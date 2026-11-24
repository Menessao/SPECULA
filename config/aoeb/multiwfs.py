"""Single- versus multi-sensor comparisons (config/analyse_errorbudget.ipynb): magnitude <-> flux, sensors, budgets at optimised frame
rates, the dual-system optimisations (split, sampling) and a disk cache of the sweeps.

Every choice is ranked as the e2e sweeps rank their runs (sweep_contrast.score): exact mean raw coronagraphic contrast in the zone
[r1, r2] lambda/D (perfect coronagraph, PSF of the budget, ~0.6 s each), relative to the peak of the long-exposure AO PSF of the same
budget (as the e2e PsfCoronagraph), among the candidates with science SR >= SR_SWITCH, any of them before one below it, and those by SR
(pure contrast would pick e.g. a 14x14 single at SR 0.003 at mag 12 in VIS); full grids, no equality tolerance. The single stage is the
best (binning, frame rate) of BINNINGS x the frame-rate grid (each binning with all its calibrated modes, the sensor's own n_subap is
ignored); a fusion keeps its sensors and gets the best frame rate; a dual system the best HO rate, then the best LO-mode rate
(coordinate search); the split and the sampling the lowest contrast.

Cache: Study.sweep / Study.cached store their results in cache_dir under a hash of (name, band, every setting of the Study). A change of
a setting recomputes; a change of the code does NOT: bump VERSION, or pass recompute=True."""
import contextlib
import dataclasses
import hashlib
import os
import os.path as op
import pickle

import numpy as np
import pandas as pd

VERSION = 4
BINNINGS = (48, 36, 24, 14)          # subapertures of the contrast-optimal single stage (those calibrated in the machine are used)
SR_SWITCH = 0.10                     # below this science SR candidates are compared by SR (e2e sweep_contrast.SR_SWITCH)


class Study:
    def __init__(self, machine, engine, seeing, latency_s, controller, area, bandw, colour, cache_dir, fs_max=2000., zone=(2.0, 5.0),
                 sr_min=0.25, recompute=False, binnings=BINNINGS):
        """machine / engine: ErrorBudgetMachine / PsfEngine; controller: 'int', 'leaky' or 'wiener'; area [m^2] and bandw [nm]: pupil and
        camera band for the photometry; colour: m(900) - m(1300) of the star; zone: separations [lambda/D] where the contrast is
        minimised; sr_min: science Strehl below which a contrast is flagged (sweeps keep the Strehl); binnings: subapertures searched
        by the contrast-optimal single stage."""
        self.binnings = tuple(n for n in binnings if n in machine.modes_per_subap)
        self._best = {}
        self.m, self.eng, self.seeing, self.colour, self.area, self.bandw = machine, engine, seeing, colour, area, bandw
        self.kw = dict(latency_s=latency_s, controller=controller, crosstalk=False, optimize_freq=False, optimize_modes=False)
        self.fs_max, self.fs_grid, self.zone, self.sr_min = float(fs_max), machine.freq_candidates(fs_max), tuple(zone), sr_min
        self.cache_dir, self.recompute = cache_dir, recompute
        os.makedirs(cache_dir, exist_ok=True)
        self._stages = {}
        self.sci_nm, self.chromatic = None, None

    # ---- band, photometry, sensors
    def set_band(self, sci_nm, chromatic):
        """Science wavelength of the contrasts and chromatic model of the following budgets (None: no chromatic term)."""
        self.sci_nm, self.chromatic = float(sci_nm), chromatic

    def flux_of(self, mag900, wl, share=1.0):
        """Photons/s at a detector sensing at wl [nm] (camera band bandw) for the star of magnitude mag900 at 900 nm."""
        from specula.lib.n_phot import n_phot
        mag = mag900 - (self.colour if wl == 1300 else 0.0)
        return share * n_phot(mag, lambda_=wl * 1e-9, width=1e-9)[0] * self.bandw * self.area

    def mag_of(self, flux, wl=900.):
        """Magnitude at 900 nm of the star giving `flux` [ph/s] at a detector sensing at wl (inverse of flux_of)."""
        return -2.5 * np.log10(flux / self.flux_of(0.0, wl))

    def sensor(self, rMod, n_subap, wl, mag900, share=1.0, fs=None, N=None):
        return dict(rMod=float(rMod), n_subap=int(n_subap), wavelength_nm=float(wl), flux=self.flux_of(mag900, wl, share),
                    fs=float(fs or self.fs_max), **({} if N is None else {'N': N}))

    # ---- budgets
    def _chrom(self, chromatic):
        return self.chromatic if chromatic is True else (None if chromatic is False else chromatic)

    def _contrast_key(self, r):
        """Sort key of a candidate, best first (e2e sweep_contrast.score)."""
        c, sr = self.contrast(r), r.strehl(self.sci_nm)
        if not (np.isfinite(c) and np.isfinite(sr)):
            return (2, 0.0)
        return (0, c) if sr >= SR_SWITCH else (1, -sr)

    def single(self, s, chromatic=True):
        """The single stage of s's modulation, wavelength and flux at the (binning, frame rate <= s['fs']) of self.binnings x the rate
        grid with the best contrast key; each binning with all its calibrated modes (s['n_subap'] and s['N'] are ignored).
        params['contrast_search']: the number of candidates."""
        cm = self._chrom(chromatic)
        key = (s['rMod'], s['wavelength_nm'], round(s['flux'], 6), float(s['fs']), repr(cm), self.sci_nm)
        if key not in self._best:
            cands = [self.m.budget(s['rMod'], n, self.seeing, s['flux'], f, N=self.m.max_modes(n), wavelength_nm=s['wavelength_nm'],
                                   chromatic=cm, warn=False, **self.kw)
                     for n in self.binnings for f in self.m.freq_candidates(s['fs'])]
            best = min(cands, key=self._contrast_key)
            best.params['contrast_search'] = len(cands)
            self._best[key] = best
        return self._best[key]

    best_single = single

    def fused(self, sensors, fs=None, chromatic=True):
        """Sensors fused in one loop (budget_fused) at the common frame rate (<= fs) of the grid with the best contrast key (the
        sensors' 'fs' are dropped)."""
        ss = [{k: v for k, v in s.items() if k != 'fs'} for s in sensors]
        return self.best_rate(lambda f: self.m.budget_fused(ss, self.seeing, f, chromatic=self._chrom(chromatic), warn=False, **self.kw),
                              self.m.freq_candidates(fs or self.fs_max))

    def best_rate(self, make, grid=None):
        """make(fs) -> BudgetResult; the one of the frame-rate grid with the best contrast key."""
        return min((make(f) for f in (self.fs_grid if grid is None else grid)), key=self._contrast_key)

    def stage(self, s, fs):
        """Budget of one sensor at frame rate fs, no chromatic term (memory cache)."""
        key = (s['rMod'], s['n_subap'], s['wavelength_nm'], round(s['flux'], 3), s.get('N'), float(fs))
        if key not in self._stages:
            self._stages[key] = self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], fs, N=s.get('N'),
                                              wavelength_nm=s['wavelength_nm'], warn=False, **self.kw)
        return self._stages[key]

    def best_residual_stage(self, s, i0=0):
        """(variance of the controlled modes i0:, result) at the frame rate minimising it."""
        return min(((float(np.sum(r.residual[i0:])), r) for f in self.fs_grid for r in [self.stage(s, f)]), key=lambda t: t[0])

    def _dual(self, lo, ho, f_lo, f_ho, chromatic):
        return self.m.budget_dual({**lo, 'fs': f_lo}, {**ho, 'fs': f_ho}, self.seeing, latency_s=self.kw['latency_s'],
                                  controller=self.kw['controller'], chromatic=chromatic, r_ho=self.stage(ho, f_ho))

    def best_dual(self, lo, ho, chromatic=True):
        """Dual system (budget_dual): the HO rate with the best contrast key of the dual system (LO-mode rate at its minimum residual),
        then the LO-mode rate with the best key (HO rate kept). -> (HO-mode residual variance, result)."""
        n_lo, cm = int(lo['N']), self._chrom(chromatic)
        f_lo0 = min(self.fs_grid, key=lambda f: float(np.sum(self._dual(lo, ho, f, self.fs_max, None).residual[:n_lo])))
        f_ho = min(self.fs_grid, key=lambda f: self._contrast_key(self._dual(lo, ho, f_lo0, f, cm)))
        r = min((self._dual(lo, ho, f, f_ho, cm) for f in self.fs_grid), key=self._contrast_key)
        return float(np.sum(self.stage(ho, f_ho).residual[n_lo:])), r

    def dual_at(self, r, lo, ho, chromatic=True):
        """The dual system of result r (best_dual / best_split / best_sampling: its frame rates and HO stage kept) with another chromatic
        model. lo, ho: its sensor dicts (the LO one of the chosen sampling)."""
        p = r.params
        out = self.m.budget_dual({**lo, 'fs': p['freqUsedInHz_lo']}, {**ho, 'fs': p['freqUsedInHz']}, self.seeing,
                                 latency_s=self.kw['latency_s'], controller=self.kw['controller'], chromatic=self._chrom(chromatic),
                                 r_ho=r.stages[1])
        out.params.update({k: p[k] for k in ('n_unmod', 'f_LO') if k in p})
        return out

    @contextlib.contextmanager
    def conditions(self, ron):
        """Temporarily another detector readout noise [e- rms] (the stage memory caches are cleared; the disk cache keys hold the
        detector)."""
        m, old = self.m, self.m.detector

        def clear():
            self._stages.clear()
            self._best.clear()
        try:
            m.detector = dataclasses.replace(m.detector, ron_e=float(ron))
            clear()
            yield self
        finally:
            m.detector = old
            clear()

    @contextlib.contextmanager
    def zenith(self, zenith_deg):
        """Temporarily the chromatic model of the band at another zenith angle (the loop and its seeing along the line of sight unchanged)."""
        cm = self.chromatic
        self.chromatic = dataclasses.replace(cm, zenith_deg=float(zenith_deg))
        try:
            yield self
        finally:
            self.chromatic = cm

    def scan(self, name, values, context, configs, mags):
        """Configurations {label: make(mag900) -> BudgetResult} over `values` of a parameter (context(v): its context manager) and
        magnitudes -> long DataFrame (x, mag, config, the row of each result). Each (value, magnitude) is cached under name (it must
        describe the configurations) + the value."""
        rows = []
        for v in values:
            with context(v):
                for mg in mags:
                    out = self.cached(f'{name} x{v:g} mag{mg:g}', lambda: {k: self.row(make(mg)) for k, make in configs.items()})
                    rows += [dict(x=v, mag=mg, config=k, **row) for k, row in out.items()]
        return pd.DataFrame(rows)

    def best_split(self, lo, ho, fracs, tol, chromatic=True):
        """lo(share), ho(share) -> sensor dicts. Share f of the light to the LO sensor with the best contrast key, with the HO-mode
        residual rms within (1 + tol) of the HO sensor with all the light (smallest share if never met); result.params['f_LO']."""
        ref = self.best_residual_stage(ho(1.0), lo(1.0)['N'])[0]
        cand = [(f, *self.best_dual(lo(f), ho(1 - f), chromatic)) for f in fracs]
        ok = [c for c in cand if np.sqrt(c[1] / ref) <= 1 + tol] or cand[:1]
        f, _, r = min(ok, key=lambda c: self._contrast_key(c[2]))
        r.params['f_LO'] = f
        return r

    def best_sampling(self, unmod, mod, chromatic=True):
        """unmod: {n: sensor dict of n x n subapertures}; the n (dual system with the modulated `mod` as HO) with the best contrast key;
        result.params['n_unmod']."""
        cands = []
        for n, s in unmod.items():
            r = self.best_dual(s, mod, chromatic)[1]
            r.params['n_unmod'] = n
            cands.append(r)
        return min(cands, key=self._contrast_key)

    # ---- contrast, sweeps, cache
    def contrast(self, r, name='total'):
        """Mean coronagraphic raw contrast over the zone at the science wavelength, relative to the AO PSF peak (NaN only if not positive:
        the model breaks down)."""
        return self._contrast_peak(r, name)[0]

    def ao_peak(self, r):
        """Peak of the long-exposure AO PSF of r at the science wavelength, diffraction-limited peak = 1."""
        return float(self.eng.psf_terms(r, self.sci_nm, names=['total']).image('total', 'psf').max())

    def _contrast_peak(self, r, name='total'):
        names = [name] if name == 'total' else [name, 'total']
        res = self.eng.psf_terms(r, self.sci_nm, names=names)
        peak = float(res.image('total', 'psf').max())
        c = float(np.mean(res.contrast_at(list(np.arange(self.zone[0], self.zone[1] + 1e-9)), name=name, kind='coro'))) / peak
        return (c if c > 0 else np.nan), peak

    def _key(self, name, extra=()):
        settings = (VERSION, name, self.sci_nm, repr(self.chromatic), self.seeing, self.kw, tuple(self.fs_grid), self.zone, self.colour,
                    self.area, self.bandw, repr(self.m.detector), self.m.paths.root, self.binnings, tuple(extra))
        h = hashlib.md5(repr(settings).encode()).hexdigest()[:12]
        slug = ''.join(c if c.isalnum() else '_' for c in name)[:60]
        return op.join(self.cache_dir, f'{slug}_{h}.pkl')

    def cached(self, name, compute, extra=()):
        """compute() stored under `name` (+ the settings of the Study and `extra`); loaded from the cache when present."""
        path = self._key(name, extra)
        if op.exists(path) and not self.recompute:
            return self._load(path)
        return self._save(path, compute())

    def lookup(self, name, extra=()):
        """The cached object of `name`, None if absent; never computes."""
        path = self._key(name, extra)
        return self._load(path) if op.exists(path) else None

    @staticmethod
    def _load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _save(path, out):
        with open(path, 'wb') as f:
            pickle.dump(out, f)
        return out

    def row(self, r):
        p = r.params
        c, peak = self._contrast_peak(r)
        return {**r.totals(), 'contrast': c, 'peak': peak, 'SR': r.strehl(self.sci_nm), 'fs': p['freqUsedInHz'],
                'fs_lo': p.get('freqUsedInHz_lo', np.nan), 'f_LO': p.get('f_LO', np.nan), 'n_unmod': p.get('n_unmod', np.nan),
                'n_subap': p.get('Nsubaps', np.nan), 'N': p.get('NmodesUsed', p.get('Nmodes', np.nan))}

    def sweep(self, name, make, mags):
        """make(mag900) -> BudgetResult; one row per magnitude (totals in nm rms, contrast, science Strehl, frame rates, chosen split /
        sampling), cached under `name`: it must describe `make` completely within the band. Each magnitude is also cached on its own
        (`name | mag <m>`) as soon as it is done: a long sweep shows its progress on disk and an interrupted one resumes."""
        row = lambda mg: self.cached(f'{name} | mag {mg:g}', lambda: {'mag': mg, **self.row(make(mg))})
        return self.cached(name, lambda: pd.DataFrame([row(mg) for mg in mags]).set_index('mag'), extra=tuple(mags))


def summary(C, mags=(6.0, 9.0, 12.0), cols=('total', 'contrast', 'fs')):
    """{band: {curve: sweep}} -> styled table, one row per (band, curve), columns (quantity, magnitude)."""
    mags = [x for x in mags if all(x in d.index for c in C.values() for d in c.values())]   # only magnitudes every sweep has
    df = pd.DataFrame({(g, k): d.loc[mags, list(cols)].unstack() for g, c in C.items() for k, d in c.items()}).T
    return df.style.format('{:.1f}').format('{:.0f}', subset=[c for c in df.columns if c[0] in ('fs', 'fs_lo', 'n_unmod')]).format(
        '{:.1e}', subset=[c for c in df.columns if c[0] == 'contrast'])
