"""Single- versus multi-sensor comparisons (config/analyse_errorbudget.ipynb): magnitude <-> flux, sensors, budgets at optimised frame
rates, the dual-system optimisations (split, sampling) and a disk cache of the sweeps.

Criteria (Study(criterion=...)):
  'strehl' (default, the sweeps before 2026-11-11): every frame rate and the sampling maximise the Strehl (minimum residual of the modes
      the loop controls: for a dual system the LO-mode loop and the HO loop separately); only the light split minimises the contrast in
      the zone [r1, r2] lambda/D, from the zone mean of the residual PSD (budgets alone, milliseconds each).
  'contrast' (2026-11-11): every choice is ranked as the e2e sweeps rank their runs (sweep_contrast.score): exact mean raw contrast in the
      zone (PSF of the budget, ~0.6 s each) among the candidates with science SR >= SR_SWITCH, any of them before one below it, and those
      by SR (pure contrast would pick e.g. a 14x14 single at SR 0.003 at mag 12 in VIS); no equality tolerance (full grids). The single
      stage is the best (binning, frame rate) of BINNINGS x the frame-rate grid (each binning with all its calibrated modes, the
      sensor's own n_subap is ignored); a fusion keeps its sensors and gets the best frame rate; a dual system the best HO rate, then the
      best LO-mode rate (coordinate search); the split and the sampling the lowest contrast. The disk-cache keys hold the criterion.

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

VERSION = 2
BINNINGS = (48, 36, 24, 14)          # subapertures of the contrast-optimal single stage (those calibrated in the machine are used)
SR_SWITCH = 0.10                     # criterion 'contrast': below this science SR candidates are compared by SR (e2e sweep_contrast.SR_SWITCH)


class Study:
    def __init__(self, machine, engine, seeing, latency_s, controller, area, bandw, colour, cache_dir, fs_max=2000., zone=(2.0, 5.0),
                 sr_min=0.25, recompute=False, criterion='strehl', binnings=BINNINGS):
        """machine / engine: ErrorBudgetMachine / PsfEngine; area [m^2] and bandw [nm]: pupil and camera band for the photometry; colour:
        m(900) - m(1300) of the star; zone: separations [lambda/D] where the contrast is maximised; sr_min: science Strehl below which
        a contrast is flagged (sweeps keep the Strehl); criterion: 'strehl' or 'contrast' (module docstring); binnings: subapertures
        searched by the contrast-optimal single stage."""
        if criterion not in ('strehl', 'contrast'):
            raise ValueError("criterion must be 'strehl' or 'contrast'")
        self.criterion = criterion
        self.binnings = tuple(n for n in binnings if n in machine.modes_per_subap)
        self._best = {}
        self.m, self.eng, self.seeing, self.colour, self.area, self.bandw = machine, engine, seeing, colour, area, bandw
        self.kw = dict(latency_s=latency_s, controller=controller, crosstalk=False, optimize_freq=False, optimize_modes=False)
        self.fs_max, self.fs_grid, self.zone, self.sr_min = float(fs_max), machine.freq_candidates(fs_max), tuple(zone), sr_min
        self.cache_dir, self.recompute = cache_dir, recompute
        os.makedirs(cache_dir, exist_ok=True)
        k = engine.kernels()
        ann = (k.k * machine.D >= zone[0]) & (k.k * machine.D <= zone[1])
        self.w_zone = k.K[:, ann].astype(float) @ k.count[ann] / k.count[ann].sum()
        self._stages = {}
        self.sci_nm, self.chromatic = None, None
        self._wind0 = machine.wind

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

    def _by_contrast(self):
        return self.criterion == 'contrast'

    def _contrast_key(self, r):
        """Sort key of criterion 'contrast', best first (e2e sweep_contrast.score)."""
        c, sr = self.contrast(r), r.strehl(self.sci_nm)
        if not (np.isfinite(c) and np.isfinite(sr)):
            return (2, 0.0)
        return (0, c) if sr >= SR_SWITCH else (1, -sr)

    def best_contrast_single(self, s, chromatic=True):
        """The single stage of s's modulation, wavelength and flux at the (binning, frame rate <= s['fs']) of self.binnings x the rate
        grid with the lowest zone contrast; each binning with all its calibrated modes (s['n_subap'] and s['N'] are ignored).
        params['contrast_search']: the number of candidates."""
        cm = self._chrom(chromatic)
        key = (s['rMod'], s['wavelength_nm'], round(s['flux'], 6), float(s['fs']), repr(cm), self.sci_nm, self.m.max_sensitivity)
        if key not in self._best:
            cands = [self.m.budget(s['rMod'], n, self.seeing, s['flux'], f, N=self.m.max_modes(n), wavelength_nm=s['wavelength_nm'],
                                   chromatic=cm, warn=False, **self.kw)
                     for n in self.binnings for f in self.m.freq_candidates(s['fs'])]
            best = min(cands, key=self._contrast_key)
            best.params['contrast_search'] = len(cands)
            self._best[key] = best
        return self._best[key]

    def single(self, s, chromatic=True, **kw):
        """One sensor at the frame rate (<= s['fs']) minimising the total residual (optimize_freq=False: at s['fs']); criterion
        'contrast' (and no kw): best_contrast_single."""
        if self._by_contrast() and not kw:
            return self.best_contrast_single(s, chromatic)
        return self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], s['fs'], N=s.get('N'), wavelength_nm=s['wavelength_nm'],
                             chromatic=self._chrom(chromatic), warn=False, **{**self.kw, 'optimize_freq': True, **kw})

    def fused(self, sensors, fs=None, chromatic=True, **kw):
        """Sensors fused in one loop at the common frame rate (<= fs) minimising the total residual (the sensors' 'fs' are dropped);
        criterion 'contrast' (and no kw): the rate of the grid minimising the zone contrast."""
        if self._by_contrast() and not kw:
            ss = [{k: v for k, v in s.items() if k != 'fs'} for s in sensors]
            return self.best_rate(lambda f: self.m.budget_fused(ss, self.seeing, f, chromatic=self._chrom(chromatic), warn=False,
                                                               **self.kw), self.m.freq_candidates(fs or self.fs_max))
        return self.m.budget_fused([{k: v for k, v in s.items() if k != 'fs'} for s in sensors], self.seeing, fs or self.fs_max,
                                   chromatic=self._chrom(chromatic), warn=False, **{**self.kw, 'optimize_freq': True, **kw})

    def best_rate(self, make, grid=None):
        """make(fs) -> BudgetResult; the one of the frame-rate grid with the lowest total residual (criterion 'contrast': zone
        contrast)."""
        key = self._contrast_key if self._by_contrast() else (lambda r: r.variances()['total'])
        return min((make(f) for f in (self.fs_grid if grid is None else grid)), key=key)

    def zone_psd(self, var, i0=0):
        """Mean residual PSD [nm^2 m^2] in the contrast zone of per-mode variances var [nm^2] of modes i0, i0 + 1, ..."""
        return float(self.w_zone[i0:i0 + len(var)] @ np.asarray(var))

    def stage(self, s, fs):
        """Standard budget of one sensor at frame rate fs, no chromatic term (memory cache)."""
        key = (s['rMod'], s['n_subap'], s['wavelength_nm'], round(s['flux'], 3), s.get('N'), float(fs), self.m.max_sensitivity)
        if key not in self._stages:
            self._stages[key] = self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], fs, N=s.get('N'),
                                              wavelength_nm=s['wavelength_nm'], warn=False, **self.kw)
        return self._stages[key]

    def best_single(self, s, chromatic=True):
        """One sensor at the frame rate of the grid maximising the Strehl (minimum residual of its controlled modes); criterion
        'contrast': best_contrast_single."""
        if self._by_contrast():
            return self.best_contrast_single(s, chromatic)
        f = min(self.fs_grid, key=lambda f: float(np.sum(self.stage(s, f).residual)))
        return self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], f, N=s.get('N'), wavelength_nm=s['wavelength_nm'],
                             chromatic=self._chrom(chromatic), warn=False, **self.kw)

    def best_residual_stage(self, s, i0=0):
        """(variance of the controlled modes i0:, result) at the frame rate minimising it."""
        return min(((float(np.sum(r.residual[i0:])), r) for f in self.fs_grid for r in [self.stage(s, f)]), key=lambda t: t[0])

    def best_dual(self, lo, ho, chromatic=True):
        """Dual system (budget_dual): HO frame rate minimising the residual of its own modes (they must also stay corrected, as the LO
        'multistage' products assume), LO-mode loop rate minimising the residual of the LO modes. -> (HO-mode residual variance, result).
        Criterion 'contrast': the HO rate minimising the zone contrast of the dual system (LO-mode rate at its Strehl optimum), then the
        LO-mode rate minimising it (HO rate kept)."""
        if self._by_contrast():
            return self._best_dual_contrast(lo, ho, chromatic)
        var_ho, r_ho = self.best_residual_stage(ho, lo['N'])
        ho_fs = {**ho, 'fs': r_ho.params['freqUsedInHz']}
        dual = lambda f, cm: self.m.budget_dual({**lo, 'fs': f}, ho_fs, self.seeing, latency_s=self.kw['latency_s'],
                                                controller=self.kw['controller'], chromatic=cm, r_ho=r_ho)
        # the chromatic term does not depend on the frame rate and is not in r.residual: choose without it, then add it once
        f_lo = min(self.fs_grid, key=lambda f: float(np.sum(dual(f, None).residual[:lo['N']])))
        return var_ho, dual(f_lo, self._chrom(chromatic))

    def _best_dual_contrast(self, lo, ho, chromatic=True):
        n_lo, cm = int(lo['N']), self._chrom(chromatic)
        dual = lambda f_lo, f_ho: self.m.budget_dual({**lo, 'fs': f_lo}, {**ho, 'fs': f_ho}, self.seeing, latency_s=self.kw['latency_s'],
                                                     controller=self.kw['controller'], chromatic=cm, r_ho=self.stage(ho, f_ho))
        f_lo0 = min(self.fs_grid, key=lambda f: float(np.sum(self.m.budget_dual({**lo, 'fs': f}, {**ho, 'fs': self.fs_max}, self.seeing,
                    latency_s=self.kw['latency_s'], controller=self.kw['controller'], chromatic=None,
                    r_ho=self.stage(ho, self.fs_max)).residual[:n_lo])))
        f_ho = min(self.fs_grid, key=lambda f: self._contrast_key(dual(f_lo0, f)))
        r = min((dual(f, f_ho) for f in self.fs_grid), key=self._contrast_key)
        return float(np.sum(self.stage(ho, f_ho).residual[n_lo:])), r

    def dual_at(self, r, lo, ho, chromatic=True, weight_shift=0.0):
        """The dual system of result r (best_dual / best_sampling: its frame rates and HO stage kept) with another chromatic model and / or
        a weight shift (budget_fused). lo, ho: its sensor dicts (the LO one of the chosen sampling)."""
        p = r.params
        out = self.m.budget_dual({**lo, 'fs': p['freqUsedInHz_lo']}, {**ho, 'fs': p['freqUsedInHz']}, self.seeing,
                                 latency_s=self.kw['latency_s'], controller=self.kw['controller'], chromatic=self._chrom(chromatic),
                                 r_ho=r.stages[1], weight_shift=weight_shift)
        out.params.update({k: p[k] for k in ('n_unmod', 'f_LO') if k in p}, weight_shift=float(weight_shift))
        return out

    def best_weight_shift(self, r, lo, ho, chromatic=True, bounds=(-4.0, 4.0), criterion='contrast'):
        """dual_at with the weight shift minimising the zone contrast (criterion 'contrast') or the total residual ('total'), the
        chromatic term INCLUDED: the fused LO modes are no longer the minimum-noise estimate but extrapolate the two bands towards the
        science wavelength (params['weight_shift']: added to the LO sensor's weight, taken from the HO one)."""
        from scipy.optimize import minimize_scalar
        cost = {'contrast': lambda x: self.contrast(x), 'total': lambda x: x.variances()['total']}[criterion]
        res = minimize_scalar(lambda d: cost(self.dual_at(r, lo, ho, chromatic, d)), bounds=bounds, method='bounded',
                              options=dict(xatol=0.01))
        return self.dual_at(r, lo, ho, chromatic, float(res.x))

    @contextlib.contextmanager
    def conditions(self, wind=None, ron=None):
        """Temporarily another effective wind [m/s] (Conan temporal spectra and the aliasing layers scaled with it) and / or another
        detector readout noise [e- rms]. Clears the stage and aliasing-spectrum memory caches (keyed on neither); the disk cache keys
        hold both (the detector always, the wind when not the default)."""
        m = self.m
        if wind is not None and m.temporal is not None:
            raise ValueError('conditions(wind=...) needs the Conan spectra of the effective wind (machine.temporal is set)')
        old = m.wind, m.alias_layers, m.detector

        def clear():
            self._stages.clear()
            self._best.clear()
            m._alias.clear()
        try:
            if wind is not None:
                m.wind = float(wind)
                m.alias_layers = tuple((c, v * float(wind) / old[0]) for c, v in old[1])
            if ron is not None:
                m.detector = dataclasses.replace(m.detector, ron_e=float(ron))
            clear()
            yield self
        finally:
            m.wind, m.alias_layers, m.detector = old
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
        """lo(share), ho(share) -> sensor dicts. Share f of the light to the LO sensor minimising the contrast zone, with the HO-mode
        residual rms within (1 + tol) of the HO sensor with all the light (smallest share if never met); result.params['f_LO']."""
        ref = self.best_residual_stage(ho(1.0), lo(1.0)['N'])[0]
        cand = [(f, *self.best_dual(lo(f), ho(1 - f), chromatic)) for f in fracs]
        ok = [c for c in cand if np.sqrt(c[1] / ref) <= 1 + tol] or cand[:1]
        f, _, r = min(ok, key=(lambda c: self._contrast_key(c[2])) if self._by_contrast() else (lambda c: self.zone_psd(c[2].residual)))
        r.params['f_LO'] = f
        return r

    def best_sampling(self, unmod, mod, chromatic=True):
        """unmod: {n: sensor dict of n x n subapertures}; the n (dual system with the modulated `mod` as HO) maximising the Strehl
        (minimum total residual); result.params['n_unmod']."""
        cands = []
        for n, s in unmod.items():
            r = self.best_dual(s, mod, chromatic)[1]
            r.params['n_unmod'] = n
            cands.append(r)
        return min(cands, key=self._contrast_key if self._by_contrast() else (lambda r: r.variances()['total']))

    # ---- contrast, sweeps, cache
    def contrast(self, r, name='total'):
        """Mean coronagraphic raw contrast over the zone at the science wavelength (NaN only if not positive: the model breaks down)."""
        res = self.eng.psf_terms(r, self.sci_nm, names=[name])
        c = float(np.mean(res.contrast_at(list(np.arange(self.zone[0], self.zone[1] + 1e-9)), name=name, kind='coro')))
        return c if c > 0 else np.nan

    def _key(self, name, extra=()):
        settings = (VERSION, name, self.sci_nm, repr(self.chromatic), self.seeing, self.kw, self.m.max_sensitivity, tuple(self.fs_grid),
                    self.zone, self.colour, self.area, self.bandw, repr(self.m.detector), self.m.paths.root, tuple(extra))
        if self.m.wind != self._wind0:                   # (only then: the keys of the default wind are unchanged)
            settings += (('wind', float(self.m.wind)),)
        if self.criterion != 'strehl':                   # (only then: the keys of the Strehl criterion are unchanged)
            settings += (('criterion', self.criterion, self.binnings),)
        h = hashlib.md5(repr(settings).encode()).hexdigest()[:12]
        slug = ''.join(c if c.isalnum() else '_' for c in name)[:60]
        return op.join(self.cache_dir, f'{slug}_{h}.pkl')

    def cached(self, name, compute, extra=()):
        """compute() stored under `name` (+ the settings of the Study and `extra`); loaded from the cache when present."""
        path = self._key(name, extra)
        if op.exists(path) and not self.recompute:
            with open(path, 'rb') as f:
                return pickle.load(f)
        out = compute()
        with open(path, 'wb') as f:
            pickle.dump(out, f)
        return out

    def row(self, r):
        p = r.params
        return {**r.totals(), 'contrast': self.contrast(r), 'SR': r.strehl(self.sci_nm), 'fs': p['freqUsedInHz'],
                'fs_lo': p.get('freqUsedInHz_lo', np.nan), 'f_LO': p.get('f_LO', np.nan), 'n_unmod': p.get('n_unmod', np.nan),
                'weight_shift': p.get('weight_shift', np.nan), 'n_subap': p.get('Nsubaps', np.nan),
                'N': p.get('NmodesUsed', p.get('Nmodes', np.nan))}

    def sweep(self, name, make, mags):
        """make(mag900) -> BudgetResult; one row per magnitude (totals in nm rms, contrast, science Strehl, frame rates, chosen split /
        sampling), cached under `name`: it must describe `make` completely within the band. Each magnitude is also cached on its own
        (`name | mag <m>`) as soon as it is done: a long sweep shows its progress on disk and an interrupted one resumes."""
        row = lambda mg: self.cached(f'{name} | mag {mg:g}', lambda: {'mag': mg, **self.row(make(mg))})
        return self.cached(name, lambda: pd.DataFrame([row(mg) for mg in mags]).set_index('mag'), extra=tuple(mags))

    # ---- chromatic limit
    def chromatic_contrast(self, r_ref, wl_wfs, zenith_deg, terms=('chrom_shear',)):
        """Contrast over the zone of the chromatic terms alone, sensing at wl_wfs, zenith angle zenith_deg (the model of set_band)."""
        cm = dataclasses.replace(self.chromatic, zenith_deg=float(zenith_deg), multiplicative=False)
        r = dataclasses.replace(r_ref, chromatic=cm, chrom_bands=None, params={**r_ref.params, 'wavelength_nm': float(wl_wfs)})
        return sum(self.contrast(r, n) for n in terms)


def farthest_wavelength(wl, contrast, floor, sci_nm):
    """WFS wavelength of the grid `wl` farthest from sci_nm with the chromatic contrast still below floor (it grows with the distance);
    NaN if even the nearest is above."""
    wl, cc = np.asarray(wl, float), np.asarray(contrast, float)
    order = np.argsort(np.abs(wl - sci_nm))
    ok = cc[order] <= floor
    if not ok[0]:
        return np.nan
    if ok.all():
        return wl[order[-1]]
    k = int(np.argmin(ok))
    a, b = order[k - 1], order[k]
    return wl[a] + (np.log(floor) - np.log(cc[a])) / (np.log(cc[b]) - np.log(cc[a])) * (wl[b] - wl[a])


def summary(C, mags=(6.0, 9.0, 12.0), cols=('total', 'contrast', 'fs')):
    """{band: {curve: sweep}} -> styled table, one row per (band, curve), columns (quantity, magnitude)."""
    mags = [x for x in mags if all(x in d.index for c in C.values() for d in c.values())]   # only magnitudes every sweep has
    df = pd.DataFrame({(g, k): d.loc[mags, list(cols)].unstack() for g, c in C.items() for k, d in c.items()}).T
    return df.style.format('{:.1f}').format('{:.0f}', subset=[c for c in df.columns if c[0] in ('fs', 'fs_lo', 'n_unmod')]).format(
        '{:.1e}', subset=[c for c in df.columns if c[0] == 'contrast'])
