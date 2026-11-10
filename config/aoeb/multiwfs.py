"""Single- versus multi-sensor comparisons (config/analyse_errorbudget.ipynb): magnitude <-> flux, sensors, budgets at optimised frame
rates, the dual-system optimisations (split, sampling) and a disk cache of the sweeps.

Criteria: every frame rate and the sampling maximise the Strehl (minimum residual of the modes the loop controls: for a dual system the
LO-mode loop and the HO loop separately); only the light split minimises the contrast in the zone [r1, r2] lambda/D. To first order the
coronagraphic halo there is the residual PSD, a sum over the modes of (variance x zone mean of the mode's PSD kernel), so the split is
chosen from budgets alone (milliseconds each); the exact contrast is computed once, at the optimum.

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


class Study:
    def __init__(self, machine, engine, seeing, latency_s, controller, area, bandw, colour, cache_dir, fs_max=2000., zone=(2.0, 5.0),
                 sr_min=0.25, recompute=False):
        """machine / engine: ErrorBudgetMachine / PsfEngine; area [m^2] and bandw [nm]: pupil and camera band for the photometry; colour:
        m(900) - m(1300) of the star; zone: separations [lambda/D] where the contrast is maximised; sr_min: science Strehl below which
        a contrast is flagged (sweeps keep the Strehl)."""
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

    def single(self, s, chromatic=True, **kw):
        """One sensor at the frame rate (<= s['fs']) minimising the total residual (optimize_freq=False: at s['fs'])."""
        return self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], s['fs'], N=s.get('N'), wavelength_nm=s['wavelength_nm'],
                             chromatic=self._chrom(chromatic), warn=False, **{**self.kw, 'optimize_freq': True, **kw})

    def fused(self, sensors, fs=None, chromatic=True, **kw):
        """Sensors fused in one loop at the common frame rate (<= fs) minimising the total residual (the sensors' 'fs' are dropped)."""
        return self.m.budget_fused([{k: v for k, v in s.items() if k != 'fs'} for s in sensors], self.seeing, fs or self.fs_max,
                                   chromatic=self._chrom(chromatic), warn=False, **{**self.kw, 'optimize_freq': True, **kw})

    def best_rate(self, make):
        """make(fs) -> BudgetResult; the one of the frame-rate grid with the lowest total residual."""
        return min((make(f) for f in self.fs_grid), key=lambda r: r.variances()['total'])

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
        """One sensor at the frame rate of the grid maximising the Strehl (minimum residual of its controlled modes)."""
        f = min(self.fs_grid, key=lambda f: float(np.sum(self.stage(s, f).residual)))
        return self.m.budget(s['rMod'], s['n_subap'], self.seeing, s['flux'], f, N=s.get('N'), wavelength_nm=s['wavelength_nm'],
                             chromatic=self._chrom(chromatic), warn=False, **self.kw)

    def best_residual_stage(self, s, i0=0):
        """(variance of the controlled modes i0:, result) at the frame rate minimising it."""
        return min(((float(np.sum(r.residual[i0:])), r) for f in self.fs_grid for r in [self.stage(s, f)]), key=lambda t: t[0])

    def best_dual(self, lo, ho, chromatic=True):
        """Dual system (budget_dual): HO frame rate minimising the residual of its own modes (they must also stay corrected, as the LO
        'multistage' products assume), LO-mode loop rate minimising the residual of the LO modes. -> (HO-mode residual variance, result)."""
        var_ho, r_ho = self.best_residual_stage(ho, lo['N'])
        ho_fs = {**ho, 'fs': r_ho.params['freqUsedInHz']}
        dual = lambda f, cm: self.m.budget_dual({**lo, 'fs': f}, ho_fs, self.seeing, latency_s=self.kw['latency_s'],
                                                controller=self.kw['controller'], chromatic=cm, r_ho=r_ho)
        # the chromatic term does not depend on the frame rate and is not in r.residual: choose without it, then add it once
        f_lo = min(self.fs_grid, key=lambda f: float(np.sum(dual(f, None).residual[:lo['N']])))
        return var_ho, dual(f_lo, self._chrom(chromatic))

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
        f, _, r = min(ok, key=lambda c: self.zone_psd(c[2].residual))
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
        return min(cands, key=lambda r: r.variances()['total'])

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
                'weight_shift': p.get('weight_shift', np.nan)}

    def sweep(self, name, make, mags):
        """make(mag900) -> BudgetResult; one row per magnitude (totals in nm rms, contrast, science Strehl, frame rates, chosen split /
        sampling), cached under `name`: it must describe `make` completely within the band."""
        return self.cached(name, lambda: pd.DataFrame([{'mag': mg, **self.row(make(mg))} for mg in mags]).set_index('mag'),
                           extra=tuple(mags))

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
    df = pd.DataFrame({(g, k): d.loc[list(mags), list(cols)].unstack() for g, c in C.items() for k, d in c.items()}).T
    return df.style.format('{:.1f}').format('{:.0f}', subset=[c for c in df.columns if c[0] in ('fs', 'fs_lo', 'n_unmod')]).format(
        '{:.1e}', subset=[c for c in df.columns if c[0] == 'contrast'])
