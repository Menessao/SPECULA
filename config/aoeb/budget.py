"""ErrorBudgetMachine: per-mode semi-analytical error budget from the SPECULA calibration products (ASSUMPTIONS.md).

Per KL mode (command units): turbulence variance V, slope-noise variance, aliasing variance, optical gain. The
integrator / leaky integrator of every mode is optimised (filter_optimizers.py); residuals are returned in nm^2 of OPD.
"""
import os.path as op
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from . import utils
from .filter_optimizers import DEFAULT_FF_GRID, IirControllers, IirTables, LoopTables
from .naming import Paths, snap_seeing
from .noise import Detector, PupilGeometry, SlopeNoise, slope_noise_coefficients
from .reconstructors import read_rec

DEFAULT_MODES_PER_SUBAP = {12: 50, 16: 150, 24: 300, 48: 1200}
SCENARIOS, CONTROLLERS = ('standard', 'multistage'), ('int', 'leaky')      # (an IirControllers is also accepted)
STEPS = dict(pupdata='ErrorBudgetGenerator.calibrate_pupdata (step 1)', slopes='calibrate_alias_sn_thrp (step 2)',
             avgframe='calibrate_alias_sn_thrp (step 2)', rec='compute_recs (step 4)', alias='compute_aliases (step 5)',
             im_pc='calibrate_im_pc (step 6)', vec='calibrate_im_pc_levels (step 6b: save_correction_vector)',
             kl='the KL basis of the SPECULA calibration')


def _need(path, step):
    if not op.exists(path):
        raise FileNotFoundError(f'{path} not found (created by {STEPS[step]})')
    return path


@dataclass
class Calibration:
    N: int                          # modes controlled by this sensor
    N_corr: int                     # modes perfectly corrected in the aliasing / PC runs
    og: np.ndarray                  # optical gain (perfect correction)
    alias_rms: np.ndarray           # std of the aliasing modal signal
    noise: SlopeNoise
    og_levels: dict = field(default_factory=dict)    # residual [nm rms] of the N controlled modes (0 = perfect correction) -> og
    mixing: Callable = None         # () -> R @ IM_PC, read from disk on demand (crosstalk only)

    @property
    def thrp(self):
        return self.noise.thrp

    @cached_property
    def ct(self):
        m = self.mixing()
        og = np.diag(m)
        return np.sqrt(np.maximum(np.sum(m ** 2, axis=0) - og ** 2, 0.0))

    @cached_property
    def xt(self):
        """xt[j, i] = (R IM_PC)_ji^2 for i != j (float32)."""
        xt = (self.mixing() ** 2).astype(np.float32)
        np.fill_diagonal(xt, 0.0)
        return xt

    def og_at(self, residual_nm):
        """Optical gain interpolated at the residual [nm rms] of the controlled modes (perfect correction is level 0)."""
        if len(self.og_levels) < 2:
            return self.og
        lv = np.array(sorted(self.og_levels))
        tab = np.stack([self.og_levels[k] for k in lv])
        r = float(np.clip(residual_nm, lv[0], lv[-1]))
        k = min(int(np.searchsorted(lv, r, side='right')) - 1, len(lv) - 2)
        t = (r - lv[k]) / (lv[k + 1] - lv[k])
        return tab[k] * (1 - t) + tab[k + 1] * t


@dataclass
class BudgetResult:
    params: dict
    gain: np.ndarray
    ff: np.ndarray
    og: np.ndarray
    lag: np.ndarray                 # per-mode variances [nm^2 OPD]
    noise_photon: np.ndarray
    noise_ron: np.ndarray
    alias: np.ndarray
    xtalk: np.ndarray
    fitting: float
    extra: dict = field(default_factory=dict)       # scalar terms [nm^2]
    chromatic: object = None
    chrom_mult: np.ndarray = None   # per-mode multiplicative chromatic term [nm^2], signed
    iterations: int = 0

    @property
    def noise(self):
        return self.noise_photon + self.noise_ron

    @property
    def residual(self):
        return self.lag + self.noise + self.alias + self.xtalk

    def totals(self):
        t = dict(lag=self.lag.sum(), noise=self.noise.sum(), noise_photon=self.noise_photon.sum(),
                 noise_ron=self.noise_ron.sum(), alias=self.alias.sum(), xtalk=self.xtalk.sum(), fitting=self.fitting,
                 **{k: float(v) for k, v in self.extra.items()})
        if self.chrom_mult is not None:
            t['chrom_mult'] = float(self.chrom_mult.sum())
        t['total'] = sum(v for k, v in t.items() if k not in ('noise_photon', 'noise_ron'))
        t['total_rms'] = float(np.sqrt(max(t['total'], 0.0)))
        return t

    def strehl(self, wavelength_nm):
        return float(np.exp(-(2 * np.pi * self.totals()['total_rms'] / wavelength_nm) ** 2))

    def totals_row(self):
        return {**self.params, **self.totals()}

    def mode_rows(self, modes=None):
        ids = np.arange(len(self.gain)) if modes is None else np.asarray(modes)
        ids = ids[ids < len(self.gain)]
        cols = {k: getattr(self, k)[ids] for k in ('gain', 'ff', 'og', 'residual', 'lag', 'noise', 'noise_photon',
                                                    'noise_ron', 'alias', 'xtalk')}
        if self.chrom_mult is not None:
            cols['chrom_mult'] = self.chrom_mult[ids]
        base = {k: np.full(len(ids), v) for k, v in self.params.items()}
        return pd.DataFrame({**base, 'modeId': ids, 'filter': self.params['controller'].upper(), **cols})


class ErrorBudgetMachine:
    """wind: effective wind [m/s] (utils.effective_wind); n_modes_ho: modes corrected by the high-order stage
    ('multistage'); wavelength_nm: default sensing wavelength of the products (None: untagged files)."""

    def __init__(self, root_dir, diameter=8.2, L0=25.0, wind=12.0, detector=None,
                 pupilstop_tag='vlt_pupil_160pixels', dm_tag='bmc2k_vlt', modes_per_subap=None,
                 n_modes_ho=1200, kl_pad=8, slopes_from_intensity=False, tables_cache=64, wavelength_nm=None):
        self.paths = Paths(root_dir, pupilstop_tag, dm_tag)
        self.wavelength_nm = wavelength_nm
        self.D, self.L0, self.wind = diameter, L0, wind
        self.detector = detector if detector is not None else Detector()
        self.modes_per_subap = dict(modes_per_subap or DEFAULT_MODES_PER_SUBAP)
        self.n_modes_ho, self.kl_pad = n_modes_ho, kl_pad
        self.slopes_from_intensity = slopes_from_intensity
        self._kl, self._cal, self._orders = None, {}, {}
        self._modes_cache, self._pupil_cache, self._recs = None, None, OrderedDict()
        self.skipped = []
        self._tables, self._tables_cache = OrderedDict(), tables_cache

    # ---- turbulence
    def pupil(self):
        if self._pupil_cache is None:
            self._pupil_cache = utils.read_fits(_need(self.paths.pupilstop(), 'kl'))
            utils.check_pupil(self._pupil_cache)
        return self._pupil_cache

    def modes(self):
        if self._modes_cache is None:
            modes = utils.kl_modes_from_inverse(utils.read_fits(_need(self.paths.kl_inv(), 'kl')))
            if modes.shape[1] != int((self.pupil() > 0).sum()):
                raise ValueError(f'KL modes have {modes.shape[1]} pixels but the pupilstop has {int((self.pupil() > 0).sum())}')
            self._modes_cache = modes
        return self._modes_cache

    def kl_variance(self):
        """(OPD variance at r0 = 1 m [nm^2], rms [nm per command unit]) of every KL mode, cached on disk."""
        if self._kl is None:
            path = self.paths.kl_variance(self.L0, self.D, self.kl_pad)
            if op.exists(path):
                var, rms = utils.read_fits(path)
            else:
                var, rms = utils.modal_variance_von_karman(1.0, self.L0, self.D, self.modes(), ~(self.pupil() > 0), pad=self.kl_pad)
                utils.write_fits(path, np.stack([var, rms]))
            self._kl = (np.asarray(var), np.asarray(rms), np.asarray(var) / np.asarray(rms) ** 2)
        return self._kl[:2]

    def turbulence(self, seeing, N):
        """(variance in command^2, rms, r0) of the first N modes."""
        self.kl_variance()
        if N > len(self._kl[0]):
            raise ValueError(f'{N} modes requested but the KL basis has {len(self._kl[0])}')
        r0 = float(utils.r0_from_seeing(seeing))
        return self._kl[2][:N] * r0 ** (-5 / 3), self._kl[1][:N], r0

    # ---- calibration products
    def calibration(self, rMod, n_subap, seeing, N=None, N_corr=None, rec=('svd',), wavelength_nm=None):
        """rec: ('svd',), ('ml', n_phot) or ('map', n_phot); the seeing is that of the (rounded) calibration."""
        N = N or self.modes_per_subap[n_subap]
        N_corr = N_corr or N
        wl = self.wavelength_nm if wavelength_nm is None else wavelength_nm
        key = (float(rMod), int(n_subap), snap_seeing(seeing), int(N), int(N_corr), tuple(rec), wl)
        if key not in self._cal:
            self._cal[key] = self._load(rMod, n_subap, snap_seeing(seeing), N, N_corr, rec, wl)
        return self._cal[key]

    def _level_residual(self, P, N, seeing, level):
        """rms [nm] of the modes left by a correction vector (max_corr, min_corr): sqrt(sum (1 - c_i)^2 sigma_i^2), analytic sigma_i."""
        if level == (1.0, 1.0):
            return 0.0
        vec = utils.read_fits(_need(P.correction_vector(N, *level), 'vec'))[:N]
        var_cmd, rms, _ = self.turbulence(seeing, N)
        return float(np.sqrt(np.sum((1.0 - vec) ** 2 * var_cmd * rms ** 2)))

    def _read_rec(self, path):
        if path not in self._recs:
            self._recs[path] = read_rec(_need(path, 'rec'))
            while len(self._recs) > 2:                                 # ~35 MB each at 1200 modes
                self._recs.popitem(last=False)
        return self._recs[path]

    def _load(self, rMod, n_subap, seeing, N, N_corr, rec_spec, wl):
        P = self.paths.at(wl)
        method = rec_spec[0]
        rpath = P.rec(rMod, n_subap, N, method, rec_spec[1] if len(rec_spec) > 1 else None, seeing)
        rec = self._read_rec(rpath)
        og_of = lambda im: np.einsum('ij,ji->i', rec, im[:, :N])
        frame = utils.read_fits(_need(P.avgframe(rMod, n_subap, seeing, N_corr), 'avgframe'))
        noise = slope_noise_coefficients(rec, frame, PupilGeometry.from_file(_need(P.pupdata(n_subap), 'pupdata'), n_subap, frame.shape[0]),
                                         self.slopes_from_intensity)
        apath = P.alias(rMod, n_subap, seeing, N, N_corr)
        alias = (utils.read_fits(apath) if method == 'svd' and op.exists(apath)
                 else np.std(rec @ utils.read_fits(_need(P.slopes(rMod, n_subap, seeing, N_corr), 'slopes')).T, axis=1))
        levels = {self._level_residual(P, N_corr, seeing, lvl):
                  og_of(utils.read_fits(P.im_pc(rMod, n_subap, seeing, N_corr, None if lvl == (1.0, 1.0) else lvl)))
                  for lvl in P.pc_levels(rMod, n_subap, seeing, N_corr)}
        im_path = _need(P.im_pc(rMod, n_subap, seeing, N_corr), 'im_pc')
        return Calibration(N, N_corr, og_of(utils.read_fits(im_path)), np.asarray(alias)[:N], noise, levels,
                           lambda: read_rec(rpath) @ utils.read_fits(im_path)[:, :N])

    # ---- loop tables
    def _loop_tables(self, fs, latency, orders, alias_order, ff_grid, n_fold, iir=None):
        key = (round(float(fs), 9), round(float(latency), 9), int(orders.max()), int(alias_order), tuple(ff_grid), n_fold,
               float(self.wind), float(self.D), None if iir is None else (iir.key(len(orders)), len(orders)))
        if key in self._tables:
            self._tables.move_to_end(key)
        else:
            self._tables[key] = (LoopTables(fs, latency, self.wind, self.D, np.arange(1, orders.max() + 1), alias_order=alias_order,
                                            n_fold=n_fold, ff_grid=ff_grid) if iir is None else
                                 IirTables(fs, latency, self.wind, self.D, orders, alias_order, iir, n_fold=n_fold))
            while len(self._tables) > self._tables_cache:
                self._tables.popitem(last=False)
        return self._tables[key]

    # ---- the budget
    def budget(self, rMod, n_subap, seeing, flux, fs, N=None, scenario='standard', latency_s=500e-6, controller='int',
               n_fold=3, fixed_point=False, ff_grid=None, og=None, rec=('svd',), wavelength_nm=None, crosstalk=False,
               extra=None, n_xt_iter=8, chromatic=None):
        """Budget of one configuration. flux: photons/s at the detector (before QE); scenario 'standard' (N modes, aliasing
        from the phase above N) or 'multistage' (aliasing and optical gain from a run with n_modes_ho modes corrected);
        latency_s: end of the exposure -> command applied, any real value (the frame of window + hold is in the model);
        controller 'int', 'leaky' or an IirControllers (per-mode IIR filters, only their global gain is optimised); og: optical gain override; crosstalk: leak of the residual of the other modes;
        extra: scalar variances [nm^2] added; chromatic: chromatic.ChromaticModel (needs wavelength_nm; the multiplicative
        term, when enabled, makes the gains optimal for the science residual). The products are those of the seeing rounded
        to 0.1 arcsec; the aliasing is rescaled to the requested seeing."""
        if scenario not in SCENARIOS:
            raise ValueError(f'scenario must be one of {SCENARIOS}, got {scenario!r}')
        iir = controller if isinstance(controller, IirControllers) else None
        if iir is None and controller not in CONTROLLERS:
            raise ValueError(f'controller must be one of {CONTROLLERS} or an IirControllers, got {controller!r}')
        ctl = 'iir' if iir is not None else controller
        N = N or self.modes_per_subap[n_subap]
        N_corr = N if scenario == 'standard' else self.n_modes_ho
        wl = self.wavelength_nm if wavelength_nm is None else wavelength_nm
        cal = self.calibration(rMod, n_subap, seeing, N, N_corr, rec, wl)
        if fixed_point and og is None and len(cal.og_levels) < 2:
            raise ValueError('fixed_point needs partial-correction IMs at >= 2 levels (generation.pc_corrections)')
        turb_cmd, rms, r0 = self.turbulence(seeing, N)
        if N not in self._orders:
            self._orders[N] = utils.radial_order(np.arange(N))
        orders, alias_order = self._orders[N], int(utils.radial_order(N_corr))
        ff_grid = tuple(ff_grid if ff_grid is not None else (DEFAULT_FF_GRID if ctl == 'leaky' else (1.0,)))
        mult = chromatic is not None and chromatic.multiplicative
        if chromatic is not None and wl is None:
            raise ValueError('chromatic terms need the sensing wavelength: give wavelength_nm')
        eta = chromatic.eta(wl) if mult else None
        tables = self._loop_tables(fs, latency_s * fs, orders, alias_order, ff_grid, n_fold, iir)
        photon, readout = cal.noise.modal_variance_parts(self.detector.n_det(flux, fs), self.detector)
        noise_cmd = photon + readout
        alias_cmd = cal.alias_rms ** 2 * (seeing / snap_seeing(seeing)) ** (5 / 3)        # Kolmogorov: alias variance ~ r0^(-5/3)
        rms2 = rms ** 2
        extra = dict(extra or {})
        if chromatic is not None:
            extra.update(chromatic.variances(r0, self.L0, utils.fitting_cutoff(N, self.D), wl, self._k_nyquist()))
        og_vec, residual, converged = (cal.og if og is None else np.asarray(og, float)), 100.0, True
        xt = cal.xt if crosstalk else None
        for it in range(30 if fixed_point and og is None else 1):
            if fixed_point and og is None:
                og_vec = cal.og_at(residual)
            opt = self._solve(tables, turb_cmd, noise_cmd, alias_cmd, og_vec, orders, xt, n_xt_iter, eta)
            res = self._result(opt, dict(scenario=scenario, controller=ctl, rMod=rMod, seeing=seeing, Nsubaps=n_subap, Nmodes=N,
                                         freqInHz=fs, photPerMs=flux / 1e3, wavelength_nm=np.nan if wl is None else float(wl),
                                         crosstalk=bool(crosstalk)), og_vec, rms2, photon / np.maximum(noise_cmd, 1e-300),
                               self._fitting(r0, N), extra, chromatic, eta, it + 1)
            if not (fixed_point and og is None):
                return res
            new = float(np.sqrt(np.sum(res.residual)))                    # rms of the controlled modes = the PC level
            if abs(new - residual) < 0.5:
                return res
            residual = 0.5 * (residual + new)
        warnings.warn('fixed_point: the optical-gain / residual iteration did not converge in 30 passes')
        return res

    @staticmethod
    def _result(opt, params, og_vec, rms2, photon_fraction, fitting, extra, chromatic, eta, iterations):
        noise = opt.noise * rms2
        return BudgetResult(
            params=params, gain=opt.gain, ff=opt.ff, og=og_vec, lag=opt.turb * rms2, noise_photon=noise * photon_fraction,
            noise_ron=noise * (1.0 - photon_fraction), alias=opt.alias * rms2, xtalk=opt.xtalk * rms2, fitting=fitting, extra=extra,
            chromatic=chromatic, chrom_mult=None if eta is None else (eta ** 2 * opt.turb0 + 2.0 * eta * opt.cross) * rms2,
            iterations=iterations)

    def _k_nyquist(self):
        return self.pupil().shape[0] / (2.0 * self.D)

    def _fitting(self, r0, N):
        return float(utils.fitting_variance_lbt(r0, N, self.D))

    @staticmethod
    def _solve(tables, turb_cmd, noise_cmd, alias_cmd, og_vec, orders, xt=None, n_iter=8, eta=None, tol=1e-3):
        """Optimise all modes; with `xt` iterate the crosstalk disturbance d = xt @ residual (damped)."""
        opt = tables.optimize(turb_cmd, noise_cmd, alias_cmd, og_vec, orders, eta=eta)
        if xt is None:
            return opt
        d2 = np.zeros_like(turb_cmd)
        for _ in range(n_iter):
            d_new = xt @ opt.resid
            if np.max(np.abs(d_new - d2)) <= tol * max(np.max(d_new), 1e-30):
                return opt
            d2 = 0.5 * (d2 + d_new) if np.any(d2) else d_new
            opt = tables.optimize(turb_cmd, noise_cmd, alias_cmd, og_vec, orders, xtalk_var=d2, eta=eta)
        warnings.warn(f'crosstalk iteration did not converge in {n_iter} passes (tolerance {tol})')
        return opt

    def scan(self, seeings, rMods, n_subaps, fluxes, freqs, scenarios=('standard',), controllers=('int',), store_modes=None,
             progress=False, wavelengths=(None,), on_missing='raise', **kw):
        """Cartesian scan -> (modes_df, totals_df). store_modes: None, 'all' or a list of mode ids; extra kwargs go to budget().
        on_missing 'skip': configurations whose products are missing are recorded in `self.skipped` instead of aborting."""
        totals, modes, last, self.skipped = [], [], None, []
        for wl, scen, rMod, seeing, n_sub, ctl, fs, flux in product(wavelengths, scenarios, rMods, seeings, n_subaps,
                                                                    controllers, freqs, fluxes):
            key = (wl, scen, rMod, seeing, n_sub)
            if key in self.skipped:
                continue
            if progress and key != last:
                last = key
                print(f'wvl={wl} {scen} rMod={rMod} seeing={seeing} n_subap={n_sub}', flush=True)
            try:
                r = self.budget(rMod, n_sub, seeing, flux, fs, scenario=scen, controller=ctl, wavelength_nm=wl, **kw)
            except FileNotFoundError as e:
                if on_missing != 'skip':
                    raise
                self.skipped.append(key)
                warnings.warn(f'skipped {key}: {e}')
                continue
            totals.append(r.totals_row())
            if store_modes is not None:
                modes.append(r.mode_rows(None if store_modes == 'all' else store_modes))
        return (pd.concat(modes, ignore_index=True) if modes else pd.DataFrame()), pd.DataFrame(totals)
