"""ErrorBudgetMachine: per-mode semi-analytical error budget from the SPECULA calibration products (ASSUMPTIONS.md).

Per KL mode (command units): turbulence variance V, slope-noise variance, aliasing variance, optical gain. The
integrator / leaky integrator of every mode is optimised (filter_optimizers.py); residuals are returned in nm^2 of OPD.
"""
import os.path as op
import warnings
from dataclasses import replace as _replace
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

from . import utils
from .filter_optimizers import DEFAULT_FF_GRID, AliasSpectrum, IirControllers, IirTables, LoopTables
from .naming import DM_KEY, Paths, nearest_wavelength, snap_seeing
from .noise import Detector, PupilGeometry, SlopeNoise, measure_slope_norm, slope_noise_coefficients, slope_noise_cov
from .reconstructors import compute_reconstructor, read_rec, save_rec

DEFAULT_MODES_PER_SUBAP = {12: 50, 16: 150, 24: 300, 48: 1200}
SCENARIOS, CONTROLLERS = ('standard', 'multistage'), ('int', 'leaky', 'wiener')      # (an IirControllers is also accepted)
STEPS = dict(pupdata='ErrorBudgetGenerator.calibrate_pupdata (step 1)', im_dl='calibrate_im_dl (step 3)', slopes='calibrate_alias_sn_thrp (step 2)',
             avgframe='calibrate_alias_sn_thrp (step 2)', rec='compute_recs (step 4)', alias='compute_aliases (step 5)',
             im_pc='calibrate_im_pc (step 6)', vec='calibrate_im_pc_levels (step 6b: save_correction_vector)',
             kl='the KL basis of the SPECULA calibration')


def regress(x, values, x0, deg=2):
    """Least-squares polynomial of degree min(deg, n - 1) through (x, values), values (n, ...), evaluated at x0 (same trailing shape)."""
    x, v = np.asarray(x, float), np.asarray(values, float)
    d = min(deg, len(x) - 1)
    coef = np.polyfit(x, v.reshape(len(x), -1), d)
    return (np.vander([float(x0)], d + 1) @ coef).reshape(v.shape[1:])


def _signed_rms(v):
    """sign(v) sqrt|v|: nm rms of a variance [nm^2] that may be negative (a cross term)."""
    v = np.asarray(v, float)
    out = np.sign(v) * np.sqrt(np.abs(v))
    return float(out) if out.ndim == 0 else out


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
    og_by_level: dict = field(default_factory=dict)  # (max_corr, min_corr) of the correction vector -> og ((1, 1) = perfect)
    alias_seeing: float = None      # seeing of the aliasing product (the budget rescales it to the requested seeing)
    source_wavelength: float = None  # wavelength of the products it comes from (A37)

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
    chrom_bands: list = None        # [(ChromaticModel, k_lo, k_hi, sensing_nm)]: chromatic PSD by spatial-frequency band (fused and
                                    # multi-stage results; PsfEngine uses it instead of `chromatic`)

    @property
    def noise(self):
        return self.noise_photon + self.noise_ron

    @property
    def residual(self):
        return self.lag + self.noise + self.alias + self.xtalk

    def variances(self):
        """Every term as a variance [nm^2] (they add: total = sum of the terms; noise = noise_photon + noise_ron)."""
        t = dict(lag=self.lag.sum(), noise=self.noise.sum(), noise_photon=self.noise_photon.sum(),
                 noise_ron=self.noise_ron.sum(), alias=self.alias.sum(), xtalk=self.xtalk.sum(), fitting=self.fitting,
                 **{k: float(v) for k, v in self.extra.items()})
        if self.chrom_mult is not None:
            t['chrom_mult'] = float(self.chrom_mult.sum())
        t['total'] = sum(v for k, v in t.items() if k not in ('noise_photon', 'noise_ron'))
        return {k: float(v) for k, v in t.items()}

    def totals(self):
        """Every term in nm rms; they add in QUADRATURE (total^2 = sum of the squares). A term whose variance is negative (the signed
        chrom_mult cross term) is given as -sqrt|v|. variances() has the nm^2 values."""
        return {k: _signed_rms(v) for k, v in self.variances().items()}

    def strehl(self, wavelength_nm):
        return float(np.exp(-(2 * np.pi / wavelength_nm) ** 2 * max(self.variances()['total'], 0.0)))

    def totals_row(self):
        return {**self.params, **self.totals()}

    def mode_rows(self, modes=None):
        ids = np.arange(len(self.gain)) if modes is None else np.asarray(modes)
        ids = ids[ids < len(self.gain)]
        cols = {k: getattr(self, k)[ids] for k in ('gain', 'ff', 'og')}
        cols.update({k: _signed_rms(getattr(self, k)[ids]) for k in ('residual', 'lag', 'noise', 'noise_photon', 'noise_ron', 'alias',
                                                                     'xtalk')})                  # nm rms per mode
        if self.chrom_mult is not None:
            cols['chrom_mult'] = _signed_rms(self.chrom_mult[ids])
        base = {k: np.full(len(ids), v) for k, v in self.params.items()}
        return pd.DataFrame({**base, 'modeId': ids, 'filter': self.params['controller'].upper(), **cols})


class ErrorBudgetMachine:
    """wind: effective wind [m/s] (utils.effective_wind); n_modes_ho: modes corrected by the high-order stage
    ('multistage'); wavelength_nm: default sensing wavelength of the products (None: untagged files)."""

    def __init__(self, root_dir, diameter=8.2, L0=25.0, wind=12.0, detector=None,
                 pupilstop_tag='vlt_pupil_160pixels', dm_tag='bmc2k_vlt', modes_per_subap=None,
                 n_modes_ho=1200, kl_pad=8, slopes_from_intensity=False, tables_cache=64, wavelength_nm=None, asm=True,
                 max_sensitivity=2.0, optimize_freq=True, optimize_modes=True, freq_min=100.0, freq_step=2 ** (1 / 3), wiener_order=2,
                 temporal=None, alias_layers=None, alias_psd=None):
        self.paths = Paths(root_dir, pupilstop_tag, dm_tag)
        self.wavelength_nm = wavelength_nm
        self.D, self.L0, self.wind = diameter, L0, wind
        self.alias_layers = ((1.0, wind),) if alias_layers is None else alias_layers    # (Cn2, speed) of the aliasing spectrum (A10)
        self._alias = {}
        self.set_alias_psd(alias_psd)
        self.temporal = temporal        # None: Conan spectra with the effective wind; or temporal.ZernikePSD (multi-layer frozen flow)
        self.detector = detector if detector is not None else Detector()
        self.modes_per_subap = dict(modes_per_subap or DEFAULT_MODES_PER_SUBAP)
        self.n_modes_ho, self.kl_pad = n_modes_ho, kl_pad
        self.slopes_from_intensity = slopes_from_intensity
        self.asm = asm                  # fitting: True = LBT-ASM law; False = TipTop (von Karman outside the corrected square)
        self.max_sensitivity = max_sensitivity    # modulus margin: gains keep max |1/(1+L)| <= this (2 = 6 dB); None: stability only
        self.optimize_freq, self.optimize_modes = optimize_freq, optimize_modes
        self.freq_min, self.freq_step = float(freq_min), float(freq_step)
        self._missing_modes = set()
        self.wiener_order = int(wiener_order)    # controller='wiener': order of the per-mode Wiener POL filters (wiener.py)
        self._kl, self._cal, self._calw, self._orders, self._derived = None, {}, {}, {}, set()
        self._basis_ok = set()
        self._modes_cache, self._pupil_cache, self._recs, self._norms = None, None, OrderedDict(), {}
        self.skipped = []
        self._tables, self._tables_cache = OrderedDict(), tables_cache

    # ---- modes per sensor
    def mode_list(self, n_subap):
        """Controlled-mode counts calibrated for a sensor (modes_per_subap value: an int or a list), sorted."""
        v = self.modes_per_subap[n_subap]
        return sorted({int(x) for x in (v if isinstance(v, (list, tuple)) else [v])})

    def max_modes(self, n_subap):
        return self.mode_list(n_subap)[-1]

    def freq_candidates(self, fs):
        """fs and the frame rates below it on the fixed grid freq_min * freq_step^k (shared by all calls: the loop tables are cached)."""
        fs = float(fs)
        k = int(np.floor(np.log(fs / self.freq_min) / np.log(self.freq_step) + 1e-9)) if fs > self.freq_min else -1
        grid = {round(self.freq_min * self.freq_step ** i, 6) for i in range(k + 1)}
        return sorted({f for f in grid if f < fs * (1 - 1e-9)} | {fs}, reverse=True)

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

    # ---- wavelengths (A37)
    def source_wavelength(self, rMod, n_subap, N_corr, wl):
        """Wavelength of the products used at sensing wavelength wl [nm]: wl itself when it has perfect-correction IMs (any seeing),
        else the nearest calibrated wavelength in log(lambda) (warning). N_corr None: wavelengths with a diffraction-limited IM (enough
        for an SVD reconstructor). wl None: the untagged products."""
        if wl is None:
            return None
        if (self._has_im(rMod, n_subap, wl) if N_corr is None else self.paths.at(wl).simulated_seeings(rMod, n_subap, N_corr)):
            return wl
        cands = self.paths.calibrated_wavelengths(rMod, n_subap, N_corr)
        if not cands:
            what = 'a diffraction-limited IM' if N_corr is None else f'{N_corr} modes corrected'
            raise FileNotFoundError(f'no calibrated wavelength for {self.paths.at(None).base(rMod, n_subap)} with {what} '
                                    f'(created by {STEPS["im_dl" if N_corr is None else "im_pc"]})')
        wl0 = nearest_wavelength(wl, cands)
        key = (float(rMod), int(n_subap), N_corr, wl, wl0)
        if key not in self._derived:
            self._derived.add(key)
            warnings.warn(f'{self.paths.at(wl).base(rMod, n_subap)}, {f"{N_corr} modes corrected" if N_corr else "IM"}: not calibrated at {wl:g} nm, derived from '
                          f'{wl0:g} nm (IM x {wl0 / wl:.4f}, equivalent seeing x {(wl0 / wl) ** 1.2:.4f})')
        return wl0

    @staticmethod
    def _wl(wl):
        return None if wl is None else float(round(float(wl)))          # tag resolution 1 nm (A29)

    # ---- calibration products
    def calibration(self, rMod, n_subap, seeing, N=None, N_corr=None, rec=('svd',), wavelength_nm=None):
        """rec: ('svd',), ('ml', n_phot) or ('map', n_phot). The products of the seeing rounded to 0.05 arcsec when it was simulated;
        otherwise they are estimated from the simulated seeings, with a warning (see _interpolate). A sensing wavelength without
        products is derived from the nearest calibrated one (A37): a pyramid at wl and seeing s sees the same phase in radians as at
        wl0 and s_eff = s (wl0 / wl)^(6/5), so the products of wl0 at s_eff are used, with the IMs scaled by wl0 / wl and the
        reconstructors recomputed from the scaled IM (ensure_rec)."""
        N = N or self.max_modes(n_subap)
        N_corr = N_corr or N
        wl = self._wl(self.wavelength_nm if wavelength_nm is None else wavelength_nm)
        wl0 = self.source_wavelength(rMod, n_subap, N_corr, wl)
        k = 1.0 if wl0 == wl else (wl0 / wl) ** 1.2                      # seeing at wl0 equivalent to seeing at wl: k * seeing
        key = (float(rMod), int(n_subap), float(seeing), int(N), int(N_corr), tuple(rec), wl)
        if key not in self._calw:
            cal = self._calibration0(rMod, n_subap, float(seeing) * k, N, N_corr, tuple(rec), wl0, wl)
            if k != 1.0:                       # back to the units of wl: seeing / k, residual [nm] of the PC levels x wl / wl0
                cal = _replace(cal, alias_seeing=cal.alias_seeing / k, og_levels={r * wl / wl0: v for r, v in cal.og_levels.items()})
            cal.source_wavelength = wl0
            self._calw[key] = cal
        return self._calw[key]

    def _calibration0(self, rMod, n_subap, seeing, N, N_corr, rec, wl0, wl):
        """Calibration from the products of wl0 at the seeing `seeing` of wl0 (alias_seeing and og_levels in the units of wl0)."""
        s_tag = snap_seeing(seeing)
        simulated = op.exists(self.paths.at(wl0).im_pc(rMod, n_subap, s_tag, N_corr))
        s_key = s_tag if simulated else ('estimated', float(seeing))
        key = (float(rMod), int(n_subap), s_key, int(N), int(N_corr), tuple(rec), wl0, wl)
        if key not in self._cal:
            self._cal[key] = (self._load(rMod, n_subap, s_tag, N, N_corr, rec, wl0, wl) if simulated else
                              self._interpolate(rMod, n_subap, float(seeing), N, N_corr, rec, wl0, wl))
        return self._cal[key]

    def _interpolate(self, rMod, n_subap, seeing, N, N_corr, rec, wl0, wl):
        """Calibration at a seeing that was not simulated: optical gains (perfect and partial correction) and noise coefficients by a
        quadratic least-squares fit against the simulated seeings (linear with 2 of them, constant with 1); aliasing and crosstalk from
        the nearest simulated seeing (the budget rescales the aliasing variance by (seeing / that seeing)^(5/3), Kolmogorov)."""
        P = self.paths.at(wl0)
        sims = P.simulated_seeings(rMod, n_subap, N_corr)
        if not sims:
            raise FileNotFoundError(f'no seeing simulated for {P.sim(rMod, n_subap, seeing, N_corr)} (created by {STEPS["im_pc"]})')
        cals = [self._calibration0(rMod, n_subap, s, N, N_corr, rec, wl0, wl) for s in sims]
        near = int(np.argmin(np.abs(np.asarray(sims) - seeing)))
        fit = lambda values: regress(sims, values, seeing)
        og = fit([c.og for c in cals])
        if np.any(og <= 0):
            og = np.maximum(og, 1e-3 * np.min([c.og.min() for c in cals]))
            warnings.warn(f'seeing {seeing}: the fitted optical gain is not positive for some modes (extrapolation): clipped')
        nz = cals[0].noise
        noise = SlopeNoise(np.maximum(fit([c.noise.a for c in cals]), 0.0), np.maximum(fit([c.noise.b for c in cals]), 0.0), nz.c_pix,
                           nz.n_norm, float(np.clip(fit([c.noise.thrp for c in cals]), 1e-6, 1.0)))
        common = set.intersection(*(set(c.og_by_level) for c in cals))
        by_level = {lvl: np.maximum(fit([c.og_by_level[lvl] for c in cals]), 1e-6) for lvl in common}
        levels = {self._level_residual(P, N_corr, seeing, lvl): v for lvl, v in by_level.items()}
        kind = {1: 'constant', 2: 'linear'}.get(len(sims), 'quadratic')
        outside = not sims[0] <= seeing <= sims[-1]
        via = '' if wl0 == wl else f' (equivalent seeing of {wl:g} nm)'
        warnings.warn(f'seeing {seeing:g} arcsec{via} not simulated for {P.base(rMod, n_subap)}, {N_corr} modes corrected: optical gains and '
                      f'noise coefficients by {kind} regression over the simulated seeings {sims}'
                      f'{" (EXTRAPOLATED)" if outside else ""}; aliasing and crosstalk from {sims[near]:g} arcsec')
        return Calibration(N, N_corr, og, cals[near].alias_rms, noise, levels, cals[near].mixing, og_by_level=by_level,
                           alias_seeing=cals[near].alias_seeing)

    def _level_residual(self, P, N, seeing, level):
        """rms [nm] of the modes left by a correction vector (max_corr, min_corr): sqrt(sum (1 - c_i)^2 sigma_i^2), analytic sigma_i."""
        if level == (1.0, 1.0):
            return 0.0
        vec = utils.read_fits(_need(P.correction_vector(N, *level), 'vec'))[:N]
        var_cmd, rms, _ = self.turbulence(seeing, N)
        return float(np.sqrt(np.sum((1.0 - vec) ** 2 * var_cmd * rms ** 2)))

    def slope_geometry(self, rMod, n_subap, seeing, N_corr, wl, frame):
        """Pupil geometry with the PyrSlopec divisor measured from the products of that run (mean slopes against mean frame); the number of
        valid subapertures, with a warning, when it cannot be measured."""
        P = self.paths.at(wl)
        geom = PupilGeometry.from_file(_need(P.pupdata(n_subap), 'pupdata'), n_subap, frame.shape[0])
        key = (P.base(rMod, n_subap), float(seeing), int(N_corr))
        if key not in self._norms:
            path = P.sn(rMod, n_subap, seeing, N_corr)
            n_norm, corr = (measure_slope_norm(frame, utils.read_fits(path), geom, self.slopes_from_intensity) if op.exists(path)
                            else (float('nan'), float('nan')))
            if not (corr > 0.9 and n_norm > 0):
                warnings.warn(f'{P.sim(rMod, n_subap, seeing, N_corr)}: the slope normalisation could not be measured (correlation {corr:.2f} of '
                              f'the mean slopes with the mean frame): using the number of valid subapertures, {geom.n_valid}')
                n_norm = float(geom.n_valid)
            elif abs(n_norm / geom.n_valid - 1) > 0.02:
                warnings.warn(f'{P.sim(rMod, n_subap, seeing, N_corr)}: measured slope normalisation {n_norm:.1f} differs from the number of '
                              f'valid subapertures {geom.n_valid}')
            self._norms[key] = n_norm
        return _replace(geom, n_norm=self._norms[key])

    def _checked(self, path):
        """The path of an IM or reconstructor, after checking it is in the modal basis of this machine (header DM_TAG): anything else
        would silently mix two bases, since the modal variances and the mode shapes come from dm_tag."""
        if path not in self._basis_ok:
            tag = utils.header_value(path, DM_KEY)
            if tag != self.paths.dm_tag:
                found = f'DM {tag}' if tag else 'no DM_TAG in its header, so its DM is unknown'
                raise ValueError(f'{op.basename(path)}: {found}, but this machine has dm_tag = {self.paths.dm_tag} (wrong modal basis). Use the '
                                 'matching dm_tag or regenerate the product; a file whose DM is known but not recorded can be tagged with '
                                 'aoeb.utils.set_header(path, dm_tag=...)')
            self._basis_ok.add(path)
        return path

    def _read_rec(self, path):
        if path not in self._recs:
            self._recs[path] = read_rec(self._checked(_need(path, 'rec')))
            while len(self._recs) > 2:                                 # ~35 MB each at 1200 modes
                self._recs.popitem(last=False)
        return self._recs[path]

    def ensure_rec(self, rMod, n_subap, N, rec_spec=('svd',), seeing=None, wavelength_nm=None, overwrite=False):
        """Path of the reconstructor of N modes at a sensing wavelength, computed and saved if missing (A37): from the IM of that
        wavelength, or from the IM of the nearest calibrated wavelength scaled by wl0 / wl. rec_spec ('svd',), ('ml', n_phot) or
        ('map', n_phot); ML / MAP need the seeing (noise weights from the average frame, MAP prior) and use the full sx-sy covariance."""
        wl = self._wl(self.wavelength_nm if wavelength_nm is None else wavelength_nm)
        wl0 = self.source_wavelength(rMod, n_subap, None if rec_spec[0] == 'svd' else N, wl)
        k = 1.0 if wl0 == wl else (wl0 / wl) ** 1.2
        s_rec = None if seeing is None else snap_seeing(seeing)
        return self._ensure_rec(rMod, n_subap, N, tuple(rec_spec), s_rec, wl, wl0,
                                None if seeing is None else snap_seeing(seeing * k), overwrite)

    def _has_im(self, rMod, n_subap, wl):
        return op.exists(self.paths.at(wl).im_dl(rMod, n_subap))

    def _ensure_rec(self, rMod, n_subap, N, rec_spec, s_rec, wl, wl0, s0, overwrite=False):
        """s_rec: seeing tag of the reconstructor (units of wl); s0: seeing of the wl0 products for the noise weights."""
        method = rec_spec[0]
        n_phot = rec_spec[1] if len(rec_spec) > 1 else None
        Pw = self.paths.at(wl)
        path = Pw.rec(rMod, n_subap, N, method, n_phot, s_rec)
        if op.exists(path) and not overwrite:
            return self._checked(path)
        own = self._has_im(rMod, n_subap, wl)
        src = wl if own else wl0
        im = utils.read_fits(self._checked(_need(self.paths.at(src).im_dl(rMod, n_subap), 'im_dl')))
        if not own:
            im = im * (wl0 / wl)                                             # IM(wl) = IM(wl0) wl0 / wl (linear regime)
        cn = cxy = ca = None
        info = dict(wvl_nm=wl, src_wvl=src, dm_tag=self.paths.dm_tag)
        if method != 'svd':
            if n_phot is None or s_rec is None:
                raise ValueError(f"reconstructor '{method}' needs n_phot and the seeing")
            P0 = self.paths.at(wl0)
            frame = utils.read_fits(_need(P0.avgframe(rMod, n_subap, s0, N), 'avgframe'))
            cn, cxy = slope_noise_cov(frame, self.slope_geometry(rMod, n_subap, s0, N, wl0, frame), n_phot, self.detector,
                                      self.slopes_from_intensity)
            ca = self.turbulence(s_rec, N)[0] if method == 'map' else None
            info.update(method=method, n_phot=float(n_phot), ron=self.detector.ron_e, f2=self.detector.f2, seeing=s_rec)
        save_rec(path, compute_reconstructor(im, N, method, cn_diag=cn, ca_diag=ca, cn_xy=cxy), **info)
        if not own:
            warnings.warn(f'saved {op.basename(path)}, derived from the {wl0:g} nm IM')
        return path

    def _load(self, rMod, n_subap, seeing, N, N_corr, rec_spec, wl0, wl):
        """Products of wl0 at the (simulated) seeing `seeing` of wl0, used at the sensing wavelength wl: IMs x wl0 / wl, the
        reconstructor of wl (ensure_rec). The slopes are those of wl (same phase in radians), so the aliasing is std(R_wl @ slopes)."""
        P = self.paths.at(wl0)
        scale = 1.0 if wl0 == wl else wl0 / wl
        method = rec_spec[0]
        rpath = self._ensure_rec(rMod, n_subap, N, rec_spec, snap_seeing(seeing / scale ** 1.2), wl, wl0, seeing)
        rec = self._read_rec(rpath)
        og_of = lambda im: np.einsum('ij,ji->i', rec, im[:, :N]) * scale
        frame = utils.read_fits(_need(P.avgframe(rMod, n_subap, seeing, N_corr), 'avgframe'))
        noise = slope_noise_coefficients(rec, frame, self.slope_geometry(rMod, n_subap, seeing, N_corr, wl0, frame), self.slopes_from_intensity)
        apath, spath = P.alias(rMod, n_subap, seeing, N, N_corr), P.slopes(rMod, n_subap, seeing, N_corr)
        if op.exists(apath) and (not op.exists(spath) or (method == 'svd' and scale == 1.0)):
            alias = utils.read_fits(apath) / scale                           # SVD aliasing of wl0: std x wl / wl0
            if method != 'svd':
                warnings.warn(f'{op.basename(spath)} missing: aliasing of the SVD reconstructor used for the {method} one')
        else:
            alias = np.std(rec @ utils.read_fits(_need(spath, 'slopes')).T, axis=1)
        by_level = {lvl: og_of(utils.read_fits(self._checked(P.im_pc(rMod, n_subap, seeing, N_corr, None if lvl == (1.0, 1.0) else lvl))))
                    for lvl in P.pc_levels(rMod, n_subap, seeing, N_corr)}
        levels = {self._level_residual(P, N_corr, seeing, lvl): og for lvl, og in by_level.items()}
        im_path = self._checked(_need(P.im_pc(rMod, n_subap, seeing, N_corr), 'im_pc'))
        return Calibration(N, N_corr, og_of(utils.read_fits(im_path)), np.asarray(alias)[:N], noise, levels,
                           lambda: read_rec(rpath) @ (utils.read_fits(im_path)[:, :N] * scale), og_by_level=by_level, alias_seeing=seeing)

    # ---- loop tables
    def set_alias_psd(self, alias_psd):
        """Measured aliasing PSDs (A10b): {N corrected: [files]} (any rMod) and / or {rMod: {N corrected: [files]}} (that modulation
        first); None: AliasSpectrum (A10) everywhere. Clears the measured spectra already built."""
        self.alias_psd = {}
        for k, v in (alias_psd or {}).items():
            if isinstance(v, dict):
                self.alias_psd.setdefault(float(k), {}).update({int(n): list(f) for n, f in v.items()})
            else:
                self.alias_psd.setdefault(None, {})[int(k)] = list(v)
        self._alias_measured = {}

    def alias_spectrum(self, wavelength_nm=None, N_corr=None, rMod=None):
        """Temporal spectrum of the aliasing at a sensing wavelength (A10; None: the reference, 900 nm). With alias_psd (A10b) and N_corr
        measured: the measured per-radial-order spectra of that N (of that rMod if listed for it; nearest measured wavelength); otherwise
        AliasSpectrum (warned once)."""
        wl = self._wl(wavelength_nm)
        if self.alias_psd and N_corr is not None:
            key = (wl, int(N_corr), None if rMod is None else float(rMod))
            if key not in self._alias_measured:
                self._alias_measured[key] = self._measured_alias(wl, int(N_corr), rMod)
            if self._alias_measured[key] is not None:
                return self._alias_measured[key]
        if wl not in self._alias:
            self._alias[wl] = AliasSpectrum(self.alias_layers, wl)
        return self._alias[wl]

    def _measured_alias(self, wl, N, rMod=None):
        """MeasuredAliasSpectrum of N corrected modes at the measured wavelength nearest wl, None (with a warning) if N is not measured."""
        from .filter_optimizers import MeasuredAliasSpectrum
        files = (self.alias_psd.get(None if rMod is None else float(rMod), {}).get(N) or self.alias_psd.get(None, {}).get(N))
        if not files:
            warnings.warn(f'alias_psd: no measured aliasing spectrum for {N} corrected modes (rMod {rMod}): AliasSpectrum (A10) used')
            return None
        with np.load(files[0]) as z:
            wls = [float(k[6:]) for k in z.files if k.startswith('modes_')]
        w = min(wls, key=lambda x: abs(np.log(x / (900.0 if wl is None else wl))))
        return MeasuredAliasSpectrum(files, w, N)

    def _loop_tables(self, fs, latency, orders, alias, ff_grid, n_fold, iir=None, wiener=False):
        key = (round(float(fs), 9), round(float(latency), 9), int(orders.max()), alias.key(), tuple(ff_grid), n_fold,
               float(self.wind) if self.temporal is None else id(self.temporal), float(self.D), None if iir is None else (iir.key(len(orders)), len(orders)), self.max_sensitivity,
               self.wiener_order if wiener else None)
        if key in self._tables:
            self._tables.move_to_end(key)
        elif wiener:
            from .wiener import WienerTables
            self._tables[key] = WienerTables(fs, latency, self.temporal or self.wind, self.D, np.arange(1, orders.max() + 1), alias, self.wiener_order,
                                             n_fold=n_fold, max_sensitivity=self.max_sensitivity)
        else:
            self._tables[key] = (LoopTables(fs, latency, self.temporal or self.wind, self.D, np.arange(1, orders.max() + 1), alias,
                                            n_fold=n_fold, ff_grid=ff_grid, max_sensitivity=self.max_sensitivity) if iir is None else
                                 IirTables(fs, latency, self.temporal or self.wind, self.D, orders, alias, iir, n_fold=n_fold,
                                           max_sensitivity=self.max_sensitivity))
            while len(self._tables) > self._tables_cache:
                self._tables.popitem(last=False)
        return self._tables[key]

    # ---- the budget
    def budget(self, rMod, n_subap, seeing, flux, fs, N=None, scenario='standard', latency_s=500e-6, controller='int',
               n_fold=3, fixed_point=False, ff_grid=None, og=None, rec=('svd',), wavelength_nm=None, crosstalk=False,
               extra=None, n_xt_iter=8, chromatic=None, optimize_freq=None, optimize_modes=None, warn=True):
        """Budget of one configuration (arguments: see _budget_at). fs and N are MAXIMA: with optimize_freq (machine default) the frame
        rates below fs on the grid freq_min * freq_step^k are tried as well, with optimize_modes the calibrated mode counts below N
        (modes_per_subap lists); the result with the lowest total residual is returned, with a warning when it is not (fs, N).
        params: freqInHz / Nmodes = requested, freqUsedInHz / NmodesUsed = used."""
        opt_f = self.optimize_freq if optimize_freq is None else optimize_freq
        opt_m = self.optimize_modes if optimize_modes is None else optimize_modes
        N_req = int(N or self.max_modes(n_subap))
        freqs = self.freq_candidates(fs) if opt_f else [float(fs)]
        modes = sorted({n for n in self.mode_list(n_subap) if n <= N_req} | {N_req}, reverse=True) if opt_m else [N_req]
        kw = dict(scenario=scenario, latency_s=latency_s, controller=controller, n_fold=n_fold, fixed_point=fixed_point, ff_grid=ff_grid,
                  og=og, rec=rec, wavelength_nm=wavelength_nm, crosstalk=crosstalk, extra=extra, n_xt_iter=n_xt_iter, chromatic=chromatic)
        best, first = None, None
        for n_m in modes:
            for f in freqs:
                try:
                    r = self._budget_at(rMod, n_subap, seeing, flux, f, N=n_m, **kw)
                except FileNotFoundError:
                    if n_m == N_req:
                        raise
                    if (n_subap, n_m) not in self._missing_modes:
                        self._missing_modes.add((n_subap, n_m))
                        warnings.warn(f'{n_m} modes on {n_subap}x{n_subap}: products missing, not considered')
                    break
                total = r.variances()['total']
                if first is None:
                    first = total                                      # the requested (fs, N): candidates are sorted from it
                if best is None or total < best[0] * (1 - 1e-9):
                    best = (total, r)
        r = best[1]
        r.params.update(freqUsedInHz=r.params['freqInHz'], NmodesUsed=r.params['Nmodes'], freqInHz=float(fs), Nmodes=N_req)
        changed = r.params['freqUsedInHz'] != float(fs) or r.params['NmodesUsed'] != N_req
        if changed and warn:
            warnings.warn(f"{n_subap}x{n_subap}, rMod {rMod}, seeing {seeing}, flux {flux:.3g}: best at {r.params['freqUsedInHz']:.0f} Hz "
                          f"(max {float(fs):.0f}) and {r.params['NmodesUsed']} modes (max {N_req}): total {np.sqrt(best[0]):.1f} nm "
                          f"instead of {np.sqrt(first):.1f} nm")
        return r

    def _budget_at(self, rMod, n_subap, seeing, flux, fs, N=None, scenario='standard', latency_s=500e-6, controller='int',
                   n_fold=3, fixed_point=False, ff_grid=None, og=None, rec=('svd',), wavelength_nm=None, crosstalk=False,
                   extra=None, n_xt_iter=8, chromatic=None):
        """Budget of one configuration. flux: photons/s at the detector (before QE); scenario 'standard' (N modes, aliasing
        from the phase above N) or 'multistage' (aliasing and optical gain from a run with n_modes_ho modes corrected);
        latency_s: end of the exposure -> command applied, any real value (the frame of window + hold is in the model);
        controller 'int', 'leaky', 'wiener' (per-mode optimal IIR filters on pseudo-open-loop measurements, order machine.wiener_order;
        result.filters = [(b, a)], wiener.to_iir_filter converts them) or an IirControllers (per-mode IIR filters, only their global
        gain is optimised); og: optical gain override; crosstalk: leak of the residual of the other modes;
        extra: scalar variances [nm^2] added; chromatic: chromatic.ChromaticModel (needs wavelength_nm; the multiplicative
        term, when enabled, makes the gains optimal for the science residual). The products are those of the seeing rounded
        to 0.1 arcsec; the aliasing is rescaled to the requested seeing."""
        if scenario not in SCENARIOS:
            raise ValueError(f'scenario must be one of {SCENARIOS}, got {scenario!r}')
        iir = controller if isinstance(controller, IirControllers) else None
        if iir is None and controller not in CONTROLLERS:
            raise ValueError(f'controller must be one of {CONTROLLERS} or an IirControllers, got {controller!r}')
        ctl = 'iir' if iir is not None else controller
        N = N or self.max_modes(n_subap)
        N_corr = N if scenario == 'standard' else self.n_modes_ho
        wl = self.wavelength_nm if wavelength_nm is None else wavelength_nm
        cal = self.calibration(rMod, n_subap, seeing, N, N_corr, rec, wl)
        if fixed_point and og is None and len(cal.og_levels) < 2:
            raise ValueError('fixed_point needs partial-correction IMs at >= 2 levels (generation.pc_corrections)')
        turb_cmd, rms, r0 = self.turbulence(seeing, N)
        if N not in self._orders:
            self._orders[N] = utils.radial_order(np.arange(N))
        orders = self._orders[N]
        ff_grid = tuple(ff_grid if ff_grid is not None else (DEFAULT_FF_GRID if ctl == 'leaky' else (1.0,)))
        mult = chromatic is not None and chromatic.multiplicative
        if chromatic is not None and wl is None:
            raise ValueError('chromatic terms need the sensing wavelength: give wavelength_nm')
        eta = chromatic.eta(wl) if mult else None
        tables = self._loop_tables(fs, latency_s * fs, orders, self.alias_spectrum(wl, N_corr, rMod), ff_grid, n_fold, iir, wiener=ctl == 'wiener')
        photon, readout = cal.noise.modal_variance_parts(self.detector.n_det(flux, fs), self.detector)
        noise_cmd = photon + readout
        alias_cmd = cal.alias_rms ** 2 * (seeing / cal.alias_seeing) ** (5 / 3)            # Kolmogorov: alias variance ~ r0^(-5/3)
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
            res.filters = getattr(opt, 'filters', None)                 # controller='wiener': per-mode (b, a) of u = F POL
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
        """Fitting variance [nm^2] of N modes: LBT-ASM law (asm=True), or TipTop's von Karman PSD outside the square |kx|, |ky| <= k_c,
        k_c = sqrt(N / pi) / D (a DM with N actuators in the pupil, sqrt(4 N / pi) across) (asm=False)."""
        if self.asm:
            return float(utils.fitting_variance_lbt(r0, N, self.D))
        return utils.fitting_variance_tiptop(r0, self.L0, utils.fitting_cutoff(N, self.D))

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
                r = self.budget(rMod, n_sub, seeing, flux, fs, scenario=scen, controller=ctl, wavelength_nm=wl, warn=False, **kw)
            except FileNotFoundError as e:
                if on_missing != 'skip':
                    raise
                self.skipped.append(key)
                warnings.warn(f'skipped {key}: {e}')
                continue
            totals.append(r.totals_row())
            if store_modes is not None:
                modes.append(r.mode_rows(None if store_modes == 'all' else store_modes))
        tot = pd.DataFrame(totals)
        if len(tot) and 'freqUsedInHz' in tot:
            nf = int((tot['freqUsedInHz'] != tot['freqInHz']).sum())
            nm = int((tot['NmodesUsed'] != tot['Nmodes']).sum())
            if nf or nm:
                warnings.warn(f'optimisation: lower frame rate in {nf} and fewer modes in {nm} of {len(tot)} configurations '
                              '(columns freqUsedInHz, NmodesUsed)')
        return (pd.concat(modes, ignore_index=True) if modes else pd.DataFrame()), tot

    # ---- several sensors
    def noise_crossover(self, rMod, n_subap, seeing, fs, N=None, wavelength_nm=None, N_corr=None):
        """Flux at which photon and readout noise are equal. Per mode the photon variance is A/n and the readout variance B/n^2 (n detected
        electrons per frame), so n* = B / A. Returns dict(flux_mode [ph/s] per mode, flux [ph/s] of the sums over the modes,
        e_frame: detected electrons per frame at that flux)."""
        N = N or self.max_modes(n_subap)
        cal = self.calibration(rMod, n_subap, seeing, N, N_corr, wavelength_nm=wavelength_nm)
        A, B = cal.noise.modal_variance_parts(1.0, self.detector)                 # n_det = 1 electron per frame
        to_flux = fs / self.detector.qe
        n_tot = float(np.sum(B) / np.sum(A))
        return dict(flux_mode=B / np.maximum(A, 1e-300) * to_flux, flux=n_tot * to_flux, e_frame=n_tot)

    def budget_fused(self, sensors, seeing, fs, N=None, latency_s=500e-6, controller='int', n_fold=3, ff_grid=None, rec=('svd',),
                     crosstalk=False, chromatic=None, alias_correlation='full', optimize_freq=None, optimize_modes=None, warn=True,
                     weight_shift=0.0):
        """Budget of ONE loop fed by several sensors at the same frame rate fs whose measurements of every mode are fused with the optimal
        (inverse-variance) weights. sensors: [dict(rMod, n_subap, wavelength_nm, flux)], flux in photons/s at that sensor's detector (each
        sensor has its own detector: the readout noise is paid once per sensor). Per mode, in optical-gain-compensated units
        (sigma_i^2 = (photon_i + readout_i) / og_i^2):
            w_i = (1 / sigma_i^2) / sum_j (1 / sigma_j^2),   fused noise = 1 / sum_i (1 / sigma_i^2) = sum_i w_i^2 sigma_i^2,
        reported as photon = sum w_i^2 photon_i / og_i^2 and readout = sum w_i^2 readout_i / og_i^2 (they add up to the fused noise).
        Aliasing (both sensors see the same phase above the corrected modes): alias_correlation 'full' (fully correlated, conservative):
        (sum w_i a_i / og_i)^2, 'none' (independent, optimistic): sum w_i^2 a_i^2 / og_i^2. Crosstalk: the fused leak matrix
        sum_i diag(w_i / og_i) (R IM_PC)_i (coherent: the same residual leaks into every sensor). Chromatic shear: one effective
        sensing beam whose refractivity is the turbulence-weighted mean of the sensors' (delta_n override); the multiplicative term is
        not supported. The gains of the result are the loop gains of the compensated measurement (og = 1).
        fs and N are maxima as in budget() (common frame-rate grid; mode counts calibrated for every sensor).
        Optional per-sensor keys (used by budget_dual): 'fs' the sensor frame rate when it differs from the loop rate fs (its noise enters
        with the same noise PSD at the loop rate: per-frame variance x fs / fs_i; its longer exposure and delay are NOT modelled, which is
        optimistic for a slower sensor); 'N_rec' the modes of its reconstructor (>= N, the first N are fused); 'N_corr' the corrected modes
        of its products (default N_rec; n_modes_ho for a 'multistage' LO sensor).
        weight_shift (two sensors only): d added to the weight of the first sensor and taken from the second, the same for every mode
        (or an array with one d per mode)
        (the weights still sum to 1 and may leave [0, 1]): no longer the minimum-noise estimate, but it moves the effective refractivity
        of the fused beam, so that two bands can be EXTRAPOLATED to the science one (chromatic shear cancelled to first order in the
        shear, the noise amplified by sum w_i^2 sigma_i^2)."""
        if chromatic is not None and chromatic.multiplicative:
            raise NotImplementedError('budget_fused: the multiplicative chromatic term is not supported')
        if alias_correlation not in ('full', 'none'):
            raise ValueError("alias_correlation must be 'full' or 'none'")
        sensors = [dict(s) for s in sensors if s['flux'] > 0]
        if not sensors:
            raise ValueError('budget_fused: no sensor with a positive flux')
        opt_f = self.optimize_freq if optimize_freq is None else optimize_freq
        opt_m = self.optimize_modes if optimize_modes is None else optimize_modes
        n_max = min(self.max_modes(s['n_subap']) for s in sensors)
        N_req = int(N or n_max)
        common = set.intersection(*(set(self.mode_list(s['n_subap'])) for s in sensors))
        modes = sorted({n for n in common if n <= N_req} | {N_req}, reverse=True) if opt_m else [N_req]
        freqs = self.freq_candidates(fs) if opt_f else [float(fs)]
        kw = dict(latency_s=latency_s, controller=controller, n_fold=n_fold, ff_grid=ff_grid, rec=rec, crosstalk=crosstalk,
                  chromatic=chromatic, alias_correlation=alias_correlation, weight_shift=weight_shift)
        best = first = None
        for n_m in modes:
            for f in freqs:
                r = self._budget_fused_at(sensors, seeing, f, n_m, **kw)
                total = r.variances()['total']
                first = total if first is None else first
                if best is None or total < best[0] * (1 - 1e-9):
                    best = (total, r)
        r = best[1]
        r.params.update(freqUsedInHz=r.params['freqInHz'], NmodesUsed=r.params['Nmodes'], freqInHz=float(fs), Nmodes=N_req)
        if warn and (r.params['freqUsedInHz'] != float(fs) or r.params['NmodesUsed'] != N_req):
            warnings.warn(f"fused {r.params['sensors']}: best at {r.params['freqUsedInHz']:.0f} Hz and {r.params['NmodesUsed']} modes "
                          f"(max {float(fs):.0f} Hz, {N_req} modes): total {np.sqrt(best[0]):.1f} nm instead of {np.sqrt(first):.1f} nm")
        return r

    def _budget_fused_at(self, sensors, seeing, fs, N, latency_s, controller, n_fold, ff_grid, rec, crosstalk, chromatic,
                         alias_correlation, weight_shift=0.0):
        if controller not in CONTROLLERS:
            raise ValueError(f'budget_fused: controller must be one of {CONTROLLERS}')
        turb_cmd, rms, r0 = self.turbulence(seeing, N)
        if N not in self._orders:
            self._orders[N] = utils.radial_order(np.arange(N))
        orders = self._orders[N]
        cals, phot, ron, alias, og = [], [], [], [], []
        for s in sensors:
            n_rec = int(s.get('N_rec') or N)                     # modes of the sensor's reconstructor (>= N; only the first N are fused)
            cal = self.calibration(s['rMod'], s['n_subap'], seeing, n_rec, int(s.get('N_corr') or n_rec), rec, s['wavelength_nm'])
            fs_i = float(s.get('fs') or fs)                      # sensor frame rate: equal noise PSD at the loop rate fs
            g = np.maximum(cal.og[:N], 1e-3)
            p, rn = cal.noise.modal_variance_parts(self.detector.n_det(s['flux'], fs_i), self.detector)
            cals.append(cal), og.append(g), phot.append(p[:N] / g ** 2 * fs / fs_i), ron.append(rn[:N] / g ** 2 * fs / fs_i)
            alias.append(cal.alias_rms[:N] * (seeing / cal.alias_seeing) ** (5 / 6) / g)
        inv = np.sum([1.0 / np.maximum(p + rn, 1e-300) for p, rn in zip(phot, ron)], axis=0)
        w = [1.0 / np.maximum(p + rn, 1e-300) / inv for p, rn in zip(phot, ron)]
        if np.any(weight_shift):                                # scalar, or one value per mode (bandsplit.py)
            if len(w) != 2:
                raise ValueError('weight_shift needs exactly two sensors')
            w = [w[0] + weight_shift, w[1] - weight_shift]
        photon =np.sum([wi ** 2 * p for wi, p in zip(w, phot)], axis=0)
        readout = np.sum([wi ** 2 * rn for wi, rn in zip(w, ron)], axis=0)
        alias_cmd = (np.sum([wi * a for wi, a in zip(w, alias)], axis=0) ** 2 if alias_correlation == 'full' else
                     np.sum([(wi * a) ** 2 for wi, a in zip(w, alias)], axis=0))
        noise_cmd = photon + readout
        xt = None
        if crosstalk:
            M = sum((wi / g)[:, None] * cal.mixing()[:N, :N] for wi, g, cal in zip(w, og, cals))
            xt = (M ** 2).astype(np.float32)
            np.fill_diagonal(xt, 0.0)
        wls = [float(s['wavelength_nm']) for s in sensors]
        wbar = np.array([float(np.sum(wi * turb_cmd * rms ** 2) / np.sum(turb_cmd * rms ** 2)) for wi in w])    # weight of each sensor
        wl_eff = float(np.sum(wbar * np.asarray(wls)))
        extra, bands = {}, None
        if chromatic is not None:
            # a sensor may carry the photon-weighted n - 1 of its band ('n_minus_1', bandsplit.py) instead of that of its wavelength
            n_eff = float(np.sum(wbar * np.array([s.get('n_minus_1') or chromatic.n_minus_1(x) for s, x in zip(sensors, wls)])))
            cm = _replace(chromatic, delta_n=n_eff - float(chromatic.n_minus_1(chromatic.science_nm)))
            k_cut = utils.fitting_cutoff(N, self.D)
            extra.update(cm.variances(r0, self.L0, k_cut, wl_eff, self._k_nyquist()))
            bands = [(cm, 0.0, k_cut, wl_eff)]
        ff = tuple(ff_grid if ff_grid is not None else (DEFAULT_FF_GRID if controller == 'leaky' else (1.0,)))
        tables = self._loop_tables(fs, latency_s * fs, orders, self.alias_spectrum(wls[int(np.argmax(wbar))], N, sensors[int(np.argmax(wbar))].get('rMod')), ff, n_fold,
                                   wiener=controller == 'wiener')
        opt = self._solve(tables, turb_cmd, noise_cmd, alias_cmd, np.ones(N), orders, xt)
        desc = ' + '.join(f"{s['n_subap']}x{s['n_subap']} rMod {s['rMod']:g} {s['wavelength_nm']:.0f}nm"
                          + (f" {float(s['fs']):.0f}Hz" if s.get('fs') else '') for s in sensors)
        params = dict(scenario='fused', controller=controller, rMod=np.nan, seeing=seeing, Nsubaps=min(s['n_subap'] for s in sensors),
                      Nmodes=N, freqInHz=fs, photPerMs=sum(s['flux'] for s in sensors) / 1e3, wavelength_nm=wl_eff,
                      crosstalk=bool(crosstalk), sensors=desc, alias_correlation=alias_correlation,
                      **{f'weight_{i}': float(x) for i, x in enumerate(wbar)})
        res = self._result(opt, params, np.ones(N), rms ** 2, photon / np.maximum(noise_cmd, 1e-300), self._fitting(r0, N), extra,
                           None, None, 1)
        res.chrom_bands, res.weights = bands, np.stack(w)
        return res

    def budget_multistage(self, lo, ho, seeing, latency_s=500e-6, controller='int', crosstalk=False, chromatic=None,
                          optimize_freq=None, optimize_modes=False, warn=True, **kw):
        """Two sensors in PARALLEL after the DM (ASSUMPTIONS A16), each mode controlled by one of them: the LO sensor controls the first
        lo['N'] modes (scenario 'multistage': aliasing, frames and optical gains of the run with n_modes_ho modes corrected), the HO
        sensor the modes lo['N'] ... n_modes_ho - 1 (scenario 'standard' with n_modes_ho modes), with its own frame rate; fitting of
        n_modes_ho modes. lo, ho: dict(rMod, n_subap, wavelength_nm, flux, fs[, N]) (fs maxima unless optimize_freq=False).
        Chromatic shear by spatial-frequency band: LO sensing wavelength below the LO cut-off, HO one between the two cut-offs.
        Returns the spliced BudgetResult; .stages = (lo_result, ho_result). The mode counts are fixed (optimize_modes is ignored: the
        LO products are those of exactly lo['N'] modes under n_modes_ho corrected)."""
        if chromatic is not None and chromatic.multiplicative:
            raise NotImplementedError('budget_multistage: the multiplicative chromatic term is not supported')
        n_ho = int(ho.get('N') or self.n_modes_ho)
        if n_ho != self.n_modes_ho:
            raise ValueError(f'the HO stage must control n_modes_ho = {self.n_modes_ho} modes (the LO products assume them corrected)')
        run = lambda s, N, scen: self.budget(s['rMod'], s['n_subap'], seeing, s['flux'], s['fs'], N=N, scenario=scen, latency_s=latency_s,
                                             controller=controller, wavelength_nm=s['wavelength_nm'], crosstalk=crosstalk,
                                             optimize_freq=optimize_freq, optimize_modes=False, warn=warn, **kw)
        return self.combine_stages(run(lo, int(lo['N']), 'multistage'), run(ho, n_ho, 'standard'), lo, ho, seeing, chromatic)

    def budget_dual(self, lo, ho, seeing, latency_s=500e-6, controller='int', crosstalk=False, chromatic=None, alias_correlation='full',
                    rec=('svd',), r_ho=None, weight_shift=0.0):
        """Two sensors with their measurements FUSED: the first lo['N'] modes are controlled by one loop at the LO frame rate lo['fs'] fed by
        both sensors (per-mode inverse-variance weights, budget_fused; the LO sensor with its 'multistage' products, i.e. n_modes_ho modes
        corrected, the HO sensor with its own n_modes_ho-mode reconstructor at its frame rate ho['fs'], equal-noise-PSD approximation), the
        modes lo['N'] ... n_modes_ho - 1 by the HO sensor alone at ho['fs'] (standard). Frame rates are used as given. r_ho: the HO standard
        budget if already computed. Returns the spliced BudgetResult (.stages = (fused LO-mode result, HO result), .weights of the LO modes).
        Chromatic shear of the LO band: the effective refractivity of the fused beam (as budget_fused), not that of its mean wavelength.
        weight_shift: see budget_fused (+d to the LO sensor, -d to the HO one, on the LO modes)."""
        n_ho, n_lo = self.n_modes_ho, int(lo['N'])
        kw = dict(latency_s=latency_s, controller=controller, crosstalk=crosstalk, rec=rec)
        if r_ho is None:
            r_ho = self.budget(ho['rMod'], ho['n_subap'], seeing, ho['flux'], ho['fs'], N=n_ho, wavelength_nm=ho['wavelength_nm'],
                               optimize_freq=False, optimize_modes=False, warn=False, **kw)
        sensors = [{**lo, 'N_rec': n_lo, 'N_corr': n_ho, 'fs': lo['fs']}, {**ho, 'N_rec': n_ho, 'N_corr': n_ho}]
        r_low = self._budget_fused_at(sensors, seeing, float(lo['fs']), n_lo, n_fold=3, ff_grid=None, chromatic=None,
                                      alias_correlation=alias_correlation, weight_shift=weight_shift, **kw)
        r_low.params['freqUsedInHz'] = float(lo['fs'])
        cm_lo = None
        if chromatic is not None:
            n_eff = sum(r_low.params[f'weight_{i}'] * chromatic.n_minus_1(s['wavelength_nm']) for i, s in enumerate(sensors))
            cm_lo = _replace(chromatic, delta_n=float(n_eff - chromatic.n_minus_1(chromatic.science_nm)))
        res = self.combine_stages(r_low, r_ho, {**lo, 'wavelength_nm': r_low.params['wavelength_nm']}, ho, seeing, chromatic,
                                  chromatic_lo=cm_lo)
        res.params['scenario'] = 'dual'
        res.weights = r_low.weights
        return res

    def combine_stages(self, r_lo, r_ho, lo, ho, seeing, chromatic=None, chromatic_lo=None):
        """Splice of an LO result (scenario 'multistage', its N modes) and an HO result (standard, n_modes_ho modes): modes < N from
        the LO stage, the others and the fitting from the HO stage, chromatic shear by spatial-frequency band (budget_multistage, which
        computes the two stage budgets; call this directly to reuse cached stage results). lo, ho: the sensor dicts (wavelength_nm, flux).
        chromatic_lo: model of the LO band (default chromatic), e.g. with the delta_n of a fused beam."""
        n_ho = len(r_ho.lag)
        n = len(r_lo.lag)
        cat = lambda a, b: np.concatenate([a[:n], b[n:]])
        fields = {f: cat(getattr(r_lo, f), getattr(r_ho, f)) for f in ('gain', 'ff', 'og', 'lag', 'noise_photon', 'noise_ron', 'alias', 'xtalk')}
        extra, bands = {}, None
        if chromatic is not None:
            r0 = float(utils.r0_from_seeing(seeing))
            k_lo, k_ho = utils.fitting_cutoff(n, self.D), utils.fitting_cutoff(n_ho, self.D)
            bands = [(chromatic_lo or chromatic, 0.0, k_lo, float(lo['wavelength_nm'])), (chromatic, k_lo, k_ho, float(ho['wavelength_nm']))]
            for cm, a, b, wl in bands:
                hi = cm.variances(r0, self.L0, b, wl, self._k_nyquist())
                low = cm.variances(r0, self.L0, a, wl, self._k_nyquist()) if a > 0 else {}
                for name, v in hi.items():
                    extra[name] = extra.get(name, 0.0) + v - low.get(name, 0.0)
        tag = lambda s, r: f"{s['n_subap']}x{s['n_subap']} rMod {s['rMod']:g} {s['wavelength_nm']:.0f}nm {r.params['freqUsedInHz']:.0f}Hz"
        params = {**r_ho.params, 'scenario': 'multistage', 'N_lo': n, 'freqUsedInHz_lo': r_lo.params['freqUsedInHz'],
                  'photPerMs': (lo['flux'] + ho['flux']) / 1e3, 'wavelength_nm': np.nan,
                  'sensors': f'LO {n} modes: {tag(lo, r_lo)} | HO: {tag(ho, r_ho)}'}
        res = _replace(r_ho, params=params, extra=extra, chromatic=None, chrom_mult=None, chrom_bands=bands, **fields)
        res.stages = (r_lo, r_ho)
        return res
