"""ErrorBudgetMachine: per-mode semi-analytical error budget from the SPECULA calibration products (ASSUMPTIONS.md).

Per KL mode (command units): turbulence variance V, slope-noise variance, aliasing variance, optical gain. The
controller of every mode (integrator, leaky integrator or Wiener filter) is optimised (filter_optimizers.py, wiener.py); residuals are
returned in nm^2 of OPD. The fitting term is the bound of the N controlled modes (ASSUMPTIONS A15, psf.kl_fitting_psd).
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
from .filter_optimizers import DEFAULT_FF_GRID, AliasSpectrum, LoopTables
from .naming import DM_KEY, Paths, nearest_wavelength, snap_seeing
from .noise import Detector, PupilGeometry, SlopeNoise, measure_slope_norm, slope_noise_coefficients, slope_noise_cov
from .reconstructors import compute_reconstructor, read_rec, save_rec

DEFAULT_MODES_PER_SUBAP = {12: 50, 16: 150, 24: 300, 48: 1200}
CONTROLLERS = ('int', 'leaky', 'wiener')
N_FOLD = 3                        # spectral images of the loop tables
WIENER_ORDER = 2                  # order of the per-mode Wiener POL filters (wiener.py)
FREQ_MIN, FREQ_STEP = 100.0, 2 ** (1 / 3)      # frame-rate grid of the rate search [Hz]
KL_PAD, PSD_PAD, TABLES_CACHE = 8, 4, 64       # padding of the KL variances, of the PSD kernels / fitting bound; loop tables kept
STEPS = dict(pupdata='ErrorBudgetGenerator.calibrate_pupdata (step 1)', im_dl='calibrate_im_dl (step 3)', slopes='calibrate_alias_sn_thrp (step 2)',
             avgframe='calibrate_alias_sn_thrp (step 2)', rec='compute_recs (step 4)', alias='compute_aliases (step 5)',
             im_pc='calibrate_im_pc (step 6)', kl='the KL basis of the SPECULA calibration')


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
    mixing: Callable = None         # () -> R @ IM_PC, read from disk on demand (crosstalk only)
    alias_seeing: float = None      # seeing of the aliasing product (the budget rescales it to the requested seeing)
    source_wavelength: float = None  # wavelength of the products it comes from (A37)
    alias_frames: Callable = None   # () -> (N, F) aliasing modal signal R @ slopes of the aliasing run (fusion: correlation between sensors)
    turb_frames: Callable = None    # () -> (F, 10) turbulence modes of the same frames (checks that two sensors' runs are paired)

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
    chrom_bands: list = None        # [(ChromaticModel, k_lo, k_hi, sensing_nm)]: chromatic PSD by spatial-frequency band (fused and
                                    # multi-stage results; PsfEngine uses it instead of `chromatic`)
    chrom_shear_modes: np.ndarray = None   # per-mode chromatic shear [nm^2] of fused results (modal, replaces the chrom_shear band PSD)

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
        t['total'] = sum(v for k, v in t.items() if k not in ('noise_photon', 'noise_ron'))
        return {k: float(v) for k, v in t.items()}

    def totals(self):
        """Every term in nm rms; they add in QUADRATURE (total^2 = sum of the squares). variances() has the nm^2 values."""
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
        base = {k: np.full(len(ids), v) for k, v in self.params.items()}
        return pd.DataFrame({**base, 'modeId': ids, 'filter': self.params['controller'].upper(), **cols})


class ErrorBudgetMachine:
    """wind: effective wind [m/s] (utils.effective_wind); n_modes_ho: modes corrected by the high-order stage (budget_dual: the
    products of the LO sensor are those of a run with n_modes_ho modes corrected); wavelength_nm: default sensing wavelength of the
    products (None: untagged files); optimize_freq / optimize_modes: defaults of budget(); alias_layers: (Cn2, speed) of the aliasing
    spectrum model (A10); alias_psd: measured aliasing spectra (A10b, set_alias_psd). The loop gains are limited only by stability (98 %
    of the critical gain)."""
    PSD_PAD = PSD_PAD

    def __init__(self, root_dir, diameter=8.2, L0=25.0, wind=12.0, detector=None,
                 pupilstop_tag='vlt_pupil_160pixels', dm_tag='bmc2k_vlt', modes_per_subap=None,
                 n_modes_ho=1200, wavelength_nm=None, optimize_freq=True, optimize_modes=True, alias_layers=None, alias_psd=None):
        self.paths = Paths(root_dir, pupilstop_tag, dm_tag)
        self.wavelength_nm = wavelength_nm
        self.D, self.L0, self.wind = diameter, L0, wind
        self.alias_layers = ((1.0, wind),) if alias_layers is None else alias_layers    # (Cn2, speed) of the aliasing spectrum (A10)
        self._alias = {}
        self.set_alias_psd(alias_psd)
        self.detector = detector if detector is not None else Detector()
        self.modes_per_subap = dict(modes_per_subap or DEFAULT_MODES_PER_SUBAP)
        self.n_modes_ho = n_modes_ho
        self.optimize_freq, self.optimize_modes = optimize_freq, optimize_modes
        self._missing_modes = set()
        self._kl, self._cal, self._calw, self._orders, self._derived = None, {}, {}, {}, set()
        self._basis_ok = set()
        self._modes_cache, self._pupil_cache, self._recs, self._norms = None, None, OrderedDict(), {}
        self._tables = OrderedDict()
        self._rho, self._mode_kern, self._warned = {}, None, set()
        self._fit, self._fit_cross, self._fit_var = {}, None, {}

    # ---- modes per sensor
    def mode_list(self, n_subap):
        """Controlled-mode counts calibrated for a sensor (modes_per_subap value: an int or a list), sorted."""
        v = self.modes_per_subap[n_subap]
        return sorted({int(x) for x in (v if isinstance(v, (list, tuple)) else [v])})

    def max_modes(self, n_subap):
        return self.mode_list(n_subap)[-1]

    def freq_candidates(self, fs):
        """fs and the frame rates below it on the fixed grid FREQ_MIN * FREQ_STEP^k (shared by all calls: the loop tables are cached)."""
        fs = float(fs)
        k = int(np.floor(np.log(fs / FREQ_MIN) / np.log(FREQ_STEP) + 1e-9)) if fs > FREQ_MIN else -1
        grid = {round(FREQ_MIN * FREQ_STEP ** i, 6) for i in range(k + 1)}
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
            path = self.paths.kl_variance(self.L0, self.D, KL_PAD)
            if op.exists(path):
                var, rms = utils.read_fits(path)
            else:
                var, rms = utils.modal_variance_von_karman(1.0, self.L0, self.D, self.modes(), ~(self.pupil() > 0), pad=KL_PAD)
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
            if k != 1.0:                       # back to the units of wl: seeing / k
                cal = _replace(cal, alias_seeing=cal.alias_seeing / k)
            cal.source_wavelength = wl0
            self._calw[key] = cal
        return self._calw[key]

    def _calibration0(self, rMod, n_subap, seeing, N, N_corr, rec, wl0, wl):
        """Calibration from the products of wl0 at the seeing `seeing` of wl0 (alias_seeing in the units of wl0)."""
        s_tag = snap_seeing(seeing)
        simulated = op.exists(self.paths.at(wl0).im_pc(rMod, n_subap, s_tag, N_corr))
        s_key = s_tag if simulated else ('estimated', float(seeing))
        key = (float(rMod), int(n_subap), s_key, int(N), int(N_corr), tuple(rec), wl0, wl)
        if key not in self._cal:
            self._cal[key] = (self._load(rMod, n_subap, s_tag, N, N_corr, rec, wl0, wl) if simulated else
                              self._interpolate(rMod, n_subap, float(seeing), N, N_corr, rec, wl0, wl))
        return self._cal[key]

    def _interpolate(self, rMod, n_subap, seeing, N, N_corr, rec, wl0, wl):
        """Calibration at a seeing that was not simulated: optical gains and noise coefficients by a
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
        kind = {1: 'constant', 2: 'linear'}.get(len(sims), 'quadratic')
        outside = not sims[0] <= seeing <= sims[-1]
        via = '' if wl0 == wl else f' (equivalent seeing of {wl:g} nm)'
        warnings.warn(f'seeing {seeing:g} arcsec{via} not simulated for {P.base(rMod, n_subap)}, {N_corr} modes corrected: optical gains and '
                      f'noise coefficients by {kind} regression over the simulated seeings {sims}'
                      f'{" (EXTRAPOLATED)" if outside else ""}; aliasing and crosstalk from {sims[near]:g} arcsec')
        return Calibration(N, N_corr, og, cals[near].alias_rms, noise, cals[near].mixing, alias_seeing=cals[near].alias_seeing, alias_frames=cals[near].alias_frames, turb_frames=cals[near].turb_frames)

    def slope_geometry(self, rMod, n_subap, seeing, N_corr, wl, frame):
        """Pupil geometry with the PyrSlopec divisor measured from the products of that run (mean slopes against mean frame); the number of
        valid subapertures, with a warning, when it cannot be measured."""
        P = self.paths.at(wl)
        geom = PupilGeometry.from_file(_need(P.pupdata(n_subap), 'pupdata'), n_subap, frame.shape[0])
        key = (P.base(rMod, n_subap), float(seeing), int(N_corr))
        if key not in self._norms:
            path = P.sn(rMod, n_subap, seeing, N_corr)
            n_norm, corr = (measure_slope_norm(frame, utils.read_fits(path), geom) if op.exists(path)
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
            cn, cxy = slope_noise_cov(frame, self.slope_geometry(rMod, n_subap, s0, N, wl0, frame), n_phot, self.detector)
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
        noise = slope_noise_coefficients(rec, frame, self.slope_geometry(rMod, n_subap, seeing, N_corr, wl0, frame))
        apath, spath = P.alias(rMod, n_subap, seeing, N, N_corr), P.slopes(rMod, n_subap, seeing, N_corr)
        if op.exists(apath) and (not op.exists(spath) or (method == 'svd' and scale == 1.0)):
            alias = utils.read_fits(apath) / scale                           # SVD aliasing of wl0: std x wl / wl0
            if method != 'svd':
                warnings.warn(f'{op.basename(spath)} missing: aliasing of the SVD reconstructor used for the {method} one')
        else:
            alias = np.std(rec @ utils.read_fits(_need(spath, 'slopes')).T, axis=1)
        im_path = self._checked(_need(P.im_pc(rMod, n_subap, seeing, N_corr), 'im_pc'))
        tpath = P.turbmodes(rMod, n_subap, seeing, N_corr)
        return Calibration(N, N_corr, og_of(utils.read_fits(im_path)), np.asarray(alias)[:N], noise,
                           lambda: read_rec(rpath) @ (utils.read_fits(im_path)[:, :N] * scale), alias_seeing=seeing,
                           alias_frames=(lambda: read_rec(rpath)[:N] @ utils.read_fits(spath).astype(float).T) if op.exists(spath) else None,
                           turb_frames=(lambda: utils.read_fits(tpath)[:, :10].astype(float)) if op.exists(tpath) else None)

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

    def _loop_tables(self, fs, latency, orders, alias, controller):
        """Gain tables of the loop (int / leaky: LoopTables, leaky over DEFAULT_FF_GRID; wiener: WienerTables), cached."""
        ff_grid = DEFAULT_FF_GRID if controller == 'leaky' else (1.0,)
        key = (round(float(fs), 9), round(float(latency), 9), int(orders.max()), alias.key(), controller, float(self.wind), float(self.D))
        if key in self._tables:
            self._tables.move_to_end(key)
        elif controller == 'wiener':
            from .wiener import WienerTables
            self._tables[key] = WienerTables(fs, latency, self.wind, self.D, np.arange(1, orders.max() + 1), alias, WIENER_ORDER,
                                             n_fold=N_FOLD, max_sensitivity=None)
        else:
            self._tables[key] = LoopTables(fs, latency, self.wind, self.D, np.arange(1, orders.max() + 1), alias, n_fold=N_FOLD,
                                           ff_grid=ff_grid, max_sensitivity=None)
        while len(self._tables) > TABLES_CACHE:
            self._tables.popitem(last=False)
        return self._tables[key]

    # ---- the budget
    def budget(self, rMod, n_subap, seeing, flux, fs, N=None, latency_s=500e-6, controller='int', rec=('svd',), wavelength_nm=None,
               crosstalk=False, extra=None, chromatic=None, optimize_freq=None, optimize_modes=None, warn=True):
        """Budget of one configuration (arguments: see _budget_at). fs and N are MAXIMA: with optimize_freq (machine default) the frame
        rates below fs on the grid FREQ_MIN * FREQ_STEP^k are tried as well, with optimize_modes the calibrated mode counts below N
        (modes_per_subap lists); the result with the lowest total residual is returned, with a warning when it is not (fs, N).
        params: freqInHz / Nmodes = requested, freqUsedInHz / NmodesUsed = used."""
        opt_f = self.optimize_freq if optimize_freq is None else optimize_freq
        opt_m = self.optimize_modes if optimize_modes is None else optimize_modes
        N_req = int(N or self.max_modes(n_subap))
        freqs = self.freq_candidates(fs) if opt_f else [float(fs)]
        modes = sorted({n for n in self.mode_list(n_subap) if n <= N_req} | {N_req}, reverse=True) if opt_m else [N_req]
        kw = dict(latency_s=latency_s, controller=controller, rec=rec, wavelength_nm=wavelength_nm, crosstalk=crosstalk, extra=extra,
                  chromatic=chromatic)
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

    def _budget_at(self, rMod, n_subap, seeing, flux, fs, N=None, latency_s=500e-6, controller='int', rec=('svd',), wavelength_nm=None,
                   crosstalk=False, extra=None, chromatic=None, N_corr=None):
        """Budget of one configuration. flux: photons/s at the detector (before QE); latency_s: end of the exposure -> command applied, any
        real value (the frame of window + hold is in the model); controller 'int', 'leaky' (gain and leak) or 'wiener' (per-mode optimal
        IIR filters on pseudo-open-loop measurements, order WIENER_ORDER; result.filters = [(b, a)], wiener.to_iir_filter converts them);
        rec: ('svd',), ('ml', n_phot) or ('map', n_phot); crosstalk: leak of the residual of the other modes; extra: scalar variances
        [nm^2] added; chromatic: chromatic.ChromaticModel (needs the sensing wavelength). N_corr (internal, budget_dual): the modes
        corrected in the products (default N). The products are those of the seeing rounded to 0.05 arcsec; the aliasing is rescaled to
        the requested seeing."""
        if controller not in CONTROLLERS:
            raise ValueError(f'controller must be one of {CONTROLLERS}, got {controller!r}')
        N = N or self.max_modes(n_subap)
        N_corr = N_corr or N
        wl = self.wavelength_nm if wavelength_nm is None else wavelength_nm
        if chromatic is not None and wl is None:
            raise ValueError('chromatic terms need the sensing wavelength: give wavelength_nm')
        cal = self.calibration(rMod, n_subap, seeing, N, N_corr, rec, wl)
        turb_cmd, rms, r0 = self.turbulence(seeing, N)
        orders = self._radial_orders(N)
        tables = self._loop_tables(fs, latency_s * fs, orders, self.alias_spectrum(wl, N_corr, rMod), controller)
        photon, readout = cal.noise.modal_variance_parts(self.detector.n_det(flux, fs), self.detector)
        noise_cmd = photon + readout
        alias_cmd = cal.alias_rms ** 2 * (seeing / cal.alias_seeing) ** (5 / 3)            # Kolmogorov: alias variance ~ r0^(-5/3)
        extra = dict(extra or {})
        if chromatic is not None:
            extra.update(chromatic.variances(r0, self.L0, utils.fitting_cutoff(N, self.D), wl))
        opt = self._solve(tables, turb_cmd, noise_cmd, alias_cmd, cal.og, orders, cal.xt if crosstalk else None)
        res = self._result(opt, dict(controller=controller, rMod=rMod, seeing=seeing, Nsubaps=n_subap, Nmodes=N, freqInHz=fs,
                                     photPerMs=flux / 1e3, wavelength_nm=np.nan if wl is None else float(wl), crosstalk=bool(crosstalk)),
                           cal.og, rms ** 2, photon / np.maximum(noise_cmd, 1e-300), self._fitting(r0, N), extra, chromatic)
        res.filters = getattr(opt, 'filters', None)                 # controller='wiener': per-mode (b, a) of u = F POL
        return res

    def _radial_orders(self, N):
        if N not in self._orders:
            self._orders[N] = utils.radial_order(np.arange(N))
        return self._orders[N]

    @staticmethod
    def _result(opt, params, og_vec, rms2, photon_fraction, fitting, extra, chromatic):
        noise = opt.noise * rms2
        return BudgetResult(
            params=params, gain=opt.gain, ff=opt.ff, og=og_vec, lag=opt.turb * rms2, noise_photon=noise * photon_fraction,
            noise_ron=noise * (1.0 - photon_fraction), alias=opt.alias * rms2, xtalk=opt.xtalk * rms2, fitting=fitting, extra=extra,
            chromatic=chromatic)

    # ---- fitting (ASSUMPTIONS A15): the bound of the N controlled modes, as a PSD and as a variance
    def mode_kernels(self):
        """Radial PSD kernels of the KL modes (psf.modal_psd_kernels, padding PSD_PAD), cached on disk; psf.py imported only here."""
        if self._mode_kern is None:
            from .psf import load_kernels, modal_psd_kernels, save_kernels
            path = self.paths.kl_psd_kernels(self.D, PSD_PAD)
            if op.exists(path):
                self._mode_kern = load_kernels(path, self.D)
            else:
                self._mode_kern = modal_psd_kernels(self.modes(), ~(self.pupil() > 0), self.D, PSD_PAD)
                save_kernels(path, self._mode_kern)
        return self._mode_kern

    def fitting_kl(self, r0, N):
        """Fitting PSD [nm^2 m^2] on the kernel grid: the first N KL modes removed exactly from the turbulence inside the pupil
        (psf.kl_fitting_psd). The cross kernels (~1 min for 1300 modes) and each N (~20 s) are cached on disk; scales as r0^-5/3."""
        N = int(N)
        if N not in self._fit:
            from .psf import kl_fitting_cross, kl_fitting_psd, load_fitting_cross, save_fitting_cross
            path = self.paths.kl_fitting_psd(self.L0, self.D, PSD_PAD, N)
            if op.exists(path):
                self._fit[N] = np.asarray(utils.read_fits(path), float)
            else:
                if self._fit_cross is None:
                    cpath = self.paths.kl_fitting_cross(self.L0, self.D, PSD_PAD)
                    if op.exists(cpath):
                        self._fit_cross = load_fitting_cross(cpath)
                    else:
                        self._fit_cross = kl_fitting_cross(self.modes(), self.pupil(), self.D, PSD_PAD, self.L0)
                        save_fitting_cross(cpath, self._fit_cross)
                self._fit[N] = kl_fitting_psd(self.modes(), self.pupil(), self.D, PSD_PAD, self._fit_cross, N)
                utils.write_fits(path, self._fit[N])
        return self._fit[N] * r0 ** (-5 / 3)

    def _fitting(self, r0, N):
        """Fitting variance [nm^2] of N modes: the variance of the bound fitting_kl on the grid plus the von Karman power above the grid's
        Nyquist frequency (far field)."""
        N = int(N)
        if N not in self._fit_var:
            from .psf import von_karman_variance_above
            kern = self.mode_kernels()
            self._fit_var[N] = kern.variance(self.fitting_kl(1.0, N)) + von_karman_variance_above(float(kern.k[-1]), 1.0, self.L0)
        return float(self._fit_var[N] * r0 ** (-5 / 3))

    @staticmethod
    def _solve(tables, turb_cmd, noise_cmd, alias_cmd, og_vec, orders, xt=None, n_iter=8, tol=1e-3):
        """Optimise all modes; with `xt` iterate the crosstalk disturbance d = xt @ residual (damped)."""
        opt = tables.optimize(turb_cmd, noise_cmd, alias_cmd, og_vec, orders)
        if xt is None:
            return opt
        d2 = np.zeros_like(turb_cmd)
        for _ in range(n_iter):
            d_new = xt @ opt.resid
            if np.max(np.abs(d_new - d2)) <= tol * max(np.max(d_new), 1e-30):
                return opt
            d2 = 0.5 * (d2 + d_new) if np.any(d2) else d_new
            opt = tables.optimize(turb_cmd, noise_cmd, alias_cmd, og_vec, orders, xtalk_var=d2)
        warnings.warn(f'crosstalk iteration did not converge in {n_iter} passes (tolerance {tol})')
        return opt

    def scan(self, seeings, rMods, n_subaps, fluxes, freqs, controllers=('int',), store_modes=None, progress=False, wavelengths=(None,), **kw):
        """Cartesian scan -> (modes_df, totals_df). store_modes: None, 'all' or a list of mode ids; extra kwargs go to budget()."""
        totals, modes, last = [], [], None
        for wl, rMod, seeing, n_sub, ctl, fs, flux in product(wavelengths, rMods, seeings, n_subaps, controllers, freqs, fluxes):
            key = (wl, rMod, seeing, n_sub)
            if progress and key != last:
                last = key
                print(f'wvl={wl} rMod={rMod} seeing={seeing} n_subap={n_sub}', flush=True)
            r = self.budget(rMod, n_sub, seeing, flux, fs, controller=ctl, wavelength_nm=wl, warn=False, **kw)
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

    # ---- multi-sensor fusion
    def alias_correlation(self, cals, N):
        """(N, K, K) per-mode correlation of the aliasing seen by the K sensors of `cals` (Calibration objects), from their aliasing runs:
        the generator gives every configuration the same seeds, hence the same turbulence frames, so the modal aliasing signals R_i @ s_i
        pair frame by frame. None when a run is missing or the runs are not paired (their turbulence modes differ)."""
        key = tuple(id(c) for c in cals) + (int(N),)
        if key not in self._rho:
            rho = None
            if all(c.alias_frames is not None and c.turb_frames is not None for c in cals):
                T = [c.turb_frames() for c in cals]
                paired = all(len(t) == len(T[0]) for t in T) and all(
                    min(np.corrcoef(T[0][:, j], t[:, j])[0, 1] for j in range(T[0].shape[1]) if T[0][:, j].std() > 0) > 0.999 for t in T[1:])
                if paired:
                    A = [np.asarray(c.alias_frames(), float)[:N] for c in cals]
                    A = [x - x.mean(axis=1, keepdims=True) for x in A]
                    A = [x / np.maximum(np.sqrt((x ** 2).mean(axis=1, keepdims=True)), 1e-300) for x in A]
                    rho = np.einsum('imf,jmf->mij', np.stack(A), np.stack(A)) / A[0].shape[1]
            self._rho[key] = rho
        return self._rho[key]

    def shear_covariance(self, chromatic, wavelengths, N, seeing):
        """(N, K, K) covariance [nm^2 OPD, unit-rms modes] of the chromatic shear delta_i = phi(sensing beam i) - phi(science beam) of K
        sensing wavelengths on the first N KL modes. Per layer the beams are displaced by d_i = h tan z (n_i - n_science) along the
        elevation, so E[delta_i delta_j](k) = Phi(k) sum_l w_l [1 - J0(2 pi k d_il) - J0(2 pi k d_jl) + J0(2 pi k (d_il - d_jl))]
        (TipTop's differentialRefractionPSD for i = j), projected on each mode with its kernel (azimuthal mean, A28)."""
        from scipy.special import j0
        kern = self.mode_kernels()
        cm = _replace(chromatic, delta_n=None)
        k = kern.k
        phi = utils.von_karman_psd(k, float(utils.r0_from_seeing(seeing)), self.L0)
        d = [cm.layer_shifts(float(wl)) for wl in wavelengths]
        J = lambda x: j0(2 * np.pi * np.asarray(x)[:, None] * k[None])
        S = np.empty((len(d), len(d), len(k)))
        for i in range(len(d)):
            for j in range(i, len(d)):
                S[i, j] = S[j, i] = phi * (cm.cn2[:, None] * (1.0 - J(d[i]) - J(d[j]) + J(d[i] - d[j]))).sum(0)
        pup = self.pupil() > 0
        area = pup.sum() * (self.D / pup.shape[0]) ** 2
        return np.einsum('mb,ijb->mij', kern.K[:N].astype(float) * kern.count * kern.df ** 2 / area, S)

    def fusion_inputs(self, sensors, seeing, N, rec=('svd',)):
        """Per-mode inputs of a fusion, in optical-gain-compensated command units: photon and readout noise variances per frame at each
        sensor's own rate (N, K), aliasing rms (N, K) at `seeing`, its correlation (N, K, K) measured from the paired aliasing runs
        (alias_correlation; fully correlated, with a warning, when they are not paired) and the calibrations. sensors as budget_fused,
        each with its frame rate 'fs'."""
        cals, phot, ron, alias, og = [], [], [], [], []
        for s in sensors:
            n_rec = int(s.get('N_rec') or N)                     # modes of the sensor's reconstructor (>= N; only the first N are fused)
            cal = self.calibration(s['rMod'], s['n_subap'], seeing, n_rec, int(s.get('N_corr') or n_rec), rec, s['wavelength_nm'])
            if not s.get('fs'):
                raise ValueError('fusion_inputs: every sensor needs its frame rate fs')
            g = np.maximum(cal.og[:N], 1e-3)
            p, rn = cal.noise.modal_variance_parts(self.detector.n_det(s['flux'], float(s['fs'])), self.detector)
            cals.append(cal), og.append(g), phot.append(p[:N] / g ** 2), ron.append(rn[:N] / g ** 2)
            alias.append(cal.alias_rms[:N] * (seeing / cal.alias_seeing) ** (5 / 6) / g)
        K = len(sensors)
        rho, measured = (self.alias_correlation(cals, N), True) if K > 1 else (None, False)
        if rho is None:
            if K > 1:
                key = ('rho', tuple(f"{s['rMod']}_{s['n_subap']}_{s['wavelength_nm']}" for s in sensors))
                if key not in self._warned:
                    self._warned.add(key)
                    warnings.warn('budget_fused: the aliasing runs of these sensors are missing or not paired (different turbulence frames): '
                                  'aliasing taken as fully correlated')
            rho, measured = np.ones((N, K, K)), False
        return dict(cals=cals, og=og, photon=np.stack(phot, 1), readout=np.stack(ron, 1), alias=np.stack(alias, 1), rho=rho,
                    alias_correlation='measured' if measured else 'full')

    @staticmethod
    def _gls_weights(noise, C):
        """Per-mode GLS weights (N, K) for the error covariance diag(noise) + C (noise (N, K), C (N, K, K))."""
        K = noise.shape[1]
        Sig = np.einsum('mi,ij->mij', noise, np.eye(K)) + C
        Sig = Sig + 1e-12 * np.einsum('mii->m', Sig)[:, None, None] * np.eye(K)
        x = np.linalg.solve(Sig, np.ones(Sig.shape[:2])[..., None])[..., 0]
        return x / x.sum(axis=1, keepdims=True)

    def budget_fused(self, sensors, seeing, fs, N=None, latency_s=500e-6, controller='int', rec=('svd',), crosstalk=False, chromatic=None,
                     optimize_freq=None, optimize_modes=None, warn=True):
        """Budget of ONE loop fed by several sensors at the frame rate fs, whose measurements of every mode are combined with static
        per-mode weights summing to 1 (ASSUMPTIONS A39). sensors: [dict(rMod, n_subap, wavelength_nm, flux)], flux in photons/s at that
        sensor's detector (each sensor has its own detector: the readout noise is paid once per sensor). Everything in
        optical-gain-compensated units. The weights are GLS on the per-mode error covariance of the sensors,
            Sigma = diag(sigma_n^2) + C_alias + C_shear,      w = Sigma^-1 1 / (1^T Sigma^-1 1),
        noise independent between sensors, C_alias = rho a_i a_j with rho measured from the paired aliasing runs, C_shear the covariance
        of the chromatic shear of the sensing beams with respect to the science one (with `chromatic`, shear_covariance), so the weights
        extrapolate towards the science wavelength when that pays; then the loop of `controller` is optimised as for one sensor. The
        fused shear is reported as chrom_shear and per mode (result.chrom_shear_modes, used by PsfEngine); the other chromatic terms use
        the refractivity of the fused beam. Crosstalk: the fused leak matrix sum_i diag(w_i / og_i) (R IM_PC)_i. fs and N are maxima as in
        budget(). Optional per-sensor keys (used by budget_dual): 'fs' the sensor frame rate when it differs from the loop rate fs (its
        noise enters with the same noise PSD at the loop rate: per-frame variance x fs / fs_i; its longer exposure and delay are NOT
        modelled, which is optimistic for a slower sensor); 'N_rec' the modes of its reconstructor (>= N, the first N are fused);
        'N_corr' the corrected modes of its products (default N_rec)."""
        sensors = [dict(s) for s in sensors if s['flux'] > 0]
        if not sensors:
            raise ValueError('budget_fused: no sensor with a positive flux')
        opt_f = self.optimize_freq if optimize_freq is None else optimize_freq
        opt_m = self.optimize_modes if optimize_modes is None else optimize_modes
        N_req = int(N or min(self.max_modes(s['n_subap']) for s in sensors))
        common = set.intersection(*(set(self.mode_list(s['n_subap'])) for s in sensors))
        modes = sorted({n for n in common if n <= N_req} | {N_req}, reverse=True) if opt_m else [N_req]
        freqs = self.freq_candidates(fs) if opt_f else [float(fs)]
        best = first = None
        for n_m in modes:
            for f in freqs:
                r = self._budget_fused_at(sensors, seeing, f, n_m, latency_s, controller, rec, crosstalk, chromatic)
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

    def _budget_fused_at(self, sensors, seeing, fs, N, latency_s, controller, rec, crosstalk, chromatic):
        if controller not in CONTROLLERS:
            raise ValueError(f'controller must be one of {CONTROLLERS}, got {controller!r}')
        turb_cmd, rms, r0 = self.turbulence(seeing, N)
        orders = self._radial_orders(N)
        sensors = [{**s, 'fs': float(s.get('fs') or fs)} for s in sensors]
        inp = self.fusion_inputs(sensors, seeing, N, rec)
        cals, og, rho = inp['cals'], inp['og'], inp['rho']
        scale = np.array([fs / s['fs'] for s in sensors])                         # slower sensor: equal noise PSD at the loop rate fs
        phot, ron, a = inp['photon'] * scale, inp['readout'] * scale, inp['alias']
        K = len(sensors)
        C_alias = rho * a[:, :, None] * a[:, None, :]
        wls = [float(s['wavelength_nm']) for s in sensors]
        rms2 = rms ** 2
        C_shear = (self.shear_covariance(chromatic, wls, N, seeing) / rms2[:, None, None] if chromatic is not None
                   else np.zeros((N, K, K)))
        w = self._gls_weights(np.maximum(phot + ron, 1e-300), C_alias + C_shear)
        photon, readout = (w ** 2 * phot).sum(1), (w ** 2 * ron).sum(1)
        alias_cmd = np.einsum('mi,mij,mj->m', w, C_alias, w)
        shear_modes = np.einsum('mi,mij,mj->m', w, C_shear, w) * rms2
        xt = None
        if crosstalk:
            M = sum((w[:, i] / og[i])[:, None] * cals[i].mixing()[:N, :N] for i in range(K))
            xt = (M ** 2).astype(np.float32)
            np.fill_diagonal(xt, 0.0)
        w_main = int(np.argmax(np.sum(1.0 / np.maximum(phot + ron, 1e-300), axis=0)))
        tables = self._loop_tables(fs, latency_s * fs, orders, self.alias_spectrum(wls[w_main], N, sensors[w_main].get('rMod')), controller)
        one = np.ones(N)
        opt = self._solve(tables, turb_cmd, photon + readout, alias_cmd, one, orders, xt)
        wbar = (w * (turb_cmd * rms2)[:, None]).sum(0) / np.sum(turb_cmd * rms2)          # turbulence-weighted weight of each sensor
        wl_eff = float(np.sum(wbar * np.asarray(wls)))
        extra, bands = {}, None
        if chromatic is not None:
            n_eff = float(np.sum(wbar * np.array([float(chromatic.n_minus_1(x)) for x in wls])))
            cm = _replace(chromatic, delta_n=n_eff - float(chromatic.n_minus_1(chromatic.science_nm)))
            k_cut = utils.fitting_cutoff(N, self.D)
            extra.update(cm.variances(r0, self.L0, k_cut, wl_eff))
            extra['chrom_shear'] = float(shear_modes.sum())
            bands = [(cm, 0.0, k_cut, wl_eff)]
        desc = ' + '.join(f"{s['n_subap']}x{s['n_subap']} rMod {s['rMod']:g} {s['wavelength_nm']:.0f}nm"
                          + (f" {float(s['fs']):.0f}Hz" if abs(s['fs'] - fs) > 1e-9 else '') for s in sensors)
        params = dict(controller=controller, rMod=np.nan, seeing=seeing, Nsubaps=min(s['n_subap'] for s in sensors), Nmodes=N, freqInHz=fs,
                      photPerMs=sum(s['flux'] for s in sensors) / 1e3, wavelength_nm=wl_eff, crosstalk=bool(crosstalk), sensors=desc,
                      alias_correlation=inp['alias_correlation'], **{f'weight_{i}': float(x) for i, x in enumerate(wbar)})
        res = self._result(opt, params, one, rms2, photon / np.maximum(photon + readout, 1e-300), self._fitting(r0, N), extra, None)
        res.chrom_bands, res.weights, res.alias_rho = bands, w.T.copy(), rho
        res.chrom_shear_modes = shear_modes if chromatic is not None else None
        res.filters = getattr(opt, 'filters', None)
        return res

    def budget_dual(self, lo, ho, seeing, latency_s=500e-6, controller='int', crosstalk=False, chromatic=None, rec=('svd',), r_ho=None):
        """Two sensors with their measurements FUSED: the first lo['N'] modes are controlled by one loop at the LO frame rate lo['fs'] fed by
        both sensors (GLS weights of budget_fused; the LO sensor with the products of a run with n_modes_ho modes corrected, the HO sensor
        with its own n_modes_ho-mode reconstructor at its frame rate ho['fs'], equal-noise-PSD approximation), the modes lo['N'] ...
        n_modes_ho - 1 by the HO sensor alone at ho['fs']. Frame rates are used as given. r_ho: the HO budget if already computed.
        Returns the spliced BudgetResult (.stages = (fused LO-mode result, HO result), .weights of the LO modes). Chromatic shear: modal,
        w^T C_shear w on the LO modes and the HO sensor's own on the others; the other chromatic terms of the LO band use the effective
        refractivity of the fused beam."""
        n_ho, n_lo = self.n_modes_ho, int(lo['N'])
        if r_ho is None:
            r_ho = self.budget(ho['rMod'], ho['n_subap'], seeing, ho['flux'], ho['fs'], N=n_ho, latency_s=latency_s, controller=controller,
                               rec=rec, wavelength_nm=ho['wavelength_nm'], crosstalk=crosstalk, optimize_freq=False, optimize_modes=False,
                               warn=False)
        sensors = [{**lo, 'N_rec': n_lo, 'N_corr': n_ho, 'fs': lo['fs']}, {**ho, 'N_rec': n_ho, 'N_corr': n_ho}]
        r_low = self._budget_fused_at(sensors, seeing, float(lo['fs']), n_lo, latency_s, controller, rec, crosstalk, chromatic)
        r_low.params['freqUsedInHz'] = float(lo['fs'])
        cm_lo = None
        if chromatic is not None:
            n_eff = sum(r_low.params[f'weight_{i}'] * chromatic.n_minus_1(s['wavelength_nm']) for i, s in enumerate(sensors))
            cm_lo = _replace(chromatic, delta_n=float(n_eff - chromatic.n_minus_1(chromatic.science_nm)))
        res = self.combine_stages(r_low, r_ho, {**lo, 'wavelength_nm': r_low.params['wavelength_nm']}, ho, seeing, chromatic,
                                  chromatic_lo=cm_lo)
        res.weights = r_low.weights
        return res

    def combine_stages(self, r_lo, r_ho, lo, ho, seeing, chromatic=None, chromatic_lo=None):
        """Splice of an LO result (its N modes) and an HO result (n_modes_ho modes): modes < N from the LO stage, the others and the
        fitting from the HO stage, chromatic shear by spatial-frequency band (budget_dual; call this directly to reuse cached stage
        results). lo, ho: the sensor dicts (wavelength_nm, flux). chromatic_lo: model of the LO band (default chromatic), e.g. with the
        delta_n of a fused beam."""
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
                hi = cm.variances(r0, self.L0, b, wl)
                low = cm.variances(r0, self.L0, a, wl) if a > 0 else {}
                for name, v in hi.items():
                    extra[name] = extra.get(name, 0.0) + v - low.get(name, 0.0)
            if getattr(r_lo, 'chrom_shear_modes', None) is not None:          # fused LO stage: modal shear, HO modes from the HO sensor alone
                shear_modes = np.concatenate([r_lo.chrom_shear_modes[:n],
                                              self.shear_covariance(chromatic, [float(ho['wavelength_nm'])], n_ho, seeing)[n:, 0, 0]])
                extra['chrom_shear'] = float(shear_modes.sum())
        tag = lambda s, r: f"{s['n_subap']}x{s['n_subap']} rMod {s['rMod']:g} {s['wavelength_nm']:.0f}nm {r.params['freqUsedInHz']:.0f}Hz"
        params = {**r_ho.params, 'scenario': 'dual', 'N_lo': n, 'freqUsedInHz_lo': r_lo.params['freqUsedInHz'],
                  'photPerMs': (lo['flux'] + ho['flux']) / 1e3, 'wavelength_nm': np.nan,
                  'sensors': f'LO {n} modes: {tag(lo, r_lo)} | HO: {tag(ho, r_ho)}'}
        res = _replace(r_ho, params=params, extra=extra, chromatic=None, chrom_bands=bands,
                       chrom_shear_modes=shear_modes if chromatic is not None and getattr(r_lo, 'chrom_shear_modes', None) is not None else None,
                       **fields)
        res.stages = (r_lo, r_ho)
        return res
