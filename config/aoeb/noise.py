"""Detector and pyramid slope-noise model: pixel variance F2 n_k + ron_e^2 (ron_e input-referred, F2 = 1 + 1/delta with EM excess
noise); PyrSlopec slopes s = (A + B - C - D) n_norm / I_total, with n_norm = PupData.n_subap. The divisor n_norm is MEASURED from the
products (measure_slope_norm: mean slopes against the mean frame of the same noise-free runs); the number of valid subapertures is the
fallback (PyrSlopec calls I_total / n_subap the counts per subaperture)."""
from dataclasses import dataclass

import numpy as np

from .utils import load_pupdata, pupil_indices


@dataclass(frozen=True)
class Detector:
    ron_e: float = 0.5               # input-referred readout noise [e- rms] (CCD `readout_level`)
    qe: float = 1.0                  # quantum efficiency x transmission (CCD `quantum_eff`)
    excess_noise: bool = False       # EM multiplication noise on?
    excess_delta: float = 1.0        # SPECULA `excess_delta`; F2 = 1 + 1/delta

    @property
    def f2(self):
        return 1.0 + 1.0 / self.excess_delta if self.excess_noise else 1.0

    def n_det(self, flux, fs):
        """Detected electrons per frame on the whole detector, from flux [photons/s at the detector]."""
        return flux * self.qe / fs


@dataclass(frozen=True)
class PupilGeometry:
    idx: tuple            # four index arrays (flat detector indices), same length
    n_norm: float         # the divisor PyrSlopec uses (PupData.n_subap): measured from the products, see measure_slope_norm
    npix: int

    @classmethod
    def from_file(cls, pupdata_path, n_norm, npix):
        ids, _ = load_pupdata(pupdata_path)
        idx = pupil_indices(ids)
        n = min(len(i) for i in idx)                     # PyrSlopec truncates to the shortest pupil
        return cls(tuple(i[:n] for i in idx), float(n_norm), int(npix))

    @property
    def n_valid(self):
        return len(self.idx[0])

    def mask(self):
        m = np.zeros(self.npix * self.npix, dtype=bool)
        for i in self.idx:
            m[i] = True
        return m.reshape(self.npix, self.npix)


def pixel_fractions(frame, geom):
    """Fractions of the detector counts in the four pupils (A, B, C, D) and the throughput (fraction of all counts inside the pupils)."""
    f = np.asarray(frame, float).ravel()
    f = f / f.sum()
    parts = [f[i] for i in geom.idx]
    return parts, float(sum(p.sum() for p in parts))


@dataclass(frozen=True)
class SlopeNoise:
    a: np.ndarray        # (N,) photon coefficient
    b: np.ndarray        # (N,) readout coefficient
    c_pix: int           # pixels per slope (4 quad-cell, 1 intensity)
    n_norm: float
    thrp: float

    def modal_variance_parts(self, n_det, detector):
        """(photon, readout) noise variance of each mode [command^2 per frame]; n_det = detected e-/frame."""
        fm = n_det * self.thrp / self.n_norm
        return (detector.f2 * n_det * self.a / fm ** 2, self.c_pix * detector.ron_e ** 2 * self.b / fm ** 2)

    def modal_variance(self, n_det, detector):
        """Noise variance of each mode [command^2 per frame], n_det = detected e-/frame (whole frame)."""
        ph, ron = self.modal_variance_parts(n_det, detector)
        return ph + ron


def slope_noise_coefficients(rec, frame, geom, slopes_from_intensity=False):
    """Coefficients (a_i, b_i) of the modal noise for reconstructor `rec` (N, n_slopes)."""
    (fa, fb, fc, fd), thrp = pixel_fractions(frame, geom)
    n = geom.n_valid
    rec = np.asarray(rec, float)
    if slopes_from_intensity:
        if rec.shape[1] != 4 * n:
            raise ValueError(f'reconstructor has {rec.shape[1]} slopes, expected {4 * n} (intensity mode)')
        r2 = rec ** 2
        a = r2 @ np.concatenate([fa, fb, fc, fd])
        return SlopeNoise(a, r2.sum(axis=1), 1, geom.n_norm, thrp)
    if rec.shape[1] != 2 * n:
        raise ValueError(f'reconstructor has {rec.shape[1]} slopes, expected {2 * n} (quad-cell)')
    rx, ry = rec[:, :n], rec[:, n:]
    f4 = fa + fb + fc + fd
    cross = fb + fd - fa - fc                           # photon covariance of sx_j and sy_j
    a = (rx ** 2 + ry ** 2) @ f4 + 2.0 * (rx * ry) @ cross
    b = (rx ** 2 + ry ** 2).sum(axis=1)
    return SlopeNoise(a, b, 4, geom.n_norm, thrp)


def slope_noise_cov(frame, geom, n_det, detector, slopes_from_intensity=False):
    """Slope noise covariance, the weights of the ML / MAP reconstructors: (var, cov_xy).
    Quad-cell: var (2 n_valid,) = [var sx, var sy] and cov_xy (n_valid,) = cov(sx_j, sy_j) = F2 n_det (fB + fD - fA - fC) / fm^2, the
    photon covariance of the two slopes of a subaperture (the readout part cancels: A, B, C, D enter sx sy with signs -, +, -, +); slopes
    of different subapertures are independent. Intensity mode (slopes_from_intensity): each slope is one pixel, var (4 n_valid,) and
    cov_xy None (exact for independent pixels; the common normalisation by the total flux is neglected, as in the quad-cell case)."""
    (fa, fb, fc, fd), thrp = pixel_fractions(frame, geom)
    fm = n_det * thrp / geom.n_norm
    ron2 = detector.ron_e ** 2
    if slopes_from_intensity:
        return (detector.f2 * n_det * np.concatenate([fa, fb, fc, fd]) + ron2) / fm ** 2, None
    v4 = detector.f2 * n_det * (fa + fb + fc + fd) + 4 * ron2
    return np.concatenate([v4, v4]) / fm ** 2, detector.f2 * n_det * (fb + fd - fa - fc) / fm ** 2


def slope_variance_diag(frame, geom, n_det, detector, slopes_from_intensity=False):
    """Variance of each slope (the diagonal of slope_noise_cov)."""
    return slope_noise_cov(frame, geom, n_det, detector, slopes_from_intensity)[0]


def measure_slope_norm(frame, sn, geom, slopes_from_intensity=False):
    """(n_norm, correlation): the PyrSlopec divisor, fitted on mean slopes `sn` against the same normalised differences computed from the
    mean frame of the same runs (sn = n_norm (A + B - C - D) / I, x then y; intensity mode: n_norm [A, B, C, D] / I). A low correlation
    means the slopes are not built as assumed (ordering, definition)."""
    f = np.asarray(frame, float).ravel()
    a, b, c, d = (f[i] for i in geom.idx)
    total = float((a + b + c + d).sum())
    pred = (np.concatenate([a, b, c, d]) if slopes_from_intensity else np.concatenate([a + b - c - d, b + c - a - d])) / total
    sn = np.asarray(sn, float).ravel()
    if len(sn) != len(pred) or not np.any(pred) or not np.any(sn):
        return float('nan'), float('nan')
    return float(sn @ pred / (pred @ pred)), float(np.corrcoef(sn, pred)[0, 1])
