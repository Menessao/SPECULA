"""Detector and pyramid slope-noise model: pixel variance F2 n_k + ron_e^2 (ron_e input-referred, F2 = 1 + 1/delta with EM excess
noise); PyrSlopec quad-cell slopes normalised by the mean counts per subaperture (n_norm = PupData.n_subap)."""
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
    n_norm: float         # PupData.n_subap, the divisor PyrSlopec uses (pupil diameter in subaps)
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


def slope_variance_diag(frame, geom, n_det, detector, slopes_from_intensity=False):
    """Variance of each slope (diagonal; the sx-sy photon covariance is neglected): the noise weights of the ML / MAP reconstructors."""
    (fa, fb, fc, fd), thrp = pixel_fractions(frame, geom)
    fm = n_det * thrp / geom.n_norm
    ron2 = detector.ron_e ** 2
    if slopes_from_intensity:
        v = detector.f2 * n_det * np.concatenate([fa, fb, fc, fd]) + ron2
    else:
        v4 = detector.f2 * n_det * (fa + fb + fc + fd) + 4 * ron2
        v = np.concatenate([v4, v4])
    return v / fm ** 2
