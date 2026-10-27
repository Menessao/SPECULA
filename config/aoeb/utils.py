"""Helpers for the error budget (corrected versions of the previous utils.py).

Conventions
-----------
* OPD in nm, r0 at 500 nm, frequencies in cycles/m, phase PSD in nm^2 m^2.
* KL modes are given as stored (command units): the DM command that multiplies the mode.
  Their rms OPD per command unit is `rms`; variances are returned both ways (see
  `modal_variance_von_karman`).
"""
import os
import os.path as op

import numpy as np
import scipy.fft as sfft
from astropy.io import fits

ARCSEC_PER_RAD = 180.0 * 3600.0 / np.pi
LAM0_NM = 500.0
RAD2_TO_NM2 = (LAM0_NM / (2 * np.pi)) ** 2          # rad^2 at 500 nm -> nm^2 OPD
VK_COEFF = 0.02289558710855519                        # von Karman phase PSD coefficient


# ----------------------------------------------------------------------------- seeing / atmosphere
def r0_from_seeing(seeing_arcsec):
    """Fried parameter [m] at 500 nm from the seeing FWHM [arcsec]."""
    return 0.98 * 500e-9 / (np.asarray(seeing_arcsec) / ARCSEC_PER_RAD)


def radial_order(i_mode):
    """Radial order n of KL/Zernike-like mode i (0-based; mode 0 is the first tilt, n = 1)."""
    noll = np.asarray(i_mode) + 2
    return np.ceil(-1.5 + np.sqrt(1 + 8 * noll) / 2).astype(int)


def effective_wind(speeds, cn2):
    """5/3-moment effective wind speed of a multi-layer profile [m/s]."""
    v, w = np.asarray(speeds, float), np.asarray(cn2, float)
    return float((np.sum(w * v ** (5 / 3)) / np.sum(w)) ** (3 / 5))


def von_karman_psd(f, r0, L0):
    """Phase PSD [nm^2 m^2] at spatial frequency f [1/m] (von Karman, r0 at 500 nm)."""
    return VK_COEFF * r0 ** (-5 / 3) * (np.asarray(f) ** 2 + 1 / L0 ** 2) ** (-11 / 6) * RAD2_TO_NM2


def square_outside_fraction(k, kc):
    """Fraction of the circle of radius k outside the square |kx|, |ky| <= kc: the azimuthal average of TipTop's corrected-area mask."""
    k = np.asarray(k, float)
    frac = (4.0 / np.pi) * np.arccos(np.clip(kc / np.maximum(k, 1e-300), 0.0, 1.0))
    return np.where(k <= kc, 0.0, np.where(k >= np.sqrt(2.0) * kc, 1.0, frac))


def fitting_variance_tiptop(r0, L0, kc, n=4000):
    """Fitting error variance [nm^2] as TipTop (P3 fittingPSD): the von Karman PSD outside the corrected square |kx|, |ky| <= kc
    (kc = 1 / (2 DM pitch)); exact radial integral (analytic beyond sqrt(2) kc)."""
    k = np.linspace(kc, np.sqrt(2.0) * kc, n)
    corners = np.trapezoid(von_karman_psd(k, r0, L0) * square_outside_fraction(k, kc) * 2 * np.pi * k, k)
    beyond = VK_COEFF * r0 ** (-5 / 3) * RAD2_TO_NM2 * (6 * np.pi / 5) * (2.0 * kc ** 2 + 1 / L0 ** 2) ** (-5 / 6)
    return float(corners + beyond)


def fitting_variance_lbt(r0, n_modes, D):
    """Fitting error variance [nm^2], LBT-ASM law 0.2778 N^-0.9 (D/r0)^(5/3) rad^2."""
    return 0.2778 * n_modes ** (-0.9) * (D / r0) ** (5 / 3) * RAD2_TO_NM2


# ----------------------------------------------------------------------------- FITS helpers
def read_fits(path, ext=None, hint=None):
    """Array of the first HDU with data (or of `ext`); `hint` names the step that creates a missing file."""
    try:
        return np.asarray(fits.getdata(path) if ext is None else fits.getdata(path, ext=ext))
    except FileNotFoundError as e:
        raise FileNotFoundError(f'{path} not found' + (f' (created by: {hint})' if hint else '')) from e


def write_fits(path, data, overwrite=True, header=None):
    os.makedirs(op.dirname(op.abspath(path)), exist_ok=True)
    fits.writeto(path, np.asarray(data), header, overwrite=overwrite)


# ----------------------------------------------------------------------------- pupil geometry
def load_pupdata(path):
    """PupData file -> (ids, radii); ids (n_pix, 4): flat detector indices of the four pupils, padded with -1."""
    with fits.open(path) as hdul:
        return np.asarray(hdul[1].data).astype(np.int64), np.asarray(hdul[2].data)


def pupil_indices(ids):
    """Valid (>= 0) indices of each of the four pupils."""
    return [ids[:, j][ids[:, j] >= 0] for j in range(ids.shape[1])]


# ----------------------------------------------------------------------------- KL modal variances
def check_pupil(pupil):
    """dx = D / n assumes the pupil touches all four edges of its array."""
    p = np.asarray(pupil) > 0
    if not (p.any(axis=0).all() and p.any(axis=1).all()):
        raise ValueError('the pupil does not touch all four edges of its array: the pixel scale D / n would be wrong')


def fitting_cutoff(N, D):
    """Cut-off frequency [cycles/m] of N corrected modes: rMax = sqrt(4N/pi) actuators across, k_c = rMax / (2D)."""
    return np.sqrt(4.0 * N / np.pi) / (2.0 * D)


def half_plane_weights(M):
    """Multiplicity of each column of an rfft2 half plane (1 for the zero and Nyquist columns, else 2)."""
    w = np.full(M // 2 + 1, 2.0, np.float32)
    w[0] = 1.0
    if M % 2 == 0:
        w[-1] = 1.0
    return w


def modal_variance_von_karman(r0, L0, D, modes, mask, pad=8, workers=-1):
    """Turbulent OPD variance of each mode [nm^2] (von Karman) and its rms: (var, rms); modes (n_modes, n_valid), mask True outside."""
    modes = np.asarray(modes, float)
    n = mask.shape[0]
    rows, cols = np.nonzero(~mask)
    dx, M = D / n, n * pad
    rms = np.sqrt(np.mean(modes ** 2, axis=1))
    f, fr = np.fft.fftfreq(M, d=dx), np.fft.rfftfreq(M, d=dx)
    phi = (von_karman_psd(np.hypot(f[:, None], fr[None, :]), r0, L0) * half_plane_weights(M)[None, :]).astype(np.float32)
    df, area = 1.0 / (M * dx), (~mask).sum() * dx * dx
    chunk = max(1, int(1e8 / (M * M * 4)))
    var = np.empty(len(modes))
    for s in range(0, len(modes), chunk):
        buf = np.zeros((min(chunk, len(modes) - s), n, n), np.float32)
        buf[:, rows, cols] = modes[s:s + chunk] / rms[s:s + chunk, None]
        ft = sfft.rfft2(buf, s=(M, M), workers=workers)                      # zero padding inside the FFT
        var[s:s + chunk] = ((ft.real ** 2 + ft.imag ** 2) * phi).sum(axis=(1, 2), dtype=np.float64) * dx ** 4 * df ** 2 / area ** 2
    return var, rms


def kl_modes_from_inverse(kl_inv):
    """KL mode shapes (n_modes, n_valid) from the stored inverse: the pseudo-inverse via the Gram matrix (checked, else pinv)."""
    a = np.asarray(kl_inv, float)
    try:
        modes = np.linalg.solve(a @ a.T, a)
        if np.max(np.abs(a @ modes.T - np.eye(len(a)))) < 1e-6:
            return modes
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(a).T


def compare_modal_variance(sim_modes, var_cmd):
    """Simulated vs analytical modal variance (command units)."""
    sim = np.var(np.asarray(sim_modes), axis=0)
    n = min(len(sim), len(var_cmd))
    return sim[:n] / np.asarray(var_cmd)[:n]
