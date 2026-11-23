"""Radial PSD of the residual terms [nm^2 m^2, k in cycles/m, separation = k D in lambda/D] and long-exposure PSFs.
Modal terms use pupil-windowed kernels, stationary terms analytic PSDs; the PSF is OTF_tel exp(-[C(0) - C(r)]). ASSUMPTIONS.md A28."""
import os
import os.path as op
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft
from astropy.io import fits
from scipy.special import gamma, kv

from . import utils


# ----------------------------------------------------------------------------- modal kernels
@dataclass
class ModalKernels:
    k: np.ndarray            # (nb,) radial frequency of the annulus centres [cycles/m]
    K: np.ndarray            # (n_modes, nb) float32, annulus mean of |FT(P m_i)|^2 / A [m^2]
    count: np.ndarray        # (nb,) number of grid points of each annulus
    df: float                # frequency step [cycles/m]
    D: float

    def psd(self, var_opd):
        """Radial PSD [nm^2 m^2] of modal variances `var_opd` [nm^2] (first len(var_opd) modes)."""
        v = np.asarray(var_opd, float)
        return v @ self.K[:len(v)].astype(float)

    def variance(self, psd):
        """Variance [nm^2] of a radial PSD defined on `k` (exact sum over the 2-D grid)."""
        return float(np.sum(np.asarray(psd) * self.count) * self.df ** 2)


def modal_psd_kernels(modes, mask, D, pad=4, workers=-1):
    """Radial kernels of the KL modes (mask True OUTSIDE the pupil); each mode is normalised to unit rms, so K_i integrates to 1."""
    modes = np.asarray(modes, float)
    n = mask.shape[0]
    rows, cols = np.nonzero(~mask)
    dx, M = D / n, n * pad
    df, area = 1.0 / (M * dx), (~mask).sum() * dx * dx
    rms = np.sqrt(np.mean(modes ** 2, axis=1))
    f, fr = np.fft.fftfreq(M, d=dx), np.fft.rfftfreq(M, d=dx)
    nb = M // 2 + 1
    b = np.minimum(np.rint(np.hypot(f[:, None], fr[None, :]) / df).astype(int), nb).ravel()      # corners (k > Nyquist) -> bin nb
    w = np.broadcast_to(utils.half_plane_weights(M)[None, :], (M, nb)).ravel()
    order = np.argsort(b, kind='stable')
    starts = np.searchsorted(b[order], np.arange(nb + 1))
    count = np.add.reduceat(w[order], starts[:-1])[:nb].astype(float)
    sel = starts[:-1] < starts[1:]
    kernels = np.empty((len(modes), nb), np.float32)
    chunk = max(1, int(1e8 / (M * nb * 4)))
    for s in range(0, len(modes), chunk):
        buf = np.zeros((min(chunk, len(modes) - s), n, n), np.float32)
        buf[:, rows, cols] = modes[s:s + chunk] / rms[s:s + chunk, None]
        ft = sfft.rfft2(buf, s=(M, M), workers=workers)
        p = ((ft.real ** 2 + ft.imag ** 2) * w.reshape(M, nb)[None]).reshape(len(buf), -1)[:, order] * np.float32(dx ** 4 / area)
        kernels[s:s + len(buf)] = np.add.reduceat(p, starts[:-1][sel], axis=1)[:, :nb] / np.maximum(count, 1)
    return ModalKernels(np.arange(nb) * df, kernels, count, df, float(D))


def save_kernels(path, kern):
    utils.write_fits(path, np.concatenate([np.vstack([kern.k, kern.count]), kern.K.astype(np.float64)]))
    # header-free: row 0 = k, row 1 = count, rows 2.. = kernels; df and D are recomputed from k and the tag


def load_kernels(path, D):
    a = utils.read_fits(path)
    k, count, K = a[0], a[1], a[2:].astype(np.float32)
    return ModalKernels(k, K, count, float(k[1] - k[0]), float(D))


# ----------------------------------------------------------------------------- analytic PSDs
fitting_cutoff = utils.fitting_cutoff


def von_karman_variance_above(k_cut, r0, L0):
    """Analytic variance [nm^2] of the von Karman PSD above k_cut (isotropic, to infinity)."""
    return float(utils.VK_COEFF * r0 ** (-5 / 3) * utils.RAD2_TO_NM2 * (6 * np.pi / 5)
                 * (k_cut ** 2 + 1 / L0 ** 2) ** (-5 / 6))


def fitting_psd(k, r0, L0, N, D, variance_nm2=None, square=False):
    """Fitting-error PSD: von Karman above the cut-off of N modes (square=True: outside the square |kx|, |ky| <= k_cut, TipTop's corrected
    area, azimuthally averaged), scaled to `variance_nm2` when given."""
    k = np.asarray(k, float)
    k_cut = fitting_cutoff(N, D)
    if square:
        v_inf = utils.fitting_variance_tiptop(r0, L0, k_cut)
        mask = utils.square_outside_fraction(k, k_cut)
    else:
        v_inf = von_karman_variance_above(k_cut, r0, L0)
        mask = (k > k_cut).astype(float)
    scale = 1.0 if variance_nm2 is None else variance_nm2 / v_inf
    psd = utils.von_karman_psd(k, r0, L0) * mask * scale
    return psd, dict(k_cut=k_cut, scale=scale, variance_inf=v_inf * scale, n_across=np.sqrt(4.0 * N / np.pi))


# ----------------------------------------------------------------------------- KL fitting PSD (modes removed from the turbulence)
def von_karman_covariance(r, r0, L0):
    """Phase covariance [nm^2] of von Karman turbulence at separation r [m] (r0 at 500 nm): the Matern form
    C(r) = C(0) 2^(1/6)/Gamma(5/6) x^(5/6) K_5/6(x), x = 2 pi r / L0, C(0) = (6 pi / 5) 0.0229 (L0/r0)^(5/3)."""
    c0 = utils.VK_COEFF * (6 * np.pi / 5) * (L0 / r0) ** (5 / 3) * utils.RAD2_TO_NM2
    x = 2 * np.pi * np.maximum(np.asarray(r, float), 1e-12) / L0
    return c0 * 2 ** (1 / 6) / gamma(5 / 6) * x ** (5 / 6) * kv(5 / 6, x)


def _radial_bins(M, dx):
    """Annulus index of every point of the full M x M FFT grid (corners beyond the Nyquist circle -> bin nb, dropped) and the counts."""
    f = np.fft.fftfreq(M, d=dx)
    nb = M // 2 + 1
    b = np.minimum(np.rint(np.hypot(f[:, None], f[None, :]) * (M * dx)).astype(int), nb).ravel()
    return b, np.maximum(np.bincount(b, minlength=nb + 1), 1), nb


def _radial_mean(p2d, bins):
    b, cnt, nb = bins
    return (np.bincount(b, weights=np.asarray(p2d).ravel(), minlength=nb + 1) / cnt)[:nb]


def _lag_covariance_ft(M, dx, r0, L0, band_limit=True):
    """FFT of the von Karman covariance sampled at the signed lags of the M x M grid (no wrap for lags < M dx / 2). Sampling folds the
    power above the pixel Nyquist frequency into the grid; band_limit=True removes it (the FFT of the sampled covariance is the folded PSD
    / dx^2: the images at k + (p, q)/dx, |p|, |q| <= 4, are subtracted), i.e. the covariance of the field band-limited to the pixel
    Nyquist square, as a phase screen sampled on the pupil grid."""
    lag = np.fft.fftfreq(M, d=1.0 / M) * dx
    Ck = sfft.fft2(von_karman_covariance(np.hypot(lag[:, None], lag[None, :]), r0, L0))
    if band_limit:
        f, ks = np.fft.fftfreq(M, d=dx), 1.0 / dx
        for p in range(-4, 5):
            for q in range(-4, 5):
                if p or q:
                    Ck = Ck - utils.von_karman_psd(np.hypot(f[:, None] + p * ks, f[None, :] + q * ks), r0, L0) / dx ** 2
    return Ck


def windowed_turbulence_psd(pupil, D, pad, r0, L0, band_limit=True):
    """Radial PSD [nm^2 m^2] of the piston-removed turbulence INSIDE the pupil, E|FT(P (phi - <phi>))|^2 / A, on the grid and with the
    normalisation of modal_psd_kernels. Exact for the pixel-sampled field: analytic covariance, pupil pair counts and piston cross terms,
    all by FFT on the padded grid. band_limit (see _lag_covariance_ft): the field below the pixel Nyquist frequency only, so that
    von_karman_variance_above(k_max) is the rest (far field)."""
    pupil = np.asarray(pupil) > 0
    n = pupil.shape[0]
    dx, M = D / n, n * pad
    Pz = np.zeros((M, M))
    Pz[:n, :n] = pupil
    npix = Pz.sum()
    Pk = sfft.fft2(Pz)
    ac = sfft.ifft2(np.abs(Pk) ** 2).real                                # pupil pair counts per lag
    Ck = _lag_covariance_ft(M, dx, r0, L0, band_limit)
    conv = sfft.ifft2(Pk * Ck).real                                      # sum_x' C(x - x') P(x')
    full = sfft.fft2(ac * sfft.ifft2(Ck).real).real                      # E|FT(P phi)|^2 (pixel units)
    cross = sfft.fft2(Pz * conv)                                         # E[<phi> FT(P phi)] * npix
    pist = float((Pz * conv).sum()) / npix ** 2                          # E[<phi>^2]
    p2d = (full - 2 * np.real(np.conj(Pk) * cross) / npix + pist * np.abs(Pk) ** 2) * dx ** 4 / (npix * dx * dx)
    return _radial_mean(p2d, _radial_bins(M, dx))


def kl_fitting_cross(modes, pupil, D, pad, L0, workers=-1):
    """Ingredients of the exact KL fitting PSD at r0 = 1 m, with a_i = <m_i psi> the projection of the piston-removed turbulence psi on the
    unit-rms mode m_i (modes orthonormal on the pupil) and g_i(x) = E[a_i psi(x)] = (C * m_i)(x) / npix, piston removed:
      W (nb,)          windowed_turbulence_psd (band limited)
      X (n_modes, nb)  annulus mean of Re[conj(FT m_i) FT(g_i)] dx^4 / A: cross spectrum of each mode with the turbulence
      G (n_modes, n_modes)  E[a_i a_j] = <m_j g_i> (a DM KL basis is not exactly diagonal: low orders correlate up to ~0.3 when its
                            statistics differ from those used here, e.g. L0)."""
    pupil = np.asarray(pupil) > 0
    modes = np.asarray(modes, float)
    n = pupil.shape[0]
    rows, cols = np.nonzero(pupil)
    dx, M = D / n, n * pad
    npix = len(rows)
    mu = modes / np.sqrt(np.mean(modes ** 2, axis=1))[:, None]
    mu = mu - mu.mean(axis=1, keepdims=True)
    gram = mu @ mu.T / npix
    if np.max(np.abs(gram - np.eye(len(mu)))) > 1e-3:
        raise ValueError('kl_fitting_cross: the modes are not orthonormal on the pupil (max |<m_i m_j> - delta_ij| '
                         f'= {np.max(np.abs(gram - np.eye(len(mu)))):.2g}); the projection formula needs them orthonormal')
    bins = _radial_bins(M, dx)
    Ck = _lag_covariance_ft(M, dx, 1.0, L0, band_limit=True)
    norm = dx ** 4 / (npix * dx * dx)
    X = np.empty((len(mu), bins[2]))
    G = np.empty((len(mu), len(mu)))
    chunk = max(1, int(1e9 / (M * M * 16 * 4)))
    for s in range(0, len(mu), chunk):
        e = min(s + chunk, len(mu))
        buf = np.zeros((e - s, M, M))
        buf[:, rows, cols] = mu[s:e]
        Fm = sfft.fft2(buf, workers=workers)
        g = sfft.ifft2(Fm * Ck[None], workers=workers).real[:, rows, cols] / npix
        g -= g.mean(axis=1, keepdims=True)
        G[s:e] = g @ mu.T / npix
        buf[:] = 0.0
        buf[:, rows, cols] = g
        cs = (np.conj(Fm) * sfft.fft2(buf, workers=workers)).real
        for j in range(e - s):
            X[s + j] = _radial_mean(cs[j], bins) * norm
    return dict(W=windowed_turbulence_psd(pupil, D, pad, 1.0, L0), X=X, G=0.5 * (G + G.T))


def kl_fitting_psd(modes, pupil, D, pad, cross, N, workers=-1):
    """Fitting PSD [nm^2 m^2, r0 = 1 m] of the turbulence inside the pupil with its projection on the first N KL modes removed, EXACT in
    expectation (no Monte Carlo): E|FT(psi - sum_{i<N} a_i m_i)|^2 / A = W - 2 sum_{i<N} X_i + sum_{i,j<N} G_ij FT(m_i) conj FT(m_j) / A.
    The last term is computed on the eigenvectors of G[:N, :N] (N FFTs). Checked against 480 von Karman phase screens: 1 % per band
    from 0.5 k_c to 4 k_c, 0.2 % on the variance (220-px VLT pupil, 300 and 1200 modes)."""
    pupil = np.asarray(pupil) > 0
    n = pupil.shape[0]
    rows, cols = np.nonzero(pupil)
    dx, M = D / n, n * pad
    npix = len(rows)
    if N > len(cross['G']):
        raise ValueError(f'{N} modes requested but the KL basis has {len(cross["G"])}')
    mu = np.asarray(modes, float)[:N]
    mu = mu / np.sqrt(np.mean(mu ** 2, axis=1))[:, None]
    mu = mu - mu.mean(axis=1, keepdims=True)
    lam, U = np.linalg.eigh(cross['G'][:N, :N])
    rot, lam = U.T @ mu, np.maximum(lam, 0.0)
    bins = _radial_bins(M, dx)
    quad = np.zeros((M, M))
    chunk = max(1, int(1e9 / (M * M * 16 * 2)))
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        buf = np.zeros((e - s, M, M))
        buf[:, rows, cols] = rot[s:e]
        F = sfft.fft2(buf, workers=workers)
        quad += np.einsum('i,ijk->jk', lam[s:e], F.real ** 2 + F.imag ** 2)
    quad = _radial_mean(quad, bins) * dx ** 4 / (npix * dx * dx)
    return np.maximum(cross['W'] - 2.0 * cross['X'][:N].sum(axis=0) + quad, 0.0)


def save_fitting_cross(path, cross):
    os.makedirs(op.dirname(op.abspath(path)), exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(cross['W']), fits.ImageHDU(cross['X'], name='X'), fits.ImageHDU(cross['G'], name='G')]).writeto(
        path, overwrite=True)


def load_fitting_cross(path):
    with fits.open(path) as h:
        return dict(W=np.asarray(h[0].data, float), X=np.asarray(h['X'].data, float), G=np.asarray(h['G'].data, float))


# ----------------------------------------------------------------------------- PSD set
@dataclass
class PsdTerm:
    name: str
    psd: np.ndarray          # radial PSD on the kernel grid [nm^2 m^2]
    windowed: bool           # True: pupil-windowed (modal), False: stationary (analytic)


@dataclass
class PsdSet:
    k: np.ndarray
    terms: OrderedDict
    df: float
    count: np.ndarray
    D: float
    meta: dict = field(default_factory=dict)

    def variance(self, name):
        return float(np.sum(self.terms[name].psd * self.count) * self.df ** 2)

    def variances(self):
        return {n: self.variance(n) for n in self.terms}

    def names(self, include=None):
        return [n for n in self.terms if include is None or n in include]


def residual_psd_terms(kern, result, r0, L0, N, fitting_scale='lbt', extra_psd=None, chromatic=None,
                       sensing_nm=None, include_open_loop=True, split_noise=True, square=False, fitting_kl=None, extra_modal=None):
    """PSD of every term of a BudgetResult. fitting_kl: the KL fitting PSD on the kernel grid (kl_fitting_psd, at the seeing), used instead
    of the stationary tail; it is band limited, the power above the grid is von_karman_variance_above(k_max) (far field)."""
    k = kern.k
    terms = OrderedDict()
    names = ('lag', 'noise_photon', 'noise_ron', 'alias', 'xtalk') if split_noise else ('lag', 'noise', 'alias', 'xtalk')
    for name in names:
        v = getattr(result, name)
        if name == 'xtalk' and not np.any(v):
            continue
        terms[name] = PsdTerm(name, kern.psd(np.asarray(v)[:N]), True)
    cm = getattr(result, 'chrom_mult', None)
    if cm is not None and np.any(cm):                       # signed modal term: the kernels are linear
        terms['chrom_mult'] = PsdTerm('chrom_mult', kern.psd(np.asarray(cm)[:N]), True)
    fit_psd, info = fitting_psd(k, r0, L0, N, kern.D, result.fitting if fitting_scale == 'lbt' else None, square)
    if fitting_kl is None:
        terms['fitting'] = PsdTerm('fitting', fit_psd, False)                      # stationary von Karman tail, sharp cut at k_c
    else:                                                                          # KL modes removed (pupil windowed)
        far = von_karman_variance_above(float(k[-1]), r0, L0)
        s = result.fitting / (kern.variance(fitting_kl) + far) if fitting_scale == 'lbt' else 1.0
        terms['fitting'] = PsdTerm('fitting', np.asarray(fitting_kl, float) * s, True)
        info = dict(info, scale=s, variance_inf=(kern.variance(fitting_kl) + far) * s)
    for name, p in (extra_psd or {}).items():
        terms[name] = PsdTerm(name, np.asarray(p, float), False)
    for name, v in (extra_modal or {}).items():                  # per-mode variances [nm^2] (e.g. the fused chromatic shear)
        terms[name] = PsdTerm(name, kern.psd(np.asarray(v, float)), True)
    if chromatic is not None:
        if sensing_nm is None:
            raise ValueError('chromatic PSDs need the sensing wavelength (sensing_nm)')
        for name, p in chromatic.psds(k, r0, L0, info['k_cut'], sensing_nm).items():
            terms[name] = PsdTerm(name, p, False)
    total = sum(t.psd for t in terms.values())
    terms['total'] = PsdTerm('total', total, True)
    if include_open_loop:
        terms['turbulence'] = PsdTerm('turbulence', utils.von_karman_psd(k, r0, L0), False)
    s = PsdSet(k, terms, kern.df, kern.count, kern.D)
    grid_fit = s.variance('fitting')
    s.meta = dict(info, grid_fitting_variance=grid_fit, far_field_variance=max(info['variance_inf'] - grid_fit, 0.0),
                  k_max=float(k[-1]))
    return s


# ----------------------------------------------------------------------------- radial profile
def radial_profile(img, pixel_scale=1.0, binsize=None, center=None, stat='mean', rmax=None, pa_range=None):
    """Radial statistic of an image in annuli of width `binsize` [units of pixel_scale]."""
    img = np.asarray(img, float)
    ny, nx = img.shape
    cy, cx = (ny // 2, nx // 2) if center is None else center
    y, x = np.indices(img.shape)
    r = np.hypot(x - cx, y - cy) * pixel_scale
    sel = np.ones(img.shape, bool)
    if pa_range is not None:
        pa = np.degrees(np.arctan2(-(x - cx), (y - cy)))
        lo, hi = pa_range
        sel = (pa >= lo) & (pa <= hi) if lo <= hi else (pa >= lo) | (pa <= hi)
    binsize = pixel_scale if binsize is None else binsize
    rmax = r.max() / np.sqrt(2) if rmax is None else rmax
    idx = np.floor(r / binsize).astype(int)
    nb = int(np.ceil(rmax / binsize))
    ok = sel & (idx < nb)
    ib, v, rr = idx[ok], img[ok], r[ok]
    cnt = np.bincount(ib, minlength=nb)
    rad = np.bincount(ib, weights=rr, minlength=nb) / np.maximum(cnt, 1)
    if stat == 'mean':
        prof = np.bincount(ib, weights=v, minlength=nb) / np.maximum(cnt, 1)
    elif stat == 'std':
        m = np.bincount(ib, weights=v, minlength=nb) / np.maximum(cnt, 1)
        prof = np.sqrt(np.maximum(np.bincount(ib, weights=v ** 2, minlength=nb) / np.maximum(cnt, 1) - m ** 2, 0))
    elif stat == 'median':
        order = np.argsort(ib, kind='stable')
        splits = np.cumsum(cnt)[:-1]
        prof = np.array([np.median(c) if len(c) else np.nan for c in np.split(v[order], splits)])
    else:
        raise ValueError(stat)
    good = cnt > 0
    return rad[good], prof[good]


# ----------------------------------------------------------------------------- PSF
@dataclass
class PsfResult:
    wavelength_nm: float
    pixel_scale: float                  # lambda/D per pixel
    psf: OrderedDict                    # name -> (M, M) contrast image (peak of the DL PSF = 1), fftshifted
    halo: OrderedDict                   # name -> first-order halo (additive over the terms)
    strehl: dict                        # exp(-sigma^2) per name
    sigma2_rad: dict                    # in-grid phase variance per name [rad^2 at wavelength]
    diffraction_limited: np.ndarray
    coro: OrderedDict = None            # name -> image after a perfect coronagraph (Cavarroc et al. 2006, eq. 1): psf - strehl * PSF_DL

    def image(self, name='total', kind='psf'):
        """kind: 'psf' (long exposure), 'coro' (after a perfect coronagraph), 'halo' (first order, additive). The perfect coronagraph is
        that of Cavarroc et al. (2006, A&A 447, 397), eq. (1): the pupil field Pi (sqrt(E_C) - exp(i phi)), E_C = exp(-sigma^2); for Gaussian
        phase its long-exposure intensity is exactly psf - exp(-sigma^2) PSF_DL, i.e. the coherent part (and the Airy pattern) removed."""
        return {'psf': self.psf, 'coro': self.coro, 'halo': self.halo}[kind][name]

    def profile(self, name='total', kind='psf', binsize=0.5, stat='mean', rmax=None, pa_range=None):
        return radial_profile(self.image(name, kind), self.pixel_scale, binsize, None, stat, rmax, pa_range)

    def contrast_at(self, rho, name='total', kind='psf', binsize=0.5, **kw):
        """Annulus-averaged contrast at separations rho [lambda/D]; kind='coro': after a perfect coronagraph (see image)."""
        r, p = self.profile(name, kind, binsize, **kw)
        return np.interp(rho, r, p)


class PsfModel:
    """Telescope OTF on the PSD grid; computes PSFs from radial PSDs."""

    def __init__(self, pupil, D, pad=4, thr=1e-3):
        n = pupil.shape[0]
        self.n, self.pad, self.D = n, pad, float(D)
        self.M = M = n * pad
        self.dx = D / n
        p = np.zeros((M, M))
        p[:n, :n] = (np.asarray(pupil) > 0)
        ac = sfft.ifft2(np.abs(sfft.fft2(p)) ** 2).real
        self.otf_tel = ac / ac[0, 0]                                   # lags on the FFT grid, zero lag at [0, 0]
        self.psf_dl = sfft.fft2(self.otf_tel).real
        self.peak = float(self.psf_dl.max())
        f = np.fft.fftfreq(M, d=self.dx)
        self.kr = np.hypot(f[:, None], f[None, :])
        self.ok = self.otf_tel > thr
        self.pixel_scale = 1.0 / pad                                    # lambda/D per pixel

    def _cov(self, k, psd_nm, wavelength_nm):
        """Covariance C(r) [rad^2 at lambda] of a radial PSD [nm^2 m^2] on the grid lags."""
        k, psd_nm = np.asarray(k, float), np.asarray(psd_nm, float)
        df = k[1] - k[0]
        if abs(df - 1.0 / (self.M * self.dx)) < 1e-9 * df:               # same grid: nearest annulus, exact sums
            ib = np.rint(self.kr / df).astype(int)
            p2d = np.where(ib < len(k), psd_nm[np.minimum(ib, len(k) - 1)], 0.0)
        else:
            p2d = np.interp(self.kr, k, psd_nm, right=0.0)
        p2d = p2d * (2 * np.pi / wavelength_nm) ** 2
        return sfft.ifft2(p2d).real / self.dx ** 2                      # sum P df^2 e^{2 pi i k r}, df = 1/(M dx)

    def _chat(self, k, term, wavelength_nm):
        c = self._cov(k, term.psd, wavelength_nm)
        if term.windowed:
            c = np.where(self.ok, c / np.where(self.ok, self.otf_tel, 1.0), 0.0)
        return c

    def _shift(self, a):
        return np.fft.fftshift(a) / self.peak

    def psfs(self, psd_set, wavelength_nm, names=None, open_loop=True):
        """PSF of every term alone, of the total (sum of the covariances of the terms, each corrected"""
        k = psd_set.k
        comps = [n for n in psd_set.terms if n not in ('total', 'turbulence')]
        chats = OrderedDict((n, self._chat(k, psd_set.terms[n], wavelength_nm)) for n in comps)
        chats['total'] = sum(chats.values(), np.zeros((self.M, self.M)))
        if open_loop and 'turbulence' in psd_set.terms:
            chats['turbulence'] = self._chat(k, psd_set.terms['turbulence'], wavelength_nm)
        keep = [n for n in chats if names is None or n in names or n in ('total', 'turbulence')]
        out, halo, coro, strehl, sig = OrderedDict(), OrderedDict(), OrderedDict(), {}, {}
        for n in keep:
            ch = chats[n]
            s2 = float(ch[0, 0])
            out[n] = self._shift(sfft.fft2(self.otf_tel * np.exp(-(s2 - ch))).real)
            # perfect coronagraph, Cavarroc et al. (2006) eq. (1) averaged over the Gaussian phase: exp(-s2) FT[OTF_tel (exp(C) - 1)]
            coro[n] = self._shift(sfft.fft2(self.otf_tel * np.exp(-s2) * np.expm1(ch)).real)
            halo[n] = self._shift(sfft.fft2(self.otf_tel * ch).real)
            strehl[n], sig[n] = float(np.exp(-s2)), s2
        return PsfResult(float(wavelength_nm), self.pixel_scale, out, halo, strehl, sig, self._shift(self.psf_dl), coro)


class PsfEngine:
    """PSD / PSF of the BudgetResults of one ErrorBudgetMachine (KL kernels cached on disk and in memory, telescope OTF built once)."""

    def __init__(self, machine, pad=4):
        self.m, self.pad = machine, pad
        self._kern = self._model = self._cross = None
        self._fit = {}

    def kernels(self):
        if self._kern is None:
            path = self.m.paths.kl_psd_kernels(self.m.D, self.pad)
            if op.exists(path):
                self._kern = load_kernels(path, self.m.D)
            else:
                self._kern = modal_psd_kernels(self.m.modes(), ~(self.m.pupil() > 0), self.m.D, self.pad)
                save_kernels(path, self._kern)
        return self._kern

    def fitting_kl(self, r0, N):
        """Fitting PSD [nm^2 m^2] with the first N KL modes removed exactly from the turbulence inside the pupil (kl_fitting_psd); the
        cross kernels (~1 min for 1300 modes) and each N (~15 s) are cached on disk; scales as r0^-5/3."""
        N = int(N)
        if N not in self._fit:
            path = self.m.paths.kl_fitting_psd(self.m.L0, self.m.D, self.pad, N)
            if op.exists(path):
                self._fit[N] = np.asarray(utils.read_fits(path), float)
            else:
                if self._cross is None:
                    cpath = self.m.paths.kl_fitting_cross(self.m.L0, self.m.D, self.pad)
                    if op.exists(cpath):
                        self._cross = load_fitting_cross(cpath)
                    else:
                        self._cross = kl_fitting_cross(self.m.modes(), self.m.pupil(), self.m.D, self.pad, self.m.L0)
                        save_fitting_cross(cpath, self._cross)
                self._fit[N] = kl_fitting_psd(self.m.modes(), self.m.pupil(), self.m.D, self.pad, self._cross, N)
                utils.write_fits(path, self._fit[N])
        return self._fit[N] * r0 ** (-5 / 3)

    def psd_terms(self, result, fitting_scale=None, include_open_loop=True, extra_psd=None, split_noise=True, fitting='kl'):
        """PsdSet of a BudgetResult. fitting: 'kl' (default) the N controlled KL modes removed exactly from the turbulence inside the
        pupil (the expectation of removing them from phase screens, fitting_kl), 'stationary' the von Karman tail cut sharply at k_c
        (outside TipTop's square with machine.asm=False). fitting_scale 'lbt': PSD scaled to the budget's fitting variance (LBT law, or
        TipTop with asm=False: only the variance, the shape stays the KL one with 'kl'), 'psd': unscaled (with 'kl': the bound of
        perfectly correcting the N defined modes). None (default): 'psd' with fitting='kl', 'lbt' with 'stationary'."""
        if fitting not in ('kl', 'stationary'):
            raise ValueError(f"fitting must be 'kl' or 'stationary', got {fitting!r}")
        if fitting_scale is None:
            fitting_scale = 'psd' if fitting == 'kl' else 'lbt'
        if fitting_scale not in ('lbt', 'psd'):
            raise ValueError(f"fitting_scale must be 'lbt' or 'psd', got {fitting_scale!r}")
        wl = result.params['wavelength_nm']
        r0, chromatic = float(utils.r0_from_seeing(result.params['seeing'])), result.chromatic
        N = int(result.params.get('NmodesUsed', result.params['Nmodes']))
        bands = getattr(result, 'chrom_bands', None)
        if bands:                                   # fused / multi-stage results: chromatic PSD by spatial-frequency band
            k, extra_psd, chromatic = self.kernels().k, dict(extra_psd or {}), None
            for cm, k_lo, k_hi, sensing_nm in bands:
                hi = cm.psds(k, r0, self.m.L0, k_hi, sensing_nm)
                lo = cm.psds(k, r0, self.m.L0, k_lo, sensing_nm) if k_lo > 0 else {}
                for name, p in hi.items():
                    extra_psd[name] = extra_psd.get(name, 0.0) + p - lo.get(name, 0.0)
        modal = None
        if getattr(result, 'chrom_shear_modes', None) is not None:      # fused results: modal shear (w^T C_shear w per mode)
            extra_psd = {k_: v for k_, v in (extra_psd or {}).items() if k_ != 'chrom_shear'}
            modal = {'chrom_shear': result.chrom_shear_modes}
        return residual_psd_terms(self.kernels(), result, r0, self.m.L0, N, fitting_scale, extra_psd, chromatic,
                                  None if np.isnan(wl) else wl, include_open_loop, split_noise, square=not getattr(self.m, 'asm', True),
                                  fitting_kl=self.fitting_kl(r0, N) if fitting == 'kl' else None, extra_modal=modal)

    def psf_terms(self, result, wavelength_nm, names=None, **kw):
        """Long-exposure PSFs of every PSD term at the SCIENCE wavelength (peak of the diffraction-limited PSF = 1)."""
        if self._model is None:
            self._model = PsfModel(self.m.pupil(), self.m.D, self.pad)
        return self._model.psfs(self.psd_terms(result, **kw), wavelength_nm, names)
