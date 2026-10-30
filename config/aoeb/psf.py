"""Radial PSD of the residual terms [nm^2 m^2, k in cycles/m, separation = k D in lambda/D] and long-exposure PSFs.
Modal terms use pupil-windowed kernels, stationary terms analytic PSDs; the PSF is OTF_tel exp(-[C(0) - C(r)]). ASSUMPTIONS.md A28."""
import os.path as op
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft

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
                       sensing_nm=None, include_open_loop=True, split_noise=True, square=False):
    """PSD of every term of a BudgetResult."""
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
    terms['fitting'] = PsdTerm('fitting', fit_psd, False)
    for name, p in (extra_psd or {}).items():
        terms[name] = PsdTerm(name, np.asarray(p, float), False)
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
        self._kern = self._model = None

    def kernels(self):
        if self._kern is None:
            path = self.m.paths.kl_psd_kernels(self.m.D, self.pad)
            if op.exists(path):
                self._kern = load_kernels(path, self.m.D)
            else:
                self._kern = modal_psd_kernels(self.m.modes(), ~(self.m.pupil() > 0), self.m.D, self.pad)
                save_kernels(path, self._kern)
        return self._kern

    def psd_terms(self, result, fitting_scale='lbt', include_open_loop=True, extra_psd=None, split_noise=True):
        """PsdSet of a BudgetResult (fitting_scale 'lbt': PSD scaled to the budget's fitting variance, 'psd': pure tail; with
        machine.asm=False the fitting PSD lies outside TipTop's square corrected area)."""
        wl = result.params['wavelength_nm']
        return residual_psd_terms(self.kernels(), result, float(utils.r0_from_seeing(result.params['seeing'])), self.m.L0,
                                  int(result.params.get('NmodesUsed', result.params['Nmodes'])), fitting_scale, extra_psd, result.chromatic,
                                  None if np.isnan(wl) else wl, include_open_loop, split_noise, square=not getattr(self.m, 'asm', True))

    def psf_terms(self, result, wavelength_nm, names=None, **kw):
        """Long-exposure PSFs of every PSD term at the SCIENCE wavelength (peak of the diffraction-limited PSF = 1)."""
        if self._model is None:
            self._model = PsfModel(self.m.pupil(), self.m.D, self.pad)
        return self._model.psfs(self.psd_terms(result, **kw), wavelength_nm, names)
