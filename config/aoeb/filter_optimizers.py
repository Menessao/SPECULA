"""Loop model and per-mode gain optimiser. Model, derivation and validation: ASSUMPTIONS.md (A11).

Physical system per mode: continuous exposure over one frame, controller C = g/(1 - ff z^-1) at the end of the exposure,
latency (frames, any real value), zero-order-hold DM. Loop gain L = x Cn B with x = g rho, Cn = 1/(1 - ff z^-1),
B = delay of 1 + latency frames. With K = x Cn/(1 + L) and the image sums P0, P1, W of frame_spectra():
  T_t = int [P0 - 2 Re(K W) + |K|^2 P1],   T_c = int [P0 - Re(K W)],   noise/aliasing gain |K|^2.
"""
import hashlib
import warnings
from dataclasses import dataclass

import numpy as np

OG_FLOOR = 1e-3            # optical gains below this are clipped (poorly sensed modes); non-positive gains are an error
X_UNBOUNDED = 10.0         # gain limit when the loop has no phase crossover (unconditionally stable)
SEARCH_CHUNK = 64          # rows per matrix product (keeps the cost matrix in cache)
DEFAULT_FF_GRID = (1.0, 0.9995, 0.999, 0.998, 0.995, 0.99, 0.98, 0.96, 0.92, 0.85, 0.7)
ALIAS_K_MIN, ALIAS_K_MAX_900, ALIAS_SLOPE = 0.461, 7.682, -1.347     # AliasSpectrum: [c/m], [c/m] at 900 nm, slope of the radial spectrum


def unit_psd(f, n, wind, D):
    """One-sided unit-variance temporal PSD of radial order n (Conan 1995); n may be an array (broadcast against f). wind: speed [m/s],
    or a temporal model with unit_psd(f, n) (temporal.ZernikePSD)."""
    if hasattr(wind, 'unit_psd'):
        return wind.unit_psd(f, n)
    f, n = np.asarray(f, float), np.asarray(n, float)
    fc = 0.3 * (n + 1) * wind / D
    tilt = n == 1
    r = f / fc
    with np.errstate(divide='ignore'):
        return np.where(r <= 1, r ** np.where(tilt, -2.0 / 3.0, 0.0), r ** (-17.0 / 3.0)) / (np.where(tilt, 45.0, 17.0) / 14.0 * fc)


def folded_psd(f, fs, n, wind, D, n_fold=3):
    return sum(unit_psd(np.abs(np.asarray(f, float) + k * fs), n, wind, D) for k in range(-n_fold, n_fold + 1))


def psd_below(n, wind, D, fs, f_min, n_fold=3):
    """Integral of the folded unit PSD over [0, f_min] (analytic); n may be an array."""
    if hasattr(wind, 'below'):
        return wind.below(n, fs, f_min, n_fold)
    n = np.asarray(n, float)
    fc = 0.3 * (n + 1) * wind / D
    own = np.where(n == 1, 3.0 * fc ** (2.0 / 3.0) * f_min ** (1.0 / 3.0) / (45.0 * fc / 14.0), f_min / (17.0 * fc / 14.0))
    return own + f_min * sum(unit_psd(abs(k) * fs, n, wind, D) for k in range(-n_fold, n_fold + 1) if k)


def delay_response(f, fs, delay):
    """Delay in frames, fractional part by linear interpolation (SPECULA convention)."""
    fl = np.floor(delay)
    m = delay - fl
    zi = -2j * np.pi * np.asarray(f, float) / fs
    return (1.0 - m) * np.exp(zi * fl) + m * np.exp(zi * (fl + 1.0))


def frame_response(f, fs):
    """One-frame boxcar (exposure window / DM hold): sinc(f/fs) exp(-j pi f/fs)."""
    f = np.asarray(f, float)
    return np.exp(-1j * np.pi * f / fs) * np.sinc(f / fs)


def loop_response(f, fs, latency, ff):
    """(L0, Cn) per unit x = g rho; latency in frames."""
    cn = 1.0 / (1.0 - ff * np.exp(-2j * np.pi * np.asarray(f, float) / fs))
    return cn * delay_response(f, fs, 1.0 + latency), cn


def frame_spectra(f, fs, n, wind, D, n_fold, latency):
    """(P0, P1, W): folded PSD, exposure-filtered PSD and hold-window cross term of radial order(s) n -> (n_ord, n_f)
    (n_f for a scalar n)."""
    n = np.asarray(n)
    fk = np.asarray(f, float)[None, :] + (np.arange(-n_fold, n_fold + 1) * fs)[:, None]
    s = unit_psd(np.abs(fk), np.atleast_1d(n)[:, None, None], wind, D)
    b = frame_response(fk, fs)
    out = (s.sum(1), np.einsum('onf,nf->of', s, np.abs(b) ** 2), np.einsum('onf,nf->of', s, b * b * np.exp(-2j * np.pi * fk * latency / fs)))
    return out if n.ndim else tuple(o[0] for o in out)


class AliasSpectrum:
    """Unit-variance temporal PSD of the aliasing (A10): frozen-flow layers (weight, speed) carrying a radial spatial spectrum k^ALIAS_SLOPE,
    k between ALIAS_K_MIN and ALIAS_K_MAX_900 * 900 / wavelength [c/m], in isotropic directions. One (k, V) gives the arcsine PSD
    (2 / pi) / sqrt((k V)^2 - f^2), f < k V; the integral over k is done with k = (f / V) cosh u. The three constants are EMPIRICAL: fitted to
    frozen-flow runs at 0.9 arcsec (N >= 100, 50-500 Hz); there the measured aliasing does not depend on N or n_subap, so neither does this."""

    def __init__(self, layers, wavelength_nm=None, n_f=2000, n_u=100):
        w, v = np.array(layers, float).T
        self.layers, self.wavelength_nm = tuple(map(tuple, np.array(layers, float))), wavelength_nm
        w = w / w.sum()
        k1, k2, s = ALIAS_K_MIN, ALIAS_K_MAX_900 * 900.0 / (900.0 if wavelength_nm is None else float(wavelength_nm)), ALIAS_SLOPE
        norm = (k2 ** (s + 1) - k1 ** (s + 1)) / (s + 1)
        self.f = np.geomspace(1e-3, k2 * v.max(), n_f)
        self.s = np.zeros(n_f)
        t = np.linspace(0.0, 1.0, n_u)[None, :]
        for wl_, vl in zip(w, v):
            on = self.f < k2 * vl
            f = self.f[on]
            u1, u2 = np.arccosh(np.maximum(1.0, k1 * vl / f)), np.arccosh(k2 * vl / f)
            u = u1[:, None] + (u2 - u1)[:, None] * t
            k = (f / vl)[:, None] * np.cosh(u)
            self.s[on] += wl_ * 2.0 / (np.pi * vl) * (u2 - u1) * np.mean(0.5 * (k[:, 1:] ** s + k[:, :-1] ** s), axis=1) / norm

    def key(self):
        return self.layers, self.wavelength_nm

    def unit_psd(self, f, n=0):
        f, _ = np.broadcast_arrays(np.asarray(f, float), np.asarray(n))
        return np.interp(f.ravel(), self.f, self.s, right=0.0).reshape(f.shape)


class MeasuredAliasSpectrum:
    """Unit-variance temporal PSD of the aliasing PER RADIAL ORDER, measured (A10b): the mean over the modes of each order (and the seeds)
    of the unit-variance Welch PSDs of the simulated aliasing modal signal (frozen flow, N modes perfectly corrected, noiseless; the files
    of the aliasing PSD study, keys modes_<wavelength>). Mean over the modes of an order = average over wind directions, as the turbulence
    spectra. Measured on [f_res, fs_meas/2]: flat below, power-law tail above (fitted on the last octave, under 0.1 % of the power). n = 0:
    the mean over all modes. Shape of N corrected modes only (it moves with N: kc V bump); not rescaled for seeing or wavelength."""

    def __init__(self, files, wavelength_nm, N, nperseg=4000, n_bins=240):
        from scipy.signal import welch
        from .utils import radial_order
        self.files, self.wavelength_nm, self.N = tuple(sorted(files)), float(wavelength_nm), int(N)
        acc, fs = 0.0, None
        for fn in self.files:
            z = np.load(fn)
            a = z[f'modes_{self.wavelength_nm:g}'].astype(np.float64)
            if a.shape[1] != self.N:
                raise ValueError(f'{fn}: {a.shape[1]} modes, expected {self.N}')
            fs = 1.0 / float(z['dt'])
            a -= a.mean(axis=0, keepdims=True)
            f, p = welch(a, fs=fs, window='hann', nperseg=nperseg, noverlap=nperseg // 2, detrend=False, axis=0)
            acc = acc + p / len(self.files)
        f, acc = f[1:], acc[1:]                                                  # no DC bin (mean removed)
        unit = acc / (acc.sum(axis=0, keepdims=True) * (f[1] - f[0]))           # unit variance per mode
        order = radial_order(np.arange(self.N))
        self.orders = np.unique(order)
        rows = [unit.mean(axis=1)] + [unit[:, order == n].mean(axis=1) for n in self.orders]
        e = np.geomspace(f[0], f[-1] * (1 + 1e-9), n_bins + 1)                    # log-binned means: smooth, keeps the bumps
        k = np.digitize(f, e) - 1
        keep = np.bincount(k, minlength=n_bins) > 0
        self.f = (np.bincount(k, f, n_bins) / np.maximum(np.bincount(k, minlength=n_bins), 1))[keep]
        tab = np.array([(np.bincount(k, r, n_bins) / np.maximum(np.bincount(k, minlength=n_bins), 1))[keep] for r in rows])
        tab /= (_trapz_weights(self.f) @ tab.T + tab[:, 0] * self.f[0])[:, None]  # unit variance incl. the flat part below f[0]
        self.table = tab                                                          # row 0: all modes; row i: radial order orders[i-1]
        hi = self.f >= 0.5 * self.f[-1]
        # tail exponent per row: the last measured octave is flattened by the folding of the sampled data, so at most -8/3 (frozen flow
        # of a k^(-11/3) spatial spectrum)
        self.slope = np.minimum(np.polyfit(np.log(self.f[hi]), np.log(tab[:, hi]).T, 1)[0], -8.0 / 3.0)
        fg = np.geomspace(self.f[0], self.f[-1], 20000)                           # exact unit variance of the interpolated spectrum:
        for r in range(len(tab)):                                                 # flat part + log-log interpolation + power-law tail
            v = self._eval(fg, r)
            total = v[0] * self.f[0] + _trapz_weights(fg) @ v - v[-1] * self.f[-1] / (self.slope[r] + 1.0)
            self.table[r] /= total

    def key(self):
        return 'measured', self.files, self.wavelength_nm, self.N

    def _row(self, n):
        n = int(n)
        if n == 0:
            return 0
        i = int(np.searchsorted(self.orders, n))
        if i >= len(self.orders) or self.orders[i] != n:
            raise ValueError(f'radial order {n} not measured (N = {self.N})')
        return i + 1

    def _eval(self, f, r):
        """Row r at f: flat below the measured band, log-log interpolation in it, power law above."""
        lf, lt, x = np.log(self.f), np.log(self.table[r]), np.log(np.maximum(f, 1e-300))
        return np.exp(np.where(x > lf[-1], lt[-1] + self.slope[r] * (x - lf[-1]), np.interp(x, lf, lt)))

    def unit_psd(self, f, n=0):
        f, n = np.broadcast_arrays(np.asarray(f, float), np.asarray(n))
        out = np.empty(f.shape)
        for u in np.unique(n):
            m = n == u
            out[m] = self._eval(f[m], self._row(u))
        return out


def aliasing_frame_fraction(fs, alias, D=None, n_fold=3, f_min=1e-3, n_f=2000):
    """Variance of the aliasing measured in ONE frame at rate fs, as a fraction of the snapshot variance the calibration gives (A10):
    integral over [0, fs/2] of the folded, exposure-filtered AliasSpectrum (P1 of frame_spectra). This is the open-loop value (K = 1) of the
    aliasing integral T_a of the loop tables: the budget applies it, and the loop rejection, by itself."""
    f = np.geomspace(f_min, 0.5 * fs, n_f)
    w = _trapz_weights(f)
    w[0] += f_min
    return float(frame_spectra(f, fs, 0, alias, D, n_fold, 0.0)[1] @ w)


def gain_margin(l0):
    """Largest stable x for the loop gain x*l0(f) (f from ~0 to Nyquist): smallest -1/Re(l0) at the phase crossovers."""
    re, im = l0.real, l0.imag
    cand = []
    for i in np.nonzero((np.sign(im[:-1]) * np.sign(im[1:]) < 0) & (0.5 * (re[:-1] + re[1:]) < 0))[0]:
        t = im[i] / (im[i] - im[i + 1])
        cand.append(-1.0 / (re[i] + t * (re[i + 1] - re[i])))
    if re[-1] < 0:
        cand.append(-1.0 / re[-1])
    return min(cand) if cand else X_UNBOUNDED


def sensitivity_limited_x(l0, x_crit, max_sensitivity, n_iter=50):
    """Largest x <= x_crit whose peak rejection max_f |1 / (1 + x l0(f))| stays <= max_sensitivity (modulus margin; 2 = 6 dB)."""
    peak = lambda x: float(np.max(np.abs(1.0 / (1.0 + x * l0))))
    lo, hi = 0.0, float(x_crit)
    if peak(hi * 0.999) <= max_sensitivity:
        return hi
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if peak(mid) <= max_sensitivity else (lo, mid)
    return lo


def gain_grid(x_crit, l0, n_x, u_min, max_gain_frac, max_sensitivity):
    """Geometric grid of x up to max_gain_frac x_crit, or up to the modulus-margin limit when that is lower."""
    top = max_gain_frac * x_crit
    if max_sensitivity is not None:
        top = min(top, sensitivity_limited_x(l0, x_crit, max_sensitivity))
    return np.geomspace(min(u_min * x_crit, 0.5 * top), top, n_x)


def critical_x(fs, latency, ff=1.0, n=6000):
    """Largest stable x = g rho of the (leaky) integrator."""
    return gain_margin(loop_response(np.linspace(1e-6, 0.5 * fs, n), fs, latency, ff)[0])


def _trapz_weights(f):
    w = np.zeros_like(f)
    d = np.diff(f)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


def _grids(fs, wind, D, orders, alias, n_fold, latency, f_min, n_f):
    """Frequency grid, quadrature weights and the order-dependent spectra shared by the table classes."""
    f = np.geomspace(f_min, 0.5 * fs, n_f)
    w = _trapz_weights(f)
    w_edge = w.copy()
    w_edge[0] += f_min                                     # [0, f_min] with the value at f_min
    p0, p1, ww = frame_spectra(f, fs, orders, wind, D, n_fold, latency)
    below = psd_below(orders, wind, D, fs, f_min, n_fold)
    g = dict(f=f, w=w, w_edge=w_edge, base=p0 @ w, below=below, a_re=(ww.real * w).T, a_im=(ww.imag * w).T, a_p1=(p1 * w).T,
             s_alias=frame_spectra(f, fs, orders, alias, D, n_fold, latency)[1] * w_edge)       # (n_orders, n_f): per radial order (A10b)
    g['T0'] = g['base'] + below
    return g


def _closed_loop_tables(k, g):
    """T_t, T_c (n_orders, n_x) and |K|^2 (n_x, n_f) of the closed-loop responses k (n_x, n_f)."""
    kr, ki = np.ascontiguousarray(k.real), np.ascontiguousarray(k.imag)
    k2 = kr * kr + ki * ki
    t2 = kr @ g['a_re'] - ki @ g['a_im']
    om = 1.0 - k[:, 0]
    t_t = (g['base'][None] - 2.0 * t2 + k2 @ g['a_p1'] + g['below'][None] * (np.abs(om) ** 2)[:, None]).T
    t_c = (g['base'][None] - t2 + g['below'][None] * om.real[:, None]).T
    return t_t, t_c, k2


@dataclass
class OptResult:
    gain: np.ndarray
    ff: np.ndarray
    x: np.ndarray
    turb: np.ndarray       # residual contributions [command^2]
    noise: np.ndarray
    alias: np.ndarray
    xtalk: np.ndarray = None
    cross: np.ndarray = None   # V T_c
    turb0: np.ndarray = None   # V T0

    @property
    def resid(self):
        return self.turb + self.noise + self.alias + (0.0 if self.xtalk is None else self.xtalk)


class LoopTables:
    """Closed-loop integrals tabulated on a (ff, x) grid; an optimisation is then one matrix product per radial order."""

    def __init__(self, fs, latency, wind, D, orders, alias, n_fold=3, ff_grid=(1.0,), f_min=1e-3, n_f=600,
                 n_x=300, u_min=1e-4, max_gain_frac=0.98, max_sensitivity=None):
        self.fs, self.latency = fs, latency
        self.ff_grid = np.asarray(ff_grid, float)
        self.orders = np.unique(np.asarray(orders, int))
        g = _grids(fs, wind, D, self.orders, alias, n_fold, latency, f_min, n_f)
        f = self.f = g['f']
        self.T0 = g['T0']
        no, nff = len(self.orders), len(self.ff_grid)
        self.x = np.empty((nff, n_x))
        self.x_crit = np.empty(nff)
        self.T_t, self.T_c = np.empty((no, nff, n_x)), np.empty((no, nff, n_x))
        self.T_w, self.T_a = np.empty((nff, n_x)), np.empty((no, nff, n_x))
        f_stab = np.linspace(1e-6, 0.5 * fs, 6000)
        for j, ff in enumerate(self.ff_grid):
            self.x_crit[j] = critical_x(fs, latency, ff)
            x = self.x[j] = gain_grid(self.x_crit[j], loop_response(f_stab, fs, latency, ff)[0], n_x, u_min, max_gain_frac,
                                      max_sensitivity)
            l0, cn = loop_response(f, fs, latency, ff)
            k = x[:, None] * cn[None, :] / (1.0 + x[:, None] * l0[None, :])         # command per unit turbulence
            self.T_t[:, j], self.T_c[:, j], k2 = _closed_loop_tables(k, g)
            self.T_w[j] = (2.0 / fs) * (k2 @ g['w_edge'])
            self.T_a[:, j] = (k2 @ g['s_alias'].T).T
        F = nff * n_x
        self._stack = np.stack([self.T_t.reshape(no, F), np.broadcast_to(self.T_w.reshape(F), (no, F)),
                                self.T_a.reshape(no, F), self.T_c.reshape(no, F)], axis=1).astype(np.float32)   # (no, 4, F)

    def optimize(self, turb_var, noise_var, alias_var, og, order, xtalk_var=None, eta=None):
        """Per-mode optimum over (g, ff); arrays (R,) in command^2. eta: multiplicative chromatic factor per mode (the cost then
        also holds 2 eta V T_c). Returns OptResult."""
        turb, noise, alias, og = (np.atleast_1d(np.asarray(a, float)) for a in (turb_var, noise_var, alias_var, og))
        if not np.all(og > 0):
            raise ValueError('optical gains must be positive and finite (a negative gain points to a sign-flipped IM)')
        og = np.maximum(og, OG_FLOOR)
        xtalk = np.zeros_like(og) if xtalk_var is None else np.atleast_1d(np.asarray(xtalk_var, float))
        chi = None if eta is None or not np.any(eta) else np.broadcast_to(np.asarray(eta, float), og.shape)
        idx = self._order_index(order)
        w_noise, w_alias = (noise + xtalk) / og ** 2, alias / og ** 2
        w_chrom = None if chi is None else 2.0 * chi * turb
        jf, ix = self._search(idx, turb, w_noise, w_alias, w_chrom)
        interp = self._refine(idx, jf, ix, turb, w_noise, w_alias, w_chrom)
        return self._assemble(idx, jf, ix, interp, turb, noise, alias, xtalk, og, chi)

    def _order_index(self, order):
        order = np.atleast_1d(order)
        idx = np.searchsorted(self.orders, order)
        if np.any(self.orders[np.minimum(idx, len(self.orders) - 1)] != order):
            raise ValueError('mode orders not in the table')
        return idx

    def _search(self, idx, w_turb, w_noise, w_alias, w_chrom):
        """Argmin of the cost over the whole (ff, x) grid: one float32 matrix product per radial order -> (jf, ix)."""
        w32 = np.stack([w_turb, w_noise, w_alias] + ([] if w_chrom is None else [w_chrom]), axis=1).astype(np.float32)
        stack = self._stack if w_chrom is not None else self._stack[:, :3]
        perm = np.argsort(idx, kind='stable')
        bounds = np.searchsorted(idx[perm], np.arange(len(self.orders) + 1))
        best = np.empty(len(idx), int)
        for o in np.nonzero(np.diff(bounds))[0]:
            for s0 in range(bounds[o], bounds[o + 1], SEARCH_CHUNK):
                rows = perm[s0:min(s0 + SEARCH_CHUNK, bounds[o + 1])]
                best[rows] = (w32[rows] @ stack[o]).argmin(axis=1)
        return np.divmod(best, self.x.shape[1])

    def _cost(self, idx, jf, ix, w_turb, w_noise, w_alias, w_chrom):
        """Cost at table points; the weights (R,) broadcast against jf / ix of any trailing shape."""
        shape = (-1,) + (1,) * (np.ndim(jf) - 1)
        w = lambda a: a.reshape(shape)
        cost = w(w_turb) * self.T_t[w(idx), jf, ix] + w(w_noise) * self.T_w[jf, ix] + w(w_alias) * self.T_a[w(idx), jf, ix]
        return cost if w_chrom is None else cost + w(w_chrom) * self.T_c[w(idx), jf, ix]

    def _refine(self, idx, jf, ix, w_turb, w_noise, w_alias, w_chrom):
        """Parabolic refinement along ln x around the discrete minimum -> (neighbour index, interpolation weight)."""
        n_x = self.x.shape[1]
        lo, hi = np.clip(ix - 1, 0, n_x - 1), np.clip(ix + 1, 0, n_x - 1)
        y0, y1, y2 = (self._cost(idx, jf, i, w_turb, w_noise, w_alias, w_chrom) for i in (lo, ix, hi))
        curv = y0 - 2 * y1 + y2
        step = np.where((ix > 0) & (ix < n_x - 1) & (curv > 0), 0.5 * (y0 - y2) / np.where(curv > 0, curv, 1.0), 0.0)
        step = np.clip(step, -0.5, 0.5)
        return np.where(step >= 0, hi, lo), np.abs(step)

    def _assemble(self, idx, jf, ix, interp, turb, noise, alias, xtalk, og, chi):
        nb, a = interp
        lerp2 = lambda t: t[jf, ix] * (1 - a) + t[jf, nb] * a                          # (nff, n_x) tables
        lerp3 = lambda t: t[idx, jf, ix] * (1 - a) + t[idx, jf, nb] * a                # (n_order, nff, n_x) tables
        x = np.exp(np.log(self.x[jf, ix]) * (1 - a) + np.log(self.x[jf, nb]) * a)
        res = dict(turb=turb * lerp3(self.T_t), noise=noise * lerp2(self.T_w) / og ** 2, xtalk=xtalk * lerp2(self.T_w) / og ** 2,
                   alias=alias * lerp3(self.T_a) / og ** 2, cross=turb * lerp3(self.T_c))
        turb0 = turb * self.T0[idx]
        e = 0.0 if chi is None else chi
        none = (res['turb'] + res['noise'] + res['alias'] + res['xtalk'] + 2.0 * e * res['cross']) >= turb0 * (1.0 + 2.0 * e)
        keep = lambda q, zero: np.where(none, zero, q)                                  # exact "no control" candidate
        return OptResult(keep(x / og, 0.0), keep(self.ff_grid[jf], 1.0), keep(x, 0.0), keep(res['turb'], turb0),
                         keep(res['noise'], 0.0), keep(res['alias'], 0.0), keep(res['xtalk'], 0.0), keep(res['cross'], turb0), turb0)

def evaluate_controllers(fs, latency, gain, ff, turb_var, noise_var, alias_var, og, order, wind, D, alias,
                         n_fold=3, f_min=1e-3, n_f=600):
    """Residuals of fixed integrators / leaky integrators (g, ff per mode); unstable loops give inf."""
    gain, ff, tv, nv, av, og = (np.atleast_1d(np.asarray(a, float)) for a in (gain, ff, turb_var, noise_var, alias_var, og))
    order = np.atleast_1d(np.asarray(order, int))
    f = np.geomspace(f_min, 0.5 * fs, n_f)
    w = _trapz_weights(f)
    w_edge = w.copy()
    w_edge[0] += f_min
    if not np.all(og > 0):
        raise ValueError('optical gains must be positive and finite')
    n_m = len(gain)
    turb, noise, alias = np.zeros(n_m), np.zeros(n_m), np.zeros(n_m)
    for ffv in np.unique(ff):
        sel = np.nonzero(ff == ffv)[0]
        l0, cn = loop_response(f, fs, latency, ffv)
        x = gain[sel] * og[sel]
        k = x[:, None] * cn[None, :] / (1.0 + x[:, None] * l0[None, :])
        un, inv = np.unique(order[sel], return_inverse=True)
        p0, p1, ww = (a[inv] for a in frame_spectra(f, fs, un, wind, D, n_fold, latency))
        below = psd_below(un, wind, D, fs, f_min, n_fold)[inv]
        s_alias = frame_spectra(f, fs, un, alias, D, n_fold, latency)[1][inv]          # per radial order (A10b)
        turb[sel] = tv[sel] * ((p0 - 2.0 * (k * ww).real + np.abs(k) ** 2 * p1) @ w + below * np.abs(1.0 - k[:, 0]) ** 2)
        hn2 = np.abs(k) ** 2 / og[sel, None] ** 2
        noise[sel] = nv[sel] * (2.0 / fs) * (hn2 @ w_edge)
        alias[sel] = av[sel] * ((hn2 * s_alias) @ w_edge)
        bad = x >= critical_x(fs, latency, ffv)
        turb[sel[bad]] = noise[sel[bad]] = alias[sel[bad]] = np.inf
    return OptResult(gain, ff, gain * og, turb, noise, alias)


# ----------------------------------------------------------------------------- IIR controllers (global gain only)
def _cpu(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a, float)


def _degree(coef):
    """Degree of polynomials stored lowest order first (rows); -1 for an all-zero row."""
    nz = coef != 0
    return np.where(nz.any(axis=1), coef.shape[1] - 1 - np.argmax(nz[:, ::-1], axis=1), -1)


@dataclass
class IirControllers:
    """Per-mode IIR controllers in the SPECULA IirFilterData convention: H(z) = (num[0] + num[1] z + ...) / (den[0] + den[1] z + ...),
    lowest order first, one row per mode (a single row applies to all modes); the gain of a filter is its LAST numerator
    coefficient, so `gain` multiplies the normalised numerator (IirFilterData.set_gain). Only that global gain is optimised; the
    shape (poles, zeros) is kept. optimize_gain=False evaluates the filters as they are."""
    num: np.ndarray
    den: np.ndarray
    optimize_gain: bool = True

    def __post_init__(self):
        self.num, self.den = np.atleast_2d(_cpu(self.num)), np.atleast_2d(_cpu(self.den))
        if self.num.shape[0] != self.den.shape[0]:
            raise ValueError('num and den need one row per filter each')
        pad = max(self.num.shape[1], self.den.shape[1])
        self.num = np.pad(self.num, ((0, 0), (0, pad - self.num.shape[1])))
        self.den = np.pad(self.den, ((0, 0), (0, pad - self.den.shape[1])))
        if np.any(_degree(self.den) < 0) or np.any(_degree(self.num) > _degree(self.den)):
            raise ValueError('every filter needs a non-zero denominator and a numerator of degree <= the denominator (causal)')

    @classmethod
    def from_gain_and_ff(cls, gain, ff=None, **kw):
        """The integrators gain / (1 - ff z^-1) of IirFilterData.from_gain_and_ff."""
        gain = np.atleast_1d(np.asarray(gain, float))
        ff = np.ones_like(gain) if ff is None else np.broadcast_to(np.asarray(ff, float), gain.shape)
        return cls(np.stack([np.zeros_like(gain), gain], axis=1), np.stack([-ff, np.ones_like(gain)], axis=1), **kw)

    @classmethod
    def from_iir_filter_data(cls, iir, **kw):
        """From a SPECULA IirFilterData object (only its .num and .den are read)."""
        return cls(iir.num, iir.den, **kw)

    @classmethod
    def from_fits(cls, path, **kw):
        """From a file saved by IirFilterData.save, without importing SPECULA."""
        from astropy.io import fits
        with fits.open(path) as h:                                  # extensions ORDNUM, ORDDEN, NUM, DEN (as IirFilterData.restore)
            return cls(np.array(h[3].data), np.array(h[4].data), **kw)

    def for_modes(self, n_modes):
        """(normalised numerator, denominator, gain) of the first n_modes modes."""
        if len(self.num) == 1:
            num, den = np.repeat(self.num, n_modes, 0), np.repeat(self.den, n_modes, 0)
        elif len(self.num) >= n_modes:
            num, den = self.num[:n_modes], self.den[:n_modes]
        else:
            raise ValueError(f'{len(self.num)} filters for {n_modes} modes')
        gain = num[:, -1].copy()
        return num / np.where(gain != 0, gain, 1.0)[:, None], den, gain

    def key(self, n_modes):
        num, den, gain = self.for_modes(n_modes)
        return (hashlib.md5(num.tobytes() + den.tobytes() + gain.tobytes()).hexdigest(), bool(self.optimize_gain))


def _poly_response(coef, z):
    return np.polyval(coef[::-1], z)


def check_closed_loop_stable(num, den, latency, x_crit, fractions=(1e-3, 0.5, 0.98)):
    """Poles of 1 + x C(z) B(z) = 0 inside the unit circle for x = fraction * x_crit (the optimisation assumes 0 < x <= 0.98 x_crit)."""
    delay = 1.0 + latency
    lag = int(np.floor(delay))
    m = delay - lag
    num_b, den_b = np.array([1.0 - m, m]), np.zeros(lag + 2)             # B = ((1 - m) z + m) / z^(lag + 1), highest order first
    den_b[0] = 1.0
    nh, dh = np.trim_zeros(num[::-1], 'f'), np.trim_zeros(den[::-1], 'f')
    for frac in fractions:
        cl = np.polyadd(np.convolve(dh, den_b), frac * x_crit * np.convolve(nh if len(nh) else [0.0], num_b))
        roots = np.roots(np.trim_zeros(cl, 'f'))
        if len(roots) and np.max(np.abs(roots)) >= 1.0 - 1e-9:
            return False, frac
    return True, None


class IirTables:
    """Tabulated closed loop of per-mode IIR controllers whose shape is fixed and whose global gain g (x = g rho) is optimised.
    Same cost and refinement as LoopTables, but each mode has its own table row (its own shape and radial order); modes that share a
    shape share the loop response. optimize() has the signature of LoopTables.optimize; `orders` must be those of the build."""

    def __init__(self, fs, latency, wind, D, orders, alias, controllers, n_fold=3, f_min=1e-3, n_f=600, n_x=300, u_min=1e-4,
                 max_gain_frac=0.98, max_sensitivity=None):
        self.fs, self.latency, self.order = float(fs), float(latency), np.atleast_1d(orders)
        self.optimize_gain = controllers.optimize_gain
        N = len(self.order)
        num, den, self.gain0 = controllers.for_modes(N)
        self.controlled = np.any(num != 0, axis=1) & (self.gain0 != 0)
        rows = np.hstack([np.round(num, 12), np.round(den, 12)])
        self.shapes, self.shape = np.unique(rows, axis=0, return_inverse=True)
        self.shape = self.shape.ravel()
        uo, self.oi = np.unique(self.order, return_inverse=True)
        self.g = g = _grids(fs, wind, D, uo, alias, n_fold, latency, f_min, n_f)
        f, z = g['f'], np.exp(2j * np.pi * g['f'] / fs)
        ns = len(self.shapes)
        self.F, self.x, self.x_crit = np.empty((ns, n_f), complex), np.empty((ns, n_x)), np.empty(ns)
        self.T_t, self.T_c, self.T_a = np.empty((N, n_x)), np.empty((N, n_x)), np.empty((N, n_x))     # T_a per mode: per-order aliasing
        self.T_w = np.empty((ns, n_x))
        f_stab = np.linspace(1e-6, 0.5 * fs, 6000)
        z_stab = np.exp(2j * np.pi * f_stab / fs)
        for s in range(ns):
            members = np.nonzero(self.shape == s)[0]
            if not self.controlled[members].any():
                num_s, den_s = np.array([0.0, 1.0]), np.array([-1.0, 1.0])       # dummy: these modes are not controlled
            else:
                num_s, den_s = self.shapes[s][:num.shape[1]], self.shapes[s][num.shape[1]:]
            self.F[s] = _poly_response(num_s, z) / _poly_response(den_s, z)
            l0 = self.F[s] * delay_response(f, fs, 1.0 + latency)
            self.x_crit[s] = gain_margin(l0)
            if self.controlled[members].any():
                ok, frac = check_closed_loop_stable(num_s, den_s, latency, self.x_crit[s])
                if not ok:
                    raise ValueError(f'the IIR filter of mode {members[0]} is not stable for x = {frac:g} x_crit (x_crit = {self.x_crit[s]:.3g}): '
                                     'the global-gain optimisation needs a loop that is stable for 0 < x <= 0.98 x_crit')
            l0_stab = (_poly_response(num_s, z_stab) / _poly_response(den_s, z_stab)) * delay_response(f_stab, fs, 1.0 + latency)
            self.x[s] = gain_grid(self.x_crit[s], l0_stab, n_x, u_min, max_gain_frac, max_sensitivity)
            k = self.x[s][:, None] * self.F[s][None, :] / (1.0 + self.x[s][:, None] * l0[None, :])
            t_t, t_c, k2 = _closed_loop_tables(k, {**g, 'a_re': g['a_re'][:, np.unique(self.oi[members])], 'a_im': g['a_im'][:, np.unique(self.oi[members])],
                                                  'a_p1': g['a_p1'][:, np.unique(self.oi[members])], 'base': g['base'][np.unique(self.oi[members])],
                                                  'below': g['below'][np.unique(self.oi[members])]})
            col = np.searchsorted(np.unique(self.oi[members]), self.oi[members])
            self.T_t[members], self.T_c[members] = t_t[col], t_c[col]
            self.T_w[s] = (2.0 / fs) * (k2 @ g['w_edge'])
            self.T_a[members] = (k2 @ g['s_alias'][self.oi[members]].T).T

    def optimize(self, turb_var, noise_var, alias_var, og, order, xtalk_var=None, eta=None):
        """Per-mode optimum of the global gain; OptResult.gain multiplies the normalised numerator (IirFilterData.set_gain)."""
        turb, noise, alias, og = (np.atleast_1d(np.asarray(a, float)) for a in (turb_var, noise_var, alias_var, og))
        if not np.all(og > 0):
            raise ValueError('optical gains must be positive and finite (a negative gain points to a sign-flipped IM)')
        og = np.maximum(og, OG_FLOOR)
        if not np.array_equal(np.atleast_1d(order), self.order):
            raise ValueError('the IIR tables were built for other mode orders')
        xtalk = np.zeros_like(og) if xtalk_var is None else np.atleast_1d(np.asarray(xtalk_var, float))
        chi = None if eta is None or not np.any(eta) else np.broadcast_to(np.asarray(eta, float), og.shape)
        if not self.optimize_gain:
            return self._as_is(turb, noise, alias, og, xtalk, chi)
        s, rows = self.shape, np.arange(len(og))
        w_noise, w_alias = (noise + xtalk) / og ** 2, alias / og ** 2
        w_chrom = None if chi is None else 2.0 * chi * turb
        cost = lambda ix: (turb * self.T_t[rows, ix] + w_noise * self.T_w[s, ix] + w_alias * self.T_a[rows, ix]
                           + (0.0 if w_chrom is None else w_chrom * self.T_c[rows, ix]))
        n_x = self.T_t.shape[1]
        full = turb[:, None] * self.T_t + w_noise[:, None] * self.T_w[s] + w_alias[:, None] * self.T_a
        if w_chrom is not None:
            full += w_chrom[:, None] * self.T_c
        ix = full.argmin(axis=1)
        lo, hi = np.clip(ix - 1, 0, n_x - 1), np.clip(ix + 1, 0, n_x - 1)
        y0, y1, y2 = cost(lo), cost(ix), cost(hi)
        curv = y0 - 2 * y1 + y2
        step = np.clip(np.where((ix > 0) & (ix < n_x - 1) & (curv > 0), 0.5 * (y0 - y2) / np.where(curv > 0, curv, 1.0), 0.0), -0.5, 0.5)
        nb, a = np.where(step >= 0, hi, lo), np.abs(step)
        lerp_m = lambda t: t[rows, ix] * (1 - a) + t[rows, nb] * a
        lerp_s = lambda t: t[s, ix] * (1 - a) + t[s, nb] * a
        x = np.exp(np.log(self.x[s, ix]) * (1 - a) + np.log(self.x[s, nb]) * a)
        res = dict(turb=turb * lerp_m(self.T_t), noise=noise * lerp_s(self.T_w) / og ** 2, xtalk=xtalk * lerp_s(self.T_w) / og ** 2,
                   alias=alias * lerp_m(self.T_a) / og ** 2, cross=turb * lerp_m(self.T_c))
        turb0 = turb * self.g['T0'][self.oi]
        e = 0.0 if chi is None else chi
        none = ((res['turb'] + res['noise'] + res['alias'] + res['xtalk'] + 2.0 * e * res['cross']) >= turb0 * (1.0 + 2.0 * e)) | ~self.controlled
        keep = lambda q, zero: np.where(none, zero, q)
        return OptResult(keep(x / og, 0.0), np.full(len(og), np.nan), keep(x, 0.0), keep(res['turb'], turb0), keep(res['noise'], 0.0),
                         keep(res['alias'], 0.0), keep(res['xtalk'], 0.0), keep(res['cross'], turb0), turb0)

    def _as_is(self, turb, noise, alias, og, xtalk, chi):
        """The filters exactly as given (gain = their last numerator coefficient); unstable modes get an infinite residual."""
        g, fs = self.g, self.fs
        x_all = self.gain0 * og
        out = {k: np.zeros(len(og)) for k in ('turb', 'noise', 'alias', 'xtalk', 'cross')}
        unstable = np.zeros(len(og), bool)
        for s in range(len(self.shapes)):
            m = np.nonzero(self.shape == s)[0]
            x = x_all[m]
            l0 = self.F[s] * delay_response(g['f'], fs, 1.0 + self.latency)
            k = x[:, None] * self.F[s][None, :] / (1.0 + x[:, None] * l0[None, :])
            o = self.oi[m]
            k2 = np.abs(k) ** 2
            t2 = np.sum(k.real * g['a_re'][:, o].T - k.imag * g['a_im'][:, o].T, axis=1)
            om = 1.0 - k[:, 0]
            out['turb'][m] = turb[m] * (g['base'][o] - 2.0 * t2 + np.sum(k2 * g['a_p1'][:, o].T, axis=1) + g['below'][o] * np.abs(om) ** 2)
            out['cross'][m] = turb[m] * (g['base'][o] - t2 + g['below'][o] * om.real)
            out['noise'][m] = noise[m] * (2.0 / fs) * (k2 @ g['w_edge']) / og[m] ** 2
            out['xtalk'][m] = xtalk[m] * (2.0 / fs) * (k2 @ g['w_edge']) / og[m] ** 2
            out['alias'][m] = alias[m] * np.sum(k2 * g['s_alias'][o], axis=1) / og[m] ** 2
            unstable[m] = (x >= self.x_crit[s]) & self.controlled[m]
        for q in out.values():
            q[unstable] = np.inf
        if unstable.any():
            warnings.warn(f'{int(unstable.sum())} IIR filters are unstable as given (gain at or above the stability limit)')
        return OptResult(self.gain0.copy(), np.full(len(og), np.nan), x_all, out['turb'], out['noise'], out['alias'], out['xtalk'], out['cross'],
                         turb * g['T0'][self.oi])
