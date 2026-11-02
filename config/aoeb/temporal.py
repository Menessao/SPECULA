"""Optional temporal model (from ao_tools.py): exact temporal PSDs of the radial orders for multi-layer frozen-flow von Karman
turbulence, instead of filter_optimizers.unit_psd (Conan 1995 asymptotes, one effective wind). Use ErrorBudgetMachine(temporal=
ZernikePSD(D, L0=L0, layers=[(cn2, speed), ...])), or atmosphere.temporal: zernike in a configuration file; only the shape matters
(unit variance), so r0 is irrelevant here."""
import numpy as np
from scipy.special import jv

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _G(n, kx, D, r0, L0, n_ky=6000):
    ky = np.r_[0.0, np.geomspace(1e-7, 200.0 / D, n_ky)]          # symmetric -> x2
    k = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    x = np.pi * D * np.maximum(k, 1e-300)
    filt = (n + 1) * (2 * jv(n + 1, x) / x) ** 2
    phi = 0.023 * r0 ** (-5 / 3) * (k ** 2 + L0 ** -2.0) ** (-11 / 6)
    return 4 * _trapz(phi * filt, ky, axis=1)        # x2 (ky<0), x2 (one-sided in f)


class ZernikePSD:
    """
    Temporal PSDs of Zernike modes for multi-layer frozen-flow von Karman turbulence.
    Per radial order n, averaged over the n+1 modes of that order (Noll normalisation),
    so the result does not depend on wind direction. Each layer l (fraction w_l of the
    total Cn^2, speed V_l) contributes (1/V_l) G_n(f/V_l), with
        G_n(kx) = int dky  Phi(k) (n+1) [2 J_{n+1}(pi D k) / (pi D k)]^2,  k = |(kx, ky)|,
        Phi(k)  = 0.023 r0^(-5/3) (k^2 + 1/L0^2)^(-11/6)   [rad^2 m^2, k in cycles/m].
    Layer altitude is irrelevant on axis. layers = [(w, V), ...]; r0 is given at
    r0_lam, PSDs are returned at lam. psd(n) -> one-sided S(f) [rad^2/Hz].
    """

    def __init__(self, D, r0=1.0, L0=25.0, layers=((1.0, 10.0),), lam=500e-9, r0_lam=500e-9,
                 fmin=1e-4, fmax=1e5, n_f=500):
        w, V = np.array(layers, float).T
        self.D, self.r0, self.L0, self.w, self.V = D, r0, L0, w / w.sum(), V
        self.scale = (r0_lam / lam) ** 2                       # phase variance scaling
        self.f = np.geomspace(fmin, fmax, n_f)
        self._cache, self._var = {}, {}

    def table(self, n):
        """PSD of radial order n on the grid self.f."""
        if n not in self._cache:
            kx = np.geomspace(self.f[0] / self.V.max(), self.f[-1] / self.V.min(), 400)
            lG = np.log(_G(n, kx, self.D, self.r0, self.L0))     # one table for all layers
            self._cache[n] = self.scale * sum(
                w / V * np.exp(np.interp(np.log(self.f / V), np.log(kx), lG))
                for w, V in zip(self.w, self.V))
        return self._cache[n]

    def psd(self, n):
        """Log-log interpolation; flat below fmin, power-law extrapolation above fmax."""
        lf, lS = np.log(self.f), np.log(self.table(n))
        slope = (lS[-1] - lS[-2]) / (lf[-1] - lf[-2])

        def S(f):
            lx = np.log(np.maximum(np.asarray(f, float), 1e-300))
            return np.exp(np.where(lx > lf[-1], lS[-1] + slope * (lx - lf[-1]),
                                   np.interp(lx, lf, lS)))
        return S

    def variance(self, n):
        S0 = self.table(n)
        return _trapz(S0, self.f) + S0[0] * self.f[0]

    # ---- aoeb interface: unit-variance spectra (filter_optimizers.unit_psd / psd_below dispatch here)
    def unit_psd(self, f, n):
        f, n = np.broadcast_arrays(np.asarray(f, float), np.asarray(n))
        out = np.empty(f.shape)
        for u in np.unique(n):
            m = n == u
            if int(u) not in self._var:
                self._var[int(u)] = self.variance(int(u))
            out[m] = self.psd(int(u))(f[m]) / self._var[int(u)]
        return out

    def below(self, n, fs, f_min, n_fold=3):
        """Integral of the folded unit PSD over [0, f_min] (the in-band part flat below the table)."""
        fb = np.geomspace(f_min * 1e-6, f_min, 200)
        own = np.array([_trapz(self.unit_psd(fb, int(k)), fb) + self.unit_psd(fb[0], int(k)) * fb[0] for k in np.atleast_1d(n)])
        return own.reshape(np.shape(n)) + f_min * sum(self.unit_psd(abs(k) * fs, n) for k in range(-n_fold, n_fold + 1) if k)
