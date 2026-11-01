"""
Helpers for the IIR Wiener study (not needed to design a filter, see wiener_ao_iir):
  1. disturbance PSDs: von Karman Zernike (frozen flow), Conan-like, vibration lines
  2. loop evaluation: closed-loop residual, optimal (leaky) integrator, transfer functions
  3. utilities: Noll index -> radial order, source-hashed cache file names,
     total residual over radial orders
Plotting is in iir_study.py. All PSDs are one-sided callables S(f) [unit^2/Hz].
"""
import json, hashlib, inspect
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, jv
from wiener_ao_iir import pol_weights

_trapz = getattr(np, "trapezoid", None) or np.trapz


# =========================================================== 1. disturbance PSDs
def r0_from_seeing(seeing_arcsec, lam=500e-9):
    return 0.98 * lam / (seeing_arcsec / 206264.806)


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

    def __init__(self, D, r0, L0, layers, lam=500e-9, r0_lam=500e-9,
                 fmin=1e-4, fmax=1e5, n_f=500):
        w, V = np.array(layers, float).T
        self.D, self.r0, self.L0, self.w, self.V = D, r0, L0, w / w.sum(), V
        self.scale = (r0_lam / lam) ** 2                       # phase variance scaling
        self.f = np.geomspace(fmin, fmax, n_f)
        self._cache = {}

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


def noll_var(n, D, r0):
    """Kolmogorov variance of one Zernike mode of radial order n [rad^2] (Noll 1976)."""
    return (0.0072 * (D / r0) ** (5 / 3) * (n + 1) * np.pi ** (8 / 3) * gamma(14 / 3)
            * gamma(n - 5 / 6) / (gamma(17 / 6) ** 2 * gamma(n + 23 / 6)))


def conan_psd(n, D, r0, V):
    """Conan (1995)-like one-sided temporal PSD of a radial-order-n Zernike mode:
    f^-2/3 (tilt, n = 1) or flat (n > 1) below fc = 0.3 (n+1) V/D, f^-17/3 above;
    normalised to the Noll per-mode variance."""
    fc = 0.3 * (n + 1) * V / D
    lo, low_int = (-2 / 3, 3.0) if n == 1 else (0.0, 1.0)    # int_0^fc (f/fc)^lo df / fc
    A = noll_var(n, D, r0) / (fc * (low_int + 3 / 14))

    def S(f):
        x = np.maximum(np.asarray(f, float), 1e-300) / fc
        return A * np.where(x < 1, x ** lo, np.maximum(x, 1.0) ** (-17 / 3))
    return S


def vibration_psd(lines):
    """
    One-sided PSD of vibration lines [(f0 [Hz], variance [unit^2], damping zeta), ...].
    Each line is a 2nd-order resonance driven by white noise, normalised to its variance:
        S(f) = 8 zeta w0^3 var / ((w0^2 - w^2)^2 + (2 zeta w0 w)^2),  w = 2 pi f, w0 = 2 pi f0
    (FWHM ~ 2 zeta f0, correlation time 1/(2 pi zeta f0)).
    """
    def S(f):
        w = 2 * np.pi * np.asarray(f, float)
        out = np.zeros_like(w)
        for f0, var, zeta in lines:
            w0 = 2 * np.pi * f0
            out = out + 8 * zeta * w0 ** 3 * var / ((w0 ** 2 - w ** 2) ** 2
                                                     + (2 * zeta * w0 * w) ** 2)
        return out
    return S


def with_vibrations(S, lines=()):
    """Disturbance PSD = S + vibration lines (S itself if there are none)."""
    if not lines:
        return S
    Sv = vibration_psd(lines)
    return lambda f: S(f) + Sv(f)



# ========================================================== 2. loop evaluation
def _poly(c, z1):
    return np.polyval(np.asarray(c)[::-1], z1)


def closed_loop_residual(spec, num, den, structure="pol", g=1.0, g_hat=1.0):
    """
    Exact residual variance of one mode in closed loop, np.inf if unstable.
    structure='pol'     : u = (num/den) * POL, POL built with g_hat
    structure='residual': u = (num/den) * y_res / g_hat   (e.g. leaky integrator)
    """
    rho = g / g_hat
    bp = np.convolve(num, spec.p)
    char = np.zeros(max(len(den), len(bp)))
    char[:len(den)] += den
    char[:len(bp)] += (-(1 - rho) if structure == "pol" else rho) * bp
    char = np.trim_zeros(char, "b")
    if len(char) > 1 and np.any(np.abs(np.roots(char)) >= 1):
        return np.inf
    # y_ol -> u, as a single ratio of polynomials (finite even where den = 0,
    # e.g. a pure integrator at DC): num / (g_hat * char)
    H = _poly(num, spec.z1) / (g_hat * _poly(char, spec.z1))
    Pyy = g**2 * (spec.Pmm + spec.Paa) + spec.Pnn        # y_ol = g(m+a) + n
    pred = np.mean(spec.Pmm - 2 * np.real(g * spec.Psm * np.conj(H)) + np.abs(H) ** 2 * Pyy)
    return pred + spec.hold_err


def optimal_integrator(spec, g=1.0, g_hat=1.0, G_range=(1e-3, 2.0)):
    """Pure integrator (leak = 1), u_k = u_{k-1} + G*y_k/g_hat; G > 0 optimised.
    Log-grid scan then bounded refinement around the best stable gain."""
    J = lambda lG: closed_loop_residual(spec, [np.exp(lG)], [1.0, -1.0], "residual", g, g_hat)
    lg = np.linspace(np.log(G_range[0]), np.log(G_range[1]), 60)
    v = np.array([J(x) for x in lg])
    i = int(np.argmin(v))
    lo, hi = lg[max(i - 1, 0)], lg[min(i + 1, len(lg) - 1)]
    r = minimize_scalar(J, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-6))
    return (np.exp(r.x), r.fun) if r.fun <= v[i] else (np.exp(lg[i]), v[i])


def optimal_leaky_integrator(spec, g=1.0, g_hat=1.0, leak_bounds=(0.0, 1.0)):
    """u_k = leak*u_{k-1} + G*y_k/g_hat; G > 0 and leak in (lo, hi] optimised jointly.
    Returns (G, leak), residual."""
    def cost(v):
        G, l = v
        if G <= 0 or not leak_bounds[0] < l <= leak_bounds[1]:
            return 1e30
        return closed_loop_residual(spec, [G], [1.0, -l], "residual", g, g_hat)

    best = min((minimize(cost, [G0, 0.995], method="Nelder-Mead",
                         options=dict(xatol=1e-6, fatol=1e-14, maxiter=800))
                for G0 in (0.1, 0.3, 0.6)), key=lambda r: r.fun)
    return best.x, best.fun


def transfer_functions(b, a, f, fs, delay, structure="pol", g=1.0, g_hat=1.0, C=None):
    """
    Continuous-time ETF, NTF and aliasing TF of the loop at frequencies f (Hz),
    in-band (spectral images neglected); cf. Males & Guyon Eq. 63.
    The controller is b/a (powers of z^-1), or C = its values at f (e.g. the
    infinite-order Wiener filter on the grid). structure as in closed_loop_residual;
    the loop is the exact discrete one (POL weights p(z)), the WFS box and the DM
    hold + delay act on the continuous signals.
      ETF: turbulence -> residual;  NTF: noise (y_ol units) -> DM;
      ATF: aliasing (WFS-input units) -> DM.
    """
    f = np.asarray(f, float)
    T, s = 1.0 / fs, 2j * np.pi * f
    H = np.where(f == 0, 1.0, (1 - np.exp(-s * T)) / np.where(f == 0, 1, s * T))
    z1 = np.exp(-s * T)
    if C is None:
        C = _poly(b, z1) / _poly(a, z1)
    P, rho = _poly(pol_weights(delay), z1), g / g_hat
    if structure == "pol":
        K = C / (1 - (1 - rho) * C * P) / g_hat          # y_ol -> u
    else:
        K = (C / g_hat) / (1 + rho * C * P)
    NTF = H * np.exp(-s * delay * T) * K
    return 1 - g * H * NTF, NTF, g * H * NTF


# ================================================================ 3. utilities
def cache_file(prefix, cfg, *objs, ext="json"):
    """File name hashed on the configuration and on the source of objs (modules or
    functions), so a cached result is never stale."""
    src = json.dumps(cfg, sort_keys=True, default=str)
    src += "".join(inspect.getsource(o) for o in objs)
    return f"{prefix}_{hashlib.sha256(src.encode()).hexdigest()[:12]}.{ext}"


def noll_radial_order(j):
    """Radial order n of the Noll-indexed Zernike mode j (j = 2, 3: tip/tilt)."""
    if j < 2:
        raise ValueError(f"Noll index {j}: piston (j = 1) or invalid")
    return int(np.ceil((-3 + np.sqrt(9 + 8 * (j - 1))) / 2))


def total_variance(res_n, n_design, n_max, res_j=(), n_j=()):
    """
    Total residual sum_{n=1..n_max} (n+1) sigma_n^2 per controller, with
    res_n[len(n_design), n_ctrl] interpolated (PCHIP, log-log) between design orders.
    Optional single modes (e.g. with vibrations): res_j[k, n_ctrl] replaces one of
    the n+1 modes of radial order n_j[k].
    """
    n_all = np.arange(1, n_max + 1)
    sig = np.exp([PchipInterpolator(np.log(n_design), np.log(r))(np.log(n_all))
                  for r in np.asarray(res_n).T])                   # [n_ctrl, n_max]
    tot = (sig * (n_all + 1)).sum(1)
    for r, n in zip(res_j, n_j):
        tot += np.asarray(r) - sig[:, n - 1]
    return tot
