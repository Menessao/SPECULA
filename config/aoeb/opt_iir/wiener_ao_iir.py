"""
Optimal IIR Wiener predictor for modal AO control, applied to pseudo-open-loop
(POL) measurements:   u = F(z) POL,   F(z) = B(z)/A(z)  (powers of z^-1, a0 = 1).

Design: (1) exact infinite-order causal Wiener filter by cepstral spectral
factorisation (Wiener-Hopf); (2) Steiglitz-McBride fit of an order-N rational
filter to it, weighted by the measurement spectrum, plus a warm start from the
order N-1 solution; (3) refinement of B and A on the exact cost, with A kept
stable through a reflection-coefficient parametrisation.  N = deg B = deg A.

ASSUMPTIONS
-----------
* Modal control: one independent loop per mode (cross-mode correlations ignored).
* Perfect DM in-mode; plant = WFS box integration + pure delay + DM zero-order hold.
* Timing (T = 1/fs): WFS integrates over [(k-1)T, kT]; u_k is held on
  [kT+tau, kT+tau+T]; tau = delay*T (non-integer allowed).
  Total delay in Males & Guyon's sense is T + tau.
* Optical gain: one scalar per mode, known and compensated (POL exact):
  POL_k = box(x + a)_k + n_k / g.
* Wide-sense stationary processes; second-order statistics only (optimal LINEAR
  filter; MMSE-optimal if Gaussian). Poisson noise enters via its variance.
* S_turb, S_alias: callables, one-sided continuous-time PSDs [unit^2/Hz] at the
  WFS input, valid well beyond fs/2 (they are folded here). Aliasing is
  uncorrelated with the turbulence.
* noise: raw WFS noise in modal units at unit optical gain, uncorrelated with x
  and a: scalar per-frame variance (white), callable one-sided discrete-time
  PSD on [0, fs/2] (Males' T_ph convention), or a pair (f, PSD) of samples of it.
* Cost: time-averaged residual over the DM hold = hold_err + E[(s_k - u_k)^2],
  with s_k the hold-interval average of x; hold_err cannot be corrected.
* Spectra live on a uniform grid of M points over [0, fs): resolution fs/M must
  resolve the lowest temporal cut-off, and M/fs must exceed the correlation time
  (the causal projection is circular).
"""
import numpy as np
from scipy.optimize import minimize

_trapz = getattr(np, "trapezoid", None) or np.trapz


def pol_weights(delay):
    """POL reconstruction: POL_k = y_k/g + sum_i p[i] u_{k-i}  (delay = d0 + phi)."""
    d0, phi = int(np.floor(delay)), delay - np.floor(delay)
    p = np.zeros(d0 + 3)
    p[1 + d0] += 1 - phi
    p[2 + d0] += phi
    return p


class ModeSpectra:
    """Discrete-time (folded) spectra of one mode. Phi convention: var = mean over grid."""

    def __init__(self, S_turb, fs, delay, S_alias=None, noise=0.0,
                 M=2**16, n_fold=40):
        T = 1.0 / fs
        self.fs, self.delay, self.M = fs, delay, M
        self.p = pol_weights(delay)
        nu = np.fft.fftfreq(M, T)                        # [-fs/2, fs/2), FFT order
        fl = nu[:, None] + fs * np.arange(-n_fold, n_fold + 1)[None, :]
        ph = np.exp(2j * np.pi * fl * (delay + 1) * T)

        def folded(S, box=True):
            """Two-sided PSD 0.5*S (times the WFS box |sinc|^2 if box) at every alias;
            the in-band column is cell-averaged (resolves sharp low-f features)."""
            fun = lambda f: 0.5 * S(np.abs(f)) * (np.sinc(f * T) ** 2 if box else 1.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                S2 = fun(fl)
            S2[:, n_fold] = _cell_mean(fun, nu, fs / M)
            return S2

        S2 = folded(S_turb)                              # WFS-averaged turbulence
        self.Pmm = fs * S2.sum(1)
        self.Psm = fs * (S2 * ph).sum(1)
        self.Paa = np.zeros(M) if S_alias is None else fs * folded(S_alias).sum(1)
        if callable(noise) or np.ndim(noise) > 0:        # PSD callable or (f, PSD) samples
            Sn = noise if callable(noise) else (lambda f: np.interp(f, *np.asarray(noise, float)))
            self.Pnn = 0.5 * fs * Sn(np.abs(nu))
        else:                                            # white, per-frame variance
            self.Pnn = np.full(M, float(noise))
        self.z1 = np.exp(-2j * np.pi * nu * T)           # z^-1 on the grid
        # same folded grid for all three -> exactly consistent
        self.open_loop = fs * folded(S_turb, box=False).sum(1).mean()
        self.var_s = self.Pmm.mean()
        self.hold_err = self.open_loop - self.var_s

    def Pyy(self, g=1.0):
        return self.Pmm + self.Paa + self.Pnn / g**2

    def pred_err(self, F, g=1.0):
        """E[(s-u)^2] for u = F POL, F given on the grid."""
        return np.mean(self.Pmm - 2 * np.real(self.Psm * np.conj(F))
                       + np.abs(F) ** 2 * self.Pyy(g))


# ---------------------------------------------------------- helpers
def _cell_mean(fun, nu, d, q=16):
    """Average of an even function over the cells [nu - d/2, nu + d/2]."""
    off = (np.arange(q) + 0.5) / q - 0.5
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = fun(nu[:, None] + d * off[None, :]).mean(1)
    fg = np.geomspace(d * 1e-12, d / 2, 4000)         # DC cell: log grid
    out[nu == 0] = 2 * _trapz(fun(fg), fg) / d
    return out


def _step_up(k):
    a = np.array([1.0])
    for ki in k:
        a = np.r_[a, 0.0] + ki * np.r_[0.0, a[::-1]]
    return a


def _step_up_jac(k):
    """Step-up recursion and its exact Jacobian da/dk (shape (N+1, N), da0/dk = 0)."""
    N = len(k)
    a, D = np.array([1.0]), np.zeros((1, N))
    for m, km in enumerate(k):
        z = np.zeros((1, N))
        Dn = np.vstack([D, z]) + km * np.vstack([z, D[::-1]])
        Dn[:, m] += np.r_[0.0, a[::-1]]
        a, D = np.r_[a, 0.0] + km * np.r_[0.0, a[::-1]], Dn
    return a, D


def _step_down(a):
    a = np.array(a, float)
    k = np.zeros(len(a) - 1)
    for i in range(len(a) - 1, 0, -1):
        k[i - 1] = a[i]
        a = (a[:i] - k[i - 1] * a[i:0:-1]) / (1 - k[i - 1] ** 2)
    return k


def _stabilise(a, rmax=0.99999):
    r = np.roots(a)
    r = np.where(np.abs(r) > 1, 1 / np.conj(r), r)
    r = np.where(np.abs(r) > rmax, rmax * r / np.abs(r), r)
    return np.real(np.poly(r))


def _b_given_a(spec, Z, a, g):
    """Exact optimal numerator for a fixed denominator (cost is quadratic in b)."""
    X = Z / (Z @ a)[:, None]
    Q = np.real(np.conj(X).T @ (X * spec.Pyy(g)[:, None]))
    r = np.real(np.conj(X).T @ spec.Psm)
    return np.linalg.lstsq(Q, r, rcond=None)[0]


# ------------------------------------------------------------------------ design
def wiener_bound(spec, g=1.0, F0=None, n_iter=2):
    """
    Infinite-order causal Wiener filter F_opt on the grid (Wiener-Hopf),
    F_opt = F0 + [ (Psm - F0 Pyy) / G* ]_+ / G,  |G|^2 = Pyy, G minimum phase.
    F0 is any causal starting filter on the grid (default 0, the textbook form).
    The causal projection is circular, so its error scales with the part of
    Psm not already explained by F0: for long-memory spectra (e.g. the f^-2/3
    tilt) F0 = 0 gives a badly wrong filter, while starting from a good
    finite-order filter (as wiener_iir does) makes the result grid-independent.
    The correction is iterated n_iter times.
    """
    Pyy = spec.Pyy(g)
    P = np.maximum(Pyy, 1e-14 * Pyy.max())
    c = np.real(np.fft.ifft(np.log(P)))                  # cepstrum
    M = spec.M
    c[1:M // 2] *= 2
    c[M // 2 + 1:] = 0
    Gm = np.exp(0.5 * np.fft.fft(c))                     # min-phase, |Gm|^2 = Pyy
    F = np.zeros(M, complex) if F0 is None else np.asarray(F0, complex).copy()
    for _ in range(n_iter if F0 is not None else 1):
        x = np.fft.ifft((spec.Psm - F * Pyy) / np.conj(Gm))
        x[M // 2:] = 0                                   # causal part [.]_+
        F = F + np.fft.fft(x) / Gm
    return F


KMAX = 0.999999                                       # |reflection coeff| < KMAX


def wiener_iir(spec, order, g=1.0, n_sm=30, n_random=1, seed=0,
               gtol=1e-10, ftol=1e-13, init=None):
    """
    Optimal order-N IIR filter for POL control (N = deg B = deg A).
    Returns b, a (powers of z^-1, a[0] = 1) and info with variances:
      residual, pred_err, bound (infinite-order residual), hold_err, open_loop,
      info['lower'] = [(b, a, residual) for orders 1..N], and
      info['F_bound'] = infinite-order Wiener filter on the grid (spec.z1).
    Orders are built up 1..N. At each order, several starting points are refined
    on the exact cost with an exact gradient (reflection-coefficient
    parametrisation of A, analytic step-up Jacobian):
      Steiglitz-McBride fit to the infinite-order filter; the order N-1 optimum
      (exactly, so the residual cannot increase with N); the order N-1 optimum
      with the new reflection coefficient at +-0.5; n_random random stable A's;
      at N = 1 also single real poles r = 0, 0.5, 0.9, 0.99, 0.999.
    For every A start, B starts at its exact optimum given A.
    init: optional {N: v} of solutions of a neighbouring problem (info['v']),
      e.g. the previous point of a noise sweep. When given, only the SM fit, the
      order N-1 optimum and init[N] are used as starts (continuation: ~2x faster).
    """
    Fopt = wiener_bound(spec, g)
    Pyy = spec.Pyy(g)
    Zfull = spec.z1[:, None] ** np.arange(order + 1)[None, :]
    rng = np.random.default_rng(seed)
    lower, prev, vs = [], None, {}
    for N in range(1, order + 1):
        Z = Zfull[:, :N + 1]

        @np.errstate(divide="ignore", invalid="ignore", over="ignore")
        def cost_grad(v):
            t = np.tanh(v[N + 1:])
            a, Dk = _step_up_jac(KMAX * t)
            b = v[:N + 1]
            A = Z @ a
            F = (Z @ b) / A
            E = F * Pyy - spec.Psm                           # dJ/dF* (x 1/2)
            J = np.mean(spec.Pmm - 2 * np.real(spec.Psm * np.conj(F))
                        + np.abs(F) ** 2 * Pyy)
            gb = 2 * np.real(np.conj(Z / A[:, None]).T @ E) / len(E)
            ga = -2 * np.real(np.conj(Z * (F / A)[:, None]).T @ E) / len(E)
            grad = np.r_[gb, (Dk.T @ ga) * KMAX * (1 - t ** 2)]
            if not (np.isfinite(J) and np.all(np.isfinite(grad))):
                return 1e30, np.zeros_like(v)                # pole hit |z| = 1
            return J, grad

        def from_theta(th):                    # A from reflection params, B optimal given A
            return np.r_[_b_given_a(spec, Z, _step_up(KMAX * np.tanh(th)), g), th]

        def from_a(a):                                       # start from a stable A
            k = np.clip(_step_down(a), -0.999 * KMAX, 0.999 * KMAX)
            return from_theta(np.arctanh(k / KMAX))

        # Steiglitz-McBride, min sum w |B - A Fopt|^2, w = Pyy/|A_prev|^2
        a = np.r_[1.0, np.zeros(N)]
        for _ in range(n_sm):
            w = np.sqrt(Pyy) / np.abs(Z @ a)
            Mx = np.hstack([Z, -Fopt[:, None] * Z[:, 1:]]) * w[:, None]
            rhs = Fopt * w
            sol = np.linalg.lstsq(np.vstack([Mx.real, Mx.imag]),
                                  np.r_[rhs.real, rhs.imag], rcond=None)[0]
            if not np.all(np.isfinite(sol)):
                break
            a = _stabilise(np.r_[1.0, sol[N + 1:]])
        starts = [from_a(a)]
        warm = init is not None and N in init
        if warm:
            starts.append(np.asarray(init[N], float))
        if N == 1 and not warm:
            starts += [from_a(np.r_[1.0, -r]) for r in (0.0, 0.5, 0.9, 0.99, 0.999)]
        if prev is not None:
            bp, thp = prev[:N], prev[N:]
            starts.append(np.r_[bp, 0.0, thp, 0.0])          # exact order N-1 optimum
            starts += [from_theta(np.r_[thp, np.arctanh(kN / KMAX)])
                       for kN in (() if warm else (0.5, -0.5))]
        starts += [from_theta(np.arctanh(rng.uniform(-0.9, 0.9, N) / KMAX))
                   for _ in range(0 if warm else n_random)]

        best, Jbest = None, np.inf
        for v0 in starts:
            res = minimize(cost_grad, v0, jac=True, method="L-BFGS-B",
                           options=dict(maxiter=5000, ftol=ftol, gtol=gtol))
            for v in (res.x, v0):
                J = cost_grad(v)[0]
                if J < Jbest:
                    best, Jbest = v, J
        b, a = best[:N + 1], _step_up(KMAX * np.tanh(best[N + 1:]))
        prev = vs[N] = best
        lower.append((b, a, Jbest + spec.hold_err))

    # infinite-order bound, refined from the best finite-order filter
    Fopt = wiener_bound(spec, g, F0=(Zfull @ b) / (Zfull @ a))
    bound = spec.pred_err(Fopt, g) + spec.hold_err
    return b, a, dict(pred_err=Jbest, residual=Jbest + spec.hold_err, bound=bound,
                      hold_err=spec.hold_err, open_loop=spec.open_loop, lower=lower,
                      F_bound=Fopt, v=vs)
