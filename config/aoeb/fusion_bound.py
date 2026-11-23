"""Causal bounds of multi-sensor fusion, per mode (ASSUMPTIONS A40): the best linear predictor of the DM command from the measurements of
several sensors, at any rates, with exposure averaging, latency and DM hold as in wiener.py / aoeb (A11).

Every quantity is a box average of a stationary process ending at some time offset:
  target    s  = mean of the turbulence x over the hold window of a command (length T_cmd, ends at t_c + T_cmd, t_c = command time);
  channel   y_c = mean of x + alpha_i over the exposure of sensor i (length T_i, ends at o_c <= t_c - latency) + white noise n_i,
with alpha_i the aliasing of sensor i (shape S_alpha, covariance C_alias between sensors, constant coherence) and noise independent
between sensors and frames. For sensors at different rates the problem is LIFTED to the common period P (the slow frame): the fast
sensor gives r = P / T_fast channels per period, and each phase p of the fast commands is a separate scalar-target problem whose channels
are the most recent samples available at its command time (causal at the lifted rate = causal in real time). Spectra at the lifted rate:
  P_cd(nu) = (1/P) sum_l S(f_l) H_c(f_l) conj(H_d(f_l)),  f_l = nu + l / P,  H_c(f) = sinc(f T_c) exp(-i pi f T_c) exp(2 pi i f o_c),
the causal Wiener filter from Wilson's matrix spectral factorisation, F = [P_sy G^-H]_+ G^-1, its error on the grid, plus the hold error
(variance of x within the hold window, which no command removes). Everything in units of the mode's turbulence variance (unit_psd).
POL with known optical gains (as wiener.py): a lower bound for any controller, static or dynamic fusion.
"""
import numpy as np

from . import utils

from .wiener import _cell_mean


def _H(X):
    return np.conj(np.swapaxes(X, -1, -2))


def wilson(S, tol=1e-12, maxit=60):
    """Minimum-phase causal factor G of a spectral density matrix on the circle, S = G G^H (Wilson 1972), S (M, K, K) in FFT order."""
    M, K, _ = S.shape
    h = np.linalg.cholesky(np.real(np.fft.ifft(S, axis=0)[0]))
    G = np.broadcast_to(h.astype(complex), S.shape).copy()
    I = np.eye(K)
    for _ in range(maxit):
        Gi = np.linalg.inv(G)
        g = Gi @ S @ _H(Gi) + I
        beta = np.fft.ifft(g, axis=0)
        beta[0] = np.triu(0.5 * beta[0])
        beta[M // 2 + 1:] = 0.0
        new = G @ np.fft.fft(beta, axis=0)
        d = np.abs(new - G).max() / np.abs(new).max()
        G = new
        if d < tol:
            break
    return G


def causal(X):
    """[X]_+: the non-negative lags (powers of z^-1) of X on the grid."""
    x = np.fft.ifft(X, axis=0)
    x[len(X) // 2:] = 0.0
    return np.fft.fft(x, axis=0)


class BoxSpectra:
    """Lifted-rate spectra of box-averaged channels of the turbulence (unit_psd S_x), aliasing (unit shape S_a) and white noise."""

    def __init__(self, S_x, S_a, period, M=2 ** 13, n_fold=None, f_max=None):
        self.S_x, self.S_a, self.P, self.M = S_x, S_a, float(period), int(M)
        self.nu = np.fft.fftfreq(M, self.P)                                   # [-1/2P, 1/2P) in FFT order
        nf = n_fold if n_fold is not None else int(np.ceil((f_max or 20.0 / self.P) * self.P))
        self.f = self.nu[:, None] + np.arange(-nf, nf + 1)[None, :] / self.P   # (M, L) spectral images
        self.l0 = nf
        two = lambda S: (lambda f: 0.5 * S(np.abs(f)))
        self.Sx = self._two(two(S_x))
        self.Sa = None if S_a is None else self._two(two(S_a))

    def _two(self, fun):
        """Two-sided PSD on the images, the in-band column averaged over its cell (resolves the low-frequency features)."""
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            S = fun(self.f)
        S[:, self.l0] = _cell_mean(fun, self.nu, 1.0 / (self.P * self.M))
        return np.nan_to_num(S)

    def h(self, length, end):
        f = self.f
        return np.sinc(f * length) * np.exp(-1j * np.pi * f * length) * np.exp(2j * np.pi * f * end)

    def cross(self, S, ha, hb):
        return (S * ha * np.conj(hb)).sum(1) / self.P


def bound(spec, target, channels, C_alias, noise, open_loop=None, n_iter=3):
    """Causal Wiener bound of ONE target from channels. target = (length, end); channels = [(sensor, length, end)];
    C_alias (S, S), noise (S,) per-sample variances per sensor (same units: turbulence variance = 1). Returns (error incl. hold, hold)."""
    hs = spec.h(*target)
    hc = [spec.h(L, e) for _, L, e in channels]
    K = len(channels)
    Pyy = np.empty((spec.M, K, K), complex)
    Psy = np.empty((spec.M, 1, K), complex)
    for c, (sc, _, _) in enumerate(channels):
        Psy[:, 0, c] = spec.cross(spec.Sx, hs, hc[c])
        for d, (sd, _, _) in enumerate(channels):
            if d < c:
                continue
            v = spec.cross(spec.Sx, hc[c], hc[d])
            if spec.Sa is not None and C_alias[sc, sd] != 0:
                v = v + C_alias[sc, sd] * spec.cross(spec.Sa, hc[c], hc[d])
            if c == d:
                v = v + noise[sc]
            Pyy[:, c, d] = v
            Pyy[:, d, c] = np.conj(v)
    Pss = np.real(spec.cross(spec.Sx, hs, hs))
    D = np.sqrt(np.real(np.einsum('mii->i', Pyy) / spec.M))
    G = wilson(Pyy / D[None, :, None] / D[None, None, :]) * D[None, :, None]
    Gi = np.linalg.inv(G)
    err = lambda F: float(np.real(Pss - 2 * np.real((Psy @ _H(F))[:, 0, 0]) + (F @ Pyy @ _H(F))[:, 0, 0]).mean())
    # The causal projection is circular: from F0 = 0 a long-memory mode (tilt) gets a badly wrong filter (wiener.wiener_bound). Each start
    # F0 = DC-unity exponential smoother on every channel (1/K each), then F <- F + [(Psy - F Pyy) G^-H]_+ G^-1 (n_iter): every iterate is a
    # causal filter evaluated exactly, so the minimum is an achievable error and converges to the bound.
    z1 = np.exp(-2j * np.pi * spec.nu * spec.P)
    E = np.inf
    for a in (None, 0.5, 0.9, 0.99, 0.999):
        F = (np.zeros((spec.M, 1, K), complex) if a is None else
             np.broadcast_to(((1 - a) / (1 - a * z1) / K)[:, None, None], (spec.M, 1, K)).astype(complex))
        for _ in range(1 if a is None else n_iter):
            F = F + causal((Psy - F @ Pyy) @ _H(Gi)) @ Gi
            E = min(E, err(F))
    if open_loop is None:                                # unfiltered variance on the same images (consistent truncation)
        open_loop = float((spec.Sx.sum(1) / spec.P).mean())
    hold = open_loop - Pss.mean()
    return float(E + hold), float(hold)


def mode_bounds(S_x, S_a, sensors, C_alias, latency_s, fs_cmd, M=2 ** 13, weights=None):
    """Bounds of one mode (turbulence variance 1). sensors: [dict(fs, noise)] (noise per frame, same units), C_alias between them.
    The commands run at fs_cmd (= the fastest sensor); the slow frame period is the common period (rates must be integer multiples).
    Returns dict: 'single_i' each sensor alone at its own rate (its commands at its rate), 'static' one channel with static weights
    (synchronous sensors only, weights given or GLS on noise + aliasing), 'optimal' all sensors, lifted."""
    fss = [float(s['fs']) for s in sensors]
    Tc = 1.0 / fs_cmd
    P = 1.0 / min(fss)
    ratios = [round(fs / min(fss)) for fs in fss]
    if any(abs(r - fs / min(fss)) > 1e-6 for r, fs in zip(ratios, fss)) or abs(fs_cmd - max(fss)) > 1e-6:
        raise ValueError('mode_bounds: sensor rates must be integer multiples of the slowest, commands at the fastest')
    r_cmd = round(P * fs_cmd)
    spec = BoxSpectra(S_x, S_a, P, M)
    noise = np.array([s['noise'] for s in sensors], float)
    out = {}
    for i, s in enumerate(sensors):                      # alone, commands at its own rate
        T = 1.0 / fss[i]
        sp = spec if abs(T - P) < 1e-12 else BoxSpectra(S_x, S_a, T, M)
        out[f'single_{i}'] = bound(sp, (T, latency_s + T), [(i, T, 0.0)], C_alias, noise)[0]
    if all(abs(fs - fss[0]) < 1e-9 for fs in fss):       # synchronous: static weights as one channel, and all channels
        T = 1.0 / fss[0]
        if weights is None:
            Sig = np.diag(noise) + C_alias
            x = np.linalg.solve(Sig, np.ones(len(noise)))
            weights = x / x.sum()
        w = np.asarray(weights, float)
        Cw = np.array([[w @ C_alias @ w]])
        out['static'] = bound(spec, (T, latency_s + T), [(0, T, 0.0)], Cw, np.array([w ** 2 @ noise]))[0]
        out['weights'] = w
        out['optimal'] = bound(spec, (T, latency_s + T), [(i, T, 0.0) for i in range(len(sensors))], C_alias, noise)[0]
        return out
    errs = []                                            # lifted: one problem per command phase within the slow period
    for p in range(r_cmd):
        t_last = p * Tc                                  # end of the latest fast exposure (phase p); slow exposures end at -Tc, P - Tc, ...
        chans = []
        for i, (fs, r) in enumerate(zip(fss, ratios)):
            T = 1.0 / fs
            for q in range(r):                           # the r most recent frames of sensor i ending at or before t_last
                ends = T * np.floor((t_last + Tc) / T + 1e-9) - Tc        # sensors' frames all end together at the period boundary
                chans.append((i, T, ends - q * T - t_last - latency_s))
        errs.append(bound(spec, (Tc, Tc), chans, C_alias, noise)[0])     # offsets relative to the command time t_last + latency
    out['optimal'] = float(np.mean(errs))
    return out


# ----------------------------------------------------------------------------- driver on an ErrorBudgetMachine
def _psd(f, n, wind, D):
    from . import filter_optimizers as fo
    return fo.unit_psd(f, n, wind, D)


def _task(args):
    n, sensors, C, latency_s, fs_cmd, M, weights, wind, D, alias = args
    Sx = lambda f: _psd(f, n, wind, D)
    Sa = (lambda f: alias.unit_psd(f, n)) if alias is not None else None
    return mode_bounds(Sx, Sa, sensors, C, latency_s, fs_cmd, M, weights)


def machine_inputs(machine, sensors, seeing, N, alias_correlation='measured', rec=('svd',)):
    """Per-mode inputs of mode_bounds from the machine (fusion_inputs): noise per frame and aliasing covariance in units of each mode's
    turbulence variance, the turbulence [nm^2 OPD], radial orders and the aliasing spectrum of the first sensor."""
    inp = machine.fusion_inputs(sensors, seeing, N, rec, alias_correlation)
    turb_cmd, rms, _ = machine.turbulence(seeing, N)
    a = inp['alias']
    s0 = sensors[0]
    return dict(noise=(inp['photon'] + inp['readout']) / turb_cmd[:, None], C=inp['rho'] * a[:, :, None] * a[:, None, :] / turb_cmd[:, None, None],
                turb_nm2=turb_cmd * rms ** 2, orders=utils.radial_order(np.arange(N)), fs=[float(s['fs']) for s in sensors],
                alias=machine.alias_spectrum(float(s0['wavelength_nm']), N, s0.get('rMod')), alias_correlation=inp['alias_correlation'])


def machine_bounds(machine, inputs, latency_s, modes, fs_cmd=None, weights=None, M=2 ** 13, jobs=32):
    """mode_bounds of the given modes, in nm^2 OPD: {key: (len(modes),)} with the keys of mode_bounds; weights: (N, K) static weights
    (default GLS on noise + aliasing). Runs on `jobs` forked processes."""
    import concurrent.futures as cf
    import multiprocessing as mp
    from . import utils as _u                                                # noqa: F401
    fs_cmd = fs_cmd or max(inputs['fs'])
    wind = machine.temporal or machine.wind
    args = [(int(inputs['orders'][i]), [dict(fs=f, noise=float(inputs['noise'][i, k])) for k, f in enumerate(inputs['fs'])],
             inputs['C'][i], latency_s, fs_cmd, M, None if weights is None else weights[i], wind, machine.D, inputs['alias']) for i in modes]
    with cf.ProcessPoolExecutor(jobs, mp_context=mp.get_context('fork')) as ex:
        res = list(ex.map(_task, args, chunksize=1))
    keys = [k for k in res[0] if k != 'weights']
    out = {k: np.array([r[k] for r in res]) * inputs['turb_nm2'][modes] for k in keys}
    if 'weights' in res[0]:
        out['weights'] = np.array([r['weights'] for r in res])
    return out
