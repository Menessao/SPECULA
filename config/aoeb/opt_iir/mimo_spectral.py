"""
All-modes (MIMO) against mode-by-mode (SISO) Wiener bound, from spectra rebuilt with the error-budget (EB) ingredients.

Inputs (rMod 0, 48x48, 900 nm, 1300-mode reconstructor, seeing 0.9", fs 2 kHz; the first NM modes are kept):
  turbulence: frozen-flow von Karman cross-spectra of the KL modes, from the EB KL shapes and projection (machine.modes(), divided by
    their rms, as utils.modal_variance_von_karman: the zero-lag variances are the EB ones) and the 10 layers of ristretto_main.yml
    (Cn2 fractions, speeds AND directions, which the EB averages out). Two-sided S_ij(f) with R_ij(tau) = E[c_i(t + tau) c_j(t)]
    = int S_ij(f) exp(2 pi i f tau) df:  S_ij(f) = sum_l w_l int Phi(k) conj(F_i(k)) F_j(k) delta(f + k.v_l) d^2k, F = FFT of the
    projector (exp(-2 pi i k.r)). Sampled at fs and averaged over the WFS frame (|sinc|^2), folded onto the M-point grid by
    linear (cloud-in-cell) binning of the k-grid points.
  aliasing: 'eb', the EB per-mode variance / og^2 with the measured shape of its radial order (as aoeb_order.py), uncorrelated between
    modes; or 'measured', the cross-spectra of the simulated aliasing (alias_psd_rmod0, rot 0 = the yml directions, 4 seeds) / og.
  noise: full modal covariance R Sigma_s R^T (EB slope-noise model, aoeb.noise.slope_noise_cov, sx-sy covariance included) / og^2,
    white; its diagonal is the EB noise.
Timing as wiener_ao_iir.ModeSpectra (POL, total delay 2 frames = 1 + tau, tau = 1): the target (hold average of the command
interval) is the WFS-averaged turbulence two frames ahead, s_k = m_{k+2}.

Bounds (prediction error E|s - u|^2 summed over the modes, + the hold error):
  SISO: per mode, wiener_ao_iir.wiener_iir (exact infinite-order bound refined from the IIR2 filter) on the diagonal spectra;
  MIMO frequency domain: Wilson's multivariate spectral factorisation Pyy = G G^H, F = F0 + [(Psm - F0 Pyy) G^-H]_+ G^-1 iterated
    from F0 = the SISO bound filters (diagonal), the error evaluated exactly on the grid;
  VAR + Kalman: VAR(p) of the turbulence (and of the aliasing) from the lag covariances of the same spectra (multivariate Yule-Walker),
    steady-state Kalman predictor of m_{k+2} (scipy DARE); 'diag' = one AR(p) per mode (the SISO counterpart).

    python aoeb/opt_iir/mimo_spectral.py check        # spectra: EB variances, comparison with the simulated atmo series
    python aoeb/opt_iir/mimo_spectral.py bounds       # SISO / MIMO frequency domain / VAR-Kalman at mags 5, 8, 11
"""
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N_THREADS = int(os.environ.get('AOEB_MIMO_THREADS', 32))       # frequency-split threads of the batched small-matrix algebra


def blas_threads(n):
    """Set the thread count of numpy's bundled OpenBLAS at run time (the batched 100x100 algebra runs on N_THREADS Python threads with one
    BLAS thread each; the dense Kalman algebra on many BLAS threads). No-op if the symbol is not found."""
    import ctypes
    for line in open('/proc/self/maps'):
        if 'openblas' in line and line.rstrip().endswith('.so') or ('openblas' in line and '.so' in line):
            lib = ctypes.CDLL(line.split()[-1])
            for name in ('openblas_set_num_threads64_', 'openblas_set_num_threads'):
                if hasattr(lib, name):
                    getattr(lib, name)(int(n))
                    return True
    return False


def par(f, *arrays, n=None):
    """f applied to frequency chunks of the arrays (axis 0) on N_THREADS threads (numpy releases the GIL), one BLAS thread each."""
    from concurrent.futures import ThreadPoolExecutor
    n = n or N_THREADS
    idx = np.array_split(np.arange(len(arrays[0])), 4 * n)
    first = f(*(a[idx[0]] for a in arrays))
    out = np.empty((len(arrays[0]),) + first.shape[1:], first.dtype)
    out[idx[0]] = first
    blas_threads(1)
    try:
        def work(ix):
            out[ix] = f(*(a[ix] for a in arrays))
        with ThreadPoolExecutor(n) as pool:
            list(pool.map(work, idx[1:]))
    finally:
        blas_threads(16)
    return out
sys.path.insert(0, HERE)
import aoeb_order as A                      # noqa: E402  (machine, EB inputs, constants)

NM = int(os.environ.get('AOEB_MIMO_NM', 100))     # modes kept (1..NM)
M = int(os.environ.get('AOEB_MIMO_M', 2 ** 13))   # frequency grid over [0, fs)
FS, T = A.FS, 1.0 / A.FS
PAD, KCUT = 8, 6.0
VAR_EPS = float(os.environ.get('AOEB_MIMO_VAR_EPS', 1e-8))                                 # FFT padding of the pupil (as the EB); |k| kept [1/m]
MAGS = (5.0, 8.0, 11.0)
# orientation of the wind in the k-grid of the projector FFT (axis 0 = rows): f = -V (sx kx cos(theta) + sy ky sin(theta)), swapped
# axes if CONV[2]; fixed by `check` against the simulated atmo series
CONV = tuple(int(v) for v in os.environ.get('AOEB_MIMO_CONV', '-1,1,0').split(','))
OUT = '/raid1/mmenessini/results/WFSW/mimo_spectral'
CACHE = os.path.join(OUT, f'turb_rmod0_N{NM}_M{M}.npy')       # Pmm (large: kept out of the repository)
ATMO = '/raid1/mmenessini/results/WFSW/alias_psd_mimo/modes/m0_rot0_seed{s}.npz'
ALIAS = '/raid1/mmenessini/results/WFSW/alias_psd_rmod0/modes/r0_rot0_seed{s}.npz'


def layers():
    """(Cn2 fraction, speed [m/s], direction [deg]) of the ristretto_main.yml layers."""
    import yaml
    Y = yaml.safe_load(open(os.path.join(A.CONFIG, 'aoeb', 'e2e', 'ristretto_main.yml')))
    w = np.array(Y['atmo']['Cn2'], float)
    return w / w.sum(), np.array(Y['wind_speed']['constant'], float), np.array(Y['wind_direction']['constant'], float)


def projector_ft(m, nm=NM, kcut=KCUT):
    """FFT of the EB projectors (mode / rms / area) of modes 1..nm on the padded grid, restricted to |k| <= kcut:
    kx, ky [1/m] (kx along the columns), F (nm, n_k) complex, dk^2 [1/m^2]."""
    pupil = m.pupil() > 0
    n = pupil.shape[0]
    dx, Mg = m.D / n, n * PAD
    modes = np.asarray(m.modes()[:nm], float)
    rms = np.sqrt(np.mean(modes ** 2, axis=1))
    area = pupil.sum() * dx * dx
    f = np.fft.fftfreq(Mg, d=dx)
    ky, kx = np.meshgrid(f, f, indexing='ij')
    keep = np.hypot(kx, ky) <= kcut
    rows, cols = np.nonzero(pupil)
    F = np.empty((nm, keep.sum()), complex)
    for i in range(nm):
        buf = np.zeros((n, n))
        buf[rows, cols] = modes[i] / rms[i]
        F[i] = np.fft.fft2(buf, s=(Mg, Mg))[keep] * dx * dx / area
    return kx[keep], ky[keep], F, (1.0 / (Mg * dx)) ** 2


def turb_spectra(m, conv=CONV, nm=NM, Mf=M):
    """Folded discrete-time turbulence cross-spectra, WFS-averaged, on the grid nu = fftfreq(Mf, T) (FFT order), Phi convention
    (covariance = mean over the grid): Pmm (Mf, nm, nm) complex. Cached (CACHE) for the default arguments."""
    from aoeb import utils
    cache = (conv, nm, Mf) == (CONV, NM, M)
    if cache and os.path.exists(CACHE):
        return np.load(CACHE)
    r0 = float(utils.r0_from_seeing(A.SEEING))
    kx, ky, F, dk2 = projector_ft(m, nm)
    if conv[2]:
        kx, ky = ky, kx
    phi = utils.von_karman_psd(np.hypot(kx, ky), r0, m.L0) * dk2
    w, V, th = layers()
    df = FS / Mf
    Pmm = np.zeros((Mf, nm, nm), complex)
    Fc = np.conj(F)
    for wl, Vl, tl in zip(w, V, np.radians(th)):
        f = -Vl * (conv[0] * kx * np.cos(tl) + conv[1] * ky * np.sin(tl))
        p = f / df
        i0 = np.floor(p).astype(np.int64)
        t = p - i0
        box = np.sinc(f * T) ** 2
        for shift, frac in ((0, 1.0 - t), (1, t)):               # cloud-in-cell split between the two nearest bins
            b = (i0 + shift) % Mf                                  # folding: the bin of f mod fs
            order = np.argsort(b, kind='stable')
            bs = b[order]
            cuts = np.flatnonzero(np.diff(bs)) + 1
            wt = (wl * phi * frac * box)[order]
            Fs, Fcs = F[:, order], Fc[:, order]
            for s, e in zip(np.r_[0, cuts], np.r_[cuts, len(bs)]):
                Pmm[bs[s]] += (Fcs[:, s:e] * wt[s:e]) @ Fs[:, s:e].T
    Pmm *= Mf
    if cache:
        os.makedirs(OUT, exist_ok=True)
        np.save(CACHE, Pmm)
    return Pmm


def lag_cov(P, lags):
    """Real lag covariances Gamma(l) = E[x_{k+l} x_k^T] (l in lags) of a spectrum on the grid (Phi convention)."""
    return np.real(np.fft.ifft(P, axis=0)[np.asarray(lags) % len(P)])


def eb_inputs(m, mags=MAGS):
    """EB values of modes 1..NM: dict(turb, alias, og, order, noise {mag: per-mode}, alias_spec) [nm^2], as aoeb_order.inputs."""
    I = A.inputs(mags=list(mags))
    js = range(1, NM + 1)
    return dict(turb=np.array([I['turb'][j] for j in js]), alias=np.array([I['alias'][j] for j in js]),
                og=np.array([I['og'][j] for j in js]), order=np.array([I['order'][j] for j in js]),
                noise={mg: np.array([I['noise'][mg][j] for j in js]) for mg in mags}, alias_spec=I['alias_spec'])


def alias_eb(E, Mf=M):
    """EB aliasing: per-mode variance / og^2, measured shape of its radial order, folded and WFS-averaged as ModeSpectra; diagonal."""
    from wiener_ao_iir import ModeSpectra
    P = np.zeros((Mf, NM, NM), complex)
    for i in range(NM):
        n, v = int(E['order'][i]), float(E['alias'][i])
        P[:, i, i] = ModeSpectra(lambda f: 0.0 * f, FS, 1.0, lambda f: v * E['alias_spec'].unit_psd(f, n), M=Mf).Paa
    return P


def alias_measured(E, Mf=M, L=1024, seeds=(1001, 1101, 1201, 1301), to_eb=True):
    """Measured aliasing cross-spectra: Welch (Hann, L frames, 50 % overlap) of the simulated aliasing of rot 0 (the yml wind
    directions), first NM modes / og, linearly interpolated onto the Mf grid (keeps every matrix positive semi-definite); to_eb:
    every mode rescaled to the EB aliasing variance (correlations and spectral shapes kept)."""
    win = np.hanning(L)
    acc, n = np.zeros((L, NM, NM), complex), 0
    for s in seeds:
        a = np.load(ALIAS.format(s=s))['modes_900'][:, :NM].astype(float) / E['og']
        a -= a.mean(0)
        for k0 in range(0, len(a) - L + 1, L // 2):
            X = np.fft.fft(a[k0:k0 + L] * win[:, None], axis=0)
            acc += X[:, :, None] * np.conj(X[:, None, :])
            n += 1
    P = acc / n / (win ** 2).sum()                                # Phi convention on the L grid
    nuL, nuM = np.fft.fftfreq(L, T), np.fft.fftfreq(Mf, T)
    o = np.argsort(nuL)
    x = np.r_[nuL[o], nuL[o][0] + FS]                              # circular
    y = np.concatenate([P[o], P[o][:1]])
    pos = np.interp(nuM, x, np.arange(len(x)))                      # fractional index on the sorted grid
    i0 = np.minimum(np.floor(pos).astype(int), len(x) - 2)
    t = (pos - i0)[:, None, None]
    P = (1 - t) * y[i0] + t * y[i0 + 1]
    if to_eb:                    # each mode scaled to the EB variance (the simulation has 0.6-0.95 of it): EB levels, measured correlations
        c = np.sqrt(E['alias'] / np.real(np.diag(P.mean(0))))
        P *= c[None, :, None] * c[None, None, :]
    return P


def noise_cov(m, E, mags=MAGS):
    """Full modal noise covariance [nm^2 per frame] of modes 1..NM per magnitude: R Sigma_s R^T with the EB slope-noise model
    (slope_noise_cov: per-slope variance and sx-sy covariance), the 1300-mode SVD reconstructor of the EB, / og^2."""
    import yaml
    from specula.lib.n_phot import n_phot
    from aoeb import utils
    from aoeb.naming import snap_seeing
    from aoeb.noise import slope_noise_cov
    rM, ns, N, wl = 0.0, A.NSUB, A.N, A.WL
    rpath = m._ensure_rec(rM, ns, N, ('svd',), snap_seeing(A.SEEING), wl, wl, A.SEEING)
    R = np.asarray(m._read_rec(rpath), float)[:NM]
    P = m.paths.at(wl)
    frame = utils.read_fits(P.avgframe(rM, ns, A.SEEING, N))
    geom = m.slope_geometry(rM, ns, A.SEEING, N, wl, frame)
    rms = m.turbulence(A.SEEING, N)[1][:NM]
    Y = yaml.safe_load(open(os.path.join(A.CONFIG, 'aoeb', 'e2e', 'ristretto_main.yml')))
    area = (m.pupil() > 0).sum() * Y['main']['pixel_pitch'] ** 2
    out = {}
    nv = R.shape[1] // 2
    Rx, Ry = R[:, :nv], R[:, nv:]
    for mg in mags:
        flux = n_phot(mg, lambda_=wl * 1e-9, width=1e-9)[0] * Y['cred_vis']['bandw'] * area
        cn, cxy = slope_noise_cov(frame, geom, m.detector.n_det(flux, FS), m.detector)
        C = (Rx * cn[:nv]) @ Rx.T + (Ry * cn[nv:]) @ Ry.T + (Rx * cxy) @ Ry.T + (Ry * cxy) @ Rx.T
        s = rms / E['og']
        out[mg] = C * s[:, None] * s[None, :]
    return out


def mode_spec(Pmm_i, Paa_i, noise_i, var0_i, Mf=M):
    """wiener_ao_iir.ModeSpectra of one mode from given diagonal spectra (total delay 2: Psm = Pmm z^-2 conj, see the docstring)."""
    from wiener_ao_iir import ModeSpectra, pol_weights
    sp = ModeSpectra.__new__(ModeSpectra)
    nu = np.fft.fftfreq(Mf, T)
    sp.fs, sp.delay, sp.M, sp.p = FS, 1.0, Mf, pol_weights(1.0)
    sp.Pmm = np.real(Pmm_i)
    sp.Psm = sp.Pmm * np.exp(2j * np.pi * nu * 2 * T)
    sp.Paa, sp.Pnn = np.real(Paa_i), np.full(Mf, float(noise_i))
    sp.z1 = np.exp(-2j * np.pi * nu * T)
    sp.open_loop, sp.var_s = float(var0_i), sp.Pmm.mean()
    sp.hold_err = sp.open_loop - sp.var_s
    return sp


def _siso_task(args):
    from wiener_ao_iir import wiener_iir
    info = wiener_iir(mode_spec(*args), 2)[2]
    return info['bound'], info['hold_err'], info['F_bound'], info['residual']


def siso_bounds(Pmm, Paa, Cn, var0, jobs=24):
    """Per-mode Wiener bound (wiener_iir, refined from IIR2) on the diagonal spectra: (bound incl. hold [nm^2], hold, F_bound (Mf, NM),
    IIR2 residual) per mode."""
    import multiprocessing as mp
    args = [(Pmm[:, i, i], Paa[:, i, i], Cn[i, i], var0[i], len(Pmm)) for i in range(NM)]
    old = {v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    os.environ.update({v: '1' for v in old})               # spawned workers: one BLAS thread each (the parent keeps its threads)
    try:
        with mp.get_context('spawn').Pool(jobs) as pool:
            r = pool.map(_siso_task, args, chunksize=1)
    finally:
        for v, x in old.items():
            os.environ[v] = x if x is not None else '16'

    return (np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.stack([x[2] for x in r], 1), np.array([x[3] for x in r]))


def _H(X):
    return np.conj(np.swapaxes(X, -1, -2))


def wilson(S, tol=1e-12, maxit=40):
    """Minimum-phase causal factor psi of a spectral density matrix on the full circle, S = psi psi^H (Wilson 1972; the plus operator
    of Dhamala et al. 2008). S (Mf, N, N) in FFT order, causal = polynomial in z^-1 = exp(-2 pi i nu T)."""
    Mf, N, _ = S.shape
    h = np.linalg.cholesky(np.real(np.fft.ifft(S, axis=0)[0]))
    psi = np.broadcast_to(h.astype(complex), S.shape).copy()
    I = np.eye(N)
    for it in range(maxit):
        g = par(lambda p_, s_: (lambda Pi: Pi @ s_ @ _H(Pi))(np.linalg.inv(p_)), psi, S) + I
        beta = np.fft.ifft(g, axis=0)
        beta[0] = np.triu(0.5 * beta[0])
        beta[Mf // 2 + 1:] = 0.0
        new = par(np.matmul, psi, np.fft.fft(beta, axis=0))
        d = np.abs(new - psi).max() / np.abs(new).max()
        psi = new
        if d < tol:
            break
    res = np.abs(par(lambda p_: p_ @ _H(p_), psi) - S).max() / np.abs(S).max()
    return psi, it + 1, res


def causal(X):
    """[X]_+ : the non-negative lags (in z^-1) of X on the grid (lag 0 kept whole), as wiener_ao_iir.wiener_bound."""
    x = np.fft.ifft(X, axis=0)
    x[len(X) // 2:] = 0.0
    return np.fft.fft(x, axis=0)


def pred_err(F, Pmm, Paa, Cn):
    """Prediction-error matrix [nm^2] of the causal filter F (Mf, N, N) on the TRUE spectra (u = F y, target m_{k+2})."""
    nu = np.fft.fftfreq(len(Pmm), T)
    Psm = Pmm * np.exp(2j * np.pi * nu * 2 * T)[:, None, None]
    Pyy = Pmm + Paa + Cn[None]
    E = par(lambda f_, m_, s_, y_: m_ - s_ @ _H(f_) - f_ @ _H(s_) + f_ @ y_ @ _H(f_), F, Pmm, Psm, Pyy)
    return np.real(E.mean(0))


def mimo_bound(Pmm, Paa, Cn, F0=None, n_iter=3):
    """All-modes causal Wiener filter on the grid, F = F0 + [(Psm - F0 Pyy) G^-H]_+ G^-1 iterated; returns the prediction-error matrix
    (mean over the grid of Pmm - Psm F^H - F Psm^H + F Pyy F^H, [nm^2]), and the Wilson diagnostics."""
    Mf = len(Pmm)
    nu = np.fft.fftfreq(Mf, T)
    Pyy = Pmm + Paa + Cn[None]
    Psm = Pmm * np.exp(2j * np.pi * nu * 2 * T)[:, None, None]
    D = np.sqrt(np.real(np.diag(Pyy.mean(0))))                  # factorise the mode-normalised spectra (variances span 7 decades)
    G, it, res = wilson(Pyy / D[None, :, None] / D[None, None, :])
    G *= D[None, :, None]                                          # Pyy = (D G')(D G')^H
    Gi = par(np.linalg.inv, G)
    del G
    F = np.zeros_like(Pmm) if F0 is None else F0.copy()
    errs = []
    for _ in range(n_iter if F0 is not None else 1):
        X = causal(par(lambda s_, f_, y_, g_: (s_ - f_ @ y_) @ _H(g_), Psm, F, Pyy, Gi))
        F = F + par(np.matmul, X, Gi)
        E = pred_err(F, Pmm, Paa, Cn)
        errs.append(np.trace(E))
    return E, dict(wilson_iter=it, wilson_res=res, err_iter=errs)


def var_fit(Gam, p, diag=False, eps=VAR_EPS):
    """VAR(p) by multivariate Yule-Walker from lag covariances Gam[l] = E[x_{k+l} x_k^T] (l = 0..p): A (p, N, N), innovation Q.
    eps: white floor added to Gamma(0) (relative, in normalised units): the Yule-Walker system of the WFS-averaged low-order turbulence
    is otherwise near-singular (its spectrum falls as f^-17/3)."""
    G = [np.diag(np.diag(g)) for g in Gam] if diag else list(Gam)
    G[0] = G[0] + eps * np.eye(len(G[0]))
    N = G[0].shape[0]
    Gl = lambda l: G[l] if l >= 0 else G[-l].T
    Rb = np.block([[Gl(l - i) for l in range(1, p + 1)] for i in range(1, p + 1)])        # block (i, l) = Gamma(l - i)
    r = np.hstack([G[l] for l in range(1, p + 1)])
    Ab = np.linalg.solve(Rb.T, r.T).T
    As = Ab.reshape(N, p, N).transpose(1, 0, 2)
    Q = G[0] - sum(As[i] @ G[i + 1].T for i in range(p))
    return As, 0.5 * (Q + Q.T)


def companion(As):
    p, N, _ = As.shape
    Phi = np.zeros((N * p, N * p))
    Phi[:N] = np.hstack(list(As))
    Phi[N:, :-N] = np.eye(N * (p - 1))
    return Phi


def dare_sda(A, G, Q, tol=1e-13, maxit=200):
    """Solution X of X = A^T X (I + G X)^-1 A + Q by the structured doubling algorithm (Chu, Fan & Lin 2005; quadratic convergence,
    robust to near-unit-root dynamics). For the Kalman prior covariance: A = Phi^T, G = H^T R^-1 H, Q = process noise."""
    I = np.eye(len(A))
    X = Q.copy()
    for it in range(maxit):
        W = np.linalg.solve(I + G @ X, np.hstack([A, G]))          # (I + G X)^-1 [A, G]
        WA, WG = W[:, :len(A)], W[:, len(A):]
        A, G, Xn = A @ WA, G + A @ WG @ A.T, X + A.T @ X @ WA
        Xn, G = 0.5 * (Xn + Xn.T), 0.5 * (G + G.T)
        d = np.abs(Xn - X).max() / np.abs(Xn).max()
        X = Xn
        if d < tol:
            break
    return X, it + 1


def kalman_filter(Gm, Ga, Cn, p, q, diag=False, Mf=M):
    """Steady-state Kalman predictor of m_{k+2} from y_<=k, y = m + a + n, designed on VAR(p) turbulence and VAR(q) aliasing fitted to
    the lag covariances Gm, Ga (diag: one AR per mode and the noise diagonal, the SISO counterpart). Returns its transfer function
    F (Mf, N, N) on the grid (from the impulse response C Phi^2 A_f^l K, A_f = (I - K H) Phi), the model's own error matrix [nm^2]
    (valid only if the VAR reproduced the spectra) and the spectral radius of the VAR companion (>= 1: unstable fit)."""
    from scipy.linalg import block_diag
    N = Cn.shape[0]
    D = np.sqrt(np.diag(Gm[0]))                                  # normalised variables (the variances span 7 decades)
    sc = lambda X: X / D[:, None] / D[None, :]
    Gm, Ga, Cn = [sc(g) for g in Gm], [sc(g) for g in Ga], sc(Cn)
    Am, Qm = var_fit(Gm, p, diag)
    Aa, Qa = var_fit(Ga, q, diag)
    Phi = block_diag(companion(Am), companion(Aa))
    rho = np.abs(np.linalg.eigvals(Phi)).max()
    nm_, na_ = N * p, N * q
    Qw = np.zeros((nm_ + na_, nm_ + na_))
    Qw[:N, :N] = Qm
    Qw[nm_:nm_ + N, nm_:nm_ + N] = Qa
    H = np.zeros((N, nm_ + na_))
    H[:, :N] = np.eye(N)
    H[:, nm_:nm_ + N] = np.eye(N)
    Rn = np.diag(np.diag(Cn)) if diag else Cn
    P, _ = dare_sda(Phi.T, H.T @ np.linalg.solve(Rn, H), Qw)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rn)
    Pf = P - K @ H @ P
    P2 = Phi @ Phi @ Pf @ Phi.T @ Phi.T + Phi @ Qw @ Phi.T + Qw
    Af = Phi - K @ (H @ Phi)
    C2 = (Phi @ Phi)[:N]                                          # m_{k+2|k} = [Phi^2 x_{k|k}]_m
    h = np.zeros((Mf, N, N))
    X = K.copy()
    for l in range(Mf):                                          # impulse response; stops once it has decayed
        h[l] = C2 @ X
        X = Af @ X
        if l > 10 and np.abs(h[l]).max() < 1e-14 * np.abs(h[:l]).max():
            break
    F = np.fft.fft(h, axis=0) * D[None, :, None] / D[None, None, :]
    return F, P2[:N, :N] * D[:, None] * D[None, :], rho


def kalman_siso(Gm, Ga, Cn, p, q, Mf=M):
    """Per-mode VAR(p) / VAR(q) + Kalman (each mode on its own spectra and noise variance): diagonal transfer function (Mf, N, N) and
    the largest VAR spectral radius."""
    N = Cn.shape[0]
    F, rho = np.zeros((Mf, N, N), complex), 0.0
    for i in range(N):
        s_ = slice(i, i + 1)
        Fi, _, r = kalman_filter([g[s_, s_] for g in Gm], [g[s_, s_] for g in Ga], Cn[s_, s_], p, q, False, Mf)
        F[:, i, i], rho = Fi[:, 0, 0], max(rho, r)
    return F, rho


def check(conv=CONV):
    """Rebuilt turbulence spectra against the EB variances; aliasing (measured / EB) and noise diagonals against the EB."""
    m = A.machine()
    E = eb_inputs(m)
    Pmm = turb_spectra(m, conv)
    var_box = np.diag(lag_cov(Pmm, [0])[0])
    print(f'WFS-averaged / EB turbulence variance: median {np.median(var_box / E["turb"]):.4f} (the box removes the fast part)')
    Pam = alias_measured(E, to_eb=False)
    va = np.diag(lag_cov(Pam, [0])[0])
    print(f'measured aliasing variance / EB (alias / og^2): min {np.min(va / E["alias"]):.3f} median {np.median(va / E["alias"]):.3f} '
          f'max {np.max(va / E["alias"]):.3f}')
    Ga = lag_cov(Pam, [0])[0]
    d = np.sqrt(np.diag(Ga))
    print(f'measured aliasing: largest zero-lag correlation between modes {np.abs(Ga / d[:, None] / d[None] - np.eye(NM)).max():.3f}')
    C = noise_cov(m, E)
    for mg, c in C.items():
        r = np.diag(c) / E['noise'][mg]
        d = np.sqrt(np.diag(c))
        print(f'mag {mg:g}: noise diagonal / EB min {r.min():.4f} max {r.max():.4f}; largest correlation between modes '
              f'{np.abs(c / d[:, None] / d[None] - np.eye(NM)).max():.3f}')


def bounds(mags=MAGS, orders_siso=(16, 32, 64), orders_mimo=(16, 32)):
    """SISO and MIMO bounds per magnitude and aliasing model: frequency domain (SISO wiener_iir per mode; MIMO Wilson + iterated
    Wiener-Hopf from the SISO filters) and VAR + steady-state Kalman (aliasing VAR(min(p, 8))), every filter's error evaluated on the
    true spectra. Prediction errors per mode [nm^2] saved to OUT/bounds_N{NM}_M{M}.npz (+ 'hold')."""
    import time
    m = A.machine()
    E = eb_inputs(m, mags)
    Pmm = turb_spectra(m)
    var0 = E['turb']                                                 # instantaneous variance (the EB one, exact)
    Cn = noise_cov(m, E, mags)
    res = {}
    pmax = max(orders_siso + orders_mimo)
    for amodel, Paa in (('eb', alias_eb(E)), ('measured', alias_measured(E))):
        Gm = lag_cov(Pmm, range(pmax + 1))
        Ga = lag_cov(Paa, range(pmax + 1))
        for mg in mags:
            t0 = time.time()
            tag = f'{amodel}_mag{mg:g}'
            sb, hold, Fb, iir2 = siso_bounds(Pmm, Paa, Cn[mg], var0)
            F0 = np.zeros_like(Pmm)
            F0[:, np.arange(NM), np.arange(NM)] = Fb
            Em, info = mimo_bound(Pmm, Paa, Cn[mg], F0)
            del F0
            res[tag + '_siso'], res[tag + '_iir2'], res[tag + '_mimo'], res['hold'] = sb - hold, iir2 - hold, np.diag(Em), hold
            rss = lambda k: np.sqrt(res[k].sum())
            print(f'{tag}: hold {np.sqrt(hold.sum()):.2f} nm; prediction error rss [nm]: frequency domain SISO {rss(tag + "_siso"):.2f} '
                  f'(IIR2 {rss(tag + "_iir2"):.2f}), MIMO {rss(tag + "_mimo"):.2f} (Wilson {info["wilson_iter"]} it, res '
                  f'{info["wilson_res"]:.1e}; Wiener-Hopf iterations {np.sqrt(info["err_iter"]).round(3)}) [{time.time() - t0:.0f} s]',
                  flush=True)
            for p in orders_siso:
                F, rho = kalman_siso(Gm, Ga, Cn[mg], p, min(p, 8))
                res[f'{tag}_var{p}_siso'] = np.diag(pred_err(F, Pmm, Paa, Cn[mg]))
                print(f'    VAR({p}) + Kalman SISO: {rss(f"{tag}_var{p}_siso"):.2f} (radius {rho:.6f}) [{time.time() - t0:.0f} s]', flush=True)
            for p in orders_mimo:
                F, Emod, rho = kalman_filter(Gm, Ga, Cn[mg], p, min(p, 8))
                res[f'{tag}_var{p}_mimo'] = np.diag(pred_err(F, Pmm, Paa, Cn[mg]))
                res[f'{tag}_var{p}_mimo_model'] = np.diag(Emod)
                print(f'    VAR({p}) + Kalman MIMO: {rss(f"{tag}_var{p}_mimo"):.2f} (model {np.sqrt(max(np.trace(Emod), 0)):.2f}, radius '
                      f'{rho:.6f}) [{time.time() - t0:.0f} s]', flush=True)
            del F
            np.savez(os.path.join(OUT, f'bounds_N{NM}_M{M}.npz'), **res)
    return res


SLOPES = '/raid1/mmenessini/results/WFSW/alias_psd_slopes/modes/s0_rot0_seed{s}.npz'
SLOPE_SEEDS = tuple(int(v) for v in os.environ.get('AOEB_MIMO_SLOPE_SEEDS', '3001,3101,3201,3301').split(','))


def welch(series, Mf=M, L=1024):
    """Cross-spectral matrices (Phi convention) of zero-mean series [(T, n), ...] (Welch, Hann, L frames, 50 % overlap), linearly
    interpolated onto the Mf grid in FFT order (every matrix stays positive semi-definite)."""
    win = np.hanning(L)
    n_ = series[0].shape[1]
    acc, cnt = np.zeros((L, n_, n_), complex), 0
    for a in series:
        a = a - a.mean(0)
        for k0 in range(0, len(a) - L + 1, L // 2):
            X = np.fft.fft(a[k0:k0 + L] * win[:, None], axis=0)
            acc += par(lambda x: x[:, :, None] * np.conj(x[:, None, :]), X)
            cnt += 1
    P = acc / cnt / (win ** 2).sum()
    nuL, nuM = np.fft.fftfreq(L, T), np.fft.fftfreq(Mf, T)
    o = np.argsort(nuL)
    x = np.r_[nuL[o], nuL[o][0] + FS]
    y = np.concatenate([P[o], P[o][:1]])
    pos = np.interp(nuM, x, np.arange(len(x)))
    i0 = np.minimum(np.floor(pos).astype(int), len(x) - 2)
    t = (pos - i0)[:, None, None]
    return (1 - t) * y[i0] + t * y[i0 + 1]


def wiener_general(Pmm, Psm, Pyy, F0, n_iter=3):
    """Causal Wiener filter u = F y (F: (Mf, N, Ny)) for the target m_{k+2} (spectra Pmm (Mf, N, N)) from a measurement of any
    dimension Ny (Pyy (Mf, Ny, Ny), cross-spectrum Psm (Mf, N, Ny)), iterated from F0 as mimo_bound; returns the prediction-error
    matrix [nm^2] and diagnostics."""
    D = np.sqrt(np.real(np.diag(Pyy.mean(0))))
    G, it, res = wilson(Pyy / D[None, :, None] / D[None, None, :])
    G *= D[None, :, None]
    Gi = par(np.linalg.inv, G)
    del G
    F, errs = F0.copy(), []
    for _ in range(n_iter):
        X = causal(par(lambda s_, f_, y_, g_: (s_ - f_ @ y_) @ _H(g_), Psm, F, Pyy, Gi))
        F = F + par(np.matmul, X, Gi)
        E = np.real(par(lambda f_, m_, s_, y_: m_ - s_ @ _H(f_) - f_ @ _H(s_) + f_ @ y_ @ _H(f_), F, Pmm, Psm, Pyy).mean(0))
        errs.append(np.trace(E))
    return E, dict(wilson_iter=it, wilson_res=res, err_iter=errs)


def slopes_bound(mags=MAGS, K=100, to_eb=True, train=None, test=None, recons=('dl', 'pc')):
    """All-modes Wiener bound of modes 1..NM with the slope information: measurement = [modal POL of modes 1..NM; K combinations of the
    slope components outside the range of the interaction matrix (no controlled turbulence there: aliasing + noise only)], the K
    combinations = the best zero-lag linear predictors of the modal aliasing at that magnitude (aliasing slopes of the new rot-0 runs +
    EB slope noise). Compared, on the same aliasing data, with the per-mode bound and the modal all-modes bound. to_eb: modal aliasing
    rescaled per mode to the EB variance (as alias_measured; the complement block is left as simulated, correlations kept).
    train / test: seeds on which the K combinations are chosen / the spectra are estimated (default: first / second half of
    SLOPE_SEEDS; the same seeds would overfit: 2300 regressors). recons: modal measurement through 'dl', the EB SVD reconstructor (from
    the diffraction-limited IM) / og, which lets ~20 % of the modal aliasing in from the complement of range(IM_pc); or 'pc', the
    pseudo-inverse of IM_pc (rejects the complement exactly: any gain left from the slopes is information, not a better static
    reconstructor). Saved to OUT/slopes_N{NM}_K{K}_M{M}.npz."""
    import time
    from aoeb import utils
    from aoeb.naming import snap_seeing
    from aoeb.noise import slope_noise_cov
    import yaml
    from specula.lib.n_phot import n_phot
    m = A.machine()
    E = eb_inputs(m, mags)
    Pmm = turb_spectra(m)
    var0 = E['turb']
    rM, ns, N, wl = 0.0, A.NSUB, A.N, A.WL
    P = m.paths.at(wl)
    IM = np.asarray(utils.read_fits(P.im_pc(rM, ns, A.SEEING, N)), float)                      # (n_slopes, N)
    R = np.asarray(m._read_rec(m._ensure_rec(rM, ns, N, ('svd',), snap_seeing(A.SEEING), wl, wl, A.SEEING)), float)[:NM]
    U, sv, Vt = np.linalg.svd(IM, full_matrices=True)
    Q = U[:, N:]                                                   # orthonormal basis of the complement of range(IM)
    Rpc = ((Vt.T / sv) @ U[:, :N].T)[:NM]                          # pinv(IM_pc), modes 1..NM
    og = E['og']
    train = train or SLOPE_SEEDS[:max(1, len(SLOPE_SEEDS) // 2)]
    test = test or (SLOPE_SEEDS[len(SLOPE_SEEDS) // 2:] if len(SLOPE_SEEDS) > 1 else SLOPE_SEEDS)
    print(f'combinations chosen on seeds {train}, spectra estimated on seeds {test}', flush=True)
    load = lambda ss: [np.load(SLOPES.format(s=s))['slopes'].astype(float) for s in ss]
    runs_tr, runs = load(train), load(test)
    ua_tr, ua = [a @ Q for a in runs_tr], [a @ Q for a in runs]   # complement components of the aliasing
    Tn = sum(len(a) for a in runs_tr)
    Cuu = sum(u.T @ u for u in ua_tr) / Tn
    al_dl = [a @ R.T / og for a in runs_tr + runs]
    Caa = sum(al.T @ al for al in al_dl) / sum(len(a) for a in al_dl)
    c = np.sqrt(E['alias'] / np.diag(Caa)) if to_eb else np.ones(NM)   # sim -> EB per mode, from the EB reconstructor; used for both
    print(f'simulated modal aliasing / EB variance: median {np.median(1 / c ** 2):.3f}', flush=True)
    del al_dl
    Rr = {'dl': (R, og), 'pc': (Rpc, np.ones(NM))}
    frame = utils.read_fits(P.avgframe(rM, ns, A.SEEING, N))
    geom = m.slope_geometry(rM, ns, A.SEEING, N, wl, frame)
    Y = yaml.safe_load(open(os.path.join(A.CONFIG, 'aoeb', 'e2e', 'ristretto_main.yml')))
    area = (m.pupil() > 0).sum() * Y['main']['pixel_pitch'] ** 2
    rms = m.turbulence(A.SEEING, N)[1][:NM]
    nv = IM.shape[0] // 2
    out = {}
    path = os.path.join(OUT, f'slopes_N{NM}_K{K}_M{M}.npz')
    for mg in mags:
        flux = n_phot(mg, lambda_=wl * 1e-9, width=1e-9)[0] * Y['cred_vis']['bandw'] * area
        cn, cxy = slope_noise_cov(frame, geom, m.detector.n_det(flux, FS), m.detector)
        Sig = lambda X: cn[:, None] * X + np.r_[cxy, cxy][:, None] * np.r_[X[nv:], X[:nv]]   # Sigma_s @ X (sx-sy 2x2 blocks)
        QSQ = Q.T @ Sig(Q)
        for rname in recons:
            t0 = time.time()
            Rm, ogr = Rr[rname]
            alpha_tr = [a @ Rm.T / ogr for a in runs_tr]
            alpha = [a @ Rm.T / ogr for a in runs]                 # modal aliasing (POL units)
            Cua = sum(u.T @ al for u, al in zip(ua_tr, alpha_tr)) / Tn
            Tm = Rm * (rms / ogr)[:, None]                         # (NM, n_slopes): slopes -> modal POL [nm]
            W = np.linalg.solve(Cuu + QSQ, Cua * c[None, :])[:, :K]
            Za = [np.hstack([al * c[None, :], u @ W]) for al, u in zip(alpha, ua)]
            Cz = sum(z.T @ z for z in Za) / sum(len(z) for z in Za)
            Cz[NM:, NM:] += W.T @ QSQ @ W
            expl = np.diag(Cz[:NM, NM:] @ np.linalg.solve(Cz[NM:, NM:], Cz[NM:, :NM])) / np.diag(Cz[:NM, :NM])
            Tz = np.vstack([Tm, (Q @ W).T])                        # (NM + K, n_slopes)
            Cn_z = Tz @ Sig(Tz.T)
            Paz = welch(Za)
            Nz = NM + K
            Pyy = Paz.copy()
            Pyy[:, :NM, :NM] += Pmm
            Pyy += Cn_z[None]
            nu = np.fft.fftfreq(M, T)
            Psm = np.zeros((M, NM, Nz), complex)
            Psm[:, :, :NM] = Pmm * np.exp(2j * np.pi * nu * 2 * T)[:, None, None]
            Paa = Paz[:, :NM, :NM]
            sb, hold, Fb, _ = siso_bounds(Pmm, Paa, Cn_z[:NM, :NM], var0)
            F0 = np.zeros((M, NM, NM), complex)
            F0[:, np.arange(NM), np.arange(NM)] = Fb
            Em, _ = mimo_bound(Pmm, Paa, Cn_z[:NM, :NM], F0)
            F0z = np.zeros((M, NM, Nz), complex)
            F0z[:, :, :NM] = F0
            del F0
            Ez, info = wiener_general(Pmm, Psm, Pyy, F0z)
            del F0z, Pyy, Psm, Paz
            tag = f'{rname}_mag{mg:g}'
            out[f'{tag}_siso'], out[f'{tag}_mimo'], out[f'{tag}_slopes'], out['hold'] = sb - hold, np.diag(Em), np.diag(Ez), hold
            out[f'{tag}_alias_explained'] = expl
            np.savez(path, **out)
            r = lambda k: np.sqrt(out[f'{tag}_{k}'].sum())
            print(f'mag {mg:g}, reconstructor {rname}: modal aliasing {np.sqrt(np.trace(Cz[:NM, :NM])):.2f} nm rss, same-frame fraction '
                  f'predicted from the slope components median {np.median(expl):.2f}; prediction error rss [nm] (hold '
                  f'{np.sqrt(hold.sum()):.2f}): per mode {r("siso"):.2f}, all modes {r("mimo"):.2f}, all modes + {K} slope components '
                  f'{r("slopes"):.2f} (Wilson {info["wilson_iter"]} it, res {info["wilson_res"]:.1e}) [{time.time() - t0:.0f} s]', flush=True)
    return out


def var_only(orders=(2,), mags=MAGS):
    """VAR(p) + Kalman (SISO and MIMO, aliasing VAR(p)) only, added to OUT/bounds_N{NM}_M{M}.npz."""
    m = A.machine()
    E = eb_inputs(m, mags)
    Pmm = turb_spectra(m)
    Cn = noise_cov(m, E, mags)
    path = os.path.join(OUT, f'bounds_N{NM}_M{M}.npz')
    res = dict(np.load(path)) if os.path.exists(path) else {}
    for amodel, Paa in (('eb', alias_eb(E)), ('measured', alias_measured(E))):
        Gm, Ga = lag_cov(Pmm, range(max(orders) + 1)), lag_cov(Paa, range(max(orders) + 1))
        for mg in mags:
            tag = f'{amodel}_mag{mg:g}'
            for p in orders:
                Fs, rs = kalman_siso(Gm, Ga, Cn[mg], p, p)
                res[f'{tag}_var{p}_siso'] = np.diag(pred_err(Fs, Pmm, Paa, Cn[mg]))
                F, Emod, rho = kalman_filter(Gm, Ga, Cn[mg], p, p)
                res[f'{tag}_var{p}_mimo'] = np.diag(pred_err(F, Pmm, Paa, Cn[mg]))
                res[f'{tag}_var{p}_mimo_model'] = np.diag(Emod)
                rss = lambda k: np.sqrt(res[k].sum())
                print(f'{tag} VAR({p}) + Kalman: SISO {rss(f"{tag}_var{p}_siso"):.2f} (radius {rs:.6f}), MIMO '
                      f'{rss(f"{tag}_var{p}_mimo"):.2f} (radius {rho:.6f})', flush=True)
            np.savez(path, **res)
    return res


if __name__ == '__main__':
    if sys.argv[1:2] == ['check']:
        check()
    elif sys.argv[1:2] == ['bounds']:
        bounds()
    elif sys.argv[1:2] == ['slopes']:         # python mimo_spectral.py slopes [K]: bound with the slope information (new rot-0 runs)
        slopes_bound(K=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif sys.argv[1:2] == ['var']:            # python mimo_spectral.py var 2 [4 ...]: VAR-Kalman orders only, added to the results
        var_only(tuple(int(v) for v in sys.argv[2:]) or (2,))
