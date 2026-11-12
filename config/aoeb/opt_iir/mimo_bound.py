"""
Linear prediction bounds from joint turbulence + aliasing time series: single mode (SISO, the mode's own measurement history) against all
modes (MIMO, every mode's history), on the same data (results/WFSW/alias_psd_mimo: rMod 0, 48x48, 900 nm, 1300 modes, 10-layer frozen
flow at 0.9", rotated winds, 2 kHz, 8 s per run, the true KL coefficients `atmo` and the pyramid-reconstructed aliasing `modes_900`).

Model per frame k (POL, optical gain compensated): y_k = x_k + a_k / og + n_k, with x the true modes [nm], a the measured aliasing, n white
Gaussian noise of the aoeb variance at the magnitude (photon + readout, measured propagation, / og^2; sampled per run). Target: the average over the hold of the command at total delay 1 + tau = 2 frames, s_k = (x_{k+1} + x_{k+2}) / 2
(samples at the frame rate: the intra-frame hold error and the WFS box average are not resolved, the same for both predictors).
Predictors, fitted on training runs and scored on test runs (other screens and wind directions; the honest error): SISO, the LMMSE
filter of each mode on the last L frames of its own measurement; MIMO, the SISO (L 64) prediction plus a ridge regression of its error
on the last L frame-to-frame changes of ALL modes' measurements (so it cannot do worse than SISO; ridge weight on a held-out training
run). Finite windows and finite data make both achievable filters, upper estimates of their bounds, converged in L. Errors are prediction errors E[(s - u)^2] per mode [nm^2] and their root sum of squares over the modes.

    python aoeb/opt_iir/mimo_bound.py [mag [L_mimo ...]]
"""
import glob
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '32')
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = '/raid1/mmenessini/results/WFSW/alias_psd_mimo/modes'
TEST = ('m0_rot90_seed2101', 'm0_rot135_seed2101')
N = 1300


def load(mag, seed=0):
    """Runs {name: (x, y)}: true modes x and the noisy POL measurement y = x + a / og + n (n Gaussian, white, the aoeb variance at the
    magnitude, sampled with a per-run seed), both [nm], means removed; and the per-mode noise variance."""
    sys.path.insert(0, HERE)
    os.environ['AOEB_ORDER_ALL'] = '1'
    import aoeb_order as A
    I = A.inputs(mags=[mag])
    og = np.array([I['og'][j] for j in range(1, N + 1)])
    noise = np.array([I['noise'][mag][j] for j in range(1, N + 1)])
    runs = {}
    for i, f in enumerate(sorted(glob.glob(os.path.join(DATA, 'm0_*.npz')))):
        d = np.load(f)
        x = d['atmo'].astype(np.float64)
        y = x + d['modes_900'].astype(np.float64) / og
        y += np.random.default_rng(seed + i).standard_normal(y.shape) * np.sqrt(noise)
        runs[os.path.basename(f)[:-4]] = (x - x.mean(0), y - y.mean(0))
    return runs, noise


def lagged(z, L, K):
    """Rows [z_k, z_{k-1}, ..., z_{k-L+1}] for k = K-1 .. T-3 (K >= L: common start; the target needs k + 2)."""
    T = len(z)
    return np.hstack([z[K - 1 - l:T - 2 - l] for l in range(L)])


def target(x, K):
    return 0.5 * (x[K:len(x) - 1] + x[K + 1:])


K = 64          # first usable frame (the longest window) so that every predictor is scored on the same frames


def siso(runs, L, train):
    """LMMSE filter per mode on the last L frames of its OWN measurement, fitted on `train`: {run: prediction (rows, N)}."""
    pred = {k: np.zeros((len(runs[k][0]) - K - 1, N)) for k in runs}
    for j in range(N):
        A = sum(lagged(runs[k][1][:, [j]], L, K).T @ lagged(runs[k][1][:, [j]], L, K) for k in train)
        b = sum(lagged(runs[k][1][:, [j]], L, K).T @ target(runs[k][0][:, j], K) for k in train)
        w = np.linalg.solve(A, b)
        for k in runs:
            pred[k][:, j] = lagged(runs[k][1][:, [j]], L, K) @ w
    return pred


def err(runs, pred, names):
    """Prediction error per mode [nm^2] over the runs `names`."""
    e = [((target(runs[k][0], K) - pred[k]) ** 2) for k in names]
    return np.concatenate(e).mean(0)


def mimo(runs, base, L, train, val, lams=10.0 ** np.arange(-3, 4)):
    """All-modes correction of the SISO prediction `base`: ridge regression of its error on the last L frame-to-frame changes of the
    measurements of ALL modes (columns scaled to unit variance), the ridge weight chosen on the run `val` (refit on all `train`).
    Returns {run: prediction} = base + correction, and the chosen weight."""
    def feats(k):
        dy = np.diff(runs[k][1], axis=0, prepend=runs[k][1][:1])
        return lagged(dy, L, K)
    sc = np.sqrt(np.concatenate([feats(k) for k in train[:2]]).var(0)) + 1e-12

    def fit(names):
        A = sum((feats(k) / sc).T @ (feats(k) / sc) for k in names)
        B = sum((feats(k) / sc).T @ (target(runs[k][0], K) - base[k]) for k in names)
        n = sum(len(base[k]) for k in names)
        return np.linalg.eigh(A / n), B / n
    (ev, V), B = fit([k for k in train if k != val])
    VB = V.T @ B
    scores = []
    for lam in lams:
        W = V @ (VB / (ev + lam)[:, None])
        r = target(runs[val][0], K) - base[val] - (feats(val) / sc) @ W
        scores.append((r ** 2).mean(0).sum())
    lam = lams[int(np.argmin(scores))]
    (ev, V), B = fit(train)
    W = V @ ((V.T @ B) / (ev + lam)[:, None])
    return {k: base[k] + (feats(k) / sc) @ W for k in runs}, lam, scores


if __name__ == '__main__':
    mag = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    Ls = [int(v) for v in sys.argv[2:]] or [1, 2, 4]
    runs, noise = load(mag)
    test = [k for k in runs if k in TEST]
    train = [k for k in runs if k not in TEST]
    print(f'mag {mag:g}: {len(train)} training runs, {len(test)} test runs, {sum(len(runs[k][0]) for k in train)} training frames', flush=True)
    res = {}
    for L in (1, 4, 16, 32, 64):
        p = siso(runs, L, train)
        res[f'siso_L{L}'] = err(runs, p, test)
        print(f'SISO L {L:2d}: test rss {np.sqrt(res[f"siso_L{L}"].sum()):.2f} nm', flush=True)
    base = p
    for L in Ls:
        pm, lam, sc = mimo(runs, base, L, train, train[-1])
        res[f'mimo_L{L}'] = err(runs, pm, test)
        print(f'MIMO L {L:2d} (+ SISO L 64, ridge {lam:g}, val scores {np.round(sc, 1)}): test rss {np.sqrt(res[f"mimo_L{L}"].sum()):.2f} nm', flush=True)
    np.savez(os.path.join(HERE, f'mimo_bound_rmod0_mag{mag:g}.npz'), noise=noise, **res)
