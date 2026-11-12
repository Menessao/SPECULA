"""
Regression laws for the leaky-integrator and IIR2 coefficients (aoeb_order.py coefficient caches, rMod 0, 2 kHz) as functions of star
magnitude, seeing, wind speed (all layers scaled) and delay, neglecting the wind-profile variations.

Collapse variable: x = log10(P), P = S_turb(f*) / S_noise, the one-sided turbulence PSD of the mode at f* = F_REF over the noise PSD
(2 sigma_n^2 / fs). For frozen flow the high-frequency tail is S ~ sigma^2 V^(14/3) f^(-17/3), so P carries magnitude (noise), seeing
(turbulence and og), wind speed and the mode (variance, radial order) in one number; the delay enters separately.

    python aoeb/opt_iir/coeff_laws.py            # collapse figure, fits, table of laws, penalty of the regressed designs
"""
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ['AOEB_ORDER_ALL'] = '1'
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aoeb_order as A                   # noqa: E402

F_REF = 50.0                              # [Hz] reference frequency of the PSD ratio (in the frozen-flow tail for every mode)
OUT = os.path.join(HERE, 'coeff_laws_rmod0.npz')
COEFS = ['G', 'leak', 'b0', 'b1', 'b2', 'a1', 'a2']


def points():
    """Every design point of the caches: dict of arrays (mode j, radial order n, delay t, mag, seeing s, wind factor k, x = log10 P,
    log10 alias/noise, the coefficients). Cached in OUT."""
    if os.path.exists(OUT):
        return dict(np.load(OUT))
    from aoeb import filter_optimizers as fo
    done = {**json.load(open(A.ENV_CACHE)), **json.load(open(A.COEF_CACHE))}
    rows = []
    conds = [(t, mg, A.SEEING, 1.0) for t, mg in A.COEF_CASES] + [(A.ENV_T, A.ENV_MAG, s, k) for s, k in A.ENV_CASES]
    js = np.arange(1, A.N + 1)
    for s in sorted({c[2] for c in conds}):
        mags = sorted({c[1] for c in conds if c[2] == s})
        I = A.inputs(s, 1.0, mags=mags)
        n = np.array([I['order'][j] for j in js])
        turb = np.array([I['turb'][j] for j in js])
        alias = np.array([I['alias'][j] for j in js])
        u = {nn: None for nn in set(n)}
        for t, mg, s2, k in conds:
            if s2 != s:
                continue
            for nn in u:
                u[nn] = float(fo.unit_psd(np.array([F_REF / k]), nn, I['wind'], I['D'])[0]) / k
            noise = np.array([I['noise'][mg][j] for j in js])
            P = turb * np.array([u[nn] for nn in n]) / (2 * noise / A.FS)
            for i, j in enumerate(js):
                c = done.get(A.ekey(t, mg, j, s, k))
                if c is None:
                    continue
                rows.append([j, n[i], t, mg, s, k, np.log10(P[i]), np.log10(alias[i] / noise[i]), c['G'], c['leak'], *c['b'], *c['a'][1:],
                             *c['res']])
    R = np.array(rows)
    d = dict(zip(['j', 'n', 't', 'mag', 's', 'k', 'x', 'xa'] + COEFS + ['res_leaky', 'res_iir2'], R.T))
    np.savez(OUT, **d)
    return d


def sweep_of(d):
    """Label of the sweep each point belongs to (for colours)."""
    lab = np.full(len(d['j']), 'mag', dtype=object)
    lab[d['t'] != 2.0] = 'delay'
    lab[d['s'] != A.SEEING] = 'seeing'
    lab[d['k'] != 1.0] = 'wind'
    return lab


# ---------------------------------------------------------------- separable power laws per mode
# Each coefficient is mapped to a variable that is positive and smooth for every design (complex or real poles):
#   leaky: log10 G, log10 (1 - leak);  IIR2: log10 b0, log10 (-b1), log10 (1 - a2), log10 A(1) = log10 (1 + a1 + a2), DC gain F(1)
# and modelled as  v(j, cond) = v_ref(j) + sum_p e_p(j) X_p,  X = (dm, dm^2, log10 k, log10 (s / 0.9), log10 (t / 2)),
# dm = mag - 8, k = wind speed / ristretto_main, s = seeing ["], t = total delay [frames]; e_p(j) = quadratic in log10 j.
# v_ref(j): the reference design (mag 8, ristretto_main winds, 0.9", delay 2). b2 = F(1) A(1) - b0 - b1, a1 = A(1) - 1 - a2.
VARS = ['lG', 'l1mleak', 'lb0', 'lmb1', 'l1ma2', 'lA1', 'F1']
TERMS = ['dm', 'dm2', 'lk', 'ls', 'lt']


def to_vars(G, leak, b0, b1, b2, a1, a2):
    A1 = 1 + a1 + a2
    return dict(lG=np.log10(G), l1mleak=np.log10(np.maximum(1 - leak, 1e-9)), lb0=np.log10(b0), lmb1=np.log10(-b1), l1ma2=np.log10(1 - a2),
                lA1=np.log10(A1), F1=(b0 + b1 + b2) / A1)


def from_vars(v):
    a2 = 1 - 10 ** v['l1ma2']
    A1 = 10 ** v['lA1']
    a1 = A1 - 1 - a2
    b0, b1 = 10 ** v['lb0'], -10 ** v['lmb1']
    return dict(G=10 ** v['lG'], leak=1 - 10 ** v['l1mleak'], b=[b0, b1, v['F1'] * A1 - b0 - b1], a=[1.0, a1, a2])


def regressors(mag, k, s, t):
    dm = np.asarray(mag, float) - 8.0
    return np.stack([dm, dm ** 2, np.log10(k), np.log10(np.asarray(s, float) / 0.9), np.log10(np.asarray(t, float) / 2.0)], -1)


def fit_laws(d, deg=2):
    """Least squares of every variable: {var: (v_ref per mode (N,), exponent coefficients (len(TERMS), deg + 1) in powers of log10 j)}."""
    J = d['j'].astype(int)
    V = to_vars(d['G'], d['leak'], d['b0'], d['b1'], d['b2'], d['a1'], d['a2'])
    X = regressors(d['mag'], d['k'], d['s'], d['t'])
    Lj = np.log10(J)
    ref = np.isclose(d['mag'], 8) & np.isclose(d['k'], 1) & np.isclose(d['s'], 0.9) & np.isclose(d['t'], 2)
    laws = {}
    for name in VARS:
        vref = np.full(A.N + 1, np.nan)
        vref[J[ref]] = V[name][ref]
        dv = V[name] - vref[J]
        M = np.concatenate([X[:, [p]] * Lj[:, None] ** np.arange(deg + 1) for p in range(len(TERMS))], 1)
        e = np.linalg.lstsq(M[~ref], dv[~ref], rcond=None)[0].reshape(len(TERMS), deg + 1)
        rms = np.sqrt(np.mean((M[~ref] @ e.ravel() - dv[~ref]) ** 2))
        laws[name] = (vref, e, rms)
    return laws


def predict(laws, j, mag, k, s, t):
    """Coefficients of mode j (from 1) at the condition, from the laws."""
    x = regressors(mag, k, s, t)
    Lj = np.log10(j) ** np.arange(next(iter(laws.values()))[1].shape[1])
    return from_vars({n: vref[j] + x @ (e @ Lj) for n, (vref, e, _) in laws.items()})


def law_table(laws, js=(1, 10, 100, 300, 1000)):
    rows = ['| variable | mode | per mag (dex) | per mag^2 | V exponent | seeing exponent | delay exponent | fit rms (dex) |', '|---|---|---|---|---|---|---|---|']
    for n, (vref, e, rms) in laws.items():
        for j in js:
            c = e @ (np.log10(j) ** np.arange(e.shape[1]))
            rows.append(f'| {n} | {j} | {c[0]:+.3f} | {c[1]:+.4f} | {c[2]:+.2f} | {c[3]:+.2f} | {c[4]:+.2f} | {rms:.3f} |')
    return '\n'.join(rows)


TESTS = [(9.5, 1.3, 0.8, 1.75), (6.5, 0.7, 1.0, 2.25), (10.0, 1.8, 0.65, 2.0), (7.0, 1.0, 0.9, 2.0)]   # (mag, wind factor, seeing, delay)


def penalty_task(args):
    """Optimal leaky / IIR2 residuals and those of the law coefficients (inf if unstable) of one mode and condition."""
    spec_args, c = args
    from ao_tools import closed_loop_residual
    spec = A.make_spec(*spec_args)
    opt = A.coef_task(spec_args)
    jl = closed_loop_residual(spec, [c['G']], [1.0, -c['leak']], 'residual')
    j2 = closed_loop_residual(spec, c['b'], c['a'], 'pol')
    return opt['res'][0], opt['res'][1], jl, j2


def penalty(laws, modes=None):
    """Residual of the law designs against the optimal ones, TESTS conditions (off the fitted grid), the plotted modes."""
    modes = modes or A.log_modes(A.N_PLOT, A.MODE_MAX)
    jobs, tags = [], []
    for mg, k, s, t in TESTS:
        I = A.inputs(s, k, mags=[mg])
        for j in modes:
            jobs.append(((t, I['order'][j], I['turb'][j], I['noise'][mg][j], I['alias'][j], I['wind'], I['D'], I['alias_spec']),
                         predict(laws, j, mg, k, s, t)))
            tags.append((mg, k, s, t, j))
    import multiprocessing as mp
    with mp.get_context('fork').Pool(A.N_JOBS) as pool:
        R = np.array(pool.map(penalty_task, jobs, chunksize=1))
    return tags, R


if __name__ == '__main__':
    d = points()
    laws = fit_laws(d)
    print(law_table(laws))
    import pickle
    pickle.dump(laws, open(os.path.join(HERE, 'coeff_laws_rmod0.pkl'), 'wb'))
    tags, R = penalty(laws)
    np.savez(os.path.join(HERE, 'coeff_laws_penalty_rmod0.npz'), tags=np.array(tags), R=R)
    for cond in TESTS:
        m = np.array([tg[:4] == cond for tg in tags])
        r = R[m]
        print(f'mag {cond[0]:g}, wind x{cond[1]:g}, seeing {cond[2]:g}", delay {cond[3]:g}: leaky law/opt residual per mode median '
              f'{np.median(r[:, 2] / r[:, 0]):.3f} max {np.max(r[:, 2] / r[:, 0]):.3f}; IIR2 law/opt median {np.median(r[:, 3] / r[:, 1]):.3f} '
              f'max {np.max(r[:, 3] / r[:, 1]):.3f}; unstable {int(np.sum(~np.isfinite(r[:, 2:])))}')
