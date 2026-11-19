"""
Regression laws for the leaky-integrator and IIR2 coefficients (aoeb_order.py coefficient caches, rMod 0, 2 kHz) as functions of star
magnitude, seeing, wind speed (all layers scaled) and delay, neglecting the wind-profile variations.

Collapse variable: x = log10(P), P = S_turb(f*) / S_noise, the one-sided turbulence PSD of the mode at f* = F_REF over the noise PSD
(2 sigma_n^2 / fs). For frozen flow the high-frequency tail is S ~ sigma^2 V^(14/3) f^(-17/3), so P carries magnitude (noise), seeing
(turbulence and og), wind speed and the mode (variance, radial order) in one number; the delay enters separately.

    python aoeb/opt_iir/coeff_laws.py            # fits, table of laws, penalty of the regressed designs
    python aoeb/opt_iir/coeff_laws.py fit        # fits and table only (no closed-loop penalty runs)
    python aoeb/opt_iir/coeff_laws.py summary    # penalty table from the saved runs (coeff_laws_penalty_rmod0.npz)
"""
import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
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
# and modelled as  v(j, cond) = v_ref(j) + sum_p e_p(j) X_p,  X = powers 1..order of (dm, log10 k, log10 (s / 0.9), log10 (t / 2)),
# dm = mag - 8, k = wind speed / ristretto_main, s = seeing ["], t = total delay [frames]; e_p(j) fitted per mode, robust spline in j above J_RM
# above J_RM (2026-11-17; before: cubic in log10 j above J_LOW = 30, whose IIR2 pole curvature terms swung wildly).
# Orders per variable (ORDERS), chosen 2026-11-17 by leave-one-condition-out error on each one-parameter sweep: the curvature in log V and log s
# helps every variable (lG wind 0.013 -> 0.002 dex, seeing 0.006 -> 0.0015; 1-leak wind 0.038 -> 0.014) except 1-leak vs seeing (kept linear);
# dm^3 helps lG and lb0 (0.067 -> 0.044, 0.085 -> 0.048) but hurts lmb1/l1ma2/lA1; plain polynomials in V, s, t are always worse than these power
# laws. Delay has 3 values only: the quadratic in log t for G and leak interpolates them exactly (fixes ~5 % low-order misses in-sample) and
# cannot be cross-validated, so keep to 1.5-2.5 frames.
# v_ref(j): the reference design (mag 8, ristretto_main winds, 0.9", delay 2). b2 = F(1) A(1) - b0 - b1, a1 = A(1) - 1 - a2.
VARS = ['lG', 'l1mleak', 'lb0', 'lmb1', 'l1ma2', 'lA1', 'F1']
TERMS = ['dm', 'dm2', 'lk', 'ls', 'lt']    # legacy term set (pickles without per-variable terms)
_IIR = dict(dm=2, lk=2, ls=2, lt=1)
ORDERS = dict(lG=dict(dm=3, lk=2, ls=2, lt=2), l1mleak=dict(dm=2, lk=2, ls=1, lt=2),
              lb0=dict(dm=3, lk=2, ls=2, lt=1), lmb1=_IIR, l1ma2=_IIR, lA1=_IIR, F1=dict(dm=3, lk=2, ls=2, lt=1))


def terms_of(orders):
    """Term names ('dm', 'dm2', 'dm3', 'lk', 'lk2', ...) of an order dict."""
    return [b + (str(q) if q > 1 else '') for b in ('dm', 'lk', 'ls', 'lt') for q in range(1, orders[b] + 1)]
J_LOW = 30                                # low-order block reported separately in the penalty summary
J_RM, KNOT = 200, 100                     # every mode fitted on its own up to J_RM; above, reference curve + smooth shape-constrained scaling
SPIKE, W_SPIKE = 4.685, 101               # reference-curve outliers: |r - local median| > SPIKE local robust sigmas (r = ref - smooth,
                                          # windows of W_SPIKE modes: the scatter grows towards the cut-off); whole design replaced
IIR_VARS = ('lb0', 'lmb1', 'l1ma2', 'lA1', 'F1', 'b2', 'a1')
# Above J_RM every law is one smooth joint law in j (shape_fit). Monotone + single curvature is imposed on the variables of the plotted
# coefficients (SHAPE_VARS); log A(1) = log (1 + a1 + a2) and the DC gain F(1) are only smooth (A(1) is not monotone in j), but fitted
# directly: fitting a1 and b2 instead (variables b2, a1, kept for tests) gave pretty curves but A(1) ~ 0.05-0.1 errors of 10-30 % ->
# IIR2 penalty 1.2-3.8x in total and up to 350x on single modes (2026-11-17).
VARS_LO = ['lG', 'l1mleak', 'lb0', 'lmb1', 'l1ma2', 'lA1', 'F1']
VARS_HI = ['lG', 'l1mleak', 'lb0', 'l1ma2', 'lA1', 'F1', 'b2']   # b1 = F(1) A(1) - b0 - b2, a1 = A(1) - 1 - a2
# (fitting -b1 instead lost the DC gain: b0 and b1 nearly cancel, a 13 % b1 bias halved F(1) -> 100x penalty on single modes)
SHAPE_VARS = ('lG', 'l1mleak', 'lb0', 'lmb1', 'l1ma2', 'b2', 'a1')
ORDERS.update(b2=_IIR, a1=_IIR)
# condition grid on which the high-order law must be monotone in j with a single curvature (covers the fitted ranges)
SHAPE_GRID = [(m, k, s, t) for m in (5, 6.5, 8, 9.5, 11) for k in (0.5, 0.7, 1.0, 1.4, 2.0) for s in (0.6, 0.9, 1.05) for t in (1.5, 2.5)]


def to_vars(G, leak, b0, b1, b2, a1, a2):
    A1 = 1 + a1 + a2
    return dict(lG=np.log10(G), l1mleak=np.log10(np.maximum(1 - leak, 1e-9)), lb0=np.log10(b0), lmb1=np.log10(-b1), l1ma2=np.log10(1 - a2),
                lA1=np.log10(A1), F1=(b0 + b1 + b2) / A1, b2=np.asarray(b2, float), a1=np.asarray(a1, float))


def from_vars(v):
    a2 = 1 - 10 ** v['l1ma2']
    if 'b2' in v:                             # high-order parametrisation (VARS_HI): b2 direct, b1 from the DC gain, a1 from A(1)
        A1 = 10 ** v['lA1']
        b0 = 10 ** v['lb0']
        return dict(G=10 ** v['lG'], leak=1 - 10 ** v['l1mleak'], b=[b0, v['F1'] * A1 - b0 - v['b2'], v['b2']], a=[1.0, A1 - 1 - a2, a2])
    A1 = 10 ** v['lA1']
    a1 = A1 - 1 - a2
    b0, b1 = 10 ** v['lb0'], -10 ** v['lmb1']
    return dict(G=10 ** v['lG'], leak=1 - 10 ** v['l1mleak'], b=[b0, b1, v['F1'] * A1 - b0 - b1], a=[1.0, a1, a2])


def regressors(mag, k, s, t, terms=TERMS):
    base = dict(dm=np.asarray(mag, float) - 8.0, lk=np.log10(np.asarray(k, float)), ls=np.log10(np.asarray(s, float) / 0.9),
                lt=np.log10(np.asarray(t, float) / 2.0))
    return np.stack([base[n[:2]] ** (int(n[2:]) if len(n) > 2 else 1) for n in terms], -1)


def branch_switch(d, j_min=None, win=21, rise=0.02):
    """Per design point, True where the optimal IIR2 design has jumped to the second branch: from the first mode >= j_min where the running
    median (win modes) of a2 rises more than `rise` above its running minimum, for that condition (a2 decreases with j on the main branch;
    the second branch jumps up and rises: mag 11 from j ~ 425, wind x2 / x1.5 from ~ 1020 / 1116, seeing 0.6" / 0.75" from ~ 1141 / 1279)."""
    from scipy.ndimage import median_filter
    j_min = J_RM if j_min is None else j_min
    C = np.stack([d['mag'], d['k'], d['s'], d['t']], 1)
    out = np.zeros(len(C), bool)
    for c in np.unique(C, axis=0):
        idx = np.nonzero(np.all(np.isclose(C, c), 1))[0]
        idx = idx[np.argsort(d['j'][idx])]
        j = d['j'][idx]
        sm = median_filter(d['a2'][idx], win, mode='nearest')
        run = np.minimum.accumulate(np.where(j >= j_min, sm, np.inf))
        up = (j >= j_min) & (sm - run > rise)
        if up.any():
            out[idx[j >= j[up][0]]] = True
    return out


def shape_basis(j, j0, j1, knot):
    """Basis of v(u) = a + b u + sum_k h_k H_k(u), u = (j - j0) / (j1 - j0) in [0, 1]: H_k = double integral of the hat function at knot k
    (knots every `knot` modes), so v'' = sum h_k hat_k (v''(t_k) = h_k) and v'(1) = b + sum h_k Phi_k(1). Returns (B (len(j), 2 + K),
    Phi_k(1) (K,))."""
    n = int(round((j1 - j0) / knot))
    t = np.linspace(0, 1, n + 1)
    g = np.arange(j0, j1 + 1, dtype=float)
    u = (g - j0) / (j1 - j0)
    du = 1.0 / (j1 - j0)
    hat = np.maximum(0, 1 - np.abs(u[:, None] - t[None, :]) * n)
    Phi = np.vstack([np.zeros(len(t)), np.cumsum((hat[1:] + hat[:-1]) / 2, 0) * du])
    H = np.vstack([np.zeros(len(t)), np.cumsum((Phi[1:] + Phi[:-1]) / 2, 0) * du])
    i = np.asarray(j, int) - int(j0)
    return np.hstack([np.ones((len(i), 1)), u[i][:, None], H[i]]), Phi[-1]


def shape_fit(J, X, y, w, j0, j1, knot, terms, n_iter=4, constrained=True):
    """Joint law of y on modes j0..j1: y = sum_p x_p (B(j) . theta_p) (x_0 = 1, then the condition terms), every coefficient curve of the
    spline basis shape_basis, so the law is a smooth curve in j at any condition. Constrained to be monotone in j with a single curvature
    on SHAPE_GRID; slope and curvature signs from the unconstrained reference curve. Tukey-reweighted (scale from the MAD), w = 0 drops
    points; constrained=False: same smooth basis, no shape constraint. Returns theta (1 + len(terms), 2 + K)."""
    from scipy.optimize import nnls
    B, Phi1 = shape_basis(J, j0, j1, knot)
    nb = B.shape[1]
    Xa = np.hstack([np.ones((len(J), 1)), X])
    A_ = (Xa[:, :, None] * B[:, None, :]).reshape(len(J), -1)
    G = np.hstack([np.ones((len(SHAPE_GRID), 1)), regressors(*np.array(SHAPE_GRID).T, terms)])
    # sign of slope / curvature: unconstrained fit, reference condition
    th0 = np.linalg.lstsq(A_[w > 0], y[w > 0], rcond=None)[0].reshape(Xa.shape[1], nb)
    sb = np.sign(th0[0, 1] + th0[0, 2:] @ Phi1 / 2) or 1.0
    sc = np.sign(np.sum(th0[0, 2:])) or 1.0
    # constraints C theta >= 0: sc * v''(t_k) (all k), sb * v'(j0), sb * v'(j1), on every grid condition
    rows = []
    for g in G:
        for k in range(nb - 2):
            r = np.zeros((len(g), nb)); r[:, 2 + k] = sc * g; rows.append(r.ravel())
        r = np.zeros((len(g), nb)); r[:, 1] = sb * g; rows.append(r.ravel())
        r = np.zeros((len(g), nb)); r[:, 1] = sb * g; r[:, 2:] = sb * g[:, None] * Phi1[None, :]; rows.append(r.ravel())
    Cm = np.array(rows)
    ww = w.astype(float).copy()
    for _ in range(n_iter):
        # whitened problem: Q = R^T R, phi = R theta -> min |phi - R^-T c|^2 s.t. Cm R^-1 phi >= 0
        Q = A_.T @ (A_ * ww[:, None])
        Q += 1e-10 * np.trace(Q) / len(Q) * np.eye(len(Q))
        R = np.linalg.cholesky(Q).T
        Ri = np.linalg.inv(R)
        z = np.linalg.solve(R.T, A_.T @ (ww * y))
        if not constrained:
            th = Ri @ z
        else:
            # projection of z on the cone {Cw f >= 0}: dual NNLS, f = z + Cw^T lam, lam = argmin_{lam >= 0} |Cw^T lam + z|
            Cw = Cm @ Ri
            th = Ri @ (z + Cw.T @ nnls(Cw.T, -z, maxiter=50 * Cw.shape[0])[0])
        r = A_ @ th - y
        k = w > 0
        s_ = 1.4826 * np.median(np.abs(r[k] - np.median(r[k]))) + 1e-12
        u = r / (4.685 * s_)
        ww = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0) * (w > 0)
    return th.reshape(Xa.shape[1], nb)


def fit_laws(d, j_rm=None, knot=None):
    """Least squares of every variable: {var: (v_ref per mode (N + 1,), None, fit rms, exponents per mode (N + 1, len(terms)), terms)}.
    Every mode is fitted on its own (the old cubic law in log10 j could not follow the IIR2 pole variables and jumped at its first mode);
    above j_rm ONE joint law in j (shape_fit: smooth, monotone in j with a single curvature at every condition of the fitted ranges, IIR2
    designs on the optimiser's second branch (branch_switch) left out) gives the scaling only: law(j, c) = ref(j) + [smooth(j, c) -
    smooth(j, ref)], i.e. the exponents are the smooth ones and v_ref keeps the optimal reference curve with its mode-to-mode shape,
    despiked: a mode whose reference design is an outlier in any variable of its filter (|ref - smooth| > SPIKE robust sigmas) takes the
    smooth design for all of them (keeps each design self-consistent). Below j_rm everything is per mode, v_ref exact.
    Variables VARS_LO below j_rm, VARS_HI above (nan where a variable is not used)."""
    j_rm = J_RM if j_rm is None else j_rm
    knot = KNOT if knot is None else knot
    second = branch_switch(d, j_rm)
    J = d['j'].astype(int)
    V = to_vars(d['G'], d['leak'], d['b0'], d['b1'], d['b2'], d['a1'], d['a2'])
    ref = np.isclose(d['mag'], 8) & np.isclose(d['k'], 1) & np.isclose(d['s'], 0.9) & np.isclose(d['t'], 2)
    laws, raw, smooth = {}, {}, {}
    hi = np.arange(j_rm + 1, A.N + 1)
    for name in dict.fromkeys(VARS_LO + VARS_HI):
        terms = terms_of(ORDERS[name])
        X = regressors(d['mag'], d['k'], d['s'], d['t'], terms)
        vref = np.full(A.N + 1, np.nan)
        vref[J[ref]] = V[name][ref]
        dv = V[name] - vref[J]
        e_pm = np.full((A.N + 1, len(terms)), np.nan)
        for j in range(1, A.N + 1):
            m = ~ref & (J == j)
            e_pm[j] = np.linalg.lstsq(X[m], dv[m], rcond=None)[0]
        e_mode, vs = e_pm.copy(), vref.copy()
        if name not in VARS_LO:
            e_mode[:j_rm + 1], vs[:j_rm + 1] = np.nan, np.nan
        if name not in VARS_HI:
            e_mode[j_rm + 1:], vs[j_rm + 1:] = np.nan, np.nan
            laws[name] = (vs, None, np.nan, e_mode, terms)
            continue
        h = J >= j_rm
        w = (~second[h] if name in IIR_VARS else np.ones(h.sum(), bool)).astype(float)
        th = shape_fit(J[h], X[h], V[name][h], w, j_rm, A.N, knot, terms, constrained=name in SHAPE_VARS)
        Bh = shape_basis(hi, j_rm, A.N, knot)[0]
        raw[name], smooth[name] = vref[hi], Bh @ th[0]
        e_mode[hi] = Bh @ th[1:].T
        laws[name] = (vs, None, np.nan, e_mode, terms)
    # reference curve above j_rm, despiked design by design (leaky and IIR2 separately)
    for group in ([n for n in VARS_HI if n not in IIR_VARS], [n for n in VARS_HI if n in IIR_VARS]):
        from scipy.ndimage import median_filter
        spike = np.zeros(len(hi), bool)
        for n in group:
            r = raw[n] - smooth[n]
            dev = np.abs(r - median_filter(r, W_SPIKE, mode='mirror'))
            spike |= dev > SPIKE * 1.4826 * median_filter(dev, W_SPIKE, mode='mirror')
        for n in group:
            laws[n][0][hi] = np.where(spike, smooth[n], raw[n])
        print(f'reference spikes replaced above j = {j_rm}, {"IIR2" if group[0] in IIR_VARS else "leaky"}: {int(spike.sum())} modes'
              f' {hi[spike][:12].tolist()}{" ..." if spike.sum() > 12 else ""}')
    for name, (vs, _, _, e_mode, terms) in laws.items():
        X = regressors(d['mag'], d['k'], d['s'], d['t'], terms)
        r = vs[J] + np.einsum('ip,ip->i', X, e_mode[J]) - V[name]
        rms = np.sqrt(np.nanmean(r[~(second & np.isin(name, IIR_VARS))] ** 2))   # second-branch IIR2 designs excluded
        laws[name] = (vs, None, rms, e_mode, terms)
    return laws


def exponents(law, j):
    """Exponents (one per term of the law) of mode j: per-mode table (law[3]) where finite, else the smooth law in log10 j (law[1])."""
    e, e_low = law[1], (law[3] if len(law) > 3 else None)
    if e is None or (e_low is not None and j < len(e_low) and np.all(np.isfinite(e_low[j]))):
        return e_low[j]
    return e @ (np.log10(j) ** np.arange(e.shape[1]))


def predict(laws, j, mag, k, s, t):
    """Coefficients of mode j (from 1) at the condition, from the laws (VARS_LO / VARS_HI below / above J_RM; legacy pickles: all)."""
    names = [n for n in laws if np.isfinite(laws[n][0][j])]
    return from_vars({n: laws[n][0][j] + regressors(mag, k, s, t, laws[n][4] if len(laws[n]) > 4 else TERMS) @ exponents(laws[n], j)
                      for n in names})


def law_table(laws, js=(1, 2, 3, 4, 5, 6, 10, 30, 100, 300, 1000)):
    """One markdown table per variable: coefficient of every term (dex per unit of dm^q, log10 k^q, ...) at the modes js."""
    out = []
    for n, law in laws.items():
        terms = law[4] if len(law) > 4 else TERMS
        out += [f'{n} (fit rms {law[2]:.3f} dex)', '| mode | ' + ' | '.join(terms) + ' |', '|---' * (len(terms) + 1) + '|']
        out += [f'| {j} | ' + ' | '.join(f'{c:+.4f}' for c in exponents(law, j)) + ' |' for j in js if np.isfinite(law[0][j])]
        out.append('')
    return '\n'.join(out)


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
    """Residual of the law designs against the optimal ones, TESTS conditions (off the fitted grid), every mode."""
    modes = modes or A.MODES
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


PENALTY = os.path.join(HERE, 'coeff_laws_penalty_rmod0.npz')


def penalty_summary(tags, R, j_low=None):
    """Markdown table of the law designs against the optimal ones per TESTS condition and filter. R columns: optimal leaky, optimal IIR2,
    law leaky, law IIR2 residual variances [nm^2] (inf if unstable). Total = rss over all modes (what the error budget sees); per-mode
    ratios of the variances: median, 99th percentile, max (with its mode), max over the low orders j <= J_LOW."""
    j_low = J_LOW if j_low is None else j_low
    tags = np.asarray(tags, float)
    rows = ['| mag | wind | seeing | delay | filter | rss opt [nm] | rss law [nm] | rss law/opt | median var ratio | p99 | max (mode) | max j<=%d | unstable |'
            % j_low, '|---' * 13 + '|']
    for cond in TESTS:
        m = np.all(np.isclose(tags[:, :4], cond), 1)
        j = tags[m, 4].astype(int)
        for f, (io, il) in (('leaky', (0, 2)), ('IIR2', (1, 3))):
            opt, law = R[m, io], R[m, il]
            ok = np.isfinite(law)
            q = law / opt
            qk = q[ok]
            low = ok & (j <= j_low)
            imax = np.argmax(np.where(ok, q, -np.inf))
            rss_o, rss_l = np.sqrt(np.sum(opt)), np.sqrt(np.sum(law[ok]))
            rows.append(f'| {cond[0]:g} | x{cond[1]:g} | {cond[2]:g}" | {cond[3]:g} | {f} | {rss_o:.2f} | {rss_l:.2f}{"" if ok.all() else "*"} | '
                        f'{rss_l / rss_o:.4f} | {np.median(qk):.3f} | {np.percentile(qk, 99):.3f} | {q[imax]:.2f} ({j[imax]}) | '
                        f'{np.max(q[low]) if low.any() else np.nan:.3f} | {int(np.sum(~ok))} |')
    rows.append('\n(* rss over the stable modes only; ratios are per-mode residual variances law/optimal)')
    return '\n'.join(rows)


if __name__ == '__main__':
    if 'summary' in sys.argv[1:]:             # summary of the saved penalty runs only
        P = np.load(PENALTY)
        print(penalty_summary(P['tags'], P['R']))
        sys.exit()
    d = points()
    laws = fit_laws(d)
    print(law_table(laws))
    import pickle
    pickle.dump(laws, open(os.path.join(HERE, 'coeff_laws_rmod0.pkl'), 'wb'))
    if 'fit' in sys.argv[1:]:                 # fits and table only, no closed-loop penalty runs
        sys.exit()
    tags, R = penalty(laws)
    np.savez(PENALTY, tags=np.array(tags), R=R)
    print(penalty_summary(tags, R))
