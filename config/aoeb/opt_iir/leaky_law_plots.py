"""
Optimal leaky-integrator coefficients (gain G, leak) per mode vs their regression laws (coeff_laws.py), one figure per swept parameter
(wind speed, seeing, magnitude, delay; the others at the reference mag 8, 0.9", nominal winds, delay 2); gain vs log mode, leak vs linear
mode. IIR2 (POL, C = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2)): one figure per coefficient, one panel per sweep, vs log mode.
Solid: optimal design (cached points, coeff_laws_rmod0.npz); dashed: law prediction (coeff_laws_rmod0.pkl). Nothing is recomputed.

    python aoeb/opt_iir/leaky_law_plots.py       # -> leaky_law_rmod0_{wind,seeing,mag,delay}.png, iir2_law_rmod0_{b0,b1,b2,a1,a2}.png
"""
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from coeff_laws import predict, terms_of, ORDERS, J_RM   # noqa: E402

REF = dict(mag=8.0, k=1.0, s=0.9, t=2.0)
TERM_TXT = dict(dm='Δm', dm2='Δm²', dm3='Δm³', lk='lV', lk2='lV²', ls='ls', ls2='ls²', lt='ld', lt2='ld²')


def law_txt(var, lhs):
    """'lhs = lhs_ref(j) + e1(j) x1 + ...' for a fitted variable, with its terms."""
    return f'{lhs}(j, c) = {lhs}_ref(j) + Σ e_p(j)·x_p,   x_p ∈ {{{", ".join(TERM_TXT[t] for t in terms_of(ORDERS[var]))}}}'


FOOT = [f'Δm = mag − 8,  lV = log10(V/V_nominal),  ls = log10(seeing/0.9"),  ld = log10(delay/2 frames);  ref = optimal reference design '
        f'(mag 8, 0.9", nominal wind, delay 2).',
        f'j ≤ {J_RM}: ref and e_p per mode.   j > {J_RM}: ref = optimal reference curve with isolated spikes replaced; e_p(j) = smooth scaling '
        f'from one joint fit,',
        '        monotone in j with a single curvature over mag 5-11, wind x0.5-2, seeing 0.6-1.05", delay 1.5-2.5; IIR2 designs on the '
        "optimiser's second branch excluded."]
LAWS = dict(
    G=[law_txt('lG', 'log10 G')], leak=[law_txt('l1mleak', 'log10 (1 − leak)')],
    b0=[law_txt('lb0', 'log10 b0')],
    b1=[f'j ≤ {J_RM}:  ' + law_txt('lmb1', 'log10 (−b1)'), f'j > {J_RM}:  b1 = F(1)·A(1) − b0 − b2   (keeps the DC gain)',
        law_txt('F1', 'F(1)') + ',   F(1) = (b0 + b1 + b2)/A(1)', law_txt('lA1', 'log10 A(1)') + ',   A(1) = 1 + a1 + a2'],
    b2=[f'j ≤ {J_RM}:  b2 = F(1)·A(1) − b0 − b1', f'j > {J_RM}:  ' + law_txt('b2', 'b2'),
        law_txt('F1', 'F(1)') + ',   F(1) = (b0 + b1 + b2)/A(1)', law_txt('lA1', 'log10 A(1)') + ',   A(1) = 1 + a1 + a2'],
    a1=['a1 = A(1) − 1 − a2', law_txt('lA1', 'log10 A(1)') + ',   A(1) = 1 + a1 + a2'],
    a2=[law_txt('l1ma2', 'log10 (1 − a2)')])


def add_law(fig, lines):
    """Formula box under the panels; the panels are laid out above it."""
    lines = lines + FOOT
    h = (len(lines) * 7.5 * 1.3 / 72 + 0.25) / fig.get_size_inches()[1]   # box height, figure fraction
    fig.tight_layout(rect=(0, h + 0.01, 1, 1))
    fig.text(0.01, 0.008, '\n'.join(lines), fontsize=7.5, va='bottom', ha='left', bbox=dict(boxstyle='round', fc='white', ec='0.7'))


SWEEPS = [('wind', 'k', 'wind speed x{:g}'), ('seeing', 's', 'seeing {:g}"'), ('mag', 'mag', 'mag {:g}'), ('delay', 't', 'delay {:g} frames')]


def main():
    d = dict(np.load(os.path.join(HERE, 'coeff_laws_rmod0.npz')))
    laws = pickle.load(open(os.path.join(HERE, 'coeff_laws_rmod0.pkl'), 'rb'))
    for name, key, label in SWEEPS:
        others = np.all([np.isclose(d[o], v) for o, v in REF.items() if o != key], 0)
        vals = np.unique(d[key][others])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(vals)))
        for v, c in zip(vals, cols):
            m = others & np.isclose(d[key], v)
            j = d['j'][m].astype(int)
            o = np.argsort(j)
            j, G, leak = j[o], d['G'][m][o], d['leak'][m][o]
            cond = {**REF, key: v}
            P = [predict(laws, jj, cond['mag'], cond['k'], cond['s'], cond['t']) for jj in j]
            ax[0].plot(j, G, '-', color=c, lw=0.8, alpha=0.45, label=label.format(v))
            ax[0].plot(j, [p['G'] for p in P], '--', color=c, lw=1.6, zorder=3)
            ax[1].plot(j, leak, '-', color=c, lw=0.8, alpha=0.45)
            ax[1].plot(j, [p['leak'] for p in P], '--', color=c, lw=1.6, zorder=3)
        ax[0].set_xscale('log')                  # gain: log x; leak: linear x (it only departs from 1 near the cut-off)
        ax[1].set_xlim(0, d['j'].max() + 1)
        for a, yl in zip(ax, ['gain G', 'leak']):
            a.set_xlabel('KL mode')
            a.set_ylabel(yl)
            a.grid(alpha=0.3, which='both')
        ax[0].plot([], [], 'k-', label='optimal')
        ax[0].plot([], [], 'k--', label='law')
        ax[0].legend(fontsize=8)
        fig.suptitle(f'Leaky integrator, rMod 0, 2 kHz: {name} sweep (others: mag 8, 0.9", nominal wind, delay 2)')
        fig.set_size_inches(12, 5.8)
        add_law(fig, LAWS['G'] + LAWS['leak'])
        f = os.path.join(HERE, f'leaky_law_rmod0_{name}.png')
        fig.savefig(f, dpi=130)
        print(f)


IIR2 = [('b0', lambda p: p['b'][0]), ('b1', lambda p: p['b'][1]), ('b2', lambda p: p['b'][2]), ('a1', lambda p: p['a'][1]),
        ('a2', lambda p: p['a'][2])]


def iir2():
    """One figure per IIR2 coefficient, 2x2 panels (wind, seeing, magnitude, delay sweeps), optimal (solid) vs law (dashed)."""
    d = dict(np.load(os.path.join(HERE, 'coeff_laws_rmod0.npz')))
    laws = pickle.load(open(os.path.join(HERE, 'coeff_laws_rmod0.pkl'), 'rb'))
    curves = {}                               # sweep -> [(value, j, law designs)]
    for name, key, label in SWEEPS:
        others = np.all([np.isclose(d[o], v) for o, v in REF.items() if o != key], 0)
        curves[name] = []
        for v in np.unique(d[key][others]):
            m = others & np.isclose(d[key], v)
            o = np.argsort(d['j'][m])
            j = d['j'][m][o].astype(int)
            cond = {**REF, key: v}
            P = [predict(laws, jj, cond['mag'], cond['k'], cond['s'], cond['t']) for jj in j]
            curves[name].append((v, m.nonzero()[0][o], j, P))
    for coef, get in IIR2:
        fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))
        for a, (name, key, label) in zip(axs.ravel(), SWEEPS):
            cols = plt.cm.viridis(np.linspace(0, 0.9, len(curves[name])))
            for (v, idx, j, P), c in zip(curves[name], cols):
                a.plot(j, d[coef][idx], '-', color=c, lw=0.8, alpha=0.45, label=label.format(v))
                a.plot(j, [get(p) for p in P], '--', color=c, lw=1.6, zorder=3)
            a.plot([], [], 'k-', label='optimal')
            a.plot([], [], 'k--', label='law')
            a.set_xscale('log')
            a.set_xlabel('KL mode')
            a.set_ylabel(coef)
            a.set_title(f'{name} sweep')
            a.grid(alpha=0.3, which='both')
            a.legend(fontsize=7)
        fig.suptitle(f'IIR2 {coef}, rMod 0, 2 kHz (others: mag 8, 0.9", nominal wind, delay 2)')
        fig.set_size_inches(13, 10)
        add_law(fig, ['IIR2 (POL): C(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)'] + LAWS[coef])
        f = os.path.join(HERE, f'iir2_law_rmod0_{coef}.png')
        fig.savefig(f, dpi=120)
        print(f)


if __name__ == '__main__':
    main()
    iir2()
