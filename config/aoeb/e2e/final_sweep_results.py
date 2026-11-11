"""Results of the final e2e sweep (sweep_final.py, results/WFSW/sweep_final_20261108): replay statistics and the comparison figures.

Magnitudes 5 and 6 were not swept: they are replays of the configuration selected at magnitude 7. Each point of the single-stage and fusion
curves is the mean over the replays (REPLAY_TIME s, other atmosphere seed and random wind direction per
layer) of the configuration the sweep selected at that magnitude; error bars: standard deviation over the replays. Only magnitudes where every replay
converged are plotted (the others are listed under the figure); a replay ran away if it is non-finite, or 550 nm SR of its last 0.5 s below RUNAWAY x that of the 0.5 s after
START_TIME, or 550 nm SR below RUNAWAY_MEDIAN x the median of the other realizations (or of the converged replays of the next fainter
magnitude, when it uses the same configuration: a brighter star cannot do worse). Only the replays of the configuration selected now count
(replays of a configuration selected before stay in replays.csv). The equal-flux dual cases were not replayed: their points
are the sweep values (one realization, the one the search was optimised on), without error bars, and so are their reference curves.
Contrast: mean raw contrast in 2-5 lambda/D, relative to the integrated AO peak. Comparison panels: each curve over (contrast: ratio) or
minus (SR: difference) the single stage with rMod 1 (same WFS wavelength and chromatic setting).
"""
import csv
import os.path as op
import textwrap

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from results_analysis import load_coro_profile, load_perfect_coro

OUT = '/raid1/mmenessini/results/WFSW/sweep_final_20261108'
MAGS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)    # 5 and 6: replays of the mag 7 configuration (not swept)
SWEPT_MAGS = MAGS[2:]
REF_MAG = 7.0
START_TIME, WINDOW, RUNAWAY, RUNAWAY_MEDIAN = 0.5, 0.5, 0.7, 0.5
PROFILE_YLIM = (1e-7, 1e-3)    # raw contrast range of the radial-profile panels
SR_MASK = 0.25            # contrast points with the science SR below this are drawn hollow
WL = {'vis': '550 nm', 'nir': '1650 nm'}
RMODS = (1, 2)            # single-stage modulations shown (rMod 0 swept and replayed too, not shown)
REF = 'r1'                # single stage every comparison is made against (rMod 1); rMod 2 shown for reference
SPLITS = (10, 30, 50, 70, 90)        # % of the band on the unmodulated pyramid

# Colors follow the entity; splits: ordinal blue ramp, light = little unmodulated flux
COLOR = {'fusion': 'tab:blue', 'swapped': 'tab:orange', 'r0': 'tab:purple', 'r1': 'tab:green', 'r2': 'tab:red'}
PROFILE_COLOR = {'single': COLOR[REF], 'fusion': 'tab:blue'}      # single stage rMod 1 and best fusion (rMod 2: COLOR['r2'])
MARKER = {'fusion': 'o', 'swapped': 's', 'r0': '^', 'r1': 'v', 'r2': 'D'}
SPLIT_COLOR = dict(zip(SPLITS, ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281')))
LW, MS, MEW, ELW, CAP = 1.3, 5.5, 0.8, 0.9, 2     # line width, marker size and edge, error-bar width and cap
FS_LEG, FS_TICK, FS_LAB, FS_TITLE, FS_SUP, FS_NOTE = 9.5, 10, 11, 12, 14, 9   # font sizes
INK, MUTED, GRID = '#1a1a19', '#6b6a63', '#e4e3dd'


def T(text):
    """Text for matplotlib: LaTeX-safe when text.usetex is on (results_analysis turns it on)."""
    if text is None or not plt.rcParams['text.usetex']:
        return text
    for a, b in (('%', r'\%'), ('λ', r'$\lambda$'), ('±', r'$\pm$'), (' < ', ' $<$ ')):
        text = text.replace(a, b)
    return text


# ----------------------------------------------------------------------------- data
def _float(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return np.nan


def _sr_windows(path, fs):
    sr = np.asarray(fits.getdata(op.join(path, 'sr_vis.fits')), dtype=float).ravel()
    i0, w = int(round(START_TIME * fs)), int(round(WINDOW * fs))
    return float(np.mean(sr[i0:i0 + w])), float(np.mean(sr[-w:]))


def _fs_of(case, key):
    """Loop rate of a selected configuration (selected.csv key): single [rmod, mag, fs, ...], dual [mag, k, n, k_unmod, ...]."""
    return float(key[2]) if case.startswith('single') else 2000.0 / float(key[3])


def selected(case):
    """Selected configuration per magnitude: {mag: row} (sweep values, one realization)."""
    path = op.join(OUT, case, 'selected.csv')
    if not op.exists(path):
        return {}
    import yaml
    out = {}
    for r in csv.DictReader(open(path)):
        r['key'] = yaml.safe_load(r['key'])
        out[float(r['mag'])] = r
    return out


def replays(case):
    """{mag: dict(n, n_ok, mean/std of contrast_vis, contrast_nir, sr_vis, sr_nir, tns, runaway tns)} over the converged replays."""
    path = op.join(OUT, case, 'replays.csv')
    sel = selected(case)
    if not op.exists(path):
        return {}
    rows = list(csv.DictReader(open(path)))
    out, fainter = {}, None        # fainter: (config, converged median 550 nm SR) of the next fainter magnitude
    for mag in sorted({float(r['mag']) for r in rows}, reverse=True):
        cur = sel.get(mag, sel.get(REF_MAG))          # current selection (mags 5, 6: that of mag 7)
        rs = [r for r in rows if float(r['mag']) == mag and (cur is None or r['config'] == cur['config'])]
        if not rs:
            continue
        fs = _fs_of(case, cur['key']) if cur else 2000.0
        sr550 = np.array([_float(r, 'sr_vis') for r in rs])
        med = np.nanmedian(sr550)
        if fainter and cur and fainter[0] == cur['config']:
            med = max(med, fainter[1])   # same configuration, brighter star: SR cannot be lower (catches runs that collapse before START_TIME)
        ok, bad = [], []
        for r, s in zip(rs, sr550):
            finite = all(np.isfinite(_float(r, k)) for k in ('contrast_vis', 'contrast_nir', 'sr_vis', 'sr_nir'))
            a, e = _sr_windows(op.join(OUT, case, 'replays', r['tn']), fs) if finite else (np.nan, np.nan)
            ran = not finite or not (e >= RUNAWAY * a) or s < RUNAWAY_MEDIAN * med
            (bad if ran else ok).append(r)
        d = dict(n=len(rs), n_ok=len(ok), tns=[r['tn'] for r in ok], runaway_tns=[r['tn'] for r in bad])
        for k in ('contrast_vis', 'contrast_nir', 'sr_vis', 'sr_nir'):
            v = np.array([_float(r, k) for r in ok])
            d[k] = float(np.mean(v)) if len(v) else np.nan
            d[k + '_std'] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        out[mag] = d
        fainter = (cur['config'], float(np.median([_float(r, 'sr_vis') for r in ok]))) if cur and ok else None
    return dict(sorted(out.items()))


def complete(st, m, k=None):
    """Is magnitude m of a replay (or sweep) statistics dict plotted: all its replays converged (and the value k finite)?"""
    d = st.get(m)
    return d is not None and d.get('n_ok', 1) == d.get('n', 1) and (k is None or np.isfinite(d[k]))


def floors():
    """Mean 2-5 lambda/D contrast of the perfect-coronagraph references (0.9" line of sight, z = 30): {case: {band: value}}."""
    out = {}
    for c in ('achrom', 'vis', 'nir'):
        rad, cv, cn = load_perfect_coro(c)
        sel = (rad >= 2) & (rad <= 5)
        out[c] = {'vis': float(np.mean(cv[sel])), 'nir': float(np.mean(cn[sel]))}
    return out


# ----------------------------------------------------------------------------- panels
def main_panels():
    """(chromatic, band) -> [(entity, label, case)] of the single / fusion / fusion swapped figure. Single stage: WFS at 900 nm for 550 nm,
    at 1300 nm for 1650 nm; there is no achromatic 1300 nm single stage, so the achromatic 1650 nm panel uses the 900 nm WFS."""
    p = {}
    for chrom in (False, True):
        t = 'chrom' if chrom else 'achrom'
        for band in ('vis', 'nir'):
            w = 1300 if (band == 'nir' and chrom) else 900
            curves = [(f'r{r}', f'single, rMod {r}', f'single_wfs{w}_{t}_r{r}') for r in RMODS]
            curves.append(('fusion', 'fusion: unmod 900 nm + rMod 2 1300 nm', f'dual_unmod900_{t}'))
            if chrom:
                curves.append(('swapped', 'fusion: unmod 1300 nm + rMod 2 900 nm', 'dual_unmod1300_chrom'))
            p[(chrom, band)] = curves
    return p


def equal_flux_panels():
    """(chromatic, band) -> (wavelength of both pyramids, equal-flux case stem, single-stage cases of the reference, fusion case)."""
    p = {}
    for chrom in (False, True):
        t = 'chrom' if chrom else 'achrom'
        for band in ('vis', 'nir'):
            w = 1300 if (band == 'nir' and chrom) else 900
            p[(chrom, band)] = (w, f'dual_eq{w}_{t}', [f'single_wfs{w}_{t}_r{r}' for r in RMODS], f'dual_unmod{w}_{t}')
    return p


def _title(chrom, band):
    """Panel title: chromatic setting, science band and WFS wavelength of the single stage."""
    w = 1300 if (band == 'nir' and chrom) else 900
    return f"{'chromatic' if chrom else 'achromatic'}, {WL[band]} (single stage: WFS {w} nm)"


def _floor_lines(ax, chrom, band, fl):
    if not chrom:
        refs = [('achrom', 'fitting floor', '--')]
    else:
        refs = [('vis', 'floor, WFS 900 nm', '--'), ('nir', 'floor, WFS 1300 nm', ':')]
    for c, label, ls in refs:
        below = c == 'vis' and chrom        # the two chromatic floors are close: one label under its line, one above
        ax.axhline(fl[c][band], color=MUTED, lw=0.9, ls=ls, zorder=1)
        ax.annotate(T(label), (1.0, fl[c][band]), xycoords=('axes fraction', 'data'), xytext=(-4, -3 if below else 3), textcoords='offset points',
                    ha='right', va='top' if below else 'bottom', fontsize=FS_NOTE, color=MUTED)


_LEG = dict(fontsize=FS_LEG, loc='best', frameon=True, facecolor='white', edgecolor='none', framealpha=0.85, handlelength=2.2)


def _style(ax, logy, mags=MAGS):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=FS_TICK)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(mags)


def _curve(ax, x, y, err, color, marker, label, hollow=None, counts=None):
    x, y = np.asarray(x), np.asarray(y)
    if err is not None:
        ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, elinewidth=ELW, capsize=CAP, zorder=3)
    ax.plot(x, y, color=color, lw=LW, marker=marker, ms=MS, mec='white', mew=MEW, zorder=4, label=T(label))
    hollow = np.zeros(len(x), bool) if hollow is None else np.asarray(hollow, bool)
    if hollow.any():     # science SR below SR_MASK: white face
        ax.plot(x[hollow], y[hollow], ls='none', marker=marker, ms=MS, mfc='white', mec=color, mew=1.1, zorder=5)
    for xi, yi, c in zip(x, y, counts or []):
        if c:
            ax.annotate(T(c), (xi, yi), xytext=(9, 5), textcoords='offset points', ha='left', va='bottom', fontsize=FS_NOTE, color=INK)


def _figure(title):
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=(3, 1.2, 3, 1.2))
    fig.suptitle(T(title), fontsize=FS_SUP, color=INK)
    return fig, gs


def _axes(fig, gs, row, col):
    ax = fig.add_subplot(gs[2 * row, col])
    rx = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    return ax, rx


def _cmp(quantity, value, ref):
    """Comparison row: contrast ratio value / ref, SR difference value - ref."""
    return value / ref if quantity == 'contrast' else value - ref


def _labels(ax, rx, quantity, chrom, band, ref_label, top_title):
    q = 'contrast 2-5 λ/D' if quantity == 'contrast' else 'SR'
    ax.set_ylabel(T(f'{q}, {WL[band]}'), color=INK, fontsize=FS_LAB)
    ax.set_title(T(top_title), fontsize=FS_TITLE, color=INK, loc='left')
    rx.set_ylabel(T(f'ratio wrt\n{ref_label}' if quantity == 'contrast' else f'difference wrt\n{ref_label}'), color=INK, fontsize=FS_TICK)
    rx.set_xlabel(T('magnitude (900 nm)'), color=INK, fontsize=FS_LAB)
    rx.axhline(1.0 if quantity == 'contrast' else 0.0, color=MUTED, lw=0.8, zorder=1)
    rx.set_ylim(*cmp_ylim(quantity))


def cmp_ylim(quantity):
    """Common y range of every comparison row (contrast ratio / SR difference) of every figure: each main-panel curve wrt single rMod 1
    and each fusion wrt single rMod 2 (replay means, every replay converged), each equal-flux split wrt single rMod 1 (sweep values)."""
    k_of = lambda band: f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
    vals = [1.0 if quantity == 'contrast' else 0.0]
    for (chrom, band), curves in main_panels().items():
        k = k_of(band)
        st = {c: replays(c) for _, _, c in curves}
        for ref_e, ents in ((REF, None), ('r2', ('fusion', 'swapped'))):
            rs = st[next(c for e, _, c in curves if e == ref_e)]
            for e, _, c in curves:
                if ents is None or e in ents:
                    vals += [_cmp(quantity, st[c][m][k], rs[m][k]) for m in MAGS if complete(st[c], m, k) and complete(rs, m, k)]
    for (chrom, band), (w, stem, singles, fusion) in equal_flux_panels().items():
        k = k_of(band)
        ref = _sweep_stats(next(c for c in singles if c.endswith('_' + REF)), band)
        for u in SPLITS:
            sel = selected(f'{stem}_u{u}')
            vals += [_cmp(quantity, float(sel[m][k]), ref[m][k]) for m in sel if m in ref]
    vals = [v for v in vals if np.isfinite(v)]
    pad = 0.05 * (max(vals) - min(vals))
    return min(vals) - pad, max(vals) + pad


def ylim(quantity):
    """Common y range of every contrast (or SR) panel of both figures: all replay means +- std, sweep values and, for the contrast, the floors."""
    import glob
    vals = []
    for d in glob.glob(op.join(OUT, '*', 'selected.csv')):
        case = op.basename(op.dirname(d))
        if case.endswith('_svd'):         # plain-SVD rMod 0 replays: stopped, not plotted (2026-11-10)
            continue
        for b in ('vis', 'nir'):
            k = f'contrast_{b}' if quantity == 'contrast' else f'sr_{b}'
            vals += [float(r[k]) for r in selected(case).values()]
            for st in replays(case).values():
                if st['n_ok'] == st['n'] and np.isfinite(st[k]):
                    e = st[k + '_std'] if np.isfinite(st[k + '_std']) else 0.0
                    vals += [st[k] - e, st[k] + e]
    if quantity == 'contrast':
        vals += [v for f in floors().values() for v in f.values()]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        return min(vals) / 1.3, max(vals) * 1.3
    vals = [v for v in vals if np.isfinite(v)]
    return 0.0, 1.0                    # SR: full range


def plot_main(quantity='contrast'):
    """Single (one curve per rMod), fusion and fusion swapped: replay mean +- std, achromatic (top) and chromatic (bottom), 550 / 1650 nm."""
    fl = floors() if quantity == 'contrast' else None
    panels = main_panels()
    stats = {c: replays(c) for curves in panels.values() for _, _, c in curves}
    fig, gs = _figure(f"{'Contrast' if quantity == 'contrast' else 'Strehl ratio'}: replay mean ± std of the selected configurations")
    missing, dropped, lim = set(), set(), ylim(quantity)
    for (chrom, band), curves in panels.items():
        ax, rx = _axes(fig, gs, int(chrom), 0 if band == 'vis' else 1)
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        singles = [c for e, _, c in curves if e.startswith('r')]
        rc = next(c for e, _, c in curves if e == REF)           # comparisons: against the single stage with rMod 1
        ref = {m: stats[rc][m][k] for m in MAGS if complete(stats[rc], m, k)}
        for e, label, case in curves:
            st = stats[case]
            dropped |= {f"{case} mag {m:g} ({st[m]['n_ok']}/{st[m]['n']})" for m in MAGS if m in st and st[m]['n_ok'] < st[m]['n']}
            if any(m not in st for m in MAGS):
                missing.add(case)
            mags = [m for m in MAGS if complete(st, m, k)]     # only magnitudes where every replay converged
            if not mags:
                continue
            y = [st[m][k] for m in mags]
            err = [st[m][k + '_std'] if np.isfinite(st[m][k + '_std']) else 0.0 for m in mags]
            hollow = [st[m][f'sr_{band}'] < SR_MASK for m in mags] if quantity == 'contrast' else None
            _curve(ax, mags, y, err, COLOR[e], MARKER[e], label, hollow)
            rm = [m for m in mags if m in ref]
            _curve(rx, rm, [_cmp(quantity, st[m][k], ref[m]) for m in rm], None, COLOR[e], MARKER[e], None)
        if fl:
            _floor_lines(ax, chrom, band, fl)
        _style(ax, quantity == 'contrast')
        _style(rx, False)
        _labels(ax, rx, quantity, chrom, band, 'single rMod 1', _title(chrom, band))
        ax.set_ylim(*lim)
        if band == 'vis':        # same curves in both panels of a row: one legend per row
            ax.legend(**_LEG)
    _footnote(fig, quantity, missing, dropped=dropped)
    return fig


def _sweep_stats(case, band):
    """Sweep values of the selected configurations, in the shape of replays() (no std): the nominal realization the search was optimised on."""
    out = {}
    for m, r in selected(case).items():
        out[m] = {f'{q}_{b}': float(r[f'{q}_{b}']) for q in ('contrast', 'sr') for b in ('vis', 'nir')}
    return out


def plot_equal_flux(quantity='contrast'):
    """Equal-flux dual (both pyramids at one wavelength, band split unmod/mod): sweep values of each split, with the best single stage and the
    fusion for reference, ALSO sweep values: all on the nominal realization, a paired comparison (the replay means differ from it, by about
    +10 % in the chromatic cases, so mixing the two would bias the ratios); comparison panels over the single stage
    with rMod 1 (contrast: ratio, SR: difference)."""
    fl = floors() if quantity == 'contrast' else None
    fig, gs = _figure(f"Equal-flux dual: {'contrast' if quantity == 'contrast' else 'Strehl ratio'} of each band split (sweep values)")
    missing, lim = set(), ylim(quantity)
    for (chrom, band), (w, stem, singles, fusion) in equal_flux_panels().items():
        ax, rx = _axes(fig, gs, int(chrom), 0 if band == 'vis' else 1)
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        stats = {c: _sweep_stats(c, band) for c in singles + [fusion]}
        rc = next(c for c in singles if c.endswith('_' + REF))   # reference and comparisons: the single stage with rMod 1
        ref = {m: stats[rc][m][k] for m in MAGS if complete(stats[rc], m, k)}
        for u in SPLITS:
            sel = selected(f'{stem}_u{u}')
            mags = [m for m in MAGS if m in sel]
            if not mags:
                missing.add(f'{stem}_u{u}')
                continue
            y = [float(sel[m][k]) for m in mags]
            hollow = [float(sel[m][f'sr_{band}']) < SR_MASK for m in mags] if quantity == 'contrast' else None
            _curve(ax, mags, y, None, SPLIT_COLOR[u], 'o', f'unmod {u}% / rMod 2 {100 - u}% of the band', hollow)
            rm = [m for m in mags if m in ref]
            _curve(rx, rm, [_cmp(quantity, float(sel[m][k]), ref[m]) for m in rm], None, SPLIT_COLOR[u], 'o', None)
        rm = sorted(ref)
        if rm:
            ax.plot(rm, [ref[m] for m in rm], color=MUTED, lw=LW, ls='--', zorder=2, label=T(f'single: WFS {w} nm, rMod 1'))
        st = stats[fusion]
        fm = [m for m in MAGS if m in st and np.isfinite(st[m][k])]
        if fm and w == 900:
            ax.plot(fm, [st[m][k] for m in fm], color=INK, lw=1.1, ls='-.', zorder=2, label=T('fusion: unmod 900 nm + rMod 2 1300 nm'))
        if fl:
            _floor_lines(ax, chrom, band, fl)
        _style(ax, quantity == 'contrast', SWEPT_MAGS)
        _style(rx, False, SWEPT_MAGS)
        _labels(ax, rx, quantity, chrom, band, 'single rMod 1', f"{'chromatic' if chrom else 'achromatic'}, both pyramids at {w} nm, {WL[band]}")
        ax.set_ylim(*lim)
        ax.legend(**_LEG)
    _footnote(fig, quantity, missing, equal_flux=True)
    return fig


def _wrap(text, width=190):
    """Footnote wrapped to the figure width (a long single line widens the saved figure with bbox_inches='tight')."""
    return '\n'.join(textwrap.wrap(text, width, break_long_words=False, break_on_hyphens=False))


def _footnote(fig, quantity, missing, equal_flux=False, dropped=()):
    notes = []
    if quantity == 'contrast':
        notes.append(f'hollow markers: science SR < {SR_MASK}')
    if equal_flux:
        notes.append('all curves: sweep values on the nominal realization (splits not replayed), no error bars')
    else:
        notes.append('points only where every replay converged')
        if dropped:
            notes.append('dropped (replays ran away): ' + ', '.join(sorted(dropped)))
    if missing:
        notes.append('incomplete: ' + ', '.join(sorted(missing)))
    fig.text(0.01, -0.01, T(_wrap('; '.join(notes))), fontsize=FS_NOTE, color=MUTED, ha='left', va='top')


def table(quantity='contrast'):
    """Rows (case, mag, mean, std, n_ok, n, runaway TNs) of the main figure, for the record."""
    out = []
    for (chrom, band), curves in main_panels().items():
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        for e, label, case in curves:
            for m, d in replays(case).items():
                out.append(dict(chromatic=chrom, band=WL[band], curve=label, case=case, mag=m, mean=d[k], std=d[k + '_std'], n_ok=d['n_ok'], n=d['n'],
                                runaways=' '.join(d['runaway_tns'])))
    return out


# ----------------------------------------------------------------------------- radial profiles
def profile(case, mag, band):
    """(rad, mean, std, label suffix) of the coronagraphic profile of the configuration selected at `mag`: mean and std over the converged
    replays or, while there are none, the sweep run (std None)."""
    st = replays(case).get(mag)
    if st and st['n_ok'] < st['n']:           # some replays ran away: not plotted
        return 'dropped'
    if st and st['tns']:
        profs = [load_coro_profile(op.join(OUT, case, 'replays', tn), band) for tn in st['tns']]
        rad = profs[0][0]
        p = np.array([q for _, q in profs])
        return rad, p.mean(axis=0), (p.std(axis=0, ddof=1) if len(p) > 1 else None), f"{st['n_ok']}/{st['n']} replays"
    sel = selected(case).get(mag)
    if sel is None or not sel.get('tn'):     # replay-only cases have no sweep run
        return None
    rad, p = load_coro_profile(op.join(OUT, case, 'runs', sel['tn']), band)
    return rad, p, None, 'sweep run'


def _best_case(cands, mag, band, notes):
    """(label, case) with the lowest mean 2-5 lambda/D contrast at `mag` in `band`: replays where every one converged or, for a case with no
    replays at that magnitude yet, its sweep run. Cases left out are added to notes."""
    vals = []
    for label, case in cands:
        st = replays(case).get(mag)
        if st is not None:
            if st['n_ok'] < st['n']:
                notes.add(f'{case} (replays ran away: not considered)')
                continue
            vals.append((st[f'contrast_{band}'], label, case))
            continue
        sel = selected(case).get(mag)
        if sel is not None and sel.get('tn'):
            notes.add(f'{case} (no replays yet: sweep run)')
            vals.append((float(sel[f'contrast_{band}']), label, case))
        else:
            notes.add(f'{case} (no data)')
    vals = [v for v in vals if np.isfinite(v[0])]
    return min(vals)[1:] if vals else None


def _values(case, mag):
    """Mean contrast and SR (both bands) of a case at mag: its replays (every one converged) or, without replays, its sweep run."""
    st = replays(case).get(mag)
    if st is not None:
        return {k: st[k] for k in ('contrast_vis', 'contrast_nir', 'sr_vis', 'sr_nir')}, f"{st['n']} replays"
    sel = selected(case)[mag]
    return {k: float(sel[k]) for k in ('contrast_vis', 'contrast_nir', 'sr_vis', 'sr_nir')}, 'sweep run'


def _single_cands(curves, single):
    """Single-stage candidates of a panel: every rMod curve (single='best') or only the one of entity `single` (e.g. 'r2')."""
    return [(lab, c) for e, lab, c in curves if (e.startswith('r') if single == 'best' else e == single)]


def best_pair_ratios(mag=7.0, single=REF):
    """Single stage of entity `single` (default rMod 1; 'best': the best rMod curve) vs best fusion of every panel of
    plot_profiles (same selection), at one magnitude or a list of magnitudes: mean 2-5 lambda/D contrast and SR in the panel's band, the
    contrast ratio single / fusion (> 1: the fusion is better) and the SR difference fusion - single (> 0: the fusion is better)."""
    rows = []
    for mg in np.atleast_1d(mag):
        mg = float(mg)
        for (chrom, band), curves in main_panels().items():
            notes = set()
            pick = {'single': _best_case(_single_cands(curves, single), mg, band, notes),
                    'fusion': _best_case([(lab, c) for e, lab, c in curves if e in ('fusion', 'swapped')], mg, band, notes)}
            if None in pick.values():
                continue
            (vs, hs), (vf, hf) = _values(pick['single'][1], mg), _values(pick['fusion'][1], mg)
            cs, cf, ss, sf = vs[f'contrast_{band}'], vf[f'contrast_{band}'], vs[f'sr_{band}'], vf[f'sr_{band}']
            rows.append({'mag': mg, 'chromatic': chrom, 'band': WL[band], 'single': pick['single'][0], 'fusion': pick['fusion'][0],
                         'contrast single': cs, 'contrast fusion': cf, 'contrast single/fusion': cs / cf,
                         'SR single': ss, 'SR fusion': sf, 'SR fusion-single': sf - ss, 'data': f'{hs} / {hf}'})
    return rows


def best_pair_table(mag=7.0, single=REF):
    """best_pair_ratios as a formatted pandas DataFrame (contrasts in scientific notation); mag: one magnitude or a list (e.g. MAGS)."""
    import pandas as pd
    df = pd.DataFrame(best_pair_ratios(mag, single))
    for c in ('contrast single', 'contrast fusion'):
        df[c] = df[c].map('{:.3e}'.format)
    for c in ('SR single', 'SR fusion'):
        df[c] = df[c].map('{:.4f}'.format)
    df['contrast single/fusion'] = df['contrast single/fusion'].map('{:.3f}'.format)
    df['SR fusion-single'] = df['SR fusion-single'].map('{:+.4f}'.format)
    if np.ndim(mag) == 0:
        df = df.drop(columns='mag')
    return df


def plot_profiles(mag=7.0, rmax=30.0):
    """Radial raw-contrast profiles at one magnitude, achromatic (top) and chromatic (bottom), 550 / 1650 nm: the single stage rMod 1 (green),
    rMod 2 (red, dashed) and the best fusion (blue, fusion or fusion swapped) at that magnitude and band (lowest mean 2-5 lambda/D contrast), replay mean (band: +- std), with the perfect-coronagraph bounds: fitting (achromatic) and fitting +
    chromatism (WFS at 900 nm, WFS at 1300 nm). Same y range in every panel; shaded: the 2-5 lambda/D of the metric."""
    refs = {c: load_perfect_coro(c) for c in ('achrom', 'vis', 'nir')}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True, sharex=True)
    fig.suptitle(T(f'Raw contrast profiles at magnitude {mag:g} (900 nm): replay mean ± std'), fontsize=FS_SUP, color=INK)
    lo, hi, notes = np.inf, 0.0, set()
    for (chrom, band), curves in main_panels().items():
        ax = axes[int(chrom), 0 if band == 'vis' else 1]
        bi = 1 if band == 'vis' else 2
        rad = refs['achrom'][0]
        sel = rad <= rmax
        ax.axvspan(2, 5, color=GRID, alpha=0.6, lw=0, zorder=0)
        bounds = [('achrom', 'fitting', '--')] + ([('vis', 'fitting + chromatism, WFS 900 nm', ':'), ('nir', 'fitting + chromatism, WFS 1300 nm', '-.')]
                                                  if chrom else [])
        for c, label, ls in bounds:
            ax.plot(rad[sel], refs[c][bi][sel], color=MUTED, lw=1.0, ls=ls, zorder=2, label=T(label))
            lo = min(lo, np.min(refs[c][bi][sel & (rad >= 1)]))
        for group in ('single', 'r2', 'fusion'):
            cands = (_single_cands(curves, REF) if group == 'single' else _single_cands(curves, 'r2') if group == 'r2'
                     else [(lab, c) for e, lab, c in curves if e in ('fusion', 'swapped')])
            best = _best_case(cands, mag, band, notes)
            if best is None:
                continue
            label, case = best
            pr = profile(case, mag, band)
            r, m, sd, how = pr
            k = r <= rmax
            col = COLOR['r2'] if group == 'r2' else PROFILE_COLOR[group]
            ax.plot(r[k], m[k], color=col, lw=1.1 if group == 'r2' else LW, ls='--' if group == 'r2' else '-', zorder=4, label=T(label))
            if sd is not None and group != 'r2':      # rMod 2: no band (readability)
                ax.fill_between(r[k], np.maximum(m[k] - sd[k], 1e-12), m[k] + sd[k], color=col, alpha=0.2, lw=0, zorder=3)
            hi = max(hi, np.max(m[k & (r >= 1)]))
            lo = min(lo, np.min(m[k & (r >= 1)]))
        _style(ax, True)
        ax.set_xticks(np.arange(0, rmax + 0.1, 5))
        ax.set_xlim(0, rmax)
        ax.set_title(T(_title(chrom, band)), fontsize=FS_TITLE, color=INK, loc='left')
        ax.set_ylabel(T(f'raw contrast, {WL[band]}'), color=INK, fontsize=FS_LAB)
        if chrom:
            ax.set_xlabel(T('separation [λ/D]'), color=INK, fontsize=FS_LAB)
        ax.legend(**_LEG)
    for ax in axes.ravel():
        ax.set_ylim(*PROFILE_YLIM)
    note = 'shaded column: 2-5 λ/D (metric); bands: ± std over the replays (curves only where every replay converged)'
    if notes:
        note += '; incomplete: ' + ', '.join(sorted(notes))
    fig.text(0.01, -0.01, T(_wrap(note)), fontsize=FS_NOTE, color=MUTED, ha='left', va='top')
    return fig


# ----------------------------------------------------------------------------- best single vs best fusion
def _best_replayed(curves, group, mag, band, single='best'):
    """(entity, label, case) of the single stage (group 'single': the rMod curves) or fusion ('fusion': fusion and swapped) with the lowest
    replay-mean 2-5 lambda/D contrast at `mag` in `band`, among the cases where every replay converged; None when there is none."""
    vals = []
    for e, label, case in curves:
        if group == 'single' and not (e.startswith('r') if single == 'best' else e == single):
            continue
        if group == 'fusion' and e not in ('fusion', 'swapped'):
            continue
        st = replays(case)
        if complete(st, mag, f'contrast_{band}'):
            vals.append((st[mag][f'contrast_{band}'], e, label, case))
    return min(vals)[1:] if vals else None


def plot_best(quantity='contrast', single=REF, extra=('r2',)):
    """Best single stage vs best fusion: at each magnitude, panel (achromatic / chromatic, 550 / 1650 nm) and band, the single stage (rMod 0, 1 or 2)
    and the fusion (fusion or fusion swapped) with the lowest replay-mean 2-5 lambda/D contrast; the winner is selected on the contrast for the
    SR figure too, so both figures show the same runs. Replay mean +- std; the marker is that of the winning case (MARKER), so a switch along the
    curve is visible. Comparison rows: contrast best fusion / single (< 1: the fusion is better), SR best fusion - single (> 0: the fusion is
    better). single: the rMod curve compared against at every magnitude (default rMod 1; 'best': the best rMod curve at each magnitude);
    extra: other single stages drawn too (default rMod 2), each with its own comparison curve against the best fusion."""
    fl = floors() if quantity == 'contrast' else None
    panels = main_panels()
    sname = 'best single' if single == 'best' else f'single rMod {single[1:]}'
    fig, gs = _figure(f"{'Contrast' if quantity == 'contrast' else 'Strehl ratio'}: {sname} stage vs best fusion (replay mean ± std)")
    lim = ylim(quantity)
    entity_label = {'r0': 'single, rMod 0', 'r1': 'single, rMod 1', 'r2': 'single, rMod 2', 'fusion': 'fusion: unmod 900 + rMod 2 1300 nm',
                    'swapped': 'swapped: unmod 1300 + rMod 2 900 nm'}
    for (chrom, band), curves in panels.items():
        ax, rx = _axes(fig, gs, int(chrom), 0 if band == 'vis' else 1)
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        picks = {g: {m: _best_replayed(curves, g, m, band, single) for m in MAGS} for g in ('single', 'fusion')}
        picks = {g: {m: p for m, p in d.items() if p is not None} for g, d in picks.items()}
        used = []
        for e2 in extra:              # other single stages (rMod 2): drawn dashed, with their comparison to the best fusion
            st2 = replays(next(c for e, _, c in curves if e == e2))
            m2 = [m for m in MAGS if complete(st2, m, k)]
            if m2:
                ax.errorbar(m2, [st2[m][k] for m in m2], yerr=[st2[m][k + '_std'] if np.isfinite(st2[m][k + '_std']) else 0.0 for m in m2],
                            fmt='none', ecolor=COLOR[e2], elinewidth=ELW * 0.8, capsize=CAP, zorder=2)
                ax.plot(m2, [st2[m][k] for m in m2], color=COLOR[e2], lw=LW * 0.85, ls='--', marker=MARKER[e2], ms=MS * 0.8, mec='white', mew=MEW, zorder=3,
                        label=T(f'single rMod {e2[1:]}'))
                b2 = [m for m in m2 if m in picks['fusion']]
                if b2:
                    _curve(rx, b2, [_cmp(quantity, replays(picks['fusion'][m][2])[m][k], st2[m][k]) for m in b2], None, COLOR[e2], MARKER[e2], None)
        for g, col in PROFILE_COLOR.items():
            mags = sorted(picks[g])
            if not mags:
                continue
            st = {m: replays(picks[g][m][2])[m] for m in mags}
            y = np.array([st[m][k] for m in mags])
            err = [st[m][k + '_std'] if np.isfinite(st[m][k + '_std']) else 0.0 for m in mags]
            ax.errorbar(mags, y, yerr=err, fmt='none', ecolor=col, elinewidth=ELW, capsize=CAP, zorder=3)
            ax.plot(mags, y, color=col, lw=LW, zorder=4, label=T(sname if g == 'single' else 'best fusion'))
            for m, yi in zip(mags, y):
                e = picks[g][m][0]
                hollow = quantity == 'contrast' and st[m][f'sr_{band}'] < SR_MASK
                ax.plot([m], [yi], ls='none', marker=MARKER[e], ms=MS, mfc='white' if hollow else col, mec=col if hollow else 'white',
                        mew=1.1 if hollow else MEW, zorder=5)
                if e not in used:
                    used.append(e)
        both = sorted(set(picks['single']) & set(picks['fusion']))
        if both:
            ratio = [_cmp(quantity, replays(picks['fusion'][m][2])[m][k], replays(picks['single'][m][2])[m][k]) for m in both]
            _curve(rx, both, ratio, None, PROFILE_COLOR['single'], 'o', None)
        if fl:
            _floor_lines(ax, chrom, band, fl)
        _style(ax, quantity == 'contrast')
        _style(rx, False)
        _labels(ax, rx, quantity, chrom, band, sname, _title(chrom, band))
        ax.set_ylim(*lim)
        handles, labels = ax.get_legend_handles_labels()
        for e in used:          # marker key: which fusion won at each point (the single stage is fixed: no key)
            if single != 'best' and e.startswith('r'):
                continue
            handles.append(plt.Line2D([], [], ls='none', marker=MARKER[e], ms=MS, mfc=MUTED, mec='white', mew=MEW))
            labels.append(T(entity_label[e]))
        ax.legend(handles, labels, **_LEG)
    who = 'the single stage (rMod 1 / 2) and the fusion' if single == 'best' else 'the fusion'
    notes = [f'at each magnitude {who} (fusion / swapped) with the lowest replay-mean 2-5 λ/D contrast '
             'in the panel band, among the cases where every replay converged; the marker shows the winning case (also for the SR: same runs)'
             + ('' if single == 'best' else f'; single stage: rMod {single[1:]} at every magnitude'),
             f"lower row: {'ratio' if quantity == 'contrast' else 'difference'} of the best fusion wrt {sname}"
             + ''.join(f" (red: {'ratio' if quantity == 'contrast' else 'difference'} wrt single rMod {e[1:]})"
                       for e in extra)]
    if quantity == 'contrast':
        notes.append(f'hollow markers: science SR < {SR_MASK}')
    fig.text(0.01, -0.01, T(_wrap('; '.join(notes))), fontsize=FS_NOTE, color=MUTED, ha='left', va='top')
    return fig
