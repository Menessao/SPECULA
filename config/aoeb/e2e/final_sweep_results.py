"""Results of the final e2e sweep (sweep_final.py, results/WFSW/sweep_final_20261108): replay statistics and the comparison figures.

Magnitudes 5 and 6 were not swept: they are replays of the configuration selected at magnitude 7. Each point of the single-stage and fusion
curves is the mean over the replays (REPLAY_TIME s, other atmosphere seed and random wind direction per
layer) of the configuration the sweep selected at that magnitude; error bars: standard deviation over the converged replays. A replay that ran
away is excluded and counted (`n/N` next to the point): non-finite, or 550 nm SR of its last 0.5 s below RUNAWAY x that of the 0.5 s after
START_TIME, or 550 nm SR below RUNAWAY_MEDIAN x the median of the other realizations (or of the converged replays of the next fainter
magnitude, when it uses the same configuration: a brighter star cannot do worse). Only the replays of the configuration selected now count
(replays of a configuration selected before stay in replays.csv). The equal-flux dual cases were not replayed: their points
are the sweep values (one realization, the one the search was optimised on), without error bars, and so are their reference curves.
Contrast: mean raw contrast in 2-5 lambda/D, relative to the integrated AO peak. Ratio panels: each curve over the best single-stage curve
(lowest mean contrast / highest mean SR of the rMods at that magnitude).
"""
import csv
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from results_analysis import load_coro_profile, load_perfect_coro

OUT = '/raid1/mmenessini/results/WFSW/sweep_final_20261108'
MAGS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0)    # 5 and 6: replays of the mag 7 configuration (not swept)
SWEPT_MAGS = MAGS[2:]
REF_MAG = 7.0
START_TIME, WINDOW, RUNAWAY, RUNAWAY_MEDIAN = 0.5, 0.5, 0.7, 0.5
SR_MASK = 0.25            # contrast points with the science SR below this are drawn hollow
WL = {'vis': '550 nm', 'nir': '1650 nm'}
RMODS = (0, 2)            # single-stage modulations shown (rMod 1 was swept and replayed too)
SPLITS = (10, 30, 50, 70, 90)        # % of the band on the unmodulated pyramid

# Colors follow the entity (reference categorical palette, fixed order); splits: ordinal blue ramp, light = little unmodulated flux
COLOR = {'fusion': '#2a78d6', 'swapped': '#eb6834', 'r0': '#1baf7a', 'r1': '#eda100', 'r2': '#e87ba4', 'r0svd': '#008300'}
MARKER = {'fusion': 'o', 'swapped': 's', 'r0': '^', 'r1': 'v', 'r2': 'D', 'r0svd': 'P'}
SPLIT_COLOR = dict(zip(SPLITS, ('#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281')))
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
            curves = [(f'r{r}', f'single, WFS {w} nm, rMod {r}', f'single_wfs{w}_{t}_r{r}') for r in RMODS]
            # full ML runs away at bright magnitudes: rMod 0 with the SVD reconstructor and Kalman model, replays at mags 5-7 only
            curves.append(('r0svd', f'single, WFS {w} nm, rMod 0, SVD rec + KF (mags 5-7)', f'single_wfs{w}_{t}_r0_svd'))
            curves.append(('fusion', 'fusion: unmod 900 nm + rMod 2 at 1300 nm', f'dual_unmod900_{t}'))
            if chrom:
                curves.append(('swapped', 'fusion swapped: unmod 1300 nm + rMod 2 at 900 nm', 'dual_unmod1300_chrom'))
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


def _floor_lines(ax, chrom, band, fl):
    if not chrom:
        refs = [('achrom', 'fitting floor', '--')]
    else:
        refs = [('vis', 'floor, WFS 900 nm', '--'), ('nir', 'floor, WFS 1300 nm', ':')]
    for c, label, ls in refs:
        below = c == 'vis' and chrom        # the two chromatic floors are close: one label under its line, one above
        ax.axhline(fl[c][band], color=MUTED, lw=1.2, ls=ls, zorder=1)
        ax.annotate(T(label), (1.0, fl[c][band]), xycoords=('axes fraction', 'data'), xytext=(-4, -3 if below else 3), textcoords='offset points',
                    ha='right', va='top' if below else 'bottom', fontsize=8, color=MUTED)


def _style(ax, logy, mags=MAGS):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(mags)


def _curve(ax, x, y, err, color, marker, label, hollow=None, counts=None):
    x, y = np.asarray(x), np.asarray(y)
    if err is not None:
        ax.errorbar(x, y, yerr=err, fmt='none', ecolor=color, elinewidth=1.4, capsize=3, zorder=3)
    ax.plot(x, y, color=color, lw=2, marker=marker, ms=8, mec='white', mew=1.5, zorder=4, label=T(label))
    hollow = np.zeros(len(x), bool) if hollow is None else np.asarray(hollow, bool)
    if hollow.any():     # science SR below SR_MASK: white face
        ax.plot(x[hollow], y[hollow], ls='none', marker=marker, ms=8, mfc='white', mec=color, mew=1.6, zorder=5)
    for xi, yi, c in zip(x, y, counts or []):
        if c:
            ax.annotate(T(c), (xi, yi), xytext=(9, 5), textcoords='offset points', ha='left', va='bottom', fontsize=8, color=INK)


def _figure(title):
    fig = plt.figure(figsize=(13, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=(3, 1.2, 3, 1.2))
    fig.suptitle(T(title), fontsize=13, color=INK)
    return fig, gs


def _axes(fig, gs, row, col):
    ax = fig.add_subplot(gs[2 * row, col])
    rx = fig.add_subplot(gs[2 * row + 1, col], sharex=ax)
    plt.setp(ax.get_xticklabels(), visible=False)
    return ax, rx


def _labels(ax, rx, quantity, chrom, band, ref_label, top_title):
    q = 'contrast 2-5 λ/D' if quantity == 'contrast' else 'SR'
    ax.set_ylabel(T(f'{q}, {WL[band]}'), color=INK)
    ax.set_title(T(top_title), fontsize=11, color=INK, loc='left')
    rx.set_ylabel(T(f'/ {ref_label}'), color=INK, fontsize=9)
    rx.set_xlabel(T('magnitude (900 nm)'), color=INK)
    rx.axhline(1.0, color=MUTED, lw=1, zorder=1)


def ylim(quantity):
    """Common y range of every contrast (or SR) panel of both figures: all replay means +- std, sweep values and, for the contrast, the floors."""
    import glob
    vals = []
    for d in glob.glob(op.join(OUT, '*', 'selected.csv')):
        case = op.basename(op.dirname(d))
        for b in ('vis', 'nir'):
            k = f'contrast_{b}' if quantity == 'contrast' else f'sr_{b}'
            vals += [float(r[k]) for r in selected(case).values()]
            for st in replays(case).values():
                if np.isfinite(st[k]):
                    e = st[k + '_std'] if np.isfinite(st[k + '_std']) else 0.0
                    vals += [st[k] - e, st[k] + e]
    if quantity == 'contrast':
        vals += [v for f in floors().values() for v in f.values()]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        return min(vals) / 1.3, max(vals) * 1.3
    vals = [v for v in vals if np.isfinite(v)]
    return max(0.0, min(vals) - 0.02), min(1.0, max(vals) + 0.02)


def _best_single(stats, cases, band, quantity, mags):
    """Best single-stage value per magnitude (lowest contrast / highest SR among the rMods)."""
    k = f'{quantity}_{band}' if quantity == 'contrast' else f'sr_{band}'
    pick = min if quantity == 'contrast' else max
    out = {}
    for m in mags:
        v = [stats[c][m][k] for c in cases if m in stats[c] and np.isfinite(stats[c][m][k])]
        if v:
            out[m] = pick(v)
    return out


def plot_main(quantity='contrast'):
    """Single (one curve per rMod), fusion and fusion swapped: replay mean +- std, achromatic (top) and chromatic (bottom), 550 / 1650 nm."""
    fl = floors() if quantity == 'contrast' else None
    panels = main_panels()
    stats = {c: replays(c) for curves in panels.values() for _, _, c in curves}
    fig, gs = _figure(f"{'Contrast' if quantity == 'contrast' else 'Strehl ratio'}: replay mean ± std of the selected configurations")
    missing, lim = set(), ylim(quantity)
    for (chrom, band), curves in panels.items():
        ax, rx = _axes(fig, gs, int(chrom), 0 if band == 'vis' else 1)
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        singles = [c for e, _, c in curves if e.startswith('r')]
        ref = _best_single(stats, singles, band, quantity, MAGS)
        for e, label, case in curves:
            st = stats[case]
            mags = [m for m in MAGS if m in st and np.isfinite(st[m][k])]
            if len(mags) < (3 if e == 'r0svd' else len(MAGS)):     # the SVD curve has mags 5-7 only
                missing.add(case)
            if not mags:
                continue
            y = [st[m][k] for m in mags]
            err = [st[m][k + '_std'] if np.isfinite(st[m][k + '_std']) else 0.0 for m in mags]
            hollow = [st[m][f'sr_{band}'] < SR_MASK for m in mags] if quantity == 'contrast' else None
            counts = [f"{st[m]['n_ok']}/{st[m]['n']}" if st[m]['n_ok'] < st[m]['n'] else '' for m in mags]
            _curve(ax, mags, y, err, COLOR[e], MARKER[e], label, hollow, counts)
            rm = [m for m in mags if m in ref]
            _curve(rx, rm, [st[m][k] / ref[m] for m in rm], None, COLOR[e], MARKER[e], None)
        if fl:
            _floor_lines(ax, chrom, band, fl)
        _style(ax, quantity == 'contrast')
        _style(rx, False)
        _labels(ax, rx, quantity, chrom, band, 'best single', f"{'chromatic' if chrom else 'achromatic'}, {WL[band]}"
                + (' (single: WFS 900 nm)' if band == 'nir' and not chrom else ''))
        ax.set_ylim(*lim)
        ax.legend(fontsize=8, loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    _footnote(fig, quantity, missing)
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
    +10 % in the chromatic cases, so mixing the two would bias the ratios); ratio panels over the best single stage."""
    fl = floors() if quantity == 'contrast' else None
    fig, gs = _figure(f"Equal-flux dual: {'contrast' if quantity == 'contrast' else 'Strehl ratio'} of each band split (sweep values)")
    missing, lim = set(), ylim(quantity)
    for (chrom, band), (w, stem, singles, fusion) in equal_flux_panels().items():
        ax, rx = _axes(fig, gs, int(chrom), 0 if band == 'vis' else 1)
        k = f'contrast_{band}' if quantity == 'contrast' else f'sr_{band}'
        stats = {c: _sweep_stats(c, band) for c in singles + [fusion]}
        ref = _best_single(stats, singles, band, quantity, MAGS)
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
            _curve(rx, rm, [float(sel[m][k]) / ref[m] for m in rm], None, SPLIT_COLOR[u], 'o', None)
        rm = sorted(ref)
        if rm:
            ax.plot(rm, [ref[m] for m in rm], color=MUTED, lw=2, ls='--', zorder=2, label=T(f'best single, WFS {w} nm'))
        st = stats[fusion]
        fm = [m for m in MAGS if m in st and np.isfinite(st[m][k])]
        if fm and w == 900:
            ax.plot(fm, [st[m][k] for m in fm], color=INK, lw=1.5, ls='-.', zorder=2, label=T('fusion: unmod 900 + rMod 2 at 1300 nm'))
        if fl:
            _floor_lines(ax, chrom, band, fl)
        _style(ax, quantity == 'contrast', SWEPT_MAGS)
        _style(rx, False, SWEPT_MAGS)
        _labels(ax, rx, quantity, chrom, band, 'best single', f"{'chromatic' if chrom else 'achromatic'}, both pyramids at {w} nm, {WL[band]}")
        ax.set_ylim(*lim)
        ax.legend(fontsize=8, loc='lower left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    _footnote(fig, quantity, missing, equal_flux=True)
    return fig


def _footnote(fig, quantity, missing, equal_flux=False):
    notes = []
    if quantity == 'contrast':
        notes.append(f'hollow markers: science SR < {SR_MASK}')
    if equal_flux:
        notes.append('all curves: sweep values on the nominal realization (splits not replayed), no error bars')
    else:
        notes.append('n/N: converged replays (runaways excluded)')
    if missing:
        notes.append('incomplete: ' + ', '.join(sorted(missing)))
    fig.text(0.01, -0.01, T('; '.join(notes)), fontsize=8, color=MUTED, ha='left', va='top')


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


def plot_profiles(mag=7.0, rmax=30.0):
    """Radial raw-contrast profiles at one magnitude, achromatic (top) and chromatic (bottom), 550 / 1650 nm: single stage (rMod 0 and 2), fusion,
    fusion swapped, mean over the converged replays (band: +- std), with the perfect-coronagraph bounds: fitting (achromatic) and fitting +
    chromatism (WFS at 900 nm, WFS at 1300 nm). Same y range in every panel; shaded: the 2-5 lambda/D of the metric."""
    refs = {c: load_perfect_coro(c) for c in ('achrom', 'vis', 'nir')}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True, sharex=True)
    fig.suptitle(T(f'Raw contrast profiles at magnitude {mag:g} (900 nm): replay mean ± std'), fontsize=13, color=INK)
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
            ax.plot(rad[sel], refs[c][bi][sel], color=MUTED, lw=1.5, ls=ls, zorder=2, label=T(label))
            lo = min(lo, np.min(refs[c][bi][sel & (rad >= 1)]))
        for e, label, case in curves:
            pr = profile(case, mag, band)
            if pr is None:
                notes.add(case)
                continue
            r, m, sd, how = pr
            k = r <= rmax
            ax.plot(r[k], m[k], color=COLOR[e], lw=2, zorder=4, label=T(f'{label} ({how})'))
            if sd is not None:
                ax.fill_between(r[k], np.maximum(m[k] - sd[k], 1e-12), m[k] + sd[k], color=COLOR[e], alpha=0.2, lw=0, zorder=3)
            if how == 'sweep run':
                notes.add(f'{case} (no replays yet)')
            hi = max(hi, np.max(m[k & (r >= 1)]))
            lo = min(lo, np.min(m[k & (r >= 1)]))
        _style(ax, True)
        ax.set_xticks(np.arange(0, rmax + 0.1, 5))
        ax.set_xlim(0, rmax)
        ax.set_title(T(f"{'chromatic' if chrom else 'achromatic'}, {WL[band]}" + (' (single: WFS 900 nm)' if band == 'nir' and not chrom else '')),
                     fontsize=11, color=INK, loc='left')
        ax.set_ylabel(T(f'raw contrast, {WL[band]}'), color=INK)
        if chrom:
            ax.set_xlabel(T('separation [λ/D]'), color=INK)
        ax.legend(fontsize=8, loc='lower right', frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
    for ax in axes.ravel():
        ax.set_ylim(lo / 60, hi * 1.5)     # room below the curves for the legend
    note = 'shaded column: 2-5 λ/D (metric); bands: ± std over the converged replays'
    if notes:
        note += '; incomplete: ' + ', '.join(sorted(notes))
    fig.text(0.01, -0.01, T(note), fontsize=8, color=MUTED, ha='left', va='top')
    return fig
