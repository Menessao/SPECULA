"""Talk: case combinations (2026-11-12). Case 2+3 contrast curves (EB and E2E) and the SR / contrast tables of the single 48x48 rMod 2 and
of the fused cases 1, 2, 3, 2+3, 1+2+3, VIS and NIR, mags 5 / 8 / 12 (E2E: 7 / 8 / 12, mag 5 was not swept).

Read-only: EB sweeps from results/multiwfs_cache (computed by results/WFSW/talk_cases_20261112/eb_sweeps.py, the definitions of the
'Talk: case combinations' cell of analyse_errorbudget.ipynb), E2E sweep values (nominal realization, selected.csv) of
results/WFSW/sweep_final_20261108 and results/WFSW/talk_cases_20261112. A missing sweep gives NaN entries, never a computation.

In wfs_talk_figures.ipynb (its setup; slide 16b and 19 cells call the functions) or standalone from config/: python talk_case_combos.py
(figures, the 6 tables, slide 19).
"""
import json
import os
import os.path as op

_STANDALONE = 'st' not in globals()             # in the notebook: only the definitions (its cells call them)
if _STANDALONE:                                  # standalone: the setup and helper cells of the talk notebook, then everything
    import matplotlib
    matplotlib.use('Agg')
    _nb = json.load(open(op.join(op.dirname(op.abspath(__file__)), 'wfs_talk_figures.ipynb')))
    def display(*a, **k):
        pass
    _src = [''.join(c['source']) for c in _nb['cells']]
    exec(_src[1], globals())
    exec(_src[3], globals())
    exec(_src[7][:_src[7].index('K12 =')], globals())             # single_stage, stage_title, coro_terms
    exec(_src[11], globals())                                     # case_figure, fitting_limit, joint_contrast

import csv as _csv
import warnings
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

E2E_FINAL = '/raid1/mmenessini/results/WFSW/sweep_final_20261108'
E2E_TALK = '/raid1/mmenessini/results/WFSW/talk_cases_20261112'
N_LO, TOL = 100, 0.20
PIXEL_SPLIT = 1 / (1 + (48 / 14) ** 2)          # flux share of the 14x14 at equal photons per subaperture (option of FRAC)
FRAC_CMB = [0.02, 0.05, 1/(1+(48/14)**2), 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NS_UNMOD = (14, 24, 36, 48)
MAGS2 = np.arange(4.0, 15.01, 1.0)
TABLE_MAGS = (5.0, 8.0, 12.0)
E2E_MAG = {5.0: 7.0, 8.0: 8.0, 12.0: 12.0}       # E2E sweeps start at mag 7
OTHER = {900.: 1300., 1300.: 900.}


REF = 'single 48x48 rMod 2 (best binning, rate)'                    # label of the reference row of the tables (rMod 2)


def ref_label(r=2):
    return f'single 48x48 rMod {r:g} (best binning, rate)'


def eb_sweeps(band, chromatic=True, ref_rmod=2):
    """(label, cache name, mags) of the band, every system at rMod 2 (the 'unmodulated' ones at rMod 0). chromatic=False: the
    achromatic counterparts (results/WFSW/talk_cases_20261112/eb_sweeps_achrom.py; reference and combined: the case 5 sweeps).
    The first entry is the reference: best single 48x48 rMod ref_rmod (rMod 1: eb_sweeps_rmod1.py)."""
    set_band(band)
    w, f = f'{wfs_nm:.0f}', FRAC_CMB
    if not chromatic:
        return [(ref_label(ref_rmod), f'c5 single rMod{ref_rmod:g} {w}', MAGS),
                ('1: two 48x48, 100% flux each', f'c1 double rMod2 {w} achrom', MAGS),
                ('2: 48x48 + 14x14, 100% flux each', f'c2 double rMod2 {w} N{N_LO} achrom', MAGS2),
                ('2: 48x48 + 14x14, optimal split', f'c2 split rMod2 {w} N{N_LO} tol{TOL:g} {f} achrom', MAGS2),
                ('3: 48x48 rMod 2 + 48x48 rMod 0, 50/50', f'c3 split rMod2+0 {w} achrom', MAGS),
                ('2+3: 48x48 rMod 2 + unmod n x n, optimal split', f'c23 split rMod2+0 {w} n{NS_UNMOD} {f} achrom', MAGS),
                ('1+2+3: mod 900 + unmod 1300, 100% each', f'c5 mod 900 + unmod 1300 rMod2 n{NS_UNMOD}', MAGS),
                ('1+2+3: unmod 900 + mod 1300, 100% each', f'c5 unmod 900 + mod 1300 rMod2 n{NS_UNMOD}', MAGS)]
    return [(ref_label(ref_rmod), f'c1 single rMod{ref_rmod:g} {w}', MAGS),
            ('1: two 48x48, 100% flux each', f'c1 double rMod2 {w}', MAGS),
            ('2: 48x48 + 14x14, 100% flux each', f'c2 double rMod2 {w} N{N_LO}', MAGS2),
            ('2: 48x48 + 14x14, optimal split', f'c2 split rMod2 {w} N{N_LO} tol{TOL:g} {f}', MAGS2),
            ('3: 48x48 rMod 2 + 48x48 rMod 0, 50/50', f'c3 split rMod2+0 {w}', MAGS),
            ('2+3: 48x48 rMod 2 + unmod n x n, optimal split', f'c23 split rMod2+0 {w} n{NS_UNMOD} {f}', MAGS),
            ('1+2+3: mod 900 + unmod 1300, 100% each', f'c5 chrom mod 900 + unmod 1300 rMod2 n{NS_UNMOD}', MAGS),
            ('1+2+3: unmod 900 + mod 1300, 100% each', f'c5 chrom unmod 900 + mod 1300 rMod2 n{NS_UNMOD}', MAGS)]


def _pct(f):
    return f'{100 * f:.3g}%'


def _sensor(rmod, n, fs, flux, wl=None, modes=None):
    return (f'rMod {rmod:g}, {n:.0f}x{n:.0f}' + (f' ({modes} modes)' if modes else '') + (f' at {wl:.0f} nm' if wl else '')
            + f', {fs:.0f} Hz, {flux} flux')


def eb_config(label, row):
    """Explicit configuration of an EB row: 'LO: ...; HO: ...' for the dual systems (LO: the sensor fused on its modes at the LO rate,
    HO: the 48x48 rMod 2 at its own rate), 'WFS 1: ...; WFS 2: ...' for the one-loop fusions (cases 1 and 3), the single as one sensor."""
    if not isinstance(row, pd.Series):
        return ''
    g = lambda k: row.get(k, np.nan)
    if label.startswith('single'):
        return _sensor(float(label.split('rMod ')[1].split()[0]), g('n_subap'), g('fs'), '100%')
    if label.startswith('1:'):
        return '; '.join(f'WFS {i}: ' + _sensor(R1, 48, g('fs'), '100%') for i in (1, 2))
    if label.startswith('3:'):
        return f"WFS 1: {_sensor(R1, 48, g('fs'), '50%')}; WFS 2: {_sensor(0, 48, g('fs'), '50%')}"
    if label.startswith('2:'):
        f = g('f_LO')
        lo, ho = (_pct(f), _pct(1 - f)) if np.isfinite(f) else ('100%', '100%')
        return f"LO: {_sensor(R1, 14, g('fs_lo'), lo, modes=N_LO)}; HO: {_sensor(R1, 48, g('fs'), ho)}"
    if label.startswith('2+3:'):
        f = g('f_LO')
        return f"LO: {_sensor(0, g('n_unmod'), g('fs_lo'), _pct(f))}; HO: {_sensor(R1, 48, g('fs'), _pct(1 - f))}"
    if label.startswith('1+2+3:'):
        wm, wu = (900., 1300.) if 'mod 900 + unmod 1300' in label else (1300., 900.)
        return f"LO: {_sensor(0, g('n_unmod'), g('fs_lo'), '100%', wl=wu)}; HO: {_sensor(R1, 48, g('fs'), '100%', wl=wm)}"
    return ''


_E2E_DUAL = re.compile(r'unmod (?P<uw>[\d.]+) nm (?P<uf>[\d.]+)% band (?P<n>\d+)x\d+ (?P<N>\d+) modes (?P<ufs>\d+) Hz '
                       r'(?:ML|rMod (?P<ur>[\d.]+) SVD), mod (?P<mw>[\d.]+) nm (?P<mf>[\d.]+)% band rMod (?P<mr>[\d.]+) (?P<mfs>\d+) Hz')
_E2E_SINGLE = re.compile(r'rMod (?P<r>[\d.]+), (?P<n>\d+)x\d+, (?P<N>\d+) modes, (?P<fs>\d+) Hz')


def e2e_config(label, text):
    """The selected.csv configuration of an E2E run written as the EB ones (LO = the 'unmodulated' chain of sweep_dual.py, which is
    rMod 2 in cases 1 and 2)."""
    if not text:
        return ''
    m1 = _E2E_SINGLE.match(text)
    if m1:
        return _sensor(float(m1['r']), int(m1['n']), float(m1['fs']), '100%')
    m2 = _E2E_DUAL.search(text)
    if not m2:
        return text
    two_bands = m2['uw'] != m2['mw']
    split = label.startswith('2: 48x48 + 14x14, split')            # 7.84 %, printed rounded in selected.csv
    lo = _sensor(float(m2['ur'] or 0), int(m2['n']), float(m2['ufs']), '7.84%' if split else f"{float(m2['uf']):g}%",
                 wl=float(m2['uw']) if two_bands else None, modes=int(m2['N']) if label.startswith('2:') else None)
    ho = _sensor(float(m2['mr']), 48, float(m2['mfs']), '92.2%' if split else f"{float(m2['mf']):g}%", wl=float(m2['mw']) if two_bands else None)
    one_loop = label.startswith(('1:', '3:'))
    return f"{'WFS 1' if one_loop else 'LO'}: {lo}; {'WFS 2' if one_loop else 'HO'}: {ho}"


def _selected(folder):
    p = op.join(folder, 'selected.csv')
    if not op.exists(p):
        return {}
    with open(p) as fh:
        return {float(r['mag']): r for r in _csv.DictReader(fh)}


def _score(c, sr):
    return (0, c) if sr >= 0.1 else (1, -sr)


def e2e_rows(band):
    """label -> {mag: (SR, contrast, config)} of the E2E sweep values (nominal realization) of the band."""
    W = 900 if band == 'VIS' else 1300
    b = 'vis' if band == 'VIS' else 'nir'
    get = lambda r: (float(r[f'sr_{b}']), float(r[f'contrast_{b}']), r['config'])
    out = {}
    src = [('single 48x48 rMod 2 (best binning, rate)', op.join(E2E_FINAL, f'single_wfs{W}_chrom_r2')),
           ('1: two 48x48, 100% flux each', op.join(E2E_TALK, f'c1_{W}')),
           ('2: 48x48 + 14x14, 100% flux each', op.join(E2E_TALK, f'c2d_{W}')),
           ('2: 48x48 + 14x14, split 7.8% (E2E: fixed)', op.join(E2E_TALK, f'c2s_{W}')),
           ('3: 48x48 rMod 2 + 48x48 rMod 0, 50/50', op.join(E2E_TALK, f'c3_{W}')),
           ('1+2+3: mod 900 + unmod 1300, 100% each', op.join(E2E_FINAL, 'dual_unmod1300_chrom')),
           ('1+2+3: unmod 900 + mod 1300, 100% each', op.join(E2E_FINAL, 'dual_unmod900_chrom'))]
    for lab, folder in src:
        out[lab] = {m: get(r) for m, r in _selected(folder).items()}
    best = {}                                                    # 2+3: best band split per magnitude
    for u in (10, 30, 50, 70, 90):
        for m, r in _selected(op.join(E2E_FINAL, f'dual_eq{W}_chrom_u{u}')).items():
            sr, c, cfg = get(r)
            if m not in best or _score(c, sr) < _score(best[m][1], best[m][0]):
                best[m] = (sr, c, cfg)
    out['2+3: 48x48 rMod 2 + unmod n x n, optimal split'] = best
    return out


def _e2e_df(d):
    """{mag: (SR, contrast, config)} -> DataFrame (index mag; SR, contrast) for plotting.fusion_gain."""
    return pd.DataFrame({'SR': {m: v[0] for m, v in d.items()}, 'contrast': {m: v[1] for m, v in d.items()}}).sort_index()


def _dmag(ref, fused, mg):
    """Magnitudes gained at equal contrast at mg (plotting.fusion_gain): NaN outside the fused sweep's contrast range."""
    if ref is None or fused is None or mg not in ref.index or len(fused) < 2:
        return np.nan
    return float(pl.fusion_gain(ref, fused, st.sr_min)['dmag'].get(mg, np.nan))


def tables():
    rows = []
    for band in BANDS:
        eb = {lab: sweep(name, mags) for lab, name, mags in eb_sweeps(band)}
        e2 = e2e_rows(band)
        ref_eb, ref_e2 = eb[next(iter(eb))], e2['single 48x48 rMod 2 (best binning, rate)']
        ref_e2df = _e2e_df(ref_e2) if ref_e2 else None
        labels = [lab for lab, _, _ in eb_sweeps(band)]
        labels.insert(4, '2: 48x48 + 14x14, split 7.8% (E2E: fixed)')
        for mg in TABLE_MAGS:
            for lab in labels:
                d = eb.get(lab)
                r = d.loc[mg] if d is not None and mg in d.index else None
                r0 = ref_eb.loc[mg] if ref_eb is not None and mg in ref_eb.index else None
                me = E2E_MAG[mg]
                e, e0 = e2.get(lab, {}).get(me), ref_e2.get(me)
                rows.append(dict(band=band, mag=mg, system=lab,
                                 EB_SR=np.nan if r is None else r['SR'], EB_contrast=np.nan if r is None else r['contrast'],
                                 EB_ratio=np.nan if r is None or r0 is None else r['contrast'] / r0['contrast'],
                                 EB_dmag=np.nan if lab.startswith('single') else _dmag(ref_eb, d, mg), EB_config=eb_config(lab, r),
                                 E2E_mag=me, E2E_SR=np.nan if e is None else e[0], E2E_contrast=np.nan if e is None else e[1],
                                 E2E_ratio=np.nan if e is None or e0 is None else e[1] / e0[1],
                                 E2E_dmag=np.nan if lab.startswith('single') or not e2.get(lab) else _dmag(ref_e2df, _e2e_df(e2[lab]), me),
                                 E2E_config='' if e is None else e2e_config(lab, e[2])))
    return pd.DataFrame(rows)


def write_tables(df):
    """One table per band and magnitude (6): talk_case_table_<band>_mag<m>_<CONTROLLER>.md / .csv; shown one by one in the notebook."""
    f = lambda x, fmt: '-' if not np.isfinite(x) else format(x, fmt)
    note = (f'Science SR and mean raw contrast {ZONE[0]}-{ZONE[1]} lambda/D (perfect coronagraph), z = 30 deg, seeing 0.9", chromatic. '
            'Ratio = contrast / single 48x48 rMod 2 (< 1: better); every modulated sensor rMod 2. EB: leaky, criterion contrast. E2E: '
            'sweep values of the selected configuration (one nominal realization, no replays); E2E sweeps start at mag 7. E2E case 2 '
            'split: fixed 7.84 % to the 14x14 (EB: optimised). Configurations: LO = the sensor fused on its modes at the LO rate, HO = '
            'the 48x48 rMod 2 at its own rate; WFS 1 / WFS 2: one-loop fusions (cases 1 and 3, one rate); flux: share of the band\'s light '
            '(100 % on both: flux added). 1+2+3: two bands, each sensor with its band\'s light. dmag: magnitudes '
            'gained at equal contrast (how much fainter the fused system reaches the single\'s contrast at this magnitude; log-linear '
            'interpolation of the fused sweep, "-" beyond it: EB sweeps end at mag 14 or 15, E2E at 12). "-": not available.')
    cols = ['system', 'EB_SR', 'EB_contrast', 'EB_ratio', 'EB_dmag', 'EB_config', 'E2E_SR', 'E2E_contrast', 'E2E_ratio', 'E2E_dmag', 'E2E_config']
    out = {}
    for (band, mg), g in df.groupby(['band', 'mag'], sort=False):
        stem = op.join(OUT, f'talk_case_table_{band}_mag{mg:g}_{CONTROLLER}')
        g[cols].to_csv(stem + '.csv', index=False)
        title = f'{band} ({S[BANDS[band]]["science_nm"]:.0f} nm), mag {mg:g} (E2E: mag {E2E_MAG[mg]:g})'
        lines = [f'# {title}', '', note, '',
                 '| system | EB SR | EB contrast | EB ratio | EB dmag | EB config | E2E SR | E2E contrast | E2E ratio | E2E dmag | E2E config |',
                 '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for _, r in g.iterrows():
            lines.append(f"| {r['system']} | {f(r['EB_SR'], '.3f')} | {f(r['EB_contrast'], '.2e')} | {f(r['EB_ratio'], '.2f')} | {f(r['EB_dmag'], '+.1f')} | "
                         f"{r['EB_config']} | {f(r['E2E_SR'], '.3f')} | {f(r['E2E_contrast'], '.2e')} | {f(r['E2E_ratio'], '.2f')} | "
                         f"{f(r['E2E_dmag'], '+.1f')} | {r['E2E_config']} |")
        with open(stem + '.md', 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        t = g[cols].set_index('system')
        out[title] = t
        print(title)
        display(t.style.format({'EB_SR': '{:.3f}', 'E2E_SR': '{:.3f}', 'EB_contrast': '{:.2e}', 'E2E_contrast': '{:.2e}',
                                'EB_ratio': '{:.2f}', 'E2E_ratio': '{:.2f}', 'EB_dmag': '{:+.1f}', 'E2E_dmag': '{:+.1f}'}, na_rep='-').set_caption(title))
    for old in (f'talk_case_tables_{CONTROLLER}.md', f'talk_case_tables_{CONTROLLER}.csv'):       # the single-file version (2026-11-12)
        if op.exists(op.join(OUT, old)):
            os.remove(op.join(OUT, old))
    return out


COMBINED_BEST = '1+2+3: better of the two arrangements'
SUMMARY_KEYS = {   # (advantage, variant, main): system label of eb_sweeps (reference: the best single rMod 2 of the band)
    ('1. more photons', 'double flux', True): '1: two 48x48, 100% flux each',
    ('2. fast LO sensor', 'optimal split', True): '2: 48x48 + 14x14, optimal split',
    ('3. unmodulated sensitivity', '50/50 split', True): '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50',
    ('2+3. unmodulated + sampling', 'optimal split', True): '2+3: 48x48 rMod 2 + unmod n x n, optimal split',
    ('Combined = 1+2+3', 'better arrangement', True): COMBINED_BEST}
SUMMARY_COLOURS = {'1. more photons': CFG['case1'], '2. fast LO sensor': CFG['case2'], '3. unmodulated sensitivity': CFG['case3'],
                   '2+3. unmodulated + sampling': 'C5', 'Combined = 1+2+3': CFG['combined']}


def combined_best(eb):
    """Per magnitude, the better of the two 1+2+3 arrangements (mod 900 + unmod 1300, unmod 900 + mod 1300) by the selection rule of the
    sweeps (lowest contrast among SR >= 10 %, else highest SR); column 'arrangement'. None if one is missing."""
    a, b = eb.get('1+2+3: mod 900 + unmod 1300, 100% each'), eb.get('1+2+3: unmod 900 + mod 1300, 100% each')
    if a is None or b is None:
        return None
    pick = [_score(a.loc[m, 'contrast'], a.loc[m, 'SR']) <= _score(b.loc[m, 'contrast'], b.loc[m, 'SR']) for m in a.index]
    out = pd.DataFrame([a.loc[m] if p else b.loc[m] for m, p in zip(a.index, pick)], index=a.index)
    out['arrangement'] = ['mod 900 + unmod 1300' if p else 'unmod 900 + mod 1300' for p in pick]
    return out


def summary_slide19(chromatic=True, ref_rmod=2, mags=(5.0, 12.0)):
    """Slide 19 as the Summary of analyse_errorbudget.ipynb (plotting.plot_fusion_gains: contrast ratio top, science SR DIFFERENCE
    fused - single bottom, VIS / NIR columns, one colour per advantage). Shown: case 1 (double flux), case 2 (optimal split only), case 3,
    case 2+3 and the combined system (= 1+2+3, the better of its two arrangements at each magnitude, combined_best); every curve against the best single 48x48 rMod 2 of the band (NIR cases 1 and 2 at rMod 2, not the rMod 0 of
    slides 14-15); ref_rmod=1: against the best single rMod 1 (the fusions unchanged; files *_ref_rmod1_*). chromatic=False: the
    achromatic sweeps (slide19_summary_achrom_*). Also slide19_ratios[_achrom]_<CONTROLLER>.csv
    (contrast ratio and SR difference at mags 5, 8, 12); returns the Summary's highlight table (ratio, magnitudes gained, 6 / 9 / 12)."""
    tag = ('' if chromatic else '_achrom') + ('' if ref_rmod == 2 else f'_ref_rmod{ref_rmod:g}')
    gains, sr_gains, absent = {}, {}, []
    for band in BANDS:
        eb = {lab: sweep(name, ms) for lab, name, ms in eb_sweeps(band, chromatic, ref_rmod)}
        ref = eb[ref_label(ref_rmod)]
        eb[COMBINED_BEST] = combined_best(eb)
        for k, lab in SUMMARY_KEYS.items():
            if ref is None or eb.get(lab) is None:
                absent.append(f'{k[0]}: {k[1]} {band}')
                continue
            gains.setdefault(k, {})[band] = pl.fusion_gain(ref, eb[lab], st.sr_min)
            g = pl.fusion_gain(ref, eb[lab], st.sr_min, col='SR')
            g['ratio'] = eb[lab]['SR'].reindex(ref.index).to_numpy(float) - ref['SR'].to_numpy(float)     # SR difference
            sr_gains.setdefault(k, {})[band] = g
    if not gains:
        print(f'slide 19{tag}: no sweeps')
        return None
    sel = lambda g: g[(g.index >= mags[0]) & (g.index <= mags[1])]['ratio']
    def lim(src, lo0, hi0, pad):
        v = np.concatenate([sel(g).to_numpy(float) for d in src.values() for g in d.values()])
        v = v[np.isfinite(v)]
        return (min(lo0, v.min() - pad), max(hi0, v.max() + pad)) if v.size else (lo0, hi0)
    c_lim = lim(gains, 0.3, 1.1, 0.03)
    fig = pl.plot_fusion_gains(gains, zone=ZONE, mags=mags, gate=False, sr_gains=sr_gains, ylim=(c_lim, lim(sr_gains, -0.02, 0.02, 0.01)),
                               colours=SUMMARY_COLOURS, title=f'Sensor fusion against the best single 48x48 rMod {ref_rmod:g}, z = {zenith:g}°, '
                               + ('chromatic' if chromatic else 'achromatic (no chromatic term)'), sr_ref=0.0)
    axs = np.array(fig.axes).reshape(2, len(BANDS))
    for j, band in enumerate(BANDS):
        axs[0, j].set_title(band_title(band))
        axs[1, j].set_xlabel('magnitude at 900 nm')
    axs[0, 0].set_ylabel(f'contrast {ZONE_TXT}\nfused / single')
    axs[1, 0].set_ylabel('science SR\nfused $-$ single')
    for a in axs[:, 1]:
        a.set_ylabel('')
    save(fig, 19, f'summary{tag}_{CONTROLLER}', f'results/multiwfs_cache: c1/c2/c3/c23/c5 sweeps{" (achromatic)" if not chromatic else ""} '
                                               f'against the best single rMod {ref_rmod:g} ({CONTROLLER}); aoeb.plotting.fusion_gain / plot_fusion_gains')
    rows = [dict(advantage=a, variant=v, band=b, quantity=q, mag=mg, value=src[(a, v, mn)][b]['ratio'].get(mg, np.nan))
            for src, q in ((gains, 'contrast ratio'), (sr_gains, 'SR difference')) for (a, v, mn), d in src.items() for b in d for mg in (5.0, 8.0, 12.0)]
    tab = pd.DataFrame(rows).pivot_table(index=['advantage', 'variant', 'band', 'quantity'], columns='mag', values='value', sort=False)
    tab.round(3).to_csv(op.join(OUT, f'slide19_ratios{tag}_{CONTROLLER}.csv'))
    if absent:
        missing(19, f'summary{tag}', 'partially produced', ', '.join(absent) + f' ({CONTROLLER})', 'the EB sweeps of those cases',
                'eb_sweeps.py / eb_sweeps_achrom.py')
    hl = pd.concat({(b, f'{a}: {v}'): g[b].loc[[6.0, 9.0, 12.0], ['ratio', 'dmag']].unstack() for (a, v, _), g in gains.items() for b in g},
                   axis=1).T
    return hl.style.format('{:.2f}', subset=[c for c in hl.columns if c[0] == 'ratio']).format('{:+.1f}', subset=[c for c in hl.columns if c[0] == 'dmag'])


def eb_case23_figures():
    for band in BANDS:
        s = dict((lab, sweep(name, mags)) for lab, name, mags in eb_sweeps(band))
        set_band(band)
        keys = ['single 48x48 rMod 2 (best binning, rate)', '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50', '2: 48x48 + 14x14, optimal split',
                '2+3: 48x48 rMod 2 + unmod n x n, optimal split']
        if any(s[k] is None for k in keys):
            print(f'EB case 2+3 {band}: missing', [k for k in keys if s[k] is None])
            continue
        curves = [('best single rMod 2, 100% flux', s[keys[0]], CFG['single'], EQ, 'o'),
                  ('case 2: 48x48 + 14x14 rMod 2, optimal split', s[keys[2]], CFG['case2'], EQ, 's'),
                  ('case 3: rMod 2 + rMod 0 (48x48), 50/50', s[keys[1]], CFG['case3'], EQ, 'o'),
                  ('case 2+3: rMod 2 48x48 + unmod. n x n, optimal split', s[keys[3]], 'C5', EQ, 'D')]
        title = f'Case 2+3, {band_title(band)}, z = {zenith:g}°: 48x48 rMod 2 + unmodulated (binning, split, rates optimised)'
        try:
            fig = case_figure(curves, curves[0][0], title, fit=fitting_limit(band), **CASE_KW)
        except AssertionError:
            set_band(band)
            fig = case_figure(curves, curves[0][0], title, **CASE_KW)
        b = fig.axes[1]
        q = max(float(np.nanmax(df['contrast'] / s[keys[0]]['contrast'].reindex(df.index))) for _, df, *_ in curves[1:])
        b.set_ylim(RATIO_YLIM_LIN[0], max(RATIO_YLIM_LIN[1], 1.05 * q))
        save(fig, 16, f'case23_{band}_{CONTROLLER}', f'results/multiwfs_cache: c1 single rMod2, c2 split rMod2, c3 split, c23 split ({CONTROLLER})')
        sp = s[keys[3]][s[keys[3]].index <= CASE_MAG_MAX + 1e-9]
        fig, ax = plt.subplots(1, 2, figsize=FULL)
        ax[0].plot(sp.index, 100 * sp['f_LO'], 'D-', color='C5')
        ax[0].axhline(100 * PIXEL_SPLIT, color='0.5', ls=':', lw=1, label='1/(1+(48/14)$^2$)')
        ax[0].set(xlabel='magnitude at 900 nm', ylabel='flux on the unmodulated [%]', ylim=(0, 100), title=f'Case 2+3, {band}: optimal split')
        ax[0].legend()
        ax[1].plot(sp.index, sp['n_unmod'], 'D-', color='C5')
        ax[1].set(xlabel='magnitude at 900 nm', ylabel='unmodulated binning', yticks=NS_UNMOD, title=f'Case 2+3, {band}: unmodulated sampling')
        fig.tight_layout()
        save(fig, 16, f'case23_config_{band}_{CONTROLLER}', f'results/multiwfs_cache: c23 split (columns f_LO, n_unmod)')


def e2e_case23_figure():
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex=True, gridspec_kw=dict(height_ratios=(2.2, 1)))
    for j, band in enumerate(BANDS):
        e2 = e2e_rows(band)
        ref = e2['single 48x48 rMod 2 (best binning, rate)']
        for lab, col, mk in (('single 48x48 rMod 2 (best binning, rate)', CFG['single'], 'o'), ('2: 48x48 + 14x14, split 7.8% (E2E: fixed)', CFG['case2'], 's'),
                             ('3: 48x48 rMod 2 + 48x48 rMod 0, 50/50', CFG['case3'], 'o'), ('2+3: 48x48 rMod 2 + unmod n x n, optimal split', 'C5', 'D')):
            d = e2.get(lab, {})
            mg = sorted(d)
            if not mg:
                continue
            axs[0, j].semilogy(mg, [d[m][1] for m in mg], '-', color=col, marker=mk, label=lab)
            if lab != 'single 48x48 rMod 2 (best binning, rate)':
                both = [m for m in mg if m in ref]
                axs[1, j].plot(both, [d[m][1] / ref[m][1] for m in both], '-', color=col, marker=mk)
        axs[0, j].set_title(f'E2E, {band_title(band)}, z = {zenith:g}°, chromatic')
        axs[1, j].axhline(1, color='gray', lw=1)
        axs[1, j].set_xlabel('magnitude at 900 nm')
    axs[0, 0].set_ylabel(f'raw contrast {ZONE_TXT}')
    axs[1, 0].set_ylabel('contrast / single')
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=12)
    save(fig, 16, 'case23_e2e', 'E2E sweep values (nominal realization): sweep_final_20261108 single_wfs*_chrom_r2, dual_eq*_chrom_u*; '
                                'talk_cases_20261112 c2s_*, c3_*')


def reference_single_figure(ref_rmod=2, mags=(4.0, 14.0)):
    """The reference of slide 19: best single 48x48 rMod ref_rmod (binning 48/36/24/14 and frame rate chosen for the lowest 2-5 lambda/D
    contrast among SR >= 10 %), contrast (top) and science SR (bottom) vs magnitude, VIS / NIR, chromatic (solid) and achromatic
    (dashed); labels: the binning where it is not 48x48."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for j, band in enumerate(BANDS):
        for chromatic, ls, dy in ((True, '-', 7), (False, '--', -9)):
            d = sweep(eb_sweeps(band, chromatic, ref_rmod)[0][1], MAGS)
            if d is None:
                continue
            d = d[(d.index >= mags[0]) & (d.index <= mags[1])]
            lab = 'chromatic' if chromatic else 'achromatic'
            axs[0, j].semilogy(d.index, d['contrast'], ls, color=CFG['single'], marker='o', ms=4, label=lab)
            axs[1, j].plot(d.index, d['SR'], ls, color=CFG['single'], marker='o', ms=4, label=lab)
            for mg, r in d.iterrows():
                if r['n_subap'] != 48:
                    axs[0, j].annotate(f"{r['n_subap']:.0f}x{r['n_subap']:.0f}", (mg, r['contrast']), xytext=(0, dy), textcoords='offset points',
                                       ha='center', va='bottom' if dy > 0 else 'top', fontsize=9, color='0.25' if chromatic else '0.5')
        axs[1, j].axhline(0.10, color='gray', ls=':', lw=1, label='SR 10 %: selection threshold')
        axs[0, j].set_title(band_title(band))
        axs[0, j].set_ylim(2e-6, 2e-3)
        axs[1, j].set_ylim(0, 1)
        axs[1, j].set_xlabel('magnitude at 900 nm')
    axs[0, 0].set_ylabel(f'raw contrast {ZONE_TXT}')
    axs[1, 0].set_ylabel('science SR')
    h, l = axs[1, 0].get_legend_handles_labels()
    fig.suptitle(f'Reference: best single rMod {ref_rmod:g} (binning, frame rate), z = {zenith:g}°; labels: binning where not 48x48')
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=12)
    name = 'reference_single' + ('' if ref_rmod == 2 else f'_rmod{ref_rmod:g}')
    save(fig, 19, f'{name}_{CONTROLLER}', f'results/multiwfs_cache: c1 single rMod{ref_rmod:g} (chromatic), c5 single rMod{ref_rmod:g} (achromatic)')


def _makers(band):
    """make(mag900) of the case 2+3 slide systems (the calls of the sweeps), used to rebuild a budget at a cached row's configuration."""
    set_band(band)
    w = wfs_nm
    lo = lambda mg, sh: st.sensor(R1, 14, w, mg, sh, N=N_LO)
    ho = lambda mg, sh: st.sensor(R1, 48, w, mg, sh)

    def c23(mg):
        cands = []
        for f in st.split_candidates(FRAC_CMB):
            r = st.best_sampling({n: st.sensor(0.0, n, w, mg, f, N=m.max_modes(n)) for n in NS_UNMOD}, ho(mg, 1 - f))
            r.params['f_LO'] = f
            cands.append(r)
        return min(cands, key=st._contrast_key)
    return {REF: lambda mg: st.single(st.sensor(R1, 48, w, mg)),
            '2: 48x48 + 14x14, optimal split': lambda mg: st.best_split(lambda f: lo(mg, f), lambda f: ho(mg, f), FRAC_CMB, TOL),
            '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50': lambda mg: st.fused([ho(mg, 0.5), st.sensor(0.0, 48, w, mg, 0.5)]),
            '2+3: 48x48 rMod 2 + unmod n x n, optimal split': c23}


def eb_case23_profiles(mags=(5.0, 8.0, 12.0), rmax=25):
    """Slide 16b: radial coronagraphic raw-contrast profiles (perfect coronagraph, azimuthal mean) of the best single rMod 2, case 2 (optimal
    split), case 3 and case 2+3 at mags 5 / 8 / 12, one figure per band. Each budget is rebuilt at the configuration of its cached sweep
    row (Study.fixed: no search). Top: 0-25 lambda/D; bottom: zoom on the 2-5 lambda/D zone, its legend the zone mean contrast of
    each curve (rebuilt budget; the cached value is checked, printed)."""
    sty = {REF: ('best single rMod 2', CFG['single'], '-'), '2: 48x48 + 14x14, optimal split': ('case 2: optimal split', CFG['case2'], '--'),
           '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50': ('case 3: rMod 2 + rMod 0, 50/50', CFG['case3'], '-'),
           '2+3: 48x48 rMod 2 + unmod n x n, optimal split': ('case 2+3: optimal split', 'C5', '-')}
    from aoeb.multiwfs import _config_of
    for band in BANDS:
        rows = {lab: sweep(name, ms) for lab, name, ms in eb_sweeps(band) if lab in sty}
        if any(v is None for v in rows.values()):
            print(f'case 2+3 profiles {band}: missing', [k for k, v in rows.items() if v is None])
            continue
        makers = _makers(band)
        fig, axs = plt.subplots(2, len(mags), figsize=(5.2 * len(mags), 8.2), squeeze=False, gridspec_kw=dict(height_ratios=(1.25, 1)))
        for j, mg in enumerate(mags):
            ax, az = axs[0, j], axs[1, j]
            ax.axvspan(*ZONE, color='gray', alpha=0.15, lw=0)
            az.axvspan(*ZONE, color='gray', alpha=0.15, lw=0)
            zoom = []
            for lab, (txt, col, ls) in sty.items():
                row = rows[lab].loc[mg]
                old = st.fixed
                st.fixed = _config_of(row.to_dict())
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        r = makers[lab](mg)
                finally:
                    st.fixed = old
                c = st.contrast(r)
                if abs(c / row['contrast'] - 1) > 1e-3:      # rows cached before 2026-11-15: SR stands for the AO peak (~3 %, legacy_contrast)
                    print(f'{band} mag {mg:g} {lab}: rebuilt contrast {c:.4e} vs cached {row["contrast"]:.4e} ({c / row["contrast"] - 1:+.1%})')
                res = eng.psf_terms(r, st.sci_nm, names=['total'])
                pk = aonorm(res)
                rr, prof = res.profile('total', 'coro', 0.5, 'mean', rmax)
                prof = prof / pk
                ax.semilogy(rr, prof, ls=ls, color=col, lw=1.6, label=txt)
                rz, pz = res.profile('total', 'coro', 0.25, 'mean', ZONE[1] + 1)          # zoom: finer radial bins
                pz = pz / pk
                k = (rz >= ZONE[0] - 0.5) & (rz <= ZONE[1] + 0.5)
                az.semilogy(rz[k], pz[k], ls=ls, color=col, lw=1.8, label=f'{c:.2e}')
                zoom.append(pz[(rz >= ZONE[0]) & (rz <= ZONE[1])])
            ax.set(xlim=(0, rmax), ylim=(1e-7, 1e-2), xlabel='separation [$\\lambda$/D]', title=f'mag {mg:g}')
            v = np.concatenate(zoom)
            az.set(xlim=(ZONE[0] - 0.5, ZONE[1] + 0.5), ylim=(v.min() / 1.6, v.max() * 1.6), xlabel='separation [$\\lambda$/D]')
            az.legend(title=f'{ZONE_TXT} mean', fontsize=11, title_fontsize=11, loc='best', framealpha=0.85)   # placed clear of the curves
        axs[0, 0].set_ylabel(f"raw contrast ({S[BANDS[band]]['science_nm']:.0f} nm)")
        axs[1, 0].set_ylabel(f'zoom on {ZONE_TXT}')
        fig.suptitle(f'Case 2+3, {band_title(band)}, z = {zenith:g}°: coronagraphic profiles (perfect coronagraph, azimuthal mean)')
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.legend(h, l, loc='lower center', ncol=len(l), fontsize=12)
        save(fig, 16, f'case23_profiles_{band}_{CONTROLLER}', f'budgets rebuilt at the cached configurations of c1 single rMod2, c2 split, '
                                                              f'c3 split, c23 split ({CONTROLLER}); PsfEngine.psf_terms')


def _e2e_runs(band):
    """System label -> {mag: (run folder, selected row)} of the E2E sweeps of the flux-equated systems (case 2+3: the best band split
    per magnitude, sweep rule)."""
    W = 900 if band == 'VIS' else 1300
    b = 'vis' if band == 'VIS' else 'nir'
    src = {REF: [op.join(E2E_FINAL, f'single_wfs{W}_chrom_r2')], '2: 48x48 + 14x14, optimal split': [op.join(E2E_TALK, f'c2s_{W}')],
           '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50': [op.join(E2E_TALK, f'c3_{W}')],
           '2+3: 48x48 rMod 2 + unmod n x n, optimal split': [op.join(E2E_FINAL, f'dual_eq{W}_chrom_u{u}') for u in (10, 30, 50, 70, 90)]}
    out = {}
    for lab, folders in src.items():
        best = {}
        for folder in folders:
            for mg, r in _selected(folder).items():
                c, sr = float(r[f'contrast_{b}']), float(r[f'sr_{b}'])
                if mg not in best or _score(c, sr) < _score(*best[mg][2]):
                    best[mg] = (folder, r, (c, sr))
        out[lab] = {mg: v[:2] for mg, v in best.items()}
    return out


def _e2e_config(lab, folder, row):
    """Study.fixed configuration of an E2E run (the selected.csv key: single [rmod, mag, fs, n, ...], dual [mag, k, n, ku, ...])."""
    k = [x.strip(" []'") for x in row['key'].split(',')]
    if lab == REF:
        return {'n_subap': float(k[3]), 'fs': float(k[2])}
    kk, n, ku = int(float(k[1])), float(k[2]), int(float(k[3]))
    fs_lo = 2000.0 / ku                                  # the 'unmodulated' chain runs at the loop rate, the modulated one every kk steps
    fs_ho = fs_lo / kk
    if lab.startswith('3:'):
        return {'fs': fs_lo}
    if lab.startswith('2:'):
        return {'fs': fs_ho, 'fs_lo': fs_lo, 'f_LO': FRAC_CMB[2]}          # 1/(1+(48/14)^2) = 7.84 %
    u = float(op.basename(folder).rsplit('_u', 1)[1]) / 100
    return {'fs': fs_ho, 'fs_lo': fs_lo, 'f_LO': u, 'n_unmod': n}


E2E_BUDGETS = op.join(op.dirname(OUT), 'e2e_config_budgets')        # EB budgets at the E2E configurations, per controller (slide 24)


def eb_at_e2e(band, lab, mg, controller, compute=True):
    """EB budget of system `lab` at the configuration of its selected E2E run at magnitude mg, with `controller` ('leaky' / 'wiener'),
    cached in figures/e2e_config_budgets (keyed on controller, band, system, magnitude, configuration and E2E run). compute=False: only
    the cache (the Wiener designs spawn worker processes: run results/WFSW/talk_cases_20261112/eb_wiener_at_e2e.py). -> (BudgetResult or
    None, cfg, run folder, selected row), None without an E2E run."""
    import hashlib
    hit = _e2e_runs(band).get(lab, {}).get(mg)
    if hit is None:
        return None
    folder, row = hit
    cfg = _e2e_config(lab, folder, row)
    tag = hashlib.md5(repr((sorted(cfg.items()), row['tn'], CRITERION)).encode()).hexdigest()[:10]
    slug = ''.join(c if c.isalnum() else '_' for c in lab)[:30]
    path = op.join(E2E_BUDGETS, f'{controller}_{band}_{slug}_m{mg:g}_{tag}.pkl')
    if op.exists(path):
        import pickle
        with open(path, 'rb') as fh:
            return pickle.load(fh), cfg, folder, row
    if not compute:
        return None, cfg, folder, row
    ctl0, old = st.kw['controller'], st.fixed
    st.kw['controller'] = controller
    st._stages.clear(); st._best.clear()                     # memory caches are not keyed on the controller
    try:
        set_band(band)
        st.fixed = cfg
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = _makers(band)[lab](mg)
    finally:
        st.kw['controller'], st.fixed = ctl0, old
        st._stages.clear(); st._best.clear()
        set_band(band)
    import pickle
    os.makedirs(E2E_BUDGETS, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(r, fh)
    return r, cfg, folder, row


def eb_vs_e2e_profiles(mags=(7.0, 8.0, 12.0), rmax=25):
    """Slide 24: coronagraphic profiles (perfect coronagraph, azimuthal mean, contrast relative to the AO PSF peak), EB (solid) vs E2E
    (dashed) and EB Wiener (dotted, from the cache of eb_wiener_at_e2e.py; absent until it has run), of the flux-equated systems: best single rMod 2, case 2 (7.84 % split), case 3 (50/50), case 2+3. Each EB budget is evaluated
    at the configuration E2E selected (binning, rates, split, sampling; Study.fixed), chromatic, z = 30: EB leaky, E2E Kalman. One figure
    per band (rows: systems, columns: magnitudes) and its zoom on 2-5 lambda/D, whose legends give each zone mean (EB: mean at 2, 3, 4, 5
    lambda/D, as Study.contrast; E2E: mean of the profile bins in [2, 5], as the sweeps)."""
    from astropy.io import fits
    sty = {REF: ('best single rMod 2', CFG['single']), '2: 48x48 + 14x14, optimal split': ('case 2: 7.84 % split', CFG['case2']),
           '3: 48x48 rMod 2 + 48x48 rMod 0, 50/50': ('case 3: 50/50', CFG['case3']),
           '2+3: 48x48 rMod 2 + unmod n x n, optimal split': ('case 2+3: best split', 'C5')}
    for band in BANDS:
        b = 'vis' if band == 'VIS' else 'nir'
        runs = _e2e_runs(band)
        makers = _makers(band)
        figs = {k: plt.subplots(len(sty), len(mags), figsize=(5.0 * len(mags), 3.4 * len(sty)), squeeze=False, sharex=True) for k in ('full', 'zoom')}
        for i, (lab, (txt, col)) in enumerate(sty.items()):
            for j, mg in enumerate(mags):
                af, az = figs['full'][1][i, j], figs['zoom'][1][i, j]
                for a in (af, az):
                    a.axvspan(*ZONE, color='gray', alpha=0.15, lw=0)
                hit = runs.get(lab, {}).get(mg)
                if hit is None:
                    for a in (af, az):
                        a.text(0.5, 0.5, 'no E2E run', transform=a.transAxes, ha='center', color='0.5')
                    continue
                folder, row = hit
                rad, prof = np.asarray(fits.getdata(op.join(folder, 'runs', row['tn'], f'coro_psf_{b}_int.fits'))[0], float)
                c_e2e = float(row[f'contrast_{b}'])
                vz = [prof[(rad >= ZONE[0]) & (rad <= ZONE[1])]]
                for ctl, ls_, lw_ in (('leaky', '-', 1.6), ('wiener', ':', 2.0)):
                    out = eb_at_e2e(band, lab, mg, ctl, compute=(ctl == 'leaky'))
                    r, cfg = out[0], out[1]
                    if r is None:
                        continue
                    set_band(band)
                    res = eng.psf_terms(r, st.sci_nm, names=['total'])
                    pk = aonorm(res)
                    c_eb = st.contrast(r)
                    x, p = res.profile('total', 'coro', 0.5, 'mean', rmax)
                    xz, pz = res.profile('total', 'coro', 0.25, 'mean', ZONE[1] + 1)
                    af.semilogy(x, p / pk, ls_, color=col, lw=lw_, label=f'EB ({ctl})')
                    k = (xz >= ZONE[0] - 0.5) & (xz <= ZONE[1] + 0.5)
                    az.semilogy(xz[k], pz[k] / pk, ls_, color=col, lw=lw_ + 0.2, label=f'EB {ctl} {c_eb:.2e} (E2E x{c_e2e / c_eb:.2f})')
                    vz.append(pz[(xz >= ZONE[0]) & (xz <= ZONE[1])] / pk)
                af.semilogy(rad, prof, '--', color=col, lw=1.4, label='E2E (Kalman)')
                ke = (rad >= ZONE[0] - 0.5) & (rad <= ZONE[1] + 0.5)
                az.semilogy(rad[ke], prof[ke], '--', color=col, lw=1.6, label=f'E2E Kalman {c_e2e:.2e}')
                v = np.concatenate(vz)
                af.set(xlim=(0, rmax), ylim=(1e-7, 1e-2))
                az.set(xlim=(ZONE[0] - 0.5, ZONE[1] + 0.5), ylim=(v.min() / 1.6, v.max() * 1.6))
                az.legend(fontsize=9, loc='best', framealpha=0.85, title=f'{ZONE_TXT} mean', title_fontsize=9)
                conf = eb_config(lab, pd.Series({'n_subap': cfg.get('n_subap', 48), 'fs': cfg['fs'], 'fs_lo': cfg.get('fs_lo', np.nan),
                                                 'f_LO': cfg.get('f_LO', np.nan), 'n_unmod': cfg.get('n_unmod', np.nan)}))
                for a in (af, az):
                    a.set_title(f'{txt}, mag {mg:g}', fontsize=12)
                    a.text(0.02, 0.97, conf.replace('; ', '\n'), transform=a.transAxes, va='top', fontsize=7.5, color='0.35')
        for key, (fig, axs) in figs.items():
            for a in axs[-1]:
                a.set_xlabel('separation [$\\lambda$/D]')
            for a in axs[:, 0]:
                a.set_ylabel(f"raw contrast ({S[BANDS[band]]['science_nm']:.0f} nm)")
            fig.suptitle(f'EB (solid: leaky, dotted: Wiener) vs E2E (dashed, Kalman) at the E2E configurations, {band_title(band)}, z = {zenith:g}°, chromatic'
                         + (', zoom on ' + ZONE_TXT if key == 'zoom' else ''))
            fig.tight_layout()
            save(fig, 24, f'eb_vs_e2e_profiles{"_zoom" if key == "zoom" else ""}_{band}_{CONTROLLER}',
                 'E2E: coro_psf_*_int.fits of the selected runs (sweep_final_20261108, talk_cases_20261112); EB: budgets at those '
                 f'configurations ({CONTROLLER}), PsfEngine.psf_terms, AO-peak normalised')


if _STANDALONE and __name__ == '__main__':
    import warnings
    eb_case23_figures()
    eb_case23_profiles()
    eb_vs_e2e_profiles()
    e2e_case23_figure()
    TALK_TABLES = write_tables(tables())
    for r_ in (2, 1):
        reference_single_figure(r_)
        summary_slide19(True, r_)
        summary_slide19(False, r_)
