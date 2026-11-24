"""Plots (matplotlib figures are returned, never saved): calibration products, PSF profiles of a BudgetResult (psf.PsfEngine) and the
single- versus multi-sensor comparisons of aoeb.multiwfs (errors in nm rms)."""
import matplotlib.pyplot as plt
import numpy as np

from . import psf as psflib

COLORS = {'lag': 'C0', 'noise': 'C1', 'noise_photon': 'C1', 'noise_ron': 'goldenrod', 'alias': 'C2', 'fitting': 'C3',
          'total': 'k', 'xtalk': 'C4', 'chrom_shear': 'C5', 'scintillation': 'teal',
          'turbulence': 'gray'}
LABELS = {'photPerMs': 'photons / ms at the detector', 'freqInHz': 'frame rate [Hz]', 'seeing': 'seeing [arcsec]',
          'rMod': 'modulation radius [$\\lambda$/D]', 'Nsubaps': 'sub-apertures across the pupil', 'Nmodes': 'controlled modes',
          'modeId': 'mode index', 'wavelength_nm': 'sensing wavelength [nm]'}


def _fig(ax, size=(6.5, 4.5)):
    return plt.subplots(figsize=size) if ax is None else (ax.figure, ax)


def _finish(fig, ax, xlabel, ylabel, title=None, logx=False, logy=False, xlim=None, ylim=None):
    ax.set_xlabel(LABELS.get(xlabel, xlabel))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xscale('log' if logx else 'linear')
    ax.set_yscale('log' if logy else 'linear')
    if xlim is not None:
        ax.set_xlim(xlim)                              # (lo, hi); None for one side keeps it automatic
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, which='both', alpha=0.4)
    fig.tight_layout()
    return fig


ERRORS = ('residual', 'lag', 'noise', 'noise_photon', 'noise_ron', 'alias', 'xtalk', 'fitting', 'chrom_shear', 'scintillation',
          'total')        # columns in nm rms (scan / mode tables)


def _pretty(name):
    """Legend text of a term: underscores as spaces ('noise_photon' -> 'noise photon')."""
    return str(name).replace('_', ' ')


def _label(name, h):
    return f'{name}={h:g}' if isinstance(h, (int, float, np.floating)) else f'{name}={h}'


def plot_calibration(machine, product, hue=None, values=None, rMod=3.0, n_subap=48, seeing=0.9, N=None, N_corr=None, flux=1e7,
                     fs=1000.0, wavelength_nm=None, logy=None, title=None, logx=False, xlim=None, ylim=None):
    """'alias' | 'og' | 'ct' | 'noise' | 'turb' versus mode (or 'thrp' versus `hue`); one curve per value of `hue`
    in {rMod, n_subap, seeing, N, wavelength_nm, flux, fs} (flux [photons/s] and fs [Hz] only change 'noise')."""
    m, base = machine, dict(rMod=rMod, n_subap=n_subap, seeing=seeing, N=N, wavelength_nm=wavelength_nm, flux=flux, fs=fs)
    if hue is not None and hue not in base:
        raise ValueError(f'hue must be one of {sorted(base)}, got {hue!r}')
    if hue is not None and values is None:
        raise ValueError('give the values of the hue')
    fig, ax = _fig(None)
    xs, ys = [], []
    for i, v in enumerate([None] if hue is None else values):
        cfg = {**base, **({} if hue is None else {hue: v})}
        N_i = cfg['N'] or m.max_modes(cfg['n_subap'])
        cal = m.calibration(cfg['rMod'], cfg['n_subap'], cfg['seeing'], N_i, N_corr, wavelength_nm=cfg['wavelength_nm'])
        if product == 'thrp':
            xs.append(v)
            ys.append(cal.thrp)
            continue
        y = {'alias': lambda: cal.alias_rms, 'og': lambda: cal.og, 'ct': lambda: cal.ct,
             'noise': lambda: np.sqrt(cal.noise.modal_variance(m.detector.n_det(cfg['flux'], cfg['fs']), m.detector)),
             'turb': lambda: np.sqrt(m.turbulence(cfg['seeing'], N_i)[0])}[product]()
        ax.plot(np.arange(len(y))+1, y, '-', lw=1, color=f'C{i % 10}', label=None if hue is None else _label(hue, v))
    if product == 'thrp':
        ax.plot(xs, ys, '-o')
        return _finish(fig, ax, hue or 'rMod', 'throughput (fraction of counts in the pupils)', title, logx, False, xlim, ylim)
    if hue is not None:
        ax.legend(fontsize=8)
    fixed = ', '.join(s for k, s in (('flux', f'{flux:.0e} ph/s'), ('fs', f'{fs:.0f} Hz')) if k != hue)
    unit = {'alias': 'aliasing std [command units]', 'og': 'optical gain', 'ct': 'crosstalk',
            'noise': f'noise rms [command units] ({fixed})', 'turb': 'turbulence rms [command units]'}[product]
    return _finish(fig, ax, 'modeId', unit, title, logx, product in ('alias', 'noise', 'turb') if logy is None else logy, xlim, ylim)


def plot_psf_terms(psf_result, kind='psf', terms=None, binsize=0.5, rmax=None, pa_range=None, show_dl=True, show_open_loop=False,
                   title=None, ax=None, ymin=None, ymax=None, logx=False, xlim=None, ylim=None, shade=None):
    """Radial raw-contrast profile of every term alone and of the total. kind: 'psf' (long-exposure PSF, Airy pattern included) or
    'coro' (after a perfect coronagraph, Cavarroc et al. 2006 eq. 1: the coherent part S * PSF_DL removed, so no Airy-ring spikes).
    The image of a term alone is NOT its contribution to the total: inside the total its halo is attenuated by the Strehl of the other
    terms, so the total can lie below a single term where that term dominates. shade: (r1, r2) separations [lambda/D] drawn as a gray
    band (e.g. where the contrast is to be maximised)."""
    fig, ax = _fig(ax, (8,5))
    if shade is not None:
        ax.axvspan(*shade, color='gray', alpha=0.15, lw=0)
    for n in (psf_result.psf if kind == 'psf' else psf_result.coro):
        if (terms is not None and n not in terms and n != 'total') or (n == 'turbulence' and not show_open_loop):
            continue
        r, p = psf_result.profile(n, kind, binsize, rmax, pa_range)
        ax.plot(r, p, color=COLORS.get(n), lw=1, ls=':' if n == 'turbulence' else '--', label=_pretty(n))
    if show_dl and kind == 'psf':                      # (a perfect coronagraph removes the diffraction-limited PSF entirely)
        r, p = psflib.radial_profile(psf_result.diffraction_limited, psf_result.pixel_scale, binsize, None, rmax, pa_range)
        ax.plot(r, p, color='gray', lw=1, label='diffraction limit')
    ax.legend(fontsize=8)
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(bottom=ymin,top=ymax)
    ax.set_xlim(right=ax.get_xlim()[1])
    return _finish(fig, ax, '[$\\lambda$/D]', f'Raw contrast ({psf_result.wavelength_nm:.0f}nm)', title, logx, True, xlim, ylim)


# ----------------------------------------------------------------------------- multi-sensor comparisons (aoeb.multiwfs sweeps)
def _contrast_line(ax, x, y, ok, color, label=None):
    """Contrast curve: solid where `ok`, faded dashed elsewhere."""
    ax.plot(x, y, '--', color=color, alpha=0.4)
    ax.plot(x, np.where(ok, y, np.nan), '-', color=color, label=label)


def plot_comparison(curves, title, ref, sci_nm, terms=None, sr_min=0.25, zone=(2, 5)):
    """Two figures of two panels against the magnitude, one colour per curve ({label: sweep DataFrame}): total residual | total relative
    to curve `ref`; contrast | contrast relative to `ref` (faded and dashed where the science Strehl < sr_min). terms: optional pairs
    (column, label) of two error terms, plotted in a third figure."""
    from matplotlib.lines import Line2D
    figs = [plt.subplots(1, 2, figsize=(13, 4.2)) for _ in range(3 if terms else 2)]
    (f1, a1), (f3, a3) = figs[0], figs[-1]
    for i, (name, df) in enumerate(curves.items()):
        c, ok = f'C{i}', df['SR'] >= sr_min
        a1[0].semilogy(df.index, df['total'], '-', color=c, label=name)
        a1[1].plot(df.index, df['total'] / curves[ref]['total'], '-', color=c, label=name)
        if terms:
            for a, (t, _) in zip(figs[1][1], terms):
                a.semilogy(df.index, df[t], '-', color=c, label=name)
        _contrast_line(a3[0], df.index, df['contrast'], ok, c, label=name)
        _contrast_line(a3[1], df.index, df['contrast'] / curves[ref]['contrast'], ok & (curves[ref]['SR'] >= sr_min), c, label=name)
    a3[0].set_yscale('log')
    labels = ['total residual [nm rms]', f'total / {ref}', *(f'{lab} [nm rms]' for _, lab in (terms or ())),
              f'raw contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D ({sci_nm:.0f} nm)', f'contrast / {ref}']
    for a, yl in zip([ax for _, axs in figs for ax in axs], labels):
        a.set_xlabel('magnitude at 900 nm'); a.set_ylabel(yl); a.grid(alpha=0.3, which='both')
    for a in (a1[1], a3[1]):
        a.axhline(1.0, color='gray', lw=0.8)
    subs = ['total residual', *([', '.join(lab for _, lab in terms)] if terms else []), 'contrast']
    for (f, a), sub in zip(figs, subs):
        a[0].legend(fontsize=8)
        f.suptitle(f'{title}: {sub}'); f.tight_layout(w_pad=3)
    a3[1].legend([Line2D([], [], color='k', ls='--', alpha=0.4)], [f'science Strehl $<$ {sr_min:g}'], fontsize=8)
    return [f for f, _ in figs]


def plot_noise_frame_rate(panels, ticks=(125, 250, 500, 1000, 2000)):
    """Photon, readout, lag and total residual against the magnitude, with the frame rate on the right axis; panels: {title: DataFrame
    with noise_photon, noise_ron, lag, total, fs (index: magnitude)}, two per row."""
    n = len(panels)
    fig, ax = plt.subplots((n + 1) // 2, min(n, 2), figsize=(13, 4.2 * ((n + 1) // 2)), squeeze=False)
    for a, (title, df) in zip(ax.flat, panels.items()):
        a.semilogy(df.index, df['noise_photon'], '-', label='photon')
        a.semilogy(df.index, df['noise_ron'], '-', label='readout')
        a.semilogy(df.index, df['lag'], '-', color='C2', label='lag')
        a.semilogy(df.index, df['total'], 'k-', label='total')
        tw = a.twinx()
        tw.semilogy(df.index, df['fs'], '-.', color='C3', label='optimal frame rate (right axis)')
        tw.set_yticks(list(ticks), [f'{t:g}' for t in ticks]); tw.minorticks_off(); tw.set_ylim(0.8 * min(ticks), 1.25 * max(ticks))
        tw.set_ylabel('optimal frame rate [Hz]', color='C3'); tw.tick_params(axis='y', colors='C3')
        a.set_xlabel('magnitude at 900 nm'); a.set_ylabel('residual [nm rms]'); a.grid(alpha=0.3); a.set_title(title)
    fig.legend(*[sum(x, []) for x in zip(a.get_legend_handles_labels(), tw.get_legend_handles_labels())], loc='lower center', ncol=5,
               fontsize=8)
    fig.tight_layout(w_pad=3, rect=(0, 0.07 / ((n + 1) // 2), 1, 1))
    return fig


# ---- sensor fusion highlights (analyse_errorbudget.ipynb, presentation figures)
def fusion_gain(ref, fused, sr_min=0.25, col='contrast'):
    """Two sweeps (index: magnitude at 900 nm; columns contrast, SR) -> DataFrame per magnitude of `ref`: 'ratio' = fused / ref contrast
    and 'dmag' = magnitudes gained at equal contrast (the fused system reaches the contrast of ref at magnitude m at m + dmag; log-linear
    interpolation within the fused sweep, NaN outside), 'ok' = both science Strehls >= sr_min. col='SR': the same for the science Strehl
    (ratio = fused / ref SR, dmag = magnitudes gained at equal Strehl)."""
    import pandas as pd
    sg = -1.0 if col == 'SR' else 1.0                                          # the Strehl falls with the magnitude: flip the sign
    f = fused.dropna(subset=[col])
    cf = np.maximum.accumulate(sg * np.log(f[col].to_numpy(float)))           # the metric grows with the magnitude: enforce monotonicity
    cr = sg * np.log(ref[col].to_numpy(float))
    m = ref.index.to_numpy(float)
    dmag = np.interp(cr, cf, f.index.to_numpy(float), left=np.nan, right=np.nan) - m
    ratio = fused[col].reindex(ref.index).to_numpy(float) / ref[col].to_numpy(float)
    ok = (ref['SR'] >= sr_min).to_numpy() & (fused['SR'].reindex(ref.index) >= sr_min).to_numpy()
    return pd.DataFrame({'ratio': ratio, 'dmag': dmag, 'ok': ok}, index=ref.index)


def plot_fusion_gains(gains, zone=(2, 5), sr_min=0.25, bands=('VIS', 'NIR'), mags=None, gate=True, sr_gains=None, ylim=None, fontsize=15,
                      colours=None, title=None, sr_ref=1.0):
    """Headline figure: for every advantage (one colour) and variant (solid: main, dashed: alternative) the contrast of the fused system
    relative to its single-sensor reference (top) and the magnitudes gained at equal contrast (bottom), one column per band.
    gains: {(advantage, variant, main): {band: fusion_gain DataFrame}}; faded where a science Strehl < sr_min (gate=True). mags: (lo, hi)
    magnitude range shown; sr_gains: same structure from fusion_gain(..., col='SR'), its ratio replaces the magnitudes gained (bottom);
    ylim: ((lo, hi) top, (lo, hi) bottom) shared by both bands; fontsize: base font size (y labels on the first column only); colours:
    {advantage: colour} (default C0, C1, ... in order); title: suptitle (default: the generic one); sr_ref: reference line of the bottom
    row with sr_gains (1 for a ratio, 0 for a difference)."""
    with plt.rc_context({'font.size': fontsize}):
        return _plot_fusion_gains(gains, zone, sr_min, bands, mags, gate, sr_gains, ylim, fontsize, colours, title, sr_ref)


def _plot_fusion_gains(gains, zone, sr_min, bands, mags, gate, sr_gains, ylim, fontsize, colours=None, title=None, sr_ref=1.0):
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, len(bands), figsize=(6.8 * len(bands), 8), sharex=True, squeeze=False)
    colours = {**{a: f'C{i}' for i, a in enumerate(dict.fromkeys(a for a, _, _ in gains))}, **(colours or {})}
    for (adv, var, main), per_band in gains.items():
        for j, band in enumerate(bands):
            if band not in per_band:
                continue
            for ax, (src, col) in zip(axs[:, j], ((gains, 'ratio'), (sr_gains or gains, 'ratio' if sr_gains else 'dmag'))):
                g = src[(adv, var, main)][band]
                if mags:
                    g = g[(g.index >= mags[0]) & (g.index <= mags[1])]
                g = g if gate else g.assign(ok=True)
                y = g[col].where(g['ok'])
                ax.plot(g.index, y, '-' if main else '--', color=colours[adv], lw=2 if main else 1.2, marker='o', ms=3,
                        label=f'{adv}: {var}')
                ax.plot(g.index, g[col].where(~g['ok']), ':', color=colours[adv], lw=1, alpha=0.5)
    for j, band in enumerate(bands):
        axs[0, j].axhline(1.0, color='gray', lw=0.8)
        axs[1, j].axhline(sr_ref if sr_gains else 0.0, color='gray', lw=0.8)
        axs[0, j].set_title(band)
        if ylim:
            axs[0, j].set_ylim(ylim[0]); axs[1, j].set_ylim(ylim[1])
        if j == 0:
            axs[0, j].set_ylabel(f'contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D: fused / single')
            axs[1, j].set_ylabel('science Strehl: fused / single' if sr_gains else 'magnitudes gained at equal contrast')
        axs[1, j].set_xlabel('magnitude' if mags else 'magnitude at 900 nm')
        for ax in axs[:, j]:
            ax.grid(alpha=0.3, which='both')
    h, l = axs[0, 0].get_legend_handles_labels()
    if gate:
        h += [Line2D([], [], color='gray', ls=':')]
        l += [f'science Strehl $<$ {sr_min:g}']
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=fontsize - 2, frameon=False)
    fig.suptitle(title or 'Sensor fusion against a single WFS: contrast and Strehl gain per advantage')
    fig.tight_layout(rect=(0, 0.02 + 0.035 * ((len(l) + 1) // 2), 1, 1))
    return fig


def plot_scan(df, pairs, xlabel, title, sr_min=0.25, zone=(2, 5), logx=False, xref=None, secondary=None, ratio_only=False):
    """Fused against single-sensor configurations over a parameter (Study.scan DataFrame): one column per pair {label: (reference, fused)};
    top: raw contrast in the zone of the reference (dashed) and of the fused system (solid), bottom: fused / reference (ratio_only: the
    bottom row alone); one colour per magnitude, faded where a science Strehl < sr_min. xref: the nominal value (dotted line);
    secondary: (label, f, f_inverse) of a second x axis on top."""
    from matplotlib.lines import Line2D
    mags = sorted(df['mag'].unique())
    nrow = 1 if ratio_only else 2
    fig, axs = plt.subplots(nrow, len(pairs), figsize=(5.4 * len(pairs), 3.9 * nrow), sharex=True, squeeze=False)
    for j, (lab, (ref, fu)) in enumerate(pairs.items()):
        top, bot = (None, axs[0, j]) if ratio_only else axs[:, j]
        for i, mg in enumerate(mags):
            d = df[df['mag'] == mg]
            a, b = (d[d['config'] == k].set_index('x').sort_index() for k in (ref, fu))
            c = f'C{i}'
            if top is not None:
                for frame, ls in ((a, '--'), (b, '-')):
                    ok = frame['SR'] >= sr_min
                    top.plot(frame.index, frame['contrast'], ls, color=c, lw=1, alpha=0.25)
                    top.plot(frame.index, frame['contrast'].where(ok), ls, color=c, lw=1.8, marker='o', ms=3)
            ratio, ok = b['contrast'] / a['contrast'], (a['SR'] >= sr_min) & (b['SR'] >= sr_min)
            bot.plot(ratio.index, ratio, '-', color=c, lw=1, alpha=0.25)
            bot.plot(ratio.index, ratio.where(ok), '-', color=c, lw=1.8, marker='o', ms=3, label=f'mag {mg:g}')
        head = top if top is not None else bot
        head.set_title(f'{lab}\n{fu} vs {ref}', fontsize=9)
        if top is not None:
            top.set_yscale('log')
            top.set_ylabel(f'raw contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D')
        bot.set(xlabel=xlabel, ylabel=f'contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D: fused / single')
        bot.axhline(1.0, color='gray', lw=0.8)
        for ax in axs[:, j]:
            if logx:
                ax.set_xscale('log')
            if xref is not None:
                ax.axvline(xref, color='gray', ls=':', lw=1)
            ax.grid(alpha=0.3, which='both')
        if secondary is not None:
            sec = head.secondary_xaxis('top', functions=secondary[1:])
            sec.set_xlabel(secondary[0])
    axs[-1, 0].legend(fontsize=8)
    if not ratio_only:
        axs[0, 0].legend([Line2D([], [], color='k', ls='--'), Line2D([], [], color='k')], ['single WFS', 'fused'], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
