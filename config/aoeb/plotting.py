"""Plots (matplotlib figures are returned, never saved): budget terms / comparisons / per-mode quantities from the scan
DataFrames (errors in nm rms), calibration products, and PSD / PSF of a BudgetResult (psf.PsfEngine)."""
import matplotlib.pyplot as plt
import numpy as np

from . import psf as psflib
from . import utils

COLORS = {'lag': 'C0', 'noise': 'C1', 'noise_photon': 'C1', 'noise_ron': 'goldenrod', 'alias': 'C2', 'fitting': 'C3',
          'total': 'k', 'xtalk': 'C4', 'chrom_shear': 'C5', 'chrom_mult': 'C6', 'chrom_fresnel': 'olive', 'scintillation': 'teal',
          'turbulence': 'gray'}
LABELS = {'photPerMs': 'photons / ms at the detector', 'freqInHz': 'frame rate [Hz]', 'seeing': 'seeing [arcsec]',
          'rMod': 'modulation radius [$\\lambda$/D]', 'Nsubaps': 'sub-apertures across the pupil', 'Nmodes': 'controlled modes',
          'modeId': 'mode index', 'wavelength_nm': 'sensing wavelength [nm]'}


def _select(df, filters):
    for k, v in (filters or {}).items():
        vals = np.atleast_1d(v)
        if df[k].dtype.kind in 'fi':
            df = df[np.any([np.isclose(df[k].to_numpy(float), x, rtol=1e-6) for x in vals], axis=0)]
        else:
            df = df[df[k].isin(vals)]
    return df


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


ERRORS = ('residual', 'lag', 'noise', 'noise_photon', 'noise_ron', 'alias', 'xtalk', 'fitting', 'chrom_mult', 'chrom_shear',
          'chrom_fresnel', 'scintillation', 'total')        # columns in nm rms (scan / mode tables)


def _value(df, q):
    return df[q].astype(float)


def _one_per(df, x, by=None):
    """Raise if the filters leave several configurations for the same x (they would be drawn as one connected line)."""
    groups = [df] if by is None else [g for _, g in df.groupby(by)]
    if any(g[x].duplicated().any() for g in groups):
        varying = [c for c in ('scenario', 'controller', 'rMod', 'seeing', 'Nsubaps', 'Nmodes', 'freqInHz', 'photPerMs', 'wavelength_nm',
                               'crosstalk') if c in df and c != x and df[c].nunique() > 1]
        raise ValueError(f'several configurations share the same {x}: pin them with filters (still varying: {varying})')


def _pretty(name):
    """Legend text of a term: underscores as spaces ('noise_photon' -> 'noise photon')."""
    return str(name).replace('_', ' ')


def _label(name, h):
    return f'{name}={h:g}' if isinstance(h, (int, float, np.floating)) else f'{name}={h}'


def plot_budget_terms(totals, x='photPerMs', filters=None, terms=('lag', 'noise', 'alias', 'fitting', 'total'), logx=True, logy=False,
                      title=None, ax=None, xlim=None, ylim=None):
    """Budget terms [nm rms] versus x for the configuration fixed by `filters` (the terms add in quadrature to the total)."""
    df = _select(totals, filters).sort_values(x)
    _one_per(df, x)
    fig, ax = _fig(ax)
    for t in terms:
        ax.plot(df[x], _value(df, t), '-o' if t == 'total' else '-', ms=3, color=COLORS.get(t), label=_pretty(t),
                lw=2 if t == 'total' else 1.5)
    ax.legend()
    return _finish(fig, ax, x, 'error [nm rms]', title, logx, logy, xlim, ylim)


def plot_budget_compare(totals, x='photPerMs', hue='freqInHz', quantity='total', filters=None, col=None, logx=True, logy=False,
                        title=None, xlim=None, ylim=None):
    """`quantity` [nm rms for the error terms] versus x, one curve per value of `hue`, panels per value of `col`."""
    df = _select(totals, filters)
    cols = [None] if col is None else sorted(df[col].unique())
    fig, axes = plt.subplots(1, len(cols), figsize=(5.5 * len(cols), 4.5), squeeze=False)
    for ax, cv in zip(axes[0], cols):
        d = df if cv is None else df[df[col] == cv]
        _one_per(d, x, hue)
        for i, (h, g) in enumerate(d.groupby(hue)):
            g = g.sort_values(x)
            ax.plot(g[x], _value(g, quantity), '-o', ms=3, color=f'C{i % 10}', label=_label(hue, h))
        ax.legend(fontsize=8)
        _finish(fig, ax, x, f"{quantity}{' [nm rms]' if quantity in ERRORS else ''}", f'{col} = {cv}' if cv is not None else title, logx, logy,
                xlim, ylim)
    return fig


def plot_per_mode(modes, y='residual', hue=None, filters=None, logy=True, title=None, logx=False, xlim=None, ylim=None):
    """Per-mode quantity versus mode index (error columns in nm rms; gain, ff, og raw)."""
    df = _select(modes, filters)
    fig, ax = _fig(None)
    for i, (h, g) in enumerate([(None, df)] if hue is None else df.groupby(hue)):
        g = g.sort_values('modeId')
        ax.plot(g['modeId'], _value(g, y), '-', lw=1, color=f'C{i % 10}', label=None if h is None else _label(hue, h))
    if hue is not None:
        ax.legend(fontsize=8)
    return _finish(fig, ax, 'modeId', f'{y} [nm rms]' if y in ERRORS else y, title, logx, logy, xlim, ylim)


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


def plot_modal_variance_check(machine, rMod=3.0, n_subap=48, seeing=0.9, N=None, wavelength_nm=None, title=None, logx=False, xlim=None,
                              ylim=None):
    """Simulated (turbmodes of the aliasing run) over analytical modal variance per mode."""
    m = machine
    N = N or m.max_modes(n_subap)
    sim = utils.read_fits(m.paths.at(m.wavelength_nm if wavelength_nm is None else wavelength_nm).turbmodes(rMod, n_subap, seeing, N))
    ratio = utils.compare_modal_variance(sim, m.turbulence(seeing, sim.shape[1])[0])
    fig, ax = _fig(None)
    ax.plot(ratio, '.', ms=3)
    ax.axhline(1.0, color='k', lw=1)
    s = np.sqrt(2.0 / sim.shape[0])
    ax.axhspan(1 - s, 1 + s, color='gray', alpha=0.2, label='1 sigma statistical')
    ax.legend()
    return _finish(fig, ax, 'modeId', 'simulated / analytical variance', title or f'{N} modes, seeing {seeing}"', logx, False, xlim, ylim)


def plot_psd_terms(psd_set, terms=None, xaxis='lamD', show_open_loop=True, title=None, ax=None, xlim=None, ylim=None):
    """Radial PSD [nm^2 m^2] of every term and the total; x = separation k D [lambda/D] or k [cycles/m]; dotted: DM cut-off."""
    fig, ax = _fig(ax, (7, 4.8))
    sc = psd_set.D if xaxis == 'lamD' else 1.0
    ok = psd_set.k > 0                                 # (x range when no term is drawn)
    for n, t in psd_set.terms.items():
        if (terms is not None and n not in terms and n != 'total') or (n == 'turbulence' and not show_open_loop):
            continue
        ok = (psd_set.k > 0) & (t.psd > 0)
        ax.plot(psd_set.k[ok] * sc, t.psd[ok], color=COLORS.get(n), lw=2.2 if n == 'total' else 1.4,
                ls=':' if n == 'turbulence' else '-', label=_pretty(n))
    # if psd_set.meta.get('k_cut'):
    #     ax.axvline(psd_set.meta['k_cut'] * sc, color='k', ls=':', lw=0.8)
    ax.legend(ncol=2, fontsize=8)
    ax.set_xlim([np.min(psd_set.k[ok] * sc)/2,np.max(psd_set.k[ok] * sc)])
    return _finish(fig, ax, '[$\\lambda$/D]' if xaxis == 'lamD' else 'spatial frequency [cycles/m]',
                   'PSD [nm$^2$ m$^2$]', title, logx=False, logy=True, xlim=xlim, ylim=ylim)


def plot_psf_terms(psf_result, kind='psf', terms=None, binsize=0.5, rmax=None, stat='mean', pa_range=None, show_dl=True,
                   show_open_loop=False, title=None, ax=None, ymin=None, ymax=None, logx=False, xlim=None, ylim=None,
                   shade=None):
    """Radial raw-contrast profile of every term alone and of the total. kind: 'psf' (long-exposure PSF, Airy pattern included),
    'coro' (after a perfect coronagraph, Cavarroc et al. 2006 eq. 1: the coherent part S * PSF_DL removed, so no Airy-ring spikes) or
    'halo' (first order, the terms add up).
    The image of a term alone is NOT its contribution to the total: inside the total its halo is attenuated by the Strehl of the other
    terms, so the total can lie below a single term where that term dominates. shade: (r1, r2) separations [lambda/D] drawn as a gray
    band (e.g. where the contrast is to be maximised)."""
    fig, ax = _fig(ax, (8,5))
    if shade is not None:
        ax.axvspan(*shade, color='gray', alpha=0.15, lw=0)
    for n in {'psf': psf_result.psf, 'coro': psf_result.coro, 'halo': psf_result.halo}[kind]:
        if (terms is not None and n not in terms and n != 'total') or (n == 'turbulence' and not show_open_loop):
            continue
        r, p = psf_result.profile(n, kind, binsize, stat, rmax, pa_range)
        ax.plot(r, p, color=COLORS.get(n), lw=1, ls=':' if n == 'turbulence' else '--',
                # label=f'{n} (S={psf_result.strehl[n]:.3f})' if kind != 'halo' else n)
                label=_pretty(n))
    if show_dl and kind == 'psf':                      # (a perfect coronagraph removes the diffraction-limited PSF entirely)
        r, p = psflib.radial_profile(psf_result.diffraction_limited, psf_result.pixel_scale, binsize, None, stat, rmax, pa_range)
        ax.plot(r, p, color='gray', lw=1, label='diffraction limit')
    ax.legend(fontsize=8)
    if ymin is None:
        ymin = ax.get_ylim()[0]
    if ymax is None:
        ymax = ax.get_ylim()[1]
    ax.set_ylim(bottom=ymin,top=ymax)
    ax.set_xlim(right=ax.get_xlim()[1])
    what = {'coro': ', perfect coronagraph', 'psf': '', 'halo': ', first-order halo'}[kind]
    return _finish(fig, ax, '[$\\lambda$/D]', f'Raw contrast ({psf_result.wavelength_nm:.0f}nm)', title, logx, True, xlim, ylim)
                   #{what})', title, logx, True, xlim, ylim)


def plot_psf_image(psf_result, name='total', half_fov=20.0, vmin=1e-7, title=None, ax=None, kind='psf', vmax=1.0):
    """2-D long-exposure image (log colour scale), field +-half_fov lambda/D; kind 'psf', 'coro' (perfect coronagraph) or 'halo'."""
    img = psf_result.image(name, kind)
    M, h = img.shape[0], int(half_fov / psf_result.pixel_scale)
    fig, ax = _fig(ax, (5.5, 5))
    im = ax.imshow(np.log10(np.maximum(img[M // 2 - h:M // 2 + h + 1, M // 2 - h:M // 2 + h + 1], vmin)), origin='lower',
                   extent=[-half_fov, half_fov, -half_fov, half_fov], vmin=np.log10(vmin), vmax=np.log10(vmax), cmap='inferno')
    fig.colorbar(im, ax=ax, label='log10 contrast',shrink=0.8)
    ax.set_xlabel('$\\lambda$/D')
    ax.set_ylabel('$\\lambda$/D')
    ax.set_title(title or f'Strehl {psf_result.strehl[name]:.3f}, name')
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- multi-sensor comparisons (aoeb.multiwfs sweeps)
def _contrast_line(ax, x, y, ok, color, label=None):
    """Contrast curve: solid where `ok`, faded dashed elsewhere."""
    ax.plot(x, y, '--', color=color, alpha=0.4)
    ax.plot(x, np.where(ok, y, np.nan), '-', color=color, label=label)


def plot_comparison(curves, title, ref, sci_nm, terms=(('noise_photon', 'photon'), ('noise_ron', 'readout')), sr_min=0.25,
                    zone=(2, 5)):
    """Three figures of two panels against the magnitude, one colour per curve ({label: sweep DataFrame}): total residual | total
    relative to curve `ref`; the two `terms`; contrast | contrast relative to `ref` (faded and dashed where the science Strehl < sr_min)."""
    from matplotlib.lines import Line2D
    figs = [plt.subplots(1, 2, figsize=(13, 4.2)) for _ in range(3)]
    (f1, a1), (f2, a2), (f3, a3) = figs
    for i, (name, df) in enumerate(curves.items()):
        c, ok = f'C{i}', df['SR'] >= sr_min
        a1[0].semilogy(df.index, df['total'], '-', color=c, label=name)
        a1[1].plot(df.index, df['total'] / curves[ref]['total'], '-', color=c, label=name)
        for a, (t, _) in zip(a2, terms):
            a.semilogy(df.index, df[t], '-', color=c, label=name)
        _contrast_line(a3[0], df.index, df['contrast'], ok, c, label=name)
        _contrast_line(a3[1], df.index, df['contrast'] / curves[ref]['contrast'], ok & (curves[ref]['SR'] >= sr_min), c, label=name)
    a3[0].set_yscale('log')
    labels = ('total residual [nm rms]', f'total / {ref}', *(f'{lab} [nm rms]' for _, lab in terms),
              f'raw contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D ({sci_nm:.0f} nm)', f'contrast / {ref}')
    for a, yl in zip([*a1, *a2, *a3], labels):
        a.set_xlabel('magnitude at 900 nm'); a.set_ylabel(yl); a.grid(alpha=0.3, which='both')
    for a in (a1[1], a3[1]):
        a.axhline(1.0, color='gray', lw=0.8)
    for f, a, sub in ((f1, a1, 'total residual'), (f2, a2, ', '.join(lab for _, lab in terms)), (f3, a3, 'contrast')):
        a[0].legend(fontsize=8)
        f.suptitle(f'{title}: {sub}'); f.tight_layout(w_pad=3)
    a3[1].legend([Line2D([], [], color='k', ls='--', alpha=0.4)], [f'science Strehl $<$ {sr_min:g}'], fontsize=8)
    return [f for f, _ in figs]


def plot_noise_frame_rate(panels, ticks=(125, 250, 500, 1000, 2000)):
    """Photon, readout and total residual against the magnitude, with the frame rate on the right axis; panels: {title: DataFrame with
    noise_photon, noise_ron, total, fs (index: magnitude)}, two per row."""
    n = len(panels)
    fig, ax = plt.subplots((n + 1) // 2, min(n, 2), figsize=(13, 4.2 * ((n + 1) // 2)), squeeze=False)
    for a, (title, df) in zip(ax.flat, panels.items()):
        a.semilogy(df.index, df['noise_photon'], '-', label='photon')
        a.semilogy(df.index, df['noise_ron'], '-', label='readout')
        a.semilogy(df.index, df['total'], 'k-', label='total')
        tw = a.twinx()
        tw.semilogy(df.index, df['fs'], '-.', color='C3', label='optimal frame rate (right axis)')
        tw.set_yticks(list(ticks), [f'{t:g}' for t in ticks]); tw.minorticks_off(); tw.set_ylim(0.8 * min(ticks), 1.25 * max(ticks))
        tw.set_ylabel('optimal frame rate [Hz]', color='C3'); tw.tick_params(axis='y', colors='C3')
        a.set_xlabel('magnitude at 900 nm'); a.set_ylabel('residual [nm rms]'); a.grid(alpha=0.3); a.set_title(title)
    fig.legend(*[sum(x, []) for x in zip(a.get_legend_handles_labels(), tw.get_legend_handles_labels())], loc='lower center', ncol=4,
               fontsize=8)
    fig.tight_layout(w_pad=3, rect=(0, 0.07 / ((n + 1) // 2), 1, 1))
    return fig


def plot_controller(results, title, extra_og=None, mark=None):
    """Integrator gain, forgetting factor and optical gain per mode of BudgetResults {label: result} (two figures: gain | forgetting
    factor, optical gain); extra_og: {label: per-mode optical gains} drawn dashed; mark: mode index of a dotted vertical line."""
    f1, a = plt.subplots(1, 2, figsize=(13, 4.2))
    f2, b = plt.subplots(1, 1, figsize=(6.5, 4.2))
    for i, (name, r) in enumerate(results.items()):
        x = np.arange(1, len(r.gain) + 1)
        a[0].semilogx(x, r.gain, '-', lw=1, color=f'C{i}', label=name)
        a[1].semilogx(x, r.ff, '-', lw=1, color=f'C{i}', label=name)
        b.semilogx(x, r.og, '-', lw=1, color=f'C{i}', label=name)
    for j, (name, og) in enumerate((extra_og or {}).items()):
        b.semilogx(np.arange(1, len(og) + 1), og, ['--', ':'][j % 2], lw=1, color='k', label=name)
    a[1].set_ylim(0.6, 1.02); b.set_ylim(0, 1.1)
    for ax_, yl in zip((*a, b), ('integrator gain', 'forgetting factor', 'optical gain')):
        ax_.set_xlabel('mode index'); ax_.set_ylabel(yl); ax_.grid(alpha=0.3, which='both'); ax_.legend(fontsize=8)
        if mark:
            ax_.axvline(mark, color='gray', ls=':', lw=1)
    f1.suptitle(title); f1.tight_layout(w_pad=3)
    f2.suptitle(title); f2.tight_layout()
    return f1, f2


def plot_chromatic_limit(band, mags, ao_contrast, ao_label, chrom_now, zenith_now, wl, cc, floors, req, wfs_nm, sci_nm):
    """Chromatic limit of one band, two figures: [AO-only contrast against the magnitude with the chromatic term alone at zenith_now |
    chromatic term alone against the WFS wavelength wl, a curve per zenith angle (cc: {z: contrasts}), AO floors {mag: contrast}];
    [limit WFS wavelength against the zenith angle (req: DataFrame, index z, a column per floor)]."""
    f1, a = plt.subplots(1, 2, figsize=(13, 4.4))
    a[0].semilogy(mags, ao_contrast, '-', color='C0', label=f'AO only, {ao_label}')
    a[0].axhline(chrom_now, color='C1', ls='--', label=f'chromatic alone, z = {zenith_now:g} deg')
    zs = list(cc)
    for j, z in enumerate(zs):
        a[1].semilogy(wl, cc[z], '-', color=plt.cm.viridis(j / max(len(zs) - 1, 1)), label=f'z = {z} deg')
    for i, (mg, fl) in enumerate(floors.items()):
        a[1].axhline(fl, color='gray', ls=['-', '--', ':'][i % 3], lw=1, label=f'AO floor, mag {mg:g}')
    a[0].set_xlabel('magnitude at 900 nm'); a[0].set_ylabel(f'raw contrast 2-5 $\\lambda$/D ({sci_nm:.0f} nm)')
    a[1].set_xlabel('WFS wavelength [nm]'); a[1].set_ylabel(f'chromatic alone ({sci_nm:.0f} nm)')
    a[0].set_title(f'{band}: z = {zenith_now:g} deg'); a[1].set_title(f'{band}: science {sci_nm:.0f} nm')
    f2, b = plt.subplots(1, 1, figsize=(6.5, 4.4))
    for i, c in enumerate(req.columns):
        b.plot(req.index, req[c], marker='o', color='k', ls=['-', '--', ':'][i % 3], label=c)
    b.axhline(wfs_nm, color='C0', lw=0.8, label=f'WFS of the band ({wfs_nm:.0f} nm)')
    b.set_xlabel('zenith angle [deg]'); b.set_ylabel('limit WFS wavelength [nm]'); b.set_title(f'{band}: chromatic term below the AO floor')
    for ax_ in (*a, b):
        ax_.grid(alpha=0.3, which='both'); ax_.legend(fontsize=7)
    f1.tight_layout(w_pad=3); f2.tight_layout()
    return f1, f2


# ---- sensor fusion highlights (analyse_errorbudget.ipynb, presentation figures)
def fusion_gain(ref, fused, sr_min=0.25):
    """Two sweeps (index: magnitude at 900 nm; columns contrast, SR) -> DataFrame per magnitude of `ref`: 'ratio' = fused / ref contrast
    and 'dmag' = magnitudes gained at equal contrast (the fused system reaches the contrast of ref at magnitude m at m + dmag; log-linear
    interpolation within the fused sweep, NaN outside), 'ok' = both science Strehls >= sr_min."""
    import pandas as pd
    f = fused.dropna(subset=['contrast'])
    cf = np.maximum.accumulate(np.log(f['contrast'].to_numpy(float)))         # contrast grows with the magnitude: enforce monotonicity
    cr = np.log(ref['contrast'].to_numpy(float))
    m = ref.index.to_numpy(float)
    dmag = np.interp(cr, cf, f.index.to_numpy(float), left=np.nan, right=np.nan) - m
    ratio = fused['contrast'].reindex(ref.index).to_numpy(float) / ref['contrast'].to_numpy(float)
    ok = (ref['SR'] >= sr_min).to_numpy() & (fused['SR'].reindex(ref.index) >= sr_min).to_numpy()
    return pd.DataFrame({'ratio': ratio, 'dmag': dmag, 'ok': ok}, index=ref.index)


def plot_fusion_gains(gains, zone=(2, 5), sr_min=0.25, bands=('VIS', 'NIR')):
    """Headline figure: for every advantage (one colour) and variant (solid: main, dashed: alternative) the contrast of the fused system
    relative to its single-sensor reference (top) and the magnitudes gained at equal contrast (bottom), one column per band.
    gains: {(advantage, variant, main): {band: fusion_gain DataFrame}}; faded where a science Strehl < sr_min."""
    from matplotlib.lines import Line2D
    fig, axs = plt.subplots(2, len(bands), figsize=(6.8 * len(bands), 8), sharex=True, squeeze=False)
    colours = {a: f'C{i}' for i, a in enumerate(dict.fromkeys(a for a, _, _ in gains))}
    for (adv, var, main), per_band in gains.items():
        for j, band in enumerate(bands):
            if band not in per_band:
                continue
            g = per_band[band]
            for ax, col in zip(axs[:, j], ('ratio', 'dmag')):
                y = g[col].where(g['ok'])
                ax.plot(g.index, y, '-' if main else '--', color=colours[adv], lw=2 if main else 1.2, marker='o', ms=3,
                        label=f'{adv}: {var}')
                ax.plot(g.index, g[col].where(~g['ok']), ':', color=colours[adv], lw=1, alpha=0.5)
    for j, band in enumerate(bands):
        axs[0, j].axhline(1.0, color='gray', lw=0.8)
        axs[1, j].axhline(0.0, color='gray', lw=0.8)
        axs[0, j].set_title(band)
        axs[0, j].set_ylabel(f'contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D: fused / single')
        axs[1, j].set_ylabel('magnitudes gained at equal contrast')
        axs[1, j].set_xlabel('magnitude at 900 nm')
        for ax in axs[:, j]:
            ax.grid(alpha=0.3, which='both')
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], color='gray', ls=':')]
    l += [f'science Strehl $<$ {sr_min:g}']
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Sensor fusion against a single WFS: contrast gain per advantage')
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig


def _logbin(y, nb=50):
    """Mean of y over log-spaced bins of the mode index (1-based) -> (centres, means)."""
    e = np.unique(np.geomspace(1, len(y) + 1, nb).astype(int))
    return np.sqrt(e[:-1] * e[1:]), np.array([np.mean(y[a - 1:b - 1]) for a, b in zip(e[:-1], e[1:])])


def config_label(p):
    """Frame rates and the optimised choices of a result's params, for titles."""
    fs = f"{p['freqUsedInHz']:.0f} Hz" + (f", LO {p['freqUsedInHz_lo']:.0f} Hz" if 'freqUsedInHz_lo' in p else '')
    return fs + (f", unmod. {p['n_unmod']}x{p['n_unmod']}" if 'n_unmod' in p else '') + (f", $f$ = {p['f_LO']:g}" if 'f_LO' in p else '')


def fusion_weights(r):
    """(per-mode inverse-variance weights [n_sensors, n_modes], sensor descriptions) of a fused / dual result; (None, None) otherwise.
    For a dual system the weights cover its LO modes only."""
    w = getattr(r, 'weights', None)
    if w is None:
        return None, None
    p = r.stages[0].params if getattr(r, 'stages', None) else r.params
    return np.atleast_2d(w), p.get('sensors', '').split(' + ')


def plot_fusion_case(res, title, zone=(2, 5), ymin=1e-7, rmax=26):
    """One 'slide' per case: one row per magnitude, every configuration in its own colour (the first is the single-sensor reference).
    Columns: (1) coronagraphic raw contrast profile of the total; (2) residual per mode (lag + noise + aliasing, log-binned): which modes
    fusion improves; (3) first-order halo of every term averaged over the zone, stacked (they add): which term fusion reduces, with the
    coronagraphic total (diamond); (4) fusion weight of every sensor per mode (line style: sensor). res: {(name, mag900): (BudgetResult,
    PsfResult)}."""
    import textwrap
    names = list(dict.fromkeys(n for n, _ in res))
    mags = sorted(dict.fromkeys(mg for _, mg in res))
    fig, axs = plt.subplots(len(mags), 4, figsize=(22, 3.9 * len(mags)), squeeze=False,
                            gridspec_kw=dict(width_ratios=(1.2, 1.1, 0.9, 1.1)))
    rho = np.arange(zone[0], zone[1] + 1e-9, 0.5)
    lss = ('-', '--', '-.', ':')
    for i_row, (row, mg) in enumerate(zip(axs, mags)):
        a0, a1, a2, a3 = row
        a0.axvspan(*zone, color='gray', alpha=0.15, lw=0)
        bottoms = np.zeros(len(names))
        terms = [t for t in res[names[0], mg][1].halo if t not in ('total', 'turbulence')]
        stacks = {t: np.array([np.mean(res[n, mg][1].contrast_at(rho, name=t, kind='halo')) if t in res[n, mg][1].halo else 0.0
                               for n in names]) for t in terms}
        for t, v in stacks.items():
            if np.any(v > 0):
                a2.bar(range(len(names)), v, bottom=bottoms, color=COLORS.get(t), label=_pretty(t), width=0.6)
                bottoms += v
        for i, n in enumerate(names):
            r, p = res[n, mg]
            c, lab = f'C{i}', f'{n} ({config_label(r.params)})'
            x, y = p.profile('total', 'coro', 0.5, 'mean', rmax)
            a0.semilogy(x, y, color=c, lw=1.5, label=lab)
            a1.loglog(*_logbin(r.residual), color=c, lw=1.5, label=n)
            a2.plot(i, np.mean(p.contrast_at(rho, kind='coro')), 'D', color='k', ms=6, label='total, coronagraphic' if i == 0 else None)
            w, desc = fusion_weights(r)
            if w is not None:
                for k in range(w.shape[0]):
                    a3.semilogx(np.arange(1, w.shape[1] + 1), w[k], ls=lss[k % 4], color=c, lw=1.5, label=f'{n}: {desc[k]}')
        a0.set_ylim(bottom=ymin)
        a0.set_xlim(0.5, rmax - 1)
        a0.set(xlabel='[$\\lambda$/D]', ylabel=f'raw contrast ({p.wavelength_nm:.0f} nm)', title=f'magnitude {mg:g}: coronagraphic profile')
        a1.set(xlabel='mode index', ylabel='residual per mode [nm$^2$]', title='residual per mode (lag + noise + aliasing)')
        a2.set_xticks(range(len(names)), [textwrap.fill(n, 16) for n in names], fontsize=7)
        a2.set(ylabel=f'mean {zone[0]:g}-{zone[1]:g} $\\lambda$/D', title='first-order halo of every term in the zone')
        a2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        a3.set(xlabel='mode index', ylabel='fusion weight', ylim=(-0.03, 1.03), title='inverse-variance weight of every sensor')
        for a in row:
            a.grid(alpha=0.3, which='both')
        a0.legend(fontsize=7)
        if i_row == 0:
            a2.legend(fontsize=7)
            if a3.lines:
                a3.legend(fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_scan(df, pairs, xlabel, title, sr_min=0.25, zone=(2, 5), logx=False, xref=None, secondary=None):
    """Fused against single-sensor configurations over a parameter (Study.scan DataFrame): one column per pair {label: (reference, fused)};
    top: raw contrast in the zone of the reference (dashed) and of the fused system (solid), bottom: fused / reference; one colour per
    magnitude, faded where a science Strehl < sr_min. xref: the nominal value (dotted line); secondary: (label, f, f_inverse) of a
    second x axis on top."""
    from matplotlib.lines import Line2D
    mags = sorted(df['mag'].unique())
    fig, axs = plt.subplots(2, len(pairs), figsize=(5.4 * len(pairs), 7.5), sharex=True, squeeze=False)
    for j, (lab, (ref, fu)) in enumerate(pairs.items()):
        top, bot = axs[:, j]
        for i, mg in enumerate(mags):
            d = df[df['mag'] == mg]
            a, b = (d[d['config'] == k].set_index('x').sort_index() for k in (ref, fu))
            c = f'C{i}'
            for frame, ls in ((a, '--'), (b, '-')):
                ok = frame['SR'] >= sr_min
                top.plot(frame.index, frame['contrast'], ls, color=c, lw=1, alpha=0.25)
                top.plot(frame.index, frame['contrast'].where(ok), ls, color=c, lw=1.8, marker='o', ms=3)
            ratio, ok = b['contrast'] / a['contrast'], (a['SR'] >= sr_min) & (b['SR'] >= sr_min)
            bot.plot(ratio.index, ratio, '-', color=c, lw=1, alpha=0.25)
            bot.plot(ratio.index, ratio.where(ok), '-', color=c, lw=1.8, marker='o', ms=3, label=f'mag {mg:g}')
        top.set_yscale('log')
        top.set(title=f'{lab}\n{fu} vs {ref}', ylabel=f'raw contrast {zone[0]:g}-{zone[1]:g} $\\lambda$/D')
        bot.set(xlabel=xlabel, ylabel='fused / single')
        bot.axhline(1.0, color='gray', lw=0.8)
        top.title.set_fontsize(9)
        for ax in (top, bot):
            if logx:
                ax.set_xscale('log')
            if xref is not None:
                ax.axvline(xref, color='gray', ls=':', lw=1)
            ax.grid(alpha=0.3, which='both')
        if secondary is not None:
            sec = top.secondary_xaxis('top', functions=secondary[1:])
            sec.set_xlabel(secondary[0])
    axs[1, 0].legend(fontsize=8)
    axs[0, 0].legend([Line2D([], [], color='k', ls='--'), Line2D([], [], color='k')], ['single WFS', 'fused'], fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
