"""Plots (matplotlib figures are returned, never saved): budget terms / comparisons / per-mode quantities from the scan
DataFrames (errors in nm rms), calibration products, and PSD / PSF of a BudgetResult (psf.PsfEngine)."""
import matplotlib.pyplot as plt
import numpy as np

from . import psf as psflib
from . import utils

COLORS = {'lag': 'C0', 'noise': 'C1', 'noise_photon': 'C1', 'noise_ron': 'goldenrod', 'alias': 'C2', 'fitting': 'C3',
          'total': 'k', 'xtalk': 'C4', 'chrom_shear': 'C5', 'chrom_mult': 'C6', 'chrom_fresnel': 'C7', 'scintillation': 'teal',
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


def _finish(fig, ax, xlabel, ylabel, title=None, logx=False, logy=False):
    ax.set_xlabel(LABELS.get(xlabel, xlabel))
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xscale('log' if logx else 'linear')
    ax.set_yscale('log' if logy else 'linear')
    ax.grid(True, which='both', alpha=0.3)
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


def _label(name, h):
    return f'{name}={h:g}' if isinstance(h, (int, float, np.floating)) else f'{name}={h}'


def plot_budget_terms(totals, x='photPerMs', filters=None, terms=('lag', 'noise', 'alias', 'fitting', 'total'), logx=True, logy=False,
                      title=None, ax=None):
    """Budget terms [nm rms] versus x for the configuration fixed by `filters` (the terms add in quadrature to the total)."""
    df = _select(totals, filters).sort_values(x)
    _one_per(df, x)
    fig, ax = _fig(ax)
    for t in terms:
        ax.plot(df[x], _value(df, t), '-o' if t == 'total' else '-', ms=3, color=COLORS.get(t), label=t,
                lw=2 if t == 'total' else 1.5)
    ax.legend()
    return _finish(fig, ax, x, 'error [nm rms]', title, logx, logy)


def plot_budget_compare(totals, x='photPerMs', hue='freqInHz', quantity='total', filters=None, col=None, logx=True, logy=False,
                        title=None):
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
        _finish(fig, ax, x, f"{quantity}{' [nm rms]' if quantity in ERRORS else ''}", f'{col} = {cv}' if cv is not None else title, logx, logy)
    return fig


def plot_per_mode(modes, y='residual', hue=None, filters=None, logy=True, title=None):
    """Per-mode quantity versus mode index (error columns in nm rms; gain, ff, og raw)."""
    df = _select(modes, filters)
    fig, ax = _fig(None)
    for i, (h, g) in enumerate([(None, df)] if hue is None else df.groupby(hue)):
        g = g.sort_values('modeId')
        ax.plot(g['modeId'], _value(g, y), '.', ms=3, color=f'C{i % 10}', label=None if h is None else _label(hue, h))
    if hue is not None:
        ax.legend(fontsize=8)
    return _finish(fig, ax, 'modeId', f'{y} [nm rms]' if y in ERRORS else y, title, False, logy)


def plot_calibration(machine, product, hue=None, values=None, rMod=3.0, n_subap=48, seeing=0.9, N=None, N_corr=None, flux=1e7,
                     fs=1000.0, wavelength_nm=None, logy=None, title=None):
    """'alias' | 'og' | 'ct' | 'noise' | 'turb' versus mode (or 'thrp' versus `hue`); one curve per value of `hue`
    in {rMod, n_subap, seeing, N, wavelength_nm}."""
    m, base = machine, dict(rMod=rMod, n_subap=n_subap, seeing=seeing, N=N, wavelength_nm=wavelength_nm)
    fig, ax = _fig(None)
    xs, ys = [], []
    for i, v in enumerate([None] if hue is None else values):
        cfg = {**base, **({} if hue is None else {hue: v})}
        N_i = cfg['N'] or m.modes_per_subap[cfg['n_subap']]
        cal = m.calibration(cfg['rMod'], cfg['n_subap'], cfg['seeing'], N_i, N_corr, wavelength_nm=cfg['wavelength_nm'])
        if product == 'thrp':
            xs.append(v)
            ys.append(cal.thrp)
            continue
        y = {'alias': lambda: cal.alias_rms, 'og': lambda: cal.og, 'ct': lambda: cal.ct,
             'noise': lambda: np.sqrt(cal.noise.modal_variance(m.detector.n_det(flux, fs), m.detector)),
             'turb': lambda: np.sqrt(m.turbulence(cfg['seeing'], N_i)[0])}[product]()
        ax.plot(np.arange(len(y)), y, '.', ms=3, color=f'C{i % 10}', label=None if hue is None else _label(hue, v))
    if product == 'thrp':
        ax.plot(xs, ys, '-o')
        return _finish(fig, ax, hue or 'rMod', 'throughput (fraction of counts in the pupils)', title)
    if hue is not None:
        ax.legend(fontsize=8)
    unit = {'alias': 'aliasing std [command units]', 'og': 'optical gain', 'ct': 'crosstalk',
            'noise': f'noise rms [command units] ({flux:.0e} ph/s, {fs:.0f} Hz)', 'turb': 'turbulence rms [command units]'}[product]
    return _finish(fig, ax, 'modeId', unit, title, False, product in ('alias', 'noise', 'turb') if logy is None else logy)


def plot_modal_variance_check(machine, rMod=3.0, n_subap=48, seeing=0.9, N=None, wavelength_nm=None, title=None):
    """Simulated (turbmodes of the aliasing run) over analytical modal variance per mode."""
    m = machine
    N = N or m.modes_per_subap[n_subap]
    sim = utils.read_fits(m.paths.at(m.wavelength_nm if wavelength_nm is None else wavelength_nm).turbmodes(rMod, n_subap, seeing, N))
    ratio = utils.compare_modal_variance(sim, m.turbulence(seeing, sim.shape[1])[0])
    fig, ax = _fig(None)
    ax.plot(ratio, '.', ms=3)
    ax.axhline(1.0, color='k', lw=1)
    s = np.sqrt(2.0 / sim.shape[0])
    ax.axhspan(1 - s, 1 + s, color='gray', alpha=0.2, label='1 sigma statistical')
    ax.legend()
    return _finish(fig, ax, 'modeId', 'simulated / analytical variance', title or f'{N} modes, seeing {seeing}"')


def plot_psd_terms(psd_set, terms=None, xaxis='lamD', show_open_loop=True, title=None, ax=None):
    """Radial PSD [nm^2 m^2] of every term and the total; x = separation k D [lambda/D] or k [cycles/m]; dotted: DM cut-off."""
    fig, ax = _fig(ax, (7, 4.8))
    sc = psd_set.D if xaxis == 'lamD' else 1.0
    for n, t in psd_set.terms.items():
        if (terms is not None and n not in terms and n != 'total') or (n == 'turbulence' and not show_open_loop):
            continue
        ok = (psd_set.k > 0) & (t.psd > 0)
        ax.plot(psd_set.k[ok] * sc, t.psd[ok], color=COLORS.get(n), lw=2.2 if n == 'total' else 1.4,
                ls=':' if n == 'turbulence' else '-', label=n)
    if psd_set.meta.get('k_cut'):
        ax.axvline(psd_set.meta['k_cut'] * sc, color='k', ls=':', lw=0.8)
    ax.legend(ncol=2, fontsize=8)
    return _finish(fig, ax, 'separation [$\\lambda$/D]  (= k D)' if xaxis == 'lamD' else 'spatial frequency [cycles/m]',
                   'PSD [nm$^2$ m$^2$]', title, True, True)


def plot_psf_terms(psf_result, kind='psf', terms=None, binsize=0.5, rmax=None, stat='mean', pa_range=None, show_dl=True,
                   show_open_loop=False, title=None, ax=None):
    """Radial contrast profile of every term alone and the total; kind='halo': first-order halos (they add up)."""
    fig, ax = _fig(ax, (7, 4.8))
    for n in (psf_result.psf if kind == 'psf' else psf_result.halo):
        if (terms is not None and n not in terms and n != 'total') or (n == 'turbulence' and not show_open_loop):
            continue
        r, p = psf_result.profile(n, kind, binsize, stat, rmax, pa_range)
        ax.plot(r, p, color=COLORS.get(n), lw=2.2 if n == 'total' else 1.4, ls=':' if n == 'turbulence' else '-',
                label=f'{n} (S={psf_result.strehl[n]:.2f})' if kind == 'psf' else n)
    if show_dl:
        r, p = psflib.radial_profile(psf_result.diffraction_limited, psf_result.pixel_scale, binsize, None, stat, rmax, pa_range)
        ax.plot(r, p, color='gray', lw=1, label='diffraction limit')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 1e-9))
    return _finish(fig, ax, 'separation [$\\lambda$/D]', f'raw contrast at {psf_result.wavelength_nm:.0f} nm', title, False, True)


def plot_psf_image(psf_result, name='total', half_fov=20.0, vmin=1e-7, title=None, ax=None):
    """2-D long-exposure PSF (log colour scale), field +-half_fov lambda/D."""
    img = psf_result.psf[name]
    M, h = img.shape[0], int(half_fov / psf_result.pixel_scale)
    fig, ax = _fig(ax, (5.5, 5))
    im = ax.imshow(np.log10(np.maximum(img[M // 2 - h:M // 2 + h + 1, M // 2 - h:M // 2 + h + 1], vmin)), origin='lower',
                   extent=[-half_fov, half_fov, -half_fov, half_fov], vmin=np.log10(vmin), vmax=0, cmap='inferno')
    fig.colorbar(im, ax=ax, label='log10 contrast')
    ax.set_xlabel('$\\lambda$/D')
    ax.set_ylabel('$\\lambda$/D')
    ax.set_title(title or f'{name}, Strehl {psf_result.strehl[name]:.3f}')
    fig.tight_layout()
    return fig
