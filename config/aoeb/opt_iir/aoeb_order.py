"""
Residual per KL mode for the integrator, the leaky integrator, the order-2 IIR Wiener filter and the Wiener bound (as iir_study.py
`order`), with the inputs of the aoeb error budget on the e2e (WFSW) products.

  cd SPECULA/config; python aoeb/opt_iir/aoeb_order.py        # the two figures, next to this file
  python aoeb/opt_iir/aoeb_order.py tf [mode [mag [delay]]]   # PSDs, |RTF|^2, |NTF|^2 of one mode (default 5, 8, 1.5)

Sensor: pyramid rMod RMOD (env AOEB_ORDER_RMOD: 0 unmodulated, default; 1), 48x48, 900 nm, 1300 modes, seeing 0.9", fs 2 kHz (fixed), magnitudes at 900 nm.
MODES: 30 KL modes log-spaced over 1..1150 (numbered from 1: 1 = tip; MODE_MAX below). Each mode is designed on ITS OWN values (nothing averaged), from
the aoeb machine (aoeb/e2e calibration.machine(), budget_wfsw.yml):
  turbulence variance (KL von Karman, r0 of the seeing); temporal PSD of its radial order (TEMPORAL: exact multi-layer frozen-flow von
  Karman spectra of the 10 ristretto_main layers, aoeb.temporal.ZernikePSD; or Conan with one effective wind); noise (photon + readout, measured slope-noise propagation, detector of budget_wfsw.yml) / og^2; aliasing (measured
  modal std, perfect correction of 1300 modes) / og^2 with the temporal shape of its radial order MEASURED in the aliasing PSD
  study (ALIAS; aoeb MeasuredAliasSpectrum, A10b) or aoeb's AliasSpectrum.
Designs in nm^2 (OPD); the figures show the rms per mode [nm]. Optical gain known (POL exact), no modulus margin (as the error-budget
notebook, max_sensitivity = None). Total delay = 1 + tau frames (tau = aoeb latency in frames).
Figures: (1) total delay 2 frames, magnitudes 5 and 11, all controllers; (2) magnitude 8, total delays 1.5, 2, 2.5 frames, leaky and IIR2.
Checked (rMod 1) against aoeb's own budget on the same modes: leaky within 1 %, IIR2 vs controller='wiener' within 3 %.
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # before numpy: single-threaded BLAS in every worker (the designs are parallel over tasks)
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, CONFIG, os.path.join(CONFIG, 'aoeb', 'e2e')]

RMOD = float(os.environ.get('AOEB_ORDER_RMOD', 0.0))           # modulation [lambda/D]: 0 (unmodulated) or 1
TEMPORAL = os.environ.get('AOEB_ORDER_TEMPORAL', 'zernike')    # 'zernike': multi-layer frozen flow (default); 'conan': one effective wind
ALIAS = os.environ.get('AOEB_ORDER_ALIAS', 'measured')          # 'measured': per-radial-order measured aliasing PSDs (A10b); 'model': AliasSpectrum
# measured aliasing runs per modulation (frozen flow, 10-layer ristretto profile, 0.9", 48x48, 2 kHz, 8 s; wind directions rotated by
# 0/45/90/135 deg, 4 seeds each at N 1300, one per rotation at N 300): results/WFSW/alias_psd_rmod<r>/modes (run_rmod<r>.py)
_A = '/raid1/mmenessini/results/WFSW/alias_psd_rmod{r:g}/modes/r{r:g}_'
ALIAS_PSD = {r: {1300: [_A.format(r=r) + f'rot{a}_seed{s}.npz' for a in (0, 45, 90, 135) for s in (1001, 1101, 1201, 1301)],
                 300: [_A.format(r=r) + f'N300_rot{a}.npz' for a in (0, 45, 90, 135)]} for r in (0.0, 1.0)}
NSUB, WL, SEEING, N, FS = 48, 900.0, 0.9, 1300, 2000.0
MODE_MAX, N_PLOT = 1150, 30
# MODE_MAX: the measured aliasing of the controlled modes grows towards the cut-off (rMod 1: 0.09 nm^2 at 779, ~0.25 at 1100-1150, ~0.5
# above 1200, 0.86 at 1300; rMod 0: ~0.1 up to 900, ~0.2-0.27 at 1050-1150, ~0.4-0.55 above 1200), which makes the last modes jump


def log_modes(n, j_max):
    """n distinct KL mode numbers (from 1), as close to log-spaced over 1..j_max as integers allow (consecutive at the start)."""
    k = n
    while len(np.unique(np.round(np.geomspace(1, j_max, k)))) < n:
        k += 1
    return [int(j) for j in np.unique(np.round(np.geomspace(1, j_max, k)))][:n - 1] + [j_max]


ALL_MODES = os.environ.get('AOEB_ORDER_ALL', '') == '1'          # every mode 1..N (figures *_all.png) instead of the N_PLOT log-spaced ones
MODES = list(range(1, N + 1)) if ALL_MODES else log_modes(N_PLOT, MODE_MAX)
FIGURES = {'delay2': ([(2.0, 5.0), (2.0, 11.0)], (0, 1, 2, 3)),   # name: ([(total delay [frames], magnitude at 900 nm)]: one line
           'mag8': ([(1.5, 8.0), (2.0, 8.0), (2.5, 8.0)], (1, 2))}  # style each, controllers shown (indices in LABELS))
LABELS = ['Integrator (leak = 1)', 'Leaky integrator', 'IIR Wiener, N=2', 'Wiener bound (N = ∞)']
COLORS, MARKERS = ['0.5', 'k', 'tab:orange', 'tab:red'], ['v', 'o', 's', '^']
CACHE = os.path.join(HERE, f'aoeb_order_cache_rmod{RMOD:g}_{TEMPORAL}' + ('_measalias2' if ALIAS == 'measured' else '') + '.json')   # measalias2: rotated per-rMod runs   # one entry per (delay, magnitude, mode): resumable
N_JOBS = int(os.environ.get('AOEB_WIENER_JOBS', 12))


def inputs():
    """Per mode number j: radial order, turbulence / noise (per magnitude) / aliasing [nm^2], og; wind, D, alias spectrum."""
    import yaml, importlib.util
    from specula.lib.n_phot import n_phot
    from aoeb import utils
    spec = importlib.util.spec_from_file_location('aoeb_e2e_utils', os.path.join(CONFIG, 'aoeb', 'e2e', 'utils.py'))
    e2e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e)
    m = e2e.machine()
    if ALIAS == 'measured':
        m.set_alias_psd(ALIAS_PSD)
    Y = yaml.safe_load(open(os.path.join(CONFIG, 'aoeb', 'e2e', 'ristretto_main.yml')))
    area = (m.pupil() > 0).sum() * Y['main']['pixel_pitch'] ** 2
    cal = m.calibration(RMOD, NSUB, SEEING, N, N, ('svd',), WL)
    turb, rms, _ = m.turbulence(SEEING, N)
    alias = cal.alias_rms ** 2 * (SEEING / cal.alias_seeing) ** (5 / 3)
    order = utils.radial_order(np.arange(N))
    at = lambda v: {j: float(v[j - 1]) for j in MODES}
    noise = {}
    for mg in sorted({mg for cases, _ in FIGURES.values() for _, mg in cases}):
        flux = n_phot(mg, lambda_=WL * 1e-9, width=1e-9)[0] * Y['cred_vis']['bandw'] * area      # [ph/s], as Study.flux_of
        ph, ro = cal.noise.modal_variance_parts(m.detector.n_det(flux, FS), m.detector)
        noise[mg] = at((ph + ro) / cal.og ** 2 * rms ** 2)
    if TEMPORAL == 'zernike':        # exact frozen-flow von Karman PSDs of the radial orders, 10 layers (Cn2, speed) of ristretto_main
        from aoeb.temporal import ZernikePSD
        wind = ZernikePSD(m.D, L0=m.L0, layers=m.alias_layers)
        for n in sorted({int(order[j - 1]) for j in MODES}):
            wind.unit_psd(1.0, n)     # tables built once here (the workers get them with the task)
        tlabel = f'frozen flow, {len(m.alias_layers)} layers of ristretto_main'
    else:
        wind, tlabel = m.wind, f'Conan, wind {m.wind:.1f} m/s'
    tlabel += (f'; aliasing PSD measured per radial order (rMod {RMOD:g}, rotated winds)' if ALIAS == 'measured' else '; aliasing PSD AliasSpectrum')
    return dict(tlabel=tlabel, order={j: int(order[j - 1]) for j in MODES}, turb=at(turb * rms ** 2), alias=at(alias / cal.og ** 2 * rms ** 2),
                og=at(cal.og), noise=noise, wind=wind, D=m.D, alias_spec=m.alias_spectrum(WL, N, RMOD))


def make_spec(total, n, turb, noise, alias, wind, D, alias_spec):
    """Folded spectra of one mode [nm^2] (the inputs of a task)."""
    from aoeb import filter_optimizers as fo
    from wiener_ao_iir import ModeSpectra
    return ModeSpectra(lambda f: turb * fo.unit_psd(f, n, wind, D), FS, total - 1.0,
                       lambda f: alias * alias_spec.unit_psd(f, n), noise=noise,
                       M=2 ** 15 if n < 4 else 2 ** 13)       # 2^13 within 0.1 % of 2^15 for n >= 4


def design(spec):
    """Controllers [(structure, b, a, C_grid, residual [nm^2])]: integrator, leaky integrator, IIR2, Wiener bound (as LABELS)."""
    from wiener_ao_iir import wiener_iir
    from ao_tools import optimal_integrator, optimal_leaky_integrator
    Gi, ji = optimal_integrator(spec)
    (G, l), jl = optimal_leaky_integrator(spec)
    info = wiener_iir(spec, 2)[2]
    b, a, j2 = info['lower'][1]
    return [('residual', [Gi], [1.0, -1.0], None, ji), ('residual', [G], [1.0, -l], None, jl), ('pol', b, a, None, j2),
            ('pol', None, None, info['F_bound'], info['bound'])]


def task(args):
    """Residuals [int, leaky, IIR2, bound] [nm^2] of one (delay, magnitude, mode)."""
    return [c[4] for c in design(make_spec(*args))]


def key(t, mg, j):
    return f'{t:g}|{mg:g}|{j}'


def compute():
    """{key: residuals} of every (delay, magnitude, mode) of FIGURES, and the inputs. Each design is cached as soon as it is done."""
    I = inputs()
    tag = repr((RMOD, NSUB, WL, SEEING, N, FS, 'IIR2'))
    done = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    done = done if done.get('tag') == tag else {'tag': tag}
    cases = list(dict.fromkeys(c for cs, _ in FIGURES.values() for c in cs))
    jobs = {key(t, mg, j): (t, I['order'][j], I['turb'][j], I['noise'][mg][j], I['alias'][j], I['wind'], I['D'], I['alias_spec'])
            for t, mg in cases for j in sorted(MODES, key=lambda j: -(I['order'][j] < 4))}          # slow low orders first
    todo = {k: v for k, v in jobs.items() if k not in done}
    print(f'{len(jobs) - len(todo)} cached, {len(todo)} to design on {N_JOBS} workers', flush=True)
    if todo:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(min(len(todo), N_JOBS)) as pool:
            for i, (k, r) in enumerate(zip(todo, pool.imap(task, todo.values(), chunksize=1))):
                done[k] = r
                json.dump(done, open(CACHE, 'w'))
                print(f'[{i + 1}/{len(todo)}] {k}: ' + ' '.join(f'{x:.3g}' for x in r), flush=True)
    return done, I


def plot(done, I):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, FormatStrFormatter, NullLocator, NullFormatter
    ls = ['-', '--', ':']
    ogs = list(I['og'].values())
    head = (f'Pyramid rMod {RMOD:g}, {NSUB}x{NSUB}, {WL:.0f} nm, {N} modes, seeing {SEEING:g}", fs = {FS:g} Hz; aoeb e2e products '
            f'({I["tlabel"]}; measured noise, aliasing, og {min(ogs):.2f}-{max(ogs):.2f})')
    out = []
    for name, (cases, show) in FIGURES.items():
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for k, (t, mg) in enumerate(cases):
            res = np.sqrt(np.array([done[key(t, mg, j)] for j in MODES]))
            for c in show:
                ax.plot(MODES, res[:, c], ls[k], color=COLORS[c], lw=0.8 if ALL_MODES else 1.2, mfc='none',
                        **({} if ALL_MODES else dict(marker=MARKERS[c], ms=3)))
        ax.set_xscale('log'); ax.set_yscale('log')
        ylim = ax.get_ylim()                     # the controllers set the limits
        ax.plot(MODES, np.sqrt([I['alias'][j] for j in MODES]), color='tab:purple', lw=1.0, zorder=0)   # aliasing rms / og
        ax.set_ylim(ylim)
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set(xlabel='KL mode', ylabel='Residual per mode [nm rms]')
        same_t = len({t for t, _ in cases}) == 1
        ax.set_title(f'Total delay {cases[0][0]:g} frames' if same_t else f'Magnitude {cases[0][1]:g}')
        ax.grid(True, which='both', alpha=0.3)
        h = [Line2D([], [], color=COLORS[c], lw=1.5, marker=MARKERS[c], mfc='none') for c in show]
        h += [Line2D([], [], color='gray', ls=s) for s in ls[:len(cases)]] + [Line2D([], [], color='tab:purple', lw=1.0)]
        lab = [LABELS[c] for c in show] + [f'mag {mg:g}' if same_t else f'total delay {t:g} frames' for t, mg in cases] + ['aliasing (rms / og, input)']
        ax.legend(h, lab, fontsize=8, ncol=2)
        fig.suptitle(head, fontsize=8, wrap=True)
        fig.tight_layout()
        fname = os.path.join(HERE, f'residual_vs_mode_aoeb_rmod{RMOD:g}_{name}' + ('_all' if ALL_MODES else '') + '.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        out.append(fname)
    return out


def plot_tf(j=5, mg=8.0, total=1.5, n_pts=3000):
    """As iir_study.plot_tf for KL mode j at magnitude mg and total delay [frames], with the aoeb inputs: left, disturbance PSD and |RTF|^2;
    right, noise and (folded, WFS-averaged) aliasing PSDs and |NTF|^2. The controllers are redesigned with the code of the sweep and
    their residuals are checked against the cache (when the case is cached)."""
    import matplotlib.pyplot as plt
    from aoeb import filter_optimizers as fo
    from ao_tools import transfer_functions
    I = inputs() if j in MODES else None
    if I is None:
        raise ValueError(f'mode {j} is not in MODES')
    n = I['order'][j]
    spec = make_spec(total, n, I['turb'][j], I['noise'][mg][j], I['alias'][j], I['wind'], I['D'], I['alias_spec'])
    ctrls = design(spec)
    done = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    if key(total, mg, j) in done:
        ref = np.array(done[key(total, mg, j)])
        new = np.array([c[4] for c in ctrls])
        print('residuals [nm^2] redesigned', np.round(new, 5).tolist(), '| cached', np.round(ref, 5).tolist(),
              '| max rel diff %.1e' % np.max(np.abs(new / ref - 1)))
    nu = np.fft.fftfreq(spec.M, 1 / FS)
    idx = np.nonzero(nu > 0)[0]
    idx = np.unique(idx[np.round(np.geomspace(1, len(idx), n_pts)).astype(int) - 1])
    f = nu[idx]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    def area(ax, y, color, label):
        ax.fill_between(f, y, 1e-300, color=color, alpha=0.18, lw=0, label=label)
        ax.loglog(f, y, color=color, lw=0.8, alpha=0.6)
        return y

    St = area(axL, I['turb'][j] * fo.unit_psd(f, n, I['wind'], I['D']), 'tab:gray', f'Disturbance PSD (mode {j}, n={n})')
    axL.set_ylim(St.min() / 10, St.max() * 10)
    axL.set(xlabel='Frequency [Hz]', ylabel='PSD [nm²/Hz]', title='Disturbance PSD and |RTF|²')
    Sn = area(axR, 2 * spec.Pnn[idx] / FS, 'tab:gray', 'Noise PSD')
    Sa = area(axR, 2 * spec.Paa[idx] / FS, 'tab:purple', 'Aliasing PSD (folded, WFS-averaged' + (', measured)' if ALIAS == 'measured' else ')'))
    v = np.r_[Sn, Sa]
    v = v[v > v.max() * 1e-8]                     # the aoeb aliasing spectrum is zero above its cut-off: keep the axis readable
    axR.set_ylim(v.min() / 10, v.max() * 10)
    axR.set(xlabel='Frequency [Hz]', ylabel='PSD [nm²/Hz]', title='Noise & aliasing PSD and |NTF|²')
    axL2, axR2 = axL.twinx(), axR.twinx()
    for c, (st, b, a, C, r) in enumerate(ctrls):
        E, Nt, _ = transfer_functions(b, a, f, FS, spec.delay, st, C=None if C is None else C[idx])
        kw = dict(color=COLORS[c], lw=1.3, ls=':' if c == len(ctrls) - 1 else '-')
        axL2.loglog(f, np.abs(E) ** 2, label=f'{LABELS[c]}  (σ² = {r:.3e} nm²)', **kw)
        axR2.loglog(f, np.abs(Nt) ** 2, label=LABELS[c], **kw)
    axL2.set_ylabel('|RTF|²'); axR2.set_ylabel('|NTF|²')
    for ax, ax2, loc in ((axL, axL2, 'lower right'), (axR, axR2, 'lower left')):
        ax.grid(True, which='both', alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, fontsize=8, loc=loc)
    fig.suptitle(f'Pyramid rMod {RMOD:g}, {NSUB}x{NSUB}, {WL:.0f} nm, seeing {SEEING:g}", fs = {FS:g} Hz; aoeb e2e products ('
                 f'{I["tlabel"]})\nKL mode {j} (radial order {n}), mag {mg:g}: noise {I["noise"][mg][j]:.3g} nm²/frame, aliasing '
                 f'{I["alias"][j]:.3g} nm², turbulence {I["turb"][j]:.3g} nm², og {I["og"][j]:.2f}; total delay {total:g} frames',
                 fontsize=9)
    fig.tight_layout()
    fname = os.path.join(HERE, f'psd_rtf_ntf_aoeb_rmod{RMOD:g}_mode{j}_mag{mg:g}_delay{total:g}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_ratios(mags=(5.0, 8.0, 11.0), total=2.0, mg_delay=8.0, delays=(1.5, 2.0, 2.5), n_bins=25):
    """From the cache only (all its modes of the cases below). Left: rms ratio leaky / IIR2 per mode at `total` for `mags`. Right: rms
    residual at each delay over that at the first delay, leaky and IIR2, at `mg_delay`. Thin: per mode; bold: median in log bins."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, FormatStrFormatter, NullFormatter
    done = json.load(open(CACHE))
    have = lambda t, mg: [j for j in range(1, N + 1) if key(t, mg, j) in done]
    js = sorted(set.intersection(*[set(have(total, mg)) for mg in mags], *[set(have(t, mg_delay)) for t in delays]))
    js = np.array(js)
    r = lambda t, mg, c: np.sqrt(np.array([done[key(t, mg, j)][c] for j in js]))
    e = np.geomspace(js.min(), js.max() + 1, n_bins + 1)
    b = np.digitize(js, e) - 1

    def binned(y):
        ok = [k for k in range(n_bins) if np.any(b == k)]
        return np.array([np.exp(np.mean(np.log(js[b == k]))) for k in ok]), np.array([np.median(y[b == k]) for k in ok])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    cm = ['tab:blue', 'tab:green', 'tab:red']
    for mg, c in zip(mags, cm):
        y = r(total, mg, 1) / r(total, mg, 2)
        a1.plot(js, y, color=c, lw=0.4, alpha=0.35)
        a1.plot(*binned(y), color=c, lw=2.0, label=f'mag {mg:g}')
    a1.set(title=f'Leaky integrator / IIR2 (residual rms), total delay {total:g} frames', ylabel='ratio')
    a1.legend(fontsize=9)
    ls = {delays[0]: '-', delays[1]: '--', delays[2]: ':'}
    for c, col, lab in ((1, 'k', 'Leaky integrator'), (2, 'tab:orange', 'IIR Wiener, N=2')):
        ref = r(delays[0], mg_delay, c)
        for t in delays[1:]:
            y = r(t, mg_delay, c) / ref
            a2.plot(js, y, color=col, lw=0.4, alpha=0.35, ls=ls[t])
            a2.plot(*binned(y), color=col, lw=2.0, ls=ls[t])
    a2.axhline(1.0, color='0.5', lw=0.8)
    h = [Line2D([], [], color='k', lw=2), Line2D([], [], color='tab:orange', lw=2)] + [Line2D([], [], color='gray', lw=2, ls=ls[t]) for t in delays[1:]]
    a2.legend(h, ['Leaky integrator', 'IIR Wiener, N=2'] + [f'total delay {t:g} / {delays[0]:g} frames' for t in delays[1:]], fontsize=9)
    a2.set(title=f'Residual rms at each delay / at {delays[0]:g} frames, mag {mg_delay:g}', ylabel='ratio')
    for ax in (a1, a2):
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('KL mode')
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'Pyramid rMod {RMOD:g}, {NSUB}x{NSUB}, {WL:.0f} nm, seeing {SEEING:g}", fs = {FS:g} Hz; {len(js)} modes; thin: per mode, '
                 f'bold: median in log bins of KL mode', fontsize=9)
    fig.tight_layout()
    fname = os.path.join(HERE, f'ratios_aoeb_rmod{RMOD:g}.png')
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def table(total=2.0, mags=(5.0, 8.0, 11.0), cache=None):
    """Root sum of squares over all N controlled modes [nm rms] per controller at `total` and `mags`, from a cache (default CACHE) whose
    cases are complete; markdown rows. The fitting error (modes above N) is not included."""
    done = json.load(open(cache or CACHE))
    rows = ['| mag | integrator | leaky | IIR2 | Wiener bound | leaky / IIR2 |', '|---|---|---|---|---|---|']
    for mg in mags:
        ks = [key(total, mg, j) for j in range(1, N + 1)]
        if not all(k in done for k in ks):
            rows.append(f'| {mg:g} | incomplete ({sum(k in done for k in ks)}/{N} modes) |')
            continue
        v = np.sqrt(np.sum([done[k] for k in ks], axis=0))
        rows.append(f'| {mg:g} | {v[0]:.1f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.1f} | {v[1] / v[2]:.2f} |')
    return '\n'.join(rows)


if __name__ == '__main__':
    if sys.argv[1:2] == ['table']:           # python aoeb_order.py table [cache.json]: total residual per controller
        print(table(cache=sys.argv[2] if len(sys.argv) > 2 else None))
    elif sys.argv[1:2] == ['ratios']:          # python aoeb_order.py ratios: from the cache only
        print(plot_ratios())
    elif sys.argv[1:2] == ['tf']:               # python aoeb_order.py tf [mode [mag [total delay]]]
        print(plot_tf(*[int(x) for x in sys.argv[2:3]], *map(float, sys.argv[3:5])))
    else:
        done, I = compute()
        for f in plot(done, I):
            print(f)
