"""
Residual per KL mode for the integrator, the leaky integrator, the order-2 IIR Wiener filter and the Wiener bound (as iir_study.py
`order`), with the inputs of the aoeb error budget on the e2e (WFSW) products.

  cd SPECULA/config; python aoeb/opt_iir/aoeb_order.py        # the two figures, next to this file

Sensor: pyramid rMod 1, 48x48, 900 nm, 1300 modes, seeing 0.9", fs 2 kHz (fixed), magnitudes at 900 nm.
MODES: 15 KL modes log-spaced over 1..1300, the last replaced by 1150 (numbered from 1: 1 = tip). Each mode is designed on ITS OWN values (nothing averaged), from
the aoeb machine (aoeb/e2e calibration.machine(), budget_wfsw.yml):
  turbulence variance (KL von Karman, r0 of the seeing); temporal PSD of its radial order (machine's model: Conan, effective wind of
  ristretto_main); noise (photon + readout, measured slope-noise propagation, detector of budget_wfsw.yml) / og^2; aliasing (measured
  modal std, perfect correction of 1300 modes) / og^2 with the machine's aliasing spectrum.
Designs in nm^2 (OPD); the figures show the rms per mode [nm]. Optical gain known (POL exact), no modulus margin (as the error-budget
notebook, max_sensitivity = None). Total delay = 1 + tau frames (tau = aoeb latency in frames).
Figures: (1) total delay 2 frames, magnitudes 5 and 12, all controllers; (2) magnitude 8, total delays 1.5, 2, 2.5 frames, leaky and IIR2.
Checked against aoeb's own budget on the same modes: leaky within 1 %, IIR2 vs controller='wiener' within 3 %.
"""
import os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')      # before numpy: single-threaded BLAS in every worker (the designs are parallel over tasks)
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path[:0] = [HERE, CONFIG, os.path.join(CONFIG, 'aoeb', 'e2e')]

RMOD, NSUB, WL, SEEING, N, FS = 1.0, 48, 900.0, 0.9, 1300, 2000.0
MODES = [int(j) for j in np.unique(np.round(np.geomspace(1, N, 15)).astype(int))][:-1] + [1150]   # KL mode numbers, from 1
# 1150 instead of 1300: the measured aliasing of the controlled modes grows towards the cut-off (0.09 nm^2 at 779, ~0.25 at 1100-1150,
# ~0.5 above 1200, 0.86 at 1300, the last mode), which makes the last modes jump; 1150 is at its local median
FIGURES = {'delay2': ([(2.0, 5.0), (2.0, 12.0)], (0, 1, 2, 3)),   # name: ([(total delay [frames], magnitude at 900 nm)]: one line
           'mag8': ([(1.5, 8.0), (2.0, 8.0), (2.5, 8.0)], (1, 2))}  # style each, controllers shown (indices in LABELS))
LABELS = ['Integrator (leak = 1)', 'Leaky integrator', 'IIR Wiener, N=2', 'Wiener bound (N = ∞)']
COLORS, MARKERS = ['0.5', 'k', 'tab:orange', 'tab:red'], ['v', 'o', 's', '^']
CACHE = os.path.join(HERE, 'aoeb_order_cache.json')   # one entry per (delay, magnitude, mode): resumable
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
    return dict(order={j: int(order[j - 1]) for j in MODES}, turb=at(turb * rms ** 2), alias=at(alias / cal.og ** 2 * rms ** 2),
                og=at(cal.og), noise=noise, wind=m.temporal or m.wind, D=m.D, alias_spec=m.alias_spectrum(WL))


def task(args):
    """Residuals [int, leaky, IIR2, bound] [nm^2] of one (delay, magnitude, mode)."""
    total, n, turb, noise, alias, wind, D, alias_spec = args
    from aoeb import filter_optimizers as fo
    from wiener_ao_iir import ModeSpectra, wiener_iir
    from ao_tools import optimal_integrator, optimal_leaky_integrator
    spec = ModeSpectra(lambda f: turb * fo.unit_psd(f, n, wind, D), FS, total - 1.0,
                       lambda f: alias * alias_spec.unit_psd(f), noise=noise,
                       M=2 ** 15 if n < 4 else 2 ** 13)       # 2^13 within 0.1 % of 2^15 for n >= 4
    _, ji = optimal_integrator(spec)
    _, jl = optimal_leaky_integrator(spec)
    info = wiener_iir(spec, 2)[2]
    return [ji, jl, info['lower'][1][2], info['bound']]


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
            f'(Conan, wind {I["wind"]:.1f} m/s; measured noise, aliasing, og {min(ogs):.2f}-{max(ogs):.2f})')
    out = []
    for name, (cases, show) in FIGURES.items():
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        for k, (t, mg) in enumerate(cases):
            res = np.sqrt(np.array([done[key(t, mg, j)] for j in MODES]))
            for c in show:
                ax.plot(MODES, res[:, c], ls[k], color=COLORS[c], marker=MARKERS[c], ms=4, lw=1.3, mfc='none')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(MODES); ax.set_xticklabels(MODES, fontsize=7); ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set(xlabel='KL mode', ylabel='Residual per mode [nm rms]')
        same_t = len({t for t, _ in cases}) == 1
        ax.set_title(f'Total delay {cases[0][0]:g} frames' if same_t else f'Magnitude {cases[0][1]:g} (900 nm)')
        ax.grid(True, which='both', alpha=0.3)
        h = [Line2D([], [], color=COLORS[c], lw=1.5, marker=MARKERS[c], mfc='none') for c in show]
        h += [Line2D([], [], color='gray', ls=s) for s in ls[:len(cases)]]
        lab = [LABELS[c] for c in show] + [f'mag {mg:g} (900 nm)' if same_t else f'total delay {t:g} frames' for t, mg in cases]
        ax.legend(h, lab, fontsize=8, ncol=2)
        fig.suptitle(head, fontsize=8, wrap=True)
        fig.tight_layout()
        fname = os.path.join(HERE, f'residual_vs_mode_aoeb_{name}.png')
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        out.append(fname)
    return out


if __name__ == '__main__':
    done, I = compute()
    for f in plot(done, I):
        print(f)
