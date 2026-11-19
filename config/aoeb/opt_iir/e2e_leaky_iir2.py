"""
Per-mode leaky integrator and IIR2 (order-2 POL Wiener) controllers for an E2E run, and the runs themselves (2026-11-13).

Case: single pyramid 900 nm, rMod 1, 48x48, 1300 modes, 2 kHz, achromatic, mag MAG (900 nm; env E2E_IIR_MAG, default 5), seeing 0.9" on the line of sight, total delay
1 + tau = 1.5 frames (tau = 250 us * fs), 5 s, PSFs integrated after 0.5 s. Reconstructor SVD, and ML at the flux of the run.

Design inputs per mode j (nm^2, optical gain compensated, as aoeb_order.py):
  turbulence: one-sided PSD MEASURED on the e2e KL time series (results/WFSW/alias_psd_mimo: true KL coefficients `atmo`, 10-layer
              ristretto profile, frozen flow, 0.9" at zenith 0, 2 kHz, 8 s; 8 runs, 4 wind rotations x 2 seeds), Welch averaged over the runs;
  aliasing:   variance of the e2e product (SVD; ML: ml_aliasing) / og^2, temporal shape of its radial order measured (aoeb MeasuredAliasSpectrum);
  noise:      e2e noise model at the run flux (get_noise_propagation / noise_variance) / og^2, white;
  og:         aoeb calibration (rMod 1, s0.9).
In SPECULA both controllers are ONE IirFilter on the closed-loop modal measurement y (delay tau), so u = C(z) y with
  leaky: C = (G / og) / (1 - leak z^-1);
  IIR2:  the POL filter F = B/A (u = F POL, POL = y / og + P u, P = (1 - tau) z^-1 + tau z^-2) in its equivalent residual form
         C = B / (og (A - B P)), identical to the POL loop for a linear plant (checked against ao_tools.closed_loop_residual).

    python aoeb/opt_iir/e2e_leaky_iir2.py design [svd|ml]     # coefficients -> OUT/coeffs_<rec>.npz (resumable cache)
    python aoeb/opt_iir/e2e_leaky_iir2.py run <rec> <ctrl> --gpu G   # ctrl leaky | iir2
    python aoeb/opt_iir/e2e_leaky_iir2.py plot                # PSF figures (550 nm)
"""
import glob
import json
import os
import os.path as op
import sys

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('AOEB_ORDER_RMOD', os.environ.get('E2E_IIR_RMOD', '1'))
import numpy as np

HERE = op.dirname(op.abspath(__file__))
E2E = op.join(HERE, '..', 'e2e')
sys.path[:0] = [HERE, E2E]
MAG = float(os.environ.get('E2E_IIR_MAG', 5.0))       # 900 nm magnitude (env E2E_IIR_MAG)
TOTAL_DELAY = float(os.environ.get('E2E_IIR_DELAY', 1.5))   # total loop delay 1 + tau [frames] (ristretto_main: 1 + 250 us * 2 kHz = 1.5)
WIND = os.environ.get('E2E_IIR_WIND', 'ristretto')          # 'ristretto': ristretto_main speeds, turbulence PSD measured on the e2e KL time
# series; 'draw3': aoeb_order.random_layers(seed 3) speeds (one 50.2 m/s layer, same V_eff), turbulence PSD aoeb analytic (ZernikePSD of those
# layers); the measured (ristretto-wind) aliasing PSD shape is kept in both
WIND_SEED = {'draw3': 3}.get(WIND)
FC = float(os.environ.get('E2E_IIR_FC', 1.0))         # flux correction: model flux x FC (measured / model of a check run): ML rec and noise
SUFFIX = ('' if WIND == 'ristretto' else f'_{WIND}') + ('' if TOTAL_DELAY == 1.5 else f'_d{TOTAL_DELAY:g}') + ('' if FC == 1.0 else f'_f{FC:.3f}')
RMOD = float(os.environ.get('E2E_IIR_RMOD', 1.0))     # modulation [lambda/D] (env E2E_IIR_RMOD)
N_SUB, N, FS, WVL = 48, 1300, 2000.0, 900.0
OUT = f'/raid1/mmenessini/results/WFSW/e2e_leaky_iir2_r{RMOD:g}_m{MAG:g}{SUFFIX}'
PSF_VMIN = {5.0: 1e-5, 11.0: 1e-4}.get(MAG, 1e-5)    # lower limit of the PSF colour scale (user's choice per magnitude)
PSF_VMAX_CORO = 1e-2                                 # upper limit of the coronagraphic PSF colour scale
PSF_HALF = 25                                        # [lambda/D] half width of the PSF images
TS = '/raid1/mmenessini/results/WFSW/alias_psd_mimo/modes/m0_*.npz'
TAU = TOTAL_DELAY - 1.0              # kpred delay [frames]: total delay 1 + TAU (the 1 is the DM step of prop)
TOTAL_TIME = 5.0
NPERSEG = 4000                       # Welch segment: 2 s, 0.5 Hz resolution
N_JOBS = int(os.environ.get('E2E_IIR_JOBS', 24))


def turbulence_psd():
    """(f, S[j, f]) one-sided PSD [nm^2/Hz] of every mode, Welch on each run, averaged over the runs."""
    path = op.join(OUT, 'turb_psd.npz')
    if op.exists(path):
        d = np.load(path)
        return d['f'], d['S']
    from scipy.signal import welch
    S, files = 0.0, sorted(glob.glob(TS))
    for fn in files:
        x = np.load(fn)['atmo'].astype(float)
        f, P = welch(x - x.mean(0), fs=FS, nperseg=NPERSEG, axis=0)
        S = S + P.T / len(files)
    np.savez(path, f=f, S=S, files=np.array(files))
    return f, S


def psd_callable(f0, S0):
    """S(f): linear interpolation of the measured PSD, flat at its first non-zero bin below it, 0 above fs/2."""
    f0, S0 = f0[1:], S0[1:]
    def S(f):
        f = np.abs(np.asarray(f, float))
        return np.where(f <= f0[-1], np.interp(f, f0, S0, left=S0[0]), 0.0)
    return S


def e2e_noise_alias(rec):
    """Per-mode noise and aliasing variance in reconstructor units (og included) of the e2e model at the run flux."""
    import sweep_contrast as sc
    from utils import get_noise_propagation, get_aliasing_power
    flux = sc.flux_per_frame(RMOD, MAG, FS, N_SUB, N) * FC
    if rec == 'svd':
        return get_noise_propagation(RMOD, N_SUB, N, WVL, sc.SIGMA_E, flux, pupil_flux=True) ** 2, get_aliasing_power(RMOD, N_SUB, N, WVL)
    R = sc.ml_rec(RMOD, MAG, FS, N_SUB, N, FC)[1]
    return sc.noise_variance(R, RMOD, flux, N_SUB, N), sc.ml_aliasing(R, RMOD, N_SUB, N)


def residual_form(b, a, og):
    """IIR2 POL filter B/A -> residual-form (num, den) in powers of z^-1 (u = num/den y)."""
    from wiener_ao_iir import pol_weights
    p = pol_weights(TAU)
    den = np.zeros(len(a) + len(p) - 1)
    den[:len(a)] += a
    bp = np.convolve(b, p)
    den[:len(bp)] -= bp
    return np.asarray(b, float) / og, np.trim_zeros(den, 'b')


_MODEL = {}                          # analytic turbulence model of the design (set before the fork)


def design_task(args):
    j, n, Sf, Ss, alias, noise, og = args       # Sf None: Ss is the turbulence variance, PSD shape analytic (_MODEL)
    from aoeb_order import machine
    from aoeb import filter_optimizers as fo
    from ao_tools import optimal_leaky_integrator, closed_loop_residual
    from wiener_ao_iir import ModeSpectra, wiener_iir
    alias_spec = machine().alias_spectrum(WVL, N, RMOD)
    S_turb = psd_callable(Sf, Ss) if Sf is not None else (lambda f: Ss * fo.unit_psd(f, n, _MODEL['wind'], _MODEL['D']))
    spec = ModeSpectra(S_turb, FS, TAU, lambda f: alias * alias_spec.unit_psd(f, n), noise=noise,
                       M=2 ** 15 if n < 4 else 2 ** 13)
    (G, l), jl = optimal_leaky_integrator(spec)
    b, a, j2 = wiener_iir(spec, 2)[2]['lower'][1]
    num, den = residual_form(b, a, og)
    jc = closed_loop_residual(spec, num * og, den, 'residual', 1.0, 1.0)  # equivalent residual form on the compensated measurement y / og
    return j, dict(G=float(G), leak=float(l), b=[float(x) for x in b], a=[float(x) for x in a], res_leaky=float(jl), res_iir2=float(j2),
                   res_iir2_resform=float(jc), open_loop=float(spec.open_loop))


def design(rec):
    import aoeb_order as A
    I = A.inputs(mags=[MAG], wind_seed=WIND_SEED)
    _MODEL.update(wind=I['wind'], D=I['D'])
    order = np.array([I['order'][j] for j in range(1, N + 1)])
    og = np.array([I['og'][j] for j in range(1, N + 1)])
    nv, av = e2e_noise_alias(rec)
    noise, alias = nv / og ** 2, av / og ** 2
    if WIND_SEED is None:
        f, S = turbulence_psd()
    else:
        f, S = None, np.array([I['turb'][j] for j in range(1, N + 1)])
        print(f'{WIND}: layer speeds {[round(v, 1) for _, v in A.random_layers(A.machine().alias_layers, WIND_SEED)]}, V_eff {I["v_eff"]:.2f} m/s')
    cache = op.join(OUT, f'coeffs_{rec}.json')
    done = json.load(open(cache)) if op.exists(cache) else {}
    jobs = [(j, int(order[j]), f, S[j], float(alias[j]), float(noise[j]), float(og[j])) for j in range(N) if str(j) not in done]
    jobs.sort(key=lambda t: -(t[1] < 4))
    print(f'{rec}: {N - len(jobs)} cached, {len(jobs)} to design on {N_JOBS} workers', flush=True)
    if jobs:
        import multiprocessing as mp
        A.machine()                                   # built before the fork
        with mp.get_context('fork').Pool(N_JOBS) as pool:
            for i, (j, r) in enumerate(pool.imap_unordered(design_task, jobs, chunksize=1)):
                done[str(j)] = r
                if i % 25 == 0 or i == len(jobs) - 1:
                    json.dump(done, open(cache + '.tmp', 'w'))
                    os.replace(cache + '.tmp', cache)
                    print(f'[{i + 1}/{len(jobs)}]', flush=True)
    c = [done[str(j)] for j in range(N)]
    G, leak = np.array([x['G'] for x in c]), np.array([x['leak'] for x in c])
    b, a = np.array([x['b'] for x in c]), np.array([x['a'] for x in c])
    rl, r2, rc = (np.array([x[k] for x in c]) for k in ('res_leaky', 'res_iir2', 'res_iir2_resform'))
    print(f'{rec}: residual rss leaky {np.sqrt(rl.sum()):.2f} nm, IIR2 {np.sqrt(r2.sum()):.2f} nm; residual form vs POL max rel diff '
          f'{np.max(np.abs(rc / r2 - 1)):.1e}')
    np.savez(op.join(OUT, f'coeffs_{rec}.npz'), G=G, leak=leak, b=b, a=a, og=og, noise=noise, alias=alias, order=order,
             res_leaky=rl, res_iir2=r2)
    save_filters(rec, G, leak, b, a, og)


def filter_tag(rec, ctrl):
    return f'e2e_leaky_iir2_r{RMOD:g}_m{MAG:g}{SUFFIX}_{rec}_{ctrl}_{N}modes'


def save_filters(rec, G, leak, b, a, og):
    """IirFilterData files (SPECULA order: oldest coefficient first, the last den one is a0) in <calib_dir>/filter."""
    from specula.data_objects.iir_filter_data import IirFilterData
    from utils import calib_dir
    L = len(residual_form(b[0], a[0], 1.0)[1]) + 1       # 5 at tau < 1, 6 at tau >= 1
    for ctrl in ('leaky', 'iir2'):
        num, den = np.zeros((N, L)), np.zeros((N, L))
        for j in range(N):
            if ctrl == 'leaky':
                nz, dz = np.array([G[j] / og[j]]), np.array([1.0, -leak[j]])
            else:
                nz, dz = residual_form(b[j], a[j], og[j])
            num[j, :len(nz)], den[j, :len(dz)] = nz, dz
        iir = IirFilterData(ordnum=np.full(N, L), ordden=np.full(N, L), num=num[:, ::-1].copy(), den=den[:, ::-1].copy())
        path = op.join(calib_dir, 'filter', filter_tag(rec, ctrl) + '.fits')
        iir.save(path)
        print('saved', path)


DIVERGED_SR, DIVERGED_DROP = 0.1, 0.7      # loop lost: SR below this, or last 0.5 s below DROP x the 0.5 s after START
CHECK_TIME = 1.5                     # [s] quick modal check (no 2D PSFs saved); statistics after START (0.5 s) as the full runs


def run(rec, ctrl, gpu, check=False):
    import sweep_contrast as sc
    sc.REC, sc.WVL, sc.CHROMATIC, sc.N_SUBAPS, sc.N_MODES = rec, WVL, False, N_SUB, N
    name = f'{"check_" if check else ""}{rec}_{ctrl}'
    sw = sc.Sweep(op.join(OUT, name), gpu, CHECK_TIME if check else TOTAL_TIME)
    ov = sw.overrides(RMOD, MAG, FS, N_SUB, N, 'unused', FC)
    ov['remove'] = ov['remove'] + ['kalman']
    del ov['kalman_override']
    ov['kpred_override'] = {'delay': TAU, 'iir_filter_data_object': filter_tag(rec, ctrl),
                            'inputs': {'delta_comm': f'pyr_modes_{sc.band()}.out_modes'}}
    if WIND_SEED is not None:
        import aoeb_order as A
        ov['wind_speed_override'] = {'constant': [float(v) for _, v in A.random_layers(A.machine().alias_layers, WIND_SEED)]}
    if not check:
        ov['data_store_override']['inputs']['input_list'] += ['psf_vis_int-coro_psf_vis.out_int_psf',
                                                              'coro_psf_vis_2d_int-coro_psf_vis.out_int_coronagraph_psf']
    tn = sw.simulate(ov, name, sw.store)
    path = op.join(sw.store, tn)
    cv, pv, sv = sc.metrics(path, 'vis', FS)
    flux = sc.measured_flux(path, sc.band(), FS) / sc.flux_per_frame(RMOD, MAG, FS, N_SUB, N)
    a, e = sc.sr_drift(path, FS)
    row = dict(rec=rec, ctrl=ctrl, check=check, rmod=RMOD, flux_corr=FC, total_delay=TOTAL_DELAY, wind=WIND, contrast_vis_2_5=cv, sr_vis=sv, sr_start=a, sr_end=e, flux_ratio=flux, tn=tn, path=path)
    with open(op.join(OUT, 'runs.jsonl'), 'a') as f:
        f.write(json.dumps(row) + '\n')
    print(row, flush=True)
    if not (np.isfinite(sv) and sv >= DIVERGED_SR and e >= DIVERGED_DROP * a):
        print(f'DIVERGED: SR {sv:.3f}, first 0.5 s {a:.3f}, last 0.5 s {e:.3f}', flush=True)
        sys.exit(3)


START = 0.5                          # [s] statistics after this (as coro_psf_vis.start_time)
CTRL_LABEL = {'leaky': 'Leaky integrator', 'iir2': 'IIR2'}
CTRL_COLOR = {'leaky': 'tab:blue', 'iir2': 'tab:orange'}


def coro_profile(path):
    """Integrated coronagraphic radial profile of a run [lambda/D, intensity / AO-corrected peak] (the contrast metric's profile)."""
    from results_analysis import load_coro_profile
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False      # results_analysis turns it on at import
    rad, prof = load_coro_profile(path, 'vis')
    return np.asarray(rad, float), np.asarray(prof, float)


def profile_axes(ax):
    from results_analysis import load_perfect_coro
    import matplotlib
    matplotlib.rcParams['text.usetex'] = False
    rad, cv, _ = load_perfect_coro('achrom', 30, '0.9')      # fitting only: 1300 modes removed exactly, achromatic, 0.9" line of sight
    ax.semilogy(rad, cv, color='k', lw=1.5, label='perfect correction (1300 modes, fitting only)')
    ax.axvspan(2, 5, color='0.9', zorder=0, label='2-5 λ/D')
    ax.set(xlim=(0, PSF_HALF), ylim=(PSF_VMIN / 10, PSF_VMAX_CORO), xlabel=r'separation [$\lambda$/D]',
           ylabel='intensity / AO-corrected peak')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=9)


def plot_profiles(folders, out):
    """One figure: coronagraphic radial profiles of every 5 s run of these result folders (label: folder settings, controller)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    styles = ['-', '--', ':', '-.']
    for i, (label, folder) in enumerate(folders):
        for ctrl in ('leaky', 'iir2'):
            rows = [json.loads(l) for l in open(op.join(folder, 'runs.jsonl'))]
            rows = [r for r in rows if not r.get('check') and r['ctrl'] == ctrl]
            if not rows:
                continue
            r = min(rows, key=lambda r: r['contrast_vis_2_5'])
            rad, prof = coro_profile(r['path'])
            ax.semilogy(rad, prof, color=CTRL_COLOR[ctrl], ls=styles[i % 4], lw=1.5,
                        label=f'{CTRL_LABEL[ctrl]}, {label} ({r["rec"].upper()}): 2-5 λ/D {r["contrast_vis_2_5"]:.2e}')
    profile_axes(ax)
    ax.set_title('Coronagraphic PSF radial profiles, 550 nm, mag 5, draw3 wind, ML rec, 5 s (statistics after 0.5 s)')
    for ext in ('png', 'pdf'):
        fig.savefig(out + '.' + ext, dpi=130)
    print('saved', out + '.png')


REC_COLOR = {'svd': 'tab:blue', 'ml': 'tab:orange'}


def runs(check):
    """{(rec, ctrl): last row of runs.jsonl} of the check or full runs."""
    rows = {}
    if op.exists(op.join(OUT, 'runs.jsonl')):
        for line in open(op.join(OUT, 'runs.jsonl')):
            r = json.loads(line)
            if bool(r.get('check')) == check:
                rows[r['rec'], r['ctrl']] = r
    return rows


def modal_rms(path, name):
    """rms per KL mode [nm] after START of a stored modal time series (first N modes), and the rms of its time mean."""
    from astropy.io import fits
    x = np.asarray(fits.getdata(op.join(path, name + '.fits')), float)[int(round(START * FS)):, :N]
    return np.sqrt(np.mean(x ** 2, 0)), np.abs(x.mean(0))


def predicted(rec, ctrl):
    return np.sqrt(np.load(op.join(OUT, f'coeffs_{rec}.npz'))[f'res_{ctrl}'])


def check_table(check=True):
    lines = ['| rec | controller | contrast 2-5 l/D 550 | SR 550 | SR 0.5-1 s / last 0.5 s | flux / model | residual rss E2E [nm] | predicted [nm] | '
             'E2E / predicted per mode (median, 10-90 %) |', '|' + '---|' * 9]
    for (rec, ctrl), r in sorted(runs(check).items()):
        e2e, _ = modal_rms(r['path'], 'res_modes')
        p = predicted(rec, ctrl)
        q = np.percentile(e2e / p, [50, 10, 90])
        lines.append(f"| {rec} | {ctrl} | {r['contrast_vis_2_5']:.3e} | {r['sr_vis']:.4f} | {r['sr_start']:.4f} / {r['sr_end']:.4f} | "
                     f"{r['flux_ratio']:.3f} | {np.sqrt(np.sum(e2e ** 2)):.1f} | {np.sqrt(np.sum(p ** 2)):.1f} | {q[0]:.2f} ({q[1]:.2f}-{q[2]:.2f}) |")
    return '\n'.join(lines)


def per_mode(y):
    """(mode number from 1, rms of every mode): no binning."""
    return np.arange(1, len(y) + 1), np.asarray(y)


def binned(y, n_bins=40):
    """Mean square of y in log-spaced mode bins -> (bin centre mode number, rms)."""
    j = np.arange(1, len(y) + 1)
    e = np.unique(np.round(np.geomspace(1, len(y) + 1, n_bins)).astype(int))
    c = [np.sqrt(np.mean(y[a - 1:b - 1] ** 2)) for a, b in zip(e[:-1], e[1:])]
    return np.sqrt(e[:-1] * (e[1:] - 1)), np.array(c)


def plot(check=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.io import fits
    plt.rcParams.update({'axes.titlesize': 10, 'figure.titlesize': 11})
    R = runs(check)
    tag = 'check_' if check else ''
    sub = f'rMod {RMOD:g}, 48x48, 1300 modes, 2 kHz, mag {MAG:g}{" (flux x %.3f)" % FC if FC != 1.0 else ""}, total delay {TOTAL_DELAY:g} frames, {WIND} wind,\nachromatic, 0.9\"; {"1.5" if check else "5"} s, statistics after {START:g} s'
    figs = []
    # modal residuals, one figure per controller: E2E vs design prediction
    for ctrl in ('leaky', 'iir2'):
        fig, ax = plt.subplots(figsize=(8, 5))
        for rec in ('svd', 'ml'):
            if (rec, ctrl) not in R:
                continue
            e2e, _ = modal_rms(R[rec, ctrl]['path'], 'res_modes')
            ax.loglog(*per_mode(e2e), color=REC_COLOR[rec], lw=1.2, label=f'E2E, {rec.upper()} rec (rss {np.sqrt(np.sum(e2e ** 2)):.1f} nm)')
            p = predicted(rec, ctrl)
            ax.loglog(*per_mode(p), color=REC_COLOR[rec], lw=1.2, ls='--', label=f'design prediction, {rec.upper()} (rss {np.sqrt(np.sum(p ** 2)):.1f} nm)')
        ax.set(xlabel='KL mode', ylabel='residual rms per mode [nm]', title=f'{CTRL_LABEL[ctrl]}: modal residual, E2E vs design\n{sub}')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=9)
        figs.append((fig, f'{tag}modal_residual_{ctrl}'))
    # modal residuals, E2E only, both controllers
    fig, ax = plt.subplots(figsize=(8, 5))
    for (rec, ctrl), r in sorted(R.items()):
        e2e, _ = modal_rms(r['path'], 'res_modes')
        ax.loglog(*per_mode(e2e), color=REC_COLOR[rec], lw=1.2, ls='-' if ctrl == 'iir2' else ':',
                  label=f'{CTRL_LABEL[ctrl]}, {rec.upper()} (rss {np.sqrt(np.sum(e2e ** 2)):.1f} nm)')
    ax.set(xlabel='KL mode', ylabel='residual rms per mode [nm]', title=f'E2E modal residual, leaky vs IIR2\n{sub}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=9)
    figs.append((fig, f'{tag}modal_residual_e2e'))
    # turbulence modal rms (open loop, from the runs: the same atmosphere in every run)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (rec, ctrl), r in sorted(R.items()):
        t, _ = modal_rms(r['path'], 'atmo_modes')
        ax.loglog(np.arange(1, N + 1), t, color='0.3', lw=1)
        break
    ax.set(xlabel='KL mode', ylabel='turbulence rms per mode [nm]',
           title=f'Turbulence (open loop) modal rms, E2E run {rec.upper()} {ctrl} (rss {np.sqrt(np.sum(t ** 2)):.0f} nm, 1300 modes)\n{sub}')
    ax.grid(True, which='both', alpha=0.25)
    figs.append((fig, f'{tag}turbulence_modal_rms'))
    # turbulence and the two E2E residuals, for the reconstructor of each controller shown in the PSF figure (the better contrast)
    best = {ctrl: min((r for (rc, c), r in R.items() if c == ctrl), key=lambda r: r['contrast_vis_2_5']) for ctrl in ('leaky', 'iir2')
            if any(c == ctrl for _, c in R)}
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(*per_mode(t), color='0.3', lw=1.2, label=f'turbulence, open loop (rss {np.sqrt(np.sum(t ** 2)):.0f} nm)')
    for ctrl, col in (('leaky', 'tab:blue'), ('iir2', 'tab:orange')):
        if ctrl in best:
            e2e, _ = modal_rms(best[ctrl]['path'], 'res_modes')
            ax.loglog(*per_mode(e2e), color=col, lw=1.2, label=f'{CTRL_LABEL[ctrl]} residual, {best[ctrl]["rec"].upper()} rec (rss {np.sqrt(np.sum(e2e ** 2)):.1f} nm)')
    ax.set(xlabel='KL mode', ylabel='rms per mode [nm]', title=f'E2E turbulence and modal residuals\n{sub}')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=9)
    figs.append((fig, f'{tag}modal_turbulence_residuals'))
    # PSFs at 550 nm, normalised to the AO peak of each run
    if not check:
        best = {ctrl: min((r for (rc, c), r in R.items() if c == ctrl), key=lambda r: r['contrast_vis_2_5']) for ctrl in ('leaky', 'iir2')
                if any(c == ctrl for _, c in R)}
        fig = plt.figure(figsize=(9.5, 15.5), constrained_layout=True)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.8])
        axs = np.array([[fig.add_subplot(gs[i, k]) for k in range(2)] for i in range(2)])
        axp = fig.add_subplot(gs[2, :])
        ims = [None, None]
        half, lod = PSF_HALF, 4.0                                # [lambda/D] shown; pixels per lambda/D (nd = 4)
        for i, ctrl in enumerate(('leaky', 'iir2')):
            r = best[ctrl]
            psf = np.asarray(fits.getdata(op.join(r['path'], 'psf_vis_int.fits')), float).squeeze()
            coro = np.asarray(fits.getdata(op.join(r['path'], 'coro_psf_vis_2d_int.fits')), float).squeeze()
            peak = psf.max()
            c = np.array(np.unravel_index(np.argmax(psf), psf.shape))
            h = int(half * lod)
            ext = [-half, half, -half, half]
            for k, (img, lab, vmax) in enumerate(((psf, 'non-coronagraphic', 1.0), (coro, 'coronagraphic (perfect)', PSF_VMAX_CORO))):
                ims[k] = axs[i, k].imshow(np.log10(np.maximum(img[c[0] - h:c[0] + h, c[1] - h:c[1] + h] / peak, 1e-12)), origin='lower',
                                          extent=ext, vmin=np.log10(PSF_VMIN), vmax=np.log10(vmax), cmap='inferno')
                axs[i, k].set(title=f'{CTRL_LABEL[ctrl]} ({r["rec"].upper()} rec), {lab}', xlabel=r'x [$\lambda$/D]', ylabel=r'y [$\lambda$/D]')
            axs[i, 0].set_xlabel(r'x [$\lambda$/D]' + f'\nSR (550 nm) = {r["sr_vis"]:.3f}', fontsize=11)
            axs[i, 1].text(0.02, 0.02, f'contrast 2-5 λ/D {r["contrast_vis_2_5"]:.2e}', transform=axs[i, 1].transAxes, color='w', fontsize=9)
            rad, prof = coro_profile(r['path'])
            axp.semilogy(rad, prof, color=CTRL_COLOR[ctrl], lw=1.5, label=f'{CTRL_LABEL[ctrl]} ({r["rec"].upper()}), 2-5 λ/D {r["contrast_vis_2_5"]:.2e}')
        profile_axes(axp)
        axp.set_title('Coronagraphic PSF radial profile (550 nm, azimuthal mean)')
        for k in (0, 1):
            fig.colorbar(ims[k], ax=axs[:, k], shrink=0.6, location='bottom', label='log10(intensity / AO-corrected peak)')
        fig.suptitle(f'550 nm PSFs: {sub}'.replace(', achromatic', ',\nachromatic'))
        figs.append((fig, 'psf_550_leaky_iir2'))
    for fig, name in figs:
        for ext in ('png', 'pdf'):
            fig.savefig(op.join(OUT, f'{name}.{ext}'), dpi=130)
        print('saved', op.join(OUT, name + '.png'))

def plot_coeffs(rec='ml', ref=None):
    """Leaky (G, leak) and IIR2 POL (b0..b2, a1, a2) coefficients used, per mode; ref: (label, folder) of another design, dashed.
    G and b are the gains of the optical-gain-compensated measurement (E2E applies them divided by og)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.titlesize': 10, 'figure.titlesize': 11})
    sets = [(f'rMod {RMOD:g}', op.join(OUT, f'coeffs_{rec}.npz'), '-')]
    if ref:
        sets.append((ref[0], op.join(ref[1], f'coeffs_{rec}.npz'), '--'))
    j = np.arange(1, N + 1)
    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for lab, path, ls in sets:
        c = np.load(path)
        axs[0, 0].semilogx(j, c['G'], color='tab:blue', ls=ls, lw=1, label=f'G, {lab}')
        axs[0, 1].semilogx(j, 1 - c['leak'], color='tab:blue', ls=ls, lw=1, label=f'1 - leak, {lab}')
        for k, col in enumerate(('tab:orange', 'tab:green', 'tab:red')):
            axs[1, 0].semilogx(j, c['b'][:, k], color=col, ls=ls, lw=1, label=f'b{k}, {lab}')
        for k, col in ((1, 'tab:purple'), (2, 'tab:brown')):
            axs[1, 1].semilogx(j, c['a'][:, k], color=col, ls=ls, lw=1, label=f'a{k}, {lab}')
    axs[0, 1].set_yscale('log')
    titles = ['Leaky integrator gain G (u_k = leak u_{k-1} + G y_k / og)', 'Leaky integrator: 1 - leak',
              'IIR2 POL numerator B = b0 + b1 z^-1 + b2 z^-2 (u = B/A POL)', 'IIR2 POL denominator A = 1 + a1 z^-1 + a2 z^-2']
    for ax, t in zip(axs.flat, titles):
        ax.set(title=t, xlabel='KL mode')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'Coefficients used ({rec.upper()} rec): 48x48, 2 kHz, mag {MAG:g}, total delay {TOTAL_DELAY:g} frames, {WIND} wind')
    for ext in ('png', 'pdf'):
        fig.savefig(op.join(OUT, f'coefficients_{rec}.{ext}'), dpi=130)
    print('saved', op.join(OUT, f'coefficients_{rec}.png'))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == 'design':
        for rec in sys.argv[2:] or ('svd', 'ml'):
            design(rec)
    elif cmd == 'run':
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument('cmd'); ap.add_argument('rec'); ap.add_argument('ctrl'); ap.add_argument('--gpu', type=int, default=0)
        ap.add_argument('--check', action='store_true')
        x = ap.parse_args()
        os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']
        run(x.rec, x.ctrl, x.gpu, x.check)
    elif cmd == 'check-table':
        print(check_table(True))
    elif cmd == 'table':
        print(check_table(False))
    elif cmd == 'plot-profiles':      # draw3 delays 1.5 / 2 / 2.5
        base = '/raid1/mmenessini/results/WFSW/e2e_leaky_iir2_r1_m5_draw3'
        plot_profiles([('delay 1.5', base), ('delay 2', base + '_d2'), ('delay 2.5', base + '_d2.5')],
                      '/raid1/mmenessini/results/WFSW/e2e_leaky_iir2_draw3_coro_profiles')
    elif cmd == 'plot-coeffs':       # [ref_label ref_folder]
        plot_coeffs('ml', tuple(sys.argv[2:4]) if len(sys.argv) > 3 else None)
    elif cmd == 'plot':
        plot(check='--check' in sys.argv)
