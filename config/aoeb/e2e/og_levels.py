"""Optical gains of the e2e VIS pyramid (rMod 3, 36x36, 600 modes, seeing 1.05", WFSW root) at four correction levels:
perfect correction, Wiener bound, integrator with one scalar gain 0.4, the same with gain 0.2 (2026-11-13: these replaced the leaky,
the optimal scalar gain 0.497 and its rounding 0.5, whose residuals were within 2 nm of each other). The partial levels are the
per-mode residuals of the aoeb budget (VIS 900 nm, mag 8 at 900 nm, 2 kHz = the residual-optimal rate of the Wiener level, latency
500 us, perfect-correction OG, one pass): residual = lag + noise + aliasing per mode. Each is turned into a correction vector
c_i = 1 - sqrt(res_i / turb_i) (modes > N uncorrected) and a PC IM of both pyramids averaged over 70 independent screens (calib_pc.yml,
the same seed for every level), OG_i = (R IM_PC)_ii, R the 600-mode SVD reconstructor.

    python og_levels.py calib  --gpu 0          # 600-mode recs, aliasing run (corr 600), perfect-correction PC IM (new files only)
    python og_levels.py budgets                 # perfect + Wiener (and the superseded leaky / optimal scalar) levels -> OUT
    python og_levels.py fixed                   # the FIXED_GAINS levels, residual plot
    python og_levels.py pc --level wiener --gpu 0
    python og_levels.py og                      # optical gains of the LEVELS, plot + table -> OUT

Products in calibration/WFSW are only ADDED (never rewritten); the level PC IMs are moved to OUT (they are not (max, min) levels and
must not be globbed by Paths.pc_levels)."""
import argparse
import json
import logging
import os
import os.path as op
import shutil
import sys

import numpy as np

E2E_DIR = op.dirname(op.abspath(__file__))
sys.path.insert(0, E2E_DIR)
sys.path.insert(0, op.dirname(op.dirname(E2E_DIR)))
import calibration as cal                                             # noqa: E402
from aoeb import utils as au                                          # noqa: E402
from aoeb.generation import merge                                     # noqa: E402

RMOD, NSUB, SEEING, N, MAG, BAND = 3.0, 36, 1.05, 600, 8.0, 'vis'
LATENCY_S, FS_MAX = 500e-6, 2000.0
FIXED_GAINS = {'int_g0.4': 0.4, 'int_g0.2': 0.2}     # 2026-11-13 (user): replace the leaky / optimal-scalar levels (too close to the
                                                      # Wiener one) by integrators with one fixed gain; same frame rate as the others
LEVELS = ('perfect', 'wiener', 'leaky', 'int_g0.4', 'int_g0.2')     # leaky re-added 2026-11-15 (user)
LABELS = {'perfect': 'perfect correction', 'wiener': 'Wiener bound', 'leaky': 'leaky integrator',
          'int_scalar': 'integrator, one scalar gain', 'int_scalar_r1': 'integrator, scalar gain rounded to 0.1',
          'int_g0.4': 'integrator, scalar gain 0.4', 'int_g0.3': 'integrator, scalar gain 0.3', 'int_g0.2': 'integrator, scalar gain 0.2'}
OUT = '/raid1/mmenessini/results/WFSW/og_levels_20261113'
log = logging.getLogger('og_levels')


def config():
    """E2EConfig of this one case (600 added to the 36x36 mode list: its recs and the aliasing of the recs <= 600 at corr 600)."""
    c = cal.E2EConfig(n_subaps=(NSUB,), rMods=(RMOD,), seeings=(SEEING,), n_corr=(N,))
    c.modes_per_subap = {NSUB: sorted(set(c.modes_per_subap[NSUB]) | {N})}
    return c


def calib(gpu):
    c = config()
    cal.compute_recs(c)
    cal.calibrate_alias(c, gpu)
    cal.calibrate_pc(c, gpu)


# ----------------------------------------------------------------------------- budgets
def machine():
    m = cal.machine(config())
    m.modes_per_subap = dict(m.modes_per_subap)
    m.modes_per_subap[NSUB] = sorted(set(m.mode_list(NSUB)) | {N})
    m.max_sensitivity = None                                           # as wfs_talk_figures.ipynb
    return m


def vis_flux(m):
    import yaml
    from specula.lib.n_phot import n_phot
    with open(op.join(E2E_DIR, 'ristretto_main.yml')) as f:
        Y = yaml.safe_load(f)
    area = (m.pupil() > 0).sum() * Y['main']['pixel_pitch'] ** 2
    return n_phot(MAG, lambda_=900e-9, width=1e-9)[0] * Y['cred_vis']['bandw'] * area


def scalar_gain_budget(m, flux, fs, gains=None):
    """Integrator with ONE gain g on the raw (not OG-compensated) measurement: effective loop gain g og_i (perfect-correction OG).
    Returns {g: (per-mode lag, noise, alias) [nm^2]}, evaluated with filter_optimizers.evaluate_controllers on the budget's inputs."""
    from aoeb.filter_optimizers import evaluate_controllers
    r0 = m.budget(RMOD, NSUB, SEEING, flux, fs, N=N, latency_s=LATENCY_S, controller='int', wavelength_nm=900.0, optimize_freq=False,
                  warn=False)
    c = m.calibration(RMOD, NSUB, SEEING, N, N, wavelength_nm=900.0)
    turb_cmd, rms, _ = m.turbulence(SEEING, N)
    photon, ron = c.noise.modal_variance_parts(m.detector.n_det(flux, fs), m.detector)
    alias_cmd = c.alias_rms ** 2 * (SEEING / c.alias_seeing) ** (5 / 3)
    orders = au.radial_order(np.arange(N))
    out = {}
    for g in gains:
        o = evaluate_controllers(fs, LATENCY_S * fs, np.full(N, g), np.ones(N), turb_cmd, photon + ron, alias_cmd, c.og, orders,
                                 m.temporal or m.wind, m.D, m.alias_spectrum(900.0, N, RMOD))
        out[g] = (o.turb * rms ** 2, o.noise * rms ** 2, o.alias * rms ** 2)
    return out, r0


def budgets():
    os.makedirs(OUT, exist_ok=True)
    m = machine()
    flux = vis_flux(m)
    turb_cmd, rms, _ = m.turbulence(SEEING, N)
    turb = turb_cmd * rms ** 2
    res = {'perfect': dict(res=np.zeros(N), fs=np.nan, info='')}
    for ctl in ('wiener', 'leaky'):
        r = m.budget(RMOD, NSUB, SEEING, flux, FS_MAX, N=N, latency_s=LATENCY_S, controller=ctl, wavelength_nm=900.0, optimize_freq=True,
                     warn=False)
        res[ctl] = dict(res=r.lag + r.noise + r.alias, fs=r.params['freqUsedInHz'], lag=r.lag, noise=r.noise, alias=r.alias,
                        info=('per-mode optimal IIR' if ctl == 'wiener' else
                              f"per-mode g, ff: median g {np.median(r.gain):.2f}, ff {np.median(r.ff):.3f}"))
    best = None
    for fs in m.freq_candidates(FS_MAX):                                  # scalar gain: best (fs, g) jointly
        gains = np.round(np.arange(0.01, 1.5, 0.01), 2)
        ev, _ = scalar_gain_budget(m, flux, fs, gains)
        tot = {g: float(np.sum(sum(v))) for g, v in ev.items()}
        g = min(tot, key=tot.get)
        if np.isfinite(tot[g]) and (best is None or tot[g] < best[0]):
            best = (tot[g], fs, g, ev)
    _, fs, g, ev = best
    g1 = round(g, 1)
    ev1, _ = scalar_gain_budget(m, flux, fs, [g1])
    for name, gg, e in (('int_scalar', g, ev[g]), ('int_scalar_r1', g1, ev1[g1])):
        res[name] = dict(res=sum(e), fs=fs, lag=e[0], noise=e[1], alias=e[2], info=f'g = {gg:g}' + (' = unrounded optimum'
                         if name == 'int_scalar_r1' and np.array_equal(e[0], ev[g][0]) else ''))
    summary = {}
    for k, v in res.items():
        frac = np.clip(v['res'] / turb, 0.0, 1.0)
        vec = np.zeros(N) if k == 'perfect' else np.sqrt(frac)
        corr = 1.0 - vec
        au.write_fits(op.join(OUT, f'correction_vector_{k}.fits'), corr)
        np.save(op.join(OUT, f'residual_{k}.npy'), v['res'])
        summary[k] = dict(label=LABELS[k], fs=float(v['fs']), rms_nm=float(np.sqrt(np.sum(v['res']))), info=v['info'],
                          **{t: float(np.sqrt(np.sum(v[t]))) for t in ('lag', 'noise', 'alias') if t in v},
                          n_clipped=int(np.sum(v['res'] > turb)))
    np.save(op.join(OUT, 'turbulence.npy'), turb)
    summary['_setup'] = dict(rMod=RMOD, n_subap=NSUB, seeing=SEEING, N=N, mag900=MAG, flux_ph_s=float(flux), latency_s=LATENCY_S,
                             band='VIS 900 nm', fitting_nm=float(np.sqrt(m._fitting(m.turbulence(SEEING, N)[2], N))))
    with open(op.join(OUT, 'levels.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    plot_residuals()
    print(json.dumps(summary, indent=1))


def fixed_gain_levels():
    """Adds the FIXED_GAINS levels (integrator, one gain on the raw measurement, at the Wiener level's frame rate) to levels.json:
    residual, correction vector (the Wiener / perfect levels of budgets() are kept)."""
    m = machine()
    flux = vis_flux(m)
    S = json.load(open(op.join(OUT, 'levels.json')))
    fs = S['wiener']['fs']
    turb = np.load(op.join(OUT, 'turbulence.npy'))
    ev, _ = scalar_gain_budget(m, flux, fs, list(FIXED_GAINS.values()))
    for name, g in FIXED_GAINS.items():
        lag, noise, alias = ev[g]
        res = lag + noise + alias
        if not np.all(np.isfinite(res)):
            raise ValueError(f'gain {g}: unstable modes')
        au.write_fits(op.join(OUT, f'correction_vector_{name}.fits'), 1.0 - np.sqrt(np.clip(res / turb, 0.0, 1.0)), overwrite=True)
        np.save(op.join(OUT, f'residual_{name}.npy'), res)
        S[name] = dict(label=LABELS[name], fs=float(fs), rms_nm=float(np.sqrt(np.sum(res))), info=f'g = {g:g}',
                       lag=float(np.sqrt(np.sum(lag))), noise=float(np.sqrt(np.sum(noise))), alias=float(np.sqrt(np.sum(alias))),
                       n_clipped=int(np.sum(res > turb)))
    with open(op.join(OUT, 'levels.json'), 'w') as f:
        json.dump(S, f, indent=1)
    plot_residuals()
    print(json.dumps({k: S[k] for k in LEVELS}, indent=1))


def _style(ax):
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')


COLOURS = {'perfect': 'k', 'wiener': 'C0', 'leaky': 'C1', 'int_scalar': 'C2', 'int_scalar_r1': 'C3', 'int_g0.4': 'C2', 'int_g0.3': 'C3', 'int_g0.2': 'C3'}


def plot_residuals():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    S = json.load(open(op.join(OUT, 'levels.json')))
    turb = np.load(op.join(OUT, 'turbulence.npy'))
    k = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(k, np.sqrt(turb), color='0.6', lw=1.2, label='turbulence (open loop)')
    for name in LEVELS[1:]:
        s = S[name]
        ax.plot(k, np.sqrt(np.load(op.join(OUT, f'residual_{name}.npy'))), color=COLOURS[name], lw=1.2,
                ls='--' if name == 'int_scalar_r1' else '-',
                label=f"{s['label']} ({s['info']}, {s['fs']:.0f} Hz): {s['rms_nm']:.0f} nm rms")
    ax.axhline(np.nan, color='k', label='perfect correction: 0 nm')
    _style(ax)
    ax.set_xlabel('KL mode')
    ax.set_ylabel('modal residual [nm rms]')
    st = S['_setup']
    ax.set_title(f"VIS pyramid rMod {st['rMod']:g}, {NSUB}x{NSUB}, {N} modes, seeing {SEEING}\", mag {MAG:g} (900 nm), latency "
                 f"{LATENCY_S * 1e6:.0f} us", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(op.join(OUT, 'residual_levels.png'), dpi=130)


# ----------------------------------------------------------------------------- PC IMs of the partial levels
def pc_level(level, gpu, n_screens=70):
    c = config()
    stem = f'correction_vector_{N}modes_oglevel_{level}'
    data = op.join(c.calib_dir, 'data', stem + '.fits')
    if not op.exists(data):
        shutil.copy(op.join(OUT, f'correction_vector_{level}.fits'), data)
    paths = {p: op.join(OUT, f'{c.paths(c.wavelengths[p]).sim(RMOD, NSUB, SEEING, N)}_im_pc_{level}.fits') for p in cal.PYRS}
    if all(op.exists(p) for p in paths.values()):
        return
    tags = {p: op.basename(path)[:-5] for p, path in paths.items()}
    ov = merge(cal._pyrs(c, RMOD, NSUB), {
        'main': {'total_time': 2 * N * n_screens * c.time_step},
        'seeing_random': {'constant': float(SEEING)}, 'atmo_random': {'update_interval': 2 * N, 'seed': c.seed},
        'modal_analysis_random': {'nmodes': N}, 'scale_random': {'constant_mul_data': stem}, 'dm_random': {'nmodes': N},
        'dm': {'nmodes': N}, 'pushpull': {'nmodes': N, 'ncycles': n_screens}},
        {f'pyr_im_calibrator_{p}': {'im_tag': tags[p], 'nmodes': N} for p in cal.PYRS})
    cal.run_specula(c, ['calib_pc.yml'], ov, gpu, tags['vis'])
    for p in cal.PYRS:
        src = op.join(c.calib_dir, 'im', tags[p] + '.fits')
        au.set_header(src, wvl_nm=float(c.wavelengths[p]), seeing=float(SEEING), n_screen=n_screens, dm_tag=c.dm_tag, n_pushpull=N,
                      level=level)
        shutil.move(src, paths[p])


# ----------------------------------------------------------------------------- optical gains
def ogs():
    c = config()
    out = {}
    for p in cal.PYRS:
        P = c.paths(c.wavelengths[p])
        rec = au.read_fits(P.rec(RMOD, NSUB, N))
        og = lambda im: np.einsum('ij,ji->i', rec, im[:, :N])
        dl = og(au.read_fits(P.im_dl(RMOD, NSUB)))
        out[p] = {'dl': dl}
        for lv in LEVELS:
            path = P.im_pc(RMOD, NSUB, SEEING, N) if lv == 'perfect' else op.join(OUT, f'{P.sim(RMOD, NSUB, SEEING, N)}_im_pc_{lv}.fits')
            if op.exists(path):
                out[p][lv] = og(au.read_fits(path))
    np.savez(op.join(OUT, 'optical_gains.npz'), **{f'{p}_{k}': v for p, d in out.items() for k, v in d.items()})
    plot_ogs(out)
    return out


def plot_ogs(out=None):
    """VIS pyramid (900 nm) only: OG per mode (OG_i = (R IM_PC)_ii, R the SVD rec of IM_DL, so OG_DL = 1), no averaging, ylim 0..1;
    below, the per-mode difference to the Wiener-bound OG (same 70 screens for every level). out None: read optical_gains.npz."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if out is None:
        with np.load(op.join(OUT, 'optical_gains.npz')) as z:
            out = {'vis': {k[4:]: z[k] for k in z.files if k.startswith('vis_')}}
    S = json.load(open(op.join(OUT, 'levels.json')))
    og = out['vis']
    k = np.arange(1, N + 1)
    lab = lambda lv: 'perfect correction' if lv == 'perfect' else f"{LABELS[lv]} ({S[lv]['rms_nm']:.0f} nm rms)"
    fig, (ax, ad) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True, gridspec_kw=dict(height_ratios=(3, 2)))
    for lv in LEVELS:
        if lv in og:
            ax.plot(k, og[lv], color=COLOURS[lv], lw=0.9, label=lab(lv))
            ad.plot(k, og[lv] - og['wiener'], color=COLOURS[lv], lw=0.9, label=lab(lv))
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('optical gain (R IM_PC)_ii')
    ax.legend(fontsize=8, loc='lower right')
    ad.set_ylabel('OG - OG(Wiener bound)')
    ad.set_xlabel('KL mode')
    for a_ in (ax, ad):
        a_.set_xscale('log')
        a_.grid(True, which='both', alpha=0.3)
    ax.set_title(f"VIS pyramid (900 nm), rMod {RMOD:g}, {NSUB}x{NSUB}, {N} modes, seeing {SEEING}\", mag {MAG:g}, "
                 f"2 kHz; 70 screens per PC IM", fontsize=10)
    fig.tight_layout()
    fig.savefig(op.join(OUT, 'optical_gains.png'), dpi=130)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('calib', 'budgets', 'fixed', 'plot', 'pc', 'og', 'plot_og'))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--level', choices=LEVELS[1:])
    a = ap.parse_args(argv)
    if a.step == 'calib':
        calib(a.gpu)
    elif a.step == 'budgets':
        budgets()
    elif a.step == 'fixed':
        fixed_gain_levels()
    elif a.step == 'plot_og':
        plot_ogs()
    elif a.step == 'plot':
        plot_residuals()
    elif a.step == 'pc':
        pc_level(a.level, a.gpu)
    else:
        ogs()


if __name__ == '__main__':
    main()
