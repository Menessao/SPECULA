"""Temporal PSDs per mode: multi-layer turbulence vs pyramid aliasing / measurement vs correction residual (psd_compare.yml on
ristretto_main.yml). Plots: config/aoeb/pol_turb/aliasing.ipynb.

Frozen-flow 10-layer ristretto atmosphere at 0.9", noiseless rMod 0 48x48 pyramids (VIS 900 nm, NIR 1300 nm), 2" FoV, 2 kHz, 8 s, no lag.
The first N KL modes are corrected in the same step by the exact modal projection times a correction vector:
    perfect : c = 1 (residual 0 on modes <= N)
    wiener  : c_i = 1 - sqrt(res_i / turb_i), res_i the aoeb Wiener-bound residual (lag + noise + aliasing) of mode i for the VIS pyramid
              (900 nm, mag 8, latency 500 us, frame rate optimised up to 2 kHz, perfect-correction OG) at the same rMod / n / seeing / N:
              the residual has the Wiener-bound variance per mode and the temporal shape of the turbulence (it is not a closed-loop
              spectrum).
Runs in its own calibration root <out>/root: pupils, ifunc, m2c, pupilstop, rec are read-only symlinks to calibration/WFSW; phase screens
and the correction vectors (<root>/data) are written there.

    PYTHONPATH=<e2e> python psd_compare.py levels [--N 1300 ...]      # Wiener budgets -> correction vectors, <out>/levels/levels.json
    PYTHONPATH=<e2e> python psd_compare.py run [--gpu 0] [--only name ...] [--time 8.0]

Output:
    <out>/runs/<name>/*.fits   raw time series: turb_modes, res_modes (true, 1835 modes), turb_rms, res_rms, meas_vis, meas_nir (measured),
                               true_alias_vis, true_alias_nir (twin pyramids on the perfectly corrected wavefront), slopes_*, slopes_pc_*
    <out>/psd/<name>.npz       one-sided Welch PSDs [nm^2/Hz] and variances per mode: turb, res, meas_*, true_alias_*, meas_minus_alias_*
Runs already done are skipped. 8 s: the fastest layer (34.3 m/s) wraps its 305 m screen after 8.6 s.
The budgets need the rMod 0 48x48 s0.9 products at every N (N 600 and 900 added on 2026-11-18, results/WFSW/psd_compare/calib_600_900.py)."""
import argparse
import json
import os
import os.path as op
import time

import numpy as np
from astropy.io import fits
from scipy.signal import welch

import calibration as c

WFSW = '/raid1/mmenessini/calibration/WFSW'
OUT = '/raid1/mmenessini/results/WFSW/psd_compare'
YML = op.join(c.E2E_DIR, 'psd_compare.yml')
SKIP = 20                                  # first frames dropped (DM / pyramid start-up)
RMOD, NSUB, SEEING, FOV = 0.0, 48, 0.9, 2.0
MAG, LATENCY_S, FS_MAX = 8.0, 500e-6, 2000.0          # Wiener-bound operating point (VIS 900 nm)
N_LIST = (1300, 900, 600, 300, 100)
RUNS = {'r0_N1300_perfect': dict(N=1300, level='perfect', seed=1001),
        **{f'r0_N{N}_wiener': dict(N=N, level='wiener', seed=1001) for N in N_LIST}}


def root_of(out):
    return op.join(out, 'root')


def vector_stem(N, level):
    return f'correction_vector_{N}modes_c1.00-1.00' if level == 'perfect' else f'correction_vector_{N}modes_wiener_mag{MAG:g}'


def setup_root(out):
    root = root_of(out)
    for d in ('phasescreens', 'data'):
        os.makedirs(op.join(root, d), exist_ok=True)
    for d in ('pupils', 'ifunc', 'm2c', 'pupilstop', 'rec'):
        if not op.exists(op.join(root, d)):
            os.symlink(op.join(WFSW, d), op.join(root, d))
    return root


def write_vector(root, stem, vec, **hdr_kw):
    hdr = fits.Header()
    hdr['VERSION'], hdr['OBJ_TYPE'], hdr['NDARRAY'] = 1, 'BaseValue', 1
    for k, v in hdr_kw.items():
        hdr[k] = v
    fits.writeto(op.join(root, 'data', stem + '.fits'), np.asarray(vec, dtype=np.float32), hdr, overwrite=True)


# ----------------------------------------------------------------------------- Wiener levels
def levels(out, n_list=N_LIST):
    import og_levels as ol
    from aoeb.generation import save_correction_vector
    root = setup_root(out)
    cfg = c.E2EConfig(n_subaps=(NSUB,), rMods=(RMOD,), seeings=(SEEING,))
    m = c.machine(cfg)
    m.modes_per_subap = dict(m.modes_per_subap)
    m.modes_per_subap[NSUB] = sorted(set(m.mode_list(NSUB)) | set(n_list))
    m.max_sensitivity = None                                       # as og_levels / wfs_talk_figures
    ol.MAG = MAG
    flux = ol.vis_flux(m)
    os.makedirs(op.join(out, 'levels'), exist_ok=True)
    js = op.join(out, 'levels', 'levels.json')
    summary = json.load(open(js)) if op.exists(js) else {}
    summary['_setup'] = (dict(rMod=RMOD, n_subap=NSUB, seeing=SEEING, mag900=MAG, flux_ph_s=float(flux), latency_s=LATENCY_S,
                               fs_max=FS_MAX, band='VIS 900 nm', controller='wiener'))
    for N in n_list:
        save_correction_vector(root, 1.0, 1.0, N)                  # perfect (ones)
        r = m.budget(RMOD, NSUB, SEEING, flux, FS_MAX, N=N, latency_s=LATENCY_S, controller='wiener', wavelength_nm=900.0,
                     optimize_freq=True, warn=False)
        turb_cmd, rms, _ = m.turbulence(SEEING, N)
        turb = turb_cmd * rms ** 2
        res = r.lag + r.noise + r.alias
        vec = 1.0 - np.sqrt(np.clip(res / turb, 0.0, 1.0))
        write_vector(root, vector_stem(N, 'wiener'), vec, MAG=MAG, FS=float(r.params['freqUsedInHz']))
        np.savez(op.join(out, 'levels', f'wiener_N{N}.npz'), res=res, lag=r.lag, noise=r.noise, alias=r.alias, turb=turb, vec=vec)
        summary[f'N{N}'] = dict(fs=float(r.params['freqUsedInHz']), res_nm=float(np.sqrt(res.sum())),
                                **{t: float(np.sqrt(np.sum(getattr(r, t)))) for t in ('lag', 'noise', 'alias')},
                                turb_nm=float(np.sqrt(turb.sum())), n_clipped=int(np.sum(res > turb)))
        print(f'N {N}: {summary[f"N{N}"]}', flush=True)
        with open(js + '.tmp', 'w') as f:
            json.dump(summary, f, indent=1)
        os.replace(js + '.tmp', js)


# ----------------------------------------------------------------------------- runs
def overrides(cfg, p, store, total_time):
    N = p['N']
    rec = {k: f'pyr{RMOD:.1f}_{NSUB}x{NSUB}_wvl{cfg.wavelengths[k]:.0f}_{N}modes_rec' for k in c.PYRS}
    ov = c.merge(c._pyrs(cfg, RMOD, NSUB), {
        'main': {'total_time': total_time},
        'seeing': {'constant': SEEING},
        'atmo': {'seed': int(p['seed'])},
        'atmo_modes': {'nmodes': N}, 'dm': {'nmodes': N}, 'dm_pc': {'nmodes': N},
        'scale_cmd': {'constant_mul_data': vector_stem(N, p['level'])},
        'data_store': {'store_dir': store, 'create_tn': False}})       # run_specula writes every block as <block>_override
    for k in c.PYRS:                                                   # main and twin (true-aliasing) chains set identically
        pyr = {kk: v for kk, v in ov[f'pyr_{k}'].items()}
        for pre in ('', 'pc_'):
            ov[f'pyr_{pre}{k}'] = {**pyr, 'fov': FOV}
            ov[f'pyr_slopes_{pre}{k}'] = dict(ov[f'pyr_slopes_{k}'])
            ov[f'pyr_modes_{pre}{k}'] = {'recmat_object': rec[k]}
    return ov


def load(store, key):
    return np.asarray(fits.getdata(op.join(store, key + '.fits')), dtype=np.float64)[SKIP:]


def psd_of(x, fs):
    f, S = welch(x, fs=fs, nperseg=min(4000, len(x)), axis=0, detrend='constant')
    return f, S.astype(np.float32)


def analyse(name, p, store, res_path, dt):
    """PSDs and variances of: true turbulence, true residual, measured, true aliasing (twin chain; = measured for perfect correction,
    whose run predates the twin) and measured - true aliasing (response to the residual of modes <= N)."""
    d = {k: load(store, k) for k in ('turb_modes', 'res_modes', 'meas_vis', 'meas_nir')}
    for cam in ('vis', 'nir'):
        f = op.join(store, f'true_alias_{cam}.fits')
        d[f'true_alias_{cam}'] = load(store, f'true_alias_{cam}') if op.exists(f) else d[f'meas_{cam}']
    out = dict(dt=dt, skip=SKIP, rmod=RMOD, n=NSUB, seeing=SEEING, fov=FOV, **p)
    out['freq'], out['turb'] = psd_of(d['turb_modes'], 1 / dt)
    _, out['res'] = psd_of(d['res_modes'], 1 / dt)
    out['var_turb'], out['var_res'] = d['turb_modes'].var(0), d['res_modes'].var(0)
    for cam in ('vis', 'nir'):
        m, a = d[f'meas_{cam}'], d[f'true_alias_{cam}']
        for key, x in ((f'meas_{cam}', m), (f'true_alias_{cam}', a), (f'meas_minus_alias_{cam}', m - a)):
            _, out[key] = psd_of(x, 1 / dt)
            out['var_' + key] = x.var(0)
    out['turb_rms'], out['res_rms'] = load(store, 'turb_rms').ravel(), load(store, 'res_rms').ravel()
    tmp = res_path[:-4] + '.tmp.npz'
    np.savez(tmp, **out)
    os.replace(tmp, res_path)


def run(out, names, gpu, total_time):
    root = setup_root(out)
    cfg = c.E2EConfig(calib_dir=root)
    for d in ('psd', 'runs'):
        os.makedirs(op.join(out, d), exist_ok=True)
    for name in names:
        p = RUNS[name]
        store, res_path = op.join(out, 'runs', name), op.join(out, 'psd', name + '.npz')
        if op.exists(res_path):
            continue
        if not op.exists(op.join(root, 'data', vector_stem(p['N'], p['level']) + '.fits')):
            raise FileNotFoundError(f'{name}: correction vector missing, run `psd_compare.py levels` first')
        t0 = time.time()
        if not op.exists(op.join(store, 'slopes_nir.fits')):
            c.run_specula(cfg, [YML], overrides(cfg, p, store, total_time), gpu, name)
        analyse(name, p, store, res_path, cfg.time_step)
        print(f'{time.strftime("%F %T")} {name}: {time.time() - t0:.0f} s', flush=True)
    print(f'{time.strftime("%F %T")} done', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('levels', 'run'))
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--time', type=float, default=8.0)
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--N', type=int, nargs='+', default=list(N_LIST), help='levels: the N to compute')
    a = ap.parse_args()
    if a.step == 'levels':
        levels(a.out, a.N)
    else:
        run(a.out, a.only or list(RUNS), a.gpu, a.time)


if __name__ == '__main__':
    main()
