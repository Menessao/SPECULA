"""Parameter sweep of the main simulation (ristretto_main.yml, run_simul.ipynb): one option changed at a time from the baseline, at
high flux (magnitudes of the yml) and low flux (magnitudes + 5). Run from anywhere, with the python of the specula env:

    python sweep_main.py [--gpu 0] [--out DIR] [case ...]         (default: all the cases; a case is e.g. low_ml_rec)

Every case gets its own Kalman data, bias and overrides ('sweep_*' tags), the existing products are not touched. Table: <out>/sweep.md, .csv

  baseline         R rescaled with the measured flux, aliasing as AR(2) states, bias estimated, SVD rec, no slope nulls
  no_flux_scaling  R at the true flux of the run, no runtime rescaling
  white_aliasing   aliasing added to R as white noise, variance per frame of each sensor
  fixed_bias       bias fixed at its initial value (the static offset): zero prior variance
  ml_rec           NIR: ML rec (old e2e 'mmse'), noise weights at the flux of the run
  slope_null       NIR: slope null, mean slopes of a calib_wfs_aliasing.yml run (the bias becomes the offset left w.r.t. the VIS reference)

Only on request (value of the runtime rescaling):
  fast_flux_scaling   as baseline, flux_update_period = FAST_PERIOD, so that VIS rescales within the run too
  wrong_r_no_scaling  R at the flux of the magnitudes of the yml (wrong in a low-flux run), no runtime rescaling
  wrong_r_scaling     same wrong R, flux_update_period = FAST_PERIOD
  true_r_scaling, true_r_no_scaling                R at the flux really collected (see S09_FRAME_STEPS: all the others assume 4 times more)
  true_r_noclip_scaling, true_r_noclip_no_scaling  same, with PyrSlopec clip_negative = False on both sensors

The measured flux of both sensors is stored in every run (flux_vis, flux_nir).

Cases run again with the same --out replace their rows in the table.
"""
import argparse
import csv
import os
import os.path as op
import shutil
import time

import numpy as np
import yaml
from astropy.io import fits

import specula
specula.init(-1)    # CPU here: the simulations run in a separate process

from specula.data_objects.slopes import Slopes                                       # noqa: E402

import calibration as cal                                                             # noqa: E402
import utils as ut                                                                    # noqa: E402
from aoeb.filter_optimizers import aliasing_frame_fraction                            # noqa: E402
from save_kalman_data import save_kalman_data                                         # noqa: E402

DT = 0.0005             # [s] loop time step
TOTAL_TIME = 1.0        # [s]
N_SUBAPS, N_MODES = 48, 1300
SIGMA_E = 0.5           # [e-] RON
SEEING = ut.SEEING      # of the calibration products
SEEING_RUN = 0.88       # seeing.constant of ristretto_main.yml
VIS_RMOD = 3.0
NEVER = 10 ** 9         # flux_update_period longer than any run
MAGS = dict(source_vis=7.00, source_nir=6.43, science_vis=8.15, science_nir=4.97)       # ristretto_main.yml, also of the calibration frames
FLUXES = {'high': 0.0, 'low': 5.0}                                                      # magnitude offset
VARIANTS = ('baseline', 'no_flux_scaling', 'white_aliasing', 'fixed_bias', 'ml_rec', 'slope_null')
FLUX_VARIANTS = ('fast_flux_scaling', 'wrong_r_no_scaling', 'wrong_r_scaling', 'true_r_scaling', 'true_r_no_scaling',
                 'true_r_noclip_scaling', 'true_r_noclip_no_scaling')     # only on request (not in the default list)
S09_FRAME_STEPS = 4     # the s0.9 average frames are 2 ms exposures (4 times the flux of the other seeings), utils.get_pupil_flux reads them as one step
FAST_PERIOD = 100       # flux_update_period [frames of each WFS] of the variants that rescale fast
SKIP_SR = 500           # loop steps skipped in the mean SR
CONTRAST_RANGE = (2.0, 5.0)     # [lambda/D]
IIR_TAG = f'plant_inverse_{N_MODES}modes'                                               # built by run_simul
RESULT_COLUMNS = ['flux', 'variant', 'TN', 'SR_VIS', 'SR_NIR', 'contrast_VIS', 'contrast_NIR']


def sensor_defs(dmag):
    vis = dict(rMod=VIS_RMOD, wvl=900, mag_key='source_vis', frame_steps=4, biased=False)    # reference sensor
    nir = dict(rMod=0.0, wvl=1300, mag_key='source_nir', frame_steps=1, biased=True)    # unmodulated, offset estimated
    for s in (vis, nir):
        s['args'] = (s['rMod'], N_SUBAPS, N_MODES, s['wvl'])
        s['flux'] = ut.get_pupil_flux(*s['args']) * s['frame_steps'] * 10 ** (-0.4 * dmag)   # as PyrSlopec 'out_flux', per frame
    return vis, nir


def rec_tag(path):
    return op.basename(path)[:-5]


def ml_rec_path(s, dmag):
    """ML rec of a sensor; noise weights at the electrons per frame of the run (same convention as calibration.compute_ml)."""
    frame = fits.getdata(ut.paths(s['wvl']).avgframe(s['rMod'], N_SUBAPS, SEEING, N_MODES))
    n_phot = float(frame.sum()) * s['frame_steps'] * 10 ** (-0.4 * dmag)
    return ut.machine().ensure_rec(s['rMod'], N_SUBAPS, N_MODES, ('ml', n_phot), SEEING, s['wvl'])


def noise_var(rec, s):
    """(total, photon) noise variance per frame of a sensor, in rec units squared (utils.get_noise_propagation with any rec)."""
    frame, pup, _ = ut._load_calib(*s['args'])
    total_flux = s['flux'] * np.sum(frame) / np.sum(frame.ravel()[pup[pup >= 0]])
    s_phot, s_read = ut.noise_sensitivities(rec, frame, pup)
    phot, read = ut.propagated_noise_var(s_phot, s_read, flux=total_flux, sigma_e=SIGMA_E)
    return phot + read, phot


def snapshot_alias_from_slopes(rec, s):
    """Snapshot aliasing variance of `rec` at SEEING: the slopes of that seeing are not on disk, so those of the two neighbouring
    seeings are used, rescaled as seeing^(5/3), and averaged."""
    out = []
    for s0 in (0.75, 1.05):
        slopes = fits.getdata(ut.paths(s['wvl']).slopes(s['rMod'], N_SUBAPS, s0, N_MODES)).astype(float)
        out.append(np.var(rec @ slopes.T, axis=1) * (SEEING / s0) ** (5 / 3))
    return np.mean(out, axis=0)


def aliasing_var(rec, s, white, is_svd):
    """Aliasing variance: loop rate with the AR(2) model (the VIS model averages it over its 4 steps), per frame with white noise."""
    fs = 1 / (s['frame_steps'] * DT) if white else 1 / DT
    if is_svd:
        return ut.get_aliasing_power(*s['args'], fs=fs)
    return snapshot_alias_from_slopes(rec, s) * aliasing_frame_fraction(fs, ut.machine().alias_spectrum(s['wvl']))


def slope_null_tag(cfg, s, gpu):
    """Tag of the NIR slope null, in the SPECULA Slopes format: time average of slopes_nir of a noiseless calib_wfs_aliasing.yml run with
    the parameters of the sweep (seeing of ristretto_main.yml). Noiseless and normalised slopes: valid at both fluxes. Made once."""
    tag = f"sweep_pyr{s['rMod']:.1f}_{N_SUBAPS}x{N_SUBAPS}_wvl{s['wvl']}_s{SEEING_RUN}_{N_MODES}modes_sn"
    path = op.join(ut.calib_dir, 'slopenulls', tag + '.fits')
    if op.exists(path):
        return tag
    store = op.join(ut.calib_dir, 'scratch', tag)
    ov = cal.merge(cal._pyrs(cfg, VIS_RMOD, N_SUBAPS, which=('vis',)), cal._pyrs(cfg, s['rMod'], N_SUBAPS, which=('nir',)))
    ov = cal.merge(ov, {'main': {'total_time': cfg.alias_time}, 'atmo_modes': {'nmodes': N_MODES}, 'atmo_random': {'seed': cfg.seed},
                        'data_store': {'store_dir': store, 'create_tn': False}})
    cal.run_specula(cfg, ['calib_wfs_aliasing.yml'], ov, gpu, tag)
    sn = fits.getdata(op.join(store, 'slopes_nir.fits')).mean(axis=0)
    Slopes(slopes=sn.astype(np.float32), target_device_idx=-1).save(path, overwrite=True)
    shutil.rmtree(store)
    return tag


def build_case(cfg, flux, variant, gpu):
    """Writes the Kalman data of the case and returns the overrides of the simulation."""
    dmag = FLUXES[flux]
    vis, nir = sensor_defs(dmag)
    a1, a2, q = ut.get_atmo_coeffs(DT)
    white = variant == 'white_aliasing'

    rec_vis = ut.get_rec(*vis['args'])
    if variant == 'ml_rec':
        rec_nir_path = ml_rec_path(nir, dmag)
        rec_nir = np.asarray(fits.getdata(rec_nir_path), float)
    else:
        rec_nir_path = ut.paths(nir['wvl']).rec(nir['rMod'], N_SUBAPS, N_MODES)
        rec_nir = ut.get_rec(*nir['args'])
    rec_vis_path = ut.paths(vis['wvl']).rec(vis['rMod'], N_SUBAPS, N_MODES)

    # Static offset of NIR w.r.t. VIS, as in run_simul, with the rec of this case. The slope null removes the static measurement of NIR
    frame_nir, pup_nir, _ = ut._load_calib(*nir['args'])
    sn_tag = None
    if variant == 'slope_null':
        sn_tag = slope_null_tag(cfg, nir, gpu)
        static_nir = 0.0
    else:
        static_nir = rec_nir @ ut.pyr_slopes_from_frame(frame_nir, pup_nir)
    bias = static_nir - ut.get_static_modes(*vis['args'])

    model_vis, model_nir = sensor_defs(0.0) if variant.startswith('wrong_r') else (vis, nir)     # flux assumed by the noise model
    if variant.startswith('true_r'):
        model_vis, model_nir = [dict(s, flux=s['flux'] / S09_FRAME_STEPS) for s in (vis, nir)]
    sensors = []
    for s, rec, is_svd in ((model_vis, rec_vis, True), (model_nir, rec_nir, variant != 'ml_rec')):   # order of kalman.inputs.in_measurements_list
        total, phot = noise_var(rec, s)
        sensors.append(dict(noise_var=total, photon_noise_var=phot, reference_flux=s['flux'],
                            aliasing=aliasing_var(rec, s, white, is_svd), frame_steps=s['frame_steps'], biased=s['biased']))
    sensors[1]['bias'] = bias
    if variant == 'fixed_bias':
        sensors[1]['bias_var'] = 0.0

    kf_tag = f'sweep_kf_{flux}_{variant}'
    save_kalman_data(kf_tag, ut.calib_dir, a1, a2, q, sensors, n_modes=N_MODES, aliasing_model='white' if white else 'ar2')

    ov = {
        'main': {'time_step': DT, 'total_time': TOTAL_TIME},
        'kalman': {'kalman_filter_data_object': kf_tag, 'time_step': DT},
        'kpred': {'iir_filter_data_object': IIR_TAG},
    }
    if variant.endswith('no_scaling') or variant == 'no_flux_scaling':
        ov['kalman']['flux_update_period'] = NEVER      # keeps the flux inputs of the yml connected, R is never rescaled
    elif variant in ('fast_flux_scaling', 'wrong_r_scaling'):
        ov['kalman']['flux_update_period'] = FAST_PERIOD
    with open(op.join(ut.E2E_DIR, cfg.main_yml)) as f:
        store_list = yaml.safe_load(f)['data_store']['inputs']['input_list']
    ov['data_store'] = {'inputs': {'input_list': [*store_list, 'flux_vis-pyr_slopes_vis.out_flux', 'flux_nir-pyr_slopes_nir.out_flux']}}
    for key, mag in MAGS.items():
        ov[key] = {'magnitude': mag + dmag}
    for p, s, path in (('vis', vis, rec_vis_path), ('nir', nir, rec_nir_path)):
        ov[f'pyr_{p}'] = {'pup_diam': float(N_SUBAPS), 'mod_amp': s['rMod'], 'wavelengthInNm': s['wvl']}
        ov[f'source_{p}']['wavelengthInNm'] = s['wvl']
        ov[f'cred_{p}'] = {'dt': s['frame_steps'] * DT, 'readout_level': SIGMA_E}
        ov[f'pyr_slopes_{p}'] = {'pupdata_object': f'pyr_pupdata_{N_SUBAPS}x{N_SUBAPS}'}
        ov[f'pyr_modes_{p}'] = {'recmat_object': rec_tag(path)}
    if sn_tag is not None:
        ov['pyr_slopes_nir']['sn_object'] = sn_tag
    if 'noclip' in variant:
        for p in ('vis', 'nir'):
            ov[f'pyr_slopes_{p}']['clip_negative'] = False
    return ov


def metrics(tn):
    """Mean SR after SKIP_SR loop steps; mean of the integrated coronagraphic profile (bins in CONTRAST_RANGE) of a run."""
    out = {}
    for band in ('vis', 'nir'):
        sr = fits.getdata(op.join(ut.result_dir, tn, f'sr_{band}.fits'))
        r, prof = fits.getdata(op.join(ut.result_dir, tn, f'coro_psf_{band}_int.fits'))[0]
        mask = (r >= CONTRAST_RANGE[0]) & (r <= CONTRAST_RANGE[1])
        out[f'SR_{band.upper()}'] = float(np.mean(sr[SKIP_SR:]))
        out[f'contrast_{band.upper()}'] = float(np.mean(prof[mask]))
    return out


def run_case(cfg, flux, variant, gpu):
    ov = build_case(cfg, flux, variant, gpu)
    before = set(os.listdir(ut.result_dir))
    cal.run_specula(cfg, [], ov, gpu, f'sweep_{flux}_{variant}')
    new = sorted(d for d in set(os.listdir(ut.result_dir)) - before if d[:8].isdigit())
    if len(new) != 1:
        raise RuntimeError(f'{flux}_{variant}: expected one new TN, found {new}')
    return dict(flux=flux, variant=variant, TN=new[0], **metrics(new[0]))


def write_table(rows, out):
    with open(op.join(out, 'sweep.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, RESULT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    lines = ['| ' + ' | '.join(RESULT_COLUMNS) + ' |', '|' + '---|' * len(RESULT_COLUMNS)]
    for r in rows:
        lines.append(f"| {r['flux']} | {r['variant']} | {r['TN']} | {r['SR_VIS']:.4f} | {r['SR_NIR']:.4f} "
                     f"| {r['contrast_VIS']:.3e} | {r['contrast_NIR']:.3e} |")
    table = '\n'.join(lines) + '\n'
    with open(op.join(out, 'sweep.md'), 'w') as f:
        f.write(table)
    return table


def main():
    all_cases = [f'{f}_{v}' for f in FLUXES for v in VARIANTS]
    known_cases = [f'{f}_{v}' for f in FLUXES for v in (*VARIANTS, *FLUX_VARIANTS)]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='*', choices=[[], *known_cases], default=[], help='default: all but the on-request variants')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--out', default=op.join(ut.result_dir, 'sweep_' + time.strftime('%Y%m%d_%H%M%S')))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = cal.E2EConfig(calib_dir=ut.calib_dir)
    rows = {}
    csv_path = op.join(args.out, 'sweep.csv')
    if op.exists(csv_path):
        with open(csv_path, newline='') as f:
            for r in csv.DictReader(f):
                rows[f"{r['flux']}_{r['variant']}"] = {k: r[k] if k in ('flux', 'variant', 'TN') else float(r[k]) for k in r}
    for case in args.cases or all_cases:
        flux, variant = case.split('_', 1)
        t0 = time.time()
        rows[case] = run_case(cfg, flux, variant, args.gpu)
        print(f'{case}: {rows[case]} [{time.time() - t0:.0f} s]', flush=True)
        table = write_table([rows[c] for c in known_cases if c in rows], args.out)     # after every run: a failure keeps what is done
    print(table)
    print('saved in', args.out)


if __name__ == '__main__':
    main()
