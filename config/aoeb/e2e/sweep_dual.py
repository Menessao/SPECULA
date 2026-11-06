"""Dual-WFS Kalman loop (unmodulated + modulated pyramid). Metrics: mean and peak raw contrast in 2-5 lambda/D at 550 nm, SR.

--compare: the three high-flux runs on the unmodulated sensor, both pyramids 48x48, 1300 modes, 2 kHz, slope nulls on both:
    svd     SVD reconstructor
    ml      ML reconstructor (noise weights at the flux of the run)
    bias    SVD reconstructor, offset of the unmodulated sensor estimated too (initial value 0, prior variance BIAS_VAR)

The modulated sensor is the reference of the filter (never biased). source_vis / pyr_vis stay at 900 nm and source_nir / pyr_nir at 1300 nm,
with the magnitudes of the calibration frames: --unmod-wvl only chooses which of the two is unmodulated. Every file this script writes has its
own tag (sweep_dual_*), not shared with the notebooks or sweep_contrast.py. Finished runs are read back from results.csv.

    python sweep_dual.py --compare --gpu 0
"""
import argparse
import csv
import os
import os.path as op
import sys

import numpy as np
import yaml
from astropy.io import fits

from sweep_contrast import DELAY, SIGMA_E, ar2_at, check_ar2, metrics       # also specula.init(-1)
from specula.data_objects.slopes import Slopes                              # noqa: E402

import calibration                                                          # noqa: E402
from save_kalman_data import save_kalman_data, save_plant_inverse_iir       # noqa: E402
from utils import (SEEING, _load_calib, calib_dir, get_aliasing_power, get_pupil_flux, get_rec, get_static_slopes, latest_tn, machine,  # noqa: E402
                   noise_sensitivities, paths, propagated_noise_var, result_dir, run_specula)

N_SUBAPS, N_MODES, RMOD_MOD, RMOD_UNMOD = 48, 1300, 2.0, 0.0
FS = 2000.0
MAGS = {900.0: 7.0, 1300.0: 6.43}       # source_vis, source_nir of ristretto_main.yml: the magnitudes of the calibration frames
BANDS = {900.0: 'vis', 1300.0: 'nir'}
BIAS_VAR = 0.1 ** 2                     # [rec units^2] prior of the offset around zero (slope nulls), as in run_simul
ALIAS_SEEINGS = (0.75, 1.05)            # the s0.9 frames were renamed from the old e2e and have no slopes file
CASES = ('svd', 'ml', 'bias')
FIELDS = ('case', 'unmod_wvl', 'contrast_vis', 'peak_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'flux_vis', 'flux_ratio_vis', 'flux_nir',
          'flux_ratio_nir', 'noise_med_unmod', 'alias_med_unmod', 'tn')


def ml_rec(rmod, wvl):
    """Path of the ML reconstructor, noise weights at the flux of one 2 kHz frame at the calibration magnitude (as calibration.compute_ml)."""
    frame_path = paths(wvl).avgframe(rmod, N_SUBAPS, SEEING, N_MODES)
    frame = fits.getdata(frame_path)
    nph = float(np.sum(frame)) / calibration.avgframe_steps(frame_path)
    return machine().ensure_rec(rmod, N_SUBAPS, N_MODES, ('ml', nph), SEEING, wvl)


def noise_var(rec, rmod, wvl, flux):
    """Per-mode noise variance of a reconstructor with the frame of that sensor (utils.get_noise_propagation for any rec)."""
    frame, pup, _ = _load_calib(rmod, N_SUBAPS, N_MODES, wvl)
    total = flux * np.sum(frame) / np.sum(frame.ravel()[pup[pup >= 0]])     # pupil flux -> frame flux, the convention of noise_sensitivities
    s_phot, s_read = noise_sensitivities(rec, frame, pup)
    phot, read = propagated_noise_var(s_phot, s_read, flux=total, sigma_e=SIGMA_E)
    return phot + read


def ml_aliasing(rec_ml, rmod, wvl):
    """SVD aliasing at s0.9 times the ML / SVD variance ratio of each mode, measured on the slopes of the nearby seeings."""
    rec_svd = get_rec(rmod, N_SUBAPS, N_MODES, wvl)
    ratios = []
    for s in ALIAS_SEEINGS:
        slopes = fits.getdata(paths(wvl).slopes(rmod, N_SUBAPS, s, N_MODES)).astype(float)
        ratios.append(np.var(rec_ml @ slopes.T, axis=1) / np.var(rec_svd @ slopes.T, axis=1))
    print(f'ML / SVD aliasing, {wvl:g} nm: median {np.median(ratios[0]):.3f} (s{ALIAS_SEEINGS[0]}), {np.median(ratios[1]):.3f} (s{ALIAS_SEEINGS[1]})', flush=True)
    return get_aliasing_power(rmod, N_SUBAPS, N_MODES, wvl) * np.mean(ratios, axis=0)


def sn_tag(rmod, wvl):
    """Slope nulls of one sensor (own tag: the file depends on the wavelength and the corrected modes)."""
    tag = f'sweep_dual_pyr{rmod:1.1f}_{N_SUBAPS}x{N_SUBAPS}_wvl{wvl:g}_corr{N_MODES}_sn'
    Slopes(slopes=get_static_slopes(rmod, N_SUBAPS, N_MODES, wvl)).save(op.join(calib_dir, 'slopenulls', tag + '.fits'))
    return tag


def build_case(case, unmod_wvl, unity):
    """Overrides and sensor quantities of one run."""
    mod_wvl = 2200.0 - unmod_wvl
    name = f'{case}_unmod{unmod_wvl:g}'
    rmods = {unmod_wvl: RMOD_UNMOD, mod_wvl: RMOD_MOD}
    ov, sensors, flux, unmod_terms = {}, [], {}, {}
    for wvl in (900.0, 1300.0):         # order of kalman.in_measurements_list
        band, rmod = BANDS[wvl], rmods[wvl]
        unmod = wvl == unmod_wvl
        flux[wvl] = get_pupil_flux(rmod, N_SUBAPS, N_MODES, wvl)
        if case == 'ml' and unmod:
            path = ml_rec(rmod, wvl)
            rec = np.asarray(fits.getdata(path))
            alias = ml_aliasing(rec, rmod, wvl)
        else:
            rec = get_rec(rmod, N_SUBAPS, N_MODES, wvl)
            path = paths(wvl).rec(rmod, N_SUBAPS, N_MODES)
            alias = get_aliasing_power(rmod, N_SUBAPS, N_MODES, wvl)
        sensor = dict(noise_var=noise_var(rec, rmod, wvl, flux[wvl]), aliasing=alias, frame_steps=1, biased=case == 'bias' and unmod)
        if sensor['biased']:
            sensor['bias_var'] = BIAS_VAR
        if unmod:
            unmod_terms = dict(noise_med_unmod=float(np.median(sensor['noise_var'])), alias_med_unmod=float(np.median(alias)))
        sensors.append(sensor)
        ov[f'source_{band}_override'] = {'magnitude': MAGS[wvl], 'wavelengthInNm': wvl}
        ov[f'pyr_{band}_override'] = {'pup_diam': N_SUBAPS, 'pup_dist': calibration.E2EConfig().pup_dist(N_SUBAPS), 'mod_amp': rmod, 'wavelengthInNm': wvl}
        ov[f'cred_{band}_override'] = {'dt': 1 / FS, 'readout_level': SIGMA_E}
        ov[f'pyr_slopes_{band}_override'] = {'pupdata_object': f'pyr_pupdata_{N_SUBAPS}x{N_SUBAPS}', 'sn_object': sn_tag(rmod, wvl), 'outputs': ['out_slopes', 'out_flux']}
        ov[f'pyr_modes_{band}_override'] = {'recmat_object': op.basename(path)[:-5]}
    kf_tag = f'sweep_dual_{name}'
    a1, a2, q = ar2_at(FS)
    save_kalman_data(kf_tag, calib_dir, a1, a2, q, sensors, n_modes=N_MODES, delay=1 + DELAY * FS)
    return name, flux, unmod_terms, {**ov,
        'kalman_override': {'kalman_filter_data_object': kf_tag, 'time_step': 1 / FS, 'gain_period': 1,
                            'inputs': {'in_measurements_list': ['pyr_modes_vis.out_modes', 'pyr_modes_nir.out_modes'], 'in_command': 'kpred.out_comm:-1'},
                            'outputs': ['out_projected_state', 'out_state'] + (['out_bias'] if case == 'bias' else [])},
        'kpred_override': {'delay': DELAY * FS, 'iir_filter_data_object': unity},
        'coro_psf_vis_override': {'verbose': False}}


def compare(out, gpu, total_time, unmod_wvl):
    store = op.join(out, 'runs')
    for d in (store, op.join(out, 'logs'), op.join(out, 'overrides')):
        os.makedirs(d, exist_ok=True)
    check_ar2()
    unity = f'sweep_dual_unity_{N_MODES}modes'
    save_plant_inverse_iir(unity, calib_dir, N_MODES, beta=0.0)
    path_csv = op.join(out, 'results.csv')
    done = set()
    if op.exists(path_csv):
        with open(path_csv) as f:
            done = {(r['case'], float(r['unmod_wvl'])) for r in csv.DictReader(f)}
    for case in CASES:
        if (case, unmod_wvl) in done:
            continue
        name, flux, terms, ov = build_case(case, unmod_wvl, unity)
        stored = ['sr_vis-coro_psf_vis.out_sr', 'coro_psf_vis_int-coro_psf_vis.out_int_coronagraph_psf_profile',
                  'sr_nir-coro_psf_nir.out_sr', 'coro_psf_nir_int-coro_psf_nir.out_int_coronagraph_psf_profile',
                  'atmo_modes-atmo_modes.out_value', 'res_modes-res_modes.out_modes',
                  'flux_vis-pyr_slopes_vis.out_flux', 'flux_nir-pyr_slopes_nir.out_flux']
        if case == 'bias':
            stored.append('kf_bias-kalman.out_bias')
        ov['main_override'] = {'time_step': 1 / FS, 'total_time': total_time, 'root_dir': calib_dir}
        ov['data_store_override'] = {'store_dir': store, 'inputs': {'input_list': stored}}
        stem = op.join(out, 'overrides', name)
        with open(stem + '.yml', 'w') as f:
            yaml.safe_dump(ov, f, sort_keys=False)
        run_specula('ristretto_main.yml', stem, log_file=op.join(out, 'logs', name + '.log'), extra_args=('--target', str(gpu)))
        tn = latest_tn(store)
        path = op.join(store, tn)
        cv, pv, sv = metrics(path, 'vis', FS)
        cn, _, sn = metrics(path, 'nir', FS)
        row = dict(case=case, unmod_wvl=unmod_wvl, contrast_vis=cv, peak_vis=pv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, tn=tn, **terms)
        for wvl, band in BANDS.items():
            row[f'flux_{band}'] = flux[wvl]
            row[f'flux_ratio_{band}'] = float(np.mean(fits.getdata(op.join(path, f'flux_{band}.fits')))) / flux[wvl]
        new = not op.exists(path_csv)
        with open(path_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{name}: contrast {cv:.3e}  peak {pv:.3e}  SR {sv:.4f}  flux ratios {row['flux_ratio_vis']:.3f} {row['flux_ratio_nir']:.3f}  {tn}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--compare', action='store_true', help='the three high-flux runs')
    ap.add_argument('--unmod-wvl', type=float, default=900.0, choices=sorted(MAGS), help='[nm] sensing wavelength of the unmodulated pyramid')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--total-time', type=float, default=3.0, help='[s] of each run')
    ap.add_argument('--out', default=op.join(result_dir, 'sweep_dual_compare_20261106'))
    a = ap.parse_args()
    if not a.compare:
        ap.error('only --compare is implemented')
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']   # the specula CLI of this environment
    compare(a.out, a.gpu, a.total_time, a.unmod_wvl)


if __name__ == '__main__':
    main()
