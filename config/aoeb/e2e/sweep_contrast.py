"""Greedy search of the best single-WFS configuration (pyr_vis only). Metric: mean raw contrast in 2-5 lambda/D at 550 nm; SR saved too.

  1. modulation at mag 7, 2 kHz; the best one is kept (--rmod: use this one instead)
  2. mag 8...12: run at the frequency of the previous magnitude, then lower it by 100 Hz until the contrast stops improving
  3. check: the other modulations at mag 12, at its best frequency (reported only)

Single-rate loop at fs: time step and camera integration 1/fs, delay tau = 250 us * fs, Kalman filter predicting 1 + tau steps ahead
with AR(2), noise and flux of that (rMod, magnitude, fs); slope nulls of the rMod. Finished runs are read back from results.csv, so an
interrupted sweep is resumed by starting it again.

    python sweep_contrast.py --gpu 0
"""
import argparse
import csv
import os
import os.path as op
import sys

import numpy as np
import yaml
from astropy.io import fits

import specula
specula.init(-1)    # CPU: only the data objects, the simulations run in a separate process
from specula.data_objects.kalman_filter_data import ar2_driving_noise_variance, ar2_stationary_variance  # noqa: E402
from specula.data_objects.slopes import Slopes                                                           # noqa: E402

from results_analysis import load_coro_profile                                                           # noqa: E402
from save_kalman_data import save_kalman_data, save_plant_inverse_iir                                    # noqa: E402
from utils import (calib_dir, get_aliasing_power, get_atmo_coeffs, get_noise_propagation, get_pupil_flux, get_rec,  # noqa: E402
                   get_static_slopes, latest_tn, paths, result_dir, run_specula)

N_SUBAPS, N_MODES, WVL, SIGMA_E = 48, 1300, 900.0, 0.5
DT0, ZETA = 5e-4, 0.7                  # AR(2) file: sampling, damping of aoeb/pol_turb/ar2_mode_laws.ar2_model
DELAY = 250e-6                         # [s]
RMODS = (0.0, 0.5, 1.0, 2.0, 3.0)
MAG_START, MAGS = 7.0, (8.0, 9.0, 10.0, 11.0, 12.0)
MAG_CAL = 7.0                          # magnitude of the calibration frames (ristretto_main.yml)
FS_START, FS_STEP = 2000.0, 100.0
START_TIME = 0.5                       # [s] coro_psf_*.start_time of ristretto_main.yml
RAD = (2.0, 5.0)                       # [lambda/D]
FLUX_TOL, SMALL_CHANGE = 0.10, 0.01    # warning thresholds
UNITY_TAG = f'sweep_contrast_unity_{N_MODES}modes'   # not shared with the notebooks: another process rewriting it mid-sweep changes every later run
FIELDS = ('phase', 'rMod', 'mag', 'fs', 'tau', 'flux_pred', 'flux_ratio', 'contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'notes', 'tn')


def ar2_at(fs):
    """AR(2) of the 2 kHz file at another rate: same natural frequency and variance of each mode."""
    a1, a2, q = get_atmo_coeffs(DT0)
    omega = -np.log(np.sqrt(-a2)) / (ZETA * DT0)
    var = ar2_stationary_variance(a1, a2, q)
    r = np.exp(-ZETA * omega / fs)
    new_a1 = 2 * r * np.cos(omega / fs * np.sqrt(1 - ZETA ** 2))
    new_a2 = -r ** 2
    return new_a1, new_a2, ar2_driving_noise_variance(new_a1, new_a2, var)


def check_ar2():
    for new, old in zip(ar2_at(1 / DT0), get_atmo_coeffs(DT0)):
        if not np.allclose(new, old, rtol=1e-6, atol=0):
            raise RuntimeError('ar2_at does not reproduce the 2 kHz AR(2) file: is its damping still ZETA?')


def flux_per_frame(rmod, mag, fs):
    """Pupil flux of one frame [e-]: get_pupil_flux is per DT0 at MAG_CAL."""
    return get_pupil_flux(rmod, N_SUBAPS, N_MODES, WVL) / (fs * DT0) * 10 ** (-0.4 * (mag - MAG_CAL))


def sn_tag(rmod):
    """Slope null of the rMod (same tag as run_simul.ipynb), saved where sn_object looks for it."""
    tag = f'pyr{rmod:1.1f}_{N_SUBAPS}x{N_SUBAPS}_vis_sn'
    os.makedirs(op.join(calib_dir, 'slopenulls'), exist_ok=True)
    Slopes(slopes=get_static_slopes(rmod, N_SUBAPS, N_MODES, WVL)).save(op.join(calib_dir, 'slopenulls', tag + '.fits'))
    return tag


def build_filter(rmod, mag, fs):
    """Kalman data of the case: one sensor, one sample per frame, so the aliasing is the snapshot one."""
    tag = f'sweep_contrast_r{rmod:1.1f}_m{mag:g}_fs{fs:g}'
    a1, a2, q = ar2_at(fs)
    args = (rmod, N_SUBAPS, N_MODES, WVL)
    sensor = dict(noise_var=get_noise_propagation(*args, SIGMA_E, flux_per_frame(rmod, mag, fs), pupil_flux=True) ** 2,
                  aliasing=get_aliasing_power(*args), frame_steps=1, biased=False)
    save_kalman_data(tag, calib_dir, a1, a2, q, [sensor], n_modes=N_MODES, delay=1 + DELAY * fs)
    return tag


def overrides(rmod, mag, fs, kf_tag, store_dir, total_time):
    dt = 1 / fs
    get_rec(rmod, N_SUBAPS, N_MODES, WVL)   # computed and saved if missing
    rec = op.basename(paths(WVL).rec(rmod, N_SUBAPS, N_MODES))[:-5]
    return {
        'remove': ['source_nir', 'pyr_nir', 'cred_nir', 'pyr_slopes_nir', 'pyr_modes_nir', 'dm_modes', 'atmo_modes'],
        'main_override': {'time_step': dt, 'total_time': total_time, 'root_dir': calib_dir},
        'prop_override': {'source_dict_ref': ['source_vis', 'science_vis', 'science_nir'],
                          'outputs': ['out_source_vis_ef', 'out_science_vis_ef', 'out_science_nir_ef']},
        'source_vis_override': {'magnitude': mag, 'wavelengthInNm': WVL},
        'pyr_vis_override': {'pup_diam': N_SUBAPS, 'mod_amp': rmod, 'wavelengthInNm': WVL},
        'cred_vis_override': {'dt': dt, 'readout_level': SIGMA_E},
        'pyr_slopes_vis_override': {'pupdata_object': f'pyr_pupdata_{N_SUBAPS}x{N_SUBAPS}', 'sn_object': sn_tag(rmod),
                                    'outputs': ['out_slopes', 'out_flux']},
        'pyr_modes_vis_override': {'recmat_object': rec},
        'kalman_override': {'kalman_filter_data_object': kf_tag, 'time_step': dt, 'gain_period': 1,
                            'inputs': {'in_measurements_list': ['pyr_modes_vis.out_modes'], 'in_command': 'kpred.out_comm:-1'},
                            'outputs': ['out_projected_state', 'out_state']},   # no biased sensor: no out_bias
        'kpred_override': {'delay': DELAY * fs, 'iir_filter_data_object': UNITY_TAG},
        'coro_psf_vis_override': {'verbose': False},
        'data_store_override': {'store_dir': store_dir, 'inputs': {'input_list': [
            'sr_vis-coro_psf_vis.out_sr', 'coro_psf_vis_int-coro_psf_vis.out_int_coronagraph_psf_profile',
            'sr_nir-coro_psf_nir.out_sr', 'coro_psf_nir_int-coro_psf_nir.out_int_coronagraph_psf_profile',
            'res_modes-res_modes.out_modes', 'flux_vis-pyr_slopes_vis.out_flux']}},
    }


def metrics(path, band, fs):
    """Mean contrast in RAD and mean SR after START_TIME."""
    rad, prof = load_coro_profile(path, band)
    sr = fits.getdata(op.join(path, f'sr_{band}.fits'))
    return float(np.mean(prof[(rad >= RAD[0]) & (rad <= RAD[1])])), float(np.mean(sr[int(round(START_TIME * fs)):]))


class Sweep:
    def __init__(self, out, gpu, total_time, rmod=None):
        self.out, self.gpu, self.total_time, self.rmod = out, gpu, total_time, rmod
        self.store = op.join(out, 'runs')
        self.csv = op.join(out, 'results.csv')
        for d in (self.store, op.join(out, 'logs'), op.join(out, 'overrides')):
            os.makedirs(d, exist_ok=True)
        self.rows = {}
        if op.exists(self.csv):
            with open(self.csv) as f:
                for r in csv.DictReader(f):
                    for k in FIELDS:
                        if k not in ('phase', 'notes', 'tn'):
                            r[k] = float(r[k])
                    self.rows[(r['rMod'], r['mag'], r['fs'])] = r
            print(f'{len(self.rows)} runs read from {self.csv}', flush=True)

    def evaluate(self, phase, rmod, mag, fs):
        key = (rmod, mag, fs)
        if key in self.rows:
            return self.rows[key]
        name = f'r{rmod:1.1f}_m{mag:g}_fs{fs:g}'
        kf_tag = build_filter(rmod, mag, fs)
        ov_stem = op.join(self.out, 'overrides', name)
        with open(ov_stem + '.yml', 'w') as f:
            yaml.safe_dump(overrides(rmod, mag, fs, kf_tag, self.store, self.total_time), f, sort_keys=False)
        # A crash is not a bad configuration: it stops the sweep (CalledProcessError), which is resumed by starting it again
        run_specula('ristretto_main.yml', ov_stem, log_file=op.join(self.out, 'logs', name + '.log'),
                    extra_args=('--target', str(self.gpu)))
        tn = latest_tn(self.store)
        path = op.join(self.store, tn)
        cv, sv = metrics(path, 'vis', fs)
        cn, sn = metrics(path, 'nir', fs)
        notes = []
        pred = flux_per_frame(rmod, mag, fs)
        ratio = float(np.mean(fits.getdata(op.join(path, 'flux_vis.fits')))) / pred
        if abs(ratio - 1) > FLUX_TOL:
            notes.append(f'flux {ratio:.2f} x model')
        if not np.isfinite(cv) or not np.isfinite(sv):
            notes.append('diverged')
            cv = np.inf
        row = dict(phase=phase, rMod=rmod, mag=mag, fs=fs, tau=DELAY * fs, flux_pred=pred, flux_ratio=ratio, contrast_vis=cv,
                   sr_vis=sv, contrast_nir=cn, sr_nir=sn, notes='; '.join(notes), tn=tn)
        self.rows[key] = row
        new = not op.exists(self.csv)
        with open(self.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{phase:9s} rMod {rmod:.1f} mag {mag:g} fs {fs:6.0f}: contrast {cv:.3e}  SR {sv:.4f}  {row['notes'] and 'WARNING ' + row['notes']}", flush=True)
        return row

    def note(self, key, text):
        """A warning found after the run: printed and in the table (results.csv keeps the run data only)."""
        row = self.rows[key]
        if text not in row['notes']:
            row['notes'] = '; '.join(x for x in (row['notes'], text) if x)
        print(f'WARNING rMod {key[0]:.1f} mag {key[1]:g} fs {key[2]:.0f}: {text}', flush=True)

    def run(self):
        check_ar2()
        save_plant_inverse_iir(UNITY_TAG, calib_dir, N_MODES, beta=0.0)
        rows = [self.evaluate('rMod', r, MAG_START, FS_START) for r in (RMODS if self.rmod is None else (self.rmod,))]
        best = min(rows, key=lambda x: x['contrast_vis'])
        rmod, fs = best['rMod'], FS_START
        selected = [(rmod, MAG_START, fs)]
        print(f'modulation: {rmod}', flush=True)
        for mag in MAGS:
            best = self.evaluate('magnitude', rmod, mag, fs)
            f = fs - FS_STEP
            while f >= FS_STEP:
                row = self.evaluate('frequency', rmod, mag, f)
                change = abs(row['contrast_vis'] - best['contrast_vis']) / best['contrast_vis'] if np.isfinite(best['contrast_vis']) else np.inf
                if change < SMALL_CHANGE:
                    self.note((rmod, mag, f), f'change {change:.2%} against {best["fs"]:.0f} Hz: may be noise')
                if not row['contrast_vis'] < best['contrast_vis']:
                    break
                best = row
                f -= FS_STEP
            fs = best['fs']
            if not np.isfinite(best['contrast_vis']):
                self.note((rmod, mag, fs), 'no converged frequency found: the search stopped at the first step')
            selected.append((rmod, mag, fs))
        checks = [self.evaluate('check', r, MAGS[-1], fs) for r in RMODS if r != rmod]
        if any(c['contrast_vis'] < self.rows[selected[-1]]['contrast_vis'] for c in checks):
            print(f'WARNING: another modulation beats rMod {rmod} at mag {MAGS[-1]:g}, {fs:.0f} Hz', flush=True)
        self.report(selected)

    def report(self, selected):
        def line(cells):
            return '| ' + ' | '.join(str(c) for c in cells) + ' |'
        def values(r):
            return [r['rMod'], f"{r['mag']:g}", f"{r['fs']:.0f}", f"{r['contrast_vis']:.3e}", f"{r['sr_vis']:.4f}"]
        head = ['rMod', 'mag', 'fs [Hz]', 'contrast 2-5 l/D, 550 nm', 'SR 550 nm']
        out = ['## Selected configuration per magnitude', '', line(head), line(['---'] * len(head))]
        out += [line(values(self.rows[k])) for k in selected]
        full = ['phase', *head, 'tau', 'flux/frame [e-]', 'flux meas/model', 'contrast 1650 nm', 'SR 1650 nm', 'selected', 'notes', 'TN']
        out += ['', '## All runs', '', line(full), line(['---'] * len(full))]
        for k, r in self.rows.items():
            out.append(line([r['phase'], *values(r), f"{r['tau']:.3f}", f"{r['flux_pred']:.3g}", f"{r['flux_ratio']:.3f}", f"{r['contrast_nir']:.3e}",
                             f"{r['sr_nir']:.4f}", 'yes' if k in selected else '', r['notes'], r['tn']]))
        text = '\n'.join(out) + '\n'
        with open(op.join(self.out, 'table.md'), 'w') as f:
            f.write(text)
        print(text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--total-time', type=float, default=3.0, help='[s] of each run')
    ap.add_argument('--out', default=op.join(result_dir, 'sweep_contrast_20261106'))
    ap.add_argument('--rmod', type=float, help='fixed modulation: skip the modulation selection')
    a = ap.parse_args()
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']   # the specula CLI of this environment
    Sweep(a.out, a.gpu, a.total_time, a.rmod).run()


if __name__ == '__main__':
    main()
