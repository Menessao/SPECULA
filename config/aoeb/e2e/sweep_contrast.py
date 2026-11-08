"""Greedy search of the best single-WFS configuration. Metric: mean raw contrast in 2-5 lambda/D in METRIC_BAND (550 nm for the 900 nm WFS,
1650 nm for the 1300 nm WFS), or SR in that band when SR < SR_SWITCH (the contrast of an uncorrected loop gets better as the correction collapses);
a run above SR_SWITCH always beats one below it.

  1. modulation at mag 7, 2 kHz, first rung; the best one is kept (--rmod: use this one instead)
  2. mag 8...14, from the (frequency, rung) of the previous magnitude: lower the frequency by 100 Hz until the metric stops improving, then
     the next rung of the LADDER (--ladder; smaller WFS, fewer modes) until it stops improving; after a change of rung the frequency is searched
     again, upwards too if the first step down does not improve. The rung never goes back.
  3. check: the other modulations at the faintest magnitude, at its best configuration (reported only; --no-check skips it)

Single-rate loop at fs: time step and camera integration 1/fs, delay tau = 250 us * fs, Kalman filter predicting 1 + tau steps ahead
with AR(2), noise and flux of that (rMod, magnitude, fs, rung); slope nulls of the rMod. `mag` is always the 900 nm magnitude, the star colour
fixed (SOURCE_MAG_CAL), so the 900 and 1300 nm sensors are compared on the same star. The seeing of the simulation is set so that the seeing along
the line of sight equals SEEING, the seeing of the calibration products (sim_seeing). --rec auto: ML reconstructor for rMod 0, SVD otherwise.
With the ML reconstructor at rMod 0 a run whose SR runs away (runaway) is repeated with the SVD filter model (noise and aliasing of the SVD
reconstructor) and the better of the two is kept, flagged in the notes.

Every Kalman/IIR file has the tag of the sweep folder, so concurrent sweeps never share a mutable file. Finished runs are read back from
results.csv, so an interrupted sweep is resumed by starting it again. --replay reruns the selected configurations (selected.csv) with other
seeds and wind directions.

    python sweep_contrast.py --gpu 0 --wvl 900 --rmod 0 --rec auto --ladder --no-check --out <folder>
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

import calibration                                                                                       # noqa: E402
from results_analysis import load_coro_profile                                                           # noqa: E402
from aoeb.noise import slope_noise_cov                                                                   # noqa: E402
from aoeb.reconstructors import compute_reconstructor, save_rec                                          # noqa: E402
from save_kalman_data import save_kalman_data, save_plant_inverse_iir                                    # noqa: E402
from utils import (E2E_DIR, SEEING, _load_calib, calib_dir, get_aliasing_power, get_atmo_coeffs, get_noise_propagation, get_pupil_flux,  # noqa: E402
                   get_rec, get_static_slopes, latest_tn, machine, noise_sensitivities, paths, propagated_noise_var, result_dir, run_specula)

N_SUBAPS, N_MODES, WVL, SIGMA_E = 48, 1300, 900.0, 0.5
LADDER = ((48, 1300), (36, 900), (24, 300), (14, 100))   # (subapertures, modes) of the --ladder search
BANDS = {900.0: 'vis', 1300.0: 'nir'}                    # chain of ristretto_main.yml of each sensing wavelength
SOURCE_MAG_CAL = {900.0: 7.0, 1300.0: 6.43}              # source_vis, source_nir of ristretto_main.yml: magnitudes of the calibration frames
REC = 'svd'                            # reconstructor: 'svd', 'ml' (ML at the flux of each run) or 'auto' (ML at rMod 0, SVD otherwise)
METRIC_BAND = 'vis'                    # science band of the metric (sweep_dual sets it too)
CHROMATIC = True                       # prop.enable_chromatic_effect
ALIAS_SEEINGS = (0.75, 1.05)           # seeings with a slopes file, for the ML aliasing where the s0.9 frames were renamed from the old e2e and have none
DT0, ZETA = 5e-4, 0.7                  # AR(2) file: sampling, damping of aoeb/pol_turb/ar2_mode_laws.ar2_model
DELAY = 250e-6                         # [s]
RMODS = (0.0, 0.5, 1.0, 2.0, 3.0)
MAG_START, MAGS = 7.0, (8.0, 9.0, 10.0, 11.0, 12.0)   # 13 and 14 dropped on 2026-11-08
MAG_CAL = 7.0                          # 900 nm magnitude of the calibration frames
FS_START, FS_STEP = 2000.0, 100.0
START_TIME = 0.5                       # [s] coro_psf_*.start_time of ristretto_main.yml
RAD = (2.0, 5.0)                       # [lambda/D]
FLUX_TOL, SMALL_CHANGE = 0.10, 0.01    # warning thresholds
FLUX_ADJUST = 0.05                     # measured pupil flux / model beyond this: the run is repeated once with the model flux x measured / model
                                       # (ML weights and KF noise; the excess is light the worse-corrected PSF puts into the valid pixels)
IMPROVED = 0.01                        # relative gain of the frequency search that counts for start_magnitude
SR_SWITCH = 0.10                       # below this SR the runs are compared by SR, not contrast
RUNAWAY, RUNAWAY_WINDOW = 0.7, 0.5     # runaway: mean 550 nm SR of the last window below RUNAWAY x the mean of the window after START_TIME [s]
STORE_ALL = ['atmo_modes-atmo_modes.out_value', 'dm_modes-dm_modes.out_modes']   # saved in the sweep, not in the replays
FIELDS = ('phase', 'rMod', 'mag', 'fs', 'n_subaps', 'n_modes', 'kf_model', 'flux_corr', 'tau', 'flux_pred', 'flux_ratio', 'contrast_vis', 'peak_vis', 'sr_vis',
          'contrast_nir', 'sr_nir', 'sr_start', 'sr_end', 'notes', 'tn')
TEXT = ('phase', 'kf_model', 'notes', 'tn')
REPLAY_FIELDS = ('mag', 'config', 'realization', 'seed', 'wind_direction', 'contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'flux_ratio', 'notes', 'tn')


def band():
    return BANDS[WVL]


def sim_seeing():
    """Seeing of the yml that gives SEEING along the line of sight (AtmoEvolution: r0 x airmass^-3/5), the seeing of the calibration products."""
    with open(op.join(E2E_DIR, 'ristretto_main.yml')) as f:
        z = yaml.safe_load(f)['main']['zenithAngleInDeg'] or 0.0
    return float(SEEING * np.cos(np.radians(z)) ** 0.6)


def rec_of(rmod):
    return ('ml' if rmod == 0.0 else 'svd') if REC == 'auto' else REC


def ml_stem(rmod, n, N, nph):
    return f'sweep_contrast_pyr{rmod:1.1f}_{n}x{n}_wvl{WVL:g}_{N}modes_rec_ml_{nph:.4g}ph_s{SEEING}'


def frame_electrons(rmod, mag, fs, n, N, fcorr=1.0):
    """Electrons of one frame on the whole detector, the flux of the noise weights of the ML reconstructor; fcorr: measured / model flux."""
    path = paths(WVL).avgframe(rmod, n, SEEING, N)
    return float(np.sum(fits.getdata(path))) / calibration.avgframe_steps(path) / (fs * DT0) * 10 ** (-0.4 * (mag - MAG_CAL)) * fcorr


def atomic_write(path, write):
    """write(tmp) then rename: an interrupted or concurrent write never leaves a partial file under the final name."""
    tmp = f'{path[:-5]}_tmp{os.getpid()}.fits'
    write(tmp)
    os.replace(tmp, path)


def ml_rec(rmod, mag, fs, n, N, fcorr=1.0):
    """Stem and matrix of the ML reconstructor at the flux of the run. Its content depends on the stem only, so sweeps may share it."""
    nph = frame_electrons(rmod, mag, fs, n, N, fcorr)
    stem = ml_stem(rmod, n, N, nph)
    path = op.join(calib_dir, 'rec', stem + '.fits')
    if not op.exists(path):
        m, P = machine(), paths(WVL)
        frame = fits.getdata(P.avgframe(rmod, n, SEEING, N))
        cn, cxy = slope_noise_cov(frame, m.slope_geometry(rmod, n, SEEING, N, WVL, frame), nph, m.detector, m.slopes_from_intensity)
        rec = compute_reconstructor(fits.getdata(P.im_dl(rmod, n)), N, 'ml', cn_diag=cn, cn_xy=cxy)
        atomic_write(path, lambda p: save_rec(p, rec, method='ml', n_phot=nph, seeing=SEEING, wvl_nm=WVL))
    return stem, np.asarray(fits.getdata(path))


def ml_aliasing(rec, rmod, n, N):
    """Aliasing variance of the ML reconstructor: from the slopes at the seeing of the run or, if there are none, the SVD aliasing times the
    ML / SVD ratio of each mode on the slopes of the nearby seeings."""
    P = paths(WVL)
    if op.exists(P.slopes(rmod, n, SEEING, N)):
        return np.var(rec @ fits.getdata(P.slopes(rmod, n, SEEING, N)).astype(float).T, axis=1)
    svd = get_rec(rmod, n, N, WVL)
    ratios = []
    for s in ALIAS_SEEINGS:
        slopes = fits.getdata(P.slopes(rmod, n, s, N)).astype(float)
        ratios.append(np.var(rec @ slopes.T, axis=1) / np.var(svd @ slopes.T, axis=1))
    return get_aliasing_power(rmod, n, N, WVL) * np.mean(ratios, axis=0)


def noise_variance(rec, rmod, flux, n, N):
    """Per-mode noise variance of any reconstructor (utils.get_noise_propagation uses the SVD one). flux: on the four pupils, per frame."""
    frame, pup, _ = _load_calib(rmod, n, N, WVL)
    total = flux * np.sum(frame) / np.sum(frame.ravel()[pup[pup >= 0]])     # pupil flux -> frame flux, the convention of noise_sensitivities
    s_phot, s_read = noise_sensitivities(rec, frame, pup)
    phot, read = propagated_noise_var(s_phot, s_read, flux=total, sigma_e=SIGMA_E)
    return phot + read


def read_rows(out, name='results.csv'):
    """Rows of a finished or running sweep, in run order (older sweeps miss some columns: filled with the defaults of this sweep)."""
    with open(op.join(out, name)) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r.setdefault('n_subaps', N_SUBAPS)
        r.setdefault('n_modes', N_MODES)
        r.setdefault('kf_model', REC)
        r.setdefault('flux_corr', 1.0)
        for k in list(r):
            if k in FIELDS and k not in TEXT and r[k] not in ('', None):
                r[k] = float(r[k])
    return rows


def score(r):
    """Sort key of a run, best first: contrast at SR >= SR_SWITCH, then SR, then the diverged runs (METRIC_BAND)."""
    c, s = r[f'contrast_{METRIC_BAND}'], r[f'sr_{METRIC_BAND}']
    if not (np.isfinite(c) and np.isfinite(s)):
        return (2, 0.0)
    return (0, c) if s >= SR_SWITCH else (1, -s)


def better(a, b):
    return score(a) < score(b)


def rel_change(a, b):
    """Relative difference of the metric of two runs against b; inf if they are not compared on the same metric or both diverged."""
    (ra, va), (rb, vb) = score(a), score(b)
    return abs(va - vb) / abs(vb) if ra == rb and ra < 2 else np.inf


def start_magnitude(out):
    """First magnitude of a sweep where the frequency search gained at least IMPROVED over the frequency that magnitude started from."""
    rows = [r for r in read_rows(out) if r['phase'] != 'check']
    for mag in sorted({r['mag'] for r in rows}):
        runs = [r for r in rows if r['mag'] == mag]
        best = min(runs, key=score)
        if better(best, runs[0]) and rel_change(best, runs[0]) >= IMPROVED:
            return mag
    print(f'WARNING: the frequency never gained {IMPROVED:.0%} in {out}: starting from mag {MAGS[-1]:g}', flush=True)
    return MAGS[-1]


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


def flux_per_frame(rmod, mag, fs, n, N):
    """Pupil flux of one frame [e-]: get_pupil_flux is per DT0 at MAG_CAL."""
    return get_pupil_flux(rmod, n, N, WVL) / (fs * DT0) * 10 ** (-0.4 * (mag - MAG_CAL))


def sn_tag(rmod, n, N):
    """Slope null of the sensor, saved where sn_object looks for it. Its content depends on the tag only, so sweeps may share it."""
    tag = f'sweep_contrast_pyr{rmod:1.1f}_{n}x{n}_wvl{WVL:g}_{N}modes_sn'
    path = op.join(calib_dir, 'slopenulls', tag + '.fits')
    if not op.exists(path):
        atomic_write(path, lambda p: Slopes(slopes=get_static_slopes(rmod, n, N, WVL)).save(p))
    return tag


def metrics(path, band, fs):
    """Mean and maximum contrast in RAD, mean SR after START_TIME."""
    rad, prof = load_coro_profile(path, band)
    sr = fits.getdata(op.join(path, f'sr_{band}.fits'))
    in_rad = prof[(rad >= RAD[0]) & (rad <= RAD[1])]
    return float(np.mean(in_rad)), float(np.max(in_rad)), float(np.mean(sr[int(round(START_TIME * fs)):]))


def sr_drift(path, fs):
    """Mean 550 nm SR of the window after START_TIME and of the last window."""
    sr = np.asarray(fits.getdata(op.join(path, 'sr_vis.fits')), dtype=float).ravel()
    i0, w = int(round(START_TIME * fs)), int(round(RUNAWAY_WINDOW * fs))
    return float(np.mean(sr[i0:i0 + w])), float(np.mean(sr[-w:]))


def measured_flux(path, band, fs, steps=1):
    """Mean pupil flux per frame after START_TIME. A sensor read every `steps` loop steps stores one sample per frame (zeros before its first
    frame excluded)."""
    f = np.asarray(fits.getdata(op.join(path, f'flux_{band}.fits')), dtype=float).ravel()[int(round(START_TIME * fs / steps)):]
    return float(np.mean(f[f != 0]))


def ran_away(r):
    a, b = r['sr_start'], r['sr_end']
    return not (np.isfinite(a) and np.isfinite(b)) or b < RUNAWAY * a


def wind_realizations(n_real, seed0):
    """(atmo seed, wind directions [deg]) of each realization: the same for every case. Seeds 100 apart, as AtmoEvolution uses seed + layer."""
    with open(op.join(E2E_DIR, 'ristretto_main.yml')) as f:
        n_layers = len(yaml.safe_load(f)['wind_direction']['constant'])
    rng = np.random.default_rng(seed0)
    return [(100 * (k + 1) + 1, [round(float(x), 2) for x in rng.uniform(0.0, 360.0, n_layers)]) for k in range(n_real)]


def replay_overrides(ov, total_time, seed, directions):
    """Overrides of a selected run for a replay: other atmosphere seed and wind directions, longer run, no atmo or DM modes saved."""
    ov = yaml.safe_load(yaml.safe_dump(ov))
    ov['main_override']['total_time'] = total_time
    ov['atmo_override'] = {'seed': int(seed)}
    ov['wind_direction_override'] = {'constant': list(directions)}
    lst = ov['data_store_override']['inputs']['input_list']
    ov['data_store_override']['inputs']['input_list'] = [x for x in lst if x not in STORE_ALL]
    return ov


class Sweep:
    def __init__(self, out, gpu, total_time, rmod=None, mag_start=None, check=True, ladder=False):
        self.out, self.gpu, self.total_time, self.rmod, self.mag_start = out, gpu, total_time, rmod, mag_start
        self.check = check and mag_start is None
        self.ladder = LADDER[LADDER.index((N_SUBAPS, N_MODES)):] if ladder else ((N_SUBAPS, N_MODES),)
        self.tag = op.basename(op.normpath(out))
        self.store = op.join(out, 'runs')
        self.csv = op.join(out, 'results.csv')
        for d in (self.store, op.join(out, 'logs'), op.join(out, 'overrides')):
            os.makedirs(d, exist_ok=True)
        self.rows = {}
        if op.exists(self.csv):
            with open(self.csv) as f:
                header = f.readline().strip().split(',')
            if list(header) != list(FIELDS):     # written by an older version: rewrite with the columns of this one, or new rows would be misaligned
                old = read_rows(out)
                with open(self.csv, 'w', newline='') as f:
                    w = csv.DictWriter(f, FIELDS, extrasaction='ignore', restval='')
                    w.writeheader()
                    w.writerows(old)
            for r in read_rows(out):
                self.rows[self.key(r['rMod'], r['mag'], r['fs'], int(r['n_subaps']), r['kf_model'], r['flux_corr'])] = r
            print(f'{len(self.rows)} runs read from {self.csv}', flush=True)

    @staticmethod
    def key(rmod, mag, fs, n, kf, fc=1.0):
        return (float(rmod), float(mag), float(fs), int(n), kf, round(float(fc), 3))

    def unity_tag(self, N):
        """Identity IIR of this sweep: another process rewriting a shared one mid-sweep changes every later run."""
        return f'{self.tag}_unity_{N}modes'

    def build_filter(self, rmod, mag, fs, n, N, kf, fc=1.0):
        """Kalman data of the case: one sensor, one sample per frame, so the aliasing is the snapshot one. kf: 'ml' or 'svd' noise/aliasing model;
        fc: measured / model flux. Rebuilt for every (rate, flux): AR(2) at fs, delay 1 + tau, noise at the flux per frame of fs."""
        tag = f'{self.tag}_kf_r{rmod:1.1f}_m{mag:g}_fs{fs:g}_n{n}_{kf}' + (f'_f{fc:.3f}' if fc != 1.0 else '')
        a1, a2, q = ar2_at(fs)
        args = (rmod, n, N, WVL)
        flux = flux_per_frame(rmod, mag, fs, n, N) * fc
        if kf == 'ml':
            rec = ml_rec(rmod, mag, fs, n, N, fc)[1]
            sensor = dict(noise_var=noise_variance(rec, rmod, flux, n, N), aliasing=ml_aliasing(rec, rmod, n, N), frame_steps=1, biased=False)
        else:
            sensor = dict(noise_var=get_noise_propagation(*args, SIGMA_E, flux, pupil_flux=True) ** 2,
                          aliasing=get_aliasing_power(*args), frame_steps=1, biased=False)
        save_kalman_data(tag, calib_dir, a1, a2, q, [sensor], n_modes=N, delay=1 + DELAY * fs)
        path = op.join(calib_dir, 'filter', self.unity_tag(N) + '.fits')
        if not op.exists(path):
            save_plant_inverse_iir(self.unity_tag(N), calib_dir, N, beta=0.0)
        return tag

    def overrides(self, rmod, mag, fs, n, N, kf_tag, fc=1.0):
        dt, b = 1 / fs, band()
        other = next(x for x in BANDS.values() if x != b)
        src = f'source_{b}'
        if rec_of(rmod) == 'ml':
            rec = ml_rec(rmod, mag, fs, n, N, fc)[0]
        else:
            get_rec(rmod, n, N, WVL)   # computed and saved if missing
            rec = op.basename(paths(WVL).rec(rmod, n, N))[:-5]
        return {
            'remove': [f'source_{other}', f'pyr_{other}', f'cred_{other}', f'pyr_slopes_{other}', f'pyr_modes_{other}'],
            'main_override': {'time_step': dt, 'total_time': self.total_time, 'root_dir': calib_dir},
            'seeing_override': {'constant': sim_seeing()},
            'prop_override': {'source_dict_ref': [src, 'science_vis', 'science_nir'], 'enable_chromatic_effect': CHROMATIC,
                              'outputs': [f'out_{src}_ef', 'out_science_vis_ef', 'out_science_nir_ef']},
            f'source_{b}_override': {'magnitude': SOURCE_MAG_CAL[WVL] + mag - MAG_CAL, 'wavelengthInNm': WVL},
            f'pyr_{b}_override': {'pup_diam': n, 'pup_dist': calibration.E2EConfig().pup_dist(n), 'mod_amp': rmod, 'wavelengthInNm': WVL},
            f'cred_{b}_override': {'dt': dt, 'readout_level': SIGMA_E},
            f'pyr_slopes_{b}_override': {'pupdata_object': f'pyr_pupdata_{n}x{n}', 'sn_object': sn_tag(rmod, n, N),
                                         'outputs': ['out_slopes', 'out_flux']},
            f'pyr_modes_{b}_override': {'recmat_object': rec},
            'kalman_override': {'kalman_filter_data_object': kf_tag, 'time_step': dt, 'gain_period': 1,
                                'inputs': {'in_measurements_list': [f'pyr_modes_{b}.out_modes'], 'in_command': 'kpred.out_comm:-1'},
                                'outputs': ['out_projected_state', 'out_state']},   # no biased sensor: no out_bias
            'kpred_override': {'delay': DELAY * fs, 'iir_filter_data_object': self.unity_tag(N)},
            'coro_psf_vis_override': {'verbose': False},
            'data_store_override': {'store_dir': self.store, 'inputs': {'input_list': [
                'sr_vis-coro_psf_vis.out_sr', 'coro_psf_vis_int-coro_psf_vis.out_int_coronagraph_psf_profile',
                'sr_nir-coro_psf_nir.out_sr', 'coro_psf_nir_int-coro_psf_nir.out_int_coronagraph_psf_profile',
                'res_modes-res_modes.out_modes', *STORE_ALL, f'flux_{b}-pyr_slopes_{b}.out_flux']}},
        }

    def simulate(self, ov, name, store):
        """One SPECULA run; returns its TN. A crash is not a bad configuration: it stops the sweep (CalledProcessError), resumed by starting again."""
        ov_stem = op.join(self.out, 'overrides', name)
        with open(ov_stem + '.yml', 'w') as f:
            yaml.safe_dump(ov, f, sort_keys=False)
        before = set(os.listdir(store))
        run_specula('ristretto_main.yml', ov_stem, log_file=op.join(self.out, 'logs', name + '.log'), extra_args=('--target', str(self.gpu)))
        new = sorted(set(os.listdir(store)) - before)
        return new[-1] if new else latest_tn(store)

    def run_one(self, phase, rmod, mag, fs, n, N, kf, fc=1.0):
        key = self.key(rmod, mag, fs, n, kf, fc)
        if key in self.rows:
            return self.rows[key]
        name = f'r{rmod:1.1f}_m{mag:g}_fs{fs:g}_n{n}_{kf}' + (f'_f{fc:.3f}' if fc != 1.0 else '')
        ov = self.overrides(rmod, mag, fs, n, N, self.build_filter(rmod, mag, fs, n, N, kf, fc), fc)
        tn = self.simulate(ov, name, self.store)
        path = op.join(self.store, tn)
        cv, pv, sv = metrics(path, 'vis', fs)
        cn, _, sn = metrics(path, 'nir', fs)
        a, e = sr_drift(path, fs)
        notes = []
        pred = flux_per_frame(rmod, mag, fs, n, N)
        ratio = measured_flux(path, band(), fs) / pred      # against the uncorrected model
        if abs(ratio / fc - 1) > FLUX_TOL:
            notes.append(f'flux {ratio / fc:.2f} x model')
        if fc != 1.0:
            notes.append(f'flux model x {fc:.3f} (measured)')
        row = dict(phase=phase, rMod=rmod, mag=mag, fs=fs, n_subaps=n, n_modes=N, kf_model=kf, flux_corr=fc, tau=DELAY * fs, flux_pred=pred, flux_ratio=ratio,
                   contrast_vis=cv, peak_vis=pv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, sr_start=a, sr_end=e, tn=tn)
        if not all(np.isfinite(row[f'{x}_{METRIC_BAND}']) for x in ('contrast', 'sr')):
            notes.append('diverged')
            row[f'contrast_{METRIC_BAND}'] = np.inf
        row['notes'] = '; '.join(notes)
        self.rows[key] = row
        new = not op.exists(self.csv)
        with open(self.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{phase:9s} rMod {rmod:.1f} mag {mag:g} fs {fs:6.0f} {n}x{n} {N} modes kf {kf}: contrast {METRIC_BAND} "
              f"{row[f'contrast_{METRIC_BAND}']:.3e}  SR {row[f'sr_{METRIC_BAND}']:.4f}  {row['notes'] and 'WARNING ' + row['notes']}", flush=True)
        return row

    def evaluate(self, phase, rmod, mag, fs, i):
        """Run of (rmod, mag, fs, rung i); at rMod 0 with the ML reconstructor a runaway is repeated with the SVD filter model."""
        n, N = self.ladder[i]
        ml = rec_of(rmod) == 'ml'
        row = self.run_one(phase, rmod, mag, fs, n, N, 'ml' if ml else 'svd')
        if ml and rmod == 0.0 and ran_away(row):
            alt = self.run_one(phase, rmod, mag, fs, n, N, 'svd')
            text = f"full ML ran away (SR {row['sr_start']:.3f} -> {row['sr_end']:.3f}): SVD filter model"
            if text not in alt['notes']:
                alt['notes'] = '; '.join(x for x in (alt['notes'], text) if x)
            row = min((row, alt), key=score)
        if float(row['flux_corr']) == 1.0 and not ran_away(row):
            ratio = measured_flux(op.join(self.store, row['tn']), band(), fs) / flux_per_frame(rmod, mag, fs, n, N)
            if np.isfinite(ratio) and abs(ratio - 1) > FLUX_ADJUST:
                row = self.run_one(phase, rmod, mag, fs, n, N, row['kf_model'], round(ratio, 3))   # replaces the uncorrected run
        return row

    def note_change(self, row, best):
        change = rel_change(row, best)
        if change < SMALL_CHANGE:
            text = f'change {change:.2%} against {best["fs"]:.0f} Hz {int(best["n_subaps"])}x: may be noise'
            if text not in row['notes']:
                row['notes'] = '; '.join(x for x in (row['notes'], text) if x)

    def frequency(self, best, rmod, mag, i, up):
        """Down by FS_STEP while the metric improves; with up (after a change of rung), up instead if the first step down does not improve."""
        for step in ((-FS_STEP, FS_STEP) if up else (-FS_STEP,)):
            f, moved = best['fs'] + step, False
            while FS_STEP <= f <= FS_START:
                row = self.evaluate('frequency', rmod, mag, f, i)
                self.note_change(row, best)
                if not better(row, best):
                    break
                best, moved = row, True
                f += step
            if moved:
                break
        return best

    def search(self, rmod, mag, fs, i):
        best = self.evaluate('magnitude', rmod, mag, fs, i)
        up = False
        while True:
            best = self.frequency(best, rmod, mag, i, up)
            moved = False
            while i + 1 < len(self.ladder):
                row = self.evaluate('ladder', rmod, mag, best['fs'], i + 1)
                self.note_change(row, best)
                if not better(row, best):
                    break
                best, i, moved = row, i + 1, True
            if not moved:
                return best, i
            up = True

    def run(self):
        check_ar2()
        fs, i = FS_START, 0
        selected = []
        if self.mag_start is None:
            rows = [self.evaluate('rMod', r, MAG_START, FS_START, 0) for r in (RMODS if self.rmod is None else (self.rmod,))]
            rmod = min(rows, key=score)['rMod']
            best, i = self.search(rmod, MAG_START, fs, 0)
            selected, mags = [best], MAGS
            fs = best['fs']
        else:
            rmod, mags = self.rmod, [m for m in MAGS if m >= self.mag_start]
        print(f'WFS {WVL:g} nm, metric {METRIC_BAND}, ladder {self.ladder}, modulation {rmod}, magnitudes {list(mags)}', flush=True)
        for mag in mags:
            best, i = self.search(rmod, mag, fs, i)
            fs = best['fs']
            if not np.isfinite(best[f'contrast_{METRIC_BAND}']):
                best['notes'] = '; '.join(x for x in (best['notes'], 'no converged configuration found') if x)
            selected.append(best)
        if self.check:
            n = self.ladder[i][0]
            checks = [self.evaluate('check', r, MAGS[-1], fs, i) for r in RMODS if r != rmod]
            if any(better(c, selected[-1]) for c in checks):
                print(f'WARNING: another modulation beats rMod {rmod} at mag {MAGS[-1]:g}, {fs:.0f} Hz, {n}x{n}', flush=True)
        self.report(selected)

    def report(self, selected):
        def line(cells):
            return '| ' + ' | '.join(str(c) for c in cells) + ' |'
        def values(r):
            return [r['rMod'], f"{r['mag']:g}", f"{r['fs']:.0f}", int(r['n_subaps']), int(r['n_modes']), r['kf_model'], f"{r['contrast_vis']:.3e}",
                    f"{r['sr_vis']:.4f}", f"{r['contrast_nir']:.3e}", f"{r['sr_nir']:.4f}"]
        head = ['rMod', 'mag', 'fs [Hz]', 'subaps', 'modes', 'KF model', 'contrast 550 nm', 'SR 550 nm', 'contrast 1650 nm', 'SR 1650 nm']
        out = [f'## Selected configuration per magnitude (WFS {WVL:g} nm, metric {METRIC_BAND}, chromatic {CHROMATIC})', '',
               line(head + ['notes', 'TN']), line(['---'] * (len(head) + 2))]
        out += [line(values(r) + [r['notes'], r['tn']]) for r in selected]
        full = ['phase', *head, 'tau', 'flux/frame [e-]', 'flux meas/model', 'SR start', 'SR end', 'selected', 'notes', 'TN']
        out += ['', '## All runs', '', line(full), line(['---'] * len(full))]
        sel = {id(r) for r in selected}
        for r in self.rows.values():
            out.append(line([r['phase'], *values(r), f"{r['tau']:.3f}", f"{r['flux_pred']:.3g}", f"{r['flux_ratio']:.3f}", f"{r['sr_start']:.4f}",
                             f"{r['sr_end']:.4f}", 'yes' if id(r) in sel else '', r['notes'], r['tn']]))
        text = '\n'.join(out) + '\n'
        with open(op.join(self.out, 'table.md'), 'w') as f:
            f.write(text)
        write_selected(self.out, [dict(mag=r['mag'], config=f"rMod {r['rMod']:g}, {int(r['n_subaps'])}x{int(r['n_subaps'])}, {int(r['n_modes'])} modes, "
                                                       f"{r['fs']:.0f} Hz, KF {r['kf_model']}" + (f", flux x{float(r['flux_corr']):.3f}" if float(r['flux_corr']) != 1.0 else ''),
                                       key=list(self.key(r['rMod'], r['mag'], r['fs'], int(r['n_subaps']), r['kf_model'], r['flux_corr'])),
                                       **{k: r[k] for k in SELECTED_FIELDS[3:]})
                                  for r in selected])
        print(text, flush=True)

    def replay(self, n_real, seed0, total_time, mags=None):
        """The selected configurations again with other atmosphere seeds and wind directions (wind_realizations), total_time [s] each.
        Rows in replays.csv; finished realizations are skipped."""
        check_ar2()
        store = op.join(self.out, 'replays')
        os.makedirs(store, exist_ok=True)
        done = {(r['mag'], int(r['realization'])) for r in read_rows(self.out, 'replays.csv')} if op.exists(op.join(self.out, 'replays.csv')) else set()
        for s in read_selected(self.out):
            rmod, mag, fs, n, kf, fc = (list(s['key']) + [1.0])[:6]
            if mags and mag not in mags:
                continue
            N = dict(LADDER).get(int(n), N_MODES)
            for k, (seed, dirs) in enumerate(wind_realizations(n_real, seed0)):
                if (mag, k) in done:
                    continue
                ov = self.overrides(rmod, mag, fs, int(n), N, self.build_filter(rmod, mag, fs, int(n), N, kf, fc), fc)
                ov = replay_overrides(ov, total_time, seed, dirs)
                ov['data_store_override']['store_dir'] = store
                tn = self.simulate(ov, f'replay_m{mag:g}_k{k}', store)
                path = op.join(store, tn)
                cv, _, sv = metrics(path, 'vis', fs)
                cn, _, sn = metrics(path, 'nir', fs)
                ratio = measured_flux(path, band(), fs) / flux_per_frame(rmod, mag, fs, int(n), N)
                append_replay(self.out, dict(mag=mag, config=s['config'], realization=k, seed=seed, wind_direction=' '.join(f'{d:g}' for d in dirs),
                                             contrast_vis=cv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, flux_ratio=ratio,
                                             notes='' if np.isfinite(cv) and np.isfinite(sv) else 'diverged', tn=tn))
                print(f'replay mag {mag:g} realization {k} seed {seed}: contrast vis {cv:.3e} nir {cn:.3e}  SR vis {sv:.4f} nir {sn:.4f}', flush=True)


SELECTED_FIELDS = ('mag', 'config', 'key', 'contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'notes', 'tn')


def write_selected(out, rows):
    """selected.csv: the configuration chosen at each magnitude; 'key' is the argument list replay uses (JSON-like)."""
    with open(op.join(out, 'selected.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, SELECTED_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, 'key': yaml.safe_dump(r['key'], default_flow_style=True).strip()})


def read_selected(out):
    with open(op.join(out, 'selected.csv')) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['key'] = yaml.safe_load(r['key'])
        for k in ('mag', 'contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir'):
            r[k] = float(r[k])
    return rows


def append_replay(out, row):
    path = op.join(out, 'replays.csv')
    new = not op.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, REPLAY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def final_table(sweeps):
    """Table of several sweeps, each 'n_subaps:n_modes:folder': the best frequency of each magnitude, then all the runs.
    Average and peak contrast are read again from the saved profiles, so sweeps saved without peak_vis are covered."""
    def line(cells):
        return '| ' + ' | '.join(str(c) for c in cells) + ' |'
    head = ['subaps', 'modes', 'rMod', 'mag', 'fs [Hz]', 'TN', 'avg contrast', 'peak contrast', 'SR 550 nm']
    best, runs = [], []
    for sweep in sweeps:
        n, modes, out = sweep.split(':', 2)
        rows = read_rows(out)
        for r in rows:
            cv, pv, sv = metrics(op.join(out, 'runs', r['tn']), 'vis', r['fs'])
            r['cells'] = [n, modes, r['rMod'], f"{r['mag']:g}", f"{r['fs']:.0f}", r['tn'], f'{cv:.3e}', f'{pv:.3e}', f'{sv:.4f}']
            runs.append(line(r['cells'] + [r['phase'], r['notes']]))
        search = [r for r in rows if r['phase'] != 'check']
        rmod = search[-1]['rMod']       # the fixed or selected modulation: the last run of the search
        for mag in sorted({r['mag'] for r in search if r['rMod'] == rmod}):
            best.append(line(min((r for r in search if r['rMod'] == rmod and r['mag'] == mag), key=score)['cells']))
    text = '\n'.join(['## Best frequency per magnitude', '', line(head), line(['---'] * len(head)), *best, '',
                      '## All runs', '', line(head + ['phase', 'notes']), line(['---'] * (len(head) + 2)), *runs]) + '\n'
    return text


def main():
    global N_SUBAPS, N_MODES, REC, WVL, METRIC_BAND, CHROMATIC
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rec', choices=('svd', 'ml', 'auto'), default=REC, help='reconstructor; ml: at the flux of each run; auto: ml at rMod 0, svd otherwise')
    ap.add_argument('--wvl', type=float, default=WVL, choices=sorted(BANDS), help='[nm] sensing wavelength; the metric is in the nearest science band')
    ap.add_argument('--achromatic', action='store_true', help='prop.enable_chromatic_effect = false')
    ap.add_argument('--ladder', action='store_true', help='search the (subapertures, modes) LADDER too, from --n-subaps/--n-modes')
    ap.add_argument('--n-subaps', type=int, default=N_SUBAPS)
    ap.add_argument('--n-modes', type=int, default=N_MODES)
    ap.add_argument('--start-from', help='earlier sweep folder: search only from its start_magnitude, 2 kHz, needs --rmod, no modulation check')
    ap.add_argument('--final-table', nargs='+', metavar='N:MODES:FOLDER', help='print the table of these sweeps and exit')
    ap.add_argument('--no-check', action='store_true', help='skip the modulation check at the faintest magnitude')
    ap.add_argument('--replay', type=int, metavar='N_REAL', help='rerun the selected configurations N_REAL times with other seeds and wind directions')
    ap.add_argument('--replay-mags', type=float, nargs='+', help='replay these magnitudes only')
    ap.add_argument('--replay-seed', type=int, default=20261108, help='seed of the random wind directions of the replays')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--total-time', type=float, default=3.0, help='[s] of each run (of each replay with --replay)')
    ap.add_argument('--out', default=op.join(result_dir, 'sweep_contrast_20261106'))
    ap.add_argument('--rmod', type=float, help='fixed modulation: skip the modulation selection')
    a = ap.parse_args()
    if a.final_table:
        print(final_table(a.final_table))
        return
    if a.start_from and a.rmod is None:
        ap.error('--start-from needs --rmod')
    N_SUBAPS, N_MODES, REC, WVL, CHROMATIC = a.n_subaps, a.n_modes, a.rec, a.wvl, not a.achromatic
    METRIC_BAND = BANDS[WVL]
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']   # the specula CLI of this environment
    sweep = Sweep(a.out, a.gpu, a.total_time, a.rmod, start_magnitude(a.start_from) if a.start_from else None, not a.no_check, a.ladder)
    if a.replay:
        sweep.replay(a.replay, a.replay_seed, a.total_time, a.replay_mags)
    else:
        sweep.run()


if __name__ == '__main__':
    main()
