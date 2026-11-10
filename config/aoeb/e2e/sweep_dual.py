"""Greedy search of the dual-WFS configuration per magnitude. Metric as in sweep_contrast.py: mean raw contrast in 2-5 lambda/D at 550 nm, or SR when
SR < SR_SWITCH; peak contrast saved too.

Modulated pyramid, reference of the filter: rMod 2, 48x48, 1300 modes, SVD, frame of k loop steps (2000 / k Hz).
Unmodulated pyramid: rMod 0, 2 kHz, ML reconstructor at the flux of the run, on the LADDER of (subapertures, modes). Modes it does not reconstruct
are padded: zero rows in its reconstructor, very large noise in the filter, so only the modulated sensor drives them.
From mag SLOW_FROM on, after the search, also the slower rates FS / K_UNMOD at the best (k, rung), the next magnitude starting its search from that
(k, rung) at 2 kHz. At a slower rate the whole loop runs at that rate, as in sweep_contrast.py (time step, unmodulated frame 1 / fs, delay
DELAY * fs, AR(2) at fs, horizon 1 + delay), and the modulated frame keeps its duration as well as a whole number of steps allows.
Both: slope nulls, no bias. Frames, nulls and aliasing of every sensor are those of N_CORR corrected modes.

source_vis / pyr_vis stay at 900 nm and source_nir / pyr_nir at 1300 nm, the star colour is fixed (MAG_CAL): --unmod-wvl chooses the unmodulated one.
Equal flux (--same-wvl W --unmod-frac f): both pyramids at W, the unmodulated on the vis chain with f of the BANDW band, the modulated on the nir
chain with 1 - f (camera bandw and flux model scaled). The metric is in the science band nearest to the unmodulated sensor (900 -> 550, 1300 -> 1650).
Flux: a run whose measured pupil flux (after START_TIME) differs from the model by more than sweep_contrast.FLUX_ADJUST on a sensor is repeated
once with that sensor's model flux (ML weights, KF noise) x measured / model; the corrected run replaces it.
`mag` is the 900 nm magnitude. Search at each magnitude, from the best (k, n) of the previous one, never going back: k + 1 until the metric
stops improving, then the next rung of the LADDER until it stops improving; if the rung changed, k again.

Every file written has its own tag (sweep_dual_*). Finished runs are read back from results.csv, so an interrupted sweep is resumed by starting it again.

    python sweep_dual.py --calibrate --gpu 0
    python sweep_dual.py --unmod-wvl 900 --gpu 0 --out <folder>
"""
import argparse
import csv
import os
import os.path as op
import sys

import numpy as np
import yaml
from astropy.io import fits

import sweep_contrast as sc                                                                                # also specula.init(-1)
from sweep_contrast import (DELAY, FLUX_ADJUST, SIGMA_E, SMALL_CHANGE, STORE_ALL, append_replay, ar2_at, atomic_write, better, check_ar2,  # noqa: E402
                            measured_flux, metrics, read_rows, read_selected, rel_change, replay_overrides, sim_seeing, wind_realizations, write_selected)
from specula.data_objects.kalman_filter_data import ar2_stationary_variance                                # noqa: E402
from specula.data_objects.slopes import Slopes                                                             # noqa: E402

import calibration                                                                                         # noqa: E402
from aoeb.filter_optimizers import aliasing_frame_fraction                                                 # noqa: E402
from aoeb.noise import slope_noise_cov                                                                     # noqa: E402
from aoeb.reconstructors import compute_reconstructor, save_rec                                            # noqa: E402
from save_kalman_data import save_kalman_data, save_plant_inverse_iir                                      # noqa: E402
from utils import (SEEING, _load_calib, calib_dir, get_aliasing_power, get_pupil_flux, get_rec, get_static_slopes, latest_tn, machine,  # noqa: E402
                   noise_sensitivities, paths, propagated_noise_var, result_dir, run_specula)

N_SUBAPS_MOD, N_CORR, RMOD_MOD, RMOD_UNMOD = 48, 1300, 2.0, 0.0
LADDER = ((48, 1300), (36, 900), (24, 300), (14, 100))     # unmodulated pyramid
FS = 2000.0
K_MAX = 20                              # slowest modulated frame: 100 Hz
K_UNMOD, SLOW_FROM = (2, 4), 12.0       # slower rates of the loop and unmodulated sensor (1 kHz, 500 Hz), tried from this magnitude
MAGS = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)   # 13 and 14 dropped on 2026-11-08
MAG_CAL = {900.0: 7.0, 1300.0: 6.43}    # source_vis, source_nir of ristretto_main.yml: the magnitudes of the calibration frames
BANDS = {900.0: 'vis', 1300.0: 'nir'}   # sensing wavelength -> nearest science band (and chain of the unequal dual)
CHAINS = ('vis', 'nir')                 # yml chains source_/pyr_/cred_/pyr_slopes_/pyr_modes_<chain>
ALIAS_SEEINGS = (0.75, 1.05)            # 48x48: the s0.9 frames were renamed from the old e2e and have no slopes file
R_PAD = 1e6                             # noise of a padded mode / its turbulence variance: gain below 1e-6
FLUX_TOL = 0.10
BANDW = 400.0                           # [nm] cred_vis / cred_nir bandw: split between the two sensors with --same-wvl
FIELDS = ('phase', 'mag', 'k', 'fs_mod', 'k_unmod', 'fs_unmod', 'n_unmod', 'N_unmod', 'fc_vis', 'fc_nir', 'contrast_vis', 'peak_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'flux_ratio_vis', 'flux_ratio_nir', 'notes', 'tn')


def frame_alias_fraction(wvl, steps, fs):
    """Aliasing of a frame of `steps` loop steps against one step (the run has one sample per step: 1 at steps = 1)."""
    spectrum = machine().alias_spectrum(wvl)
    return aliasing_frame_fraction(fs / steps, spectrum) / aliasing_frame_fraction(fs, spectrum)


def ml_rec(n, N, wvl, nph):
    """Stem and first N rows of the ML reconstructor at nph frame electrons, noise weights from the frame with N_CORR modes corrected.
    Saved with 1300 rows (zero rows for the modes it does not reconstruct)."""
    stem = f'sweep_dual_pyr{RMOD_UNMOD:1.1f}_{n}x{n}_wvl{wvl:g}_{N}modes_corr{N_CORR}_rec_ml_{nph:.4g}ph_s{SEEING}'
    path = op.join(calib_dir, 'rec', stem + '.fits')
    if not op.exists(path):
        m, P = machine(), paths(wvl)
        frame = fits.getdata(P.avgframe(RMOD_UNMOD, n, SEEING, N_CORR))
        cn, cxy = slope_noise_cov(frame, m.slope_geometry(RMOD_UNMOD, n, SEEING, N_CORR, wvl, frame), nph, m.detector, m.slopes_from_intensity)
        rec = compute_reconstructor(fits.getdata(P.im_dl(RMOD_UNMOD, n)), N, 'ml', cn_diag=cn, cn_xy=cxy)
        atomic_write(path, lambda p: save_rec(p, np.pad(rec, ((0, N_CORR - N), (0, 0))), method='ml', n_phot=nph, seeing=SEEING, wvl_nm=wvl))
    return stem, np.asarray(fits.getdata(path))[:N]


def ml_aliasing(rec, n, N, wvl):
    """Aliasing variance of the ML reconstructor: from the slopes at the seeing of the run or, if there are none, the SVD aliasing times the
    ML / SVD ratio of each mode on the slopes of the nearby seeings."""
    P = paths(wvl)
    if op.exists(P.slopes(RMOD_UNMOD, n, SEEING, N_CORR)):
        return np.var(rec @ fits.getdata(P.slopes(RMOD_UNMOD, n, SEEING, N_CORR)).astype(float).T, axis=1)
    svd = get_rec(RMOD_UNMOD, n, N, wvl)
    ratios = []
    for s in ALIAS_SEEINGS:
        slopes = fits.getdata(P.slopes(RMOD_UNMOD, n, s, N_CORR)).astype(float)
        ratios.append(np.var(rec @ slopes.T, axis=1) / np.var(svd @ slopes.T, axis=1))
    return get_aliasing_power(RMOD_UNMOD, n, N, wvl, nCorrModes=N_CORR) * np.mean(ratios, axis=0)


def noise_var(rec, rmod, n, N, wvl, flux):
    """Per-mode noise variance of a reconstructor (utils.get_noise_propagation for any rec). flux: on the four pupils, per frame."""
    frame, pup, _ = _load_calib(rmod, n, N, wvl, N_CORR)
    total = flux * np.sum(frame) / np.sum(frame.ravel()[pup[pup >= 0]])     # pupil flux -> frame flux, the convention of noise_sensitivities
    s_phot, s_read = noise_sensitivities(rec, frame, pup)
    phot, read = propagated_noise_var(s_phot, s_read, flux=total, sigma_e=SIGMA_E)
    return phot + read


def sn_tag(rmod, n, N, wvl):
    """Slope nulls of one sensor; the file depends on the wavelength and the corrected modes only, so it is written once."""
    tag = f'sweep_dual_pyr{rmod:1.1f}_{n}x{n}_wvl{wvl:g}_corr{N_CORR}_sn'
    path = op.join(calib_dir, 'slopenulls', tag + '.fits')
    if not op.exists(path):
        atomic_write(path, lambda p: Slopes(slopes=get_static_slopes(rmod, n, N, wvl, nCorrModes=N_CORR)).save(p))
    return tag


class Sweep:
    def __init__(self, out, gpu, total_time, unmod_wvl, chromatic=True, same_wvl=None, unmod_frac=1.0, bands=None):
        """bands: (unmod nm, unmod bandw nm, mod nm, mod bandw nm): each pyramid at its own wavelength and camera band (unmodulated on the vis
        chain, products calibrated on that chain: MAG_CAL of the chain), the star magnitude at that wavelength from the colour of MAG_CAL,
        linear in the wavelength, as the 1300 nm source of the other cases."""
        self.out, self.gpu, self.total_time, self.chromatic = out, gpu, total_time, chromatic
        colour = MAG_CAL[900.0] - MAG_CAL[1300.0]
        if bands:
            uw, ubw, mw, mbw = (float(x) for x in bands)
            self.wvl, self.unmod_chain = {'vis': uw, 'nir': mw}, 'vis'
            self.frac = {'vis': ubw / BANDW, 'nir': mbw / BANDW}
            MAG_CAL.setdefault(uw, MAG_CAL[900.0])          # calibration.py: vis chain at uw, nir chain at mw, sources of ristretto_main.yml
            MAG_CAL.setdefault(mw, MAG_CAL[1300.0])
            self.mag_offset = {c: -colour * (w - 900.0) / 400.0 for c, w in self.wvl.items()}
        elif same_wvl:      # both sensors at one wavelength, band split: unmodulated on the vis chain
            self.wvl, self.unmod_chain = {'vis': float(same_wvl), 'nir': float(same_wvl)}, 'vis'
            self.frac = {'vis': float(unmod_frac), 'nir': 1.0 - float(unmod_frac)}
        else:
            self.wvl, self.unmod_chain = {b: w for w, b in BANDS.items()}, BANDS[unmod_wvl]
            self.frac = {'vis': 1.0, 'nir': 1.0}
        if not bands:                                       # source magnitude of a chain = 900 nm magnitude + offset
            self.mag_offset = {c: MAG_CAL[w] - MAG_CAL[900.0] for c, w in self.wvl.items()}
        self.mod_chain = next(c for c in CHAINS if c != self.unmod_chain)
        self.unmod_wvl, self.mod_wvl = self.wvl[self.unmod_chain], self.wvl[self.mod_chain]
        sc.METRIC_BAND = BANDS.get(self.unmod_wvl, 'vis')  # metric in the science band nearest to the unmodulated sensor (bands: 550 nm)
        self.tag = op.basename(op.normpath(out))
        self.store = op.join(out, 'runs')
        self.csv = op.join(out, 'results.csv')
        self.unity = f'{self.tag}_unity_{N_CORR}modes'   # of this sweep: another process rewriting a shared one changes every later run
        for d in (self.store, op.join(out, 'logs'), op.join(out, 'overrides')):
            os.makedirs(d, exist_ok=True)
        self.rows = {}
        if op.exists(self.csv):
            with open(self.csv) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                r.setdefault('fc_vis', 1.0)
                r.setdefault('fc_nir', 1.0)
                for k in FIELDS:
                    if k not in ('phase', 'notes', 'tn'):
                        r[k] = float(r[k])
            with open(self.csv) as f:
                header = f.readline().strip().split(',')
            if header != list(FIELDS):     # written by an older version: rewrite with the columns of this one
                with open(self.csv, 'w', newline='') as f:
                    w = csv.DictWriter(f, FIELDS, extrasaction='ignore')
                    w.writeheader()
                    w.writerows(rows)
            for r in rows:
                self.rows[self.key(r['mag'], r['k'], r['n_unmod'], r['k_unmod'], {'vis': r['fc_vis'], 'nir': r['fc_nir']})] = r
            print(f'{len(self.rows)} runs read from {self.csv}', flush=True)

    @staticmethod
    def key(mag, k, n, ku, fc):
        return (float(mag), int(k), int(n), int(ku), round(float(fc['vis']), 3), round(float(fc['nir']), 3))

    def model_flux(self, chain, rmod, n, N, mag, exposure):
        """Pupil flux of one frame [e-]: calibration frame flux scaled to the chain's source magnitude and camera band."""
        wvl = self.wvl[chain]
        return (get_pupil_flux(rmod, n, N, wvl, nCorrModes=N_CORR) * exposure * FS * 10 ** (-0.4 * (mag + self.mag_offset[chain] - MAG_CAL[wvl]))
                * self.frac[chain])

    def sensor(self, chain, mag, steps, n, N, fs, fc=1.0):
        """(overrides, sensor of the filter, model flux per frame) of the pyramid of a chain, frame of `steps` steps of the loop at fs;
        fc: measured / model flux, applied to the ML weights and the KF noise. Rebuilt for every (rate, frame, flux)."""
        wvl = self.wvl[chain]
        unmod = chain == self.unmod_chain
        rmod = RMOD_UNMOD if unmod else RMOD_MOD
        exposure = steps / fs
        flux = self.model_flux(chain, rmod, n, N, mag, exposure)
        if unmod:
            P = paths(wvl)
            path = P.avgframe(rmod, n, SEEING, N_CORR)
            frame_flux = (float(np.sum(fits.getdata(path))) / calibration.avgframe_steps(path) * exposure * FS
                          * 10 ** (-0.4 * (mag + self.mag_offset[chain] - MAG_CAL[wvl]))
                          * self.frac[chain] * fc)
            rec_tag, rec = ml_rec(n, N, wvl, frame_flux)
            alias = ml_aliasing(rec, n, N, wvl) * frame_alias_fraction(wvl, steps, fs)
        else:
            rec = get_rec(rmod, n, N, wvl)
            rec_tag = op.basename(paths(wvl).rec(rmod, n, N))[:-5]
            alias = get_aliasing_power(rmod, n, N, wvl, nCorrModes=N_CORR) * frame_alias_fraction(wvl, steps, fs)
        a1, a2, q = ar2_at(fs)
        pad = R_PAD * ar2_stationary_variance(a1, a2, q)[N:N_CORR]
        sensor = dict(noise_var=np.concatenate([noise_var(rec, rmod, n, N, wvl, flux * fc), pad]), aliasing=np.concatenate([alias, np.zeros(N_CORR - N)]),
                      frame_steps=steps, biased=False)
        ov = {f'source_{chain}_override': {'magnitude': mag + self.mag_offset[chain], 'wavelengthInNm': wvl},
              f'pyr_{chain}_override': {'pup_diam': n, 'pup_dist': calibration.E2EConfig().pup_dist(n), 'mod_amp': rmod, 'wavelengthInNm': wvl},
              f'cred_{chain}_override': {'dt': exposure, 'readout_level': SIGMA_E, 'bandw': BANDW * self.frac[chain]},
              f'pyr_slopes_{chain}_override': {'pupdata_object': f'pyr_pupdata_{n}x{n}', 'sn_object': sn_tag(rmod, n, N, wvl), 'outputs': ['out_slopes', 'out_flux']},
              f'pyr_modes_{chain}_override': {'recmat_object': rec_tag}}
        return ov, sensor, flux

    def overrides(self, mag, k, i, ku, name, fc):
        """(overrides, model flux per frame of each chain) of one configuration; writes its Kalman data (AR(2) at fs, delay 1 + tau, noise at the flux)."""
        n, N = LADDER[i]
        fs = FS / ku
        ov, sensors, flux = {}, [], {}
        for chain in CHAINS:                    # order of kalman.in_measurements_list
            if chain == self.unmod_chain:
                o, s, flux[chain] = self.sensor(chain, mag, 1, n, N, fs, fc[chain])
            else:
                o, s, flux[chain] = self.sensor(chain, mag, k, N_SUBAPS_MOD, N_CORR, fs, fc[chain])
            ov.update(o)
            sensors.append(s)
        kf_tag = f'{self.tag}_kf_{name}'
        a1, a2, q = ar2_at(fs)
        save_kalman_data(kf_tag, calib_dir, a1, a2, q, sensors, n_modes=N_CORR, delay=1 + DELAY * fs)
        ov.update({
            'main_override': {'time_step': 1 / fs, 'total_time': self.total_time, 'root_dir': calib_dir},
            'seeing_override': {'constant': sim_seeing()},
            'prop_override': {'enable_chromatic_effect': self.chromatic},
            'kalman_override': {'kalman_filter_data_object': kf_tag, 'time_step': 1 / fs, 'gain_period': k,
                                'inputs': {'in_measurements_list': ['pyr_modes_vis.out_modes', 'pyr_modes_nir.out_modes'], 'in_command': 'kpred.out_comm:-1'},
                                'outputs': ['out_projected_state', 'out_state']},
            'kpred_override': {'delay': DELAY * fs, 'iir_filter_data_object': self.unity},
            'coro_psf_vis_override': {'verbose': False},
            'data_store_override': {'store_dir': self.store, 'inputs': {'input_list': [
                'sr_vis-coro_psf_vis.out_sr', 'coro_psf_vis_int-coro_psf_vis.out_int_coronagraph_psf_profile',
                'sr_nir-coro_psf_nir.out_sr', 'coro_psf_nir_int-coro_psf_nir.out_int_coronagraph_psf_profile',
                'res_modes-res_modes.out_modes', *STORE_ALL,
                'flux_vis-pyr_slopes_vis.out_flux', 'flux_nir-pyr_slopes_nir.out_flux']}}})
        return ov, flux

    def simulate(self, ov, name, store):
        """One SPECULA run; returns its TN. A crash is not a bad configuration: it stops the sweep (CalledProcessError), resumed by starting it again."""
        stem = op.join(self.out, 'overrides', name)
        with open(stem + '.yml', 'w') as f:
            yaml.safe_dump(ov, f, sort_keys=False)
        before = set(os.listdir(store))
        run_specula('ristretto_main.yml', stem, log_file=op.join(self.out, 'logs', name + '.log'), extra_args=('--target', str(self.gpu)))
        new = sorted(set(os.listdir(store)) - before)
        return new[-1] if new else latest_tn(store)

    @staticmethod
    def name(mag, k, n, ku, fc=None):
        corr = ''.join(f'_f{c}{fc[c]:.3f}' for c in CHAINS if fc and fc[c] != 1.0)
        return f'm{mag:g}_k{k}_n{n}' + (f'_ku{ku}' if ku > 1 else '') + corr

    def run_one(self, phase, mag, k, i, ku, fc):
        n, N = LADDER[i]
        key = self.key(mag, k, n, ku, fc)
        if key in self.rows:
            return self.rows[key]
        fs = FS / ku
        name = self.name(mag, k, n, ku, fc)
        ov, flux = self.overrides(mag, k, i, ku, name, fc)
        tn = self.simulate(ov, name, self.store)
        path = op.join(self.store, tn)
        cv, pv, sv = metrics(path, 'vis', fs)
        cn, _, sn = metrics(path, 'nir', fs)
        row = dict(phase=phase, mag=mag, k=k, fs_mod=fs / k, k_unmod=ku, fs_unmod=fs, n_unmod=n, N_unmod=N, fc_vis=fc['vis'], fc_nir=fc['nir'],
                   contrast_vis=cv, peak_vis=pv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, tn=tn)
        notes = []
        for chain in CHAINS:
            steps = 1 if chain == self.unmod_chain else k
            row[f'flux_ratio_{chain}'] = measured_flux(path, chain, fs, steps) / flux[chain]     # against the uncorrected model
            if abs(row[f'flux_ratio_{chain}'] / fc[chain] - 1) > FLUX_TOL:
                notes.append(f"flux {chain} {row[f'flux_ratio_{chain}'] / fc[chain]:.2f} x model")
            if fc[chain] != 1.0:
                notes.append(f'flux model {chain} x {fc[chain]:.3f} (measured)')
        mb = sc.METRIC_BAND
        if not np.isfinite(row[f'contrast_{mb}']) or not np.isfinite(row[f'sr_{mb}']):
            notes.append('diverged')
            row[f'contrast_{mb}'] = np.inf
        row['notes'] = '; '.join(notes)
        self.rows[key] = row
        new = not op.exists(self.csv)
        with open(self.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{phase:10s} mag {mag:g} modulated {fs / k:6.0f} Hz, unmodulated {fs:6.0f} Hz {n}x{n} {N} modes, flux x {fc['vis']:.3f}/{fc['nir']:.3f}: "
              f"contrast {mb} {row[f'contrast_{mb}']:.3e}  SR {row[f'sr_{mb}']:.4f}  {row['notes'] and 'WARNING ' + row['notes']}", flush=True)
        return row

    def evaluate(self, phase, mag, k, i, ku=1):
        """k: modulated frame in steps of the loop at FS / ku. A sensor whose measured flux is off by more than FLUX_ADJUST: the run is repeated once
        with its model flux corrected, and the corrected run is the result."""
        one = {c: 1.0 for c in CHAINS}
        row = self.run_one(phase, mag, k, i, ku, one)
        if not np.isfinite(row[f'contrast_{sc.METRIC_BAND}']):
            return row
        n, N = LADDER[i]
        path, fs = op.join(self.store, row['tn']), FS / ku
        fc = {}
        for chain in CHAINS:
            unmod = chain == self.unmod_chain
            model = self.model_flux(chain, RMOD_UNMOD if unmod else RMOD_MOD, n if unmod else N_SUBAPS_MOD, N if unmod else N_CORR, mag, (1 if unmod else k) / fs)
            ratio = measured_flux(path, chain, fs, 1 if unmod else k) / model
            fc[chain] = round(ratio, 3) if np.isfinite(ratio) and abs(ratio - 1) > FLUX_ADJUST else 1.0
        return row if fc == one else self.run_one(phase, mag, k, i, ku, fc)

    def note(self, row, against):
        change = rel_change(row, against)
        if change < SMALL_CHANGE:
            text = f'change {change:.2%}: may be noise'
            if text not in row['notes']:
                row['notes'] = '; '.join(x for x in (row['notes'], text) if x)

    def search(self, mag, k, i):
        """Best (row, k, ladder index) at this magnitude."""
        best = self.evaluate('magnitude', mag, k, i)
        while True:
            while k < K_MAX:
                row = self.evaluate('rate', mag, k + 1, i)
                self.note(row, best)
                if not better(row, best):
                    break
                best, k = row, k + 1
            moved = False
            while i + 1 < len(LADDER):
                row = self.evaluate('ladder', mag, k, i + 1)
                self.note(row, best)
                if not better(row, best):
                    break
                best, i, moved = row, i + 1, True
            if not moved:
                return best, k, i

    def run(self):
        check_ar2()
        save_plant_inverse_iir(self.unity, calib_dir, N_CORR, beta=0.0)
        k, i, selected = 1, 0, []
        for mag in MAGS:
            best, k, i = self.search(mag, k, i)
            if mag >= SLOW_FROM:
                for ku in K_UNMOD:
                    k_slow = max(1, round(k / ku))      # same modulated frame duration, as well as whole steps allow
                    row = self.evaluate('unmod rate', mag, k_slow, i, ku)
                    self.note(row, best)
                    if better(row, best):
                        best = row
            selected.append(best)
        self.report(selected)

    def describe(self, r):
        u, m = self.unmod_chain, self.mod_chain
        fc = f", flux x {r[f'fc_{u}']:.3f}/{r[f'fc_{m}']:.3f}" if r[f'fc_{u}'] != 1.0 or r[f'fc_{m}'] != 1.0 else ''
        return (f"unmod {self.unmod_wvl:g} nm {self.frac[u]:.0%} band {int(r['n_unmod'])}x{int(r['n_unmod'])} {int(r['N_unmod'])} modes {r['fs_unmod']:.0f} Hz ML, "
                f"mod {self.mod_wvl:g} nm {self.frac[m]:.0%} band rMod {RMOD_MOD:g} {r['fs_mod']:.0f} Hz{fc}")

    def report(self, selected):
        def line(cells):
            return '| ' + ' | '.join(str(c) for c in cells) + ' |'
        def values(r):
            return [f"{r['mag']:g}", f"{r['fs_mod']:.0f}", f"{r['fs_unmod']:.0f}", f"{int(r['n_unmod'])}", f"{int(r['N_unmod'])}",
                    f"{r['fc_' + self.unmod_chain]:.3f}", f"{r['fc_' + self.mod_chain]:.3f}", f"{r['contrast_vis']:.3e}",
                    f"{r['sr_vis']:.4f}", f"{r['contrast_nir']:.3e}", f"{r['sr_nir']:.4f}"]
        head = ['mag 900 nm', 'modulated [Hz]', 'unmod [Hz]', 'unmod subaps', 'unmod modes', 'flux corr unmod', 'flux corr mod', 'contrast 550 nm', 'SR 550 nm',
                'contrast 1650 nm', 'SR 1650 nm']
        title = (f'unmodulated at {self.unmod_wvl:g} nm ({self.frac[self.unmod_chain]:.0%} band), modulated at {self.mod_wvl:g} nm '
                 f'({self.frac[self.mod_chain]:.0%} band), metric {sc.METRIC_BAND}, chromatic {self.chromatic}')
        out = [f'## Best configuration per magnitude: {title}', '', line(head + ['notes', 'TN']), line(['---'] * (len(head) + 2))]
        out += [line(values(r) + [r['notes'], r['tn']]) for r in selected]
        full = ['phase', *head, 'flux vis meas/model', 'flux nir meas/model', 'selected', 'notes', 'TN']
        out += ['', f'## All runs: {title}', '', line(full), line(['---'] * len(full))]
        sel = {id(r) for r in selected}
        for r in self.rows.values():
            out.append(line([r['phase'], *values(r), f"{r['flux_ratio_vis']:.3f}", f"{r['flux_ratio_nir']:.3f}",
                             'yes' if id(r) in sel else '', r['notes'], r['tn']]))
        text = '\n'.join(out) + '\n'
        with open(op.join(self.out, 'table.md'), 'w') as f:
            f.write(text)
        write_selected(self.out, [dict(mag=r['mag'], key=[float(r['mag']), int(r['k']), int(r['n_unmod']), int(r['k_unmod']), float(r['fc_vis']), float(r['fc_nir'])],
                                       config=self.describe(r), **{x: r[x] for x in ('contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'notes', 'tn')})
                                  for r in selected])
        print(text, flush=True)

    def replay(self, n_real, seed0, total_time, mags=None, extra_mags=(), ref_mag=7.0):
        """The selected configurations again with other atmosphere seeds and wind directions (sweep_contrast.wind_realizations), total_time [s]
        each, no atmo or DM modes saved; the configuration (and its flux correction) is not searched again. extra_mags: magnitudes not swept,
        replayed with the configuration selected at ref_mag (ML weights and KF noise at their own flux). Rows in replays.csv."""
        check_ar2()
        save_plant_inverse_iir(self.unity, calib_dir, N_CORR, beta=0.0)
        store = op.join(self.out, 'replays')
        os.makedirs(store, exist_ok=True)
        # realizations already replayed with the configuration selected now (rows of a configuration selected before stay in the file)
        done = {(r['mag'], int(r['realization']), r['config']) for r in read_rows(self.out, 'replays.csv')} if op.exists(op.join(self.out, 'replays.csv')) else set()
        sel = read_selected(self.out)
        ref = [x for x in sel if float(x['mag']) == ref_mag]
        items = [(x, float(x['mag'])) for x in sel] + [(ref[0], float(m)) for m in extra_mags if ref]
        for s, mag in items:
            _, k, n, ku, fv, fn = (list(s['key']) + [1.0, 1.0])[:6]
            if mags and mag not in mags:
                continue
            fc = {'vis': float(fv), 'nir': float(fn)}
            i = [x[0] for x in LADDER].index(int(n))
            for j, (seed, dirs) in enumerate(wind_realizations(n_real, seed0)):
                if (mag, j, s['config']) in done:
                    continue
                ov, flux = self.overrides(mag, int(k), i, int(ku), self.name(mag, int(k), int(n), int(ku), fc), fc)
                ov = replay_overrides(ov, total_time, seed, dirs)
                ov['data_store_override']['store_dir'] = store
                tn = self.simulate(ov, f'replay_m{mag:g}_k{j}', store)
                path = op.join(store, tn)
                fs = FS / ku
                cv, _, sv = metrics(path, 'vis', fs)
                cn, _, sn = metrics(path, 'nir', fs)
                ratio = measured_flux(path, self.unmod_chain, fs) / flux[self.unmod_chain]
                append_replay(self.out, dict(mag=mag, config=s['config'], realization=j, seed=seed, wind_direction=' '.join(f'{d:g}' for d in dirs),
                                             contrast_vis=cv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, flux_ratio=ratio,
                                             notes='' if np.isfinite(cv) and np.isfinite(sv) else 'diverged', tn=tn))
                print(f'replay mag {mag:g} realization {j} seed {seed}: contrast vis {cv:.3e} nir {cn:.3e}  SR vis {sv:.4f} nir {sn:.4f}', flush=True)


def calibrate(gpu):
    """Aliasing runs with N_CORR modes corrected on the smaller rungs of the LADDER (both wavelengths; existing products are skipped)."""
    rungs = LADDER[1:]
    cfg = calibration.E2EConfig(n_subaps=tuple(n for n, _ in rungs), rMods=(RMOD_UNMOD,), seeings=(SEEING,),
                                modes_per_subap={n: [N, N_CORR] for n, N in rungs})
    calibration.calibrate_alias(cfg, gpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibrate', action='store_true', help='make the missing aliasing products and exit')
    ap.add_argument('--unmod-wvl', type=float, default=900.0, choices=sorted(BANDS), help='[nm] sensing wavelength of the unmodulated pyramid')
    ap.add_argument('--case', nargs=4, type=float, metavar=('MAG', 'K', 'RUNG', 'KU'), help='one run only (test): magnitude, modulated frame in loop steps, LADDER index, loop rate = 2000 / KU')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--achromatic', action='store_true', help='prop.enable_chromatic_effect = false')
    ap.add_argument('--bands', type=float, nargs=4, metavar=('UNMOD_NM', 'UNMOD_BANDW', 'MOD_NM', 'MOD_BANDW'),
                    help='each pyramid at its own wavelength and camera band [nm] (unmodulated first); products must exist at both')
    ap.add_argument('--same-wvl', type=float, choices=sorted(BANDS), help='[nm] equal flux: both pyramids at this wavelength, band split by --unmod-frac')
    ap.add_argument('--unmod-frac', type=float, default=0.5, help='with --same-wvl: fraction of the band of the unmodulated pyramid')
    ap.add_argument('--replay', type=int, metavar='N_REAL', help='rerun the selected configurations N_REAL times with other seeds and wind directions')
    ap.add_argument('--replay-mags', type=float, nargs='+', help='replay these magnitudes only')
    ap.add_argument('--replay-extra-mags', type=float, nargs='+', default=(), help='also replay these (not swept) magnitudes with the mag 7 configuration')
    ap.add_argument('--replay-seed', type=int, default=20261108, help='seed of the random wind directions of the replays')
    ap.add_argument('--total-time', type=float, default=3.0, help='[s] of each run (of each replay with --replay)')
    ap.add_argument('--out')
    a = ap.parse_args()
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']   # the specula CLI of this environment
    if a.calibrate:
        calibrate(a.gpu)
        return
    sweep = Sweep(a.out or op.join(result_dir, f'sweep_dual_unmod{a.unmod_wvl:g}_20261106'), a.gpu, a.total_time, a.unmod_wvl, not a.achromatic, a.same_wvl, a.unmod_frac, a.bands)
    if a.case:
        check_ar2()
        save_plant_inverse_iir(sweep.unity, calib_dir, N_CORR, beta=0.0)
        sweep.evaluate('test', a.case[0], int(a.case[1]), int(a.case[2]), int(a.case[3]))
    elif a.replay:
        sweep.replay(a.replay, a.replay_seed, a.total_time, a.replay_mags, a.replay_extra_mags)
    else:
        sweep.run()


if __name__ == '__main__':
    main()
