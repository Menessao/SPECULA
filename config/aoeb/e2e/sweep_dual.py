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

from sweep_contrast import DELAY, SIGMA_E, SMALL_CHANGE, ar2_at, better, check_ar2, metrics, rel_change    # also specula.init(-1)
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
MAGS = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
MAG_CAL = {900.0: 7.0, 1300.0: 6.43}    # source_vis, source_nir of ristretto_main.yml: the magnitudes of the calibration frames
BANDS = {900.0: 'vis', 1300.0: 'nir'}
ALIAS_SEEINGS = (0.75, 1.05)            # 48x48: the s0.9 frames were renamed from the old e2e and have no slopes file
R_PAD = 1e6                             # noise of a padded mode / its turbulence variance: gain below 1e-6
FLUX_TOL = 0.10
FIELDS = ('phase', 'mag', 'k', 'fs_mod', 'k_unmod', 'fs_unmod', 'n_unmod', 'N_unmod', 'contrast_vis', 'peak_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'flux_ratio_vis', 'flux_ratio_nir', 'notes', 'tn')


def flux_per_frame(rmod, n, N, wvl, mag, exposure):
    """Pupil flux of one frame of `exposure` [s] [e-], at the magnitude of that band for the 900 nm magnitude `mag`."""
    return get_pupil_flux(rmod, n, N, wvl, nCorrModes=N_CORR) * exposure * FS * 10 ** (-0.4 * (mag - MAG_CAL[900.0]))


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
        save_rec(path, np.pad(rec, ((0, N_CORR - N), (0, 0))), method='ml', n_phot=nph, seeing=SEEING, wvl_nm=wvl)
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
        Slopes(slopes=get_static_slopes(rmod, n, N, wvl, nCorrModes=N_CORR)).save(path)
    return tag


class Sweep:
    def __init__(self, out, gpu, total_time, unmod_wvl):
        self.out, self.gpu, self.total_time, self.unmod_wvl = out, gpu, total_time, unmod_wvl
        self.mod_wvl = next(w for w in BANDS if w != unmod_wvl)
        self.store = op.join(out, 'runs')
        self.csv = op.join(out, 'results.csv')
        self.unity = f'sweep_dual_unity_{N_CORR}modes_unmod{unmod_wvl:g}'
        for d in (self.store, op.join(out, 'logs'), op.join(out, 'overrides')):
            os.makedirs(d, exist_ok=True)
        self.rows = {}
        if op.exists(self.csv):
            with open(self.csv) as f:
                for r in csv.DictReader(f):
                    for k in FIELDS:
                        if k not in ('phase', 'notes', 'tn'):
                            r[k] = float(r[k])
                    self.rows[(r['mag'], int(r['k']), int(r['n_unmod']), int(r['k_unmod']))] = r
            print(f'{len(self.rows)} runs read from {self.csv}', flush=True)

    def sensor(self, wvl, mag, steps, n, N, fs):
        """(overrides, sensor of the filter, flux per frame) of one pyramid, frame of `steps` steps of the loop at fs."""
        band = BANDS[wvl]
        unmod = wvl == self.unmod_wvl
        rmod = RMOD_UNMOD if unmod else RMOD_MOD
        exposure = steps / fs
        flux = flux_per_frame(rmod, n, N, wvl, mag, exposure)
        if unmod:
            P = paths(wvl)
            path = P.avgframe(rmod, n, SEEING, N_CORR)
            frame_flux = float(np.sum(fits.getdata(path))) / calibration.avgframe_steps(path) * exposure * FS * 10 ** (-0.4 * (mag - MAG_CAL[900.0]))
            rec_tag, rec = ml_rec(n, N, wvl, frame_flux)
            alias = ml_aliasing(rec, n, N, wvl) * frame_alias_fraction(wvl, steps, fs)
        else:
            rec = get_rec(rmod, n, N, wvl)
            rec_tag = op.basename(paths(wvl).rec(rmod, n, N))[:-5]
            alias = get_aliasing_power(rmod, n, N, wvl, nCorrModes=N_CORR) * frame_alias_fraction(wvl, steps, fs)
        a1, a2, q = ar2_at(fs)
        pad = R_PAD * ar2_stationary_variance(a1, a2, q)[N:N_CORR]
        sensor = dict(noise_var=np.concatenate([noise_var(rec, rmod, n, N, wvl, flux), pad]), aliasing=np.concatenate([alias, np.zeros(N_CORR - N)]),
                      frame_steps=steps, biased=False)
        ov = {f'source_{band}_override': {'magnitude': MAG_CAL[wvl] + mag - MAG_CAL[900.0], 'wavelengthInNm': wvl},
              f'pyr_{band}_override': {'pup_diam': n, 'pup_dist': calibration.E2EConfig().pup_dist(n), 'mod_amp': rmod, 'wavelengthInNm': wvl},
              f'cred_{band}_override': {'dt': exposure, 'readout_level': SIGMA_E},
              f'pyr_slopes_{band}_override': {'pupdata_object': f'pyr_pupdata_{n}x{n}', 'sn_object': sn_tag(rmod, n, N, wvl), 'outputs': ['out_slopes', 'out_flux']},
              f'pyr_modes_{band}_override': {'recmat_object': rec_tag}}
        return ov, sensor, flux

    def evaluate(self, phase, mag, k, i, ku=1):
        """k: modulated frame in steps of the loop at FS / ku."""
        n, N = LADDER[i]
        key = (mag, k, n, ku)
        if key in self.rows:
            return self.rows[key]
        fs = FS / ku
        name = f'm{mag:g}_k{k}_n{n}' + (f'_ku{ku}' if ku > 1 else '')
        ov, sensors, flux = {}, [], {}
        for wvl in BANDS:                       # order of kalman.in_measurements_list
            o, s, flux[wvl] = self.sensor(wvl, mag, 1, n, N, fs) if wvl == self.unmod_wvl else self.sensor(wvl, mag, k, N_SUBAPS_MOD, N_CORR, fs)
            ov.update(o)
            sensors.append(s)
        kf_tag = f'sweep_dual_unmod{self.unmod_wvl:g}_{name}'
        a1, a2, q = ar2_at(fs)
        save_kalman_data(kf_tag, calib_dir, a1, a2, q, sensors, n_modes=N_CORR, delay=1 + DELAY * fs)
        ov.update({
            'main_override': {'time_step': 1 / fs, 'total_time': self.total_time, 'root_dir': calib_dir},
            'kalman_override': {'kalman_filter_data_object': kf_tag, 'time_step': 1 / fs, 'gain_period': k,
                                'inputs': {'in_measurements_list': ['pyr_modes_vis.out_modes', 'pyr_modes_nir.out_modes'], 'in_command': 'kpred.out_comm:-1'},
                                'outputs': ['out_projected_state', 'out_state']},
            'kpred_override': {'delay': DELAY * fs, 'iir_filter_data_object': self.unity},
            'coro_psf_vis_override': {'verbose': False},
            'data_store_override': {'store_dir': self.store, 'inputs': {'input_list': [
                'sr_vis-coro_psf_vis.out_sr', 'coro_psf_vis_int-coro_psf_vis.out_int_coronagraph_psf_profile',
                'sr_nir-coro_psf_nir.out_sr', 'coro_psf_nir_int-coro_psf_nir.out_int_coronagraph_psf_profile',
                'atmo_modes-atmo_modes.out_value', 'res_modes-res_modes.out_modes',
                'flux_vis-pyr_slopes_vis.out_flux', 'flux_nir-pyr_slopes_nir.out_flux']}}})
        stem = op.join(self.out, 'overrides', name)
        with open(stem + '.yml', 'w') as f:
            yaml.safe_dump(ov, f, sort_keys=False)
        # A crash is not a bad configuration: it stops the sweep (CalledProcessError), which is resumed by starting it again
        run_specula('ristretto_main.yml', stem, log_file=op.join(self.out, 'logs', name + '.log'), extra_args=('--target', str(self.gpu)))
        tn = latest_tn(self.store)
        path = op.join(self.store, tn)
        cv, pv, sv = metrics(path, 'vis', fs)
        cn, _, sn = metrics(path, 'nir', fs)
        row = dict(phase=phase, mag=mag, k=k, fs_mod=fs / k, k_unmod=ku, fs_unmod=fs, n_unmod=n, N_unmod=N, contrast_vis=cv, peak_vis=pv, sr_vis=sv, contrast_nir=cn, sr_nir=sn, tn=tn)
        notes = []
        for wvl, band in BANDS.items():
            frame = fits.getdata(op.join(path, f'flux_{band}.fits'))
            row[f'flux_ratio_{band}'] = float(np.mean(frame[frame != 0])) / flux[wvl]   # a slower sensor may hold zeros before its first frame
            if abs(row[f'flux_ratio_{band}'] - 1) > FLUX_TOL:
                notes.append(f"flux {band} {row[f'flux_ratio_{band}']:.2f} x model")
        if not np.isfinite(cv) or not np.isfinite(sv):
            notes.append('diverged')
            row['contrast_vis'] = np.inf
        row['notes'] = '; '.join(notes)
        self.rows[key] = row
        new = not op.exists(self.csv)
        with open(self.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"{phase:10s} mag {mag:g} modulated {fs / k:6.0f} Hz, unmodulated {fs:6.0f} Hz {n}x{n} {N} modes: contrast {row['contrast_vis']:.3e}  peak {pv:.3e}  SR {sv:.4f}  {row['notes'] and 'WARNING ' + row['notes']}", flush=True)
        return row

    def note(self, row, against):
        change = rel_change(row, against)
        if change < SMALL_CHANGE:
            text = f'change {change:.2%}: may be noise'
            if text not in row['notes']:
                row['notes'] = '; '.join(x for x in (row['notes'], text) if x)

    def search(self, mag, k, i):
        """Best (k, ladder index) at this magnitude."""
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
                return k, i

    def run(self):
        check_ar2()
        save_plant_inverse_iir(self.unity, calib_dir, N_CORR, beta=0.0)
        k, i, selected = 1, 0, []
        for mag in MAGS:
            k, i = self.search(mag, k, i)
            best = self.evaluate('magnitude', mag, k, i)
            sel = (mag, k, LADDER[i][0], 1)
            if mag >= SLOW_FROM:
                for ku in K_UNMOD:
                    k_slow = max(1, round(k / ku))      # same modulated frame duration, as well as whole steps allow
                    row = self.evaluate('unmod rate', mag, k_slow, i, ku)
                    self.note(row, best)
                    if better(row, best):
                        best, sel = row, (mag, k_slow, LADDER[i][0], ku)
            selected.append(sel)
        self.report(selected)

    def report(self, selected):
        def line(cells):
            return '| ' + ' | '.join(str(c) for c in cells) + ' |'
        def values(r):
            return [f"{r['mag']:g}", f"{r['fs_mod']:.0f}", f"{r['fs_unmod']:.0f}", f"{int(r['n_unmod'])}", f"{int(r['N_unmod'])}", f"{r['contrast_vis']:.3e}", f"{r['peak_vis']:.3e}", f"{r['sr_vis']:.4f}"]
        head = ['mag 900 nm', 'modulated [Hz]', 'unmod [Hz]', 'unmod subaps', 'unmod modes', 'avg contrast', 'peak contrast', 'SR 550 nm']
        title = f'unmodulated at {self.unmod_wvl:g} nm, modulated at {self.mod_wvl:g} nm'
        out = [f'## Best configuration per magnitude: {title}', '', line(head), line(['---'] * len(head))]
        out += [line(values(self.rows[k])) for k in selected]
        full = ['phase', *head, 'contrast 1650 nm', 'SR 1650 nm', 'flux vis meas/model', 'flux nir meas/model', 'selected', 'notes', 'TN']
        out += ['', f'## All runs: {title}', '', line(full), line(['---'] * len(full))]
        for k, r in self.rows.items():
            out.append(line([r['phase'], *values(r), f"{r['contrast_nir']:.3e}", f"{r['sr_nir']:.4f}", f"{r['flux_ratio_vis']:.3f}", f"{r['flux_ratio_nir']:.3f}",
                             'yes' if k in selected else '', r['notes'], r['tn']]))
        text = '\n'.join(out) + '\n'
        with open(op.join(self.out, 'table.md'), 'w') as f:
            f.write(text)
        print(text, flush=True)


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
    ap.add_argument('--total-time', type=float, default=3.0, help='[s] of each run')
    ap.add_argument('--out')
    a = ap.parse_args()
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']   # the specula CLI of this environment
    if a.calibrate:
        calibrate(a.gpu)
        return
    sweep = Sweep(a.out or op.join(result_dir, f'sweep_dual_unmod{a.unmod_wvl:g}_20261106'), a.gpu, a.total_time, a.unmod_wvl)
    if a.case:
        check_ar2()
        save_plant_inverse_iir(sweep.unity, calib_dir, N_CORR, beta=0.0)
        sweep.evaluate('test', a.case[0], int(a.case[1]), int(a.case[2]), int(a.case[3]))
    else:
        sweep.run()


if __name__ == '__main__':
    main()
