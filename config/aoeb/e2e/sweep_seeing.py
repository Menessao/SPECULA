"""Seeing scan of three configurations at the bright magnitude (MAG), achromatic: single-stage rMod 2, single-stage rMod 0 and their fusion
with the 400 nm band split between the two pyramids (both at 900 nm, UNMOD_FRAC of the band on the unmodulated one). N_REAL realizations
(atmosphere seed and uniform random wind direction per layer, sweep_contrast.wind_realizations, the same for every case and seeing) of
TOTAL_TIME s at every seeing of SEEINGS along the line of sight (zenith angle of ristretto_main.yml). Only the SR (550 and 1650 nm) is saved.

Configurations: those selected by the final sweep (sweep_final.py) at mag 7 for the achromatic 900 nm cases, all at 2 kHz: 48x48, 1300 modes;
rMod 2 with the SVD reconstructor and filter model, rMod 0 with the ML reconstructor and filter model; dual: unmodulated 48x48 1300 modes ML
2 kHz, modulated rMod 2 2 kHz, both at 900 nm. The Kalman model (ML weights, noise, aliasing, AR(2)) is the one of the sweeps, built on the
0.9" products; the turbulence variance of the AR(2) and the aliasing are scaled by (seeing / 0.9)^(5/3) at the other seeings (KF_SCALE).

Every (case, seeing) is a folder of OUT with its own Kalman data; the runs go to <case>_s<seeing>/runs/k<realization>/<TN>. Rows in
results.csv; finished realizations are skipped, so the scan is resumed by starting it again.

    python -u sweep_seeing.py --gpus 0 1
    python -u sweep_seeing.py --plot            # table and figure from results.csv
"""
import argparse
import csv
import os
import os.path as op
import queue
import sys
import threading
import time

import numpy as np
import yaml
from astropy.io import fits

import sweep_contrast as sc          # also specula.init(-1)
import sweep_dual as sd              # noqa: E402
import save_kalman_data as skd       # noqa: E402
from save_kalman_data import save_plant_inverse_iir   # noqa: E402
from specula.data_objects.iir_filter_data import IirFilterData   # noqa: E402
from utils import E2E_DIR, SEEING, calib_dir, latest_tn, result_dir, run_specula   # noqa: E402

OUT = op.join(result_dir, 'sweep_seeing_20261110')
SEEINGS = (0.7, 0.9, 1.1, 1.3)      # [arcsec] along the line of sight
N_REAL, SEED0 = 10, 20261110
TOTAL_TIME, MAG, FS, WVL = 2.0, 7.0, 2000.0, 900.0
UNMOD_FRAC = 0.5                    # dual: fraction of the band on the unmodulated pyramid
KF_SCALE = True                     # scale the AR(2) variance and the aliasing of the filter model with seeing^(5/3)
CASES = {'single_r2': 'single stage, rMod 2 (SVD)', 'single_r0': 'single stage, rMod 0 (ML)',
         'dual': f'fusion rMod 0 ({UNMOD_FRAC:.0%} of the flux) + rMod 2 ({1 - UNMOD_FRAC:.0%})'}
INT_GAIN = 0.5      # integrator cases: no Kalman filter, c[k] = c[k-1] + gain x measured residual modes, same 0.5 step fractional delay
SINGLE = {'single_r2': (2.0, 'svd', 'svd', None), 'single_r0': (0.0, 'ml', 'ml', None),      # rMod, reconstructor, KF model, integrator gain
          'single_r0_svd': (0.0, 'svd', 'svd', None), 'single_r0_mlsvd': (0.0, 'ml', 'svd', None),
          'int_r0_ml': (0.0, 'ml', 'ml', INT_GAIN), 'int_r0_svd': (0.0, 'svd', 'svd', INT_GAIN), 'int_r2_svd': (2.0, 'svd', 'svd', INT_GAIN)}
CMD_GAIN = {'single_r0_g95': 0.95, 'single_r0_svd_g95': 0.95}      # Kalman cases: command scaled before the DM (identity IIR stage x gain)
CMD_FF = {'single_r0_lp95': 0.05}       # forgetting factor of the command stage: c = gain u + ff c_prev, unit DC gain when gain = 1 - ff
CMD_GAIN['single_r0_lp95'] = 0.95
SINGLE.update({'single_r0_lp95': (0.0, 'ml', 'ml', None), 'single_r0_g95': (0.0, 'ml', 'ml', None), 'single_r0_svd_g95': (0.0, 'svd', 'svd', None)})
EXTRA = {'single_r0_svd': 'single stage, rMod 0 (SVD rec and KF model)', 'single_r0_mlsvd': 'single stage, rMod 0 (ML rec, SVD KF model)',
         'int_r0_ml': 'rMod 0 (ML), integrator 0.5', 'int_r0_svd': 'rMod 0 (SVD), integrator 0.5', 'int_r2_svd': 'rMod 2 (SVD), integrator 0.5'}
FIELDS = ('case', 'seeing', 'realization', 'seed', 'wind_direction', 'sr_vis', 'sr_nir', 'sr_start', 'sr_end', 'flux_ratio_unmod', 'notes', 'tn')
STORE = ['sr_vis-coro_psf_vis.out_sr', 'sr_nir-coro_psf_nir.out_sr']
REMOVE = ['res_modes', 'dm_modes', 'atmo_modes']

_state = {'seeing': SEEING}
_lock = threading.Lock()


def log(text):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {text}', flush=True)


# ---- hooks into the sweep modules: seeing of the yml and scaling of the filter model ----
def _sim_seeing():
    with open(op.join(E2E_DIR, 'ristretto_main.yml')) as f:
        z = yaml.safe_load(f)['main']['zenithAngleInDeg'] or 0.0
    return float(_state['seeing'] * np.cos(np.radians(z)) ** 0.6)


_save_kalman_data = skd.save_kalman_data


def _scaled_save_kalman_data(tag, root_dir, a1, a2, q, sensors, **kw):
    f = (_state['seeing'] / SEEING) ** (5 / 3) if KF_SCALE else 1.0
    sensors = [dict(s, aliasing=np.asarray(s.get('aliasing', 0.0)) * f) for s in sensors]
    return _save_kalman_data(tag, root_dir, a1, a2, np.asarray(q) * f, sensors, **kw)


sc.sim_seeing = sd.sim_seeing = _sim_seeing
sc.save_kalman_data = sd.save_kalman_data = _scaled_save_kalman_data
sc.WVL, sc.REC, sc.CHROMATIC = WVL, 'auto', False


def case_dir(case, seeing, fc=1.0):
    return op.join(OUT, f'{case}_s{seeing:g}' + (f'_f{fc:.3f}' if fc != 1.0 else ''))


def overrides(case, seeing, fc=1.0):
    """Overrides of one (case, seeing) before the realization (seed, wind, store) is set; writes the Kalman data of this case and seeing."""
    _state['seeing'] = seeing
    out = case_dir(case, seeing, fc)
    if case in SINGLE:
        rmod, rec, kf, gain = SINGLE[case]
        sc.REC = rec
        sw = sc.Sweep(out, 0, TOTAL_TIME, rmod)
        ov = sw.overrides(rmod, MAG, FS, 48, 1300, sw.build_filter(rmod, MAG, FS, 48, 1300, kf, fc), fc)
        ov['remove'] = ov['remove'] + REMOVE
        if gain:        # integrator in place of the Kalman filter: the IIR filter takes the modal measurement itself
            tag = f'seeing_integrator_g{gain:g}_{sd.N_CORR}modes'
            IirFilterData.from_gain_and_ff(np.full(sd.N_CORR, gain)).save(op.join(calib_dir, 'filter', tag + '.fits'))
            ov.pop('kalman_override')
            ov['remove'] = ov['remove'] + ['kalman']
            ov['kpred_override'] = {'delay': sc.DELAY * FS, 'iir_filter_data_object': tag, 'inputs': {'delta_comm': f'pyr_modes_{sc.band()}.out_modes'}}
        if case in CMD_GAIN:
            ff = CMD_FF.get(case, 0.0)
            tag = f'seeing_cmdgain_{CMD_GAIN[case]:g}_ff{ff:g}_{sd.N_CORR}modes'
            IirFilterData.from_gain_and_ff(np.full(sd.N_CORR, CMD_GAIN[case]), ff=np.full(sd.N_CORR, ff)).save(op.join(calib_dir, 'filter', tag + '.fits'))
            ov['kpred_override'] = dict(ov['kpred_override'], iir_filter_data_object=tag)
        flux_key, flux_model = f'flux_{sc.band()}', sc.flux_per_frame(rmod, MAG, FS, 48, 1300)
    else:
        sw = sd.Sweep(out, 0, TOTAL_TIME, WVL, chromatic=False, same_wvl=WVL, unmod_frac=UNMOD_FRAC)
        sd.check_ar2()
        save_plant_inverse_iir(sw.unity, calib_dir, sd.N_CORR, beta=0.0)
        one = {c: 1.0 for c in sd.CHAINS}
        ov, flux = sw.overrides(MAG, 1, 0, 1, sw.name(MAG, 1, 48, 1, one), one)
        ov['remove'] = REMOVE
        flux_key, flux_model = f'flux_{sw.unmod_chain}', flux[sw.unmod_chain]
    assert ov['seeing_override']['constant'] == _sim_seeing()
    ov['main_override']['total_time'] = TOTAL_TIME
    for b in ('vis', 'nir'):
        ov.setdefault(f'coro_psf_{b}_override', {}).update({'verbose': False, 'compute_profile_metrics': False})
    lst = ov['data_store_override']['inputs']['input_list']
    ov['data_store_override']['inputs']['input_list'] = STORE + [x for x in lst if x.startswith(flux_key)]
    return ov, flux_key, flux_model


def read_rows():
    path = op.join(OUT, 'results.csv')
    if not op.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('seeing', 'sr_vis', 'sr_nir', 'sr_start', 'sr_end', 'flux_ratio_unmod'):
            r[k] = float(r[k]) if r[k] not in ('', None) else np.nan
        r['realization'] = int(r['realization'])
    return rows


def append_row(row):
    path = op.join(OUT, 'results.csv')
    with _lock:
        new = not op.exists(path)
        with open(path, 'a', newline='') as f:
            w = csv.DictWriter(f, FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)


def mean_sr(path, band):
    sr = np.asarray(fits.getdata(op.join(path, f'sr_{band}.fits')), float).ravel()
    i0 = int(round(sc.START_TIME * FS))
    return float(np.mean(sr[i0:]))


def run_job(case, seeing, j, seed, dirs, ov, flux_key, flux_model, gpu, fc=1.0):
    out = case_dir(case, seeing, fc)
    label = case + (f'_f{fc:.3f}' if fc != 1.0 else '')
    store = op.join(out, 'runs', f'k{j}')
    os.makedirs(store, exist_ok=True)
    ov = sc.replay_overrides(ov, TOTAL_TIME, seed, dirs)
    ov['data_store_override']['store_dir'] = store
    name = f'k{j}'
    stem = op.join(out, 'overrides', name)
    with open(stem + '.yml', 'w') as f:
        yaml.safe_dump(ov, f, sort_keys=False)
    before = set(os.listdir(store))
    run_specula('ristretto_main.yml', stem, log_file=op.join(out, 'logs', name + '.log'), extra_args=('--target', str(gpu)))
    new = sorted(set(os.listdir(store)) - before)
    tn = new[-1] if new else latest_tn(store)
    path = op.join(store, tn)
    sv, sn = mean_sr(path, 'vis'), mean_sr(path, 'nir')
    a, e = sc.sr_drift(path, FS)
    flux = np.asarray(fits.getdata(op.join(path, flux_key + '.fits')), float).ravel()[int(round(sc.START_TIME * FS)):]
    row = dict(case=label, seeing=seeing, realization=j, seed=seed, wind_direction=' '.join(f'{d:g}' for d in dirs), sr_vis=sv, sr_nir=sn,
               sr_start=a, sr_end=e, flux_ratio_unmod=float(np.mean(flux[flux != 0])) / flux_model,
               notes='' if np.isfinite(sv) and np.isfinite(sn) else 'diverged', tn=tn)
    append_row(row)
    log(f'{label} seeing {seeing:g} realization {j} (GPU {gpu}): SR 550 nm {sv:.4f}  1650 nm {sn:.4f}  {row["notes"]}')


def run(gpus, cases=None, seeings=None, reals_sel=None, fc=1.0):
    done = {(r['case'], r['seeing'], r['realization']) for r in read_rows()}
    sc.check_ar2()
    reals = sc.wind_realizations(N_REAL, SEED0)
    jobs = queue.Queue()
    n = 0
    for seeing in seeings or SEEINGS:
        for case in cases or CASES:
            label = case + (f'_f{fc:.3f}' if fc != 1.0 else '')
            todo = [j for j in (reals_sel or range(N_REAL)) if (label, float(seeing), j) not in done]
            if not todo:
                continue
            ov, flux_key, flux_model = overrides(case, seeing, fc)
            for j in todo:
                jobs.put((case, seeing, j, *reals[j], ov, flux_key, flux_model))
                n += 1
    log(f'{n} runs to do, {len(done)} done')

    def worker(gpu):
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            try:
                run_job(*job, gpu, fc)
            except Exception as e:      # a crashed run: logged, the scan goes on (resumed by starting again)
                log(f'FAILED {job[0]} seeing {job[1]:g} realization {job[2]} (GPU {gpu}): {e!r}')

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()
    log('all done')


LOST = 0.8      # a run whose 550 nm SR is below LOST x the mean of the single-stage rMod 2 runs at the same seeing has lost the loop


def stats():
    """{case: {seeing: dict}}: mean/std/min/max of sr_vis and sr_nir over ALL realizations, and `lost`, the number that lost the loop
    (the drift test of sweep_contrast.ran_away misses a loop lost before START_TIME, so the reference is the rMod 2 mean)."""
    rows = read_rows()
    out = {}
    for case in CASES:
        out[case] = {}
        for s in SEEINGS:
            rs = [r for r in rows if r['case'] == case and r['seeing'] == s]
            if not rs:
                continue
            ref = np.mean([r['sr_vis'] for r in rows if r['case'] == 'single_r2' and r['seeing'] == s])
            d = dict(n=len(rs), lost=sum(not (r['sr_vis'] >= LOST * ref) for r in rs))
            for b in ('vis', 'nir'):
                v = np.array([r[f'sr_{b}'] for r in rs])
                d[f'sr_{b}'] = (float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max()))
            out[case][s] = d
    return out


def table():
    st = stats()
    line = lambda cells: '| ' + ' | '.join(str(c) for c in cells) + ' |'   # noqa: E731
    head = ['case', 'seeing LOS ["]', 'N (loop lost)', 'SR 550 nm', 'SR 1650 nm', 'min / max SR 550 nm']
    out = ['# Seeing scan at mag 7 (900 nm), achromatic', '',
           f'{N_REAL} realizations (atmosphere seed and random wind directions, seed {SEED0}) of {TOTAL_TIME:g} s per case and seeing; SR averaged '
           f'after {sc.START_TIME:g} s, mean +- std over all the realizations; loop lost: SR below '
           f'{LOST:g} x the rMod 2 mean at that seeing. All sensors 48x48, 1300 modes, 2 kHz, 900 nm; Kalman model of the 0.9" sweeps, AR(2) variance and '
           f'aliasing scaled with seeing^(5/3): {KF_SCALE}.', '', line(head), line(['---'] * len(head))]
    for case, label in CASES.items():
        for s, d in st[case].items():
            out.append(line([label, f'{s:g}', f"{d['n']} ({d['lost']})", f"{d['sr_vis'][0]:.3f} +- {d['sr_vis'][1]:.3f}",
                             f"{d['sr_nir'][0]:.3f} +- {d['sr_nir'][1]:.3f}", f"{d['sr_vis'][2]:.3f} / {d['sr_vis'][3]:.3f}"]))
    text = '\n'.join(out) + '\n'
    with open(op.join(OUT, 'table.md'), 'w') as f:
        f.write(text)
    return text


def plot(path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    st = stats()
    rows = read_rows()
    colors = {'single_r2': '#4477AA', 'single_r0': '#EE6677', 'dual': '#228833'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, b, wl in zip(axes, ('vis', 'nir'), ('550 nm', '1650 nm')):
        for i, (case, label) in enumerate(CASES.items()):
            dx = (i - 1) * 0.012
            ss = sorted(st[case])
            m = [st[case][s][f'sr_{b}'][0] for s in ss]
            e = [st[case][s][f'sr_{b}'][1] for s in ss]
            ax.errorbar(np.array(ss) + dx, m, yerr=e, fmt='o-', capsize=3, color=colors[case], label=label, lw=1.5, ms=5)
            pts = [(r['seeing'] + dx, r[f'sr_{b}']) for r in rows if r['case'] == case and np.isfinite(r[f'sr_{b}'])]
            if pts:
                ax.scatter(*zip(*pts), s=8, color=colors[case], alpha=0.35, zorder=1)
            for s in ss:
                if st[case][s]['lost']:
                    ax.annotate(f"{st[case][s]['lost']}/{st[case][s]['n']} lost", (s + dx, st[case][s][f'sr_{b}'][0]), textcoords='offset points',
                                xytext=(6, -12), fontsize=7, color=colors[case])
        ax.set_xlabel('seeing along the line of sight [arcsec]')
        ax.set_ylabel(f'SR {wl}')
        ax.set_xticks(SEEINGS)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle(f'Mag 7 (900 nm), achromatic, {N_REAL} realizations of {TOTAL_TIME:g} s per point: mean +- std, dots: single runs', fontsize=10)
    fig.tight_layout()
    path = path or op.join(OUT, 'sr_vs_seeing.png')
    fig.savefig(path, dpi=150)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0])
    ap.add_argument('--plot', action='store_true', help='table and figure only, from results.csv')
    ap.add_argument('--dry', action='store_true', help='build the Kalman data and print the overrides of every (case, seeing) without simulating')
    ap.add_argument('--cases', nargs='+', choices=sorted(set(CASES) | set(SINGLE)), help='these cases only (default: the three of the scan)')
    ap.add_argument('--seeings', type=float, nargs='+', help='these seeings only')
    ap.add_argument('--reals', type=int, nargs='+', help='these realizations only')
    ap.add_argument('--fc', type=float, default=1.0, help='single cases: measured / model flux applied to the ML weights and KF noise (the case is tagged _f<fc>)')
    a = ap.parse_args()
    os.environ['PATH'] = op.dirname(sys.executable) + os.pathsep + os.environ['PATH']
    os.makedirs(OUT, exist_ok=True)
    if a.dry:
        for seeing in SEEINGS:
            for case in CASES:
                ov, fk, fm = overrides(case, seeing)
                print(f'### {case} seeing {seeing}: {fk} model {fm:.4g} e-/frame\n', yaml.safe_dump(ov, sort_keys=False)[:3000] if case == 'dual' and seeing == SEEINGS[0] else
                      {k: ov[k] for k in ('seeing_override', 'kalman_override', 'data_store_override', 'remove')}, flush=True)
        return
    if not a.plot:
        run(a.gpus, a.cases, a.seeings, a.reals, a.fc)
    print(table())
    print(plot())


if __name__ == '__main__':
    main()
