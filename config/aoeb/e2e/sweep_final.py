"""Final sweep of the e2e configurations, unattended: the single-stage and dual-WFS sweeps of CASES on the GPUs, the table of the selected
configurations, an automatic sanity check and, if it passes, REPLAYS realizations of every selected configuration (other atmosphere seed and
random wind direction of each layer, the same for every case, REPLAY_TIME s, no atmo or DM modes saved) with the mean and std of the results.

Equal-flux dual cases (dual_eq*): both pyramids at the single-stage wavelength, the 400 nm band split between them (SPLITS), swept but not replayed.
Flux: a run whose measured pupil flux differs from the model by more than 5 % is repeated once with the model flux (ML weights, KF noise)
corrected (sweep_contrast.FLUX_ADJUST). Magnitudes 7-12.

Metric of every sweep: mean raw contrast in 2-5 lambda/D in the science band nearest to the WFS (single stage) or to the unmodulated pyramid
(dual): 550 nm for 900 nm, 1650 nm for 1300 nm; SR in that band when SR < 10 %. ML reconstructor on every unmodulated pyramid. Simulation seeing
set so that the seeing along the line of sight is the 0.9" of the calibration products (z = 30 deg of ristretto_main.yml).

Each case is a folder of OUT; every step reads back what is done, so the whole chain is resumed by starting it again:

    python -u sweep_final.py --gpus 0 1                 # sweeps, table, check, replays, replay table
    python -u sweep_final.py --tables                   # tables only, from what is on disk
    python -u sweep_final.py --replay-only --gpus 0 1   # replays even if the check failed (after a look at check.md)
"""
import argparse
import csv
import os
import os.path as op
import queue
import subprocess
import sys
import threading
import time

import numpy as np

HERE = op.dirname(op.abspath(__file__))
REPO = op.abspath(op.join(HERE, '..', '..', '..'))
RESULT_DIR = '/raid1/mmenessini/results/WFSW'
OUT = op.join(RESULT_DIR, 'sweep_final_20261108')
MAGS = (7.0, 8.0, 9.0, 10.0, 11.0, 12.0)   # 13 and 14 dropped on 2026-11-08
SWEEP_TIME, REPLAY_TIME, N_REPLAYS, REPLAY_SEED = 3.0, 5.0, 10, 20261108
RETRIES, RETRY_WAIT = 5, 300           # a crashed sweep is restarted (it resumes from its results.csv)

SINGLE = ['--rec', 'auto', '--ladder', '--no-check']
SETTINGS = ((900, 'achrom'), (900, 'chrom'), (1300, 'chrom'))     # sensing wavelength and chromatism of the single-stage cases
SPLITS = (50, 70, 90, 30, 10)                                     # equal-flux dual: % of the 400 nm band on the unmodulated pyramid


def _refs(w, c):
    return ('achrom',) if c == 'achrom' else (('vis',) if w == 900 else ('nir',))


def _chrom(c):
    return ['--achromatic'] if c == 'achrom' else []


# name, script, arguments, metric band, perfect-coronagraph references (results_analysis.load_perfect_coro) of the floor check, replayed
CASES = [
    ('dual_unmod900_achrom', 'sweep_dual.py', ['--unmod-wvl', '900', '--achromatic'], 'vis', ('achrom',), True),
    ('dual_unmod900_chrom', 'sweep_dual.py', ['--unmod-wvl', '900'], 'vis', ('vis', 'nir'), True),     # two WFS wavelengths: the lower floor
    ('dual_unmod1300_chrom', 'sweep_dual.py', ['--unmod-wvl', '1300'], 'nir', ('vis', 'nir'), True),
] + [(f'dual_eq{w}_{c}_u{u}', 'sweep_dual.py', ['--same-wvl', str(w), '--unmod-frac', f'{u / 100:g}', *_chrom(c)], 'vis' if w == 900 else 'nir', _refs(w, c), False)
     for w, c in SETTINGS for u in SPLITS
] + [(f'single_wfs{w}_{c}_r{r}', 'sweep_contrast.py', ['--wvl', str(w), '--rmod', str(r), *SINGLE, *_chrom(c)], 'vis' if w == 900 else 'nir', _refs(w, c), True)
     for w, c in SETTINGS for r in (0, 1, 2)]
DESCRIPTION = {'dual_unmod900_achrom': 'dual: unmod 900 nm, rMod 2 at 1300 nm, achromatic',
               'dual_unmod900_chrom': 'dual: unmod 900 nm, rMod 2 at 1300 nm, chromatic',
               'dual_unmod1300_chrom': 'dual: unmod 1300 nm, rMod 2 at 900 nm, chromatic'}
for _w, _c in SETTINGS:
    _t = 'achromatic' if _c == 'achrom' else 'chromatic'
    for _r in (0, 1, 2):
        DESCRIPTION[f'single_wfs{_w}_{_c}_r{_r}'] = f'single: {_w} nm, rMod {_r}, {_t}'
    for _u in SPLITS:
        DESCRIPTION[f'dual_eq{_w}_{_c}_u{_u}'] = f'dual equal flux: {_w} nm, unmod {_u}% / rMod 2 {100 - _u}% of the band, {_t}'
# sanity check: tolerance of the "fainter is not better" test, minimum ratio to the perfect-coronagraph floor (references at the sweep seeing and zenith)
MONOTONE_TOL, FLOOR_RATIO = 1.2, 0.9


def log(text):
    print(f'{time.strftime("%Y-%m-%d %H:%M:%S")} {text}', flush=True)


def env():
    e = dict(os.environ)
    e['PYTHONPATH'] = REPO + (os.pathsep + e['PYTHONPATH'] if e.get('PYTHONPATH') else '')
    e['PATH'] = op.dirname(sys.executable) + os.pathsep + e['PATH']
    return e


def run_job(name, cmd, gpu, logfile):
    """cmd with --gpu, retried RETRIES times; True if it finished."""
    for k in range(RETRIES + 1):
        with open(logfile, 'a') as f:
            f.write(f'\n##### {time.ctime()} attempt {k + 1}: {" ".join(cmd)} --gpu {gpu}\n')
            f.flush()
            rc = subprocess.run([sys.executable, '-u', *cmd, '--gpu', str(gpu)], cwd=HERE, env=env(), stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            log(f'DONE {name} (GPU {gpu})')
            return True
        log(f'RETRY {k + 1} {name} (GPU {gpu}): exit {rc}, see {logfile}')
        if k < RETRIES:
            time.sleep(RETRY_WAIT)
    log(f'FAILED {name} (GPU {gpu}) after {RETRIES + 1} attempts')
    return False


def run_queue(jobs, gpus):
    """jobs: (name, cmd, logfile), run in order on the first free GPU. Returns the names that failed."""
    q, failed = queue.Queue(), []
    for j in jobs:
        q.put(j)

    def worker(gpu):
        while True:
            try:
                name, cmd, logfile = q.get_nowait()
            except queue.Empty:
                return
            if not run_job(name, cmd, gpu, logfile):
                failed.append(name)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus]
    for t in threads:
        t.start()
        time.sleep(60)      # staggered: the first runs of a sweep write shared products (atomic, but no need to compute them twice)
    for t in threads:
        t.join()
    return failed


def case_dir(name):
    return op.join(OUT, name)


def sweep_jobs():
    return [(name, [script, *args, '--total-time', str(SWEEP_TIME), '--out', case_dir(name)], op.join(OUT, f'{name}.log'))
            for name, script, args, *_ in CASES]


def replay_jobs(mags=None):
    extra = ['--replay-mags', *[str(m) for m in (mags or MAGS)]]
    return [(f'{name} replay', [script, *args, '--replay', str(N_REPLAYS), '--replay-seed', str(REPLAY_SEED), '--total-time', str(REPLAY_TIME),
                                '--out', case_dir(name), *extra], op.join(OUT, f'{name}_replay.log'))
            for name, script, args, _, _, replayed in CASES if replayed]


def read_csv(path):
    if not op.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('mag', 'contrast_vis', 'sr_vis', 'contrast_nir', 'sr_nir', 'flux_ratio'):
            if k in r and r[k] not in ('', None):
                r[k] = float(r[k])
    return rows


def line(cells):
    return '| ' + ' | '.join(str(c) for c in cells) + ' |'


def sweep_table():
    """Selected configuration of every case and magnitude."""
    head = ['case', 'mag 900 nm', 'configuration', 'contrast 550 nm', 'contrast 1650 nm', 'SR 550 nm', 'SR 1650 nm', 'TN', 'notes']
    out = ['# Final sweep: selected configurations', '',
           f'Mean raw contrast in 2-5 lambda/D (relative to the AO peak), mean SR after 0.5 s; {SWEEP_TIME:g} s runs. Metric band of the selection: '
           '550 nm for a 900 nm WFS / unmodulated pyramid, 1650 nm for 1300 nm. TN folders: <case>/runs/<TN>.', '',
           line(head), line(['---'] * len(head))]
    pivot = {}
    for name, _, _, mb, *_ in CASES:
        rows = read_csv(op.join(case_dir(name), 'selected.csv'))
        for r in rows:
            out.append(line([DESCRIPTION[name], f"{r['mag']:g}", r['config'], f"{r['contrast_vis']:.3e}", f"{r['contrast_nir']:.3e}",
                             f"{r['sr_vis']:.4f}", f"{r['sr_nir']:.4f}", r['tn'], r['notes']]))
            pivot.setdefault(name, {})[r['mag']] = r
        if not rows:
            out.append(line([DESCRIPTION[name], '-', 'not finished', '', '', '', '', '', '']))
    for b, wl in (('vis', '550 nm'), ('nir', '1650 nm')):
        h = ['case', *[f'{m:g}' for m in MAGS]]
        out += ['', f'## Contrast {wl} of the selected configuration, per 900 nm magnitude', '', line(h), line(['---'] * len(h))]
        for name, *_ in CASES:
            out.append(line([DESCRIPTION[name], *[f"{pivot[name][m][f'contrast_{b}']:.2e}" if m in pivot.get(name, {}) else '-' for m in MAGS]]))
    text = '\n'.join(out) + '\n'
    with open(op.join(OUT, 'sweep_table.md'), 'w') as f:
        f.write(text)
    return text


def check():
    """Automatic sanity check of the sweeps before the replays; writes check.md, returns True if it passes.
    Fails on: a case not finished, a selected configuration that diverged, a selected configuration beating the perfect-coronagraph
    (fitting-only) floor, or a metric contrast that gets better by more than MONOTONE_TOL at a fainter magnitude (both with SR >= 10 %, as below
    that the SR selects). Flux and runaway notes are warnings (flux offsets up to ~1.4 on the small unmodulated rungs were already in the 2026-11-06 sweeps)."""
    sys.path.insert(0, HERE)
    from results_analysis import load_perfect_coro
    problems, warnings = [], []
    for name, _, _, mb, refs, _ in CASES:
        rows = {r['mag']: r for r in read_csv(op.join(case_dir(name), 'selected.csv'))}
        missing = [m for m in MAGS if m not in rows]
        if missing:
            problems.append(f'{name}: no selected configuration at mag {missing}')
            continue
        floor = {'vis': np.inf, 'nir': np.inf}
        for ref in refs:
            rad, cv, cn = load_perfect_coro(ref)
            sel = (rad >= 2.0) & (rad <= 5.0)
            floor = {'vis': min(floor['vis'], float(np.mean(cv[sel]))), 'nir': min(floor['nir'], float(np.mean(cn[sel])))}
        for m in MAGS:
            r = rows[m]
            c = r[f'contrast_{mb}']
            if not np.isfinite(c) or 'diverged' in r['notes']:
                problems.append(f'{name} mag {m:g}: selected run diverged')
            if 'flux' in r['notes'] or 'ran away' in r['notes']:
                warnings.append(f'{name} mag {m:g}: {r["notes"]}')
            for b in ('vis', 'nir'):
                if r[f'contrast_{b}'] < FLOOR_RATIO * floor[b]:
                    problems.append(f'{name} mag {m:g}: contrast {b} {r[f"contrast_{b}"]:.3e} below the perfect-coronagraph floor {floor[b]:.3e}')
        for m0, m1 in zip(MAGS, MAGS[1:]):
            c0, c1 = rows[m0][f'contrast_{mb}'], rows[m1][f'contrast_{mb}']
            if min(rows[m0][f'sr_{mb}'], rows[m1][f'sr_{mb}']) >= 0.10 and np.isfinite(c0) and np.isfinite(c1) and c1 * MONOTONE_TOL < c0:
                problems.append(f'{name}: contrast {mb} better at mag {m1:g} ({c1:.3e}) than at mag {m0:g} ({c0:.3e})')
    ok = not problems
    text = '\n'.join([f'# Sanity check of the final sweep: {"PASSED" if ok else "FAILED"}', '', '## Problems', '', *(f'- {p}' for p in problems or ['none']),
                      '', '## Warnings', '', *(f'- {w}' for w in warnings or ['none'])]) + '\n'
    with open(op.join(OUT, 'check.md'), 'w') as f:
        f.write(text)
    log(f'CHECK {"PASSED" if ok else "FAILED"}: {len(problems)} problems, {len(warnings)} warnings (check.md)')
    return ok


def replay_table():
    """Mean and std over the replays of every selected configuration, then every replay run with its TN."""
    head = ['case', 'mag 900 nm', 'configuration', 'N', 'contrast 550 nm', 'contrast 1650 nm', 'SR 550 nm', 'SR 1650 nm', 'sweep contrast (metric)']
    out = ['# Final sweep: repeatability', '',
           f'{N_REPLAYS} realizations of every selected configuration, {REPLAY_TIME:g} s each: atmosphere seed 100 k + 1 and an independent uniform '
           f'random wind direction per layer (seed {REPLAY_SEED}), the same for every case; wind speeds and Kalman model unchanged. '
           'Values: mean +- std over the realizations (diverged runs excluded and counted in the notes). TN folders: <case>/replays/<TN>.', '',
           line(head + ['notes']), line(['---'] * (len(head) + 1))]
    runs = []
    for name, _, _, mb, _, replayed in CASES:
        if not replayed:
            continue
        sel = {r['mag']: r for r in read_csv(op.join(case_dir(name), 'selected.csv'))}
        rows = read_csv(op.join(case_dir(name), 'replays.csv'))
        for m in MAGS:
            rs = [r for r in rows if r['mag'] == m]
            if not rs:
                continue
            ok = [r for r in rs if np.isfinite(r['contrast_vis']) and np.isfinite(r['sr_vis'])]

            def ms(k, fmt):
                v = np.array([r[k] for r in ok])
                return f'{np.mean(v):{fmt}} +- {np.std(v, ddof=1) if len(v) > 1 else 0:{fmt}}' if len(v) else '-'
            notes = f'{len(rs) - len(ok)} diverged' if len(ok) < len(rs) else ''
            out.append(line([DESCRIPTION[name], f'{m:g}', rs[0]['config'], len(ok), ms('contrast_vis', '.3e'), ms('contrast_nir', '.3e'),
                             ms('sr_vis', '.4f'), ms('sr_nir', '.4f'), f"{sel[m][f'contrast_{mb}']:.3e}" if m in sel else '-', notes]))
            runs += [line([DESCRIPTION[name], f'{m:g}', r['realization'], r['seed'], f"{r['contrast_vis']:.3e}", f"{r['contrast_nir']:.3e}",
                           f"{r['sr_vis']:.4f}", f"{r['sr_nir']:.4f}", r['tn'], r['notes']]) for r in rs]
    h = ['case', 'mag', 'realization', 'atmo seed', 'contrast 550 nm', 'contrast 1650 nm', 'SR 550 nm', 'SR 1650 nm', 'TN', 'notes']
    out += ['', '## All replay runs', '', line(h), line(['---'] * len(h)), *runs]
    text = '\n'.join(out) + '\n'
    with open(op.join(OUT, 'replay_table.md'), 'w') as f:
        f.write(text)
    return text


def replays(gpus):
    # The first replays generate the phase screens of every seed: serially, on one GPU, so no two processes write the same screen file
    first = replay_jobs(mags=[MAGS[0]])[-1]     # a single-stage case: the fastest
    if not run_job(*first[:2], gpus[0], first[2]):
        log('CHAIN STOPPED: the phase-screen replays failed')
        return False
    failed = run_queue(replay_jobs(), gpus)
    replay_table()
    log(f'REPLAYS DONE, failed: {failed or "none"}')
    return not failed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--tables', action='store_true', help='write the tables and the check from what is on disk, and exit')
    ap.add_argument('--replay-only', action='store_true', help='skip the sweeps and the check')
    ap.add_argument('--no-replay', action='store_true', help='stop after the sweeps and the check')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.tables:
        sweep_table()
        check()
        replay_table()
        return
    if not a.replay_only:
        log(f'SWEEPS START: {len(CASES)} cases on GPUs {a.gpus}, output {OUT}')
        failed = run_queue(sweep_jobs(), a.gpus)
        sweep_table()
        log(f'SWEEPS DONE, failed: {failed or "none"}')
        if not check() or failed:
            log('CHAIN STOPPED before the replays: see check.md and the logs')
            return
        if a.no_replay:
            return
    replays(a.gpus)
    log('CHAIN DONE')


if __name__ == '__main__':
    main()
