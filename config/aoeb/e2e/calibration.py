"""Calibration of the e2e system (ristretto_main.yml: VIS and NIR pyramids) in the aoeb layout (aoeb/naming.py), so that the budget
machine reads the products like any aoeb root (aoeb/config/budget_wfsw.yml, machine()). The sensors, seeings, wavelengths and tags come from that file. Used by main_simul.ipynb and runnable offline:

    python calibration.py pc --gpu 0 --shard 0/2                          # one shard of 2 (start one process per shard)
    setsid nohup python calibration.py pc --gpu 1 --shard 1/2 > pc_1.log 2>&1 < /dev/null &   # detached: survives the session
    python calibration.py all --seeings 0.7 0.9 1.1                       # every step, in order
    python calibration.py --list                                          # the cases of every step and whether they are done

Steps (existing products are skipped, so an interrupted run is resumed by starting it again):
  pupdata  pupils/pyr_pupdata_{n}x{n}                                  (calib_pupdata.yml)
  im       im/<base>_im_dl  per sensing wavelength, pyr_vis             (calib_im.yml)
  recs     rec/<base>_{N}modes_rec  SVD                                (aoeb.reconstructors)
  alias    aliasing/, frames/, slopenulls/, throughput/ per (rMod, n, seeing, N corrected), both pyramids in one run
           (calib_wfs_aliasing.yml; aoeb.generation.process_aliasing_runs / compute_alias_product)
  pc       im/<base>_s{seeing}_{N}modes_im_pc  both pyramids in one run of `n_screens` push-pull cycles, one screen each (calib_pc.yml)
  ml       rec/<base>_{N}modes_rec_ml_{nph}ph_s{seeing}  ML with the sx-sy noise covariance (machine.ensure_rec)
<base> = pyr{rMod}_{n}x{n}_wvl{nm}. Both pyramids of a run get the same rMod; the VIS one senses at wavelengths['vis'], the NIR one at
wavelengths['nir']. Every calibration camera integrates ONE time step (cred_*.dt = time_step: the push-pull IMs need it, the loop
uses cred_vis.dt = 4 steps). Every product carries WVL_NM (and SEEING) in its header.
"""
import argparse
import glob
import logging
import math
import os
import os.path as op
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import yaml

E2E_DIR = op.dirname(op.abspath(__file__))
sys.path.insert(0, op.dirname(op.dirname(E2E_DIR)))          # config/: the aoeb package
from aoeb import utils as au                                   # noqa: E402
from aoeb.generation import compute_alias_product, merge, process_aliasing_runs, save_correction_vector, write_overrides_yaml  # noqa: E402
from aoeb.naming import Paths, snap_seeing                     # noqa: E402
from aoeb.reconstructors import compute_reconstructor, save_rec  # noqa: E402

log = logging.getLogger('e2e.calibration')
BUDGET_YML = op.join(op.dirname(E2E_DIR), 'config', 'budget_wfsw.yml')
with open(BUDGET_YML) as _f:
    _STUDY = yaml.safe_load(_f)
_SPECULA = op.join(op.dirname(sys.executable), 'specula')
SPECULA = _SPECULA if op.exists(_SPECULA) else (shutil.which('specula') or 'specula')
PYRS = ('vis', 'nir')


@dataclass
class E2EConfig:
    calib_dir: str = _STUDY['paths']['root_dir']
    n_subaps: tuple = tuple(_STUDY['sensors']['n_subaps'])
    rMods: tuple = tuple(_STUDY['sensors']['rMods'])
    modes_per_subap: dict = field(default_factory=lambda: dict(_STUDY['sensors']['modes_per_subap']))
    wavelengths: dict = field(default_factory=lambda: dict(zip(PYRS, _STUDY['wavelengths']['sensing_nm'])))   # vis, nir
    seeings: tuple = tuple(_STUDY['scan']['seeings'])
    time_step: float = 5e-4
    alias_time: float = 1.0                  # [s] aliasing run: one independent screen per step (2000 frames)
    n_screens: int = 70                      # PC IM: push-pull cycles, one screen each, averaged
    n_modes_im: int = 1800                   # columns of the diffraction-limited IM
    pupilstop_tag: str = _STUDY['paths']['pupilstop_tag']
    dm_tag: str = _STUDY['paths']['dm_tag']
    main_yml: str = _STUDY['paths']['main']
    seed: int = 1
    overwrite: bool = False

    def paths(self, wvl):
        return Paths(self.calib_dir, self.pupilstop_tag, self.dm_tag).at(float(wvl))

    def pup_dist(self, n):
        return float(max(14, math.ceil(n * 60 / 48)))

    def mode_list(self, n):
        return sorted(int(x) for x in self.modes_per_subap[n])

    def sensors(self):
        return [(float(r), int(n)) for n in self.n_subaps for r in self.rMods]

    def cases(self):
        """(rMod, n, seeing, N corrected) of the aliasing and PC runs, cheapest first."""
        out = [(r, n, snap_seeing(s), N) for r, n in self.sensors() for s in self.seeings for N in self.mode_list(n)]
        return sorted(out, key=lambda c: (c[3], c[1], c[0], c[2]))


# ----------------------------------------------------------------------------- SPECULA
def run_specula(cfg, extra_ymls, overrides, gpu=0, tag='run'):
    """specula main + extra ymls + a unique override file; stdout / stderr -> <calib_dir>/logs/<tag>.log (kept on failure)."""
    ov = merge({'main': {'root_dir': cfg.calib_dir, 'time_step': cfg.time_step},
                'cred_vis': {'dt': cfg.time_step}, 'cred_nir': {'dt': cfg.time_step}}, overrides)
    logs, scratch = op.join(cfg.calib_dir, 'logs'), op.join(cfg.calib_dir, 'scratch')
    os.makedirs(logs, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)
    ymls = [op.join(E2E_DIR, y) for y in [cfg.main_yml, *extra_ymls]]
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        path = op.join(tmp, 'overrides')
        write_overrides_yaml(ov, path)
        log_path = op.join(logs, f'{tag}.log')
        t0 = time.time()
        with open(log_path, 'w') as f:
            r = subprocess.run([SPECULA, '--target', str(gpu), *ymls, f'{path}.yml'], cwd=E2E_DIR, stdout=f, stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f'SPECULA failed ({r.returncode}): see {log_path}')
        log.info('%s: %.0f s', tag, time.time() - t0)


def _pyrs(cfg, rMod, n, which=PYRS):
    """Geometry and wavelength of the pyramids (each pyramid and its source get the same wavelength)."""
    ov = {}
    for p in which:
        wl = float(cfg.wavelengths[p])
        ov[f'pyr_{p}'] = {'pup_diam': float(n), 'pup_dist': cfg.pup_dist(n), 'mod_amp': float(rMod), 'wavelengthInNm': wl}
        ov[f'source_{p}'] = {'wavelengthInNm': wl}
        ov[f'pyr_slopes_{p}'] = {'pupdata_object': f'pyr_pupdata_{n}x{n}'}
    return ov


def _correction_vector(cfg, N):
    """Stem of the perfect-correction vector of N modes in <calib_dir>/data, written once and atomically (shards run in parallel and
    SPECULA may be reading it)."""
    stem = f'correction_vector_{N}modes_c1.00-1.00'
    path = op.join(cfg.calib_dir, 'data', stem + '.fits')
    if not op.exists(path):
        os.makedirs(op.join(cfg.calib_dir, 'scratch'), exist_ok=True)
        with tempfile.TemporaryDirectory(dir=op.join(cfg.calib_dir, 'scratch')) as tmp:
            os.replace(op.join(tmp, 'data', save_correction_vector(tmp, 1.0, 1.0, N) + '.fits'), path)
    return stem


def _needed(cfg, *paths):
    return cfg.overwrite or not all(op.exists(p) for p in paths)


def _shard(items, shard):
    k, m = shard
    return [x for i, x in enumerate(items) if i % m == k]


# ----------------------------------------------------------------------------- steps
def calibrate_pupdata(cfg, gpu=0, shard=(0, 1)):
    for n in _shard(list(cfg.n_subaps), shard):
        path = cfg.paths(cfg.wavelengths['vis']).pupdata(n)
        if _needed(cfg, path):
            run_specula(cfg, ['calib_pupdata.yml'], {'pyr_vis': {'pup_diam': float(n), 'pup_dist': cfg.pup_dist(n)},
                                                     'pyr_pupdata': {'output_tag': f'pyr_pupdata_{n}x{n}'}}, gpu, f'pupdata_{n}x{n}')


def calibrate_im(cfg, gpu=0, shard=(0, 1)):
    """Diffraction-limited IM of every sensor at every sensing wavelength (pyr_vis set to that wavelength)."""
    jobs = [(wl, r, n) for wl in sorted(set(cfg.wavelengths.values())) for r, n in cfg.sensors()]
    for wl, r, n in _shard(jobs, shard):
        path = cfg.paths(wl).im_dl(r, n)
        if not _needed(cfg, path):
            continue
        ov = _pyrs(cfg, r, n, ('vis',))
        ov['pyr_vis']['wavelengthInNm'] = ov['source_vis']['wavelengthInNm'] = float(wl)
        ov.update({'pyr_im_calibrator': {'im_tag': op.basename(path)[:-5], 'nmodes': cfg.n_modes_im},
                   'pushpull': {'nmodes': cfg.n_modes_im}, 'main': {'total_time': cfg.n_modes_im * 2 * cfg.time_step}})
        run_specula(cfg, ['calib_im.yml'], ov, gpu, op.basename(path)[:-5])
        au.set_header(path, wvl_nm=float(wl), dm_tag=cfg.dm_tag)


def compute_recs(cfg, gpu=0, shard=(0, 1)):
    for wl in sorted(set(cfg.wavelengths.values())):
        P = cfg.paths(wl)
        for r, n in cfg.sensors():
            for N in cfg.mode_list(n):
                path = P.rec(r, n, N)
                if _needed(cfg, path) and op.exists(P.im_dl(r, n)):
                    save_rec(path, compute_reconstructor(au.read_fits(P.im_dl(r, n)), N), wvl_nm=float(wl), src_wvl=float(wl), dm_tag=cfg.dm_tag)


def calibrate_alias(cfg, gpu=0, shard=(0, 1)):
    """Aliasing run (independent screens, N modes removed exactly) of both pyramids; products of each pyramid at its wavelength,
    aliasing of every reconstructor with N_rec <= N. Cases already on disk (also those renamed from the old e2e names, which have
    no slopes) are skipped."""
    names = {p: (f'slopes_{p}', f'{p}_frames', 'turb_modes') for p in PYRS}
    for r, n, s, N in _shard(cfg.cases(), shard):
        Ps = {p: cfg.paths(cfg.wavelengths[p]) for p in PYRS}
        done = all(op.exists(P.slopes(r, n, s, N)) or (op.exists(P.avgframe(r, n, s, N)) and op.exists(P.alias(r, n, s, N, N)))
                   for P in Ps.values())
        if done and not cfg.overwrite:
            continue
        tag = Ps['vis'].sim(r, n, s, N).replace('_wvl', '_alias_wvl')
        store = op.join(cfg.calib_dir, 'scratch', tag)
        run_specula(cfg, ['calib_wfs_aliasing.yml'], merge(_pyrs(cfg, r, n), {
            'main': {'total_time': cfg.alias_time}, 'seeing': {'constant': float(s)}, 'atmo_modes': {'nmodes': N},
            'atmo_random': {'seed': cfg.seed}, 'data_store': {'store_dir': store, 'create_tn': False}}), gpu, tag)
        for p, P in Ps.items():
            process_aliasing_runs(P, r, n, s, N, [store], names=names[p])
            for N_rec in cfg.mode_list(n):
                if N_rec <= N:
                    if not op.exists(P.rec(r, n, N_rec)):
                        raise FileNotFoundError(f'{P.rec(r, n, N_rec)} missing: run the recs step first')
                    compute_alias_product(P, r, n, s, N_rec, N)
        shutil.rmtree(store, ignore_errors=True)


def calibrate_pc(cfg, gpu=0, shard=(0, 1)):
    """PC IM of both pyramids: one run of n_screens push-pull cycles, a new screen every cycle (calib_pc.yml)."""
    for r, n, s, N in _shard(cfg.cases(), shard):
        Ps = {p: cfg.paths(cfg.wavelengths[p]) for p in PYRS}
        paths = {p: P.im_pc(r, n, s, N) for p, P in Ps.items()}
        if not _needed(cfg, *paths.values()):
            continue
        vec = _correction_vector(cfg, N)
        ov = merge(_pyrs(cfg, r, n), {
            'main': {'total_time': 2 * N * cfg.n_screens * cfg.time_step},
            'seeing_random': {'constant': float(s)}, 'atmo_random': {'update_interval': 2 * N, 'seed': cfg.seed},
            'modal_analysis_random': {'nmodes': N}, 'scale_random': {'constant_mul_data': vec}, 'dm_random': {'nmodes': N},
            'dm': {'nmodes': N}, 'pushpull': {'nmodes': N, 'ncycles': cfg.n_screens}},
            {f'pyr_im_calibrator_{p}': {'im_tag': op.basename(path)[:-5], 'nmodes': N} for p, path in paths.items()})
        run_specula(cfg, ['calib_pc.yml'], ov, gpu, op.basename(paths['vis'])[:-5])
        for p, path in paths.items():
            au.set_header(path, wvl_nm=float(cfg.wavelengths[p]), seeing=float(s), n_screen=cfg.n_screens, dm_tag=cfg.dm_tag)


def avgframe_steps(frame_path):
    """Exposure of a saved average frame, in loop steps. The frames renamed from the old e2e (s0.9) are longer exposures than one step
    (2 ms or 1 ms against 0.5 ms) and nothing in the file says so: measured against the median flux of the same case at all its seeings."""
    pattern = op.join(op.dirname(frame_path), re.sub(r'_s[0-9.]+_', '_s*_', op.basename(frame_path)))
    fluxes = [float(au.read_fits(p).sum()) for p in glob.glob(pattern)]
    return max(1, round(float(au.read_fits(frame_path).sum()) / float(np.median(fluxes))))


def compute_ml(cfg, gpu=0, shard=(0, 1), rMods=(0.0,), n_phot=None):
    """ML reconstructors (sx-sy noise covariance, aoeb machine.ensure_rec) of the sensors with rMod in `rMods`; n_phot: detected
    electrons per frame of the noise weights (None: those of the average frame of the aliasing run, as the old e2e `mmse` recs)."""
    m = machine(cfg)
    for wl in sorted(set(cfg.wavelengths.values())):
        P = cfg.paths(wl)
        for r, n in cfg.sensors():
            if float(r) not in {float(x) for x in rMods}:
                continue
            for s in cfg.seeings:
                for N in cfg.mode_list(n):
                    frame = P.avgframe(r, n, snap_seeing(s), N)
                    if op.exists(frame):
                        nph = float(au.read_fits(frame).sum()) / avgframe_steps(frame) if n_phot is None else float(n_phot)
                        m.ensure_rec(r, n, N, ('ml', nph), s, wl, overwrite=cfg.overwrite)


STEPS = {'pupdata': calibrate_pupdata, 'im': calibrate_im, 'recs': compute_recs, 'alias': calibrate_alias, 'pc': calibrate_pc,
         'ml': compute_ml}


def machine(cfg=None, **kw):
    """aoeb ErrorBudgetMachine of the e2e products (budget_wfsw.yml, root = cfg.calib_dir)."""
    from aoeb.config import Config
    cfg = cfg or E2EConfig()
    return Config.load(BUDGET_YML, [f'paths.specula_dir={E2E_DIR}', f'paths.root_dir={cfg.calib_dir}']).machine(**kw)


def status(cfg):
    """{step: (done, total)} of the per-case steps."""
    out = {}
    for name, test in (('alias', lambda P, c: op.exists(P.avgframe(*c)) and op.exists(P.alias(*c, c[3]))),
                       ('pc', lambda P, c: op.exists(P.im_pc(*c)))):
        cases = cfg.cases()
        done = sum(all(test(cfg.paths(cfg.wavelengths[p]), c) for p in PYRS) for c in cases)
        out[name] = (done, len(cases))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', help=f'{list(STEPS)} or all')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', default='0/1', help='k/m: run the cases i with i %% m == k (one process per GPU)')
    ap.add_argument('--seeings', type=float, nargs='+')
    ap.add_argument('--n-subaps', type=int, nargs='+')
    ap.add_argument('--rmods', type=float, nargs='+')
    ap.add_argument('--n-screens', type=int)
    ap.add_argument('--calib-dir')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the status of the alias and pc cases and exit')
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cfg = E2EConfig()
    for k, v in (('seeings', a.seeings), ('n_subaps', a.n_subaps), ('rMods', a.rmods), ('n_screens', a.n_screens),
                 ('calib_dir', a.calib_dir)):
        if v is not None:
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg.overwrite = a.overwrite
    if a.list:
        for k, (d, t) in status(cfg).items():
            print(f'{k}: {d}/{t} cases done')
        return
    steps = list(STEPS) if a.steps in (['all'], []) else a.steps
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        ap.error(f'unknown steps {unknown}')
    shard = tuple(int(x) for x in a.shard.split('/'))
    for s in steps:
        log.info('step %s (gpu %d, shard %d/%d)', s, a.gpu, *shard)
        STEPS[s](cfg, a.gpu, shard)
    log.info('done')


if __name__ == '__main__':
    main()
