"""Does the aliasing variance per frame follow the temporal model of the budget (A10, aoeb.filter_optimizers.aliasing_frame_fraction)?
Frozen-flow run (calib_alias_frozen.yml), N modes removed exactly, frames of one time step binned to longer exposures; variance of
R @ slopes against time for each exposure, relative to one step, against the model. Results: <out>/results.txt (written after each case).
    python test_alias_rate.py [--gpu 1] [--time 3.0] [--seeing 0.9]"""
import argparse
import os
import os.path as op
import shutil
import sys

import numpy as np
from astropy.io import fits

import calibration as c
from aoeb.filter_optimizers import aliasing_frame_fraction
from aoeb.utils import effective_wind, radial_order

WFSW = '/raid1/mmenessini/calibration/WFSW'
OUT = '/raid1/mmenessini/results/WFSW/alias_rate_test'
WIND = effective_wind([6.6, 5.9, 5.1, 4.5, 5.1, 8.3, 16.3, 30.2, 34.3, 17.5], [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05])
CASES = ((2.0, 10, 54), (2.0, 24, 300), (3.0, 48, 1300))
BINS = (1, 2, 4, 10, 20)                 # exposure in time steps: 2000, 1000, 500, 200, 100 Hz


def rec_file(r, n, N, wl):
    """The reconstructor, under its new name or (before the migration) the old one."""
    new = f'{WFSW}/rec/pyr{r:.1f}_{n}x{n}_wvl{wl}_{N}modes_rec.fits'
    return new if op.exists(new) else f'{WFSW}/rec/pyr{r:.1f}_{n}x{n}_wvl{wl}_bmc2k_vlt_{N}modes_rec.fits'


def slopes_of(frames, pup):
    """PyrSlopec slopes (no slope nulls), frames (T, ny, nx)."""
    f = frames.reshape(len(frames), -1).astype(np.float64)
    A, B, C, D = (f[:, pup[:, k][pup[:, k] >= 0]] for k in range(4))
    norm = (A + B + C + D).sum(axis=1, keepdims=True) / pup.shape[0]
    return np.hstack([A + B - C - D, B + C - A - D]) / norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--time', type=float, default=3.0)
    ap.add_argument('--seeing', type=float, default=0.9)
    a = ap.parse_args()
    root = op.join(OUT, 'root')
    os.makedirs(op.join(root, 'phasescreens'), exist_ok=True)
    for d in ('pupils', 'ifunc', 'm2c', 'pupilstop'):
        if not op.exists(op.join(root, d)):
            os.symlink(op.join(WFSW, d), op.join(root, d))
    cfg = c.E2EConfig(calib_dir=root)
    dt = cfg.time_step
    out = open(op.join(OUT, 'results.txt'), 'a')
    def say(*x):
        print(*x, file=out, flush=True)
        print(*x, flush=True)
    say(f'# frozen flow, seeing {a.seeing}, {a.time} s, wind {WIND:.2f} m/s; ratio = per-frame variance / variance at one step; model = aliasing_frame_fraction ratio')
    for r, n, N in CASES:
        recs = {wl: np.asarray(fits.getdata(rec_file(r, n, N, wl))) for wl in (900, 1300)}   # read before the migration may rename them
        pup = np.asarray(fits.getdata(f'{WFSW}/pupils/pyr_pupdata_{n}x{n}.fits', 1))
        store = op.join(root, 'scratch', f'frozen_{r}_{n}_{N}')
        ov = c.merge(c._pyrs(cfg, r, n), {'main': {'total_time': a.time}, 'seeing': {'constant': a.seeing}, 'atmo_modes': {'nmodes': N},
                                          'data_store': {'store_dir': store, 'create_tn': False}})
        c.run_specula(cfg, ['calib_alias_frozen.yml'], ov, a.gpu, f'frozen_{r}_{n}x{n}_{N}')
        order = int(radial_order(N))
        for name, wl in (('vis', 900), ('nir', 1300)):
            frames = np.asarray(fits.getdata(op.join(store, f'{name}_frames.fits')))
            frames = frames[np.any(frames != 0, axis=(1, 2))]
            base = None
            for k in BINS:
                m = len(frames) // k
                binned = frames[:m * k].reshape(m, k, *frames.shape[1:]).mean(axis=1)
                modes = recs[wl] @ slopes_of(binned, pup).T
                v = np.var(modes, axis=1)
                if k == 1:
                    base, f1 = v, aliasing_frame_fraction(1 / dt, order, WIND, 8.2)
                fk = aliasing_frame_fraction(1 / (k * dt), order, WIND, 8.2)
                third = N // 3
                say(f'pyr{r} {n}x{n} N{N} {wl} nm {1 / (k * dt):6.0f} Hz ({m:5d} frames): total {v.sum() / base.sum():.3f}  model {fk / f1:.3f} | '
                    f'first third {v[:third].sum() / base[:third].sum():.3f}  last third {v[-third:].sum() / base[-third:].sum():.3f} | var(1 step) {base.sum():.4g}')
        shutil.rmtree(store, ignore_errors=True)
    say('# done')


if __name__ == '__main__':
    main()
