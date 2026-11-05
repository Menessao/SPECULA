"""One-off migration of the e2e calibration files to the aoeb layout (naming.py), so that the budget machine reads them.

    python migrate_names.py                # dry run: prints every rename / conversion / deletion and the unmatched files
    python migrate_names.py --apply

  im/pyr{r}_{n}x{n}_wvl{l}_bmc2k_vlt_im                       -> im/pyr{r}_{n}x{n}_wvl{l}_im_dl                       (+ WVL_NM)
  rec/pyr{r}_{n}x{n}_wvl{l}_bmc2k_vlt_{N}modes_rec            -> rec/pyr{r}_{n}x{n}_wvl{l}_{N}modes_rec               (+ WVL_NM)
  rec/..._{N}modes_mmse_rec  (ML with the sx-sy covariance)  -> rec/..._{N}modes_rec_ml_{nph}ph_s{s}  (SPECULA format, + header)
  aliasing/..._{Nc}corrmodes_wvl{l}_frame                     -> frames/pyr{r}_{n}x{n}_wvl{l}_s{s}_{Nc}modes_avgframe
  aliasing/..._{Nc}corrmodes_{N}modes_wvl{l}_2000Hz_alias     -> aliasing/pyr{r}_{n}x{n}_wvl{l}_s{s}_{N}modes_corr{Nc}_alias
                                                                 (the VARIANCE is converted to the std aoeb stores)
  aliasing/... at any rate but 2000 Hz (_100Hz, _200Hz, _500Hz) and ..._{N}modes_wvl{l}_2000Hz_alias (no corrmodes): deleted (the frame rate does not change the
  calibrated aliasing: independent screens every frame)
The runs were made at seeing 0.88 (ristretto_main.yml): named s0.9, header SEEING = 0.88. nph of the ML reconstructors: the sum of the
average frame they were computed with (rec_mmse with flux=None).
"""
import argparse
import os
import os.path as op
import re
import sys

import numpy as np
from astropy.io import fits

E2E_DIR = op.dirname(op.abspath(__file__))
sys.path.insert(0, op.dirname(op.dirname(E2E_DIR)))
from aoeb.naming import Paths                                  # noqa: E402
from aoeb.reconstructors import save_rec                       # noqa: E402

SEEING_TAG, SEEING = 0.9, 0.88
B = r'pyr(?P<r>[0-9.]+)_(?P<n>[0-9]+)x[0-9]+'


def plan(root, pupilstop_tag='vlt_pupil_220pixels', dm_tag='bmc2k_vlt'):
    """[(action, src, dst, info)], unmatched files."""
    P = lambda wl: Paths(root, pupilstop_tag, dm_tag).at(float(wl))
    acts, unmatched = [], []
    for f in sorted(os.listdir(op.join(root, 'im'))):
        m = re.fullmatch(B + r'_wvl(?P<w>[0-9]+)_bmc2k_vlt_im\.fits', f)
        if m:
            acts.append(('rename', op.join(root, 'im', f), P(m['w']).im_dl(float(m['r']), int(m['n'])), dict(wvl_nm=float(m['w']), dm_tag=dm_tag)))
    frames = {}
    for f in sorted(os.listdir(op.join(root, 'aliasing'))):
        src = op.join(root, 'aliasing', f)
        m = re.fullmatch(B + r'_(?P<nc>[0-9]+)corrmodes_wvl(?P<w>[0-9]+)_frame\.fits', f)
        if m:
            dst = P(m['w']).avgframe(float(m['r']), int(m['n']), SEEING_TAG, int(m['nc']))
            frames[(m['r'], m['n'], m['nc'], m['w'])] = src
            acts.append(('frame', src, dst, dict(wvl_nm=float(m['w']), seeing=SEEING)))
            continue
        m = re.fullmatch(B + r'_(?P<nc>[0-9]+)corrmodes_(?P<N>[0-9]+)modes_wvl(?P<w>[0-9]+)_2000Hz_alias\.fits', f)
        if m:
            dst = P(m['w']).alias(float(m['r']), int(m['n']), SEEING_TAG, int(m['N']), int(m['nc']))
            acts.append(('alias', src, dst, dict(wvl_nm=float(m['w']), seeing=SEEING)))
            continue
        if re.fullmatch(B + r'_.*_(?!2000Hz)[0-9]+Hz_alias\.fits', f) or re.fullmatch(B + r'_[0-9]+modes_wvl[0-9]+_2000Hz_alias\.fits', f):
            acts.append(('delete', src, None, {}))
            continue
        unmatched.append(src)
    for f in sorted(os.listdir(op.join(root, 'rec'))):
        src = op.join(root, 'rec', f)
        m = re.fullmatch(B + r'_wvl(?P<w>[0-9]+)_bmc2k_vlt_(?P<N>[0-9]+)modes_rec\.fits', f)
        if m:
            acts.append(('rename', src, P(m['w']).rec(float(m['r']), int(m['n']), int(m['N'])), dict(wvl_nm=float(m['w']), dm_tag=dm_tag,
                                                                                                      src_wvl=float(m['w']))))
            continue
        m = re.fullmatch(B + r'_wvl(?P<w>[0-9]+)_bmc2k_vlt_(?P<N>[0-9]+)modes_mmse_rec\.fits', f)
        if m:
            fr = frames.get((m['r'], m['n'], m['N'], m['w']))
            if fr is None:
                unmatched.append(src + '  (no average frame to name it)')
                continue
            frame = fits.getdata(fr)
            nph = float((frame.mean(axis=0) if frame.ndim == 3 else frame).sum())
            dst = P(m['w']).rec(float(m['r']), int(m['n']), int(m['N']), 'ml', nph, SEEING_TAG)
            acts.append(('ml', src, dst, dict(wvl_nm=float(m['w']), src_wvl=float(m['w']), dm_tag=dm_tag, method='ml', n_phot=nph, ron=0.5, f2=1.0,
                                             seeing=SEEING_TAG, xycov=True)))
            continue
        if not re.fullmatch(B + r'_wvl[0-9]+_[0-9]+modes_rec.*\.fits', f):      # already in the new layout: not listed
            unmatched.append(src)
    return acts, unmatched


def apply(acts):
    from aoeb import utils as au
    for kind, src, dst, info in acts:
        if kind == 'delete':
            os.remove(src)
            continue
        if op.exists(dst):
            print('exists, kept the old file:', dst)
            continue
        os.makedirs(op.dirname(dst), exist_ok=True)
        if kind == 'rename':
            os.rename(src, dst)
            au.set_header(dst, **info)
        elif kind == 'frame':
            f = fits.getdata(src)
            au.write_fits(dst, f.mean(axis=0) if f.ndim == 3 else f, **info)
            os.remove(src)
        elif kind == 'alias':
            v = np.asarray(fits.getdata(src), float)
            au.write_fits(dst, np.sqrt(np.var(v, axis=0) if v.ndim == 2 else v), **info)
            os.remove(src)
        elif kind == 'ml':
            save_rec(dst, fits.getdata(src), **info)
            os.remove(src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='/raid1/mmenessini/calibration/WFSW')
    ap.add_argument('--apply', action='store_true')
    ap.add_argument('-v', '--verbose', action='store_true')
    a = ap.parse_args()
    acts, unmatched = plan(a.root)
    from collections import Counter
    print('planned:', dict(Counter(k for k, *_ in acts)))
    clash = [d for _, _, d, _ in acts if d and op.exists(d)]
    print('targets already on disk (old file kept):', len(clash))
    for k, s, d, _ in (acts if a.verbose else acts[:3] + [x for x in acts if x[0] in ('ml', 'delete')][:4]):
        print(f'  {k:6s} {op.relpath(s, a.root)}  ->  {op.relpath(d, a.root) if d else "-"}')
    print('unmatched (left as they are):', len(unmatched), *[op.relpath(u, a.root) for u in unmatched[:20]], sep='\n  ')
    if a.apply:
        apply(acts)
        print('applied')


if __name__ == '__main__':
    main()
