"""WFSW products at two extra sensing wavelengths for the e2e: the vis chain (pyr_vis, source_vis) at --vis nm, the nir chain at --nir nm,
48x48, rMod 0 and 2, seeing 0.9 (at zenith: calibration.run_specula forces zenith 0), 1300 corrected modes: diffraction-limited IMs, SVD
reconstructors, aliasing run (frames, slopes, slope-null inputs, aliasing products). The pupil data (pyr_pupdata_48x48) do not depend on the
wavelength. Existing products are kept, except reconstructors at these wavelengths DERIVED by the error budget from another wavelength's IM
(header SRC_WVL != WVL_NM): they are removed first, so that the reconstructors come from the new IMs. The PC IMs are not needed by the e2e.

    python calibrate_bands.py --vis 750 --nir 950 --gpu 0
    python calibrate_bands.py --vis 750 --nir 450 --rmods 0 3 --steps im --shard 1/2 --gpu 1 --calib-dir <staging root>
"""
import argparse
import glob
import logging
import os
import os.path as op

from astropy.io import fits

import calibration


def remove_derived_recs(cfg, wavelengths):
    for wl in wavelengths:
        for path in glob.glob(op.join(cfg.calib_dir, 'rec', f'pyr*_wvl{wl:g}_*modes_rec.fits')):
            h = fits.getheader(path)
            src, own = h.get('SRC_WVL'), h.get('WVL_NM', wl)
            if src is not None and float(src) != float(own):
                logging.info('removing %s (derived from the %g nm IM)', path, float(src))
                os.remove(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vis', type=float, required=True, help='[nm] wavelength of the vis chain (unmodulated sensor of the e2e split case)')
    ap.add_argument('--nir', type=float, required=True, help='[nm] wavelength of the nir chain (modulated sensor)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--rmods', type=float, nargs='+', default=[0.0, 2.0], help='modulation radii (both chains get every one)')
    ap.add_argument('--steps', nargs='+', default=['im', 'recs', 'alias'], help="subset of im recs alias pc, in this order ('pc': the "
                    "perfect-correction IMs the error budget needs for the optical gains)")
    ap.add_argument('--shard', default='0/1', help="k/m of the im / alias / pc cases ('recs' is not sharded: run it from one process)")
    ap.add_argument('--calib-dir', help='products root (default: budget_wfsw.yml, the shared WFSW root)')
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cfg = calibration.E2EConfig(n_subaps=(48,), rMods=tuple(a.rmods), seeings=(0.9,), modes_per_subap={48: [1300]},
                                wavelengths={'vis': a.vis, 'nir': a.nir})
    if a.calib_dir:
        cfg.calib_dir = a.calib_dir
    shard = tuple(int(x) for x in a.shard.split('/'))
    if 'recs' in a.steps:
        remove_derived_recs(cfg, (a.vis, a.nir))
    for step in a.steps:
        logging.info('step %s (shard %d/%d)', step, *shard)
        calibration.STEPS[step](cfg, a.gpu, (0, 1) if step == 'recs' else shard)
    logging.info('done')


if __name__ == '__main__':
    main()
