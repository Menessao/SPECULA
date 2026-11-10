"""WFSW products at two extra sensing wavelengths for the e2e: the vis chain (pyr_vis, source_vis) at --vis nm, the nir chain at --nir nm,
48x48, rMod 0 and 2, seeing 0.9 (at zenith: calibration.run_specula forces zenith 0), 1300 corrected modes: diffraction-limited IMs, SVD
reconstructors, aliasing run (frames, slopes, slope-null inputs, aliasing products). The pupil data (pyr_pupdata_48x48) do not depend on the
wavelength. Existing products are kept, except reconstructors at these wavelengths DERIVED by the error budget from another wavelength's IM
(header SRC_WVL != WVL_NM): they are removed first, so that the reconstructors come from the new IMs. The PC IMs are not needed by the e2e.

    python calibrate_bands.py --vis 750 --nir 950 --gpu 0
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
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
    cfg = calibration.E2EConfig(n_subaps=(48,), rMods=(0.0, 2.0), seeings=(0.9,), modes_per_subap={48: [1300]},
                                wavelengths={'vis': a.vis, 'nir': a.nir})
    remove_derived_recs(cfg, (a.vis, a.nir))
    for step in ('im', 'recs', 'alias'):
        logging.info('step %s', step)
        calibration.STEPS[step](cfg, a.gpu, (0, 1))
    logging.info('done')


if __name__ == '__main__':
    main()
