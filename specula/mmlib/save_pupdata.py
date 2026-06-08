import numpy as np
import os

import specula
specula.init(0)

from astropy.io import fits
from specula.data_objects.pupdata import PupData


def pupdata_from_boolean_mask(mask: np.ndarray, radius: float = None) -> PupData:
    """
    Create a `PupData` object from a boolean pupils mask.

    Parameters
    ----------
    mask : ndarray (2D, bool)
        Boolean mask with pupil pixels set to True. Expected to contain four
        pupils arranged in quadrants (top-left, top-right, bottom-left, bottom-right).
    radius : float or None
        If provided, used as the radius for all pupils. If None, radius for each
        pupil is estimated from the pixel area as sqrt(Npix/pi).

    Returns
    -------
    PupData
        Initialized PupData instance with `ind_pup`, `radius`, `cx`, `cy`, and `framesize`.

    Notes
    -----
    The function splits the image in four quadrant regions and computes the
    centroid and pixel indices for each quadrant. If a quadrant contains no
    True pixels, its centroid is set to the quadrant center and indices left empty.
    """

    if mask.ndim != 2:
        raise ValueError('mask must be a 2D array')

    h, w = mask.shape
    cy, cx = h // 2, w // 2

    # Define quadrant slices: TL, TR, BL, BR
    quads = [ (slice(0, cy), slice(0, cx)),
              (slice(0, cy), slice(cx, w)),
              (slice(cy, h), slice(0, cx)),
              (slice(cy, h), slice(cx, w)) ]

    centers = []
    radii = []
    indices_list = []

    for ys, xs in quads:
        sub = mask[ys, xs]
        if np.any(sub):
            ys_idx, xs_idx = np.nonzero(sub)
            # convert to full-image coordinates
            full_y = ys_idx + (ys.start or 0)
            full_x = xs_idx + (xs.start or 0)
            # centroid
            x_center = full_x.mean()
            y_center = full_y.mean()
            centers.append((x_center, y_center))
            # pixel indices (linear)
            linear = (full_y * w + full_x).astype(int)
            indices_list.append(linear)
            # estimate radius if not provided
            if radius is None:
                radii.append(np.sqrt(linear.size / np.pi))
            else:
                radii.append(float(radius))
        else:
            # fallback to quadrant center
            x_center = ((xs.start or 0) + (xs.stop or w)) / 2.0
            y_center = ((ys.start or 0) + (ys.stop or h)) / 2.0
            centers.append((x_center, y_center))
            indices_list.append(np.array([], dtype=int))
            radii.append(float(radius) if radius is not None else 0.0)

    # Build ind_pup array with -1 padding like other code expects
    max_pixels = max((arr.size for arr in indices_list), default=0)
    ind_pup = -1 * np.ones((max_pixels, 4), dtype=int)
    for i, arr in enumerate(indices_list):
        if arr.size > 0:
            ind_pup[:arr.size, i] = arr

    # Unpack centers into cx, cy arrays
    cx_arr = np.array([c[0] for c in centers], dtype=float)
    cy_arr = np.array([c[1] for c in centers], dtype=float)
    radii_arr = np.array(radii, dtype=float)

    pup = PupData(ind_pup=ind_pup, radius=radii_arr, cx=cx_arr, cy=cy_arr, framesize=(h, w))
    return pup


def save_lbt_pupil(destination_dir:str='/raid1/mmenessini/calibration/SOUL/KLv30dx/pupils'):
    pupil = fits.getdata(os.path.join(destination_dir, 'lbt_pupmask.fits'))
    os.makedirs(destination_dir,exist_ok=True)
    fname = os.path.join(destination_dir, 'lbt_pupdata.fits')
    pupilstop = pupdata_from_boolean_mask(pupil,radius=20)
    pupilstop.save(fname,overwrite=True)

if __name__ == "__main__":
    save_lbt_pupil()