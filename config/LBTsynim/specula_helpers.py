"""
specula_helpers.py
===================

Stateless, low-level helpers that drive SPECULA on behalf of `LBTSynIM`
(see lbt_synim.py). Nothing here knows about a particular system
(LUCIdx/LUCIsx/LBTIdx/LBTIsx) or reads LBT_synim_config.yaml -- it's pure
plumbing: pupil-plane geometric warps, saving calibration products in
SPECULA's own formats, writing "*_override" yaml files, and running the
`specula` CLI as a subprocess.

IMPORTANT -- no live SPECULA source in this environment: block/field
names used here (Pupilstop, SimulParams, the "*_override" yaml merge
convention, toccd's signature) are ported from the reference scripts
provided during development, not verified against SPECULA's current
source. See the main lbt_synim.py docstring for the full caveat list.
"""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from astropy.io import fits
from skimage.transform import AffineTransform, warp

from specula import cpuArray
from specula.data_objects.pupdata import PupData
from specula.data_objects.simul_params import SimulParams
from specula.data_objects.pupilstop import Pupilstop
from specula.lib.toccd import toccd  # noqa: F401  (re-exported for lbt_synim.py)


# =============================================================================
# geometric warps (registration)
# =============================================================================

def warp_image(ifunc, pupmask, flip: bool = False,
                shiftX: float = 0.0, shiftY: float = 0.0,
                rot: float = 0.0, mag: float = 1.0, oldpup=None):
    """Warp a set of influence-function-like columns onto a new pupil mask
    (rotation + shift + magnification; no shear)."""
    pup_mask = pupmask.astype(bool)
    ifunc_new = np.zeros([int(np.sum(pup_mask)), ifunc.shape[1]])
    center_y, center_x = oldpup.shape[0] / 2.0, oldpup.shape[1] / 2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    rot_and_scale = AffineTransform(rotation=rot * np.pi / 180, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x + shiftX, center_y + shiftY))
    trf = shift_to_origin + rot_and_scale + shift_to_center
    for j in range(ifunc.shape[1]):
        # Fresh image per mode: re-using one image across modes while
        # flipping it (img = img[::-1]) leaks the previous mode's values
        # into the next whenever the pupil isn't exactly up/down
        # symmetric -- i.e. it corrupted every mode but the first for
        # flip=True systems (LBTIdx, LUCIsx).
        img = np.zeros(oldpup.shape)
        img[oldpup.astype(bool)] = ifunc[:, j]
        if flip:
            img = img[::-1, :]
        ifunc_new[:, j] = warp(img, inverse_map=trf.inverse)[pup_mask]
    return ifunc_new


def warp_mask(pup, shiftX: float = 0.0, shiftY: float = 0.0,
              mag: float = 1.0, rot: float = 0.0):
    """Warp a pupil mask the same way (rotation + shift + magnification;
    no shear)."""
    center_y, center_x = pup.shape[0] / 2.0, pup.shape[1] / 2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    rot_and_scale = AffineTransform(rotation=rot * np.pi / 180, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x + shiftX, center_y + shiftY))
    trf = shift_to_origin + rot_and_scale + shift_to_center
    warp_pup = warp(pup.astype(float), inverse_map=trf.inverse) > 0.9
    return warp_pup.astype(float)


# =============================================================================
# pupil data
# =============================================================================

def pup_ids_from_pupdata(pupdata: Union[PupData, str, Path], n_pupils: int = 2) -> np.ndarray:
    """Pixel indices ("pup_ids") of the pupils the slopes are rastered into,
    from a SPECULA `PupData` object or the path of a saved one.

    `PupData.ind_pup` is `[n_subap, 4]`: flat indices into the detector
    frame (`pupdata.framesize`), one column per pyramid pupil, padded with
    -1 where pupils have different pixel counts. `LBTSynIM._raster` places
    the x slopes at column 0 and the y slopes at column 1, so by default
    this returns just those first two columns.

    Rows with any -1 padding in the returned columns are dropped (with a
    warning): `np.put` with an index of -1 doesn't skip it, it silently
    writes into the last pixel of the frame.

    Returns an int array `[n_valid, n_pupils]`.
    """
    if not isinstance(pupdata, PupData):
        pupdata = PupData.restore(str(pupdata))
    ind = np.asarray(cpuArray(pupdata.ind_pup))[:, :n_pupils].astype(int)
    valid = np.all(ind >= 0, axis=1)
    if not valid.all():
        warnings.warn(f"pup_ids_from_pupdata: dropped {int((~valid).sum())} of {len(ind)} rows "
                      f"padded with -1 in the first {n_pupils} pupils.")
    return ind[valid]


# =============================================================================
# saving calibration products
# =============================================================================

def save_pupil(pupil_mask, fname: str, Npix: int, D: float):
    """Save a boolean pupil mask as a SPECULA Pupilstop object. `fname`
    has no extension -- `.fits` is appended."""
    simul_params = SimulParams(pixel_pupil=Npix, pixel_pitch=D / Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=pupil_mask)
    pupilstop.save(fname + ".fits")


def save_perfect_correction_vector(fname: str, dest_dir: str, Nmodes: int = 672,
                                    Ncorrmodes: Optional[int] = None) -> str:
    """Save a perfect-correction vector (1.0 for the first `Ncorrmodes`
    modes, 0.0 up to `Nmodes`; all ones if `Ncorrmodes` is None) as a SPECULA
    BaseValue-tagged fits file `dest_dir/fname.fits`. This is only the
    fallback for when no correction vector file is configured -- a
    user-provided vector is used as-is, never copied through here."""
    correction = np.zeros(Nmodes)
    correction[:Ncorrmodes] = 1.0
    hdr = fits.Header()
    hdr["VERSION"] = 1
    hdr["OBJ_TYPE"] = "BaseValue"
    fits.writeto(str(Path(dest_dir) / fname) + ".fits", correction, hdr, overwrite=True)
    return fname


# =============================================================================
# running SPECULA
# =============================================================================

def write_overrides_yaml(overrides: dict, path: Union[str, Path]):
    """Write a SPECULA "*_override" yaml file from a plain nested dict
    (each top-level key gets an "_override" suffix)."""
    payload = {f"{k}_override": v for k, v in overrides.items()}
    with open(str(path) + ".yml", "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)


def run_specula(main_yaml: Union[str, Path], override_path: Union[str, Path],
                 extra_yaml: Optional[Union[str, Path]] = None):
    """Run the `specula` CLI: main config, optional extra static yaml
    (e.g. the PC blocks), then the override yaml written by
    `write_overrides_yaml` (same `override_path`, ".yml" appended)."""
    cmd = ["specula", str(main_yaml)]
    if extra_yaml:
        cmd.append(str(extra_yaml))
    cmd.append(str(override_path) + ".yml")
    subprocess.run(cmd, check=True)
