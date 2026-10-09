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
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from astropy.io import fits
from skimage.transform import AffineTransform, warp

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
    img = np.zeros(oldpup.shape)
    center_y, center_x = img.shape[0] / 2.0, img.shape[1] / 2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    rot_and_scale = AffineTransform(rotation=rot * np.pi / 180, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x + shiftX, center_y + shiftY))
    trf = shift_to_origin + rot_and_scale + shift_to_center
    for j in range(ifunc.shape[1]):
        img[oldpup.astype(bool)] = ifunc[:, j]
        if flip:
            img = img[::-1, :]
        warp_img = warp(img, inverse_map=trf.inverse)
        ifunc_new[:, j] = warp_img[pup_mask]
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
# saving calibration products
# =============================================================================

def save_pupil(pupil_mask, fname: str, Npix: int, D: float):
    """Save a boolean pupil mask as a SPECULA Pupilstop object. `fname`
    has no extension -- `.fits` is appended."""
    simul_params = SimulParams(pixel_pupil=Npix, pixel_pitch=D / Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=pupil_mask)
    pupilstop.save(fname + ".fits")


def save_perfect_correction_vector(fname: str, dest_dir: str, full_path: str = "",
                                    Nmodes: int = 672, Ncorrmodes: Optional[int] = None) -> str:
    """Build (or copy) a per-mode PC correction vector and save it as a
    SPECULA BaseValue-tagged fits file. Empty `full_path` (the default)
    builds a perfect-correction vector: 1.0 for the first `Ncorrmodes`
    modes, 0.0 beyond -- avoids ever needing to select/deselect whether
    `scale_random` is wired in (see lbt_synim.py)."""
    if full_path:
        correction = fits.getdata(full_path)
    else:
        correction = np.zeros(Nmodes)
        correction[:Ncorrmodes] = 1.0
    filepath = Path(dest_dir) / fname
    hdr = fits.Header()
    hdr["VERSION"] = 1
    hdr["OBJ_TYPE"] = "BaseValue"
    fits.writeto(str(filepath) + ".fits", correction, hdr, overwrite=True)
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
