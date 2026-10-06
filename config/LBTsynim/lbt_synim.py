"""
lbt_synim.py
============

Standalone utility class to simulate LBT (Large Binocular Telescope) AO
interaction matrices and reconstructors for the 4 available systems:

    LUCIdx, LUCIsx, LBTIdx, LBTIsx

built on top of the SPECULA end-to-end AO simulation package
(https://github.com/Menessao/SPECULA/tree/xao).

This generalises three scripts that used to be hand-tuned per system/side:

    * synim_sprint.py       -> LBTSynIM.update_registration
    * compute_lbt_recdx.py  -> LBTSynIM.compute_reconstructor
    * compute_lbt_recsx.py  -> LBTSynIM.compute_reconstructor

Typical usage
-------------
    from lbt_synim import LBTSynIM

    lucidx = LBTSynIM('LUCIdx')
    misreg = lucidx.update_registration(meas_imat)
    imat   = lucidx.compute_interaction_matrix(rMod=2, seeing=1.0)
    rec    = lucidx.compute_reconstructor(imat, Nmodes=550)

"""

from __future__ import annotations

import datetime
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import os.path as op

import numpy as np
import yaml
from astropy.io import fits
from skimage.transform import AffineTransform, warp
import matplotlib.pyplot as plt
from scipy.io import readsav

import specula
specula.init(0)

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv

from specula.data_objects.pupilstop import Pupilstop
from specula.data_objects.simul_params import SimulParams

# from specula.lib.toccd import toccd


VALID_SYSTEMS = ("LUCIdx", "LUCIsx", "LBTIdx", "LBTIsx")


# =============================================================================
# small module-level helpers
# =============================================================================

def _tn_now() -> str:
    """Tracking-number timestamp, e.g. '20260927_153000'."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _side_for(system: str) -> str:
    return "dx" if system.endswith("dx") else "sx"


def save_pupil(pupil_mask, fname:str, Npix:int, D:float):
    simul_params = SimulParams(pixel_pupil=Npix,pixel_pitch=D/Npix)
    pupilstop = Pupilstop(simul_params=simul_params, input_mask=pupil_mask)
    pupilstop.save(fname+'.fits')


def save_perfect_correction_vector(fname:str, dest_dir:str, full_path:str='', Nmodes: int = 672, Ncorrmodes: int = None):
    """
    Generates a correction vector with logarithmic scaling to maintain 
    constant power-law slopes in residual turbulence PSDs.
    """
    if full_path == '':
        correction = np.zeros(Nmodes)
        correction[:Ncorrmodes] = np.ones(Ncorrmodes)
    else:
        correction = fits.getdata(full_path)
    filepath = op.join(dest_dir,fname)
    hdr = fits.Header()
    hdr['VERSION'] = 1
    hdr['OBJ_TYPE'] = 'BaseValue'
    fits.writeto(filepath+'.fits', correction, hdr, overwrite=True)
    print(f'Saved {filepath}')
    return fname
    

def warp_image(ifunc, pupmask, flip: bool = False,
               shiftX: float = 0.0, shiftY: float = 0.0,
               rot: float = 0.0, mag: float = 1.0, oldpup=None):
    """Warp a set of influence-function-like columns onto a new pupil mask.

    Direct port of ``warp_image`` from synim_sprint.py, with the shear
    terms removed.
    """
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
    """Warp a pupil mask. Direct port of ``warp_mask`` (shear removed)."""
    center_y, center_x = pup.shape[0] / 2.0, pup.shape[1] / 2.0
    shift_to_origin = AffineTransform(translation=(-center_x, -center_y))
    rot_and_scale = AffineTransform(rotation=rot * np.pi / 180, scale=mag)
    shift_to_center = AffineTransform(translation=(center_x + shiftX, center_y + shiftY))
    trf = shift_to_origin + rot_and_scale + shift_to_center
    warp_pup = warp(pup.astype(float), inverse_map=trf.inverse) > 0.9
    return warp_pup.astype(float)


@dataclass
class RegistrationResult:
    """Result of a successful `update_registration()` call."""
    rotation: float
    shift_x: float
    shift_y: float
    magnification: float
    converged: bool
    iterations: int
    tn: str

# =============================================================================
# main class
# =============================================================================

class LBTSynIM:
    """Simulate LBT AO interaction matrices / reconstructors for one system.

    Parameters
    ----------
    system : str
        One of 'LUCIdx', 'LUCIsx', 'LBTIdx', 'LBTIsx'.
    config_path : str or Path
        Path to the YAML configuration file (default './config/LBT_synim_config.yaml').
    """

    def __init__(self, system: str,
                 config_path: Union[str, Path] = "./config/LBT_synim_config.yaml"):
        if system not in VALID_SYSTEMS:
            raise ValueError(f"Unknown system '{system}'. Must be one of {VALID_SYSTEMS}")

        self.system = system
        self.side = _side_for(system)
        self.config_path = Path(config_path)
        self._load_config()

        sys_cfg = self.config["systems"][system]
        self.kl_version = sys_cfg["kl_version"]
        self.flip = bool(sys_cfg["flip"])
        self.xsign = float(sys_cfg.get("xsign", 1.0))
        self.ysign = float(sys_cfg.get("ysign", 1.0))
        guess = sys_cfg["misreg_guess"]
        self.misreg_guess = np.array([
            guess["rotation"], guess["shift_x"], guess["shift_y"], guess["magnification"],
        ], dtype=float)

        self._resolve_paths()
        self._load_calib_data()

    # ------------------------------------------------------------------
    # configuration / path handling
    # ------------------------------------------------------------------
    def _load_config(self):
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _fmt(self, template: str) -> str:
        return template.format(kl=self.kl_version, side=self.side)

    def _resolve_paths(self):
        p = self.config["paths"]
        self.root_dir = Path(self._fmt(p["root_dir_template"]))
        self.ifunc_path = self.root_dir / self._fmt(p["ifunc_template"])
        self.ifunc_inv_path = self.root_dir / self._fmt(p["ifunc_inv_template"])
        self.m2c_path = self.root_dir / self._fmt(p["m2c_template"])
        self.pupilstop_path = self.root_dir / self._fmt(p["pupilstop_template"])
        self.pupil_mask_path = self.root_dir / self._fmt(p["pupil_mask_template"])
        self.pupdata_path = self.root_dir / self._fmt(p["pupdata_template"])
        self.pupids_path = self.root_dir / self._fmt(p["pupids_template"])

        self.data_dir = self.root_dir / p.get("data_dir", "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sav_dir = Path(self._fmt(p["sav_dir_template"]))
        self.sav_file = self.sav_dir / p["sav_file_template"]
        
        self.telescope_diameter = p.get("telescope_diameter", 8.222)

    # ------------------------------------------------------------------
    # calibration-data loading
    # ------------------------------------------------------------------
    def _load_calib_data(self):
        needed = [self.ifunc_path, self.ifunc_inv_path, self.m2c_path, self.pupilstop_path]
        if all(path.exists() for path in needed):
            self.ifunc = fits.getdata(self.ifunc_path)
            self.ifunc_inv = fits.getdata(self.ifunc_inv_path)
            self.m2c = fits.getdata(self.m2c_path)
            self.pupilstop = fits.getdata(self.pupilstop_path).astype(bool)
            self.pixel_pupil = int(self.pupilstop.shape[0])
        else:
            self.ifunc = fits.getdata(self.ifunc_path)
            self.ifunc_inv, self.m2c, self.pupilstop, self.pixel_pupil = (
                self._build_calib_data_from_sav()
            )

        self.pixel_pitch = self.telescope_diameter / self.pixel_pupil

        self.pupil_mask = fits.getdata(self.pupil_mask_path).astype(bool)
        pupdata_hdu = fits.open(self.pupdata_path)
        self.pup_ids = pupdata_hdu[1].data              # columns 0/1 -> raster positions
        self.pupids = fits.getdata(self.pupids_path)     # hardware slope-order indices

        ic_cfg = self.config["interaction_matrix"]
        self.npix = ic_cfg["npix"]
        self.nslopes = ic_cfg["nslopes"]
        self.half_mask = self.pupil_mask[: self.npix // 2, : self.npix].astype(bool)

    def _build_calib_data_from_sav(self):
        """
          1. read raw ifunc / ifunc_inv / m2c / pupilstop arrays from
             ``self.sav_file``
          2. save them at self.ifunc_path / self.ifunc_inv_path /
             self.m2c_path / self.pupilstop_path (ifunc & ifunc_inv via
             specula's IFunc/IFuncInv .save(), pupilstop via save_pupil,
             m2c as a plain FITS array) so future calls hit the fast path
             above
          3. return the five arrays (ifunc, ifunc_inv, m2c, pupilstop, pixel_pupil)
        """
        data = readsav(self.sav_file)
        pixel_pupil = data['dpix']
        mask = np.zeros(pixel_pupil**2)
        mask[data['idx_mask']] = 1
        pupil_mask = mask.reshape([pixel_pupil,pixel_pupil])

        save_pupil(pupil_mask,self.pupilstop_path,Npix=pixel_pupil,D=self.telescope_diameter)

        m2c_full = data['klm2c']
        act_ids = np.sum(abs(m2c_full),axis=0)>0
        mode_ids = np.sum(abs(m2c_full),axis=1)>0
        m2c = m2c_full[:,act_ids]
        m2c = m2c[mode_ids,:]    
        m2c_obj = M2C(m2c=m2c)
        m2c_obj.save(self.m2c_path)

        kl = data['klmatrix']
        ifunc_inv = np.linalg.pinv(kl)
        ifunc_inv_obj = IFuncInv(ifunc_inv=ifunc_inv,mask=pupil_mask)
        ifunc_inv_obj.save(self.ifunc_inv_path)

        return ifunc_inv, m2c, pupil_mask, pixel_pupil

    # ------------------------------------------------------------------
    # binning geometry (used only by compute_interaction_matrix /
    # compute_reconstructor -- registration always uses the bin-1 geometry
    # loaded above in _load_calib_data)
    # ------------------------------------------------------------------
    def _binning_config(self, binning: int) -> dict:
        """Validate and return the raw config entry for a WFS binning
        factor > 1. Raises a clear error if that binning hasn't been
        filled in yet.

        TODO: only binning=1 (the default paths/interaction_matrix
        config) is real. Fill in config['binning']['configs'][2|3|4] with
        real CCD sizes and pupil_mask/pupdata/pupids filenames once known.
        """
        configs = self.config.get("binning", {}).get("configs", {})
        cfg = configs.get(binning, configs.get(str(binning)))
        if cfg is None:
            raise ValueError(
                f"No binning={binning} entry under config['binning']['configs'] "
                f"in {self.config_path}."
            )
        required = ("ccd_size", "npix", "nslopes", "total_slopes",
                    "pupil_mask_template", "pupdata_template", "pupids_template")
        missing = [k for k in required if cfg.get(k) is None]
        if missing:
            raise ValueError(
                f"binning={binning} is missing {missing} under "
                f"config['binning']['configs'][{binning}] in {self.config_path} "
                f"-- these are placeholders and must be filled in before this "
                f"binning can be used."
            )
        return cfg

    def _binning_tags(self, binning: int) -> dict:
        """Scalar/tag info needed to run the SPECULA simulation at a given
        binning (doesn't require loading any arrays)."""
        if binning == 1:
            return {
                "ccd_size": [self.npix, self.npix],
                "npix": self.npix,
                "nslopes": self.nslopes,
                "total_slopes": self.config["reconstructor"]["total_slopes"],
                "pupdata_tag": self.pupdata_path.stem,
                "rec_header_template": self.config["reconstructor"]["header_template"],
            }
        cfg = self._binning_config(binning)
        return {
            "ccd_size": cfg["ccd_size"],
            "npix": cfg["npix"],
            "nslopes": cfg["nslopes"],
            "total_slopes": cfg["total_slopes"],
            "pupdata_tag": Path(self._fmt(cfg["pupdata_template"])).stem,
            "rec_header_template": cfg.get("rec_header_template")
                                    or self.config["reconstructor"]["header_template"],
        }

    def _binning_arrays(self, binning: int) -> dict:
        """Pupil-mapping arrays needed by compute_reconstructor at a given
        binning (loads fits files; bin1 reuses what's already in memory)."""
        if binning == 1:
            return {"pup_ids": self.pup_ids, "pupids": self.pupids,
                    "half_mask": self.half_mask, "npix": self.npix}
        cfg = self._binning_config(binning)
        npix = cfg["npix"]
        pupil_mask = fits.getdata(self.root_dir / self._fmt(cfg["pupil_mask_template"])).astype(bool)
        pupdata_hdu = fits.open(self.root_dir / self._fmt(cfg["pupdata_template"]))
        pup_ids = pupdata_hdu[1].data
        pupids = fits.getdata(self.root_dir / self._fmt(cfg["pupids_template"]))
        half_mask = pupil_mask[: npix // 2, : npix].astype(bool)
        return {"pup_ids": pup_ids, "pupids": pupids, "half_mask": half_mask, "npix": npix}

    # ------------------------------------------------------------------
    # registration warping
    # ------------------------------------------------------------------
    def _register_ifunc_and_klinv(self, alpha):
        """Warp ifunc / inverse-KL by alpha = [rot, shiftX, shiftY, mag]."""
        rot, shiftX, shiftY, mag = alpha
        warped_mask = warp_mask(self.pupilstop, shiftX=shiftX, shiftY=shiftY, rot=rot, mag=mag)
        ifunc_new = warp_image(self.ifunc, warped_mask, flip=self.flip,
                                shiftX=shiftX, shiftY=shiftY, rot=rot, mag=mag,
                                oldpup=self.pupilstop)
        # NOTE: simplified relative to synim_sprint.py's save_ifunc_pars,
        # which applied a warp_image(...).T followed by another .T when
        # building the IFuncInv object -- the two transposes cancel out.
        ifunc_inv_new = warp_image(self.ifunc_inv.T, warped_mask, flip=self.flip,
                                    shiftX=shiftX, shiftY=shiftY, rot=rot, mag=mag,
                                    oldpup=self.pupilstop)
        return ifunc_new, ifunc_inv_new, warped_mask

    def _save_ifunc_products(self, ifunc_new, ifunc_inv_new, mask_new, tag):
        """Save registered ifunc / inverse-KL / pupilstop as their own
        SPECULA-format files, tagged with `tag`, so `compute_*` calls can
        find "the latest registered ifunc for this system". `tag` is
        either a real TN (permanent product) or a reused temp tag (see
        `_cleanup_tmp_products`)."""
        ifunc_dir = self.ifunc_path.parent
        pupilstop_dir = self.pupilstop_path.parent

        ifunc_file = ifunc_dir / f"IFunc_{self.system}_{tag}.fits"
        ifunc_inv_file = ifunc_dir / f"IFuncInv_{self.system}_{tag}.fits"
        pupilstop_fname = f"Pupilstop_{self.system}_{tag}"

        IFunc(ifunc=ifunc_new.T, mask=mask_new).save(str(ifunc_file), overwrite=True)
        IFuncInv(ifunc_inv=ifunc_inv_new, mask=mask_new).save(str(ifunc_inv_file), overwrite=True)
        save_pupil(mask_new, str(pupilstop_dir) + "/" + pupilstop_fname,
                   Npix=self.pixel_pupil, D=self.telescope_diameter)

        return ifunc_file, ifunc_inv_file, pupilstop_dir / f"{pupilstop_fname}.fits"

    def _cleanup_tmp_products(self, tag: str):
        """Best-effort removal of the temporary ifunc/ifunc_inv/pupilstop/im
        files reused across one update_registration() run."""
        candidates = [
            self.ifunc_path.parent / f"IFunc_{self.system}_{tag}.fits",
            self.ifunc_path.parent / f"IFuncInv_{self.system}_{tag}.fits",
            self.pupilstop_path.parent / f"Pupilstop_{self.system}_{tag}.fits",
            self._im_output_path(f"tmp_synim_{tag}"),
        ]
        for f in candidates:
            try:
                f.unlink(missing_ok=True)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # SPECULA simulation runner
    # ------------------------------------------------------------------
    def _write_overrides_yaml(self, overrides: dict, path: Path):
        """Write a SPECULA "*_override" yaml file from a plain nested dict."""
        payload = {f"{k}_override": v for k, v in overrides.items()}
        with open(str(path) + ".yml", "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    def _im_output_path(self, tag: str) -> Path:
        return self.root_dir / "im" / f"{tag}.fits"

    def _run_specula_simulation(self, ifunc_tag: str, pupilstop_tag: str,
                                 nmodes: int, im_tag: str, mod_amp: Optional[float] = None,
                                 binning_tags: Optional[dict] = None):
        """ Diffraction-limited push-pull calibration. """
        m2c_tag = self.m2c_path.stem
        overrides = {
            "pyr_im_calibrator": {"im_tag": im_tag, "nmodes": nmodes, "overwrite": True},
            "pushpull": {"nmodes": nmodes},
            "dm": {"ifunc_object": ifunc_tag, "m2c_object": m2c_tag, "nmodes": nmodes},
            "pupilstop": {"tag": pupilstop_tag},
            "main": {"pixel_pupil": self.pixel_pupil, "pixel_pitch": self.pixel_pitch,
                     "root_dir": str(self.root_dir), "total_time": nmodes*0.001*2},
        }
        pyr_override = {}
        if mod_amp is not None:
            pyr_override["mod_amp"] = mod_amp
        if binning_tags is not None:
            pyr_override["output_resolution"] = binning_tags["ccd_size"][0]
            overrides["ocam"] = {"size": binning_tags["ccd_size"]}
            overrides["pyr_slopes"] = {"pupdata_object": binning_tags["pupdata_tag"]}
        if pyr_override:
            overrides["pyr"] = pyr_override

        override_path = self.data_dir / f"_overrides_{_tn_now()}"
        self._write_overrides_yaml(overrides, override_path)
        main_yaml = self.config["registration"]["main_simul_yaml"]
        subprocess.run(["specula", str(main_yaml), str(override_path) + ".yml"], check=True)

    def _run_pc_specula_simulation(self, ifunc_tag: str, ifunc_inv_tag: str, pupilstop_tag: str,
                                    nmodes: int, im_tag: str, mod_amp: float,
                                    seeing: float, pc_cfg: dict,
                                    binning_tags: Optional[dict] = None):
        """Partial-correction push-pull calibration, adapted from
        pc_calib_yml.txt / config/pc_blocks.yml.

        'Perfect correction' (empty correction_vector_path) feeds the
        analysed atmospheric modes directly to the DM (no attenuation);
        a non-empty path instead scales them by a per-mode correction
        vector before commanding the DM.
        """
        nmodes_pc = pc_cfg["nmodes_perfect_correction"]
        corr_vec_path = pc_cfg.get("correction_vector_path", "") or ""
        n_avg = pc_cfg.get("n_screens_average", 1)
        corr_vec_name = f'corr_vec_{_tn_now()}'
        save_perfect_correction_vector(fname=corr_vec_name,dest_dir=self.data_dir,full_path=corr_vec_path,Nmodes=nmodes,Ncorrmodes=nmodes_pc)

        extra_blocks_yaml = self.config["interaction_matrix"].get("pc_extra_blocks_yaml")
        main_yaml = self.config["registration"]["main_simul_yaml"]
        m2c_tag = self.m2c_path.stem

        accum = None
        for i in range(n_avg):
            overrides = {
                "pyr_im_calibrator": {"im_tag": im_tag, "nmodes": nmodes, "overwrite": True},
                "pushpull": {"nmodes": nmodes},
                "pyr": {"mod_amp": mod_amp},
                "dm": {"ifunc_object": ifunc_tag, "m2c_object": m2c_tag, "nmodes": nmodes},
                "pupilstop": {"tag": pupilstop_tag},
                "main": {"pixel_pupil": self.pixel_pupil, "pixel_pitch": self.pixel_pitch,
                         "root_dir": str(self.root_dir), "total_time": nmodes*0.001*2},
                "seeing_random": {"constant": seeing},
                "atmo_random": {"update_interval": int(nmodes*2), "seed": int(i+1)},
                "scale_random": {"constant_mul_data": str(corr_vec_name)},
                "modal_analysis_random": {"ifunc_inv_object": ifunc_inv_tag, "nmodes": nmodes_pc},
                "dm_random": {"ifunc_object": ifunc_tag, "m2c_object": self.m2c_path.stem,
                               "nmodes": nmodes_pc},
            }
            overrides['pyr']['inputs'] = {"in_ef": 'ef_mode.out_ef'}
            if binning_tags is not None:
                overrides["pyr"]["output_resolution"] = binning_tags["ccd_size"][0]
                overrides["ocam"] = {"size": binning_tags["ccd_size"]}
                overrides["pyr_slopes"] = {"pupdata_object": binning_tags["pupdata_tag"]}

            override_path = self.data_dir / f"_overrides_pc_{_tn_now()}_{i}"
            self._write_overrides_yaml(overrides, override_path)

            cmd = ["specula", str(main_yaml)]
            if extra_blocks_yaml:
                cmd.append(str(extra_blocks_yaml))
            cmd.append(str(override_path) + ".yml")
            subprocess.run(cmd, check=True)

            run_im = fits.getdata(self._im_output_path(im_tag))
            accum = run_im.copy() if accum is None else accum + run_im

        averaged = accum / n_avg
        fits.writeto(self._im_output_path(im_tag), averaged, overwrite=True)

        return {
            "nmodes_perfect_correction": nmodes_pc,
            "correction_vector_path": corr_vec_path,
            "n_screens_average": n_avg,
        }

    # ------------------------------------------------------------------
    # slope re-ordering (registration-fit space only, see README)
    # ------------------------------------------------------------------
    def _reshape_to_pupil_frame(self, im: np.ndarray, nmodes: int) -> np.ndarray:
        """Re-order a raw (hardware-order) IM into the 'true pupil' slope
        ordering used for the registration sensitivity-matrix fit. Port of
        `get_refim` in synim_sprint.py."""
        im = im[: self.nslopes, :nmodes]
        out = np.zeros([int(self.half_mask.sum()), nmodes])
        for j in range(nmodes):
            img = np.zeros(self.half_mask.size)
            img[self.half_mask.flatten()] = im[self.pupids, j]
            img = img.reshape(self.half_mask.shape)
            out[:, j] = img[self.half_mask]
        return out

    def _synthetic_im(self, alpha, nmodes: int, tmp_tag: Optional[str] = None) -> np.ndarray:
        """
        Register ifunc/pupil by alpha, run one push-pull SPECULA
        calibration, and return the result reshaped into the same 'true
        pupil' ordering as `_reshape_to_pupil_frame`. Port of `get_synim`
        in synim_sprint.py.

        `tmp_tag`, when given, is reused across every call within one
        `update_registration()` run so the (many) intermediate ifunc /
        pupilstop / im files it produces overwrite each other in place
        rather than littering disk with one timestamped set per iteration
        per finite-difference parameter (they're cleaned up at the end of
        `update_registration` regardless).
        """
        ifunc_new, ifunc_inv_new, mask_new = self._register_ifunc_and_klinv(alpha)
        tag = tmp_tag or f"tmp"#_{_tn_now()}"
        ifunc_file, _, pupilstop_file = self._save_ifunc_products(
            ifunc_new, ifunc_inv_new, mask_new, tag)
        im_tag = f"tmp_synim_{tag}"
        self._run_specula_simulation(ifunc_tag=ifunc_file.stem,
                                      pupilstop_tag=pupilstop_file.stem,
                                      nmodes=nmodes, im_tag=im_tag)
        raw = fits.getdata(self._im_output_path(im_tag))[:, :nmodes]

        # NOTE: xsign/ysign generalise what was a fixed (+1, -1) flip in
        # the original get_synim -- see README "Open items" (xsign/ysign).
        aux = raw.copy()
        half = self.nslopes // 2
        aux[:half, :] = raw[half:, :] * self.xsign
        aux[half:, :] = raw[:half, :] * self.ysign

        out = np.zeros([int(self.half_mask.sum()), nmodes])
        fimg = np.zeros(self.npix ** 2)
        for j in range(nmodes):
            np.put(fimg, self.pup_ids[:, 0], aux[:half, j])
            np.put(fimg, self.pup_ids[:, 1], aux[half:, j])
            f2d = fimg.reshape([self.npix, self.npix])
            fcut = f2d[: self.npix // 2, : self.npix]
            out[:, j] = fcut[self.half_mask]
        return out

    def _sensitivity_matrix(self, alpha, eps, nmodes, tmp_tag):
        sens = []
        for k, e in enumerate(eps):
            a_plus = alpha.copy()
            a_plus[k] += e
            push = self._synthetic_im(a_plus, nmodes, tmp_tag=tmp_tag)
            a_minus = alpha.copy()
            a_minus[k] -= e
            pull = self._synthetic_im(a_minus, nmodes, tmp_tag=tmp_tag)
            sens.append(((push - pull) / (2 * e)).flatten())
        return np.array(sens).T

    def _mode_slope_map(self, slopes_1d: np.ndarray) -> np.ndarray:
        """Reconstruct a 1D 'true pupil frame' slope vector into a 2D image
        for display. Port of `show_im_slopes_idx` in the plotting notebook."""
        img = np.zeros(self.half_mask.size)
        img[self.half_mask.flatten()] = slopes_1d
        return img.reshape(self.half_mask.shape)

    def _plot_mode_slopes_check(self, refim_mode, before_mode, after_mode, tn, mode_idx):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        panels = [("Measured", refim_mode),
                  ("Synthetic (before)", before_mode),
                  ("Synthetic (after)", after_mode)]
        for ax, (title, data) in zip(axes, panels):
            img = self._mode_slope_map(data)
            im = ax.imshow(np.ma.masked_array(img, mask=img == 0), origin="lower", cmap="RdBu")
            ax.set_title(f"{title}\nmode {mode_idx}")
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6)
        fig.suptitle(f"{self.system} registration check -- TN {tn}")
        fig.tight_layout()
        out_png = self.data_dir / f"MisRegCheck_{self.system}_{tn}.png"
        fig.savefig(out_png, dpi=120)
        return fig

    # ------------------------------------------------------------------
    # misc small helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_imat(imat: Union[np.ndarray, str, Path]) -> np.ndarray:
        if isinstance(imat, (str, Path)):
            return fits.getdata(imat)
        return np.asarray(imat)

    @staticmethod
    def _find_latest(directory: Union[str, Path], pattern: str) -> Optional[Path]:
        matches = sorted(Path(directory).glob(pattern))  # TN prefix sorts chronologically
        return matches[-1] if matches else None

    @staticmethod
    def _tn_from_filename(path: Union[str, Path]) -> Optional[str]:
        m = re.search(r"(\d{8}_\d{6})", Path(path).stem)
        return m.group(1) if m else None

    def _latest_registered_products(self):
        # NOTE: the "[0-9]*" glob (rather than "*") deliberately excludes
        # the "tmp_..." reused-temp-tag files from _synthetic_im, which
        # always start with a letter, not a digit.
        ifunc_path = self._find_latest(self.ifunc_path.parent, f"IFunc_{self.system}_[0-9]*.fits")
        ifunc_inv_path = self._find_latest(self.ifunc_path.parent, f"IFuncInv_{self.system}_[0-9]*.fits")
        pupilstop_path = self._find_latest(self.pupilstop_path.parent, f"Pupilstop_{self.system}_[0-9]*.fits")
        if not (ifunc_path and ifunc_inv_path and pupilstop_path):
            raise FileNotFoundError(
                f"No registered ifunc/pupilstop found for {self.system} -- "
                f"run update_registration() first."
            )
        tn = self._tn_from_filename(ifunc_path)
        return ifunc_path.stem, ifunc_inv_path.stem, pupilstop_path.stem, tn

    def _resolve_imat(self, imat):
        if imat is None:
            path = self._find_latest(self.data_dir, f"IntMat_{self.system}_[0-9]*.fits")
            if path is None:
                raise FileNotFoundError(f"No IntMat found for {self.system} in {self.data_dir}")
            return fits.getdata(path), path, self._tn_from_filename(path)
        if isinstance(imat, (str, Path)):
            return fits.getdata(imat), Path(imat), self._tn_from_filename(Path(imat))
        return np.asarray(imat), None, None

    def _update_config_misreg(self, alpha, tn):
        """Persist the newly converged registration as the system's new
        default initial guess, keeping a timestamped backup of the
        previous config alongside it."""
        backup_path = self.config_path.with_name(
            f"{self.config_path.stem}_backup_{tn}{self.config_path.suffix}")
        shutil.copy2(self.config_path, backup_path)

        self.config["systems"][self.system]["misreg_guess"] = {
            "rotation": float(alpha[0]), "shift_x": float(alpha[1]),
            "shift_y": float(alpha[2]), "magnification": float(alpha[3]),
        }
        with open(self.config_path, "w") as f:
            yaml.safe_dump(self.config, f, sort_keys=False, default_flow_style=False)
        self.misreg_guess = np.asarray(alpha, dtype=float)

    def _save_misreg_fits(self, result: RegistrationResult, measured_imat, reg_cfg, tn,
                           ifunc_file, ifunc_inv_file, pupilstop_file, alpha0):
        hdr = fits.Header()
        hdr['SYNTH'] = True
        hdr["SYSTEM"] = self.system
        hdr["SIDE"] = self.side
        hdr["KLVER"] = self.kl_version
        hdr["FLIP"] = self.flip
        hdr["XSIGN"] = self.xsign
        hdr["YSIGN"] = self.ysign
        hdr["TN"] = tn
        hdr["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hdr["NMODES"] = reg_cfg["nmodes"]
        hdr["DROT"] = reg_cfg["delta_rotation"]
        hdr["DSHIFT"] = reg_cfg["delta_shift"]
        hdr["DMAG"] = reg_cfg["delta_magnification"]
        hdr["TOL"] = reg_cfg["tolerance"]
        hdr["MAXITER"] = reg_cfg["max_iterations"]
        hdr["ROT0"] = float(alpha0[0])
        hdr["SHIFTX0"] = float(alpha0[1])
        hdr["SHIFTY0"] = float(alpha0[2])
        hdr["MAG0"] = float(alpha0[3])
        hdr["ROT"] = result.rotation
        hdr["SHIFTX"] = result.shift_x
        hdr["SHIFTY"] = result.shift_y
        hdr["MAG"] = result.magnification
        hdr["NITER"] = result.iterations
        hdr["CONVRG"] = result.converged
        hdr["IFUNCF"] = str(ifunc_file)
        hdr["IFINVF"] = str(ifunc_inv_file)
        hdr["PUPF"] = str(pupilstop_file)
        hdr["MEASIM"] = str(measured_imat) if isinstance(measured_imat, (str, Path)) else "array_input"

        data = np.array([result.rotation, result.shift_x, result.shift_y, result.magnification])
        out_path = self.data_dir / f"MisReg_{self.system}_{tn}.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(out_path, overwrite=True)
        return out_path

    def _build_rec_header(self, rec_cfg, Nmodes, argos, imat_tn, tn, binning, bin_tags):
        template_path = bin_tags.get("rec_header_template") or rec_cfg["header_template"]
        hdr = fits.getheader(template_path).copy()
        hdr["BINNING"] = binning
        hdr["CCDSZX"] = bin_tags["ccd_size"][0]
        hdr["CCDSZY"] = bin_tags["ccd_size"][1]
        hdr["IM_MODES"] = Nmodes
        hdr["M2C"] = f"KL_v{self.kl_version}"
        hdr["ORIG_REC"] = "synth_rec"
        hdr["C_DIST_F"] = "synth_pp"
        hdr["M_DIST_F"] = "synth_pp"
        hdr["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d")
        hdr["SYSTEM"] = self.system
        hdr["SIDE"] = self.side
        hdr["ARGOS"] = argos
        hdr["XSIGN"] = self.xsign
        hdr["YSIGN"] = self.ysign
        hdr["IMATTN"] = imat_tn or "unknown"
        hdr["RECTN"] = tn
        hdr["SYNTH"] = True
        return hdr

    # ==================================================================
    # PUBLIC API
    # ==================================================================
    def update_registration(self, measured_imat: Union[np.ndarray, str, Path],
                             save: bool = True, show_plot: bool = True
                             ) -> Optional[RegistrationResult]:
        """Estimate misregistration parameters (rotation, shift_x, shift_y,
        magnification -- shear intentionally removed) between the nominal
        influence functions and a measured interaction matrix. Always
        operates at the default (bin-1) WFS geometry.

        On convergence: saves the registration parameters to
        ``MisReg_<system>_<TN>.fits`` (header only), saves the registered
        influence functions / inverse KL / pupil mask as their own
        SPECULA-format files (``IFunc_<system>_<TN>.fits`` /
        ``IFuncInv_<system>_<TN>.fits`` / ``Pupilstop_<system>_<TN>.fits``),
        overwrites this system's `misreg_guess` in the config file (keeping
        a timestamped backup), and produces a before/after slope-map plot
        for the configured check mode (default: mode 30).

        Parameters
        ----------
        measured_imat : ndarray or path
            Measured interaction matrix (hardware slope order).
        save : bool
            Whether to write the MisReg/IFunc/IFuncInv/pupilstop files and
            update the config (default True).
        show_plot : bool
            Whether to produce the before/after slope-map check (default True).

        Returns
        -------
        RegistrationResult, or None if the fit did not converge within
        the configured number of iterations.
        """
        reg_cfg = self.config["registration"]
        nmodes = reg_cfg["nmodes"]
        eps = np.array([
            reg_cfg["delta_rotation"], reg_cfg["delta_shift"],
            reg_cfg["delta_shift"], reg_cfg["delta_magnification"],
        ])
        tol = reg_cfg["tolerance"]
        max_its = reg_cfg["max_iterations"]
        mode_idx = reg_cfg.get("slopes_mode_check", 30)

        measured = self._load_imat(measured_imat)
        refim = self._reshape_to_pupil_frame(measured, nmodes)

        alpha0 = self.misreg_guess.copy()
        alpha = alpha0.copy()

        # Reused across every iteration/finite-difference call in this run
        # so intermediate products overwrite each other instead of
        # littering disk -- see _synthetic_im / _cleanup_tmp_products.
        tmp_tag = f"tmp"#_{_tn_now()}"

        err = tol + 1
        k = 0
        while err > tol and k < max_its:
            sens = self._sensitivity_matrix(alpha, eps, nmodes, tmp_tag)
            synim = self._synthetic_im(alpha, nmodes, tmp_tag=tmp_tag)
            gain = np.diag(np.linalg.pinv(synim) @ refim)
            residual = (refim @ np.diag(1 / gain)) - synim
            dalpha = np.linalg.pinv(sens) @ residual.flatten()
            alpha_new = alpha + dalpha
            err = np.max(np.minimum(np.abs(dalpha) / np.abs(alpha_new), np.abs(alpha_new)))
            alpha = alpha_new
            k += 1

        if err > tol:
            self._cleanup_tmp_products(tmp_tag)
            return None

        tn = _tn_now()
        ifunc_new, ifunc_inv_new, mask_new = self._register_ifunc_and_klinv(alpha)
        ifunc_file, ifunc_inv_file, pupilstop_file = self._save_ifunc_products(
            ifunc_new, ifunc_inv_new, mask_new, tn)

        result = RegistrationResult(
            rotation=float(alpha[0]), shift_x=float(alpha[1]), shift_y=float(alpha[2]),
            magnification=float(alpha[3]), converged=True, iterations=k, tn=tn,
        )

        if save:
            self._save_misreg_fits(result, measured_imat, reg_cfg, tn,
                                    ifunc_file, ifunc_inv_file, pupilstop_file, alpha0)
            self._update_config_misreg(alpha, tn)

        if show_plot and nmodes > mode_idx:
            before = self._synthetic_im(alpha0, nmodes, tmp_tag=tmp_tag)[:, mode_idx]
            after = self._synthetic_im(alpha, nmodes, tmp_tag=tmp_tag)[:, mode_idx]
            self._plot_mode_slopes_check(refim[:, mode_idx], before, after, tn, mode_idx)

        self._cleanup_tmp_products(tmp_tag)
        return result

    def compute_interaction_matrix(self, rMod: float = 3.0,
                                    seeing: Optional[float] = None,
                                    nmodes: Optional[int] = None,
                                    binning: int = 1) -> np.ndarray:
        """Simulate the interaction matrix for this system, using the
        latest registered influence functions (see `update_registration`).

        Parameters
        ----------
        rMod : float
            Pyramid modulation radius, lambda/D units (default 3).
        seeing : float, optional
            If given (arcsec), calibrates a partial-correction (PC) IM for
            that seeing value instead of a diffraction-limited one, using
            the `interaction_matrix.partial_correction` config section.
        nmodes : int, optional
            Number of modes to calibrate. Defaults to
            `interaction_matrix.default_nmodes`.
        binning : int
            WFS CCD binning factor: 1 (default), 2, 3 or 4. Selects the CCD
            size and pupil-mapping files from `config['binning']['configs']`
            (binning=1 reuses the system's default paths). TODO(user):
            2/3/4 are placeholders until real CCD sizes / pupil files are
            filled in.

        Returns
        -------
        The interaction matrix (also saved as
        ``IntMat_<system>_<TN>.fits``, with all relevant parameters in the
        header).
        """
        ic_cfg = self.config["interaction_matrix"]
        nmodes = nmodes or ic_cfg.get("default_nmodes", 600)
        bin_tags = self._binning_tags(binning)

        ifunc_tag, ifunc_inv_tag, pupilstop_tag, reg_tn = self._latest_registered_products()

        tn = _tn_now()
        calib_tag = f"_calib_{self.system}_{tn}"

        pc_info = None
        if seeing is None:
            self._run_specula_simulation(ifunc_tag=ifunc_tag, pupilstop_tag=pupilstop_tag,
                                          nmodes=nmodes, im_tag=calib_tag, mod_amp=rMod,
                                          binning_tags=bin_tags if binning != 1 else None)
        else:
            pc_cfg = ic_cfg["partial_correction"]
            pc_info = self._run_pc_specula_simulation(
                ifunc_tag=ifunc_tag, ifunc_inv_tag=ifunc_inv_tag, pupilstop_tag=pupilstop_tag,
                nmodes=nmodes, im_tag=calib_tag, mod_amp=rMod, seeing=seeing, pc_cfg=pc_cfg,
                binning_tags=bin_tags if binning != 1 else None)

        imat = fits.getdata(self._im_output_path(calib_tag))[:, :nmodes]

        hdr = fits.Header()
        hdr["SYSTEM"] = self.system
        hdr["SIDE"] = self.side
        hdr["KLVER"] = self.kl_version
        hdr["TN"] = tn
        hdr["REGTN"] = reg_tn or "unknown"
        hdr["RMOD"] = rMod
        hdr["NMODES"] = nmodes
        hdr["SEEING"] = seeing if seeing is not None else -1.0
        hdr["ISPC"] = seeing is not None
        hdr["BINNING"] = binning
        hdr["CCDSZX"] = bin_tags["ccd_size"][0]
        hdr["CCDSZY"] = bin_tags["ccd_size"][1]
        if pc_info is not None:
            hdr["PCNPERF"] = pc_info["nmodes_perfect_correction"]
            hdr["PCVEC"] = pc_info["correction_vector_path"] or "NONE"
            hdr["PCNAVG"] = pc_info["n_screens_average"]
        hdr["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        out_path = self.data_dir / f"IntMat_{self.system}_{tn}.fits"
        fits.writeto(out_path, imat, header=hdr, overwrite=True)
        return imat


    def compute_reconstructor(self, imat: Union[np.ndarray, str, Path, None] = None,
                               Nmodes: Optional[int] = None,
                               argos: Optional[bool] = None,
                               binning: int = 1) -> np.ndarray:
        """Compute the reconstructor for this system from an interaction
        matrix (generalises compute_lbt_recdx.py / compute_lbt_recsx.py).

        Parameters
        ----------
        imat : ndarray, path, or None
            Interaction matrix (hardware slope order) or its full path.
            Defaults to the latest ``IntMat_<system>_*.fits`` for this
            system.
        Nmodes : int, optional
            Number of modes to keep in the reconstructor. Defaults to the
            number of columns in `imat`.
        argos : bool, optional
            Whether to apply the ARGOS half-gain convention. Defaults to
            `reconstructor.argos_default` in the config (True).
        binning : int
            WFS CCD binning factor the `imat` was computed at: 1 (default),
            2, 3 or 4. Must match what was used in
            `compute_interaction_matrix`, since it selects the matching
            pupil-mapping arrays. TODO(user): 2/3/4 are placeholders.

        Returns
        -------
        The reconstructor matrix (also saved as
        ``RecMat_<system>_<TN>.fits``, with all relevant parameters in the
        header).
        """
        rec_cfg = self.config["reconstructor"]
        if argos is None:
            argos = rec_cfg.get("argos_default", True)

        bin_tags = self._binning_tags(binning)
        bin_arrays = self._binning_arrays(binning)
        npix = bin_arrays["npix"]
        nslopes = bin_tags["nslopes"]
        pup_ids = bin_arrays["pup_ids"]
        pupids = bin_arrays["pupids"]
        half_mask = bin_arrays["half_mask"]
        total_commands = rec_cfg["total_commands"]
        total_slopes = bin_tags["total_slopes"]

        imat_arr, _, imat_tn = self._resolve_imat(imat)
        if Nmodes is None:
            Nmodes = imat_arr.shape[1]
        imat_arr = imat_arr[:, :Nmodes]

        half = nslopes // 2
        aux = np.zeros_like(imat_arr)
        aux[:half, :] = imat_arr[half:, :] * self.xsign
        aux[half:, :] = imat_arr[:half, :] * self.ysign
        aux *= rec_cfg.get("slopes_scale", 4.0e9)

        IM = np.zeros_like(imat_arr)
        fimg = np.zeros(npix ** 2)
        for i in range(Nmodes):
            np.put(fimg, pup_ids[:, 0], aux[:half, i])
            np.put(fimg, pup_ids[:, 1], aux[half:, i])
            f2d = fimg.reshape([npix, npix])
            img = f2d[: npix // 2, :]
            IM[pupids, i] = img.flatten()[half_mask.flatten()]

        IMinv = np.linalg.pinv(IM[:nslopes, :Nmodes])
        Rec = np.pad(IMinv, pad_width=((0, total_commands - Nmodes),
                                        (0, total_slopes - nslopes)),
                     mode="constant", constant_values=0.0)

        iir_rows = rec_cfg["iir_command_rows"]
        iir_source_modes = rec_cfg.get("iir_source_modes", [0] * len(iir_rows))
        for row, mode in zip(iir_rows, iir_source_modes):
            src_mode = mode if mode < IMinv.shape[0] else 0
            Rec[row, :] = np.pad(IMinv[src_mode, :], pad_width=(0, total_slopes - nslopes),
                                  mode="constant", constant_values=0.0)

        Rec = Rec.astype(">f4")
        if argos:
            Rec /= 2

        tn = _tn_now()
        hdr = self._build_rec_header(rec_cfg, Nmodes, argos, imat_tn, tn, binning, bin_tags)
        out_path = self.data_dir / f"RecMat_{self.system}_{tn}.fits"
        fits.writeto(out_path, Rec, header=hdr, overwrite=True)
        return Rec
