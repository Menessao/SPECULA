"""
lbt_synim.py
============

Simulate LBT (Large Binocular Telescope) AO interaction matrices and
reconstructors for the 4 available systems:

    LUCIdx, LUCIsx, LBTIdx, LBTIsx

built on top of the SPECULA end-to-end AO simulation package
(https://github.com/Menessao/SPECULA/tree/xao). Low-level, stateless
SPECULA-driving plumbing (geometric warps, saving calibration products,
writing override yamls, running the `specula` CLI) lives in
specula_helpers.py; this file holds the `LBTSynIM` class itself.

Typical usage
-------------
    from lbt_synim import LBTSynIM

    lucidx = LBTSynIM('LUCIdx')
    misreg = lucidx.update_registration(meas_imat)
    imat   = lucidx.compute_interaction_matrix(rMod=2, seeing=1.0)
    rec    = lucidx.compute_reconstructor(imat, Nmodes=550)
    fig    = lucidx.plot_registration_check(mode_idx=30)

IMPORTANT -- no live SPECULA source in this environment: block/field
names (DM, PyrSlopec, CCD, ModulatedPyramid, the "*_override" yaml merge
convention, toccd's signature) are ported from the reference
scripts/configs provided during development, not verified against
SPECULA's current source. Sanity-check before relying on this in
production, especially the binning overrides in `_base_overrides`.
"""

from __future__ import annotations

import datetime
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import yaml
from astropy.io import fits
from scipy.io import readsav
import matplotlib.pyplot as plt

import specula
specula.init(0)

from specula.data_objects.m2c import M2C
from specula.data_objects.ifunc import IFunc
from specula.data_objects.ifunc_inv import IFuncInv

from specula_helpers import (
    warp_image, warp_mask, save_pupil, save_perfect_correction_vector,
    write_overrides_yaml, run_specula, toccd,
)

VALID_SYSTEMS = ("LUCIdx", "LUCIsx", "LBTIdx", "LBTIsx")


def _tn_now() -> str:
    """Tracking-number timestamp, e.g. '20260927_153000'."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def _side_for(system: str) -> str:
    return "dx" if system.endswith("dx") else "sx"


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
        self._load_latest_registration()

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

        self.output_dir = self.root_dir / p.get("output_dir", ".output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.sav_dir = Path(self._fmt(p["sav_dir_template"]))
        self.sav_file = self.sav_dir / p["sav_file_template"]

        optics = self.config["optics"]
        self.telescope_diameter = optics["telescope_diameter"]
        self.base_pup_diam = optics["pup_diam"]
        self.base_pup_dist = optics["pup_dist"]
        self.base_npix = optics["npix"]

    def _binned(self, path: Path, binning: int) -> Path:
        """`path` unchanged for binning=1, or with "_{binning}x{binning}"
        inserted before the extension for binning>1 -- the on-disk naming
        convention for pupdata/pupids/pupil_mask variants."""
        if binning == 1:
            return path
        return path.with_name(f"{path.stem}_{binning}x{binning}{path.suffix}")

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
        self.default_nmodes = int(self.m2c.shape[0])

        geom = self._pupil_geometry(1)
        self.npix, self.nslopes = geom["npix"], geom["nslopes"]
        self.pup_ids, self.pupids, self.half_mask = geom["pup_ids"], geom["pupids"], geom["half_mask"]

    def _build_calib_data_from_sav(self):
        """Read raw ifunc_inv/m2c/pupilstop out of `self.sav_file` and
        cache them at self.ifunc_inv_path/self.m2c_path/self.pupilstop_path
        (SPECULA's own IFuncInv/M2C/Pupilstop formats) so future __init__
        calls hit the fast path above. `self.ifunc` itself is always read
        directly from self.ifunc_path (not rebuilt from .sav)."""
        data = readsav(self.sav_file)
        pixel_pupil = data["dpix"]
        mask = np.zeros(pixel_pupil ** 2)
        mask[data["idx_mask"]] = 1
        pupil_mask = mask.reshape([pixel_pupil, pixel_pupil])
        save_pupil(pupil_mask, str(self.pupilstop_path.with_suffix("")),
                   Npix=pixel_pupil, D=self.telescope_diameter)

        m2c_full = data["klm2c"]
        act_ids = np.sum(abs(m2c_full), axis=0) > 0
        mode_ids = np.sum(abs(m2c_full), axis=1) > 0
        m2c = m2c_full[:, act_ids][mode_ids, :]
        M2C(m2c=m2c).save(str(self.m2c_path))

        ifunc_inv = np.linalg.pinv(data["klmatrix"])
        IFuncInv(ifunc_inv=ifunc_inv, mask=pupil_mask).save(str(self.ifunc_inv_path))

        return ifunc_inv, m2c, pupil_mask, pixel_pupil

    # ------------------------------------------------------------------
    # binning geometry (used by compute_interaction_matrix /
    # compute_reconstructor / plot_registration_check -- registration
    # itself always uses the bin-1 geometry loaded above)
    # ------------------------------------------------------------------
    def _ensure_pupil_mask(self, binning: int, pup_ids, npix: int) -> np.ndarray:
        """Load this binning's half-pupil mask from disk, or build it from
        `pup_ids` (every raster position it references, set True, keeping
        the top `npix//2` rows -- same convention as the nominal mask) and
        cache it at the standard (binned) path if it doesn't exist yet."""
        path = self._binned(self.pupil_mask_path, binning)
        if path.exists():
            full_mask = fits.getdata(path).astype(bool)
        else:
            full_mask = np.zeros(npix * npix, dtype=bool)
            np.put(full_mask, pup_ids[:, 0], True)
            np.put(full_mask, pup_ids[:, 1], True)
            full_mask = full_mask.reshape(npix, npix)
            path.parent.mkdir(parents=True, exist_ok=True)
            fits.writeto(path, full_mask.astype(np.uint8), overwrite=True)
        return full_mask[: npix // 2, :npix]

    def _pupil_geometry(self, binning: int) -> dict:
        """Resolve every binning-dependent quantity needed to run a
        push-pull calibration and reshape its output: the pyramid/CCD
        sizes to override, the pupdata SPECULA tag, and the loaded
        pup_ids/pupids/half_mask arrays. binning=1 reuses the standard
        (un-suffixed) files; binning>1 uses the same files with
        "_{binning}x{binning}" appended (see `_binned`) -- pupdata/pupids
        must already exist for a given binning (real SPECULA calibration
        products), only the pupil_mask is built on-demand if missing."""
        npix = self.base_npix // binning
        pupdata_path = self._binned(self.pupdata_path, binning)
        pupids_path = self._binned(self.pupids_path, binning)
        pup_ids = fits.open(pupdata_path)[1].data
        pupids = fits.getdata(pupids_path)
        return {
            "binning": binning,
            "npix": npix,
            "nslopes": len(pupids),
            "ccd_size": [npix, npix],
            "pup_diam": self.base_pup_diam / binning,
            "pup_dist": self.base_pup_dist / binning,
            "pupdata_tag": pupdata_path.stem,
            "pup_ids": pup_ids,
            "pupids": pupids,
            "half_mask": self._ensure_pupil_mask(binning, pup_ids, npix),
        }

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
        ifunc_inv_new = warp_image(self.ifunc_inv.T, warped_mask, flip=self.flip,
                                    shiftX=shiftX, shiftY=shiftY, rot=rot, mag=mag,
                                    oldpup=self.pupilstop)
        return ifunc_new, ifunc_inv_new, warped_mask

    def _save_ifunc_products(self, ifunc_new, ifunc_inv_new, mask_new, tag):
        """Save registered ifunc / inverse-KL / pupilstop as their own
        SPECULA-format files, tagged with `tag` -- either a real TN
        (permanent product) or a reused temp tag (see
        `_cleanup_tmp_products`)."""
        ifunc_dir = self.ifunc_path.parent
        pupilstop_dir = self.pupilstop_path.parent

        ifunc_file = ifunc_dir / f"IFunc_{self.system}_{tag}.fits"
        ifunc_inv_file = ifunc_dir / f"IFuncInv_{self.system}_{tag}.fits"
        pupilstop_fname = f"Pupilstop_{self.system}_{tag}"

        IFunc(ifunc=ifunc_new.T, mask=mask_new).save(str(ifunc_file), overwrite=True)
        IFuncInv(ifunc_inv=ifunc_inv_new, mask=mask_new).save(str(ifunc_inv_file), overwrite=True)
        save_pupil(mask_new, str(pupilstop_dir / pupilstop_fname),
                   Npix=self.pixel_pupil, D=self.telescope_diameter)

        return ifunc_file, ifunc_inv_file, pupilstop_dir / f"{pupilstop_fname}.fits"

    @staticmethod
    def _safe_unlink(path: Union[str, Path]):
        """Best-effort delete -- never raises (used for temp/override files
        that are reused/overwritten in place and only need cleaning up
        once a run is done)."""
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass

    def _cleanup_tmp_products(self, tag: str):
        """Remove the temporary ifunc/ifunc_inv/pupilstop/im/override files
        reused across one update_registration() run (all overwritten in
        place during the run -- see `_synthetic_im` -- so there's exactly
        one of each to remove)."""
        for f in (self.ifunc_path.parent / f"IFunc_{self.system}_{tag}.fits",
                  self.ifunc_path.parent / f"IFuncInv_{self.system}_{tag}.fits",
                  self.pupilstop_path.parent / f"Pupilstop_{self.system}_{tag}.fits",
                  self._im_output_path(f"tmp_synim_{tag}"),
                  self.output_dir / "_overrides_dl.yml"):
            self._safe_unlink(f)

    # ------------------------------------------------------------------
    # running SPECULA (one merged function for both DL and PC)
    # ------------------------------------------------------------------
    def _im_output_path(self, tag: str) -> Path:
        return self.root_dir / "im" / f"{tag}.fits"

    def _base_overrides(self, geom: dict, ifunc_tag: str, nmodes: int, im_tag: str,
                         pupilstop_tag: Optional[str] = None,
                         mod_amp: Optional[float] = None) -> dict:
        """Override blocks common to every push-pull run -- including the
        binning-derived pyramid/CCD/pupdata overrides, always applied
        (binning=1 just reproduces the nominal values, so no special-casing
        is needed)."""
        overrides = {
            "pyr_im_calibrator": {"im_tag": im_tag, "nmodes": nmodes, "overwrite": True},
            "pushpull": {"nmodes": nmodes},
            "dm": {"ifunc_object": ifunc_tag, "m2c_object": self.m2c_path.stem, "nmodes": nmodes},
            "main": {"pixel_pupil": self.pixel_pupil, "pixel_pitch": self.pixel_pitch,
                     "root_dir": str(self.root_dir), "total_time": nmodes * 0.002},
            "pyr": {"pup_diam": geom["pup_diam"], "pup_dist": geom["pup_dist"],
                    "output_resolution": geom["npix"]},
            "ocam": {"size": geom["ccd_size"]},
            "pyr_slopes": {"pupdata_object": geom["pupdata_tag"]},
        }
        if pupilstop_tag is not None:
            overrides["pupilstop"] = {"tag": pupilstop_tag}
        if mod_amp is not None:
            overrides["pyr"]["mod_amp"] = mod_amp
        return overrides

    def _run_specula_simulation(self, ifunc_tag: str, pupilstop_tag: str, nmodes: int, im_tag: str,
                                 mod_amp: Optional[float] = None, binning: int = 1,
                                 seeing: Optional[float] = None,
                                 ifunc_inv_tag: Optional[str] = None,
                                 pc_cfg: Optional[dict] = None) -> Optional[dict]:
        """Run one push-pull IM calibration through the `specula` CLI.

        Diffraction-limited when `seeing` is None (the default): a single
        run, returns None. Partial-correction when `seeing` is given
        (`ifunc_inv_tag`/`pc_cfg` then required): builds a correction
        vector (perfect correction by default -- see
        `save_perfect_correction_vector`), appends
        `config['interaction_matrix']['pc_extra_blocks_yaml']`, averages
        `pc_cfg['n_screens_average']` independent atmosphere realizations,
        and returns a dict of the PC parameters used.

        Writes its override yaml to a fixed, reused filename
        (`_overrides_dl.yml` / `_overrides_pc.yml`) rather than one per
        call -- the caller deletes it once the whole operation (a full
        `update_registration` run, or one `compute_interaction_matrix`
        call) is done.
        """
        geom = self._pupil_geometry(binning)
        overrides = self._base_overrides(geom, ifunc_tag, nmodes, im_tag, pupilstop_tag, mod_amp)
        main_yaml = self.config["registration"]["main_simul_yaml"]

        if seeing is None:
            override_path = self.output_dir / "_overrides_dl"
            write_overrides_yaml(overrides, override_path)
            run_specula(main_yaml, override_path)
            return None

        nmodes_pc = pc_cfg["nmodes_perfect_correction"]
        corr_vec_path = pc_cfg.get("correction_vector_path") or ""
        n_avg = pc_cfg.get("n_screens_average", 1)
        corr_vec_name = f"corr_vec_{self.system}"
        save_perfect_correction_vector(fname=corr_vec_name, dest_dir=str(self.output_dir),
                                        full_path=corr_vec_path, Nmodes=nmodes, Ncorrmodes=nmodes_pc)

        extra_blocks_yaml = self.config["interaction_matrix"].get("pc_extra_blocks_yaml")
        override_path = self.output_dir / "_overrides_pc"
        overrides["pyr"]["inputs"] = {"in_ef": "ef_mode.out_ef"}  # see atmo + DM + pushpull combiner
        overrides.update({
            "seeing_random": {"constant": seeing},
            "scale_random": {"constant_mul_data": corr_vec_name},
            "modal_analysis_random": {"ifunc_inv_object": ifunc_inv_tag, "nmodes": nmodes_pc},
            "dm_random": {"ifunc_object": ifunc_tag, "m2c_object": self.m2c_path.stem, "nmodes": nmodes_pc},
        })

        accum = None
        for i in range(n_avg):
            overrides["atmo_random"] = {"update_interval": int(nmodes * 2), "seed": i + 1}
            write_overrides_yaml(overrides, override_path)
            run_specula(main_yaml, override_path, extra_blocks_yaml)

            run_im = fits.getdata(self._im_output_path(im_tag))
            accum = run_im if accum is None else accum + run_im

        fits.writeto(self._im_output_path(im_tag), accum / n_avg, overwrite=True)
        return {"nmodes_perfect_correction": nmodes_pc, "correction_vector_path": corr_vec_path,
                "n_screens_average": n_avg}

    # ------------------------------------------------------------------
    # slope re-ordering
    # ------------------------------------------------------------------
    @staticmethod
    def _raster(fimg: np.ndarray, pup_ids, first_half, second_half, npix: int) -> np.ndarray:
        """Place the two pupil-image halves at their raster positions
        (`pup_ids` columns 0/1) and reshape to a 2D `npix x npix` image.
        `fimg` is a scratch buffer reused by the caller across modes."""
        np.put(fimg, pup_ids[:, 0], first_half)
        np.put(fimg, pup_ids[:, 1], second_half)
        return fimg.reshape(npix, npix)

    def _reshape_to_pupil_frame(self, im: np.ndarray, nmodes: int) -> np.ndarray:
        """Re-order a raw (hardware-order) IM into the 'true pupil' slope
        ordering used for the registration sensitivity-matrix fit."""
        im = im[: self.nslopes, :nmodes]
        out = np.zeros([int(self.half_mask.sum()), nmodes])
        for j in range(nmodes):
            img = np.zeros(self.half_mask.size)
            img[self.half_mask.flatten()] = im[self.pupids, j]
            out[:, j] = img.reshape(self.half_mask.shape)[self.half_mask]
        return out

    def _synthetic_im(self, alpha, nmodes: int, tmp_tag: Optional[str] = None) -> np.ndarray:
        """Register ifunc/pupil by alpha, run one push-pull SPECULA
        calibration (always at the bin-1 geometry), and return the result
        reshaped into the same 'true pupil' ordering as
        `_reshape_to_pupil_frame`.

        `tmp_tag`, when given, is reused across every call within one
        `update_registration()` run so the intermediate files it produces
        overwrite each other in place instead of littering disk."""
        ifunc_new, ifunc_inv_new, mask_new = self._register_ifunc_and_klinv(alpha)
        tag = tmp_tag or "tmp"
        ifunc_file, _, pupilstop_file = self._save_ifunc_products(
            ifunc_new, ifunc_inv_new, mask_new, tag)
        im_tag = f"tmp_synim_{tag}"
        self._run_specula_simulation(ifunc_tag=ifunc_file.stem, pupilstop_tag=pupilstop_file.stem,
                                      nmodes=nmodes, im_tag=im_tag)
        raw = fits.getdata(self._im_output_path(im_tag))[:, :nmodes]

        # NOTE: xsign/ysign generalise what was once a fixed (+1, -1) flip
        # (unconfirmed for LUCIdx/LUCIsx -- see the config).
        half = self.nslopes // 2
        aux = raw.copy()
        aux[:half, :] = raw[half:, :] * self.xsign
        aux[half:, :] = raw[:half, :] * self.ysign

        out = np.zeros([int(self.half_mask.sum()), nmodes])
        fimg = np.zeros(self.npix ** 2)
        for j in range(nmodes):
            f2d = self._raster(fimg, self.pup_ids, aux[:half, j], aux[half:, j], self.npix)
            out[:, j] = f2d[: self.npix // 2, : self.npix][self.half_mask]
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

    def _imat_mode_to_2d(self, imat: np.ndarray, mode_idx: int, geom: dict) -> np.ndarray:
        """Full `npix x npix` raster of one imat column (sign-corrected,
        placed via this binning's pup_ids) -- shared by
        `plot_registration_check`."""
        half = geom["nslopes"] // 2
        col = imat[: geom["nslopes"], mode_idx]
        fimg = np.zeros(geom["npix"] ** 2)
        return self._raster(fimg, geom["pup_ids"], col[half:] * self.xsign, col[:half] * self.ysign,
                             geom["npix"])

    # ------------------------------------------------------------------
    # plotting
    # ------------------------------------------------------------------
    def _plot_slope_panels(self, panels, half_mask, title: str, out_name: Optional[str] = None):
        """Render a row of masked 2D slope-map panels, each built from a
        1D 'true pupil frame' vector via `half_mask`. Shared by
        `update_registration`'s before/after check and
        `plot_registration_check`."""
        fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4))
        for ax, (panel_title, data) in zip(np.atleast_1d(axes), panels):
            img = np.zeros(half_mask.size)
            img[half_mask.flatten()] = data
            img = img.reshape(half_mask.shape)
            im = ax.imshow(np.ma.masked_array(img, mask=img == 0), origin="lower", cmap="RdBu")
            ax.set_title(panel_title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.6)
        fig.suptitle(title)
        fig.tight_layout()
        if out_name:
            fig.savefig(self.output_dir / out_name, dpi=120)
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

    def _load_latest_registration(self):
        """Cache this system's current registration TN as `self.reg_tn`
        (or None if `update_registration()` has never succeeded for it).
        Updated again on every successful `update_registration()` call."""
        path = self._find_latest(self.ifunc_path.parent, f"IFunc_{self.system}_[0-9]*.fits")
        self.reg_tn = self._tn_from_filename(path) if path else None

    def _registered_tags(self):
        """ifunc/ifunc_inv/pupilstop SPECULA tags for the current
        registration (`self.reg_tn`). Raises if none exists yet."""
        if self.reg_tn is None:
            raise FileNotFoundError(
                f"No registered ifunc/pupilstop found for {self.system} -- "
                f"run update_registration() first.")
        suffix = f"{self.system}_{self.reg_tn}"
        return f"IFunc_{suffix}", f"IFuncInv_{suffix}", f"Pupilstop_{suffix}", self.reg_tn

    def _resolve_imat(self, imat):
        if imat is None:
            path = self._find_latest(self.output_dir, f"IntMat_{self.system}_[0-9]*.fits")
            if path is None:
                raise FileNotFoundError(f"No IntMat found for {self.system} in {self.output_dir}")
            return fits.getdata(path), path, self._tn_from_filename(path)
        if isinstance(imat, (str, Path)):
            return fits.getdata(imat), Path(imat), self._tn_from_filename(Path(imat))
        return np.asarray(imat), None, None

    def _cached_imat(self, rMod, nmodes, seeing, binning, reg_tn, pc_cfg):
        """Return (array, path) of an existing IntMat for this system whose
        header matches every parameter of the requested call -- including
        REGTN, so a new `update_registration()` run always invalidates the
        cache -- or (None, None) if there's no match."""
        want_pc = seeing is not None
        for path in sorted(self.output_dir.glob(f"IntMat_{self.system}_[0-9]*.fits"), reverse=True):
            hdr = fits.getheader(path)
            if (hdr.get("REGTN") != reg_tn or hdr.get("NMODES") != nmodes or
                    hdr.get("BINNING", 1) != binning or
                    abs(hdr.get("RMOD", -1) - rMod) > 1e-9 or
                    bool(hdr.get("ISPC", False)) != want_pc):
                continue
            if want_pc and (
                    abs(hdr.get("SEEING", -1) - seeing) > 1e-9 or
                    hdr.get("PCNPERF") != pc_cfg.get("nmodes_perfect_correction") or
                    (hdr.get("PCVEC") or "NONE") != (pc_cfg.get("correction_vector_path") or "NONE") or
                    hdr.get("PCNAVG") != pc_cfg.get("n_screens_average", 1)):
                continue
            return fits.getdata(path), path
        return None, None

    def _resolve_ref_imat(self, ref_imat) -> np.ndarray:
        """Default: the imat used in the current registration's (self.reg_tn)
        update_registration() call, found via that MisReg file's MEASIM
        header entry (rather than a separately-tracked path, so a
        user-supplied path and a freshly-saved array both resolve the
        same way)."""
        if ref_imat is not None:
            return self._load_imat(ref_imat)
        if self.reg_tn is None:
            raise RuntimeError(f"No registration on record for {self.system} -- "
                                f"run update_registration() or pass ref_imat explicitly.")
        misreg_path = self.output_dir / f"MisReg_{self.system}_{self.reg_tn}.fits"
        return fits.getdata(fits.getheader(misreg_path)["MEASIM"])

    def _resolve_calib_imat(self, calib_imat, binning: int, nmodes:int=None) -> np.ndarray:
        """Default: the latest IntMat matching the current registration
        and compute_interaction_matrix's own defaults -- computed if none
        exists yet."""
        if calib_imat is not None:
            return self._load_imat(calib_imat)
        if self.reg_tn is None:
            raise RuntimeError(f"No registration on record for {self.system} -- "
                                f"run update_registration() first.")
        rMod = self.config["interaction_matrix"].get("default_modulation_radius", 3.0)
        cached, _ = self._cached_imat(rMod, self.default_nmodes, None, binning, self.reg_tn, {})
        return cached if cached is not None else self.compute_interaction_matrix(rMod=rMod, binning=binning, nmodes=nmodes)

    @staticmethod
    def _round_alpha(alpha) -> np.ndarray:
        """Round registration params to the precision they're saved at
        everywhere (2 decimals for rotation/shifts, 4 for magnification)."""
        return np.array([round(float(alpha[0]), 2), round(float(alpha[1]), 2),
                          round(float(alpha[2]), 2), round(float(alpha[3]), 4)])

    def _save_measured_imat(self, measured: np.ndarray, tn: str) -> Path:
        path = self.output_dir / f"MeasIM_{self.system}_{tn}.fits"
        fits.writeto(path, measured, overwrite=True)
        return path

    def _update_config_misreg(self, alpha, tn):
        """Persist the newly converged registration as this system's new
        default initial guess. Edits just the one `misreg_guess:` line for
        this system via a text patch (not a full yaml.safe_dump round-trip,
        which would strip every comment in the file) -- keeps a timestamped
        backup of the previous config alongside it."""
        backup_path = self.config_path.with_name(
            f"{self.config_path.stem}_backup_{tn}{self.config_path.suffix}")
        shutil.copy2(self.config_path, backup_path)

        rot, sx, sy, mag = (float(a) for a in alpha)
        new_line = (f"    misreg_guess: {{rotation: {rot:.2f}, shift_x: {sx:.2f}, "
                    f"shift_y: {sy:.2f}, magnification: {mag:.4f}}}")
        pattern = re.compile(
            rf"(^  {re.escape(self.system)}:\n(?:.*\n)*?)^    misreg_guess:.*$", re.MULTILINE)
        text, n = pattern.subn(lambda m: m.group(1) + new_line, self.config_path.read_text(), count=1)
        if n != 1:
            raise RuntimeError(f"Could not find misreg_guess for {self.system} in {self.config_path}")
        self.config_path.write_text(text)

        self.config["systems"][self.system]["misreg_guess"] = {
            "rotation": rot, "shift_x": sx, "shift_y": sy, "magnification": mag}
        self.misreg_guess = np.asarray(alpha, dtype=float)

    def _base_header(self) -> fits.Header:
        """Fields common to every fits product this class writes."""
        hdr = fits.Header()
        hdr["SYSTEM"] = self.system
        hdr["SIDE"] = self.side
        hdr["KLVER"] = self.kl_version
        hdr["XSIGN"] = self.xsign
        hdr["YSIGN"] = self.ysign
        hdr["SYNTH"] = True
        hdr["DATE"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return hdr

    def _save_misreg_fits(self, result: RegistrationResult, meas_path: Path, reg_cfg, tn,
                           ifunc_file, ifunc_inv_file, pupilstop_file, alpha0):
        hdr = self._base_header()
        hdr["FLIP"] = self.flip
        hdr["TN"] = tn
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
        hdr["MEASIM"] = str(meas_path)

        data = np.array([result.rotation, result.shift_x, result.shift_y, result.magnification])
        out_path = self.output_dir / f"MisReg_{self.system}_{tn}.fits"
        fits.PrimaryHDU(data=data, header=hdr).writeto(out_path, overwrite=True)
        return out_path

    def _build_rec_header(self, rec_cfg, Nmodes, argos, imat_tn, tn, binning):
        hdr = fits.getheader(rec_cfg["header_template"]).copy()
        hdr.update(self._base_header())  # SYSTEM/SIDE/KLVER/XSIGN/YSIGN/SYNTH/DATE
        hdr["BINNING"] = binning
        hdr["IM_MODES"] = Nmodes
        hdr["M2C"] = f"KL_v{self.kl_version}"
        hdr["ORIG_REC"] = "synth_rec"
        hdr["C_DIST_F"] = "synth_pp"
        hdr["M_DIST_F"] = "synth_pp"
        hdr["ARGOS"] = argos
        hdr["IMATTN"] = imat_tn or "unknown"
        hdr["RECTN"] = tn
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

        On convergence: rounds the result (2 decimals for
        rotation/shifts, 4 for magnification), saves it to
        ``MisReg_<system>_<TN>.fits`` (header + a 4-element data array),
        saves the registered influence functions / inverse KL / pupil
        mask as their own SPECULA-format files
        (``IFunc_<system>_<TN>.fits`` / ``IFuncInv_<system>_<TN>.fits`` /
        ``Pupilstop_<system>_<TN>.fits``), saves `measured_imat` itself as
        ``MeasIM_<system>_<TN>.fits`` if it wasn't already a file,
        overwrites this system's `misreg_guess` in the config file
        (keeping a timestamped backup), updates `self.reg_tn`, and
        produces a before/after slope-map plot for the configured check
        mode (default: mode 30).

        Parameters
        ----------
        measured_imat : ndarray or path
            Measured interaction matrix (hardware slope order).
        save : bool
            Whether to write the MisReg fits file and update the config
            (default True). The registered ifunc/pupilstop/measured-imat
            files and `self.reg_tn` are always saved/updated regardless,
            since later calls (`compute_interaction_matrix`,
            `plot_registration_check`) depend on them.
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
        # Reused (overwritten in place) across every iteration/finite-
        # difference call in this run, then deleted at the very end.
        tmp_tag = "tmp"
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
        # Round now so the registered ifunc/pupilstop actually saved
        # below, the MisReg fits file, and the config update all agree.
        alpha = self._round_alpha(alpha)
        tn = _tn_now()
        ifunc_new, ifunc_inv_new, mask_new = self._register_ifunc_and_klinv(alpha)
        ifunc_file, ifunc_inv_file, pupilstop_file = self._save_ifunc_products(
            ifunc_new, ifunc_inv_new, mask_new, tn)
        meas_path = (measured_imat if isinstance(measured_imat, (str, Path))
                     else self._save_measured_imat(measured, tn))
        result = RegistrationResult(
            rotation=float(alpha[0]), shift_x=float(alpha[1]), shift_y=float(alpha[2]),
            magnification=float(alpha[3]), converged=True, iterations=k, tn=tn,
        )
        if save:
            self._save_misreg_fits(result, meas_path, reg_cfg, tn,
                                    ifunc_file, ifunc_inv_file, pupilstop_file, alpha0)
            self._update_config_misreg(alpha, tn)
        if show_plot and nmodes > mode_idx:
            # NOTE: both re-derived at the *converged* alpha (and the
            # initial alpha0) rather than the loop's last `synim`, which
            # corresponds to the second-to-last iterate, not the final one.
            before = self._synthetic_im(alpha0, nmodes, tmp_tag=tmp_tag)[:, mode_idx]
            after = self._synthetic_im(alpha, nmodes, tmp_tag=tmp_tag)[:, mode_idx]
            self._plot_slope_panels(
                [("Measured", refim[:, mode_idx]), ("Synthetic (before)", before),
                 ("Synthetic (after)", after)],
                self.half_mask, f"{self.system} registration check -- TN {tn}",
                f"MisRegCheck_{self.system}_{tn}.png")
        self._cleanup_tmp_products(tmp_tag)
        self.reg_tn = tn
        return result

    def compute_interaction_matrix(self, rMod: float = 3.0,
                                    seeing: Optional[float] = None,
                                    nmodes: Optional[int] = None,
                                    binning: int = 1) -> np.ndarray:
        """Simulate the interaction matrix for this system, using the
        current registration (`self.reg_tn` -- see `update_registration`).

        Parameters
        ----------
        rMod : float
            Pyramid modulation radius, lambda/D units (default 3).
        seeing : float, optional
            If given (arcsec), calibrates a partial-correction (PC) IM for
            that seeing value instead of a diffraction-limited one, using
            the `interaction_matrix.partial_correction` config section.
        nmodes : int, optional
            Number of modes to calibrate. Defaults to `self.default_nmodes`
            (the m2c's own mode count).
        binning : int
            WFS CCD binning factor: 1 (default), 2, 3 or 4. pyr.pup_diam/
            pup_dist/output_resolution, ocam.size, and pyr_slopes.pupdata_object
            all scale with it automatically -- see `_pupil_geometry`.

        Returns
        -------
        The interaction matrix (also saved as
        ``IntMat_<system>_<TN>.fits``, with all relevant parameters in the
        header). If an existing IntMat already matches every parameter of
        this call -- including the registration used -- it's loaded and
        returned directly instead of recomputing.
        """
        ic_cfg = self.config["interaction_matrix"]
        nmodes = nmodes or self.default_nmodes
        pc_cfg = ic_cfg.get("partial_correction", {})

        ifunc_tag, ifunc_inv_tag, pupilstop_tag, reg_tn = self._registered_tags()

        cached, _ = self._cached_imat(rMod, nmodes, seeing, binning, reg_tn, pc_cfg)
        if cached is not None:
            return cached

        tn = _tn_now()
        calib_tag = f"_calib_{self.system}_{tn}"
        pc_info = self._run_specula_simulation(
            ifunc_tag=ifunc_tag, pupilstop_tag=pupilstop_tag, nmodes=nmodes, im_tag=calib_tag,
            mod_amp=rMod, binning=binning, seeing=seeing, ifunc_inv_tag=ifunc_inv_tag, pc_cfg=pc_cfg)
        self._safe_unlink(self.output_dir / ("_overrides_pc.yml" if seeing is not None else "_overrides_dl.yml"))

        imat = fits.getdata(self._im_output_path(calib_tag))[:, :nmodes]
        hdr = self._base_header()
        hdr["TN"] = tn
        hdr["REGTN"] = reg_tn
        hdr["RMOD"] = rMod
        hdr["NMODES"] = nmodes
        hdr["SEEING"] = seeing if seeing is not None else -1.0
        hdr["ISPC"] = seeing is not None
        hdr["BINNING"] = binning
        if pc_info is not None:
            hdr["PCNPERF"] = pc_info["nmodes_perfect_correction"]
            hdr["PCVEC"] = pc_info["correction_vector_path"] or "NONE"
            hdr["PCNAVG"] = pc_info["n_screens_average"]
        out_path = self.output_dir / f"IntMat_{self.system}_{tn}.fits"
        fits.writeto(out_path, imat, header=hdr, overwrite=True)
        return imat

    def compute_reconstructor(self, imat: Union[np.ndarray, str, Path, None] = None,
                               Nmodes: Optional[int] = None,
                               argos: Optional[bool] = None,
                               binning: int = 1,
                               output_dir: Union[str, Path, None] = None) -> np.ndarray:
        """Compute the reconstructor for this system from an interaction
        matrix.

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
            2, 3 or 4 -- selects the matching pupdata/pupids/pupil_mask
            (see `_pupil_geometry`). `total_commands`/`total_slopes`/the
            Rec header template stay constant regardless of binning.
        output_dir : str, Path, or None
            Full directory to write ``RecMat_<system>_<TN>.fits`` into.
            Defaults to this system's output folder (``root_dir/.output``).

        Returns
        -------
        The reconstructor matrix (also saved as
        ``RecMat_<system>_<TN>.fits``, with all relevant parameters in the
        header).
        """
        rec_cfg = self.config["reconstructor"]
        if argos is None:
            argos = rec_cfg.get("argos_default", True)

        geom = self._pupil_geometry(binning)
        npix, nslopes = geom["npix"], geom["nslopes"]
        pup_ids, pupids, half_mask = geom["pup_ids"], geom["pupids"], geom["half_mask"]
        total_commands, total_slopes = rec_cfg["total_commands"], rec_cfg["total_slopes"]

        imat_arr, _, imat_tn = self._resolve_imat(imat)
        if Nmodes is None:
            Nmodes = imat_arr.shape[1]
        imat_arr = imat_arr[:, :Nmodes]  # only reshuffle the columns we keep

        half = nslopes // 2
        # NOTE: xsign/ysign generalise what used to be a `side`-conditional
        # sign flip (unconfirmed for LUCIdx/LUCIsx -- see the config).
        aux = np.zeros_like(imat_arr)
        aux[:half, :] = imat_arr[half:, :] * self.xsign
        aux[half:, :] = imat_arr[:half, :] * self.ysign
        aux *= rec_cfg.get("slopes_scale", 4.0e9)

        IM = np.zeros_like(imat_arr)
        fimg = np.zeros(npix ** 2)
        for i in range(Nmodes):
            f2d = self._raster(fimg, pup_ids, aux[:half, i], aux[half:, i], npix)
            img = f2d[: npix // 2, :]
            IM[pupids, i] = img.flatten()[half_mask.flatten()]

        IMinv = np.linalg.pinv(IM[:nslopes, :Nmodes])
        Rec = np.pad(IMinv, ((0, total_commands - Nmodes), (0, total_slopes - nslopes)))

        # Row 661 <- mode 0, row 668 <- mode 1 (falls back to mode 0 if
        # Nmodes < 2, i.e. mode 1 doesn't exist).
        iir_rows = rec_cfg["iir_command_rows"]
        iir_source_modes = rec_cfg.get("iir_source_modes", [0] * len(iir_rows))
        for row, mode in zip(iir_rows, iir_source_modes):
            src_mode = mode if mode < IMinv.shape[0] else 0
            Rec[row, :] = np.pad(IMinv[src_mode, :], (0, total_slopes - nslopes))

        Rec = Rec.astype(">f4")
        if argos:
            Rec /= 2

        tn = _tn_now()
        hdr = self._build_rec_header(rec_cfg, Nmodes, argos, imat_tn, tn, binning)
        out_dir = Path(output_dir) if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fits.writeto(out_dir / f"RecMat_{self.system}_{tn}.fits", Rec, header=hdr, overwrite=True)
        return Rec

    def plot_registration_check(self, mode_idx: int,
                                 ref_imat: Union[np.ndarray, str, Path, None] = None,
                                 calib_imat: Union[np.ndarray, str, Path, None] = None,
                                 ref_binning: int = 1, calib_binning: int = 1):
        """Visually compare one mode's slopes between a reference
        (typically measured) and a calibrated (simulated) interaction
        matrix -- a generalisation of `update_registration`'s before/after
        check that works from saved products instead of the live fit.

        Parameters
        ----------
        mode_idx : int
            Mode/column index to visualize.
        ref_imat : ndarray, path, or None
            Reference interaction matrix. Defaults to the imat used in the
            current registration's (`self.reg_tn`) `update_registration()`
            call.
        calib_imat : ndarray, path, or None
            Calibrated interaction matrix. Defaults to the latest IntMat
            matching the current registration and
            `compute_interaction_matrix`'s own defaults -- computed if
            none exists yet.
        ref_binning, calib_binning : int
            WFS binning each imat was computed at (default 1 each). If
            they differ, the finer (larger-npix) one is binned down to
            match the coarser one via `toccd` before the difference is
            taken.

        Returns the matplotlib Figure (also saved as
        ``RegCheck_<system>_mode<mode_idx>.png``). The third panel is
        each map normalized by its own STD before differencing.
        """
        ref = self._resolve_ref_imat(ref_imat)
        calib = self._resolve_calib_imat(calib_imat, calib_binning, nmodes=mode_idx + 1)
        ref_geom = self._pupil_geometry(ref_binning)
        calib_geom = self._pupil_geometry(calib_binning)

        ref_2d = self._imat_mode_to_2d(ref, mode_idx, ref_geom)
        calib_2d = self._imat_mode_to_2d(calib, mode_idx, calib_geom)

        # Reconcile different binnings: bin the finer side down to match
        # the coarser one (TODO: toccd's exact averaging/summing behavior
        # is unverified -- no live SPECULA source in this environment).
        if ref_geom["npix"] > calib_geom["npix"]:
            ref_2d, mask_geom = toccd(ref_2d, (calib_geom["npix"],) * 2), calib_geom
        elif calib_geom["npix"] > ref_geom["npix"]:
            calib_2d, mask_geom = toccd(calib_2d, (ref_geom["npix"],) * 2), ref_geom
        else:
            mask_geom = ref_geom

        half_mask, npix = mask_geom["half_mask"], mask_geom["npix"]
        ref_1d = ref_2d[: npix // 2, :][half_mask]
        calib_1d = calib_2d[: npix // 2, :][half_mask]
        diff_1d = ref_1d / ref_1d.std() - calib_1d / calib_1d.std()

        return self._plot_slope_panels(
            [("Reference", ref_1d), ("Calibrated", calib_1d), ("Difference (normalized)", diff_1d)],
            half_mask, f"{self.system} IM check -- mode {mode_idx}",
            f"RegCheck_{self.system}_mode{mode_idx}.png")
