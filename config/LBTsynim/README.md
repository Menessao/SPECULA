# LBTSynIM — user guide

Simulate LBT pyramid-WFS interaction matrices (IMs) and reconstructors with SPECULA, and register the DM to the WFS
from a measured IM. Systems: `LUCIdx`, `LUCIsx`, `LBTIdx`, `LBTIsx`.

## Essentials

- **Run Python from the folder that contains `config/`.** Relative paths (the config file and every path inside it)
  are relative to that folder and become absolute when the object is created. Results go to `./output`
  (`paths.output_dir`); nothing is written to hidden folders or into `root_dir` except what SPECULA must find by tag
  (registered `ifunc/`, `pupilstop/`, and SPECULA's `im/`).
- **A good starting registration is mandatory** before `update_registration`. It is a linearised fit: with a wrong
  `flip`, or a rotation off by more than about a degree, it does not converge. Get the start from
  `initial_registration` (§1).
- **Every IM is in the real system's ("hardware") convention** — row order, `xsign`/`ysign` and units
  (`slopes_scale`) — so simulated and measured IMs are interchangeable for comparing, registering and building
  reconstructors.
- **Binning is not fully tested** (§4).

```python
from lbt_synim import LBTSynIM

lbtidx = LBTSynIM('LBTIdx')                         # reads ./config/LBT_synim_config.yaml
meas = '/path/to/measured_imat.fits'                # measured IM (hardware order), path or array

lbtidx.initial_registration(meas)                   # 1) flip + rotation  (new registration only)
lbtidx.plot_registration_check(30, ref_imat=meas)   #    look at it
lbtidx.update_registration(meas)                    # 2) fit rotation, shifts, magnification
imat = lbtidx.compute_interaction_matrix(rMod=3.0, seeing=1.0)   # 3) simulate IMs (§2)
rec  = lbtidx.compute_reconstructor(imat, Nmodes=550)            # 4) reconstructor (§3)
```

## 1. Registration

Registration = `rotation` [deg], `shift_x`, `shift_y` [pixels] and `magnification` (saved with 2, 2, 2 and 4
decimals), plus the fixed properties `flip`, `xsign`, `ysign` of each system (`flip` mirrors the influence functions
up/down; `xsign`/`ysign` are confirmed for `LBTIdx` and `LBTIsx`, guesses for `LUCIdx`/`LUCIsx`).

### 1.1 Why the starting point matters

`update_registration` measures with small steps (`delta_*`) how the simulated IM changes with each parameter and solves
for a correction. That only works near the solution: if `flip` is wrong, or the rotation is not right to about a
degree, no small correction maps the simulation onto the measurement, and more iterations do not help. Shifts and
magnification matter less.

### 1.2 `initial_registration` — the starting point of a new registration

```python
r = lbtidx.initial_registration(meas)               # save=False to only try it
```

`xsign`, `ysign` and the x/y order are used as configured. It then:

1. simulates a few modes (`calibration_modes`, 5) at the current registration;
2. **rotation from the tilts**: the mean slopes of a tilt are a point in the (pupil 0, pupil 1) plane; the angle between
   the simulated and the measured point is the rotation, averaged over the two tilt modes (found automatically, or
   `tilt_modes`). Both senses of rotation are simulated and the lower residual is kept;
3. repeats with **`flip` negated**; the flip with the lower residual wins (non-negative per-mode gain, so a sign error
   is not hidden);
4. refines the rotation with the standard registration algorithm, **rotation only** (shifts and magnification have
   step 0, so they are not fitted), coarse to fine: `stages` = [number of modes, rotation step in degrees], by default
   10 modes/5°, 20/2°, 50/1°, 100/0.5°.

It sets `flip` and `misreg_guess` on the object (and resets the registration, so IMs use this start) and writes them to
the config (comments kept, backup next to it). It returns an `InitialRegistration` (`flip`, `rotation`,
`rotation_from_tilts`, `flip_residuals`, `tilt_modes`, `tilt_angle_spread` — near 0° when `flip` is right —,
`stages`). It warns if the two flips fit about equally well or neither fits. The measured IM needs at least as many
modes as the largest stage. Settings: `initial_registration` section of the config.

Doing it by hand: `compute_interaction_matrix(nmodes=40)` simulates with the config's `misreg_guess` (no registration
needed); compare with `plot_registration_check(mode, ref_imat=meas, calib_imat=im40)`, edit `flip`/`rotation` in the
config, create a new `LBTSynIM` and repeat.

### 1.3 `update_registration`

```python
res = lbtidx.update_registration(meas)              # path or array; save=False / show_plot=False optional
```

- Needs `registration.nmodes` modes (500) in `meas`; runs at 1×1 binning with the modulation of `syn_soul_im.yml`
  (`pyr.mod_amp`) — keep it equal to the measurement's. Each iteration is 9 SPECULA runs of `nmodes` modes.
- **A parameter whose step (`delta_rotation`, `delta_shift`, `delta_magnification`) is 0 is fixed**: it is not fitted
  and not part of the sensitivity matrix.
- Returns a `RegistrationResult` (`rotation`, `shift_x`, `shift_y`, `magnification`, `iterations`, `tn`) and makes it the
  current registration: registered `IFunc`/`IFuncInv`/`Pupilstop`, `MisReg_<system>_<TN>.fits`, the new `misreg_guess`
  in the config, the plot `MisRegCheck_…png`, and `MeasIM_…fits` if you passed an array.
- **No convergence**: returns `None`, saves nothing, and warns with the final `err` (must fall below `tolerance`).
  Go back to §1.2 — do not raise `max_iterations`. Also check the measurement's modulation and binning. The converged
  values become the next start, so a bad `misreg_guess` keeps hurting: restore a `…_backup_<TN>.yaml` or reset it.

### 1.4 Misregistration from one mode

```python
mm = lbtidx.misregistration_matrix(mode=6)            # 7 single-mode SPECULA runs, then cached
rotation, shift_x, shift_y = mm.estimate(meas[:, 6])  # degrees, pixels, pixels
```

The slopes of one mode, linearised around the **current registration** (`misreg_guess`: the last fit,
`initial_registration`, or the config): `slopes(alpha) ≈ s0 + S·(alpha − alpha0)`, `alpha` = (rotation, shift_x,
shift_y). Steps come from `registration.delta_rotation` and `delta_shift`; magnification is not included. It is much
cheaper than a fit and needs one mode only, but is only valid close to `alpha0`.

**Units and convention** — like every IM here, in hardware order and hardware units (signs and `slopes_scale`
applied), so a column of a measured IM is used as is. Columns of `S` are per **degree** (rotation) and per **pixel**
(shifts) of the DM pupil grid.

`mm` is a `MisregistrationMatrix`: `S` `(nslopes, 3)`, `R = pinv(S)` `(3, nslopes)`, `s0`, `alpha0`, `sensitivity`.
`mm.estimate(measured_column, gain=None)` returns the rotation [deg], shift_x [px], shift_y [px] that match the
measured column; the positive optical gain is estimated unless given. Saved as `MisRegIM_<system>_mode<m>_<TN>.fits`
(columns: S rotation, S shift_x, S shift_y, `s0`); identical calls (same mode, `alpha0`, steps, `flip`, conventions) are
loaded from it.

**Warning for low signal**: a mode cannot determine every parameter (defocus does not respond to rotation, tilts hardly
to shifts). A column is flagged when, in the same unit (response per pixel of displacement at the pupil edge), it is
below `min_relative` (0.2) of the strongest column or below `min_sensitivity` (1e-3). Choose modes that are not
flagged; `mm.sensitivity` shows the values.

### 1.5 Checking

```python
lbtidx.plot_registration_check(mode_idx, ref_imat=None, calib_imat=None, ref_binning=1, calib_binning=1)
```

Reference, calibrated and the difference after dividing each by its standard deviation, for one mode; saved as
`RegCheck_<system>_mode<N>.png`. Defaults: `ref_imat` = the measured IM of the current registration (pass it if there
is none, or its files are gone); `calib_imat` = the saved IM for the current registration — or the config's
`misreg_guess` if none — with default `rMod` and all modes, simulated if missing (slow: pass a quick
`compute_interaction_matrix(nmodes=…)` instead). Judge patterns: low-order modes are easiest; a leftover structure in the
difference means the registration is still off.

## 2. Interaction matrices: rMod, nmodes, seeing

```python
imat = lbtidx.compute_interaction_matrix(rMod=3.0, seeing=None, nmodes=None, binning=1)
```

| Argument | Meaning | Default |
|---|---|---|
| `rMod` | modulation radius [λ/D] | `3.0` |
| `seeing` | arcsec → partial-correction (PC) IM; `None` → diffraction-limited | `None` |
| `nmodes` | modes to calibrate | all m2c modes |
| `binning` | 1–4, see §4 | `1` |

It uses the current registration (or the config's `misreg_guess` if none), runs SPECULA, saves
`IntMat_<system>_<TN>.fits` in the output folder and returns the IM. **Identical calls are cached** (same `rMod`,
`nmodes`, `seeing`, PC settings, `binning`, registration, conventions) — changing any of them recomputes; delete the
file to force it.

```python
imats = {(r, n): lbtidx.compute_interaction_matrix(rMod=r, nmodes=n) for r in (1.5, 2.0, 3.0) for n in (300, 649)}
pc    = {s: lbtidx.compute_interaction_matrix(rMod=3.0, seeing=s) for s in (0.6, 0.8, 1.0, 1.5)}
```

- Fewer modes without recomputing: take columns (`imat[:, :n]`) or use `compute_reconstructor(imat, Nmodes=n)`.
  Exact for diffraction-limited IMs; for PC the modes share one simulated atmosphere, so it is equivalent but not
  identical to a separate run.
- PC settings (`interaction_matrix.partial_correction`): `nmodes_perfect_correction` (modes corrected by the simulated
  loop, 649); `correction_vector_path` — **full path** to a per-mode gain file, used as-is (empty = perfect correction;
  only then `<system>_corr_vec.fits` of ones is generated in the output folder; a path that does not exist is an
  error); `n_screens_average` — atmospheric realisations averaged (cost scales with it).

## 3. Reconstructors

```python
rec = lbtidx.compute_reconstructor(imat, Nmodes=550, argos=True, binning=1, output_dir=None)
```

Pseudo-inverse of the first `Nmodes` columns of a hardware-order IM (simulated **or measured**), padded to the
real-time computer frame (672 × 2848), IIR rows (661, 668) from modes 0 and 1, ARGOS half-gain if `argos`
(default from the config). Saved as `RecMat_<system>_<TN>.fits` (big-endian float32, header from
`reconstructor.header_template`) in the output folder, or `output_dir`. With `imat=None` it uses the **latest**
`IntMat_<system>_*` whatever its parameters — pass the IM explicitly when you have several.

```python
for s, im in pc.items():
    lbtidx.compute_reconstructor(im, Nmodes=550, output_dir=f'./recs/seeing_{s}')
```

## 4. Binning — not fully tested

> **Implemented but not fully tested.** Only 1×1 has been exercised on real pupil files; the SPECULA settings below
> have not been verified on live 2×2/3×3/4×4 runs. Check the first results carefully.

```python
imat2 = lbtidx.compute_interaction_matrix(binning=2)
rec2  = lbtidx.compute_reconstructor(imat2, Nmodes=500, binning=2)
lbtidx.plot_registration_check(30, ref_imat=meas_2x2, ref_binning=2, calib_binning=2)
```

For `binning=b` the SPECULA run gets `pyr.pup_diam`, `pyr.pup_dist`, `pyr.output_resolution` and both `ocam.size`
values divided by `b` (from `optics`; `npix` must be divisible by `b`) and `pyr_slopes.pupdata_object =
lbt_pupdata_{b}x{b}`. You must provide, next to the 1×1 files: `pupils/lbt_pupdata_{b}x{b}.fits` (frame size
`npix/b`) and `pupils/pup_ids_{b}x{b}.fits` (hardware slope-order table; cannot be derived from the pupdata);
`pupils/lbt_pupmask_{b}x{b}.fits` is built automatically if missing.

Limits: registration always runs at 1×1 and binned IMs reuse it; `total_commands`, `total_slopes` and the Rec header
template are assumed identical for all binnings; with different binnings in `plot_registration_check` the finer IM is
binned down with SPECULA's `toccd` (its exact behaviour is unchecked); a binning's geometry is read once per object.

## 5. Reference

**Config** (read when the object is created — create a new one after editing).

| Section | Keys |
|---|---|
| `paths` | `root_dir_template`, file templates (`{kl}`, `{side}`), `.sav` location, `output_dir` |
| `optics` | `telescope_diameter`; 1×1 `pup_diam` (40), `pup_dist` (48), `npix` (120) |
| `systems.<name>` | `kl_version`, `flip`, `xsign`, `ysign`, `misreg_guess` |
| `registration` | `nmodes`, `delta_rotation`/`delta_shift`/`delta_magnification` (0 = fixed), `tolerance`, `max_iterations`, `main_simul_yaml`, `slopes_mode_check` |
| `initial_registration` | `calibration_modes`, `tilt_modes`, `max_iterations`, `stages` |
| `interaction_matrix` | `default_modulation_radius`, `slopes_scale`, `partial_correction`, `pc_extra_blocks_yaml` |
| `reconstructor` | `header_template`, `argos_default`, `total_commands`, `total_slopes`, `iir_command_rows`, `iir_source_modes` |

The registration functions edit only their system's lines (`flip`, `misreg_guess`) after saving
`LBT_synim_config_backup_<TN>.yaml`, and check the file afterwards.

**API**

| Call | Returns |
|---|---|
| `LBTSynIM(system, config_path='./config/LBT_synim_config.yaml')` | the object |
| `initial_registration(measured_imat, save=True)` | `InitialRegistration` |
| `update_registration(measured_imat, save=True, show_plot=True)` | `RegistrationResult` or `None` |
| `misregistration_matrix(mode, min_sensitivity=1e-3, min_relative=0.2)` | `MisregistrationMatrix` (`.estimate(column)`) |
| `compute_interaction_matrix(rMod=3.0, seeing=None, nmodes=None, binning=1)` | IM `(nslopes, nmodes)` |
| `compute_reconstructor(imat=None, Nmodes=None, argos=None, binning=1, output_dir=None)` | Rec `(672, 2848)` |
| `plot_registration_check(mode_idx, ref_imat=None, calib_imat=None, ref_binning=1, calib_binning=1)` | figure |

Attributes: `reg_tn` (current registration, `None` if none), `misreg_guess`, `flip`, `output_dir`, `root_dir`.

**Files** — output folder: `IntMat_`, `RecMat_`, `MisReg_`, `MeasIM_` (`<system>_<TN>.fits`), `MisRegIM_<system>_mode<m>_<TN>.fits`, `MisRegCheck_…png`,
`RegCheck_<system>_mode<N>.png`, `<system>_corr_vec.fits` (fallback only), `_overrides_dl.yml`/`_overrides_pc.yml`
(temporary SPECULA overrides, deleted after a successful run, kept if it fails). Under `root_dir`:
`ifunc/IFunc_`, `ifunc/IFuncInv_`, `pupilstop/Pupilstop_` (`<system>_<TN>` of a fit, or `<system>_cfg_flip<0|1>_<rot>_<sx>_<sy>_<mag>`
for a registration built from the config, `p` = decimal point, `n` = minus), `im/_calib_<system>_<TN>.fits` (raw SPECULA
output, kept), `pupils/lbt_pupmask_{b}x{b}.fits`. `LUCIdx`/`LBTIdx` (and `LUCIsx`/`LBTIsx`) share `root_dir`: do not run
two of them at the same time. Products from an older layout in `<root_dir>/.output` or `.data` must be moved to the
output folder — only that folder is searched.

**Troubleshooting**

| You see | Do |
|---|---|
| "did not converge … returning None" | §1.2; check modulation/binning of the measurement |
| `No registration on record … pass ref_imat` / `Can't find the measured imat …` | pass `ref_imat=` (nothing fitted, or the fit's files are gone) |
| `… saved in raw SPECULA order` | old `IntMat_<system>_<TN>.fits` without `ORDER='hardware'`: delete and recompute |
| `ValueError … entries but … subapertures` / mask does not cover | pupdata, `pup_ids` and mask do not belong together (wrong binning files) |
| warning "frame size …, expected …" | pupdata frame size ≠ `optics.npix / binning` |
| `FileNotFoundError: correction_vector_path` | the configured path does not exist |
| `CalledProcessError … specula` | re-run by hand with the override yaml left in the output folder. A failed registration may leave `*_tmp.fits` files (`IFunc_`, `IFuncInv_`, `Pupilstop_`, `im/tmp_synim_tmp.fits`) — safe to delete |
| `Could not update … in the config` | config left unchanged (backup next to it); check the system's block |
| edits to the config have no effect | create a new `LBTSynIM` |

**Unverified assumptions**: binning (§4); `xsign`/`ysign` for `LUCIdx`/`LUCIsx` (they set the sign of every saved IM
and reconstructor); the vector for `scale_random.constant_mul_data` is passed as the file's full path without `.fits`
(change the `return` in `_correction_vector_tag` to `path.stem` if your SPECULA looks tags up in its data directory);
IIR row 668 fed from mode 1; `initial_registration` assumes the first modes include two tilts and that the
configured `xsign`/`ysign` and x/y order are right.
