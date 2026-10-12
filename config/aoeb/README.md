# aoeb — error budget for pyramid-WFS AO

Two main pieces: `generation.ErrorBudgetGenerator` runs the SPECULA calibrations and post-processes them;
`budget.ErrorBudgetMachine` reads the products and computes the per-mode budget (assumptions in
`ASSUMPTIONS.md`). The budget side needs only numpy, scipy, pandas, astropy (no SPECULA); matplotlib for plots.

```
aoeb/
  generation.py         ErrorBudgetGenerator           SPECULA runs + post-processing
  budget.py             ErrorBudgetMachine             fast per-mode budget, scans (numpy / scipy / pandas / astropy only; does not import psf.py)
  filter_optimizers.py  LoopTables (integrator / leaky), IirControllers + IirTables (IIR, global gain), evaluate_controllers
  noise.py              Detector, PupilGeometry, slope noise coefficients
  reconstructors.py     SVD / ML / MAP
  utils.py              r0, radial order, KL variances, pupil geometry, KL checks
  naming.py             the file layout (one place)
  psf.py                PsfEngine: radial PSD of every error term and long-exposure PSF (terms, total, first-order halos)
  chromatic.py          shear (always) and optional multiplicative / Fresnel OPD / scintillation terms; TipTop's airRefraction for n - 1
  plotting.py           budget, calibration, PSD and PSF figures (returned, never saved)
  config.py             one YAML file for a whole study -> machine / generator / chromatic model / scan
  config/defaults.yml   THE defaults of the study configuration, every key documented;  config/calib_alias_sn_thrp.yml  SPECULA aliasing run
  scripts/run.py        python -m aoeb.scripts.run {scan|generate|show} study.yml [section.key=value ...]
  tests/                106 tests, synthetic data: `pip install -e .[test]; pytest` (pyproject.toml), or `python aoeb/tests/run_all.py` without pytest
```

## 1. Generate the products (once per configuration)

From the command line (nothing to write, every ingredient is in `aoeb/config`):

```
python -m aoeb.scripts.generate --root /raid1/.../RISTRETTO                                  # all steps, packaged ymls, defaults
python -m aoeb.scripts.generate study.yml wavelengths.sensing_nm=[750,1650] --steps im_dl recs # a study file / overrides, some steps
aoeb-generate --list-steps          # (installed by pip install -e .)   steps: pupdata alias im_dl recs aliases pc pc_levels
```
`--overwrite` redoes existing products, `-q` logs only warnings; every SPECULA run and its progress are logged. The study file and the
`section.key=value` overrides are those of the configuration (section 5; `config/defaults.yml` lists every key). From Python:

```python
from aoeb.generation import ErrorBudgetGenerator, GeneratorConfig
gen = ErrorBudgetGenerator(GeneratorConfig(root_dir='/raid1/.../RISTRETTO', wavelengths_nm=(750., 1650.), science_nm=1650.,
                                           pc_corrections=((0.99, 0.2), (0.9, 0.2))))      # the rest: defaults
gen.run()                       # or gen.run(['pupdata', 'im_dl', 'recs']); gen.runs keeps every SPECULA call and its overrides
```

**The SPECULA files** are all in `aoeb/config` (`generation.config_dir` / `paths.specula_dir` to use your own): `ristretto_main.yml`
(the base simulation, adapted from your example: pyramid and NGS at the same wavelength, science source propagated, `prop` output typo
fixed), and, applied on top of it as extra ymls: `calib_pupdata.yml`, `calib_im.yml` (diffraction-limited IM),
`calib_alias_sn_thrp.yml` (aliasing, slope null, throughput, noise illumination), `calib_pc.yml` (IM on a partially corrected
atmosphere: **one file for perfect and partial correction, only the correction vector differs**). Their headers list what the generator
overrides.

**How SPECULA is driven** (the logic of `LBTSynIM._run_specula_simulation`): overrides are nested dicts `{block: {param: value}}`
written as `<block>_override` YAML (`write_overrides_yaml`) and passed as `specula main.yml extra.yml overrides.yml` (`run_specula`, an
argument list, one unique temporary override file per run; the only place where SPECULA is started is `ErrorBudgetGenerator._launch`);
`main.root_dir` is the products root, and `main.pixel_pupil` / `main.pixel_pitch` (= diameter / pixel_pupil) always come from the size of the
pupilstop array (`generation.pixel_pupil` to force it), so the atmosphere, the pupil stop and the DM influence functions agree. **Every independent atmospheric realisation is its own SPECULA run with
`atmo_random.seed = seed0 + i + 1`**: the aliasing runs (`n_screens_alias` per case) are pooled, the PC IMs (`n_screens_pc` = 70 runs of
ONE push-pull cycle, `total_time = 2 N time_step`, one screen each) are averaged. Leading all-zero frames are dropped from the aliasing
runs (plus `discard_frames`, default 0) and the pooled frames must pass a lag-1 independence check (a warning otherwise).

Order of the steps: pupil data (once, shared) -> aliasing runs per (wavelength, rMod, n_subap, seeing, N corrected) that yield the
aliasing, the slope null, the throughput and the noise illumination -> diffraction-limited IM -> SVD reconstructors -> aliasing modal
signal std(R slopes) -> IM with N modes perfectly corrected -> (optional) IMs with a partial correction. Existing products are skipped
(`overwrite=True` to redo). The static inputs (pupilstop, KL basis) are checked before anything starts.

* **Wavelengths.** The pyramid and its source must have the same wavelength: `wavelengths_nm` (sensing, required) sets
  `pyr.wavelengthInNm` and `source_ngs.wavelengthInNm` together in every run. The science source (`science_nm`) is propagated like the NGS
  (same screens, same DM) in the aliasing run. The pupil data are made once (first wavelength), the other products carry `_wvl{nm}`
  (`pyr3.0_48x48_wvl1650_s0.9_1200modes_im_pc.fits`).
* **DM objects.** One tag, `dm_tag` (`bmc2k_vlt`), names them: `{dm_tag}_ifunc`, `{dm_tag}_m2c`, `{dm_tag}_kl_inv`, in SPECULA (overrides)
  and in the budget machine alike.
* **Correction vectors** (`generation.save_correction_vector`, as your reference script): `<root>/data/correction_vector_{N}modes_c{max}-{min}.fits`
  (BaseValue; header VERSION, OBJ_TYPE, NDARRAY, MAX_CORR, MIN_CORR; ONE function for the perfect and the partial correction); the vector multiplies the
  reconstructed modes that command the DM, which SPECULA applies as a correction (negative sign): 1 = corrected, 0 = uncorrected. Factor `max_corr` at the tip-tilt decreasing linearly with the radial order to
  `min_corr` at the highest order (the same for all modes of an order, `min_corr` for modes no order covers, 0 beyond the corrected
  modes). `max = min = 1` is the perfect correction (`..._im_pc.fits`); `generation.pc_corrections: [[0.99, 0.2], ...]` also makes
  `..._im_pc_c0.99-0.20.fits`, each costs one more PC set. The budget machine turns every level into the rms residual of the
  controlled modes, sqrt(sum (1 - c_i)^2 sigma_i^2) (analytic), and `budget(..., fixed_point=True)` interpolates the optical gain there.

ML / MAP reconstructors matched to a flux, rate, detector and seeing: `gen.compute_recs_weighted('map', flux, fs, detector, seeing, machine)`,
used with `machine.budget(..., rec=('map', n_phot))` (the seeing is that of the calibration).

## 2. Budget

```python
from aoeb.budget import ErrorBudgetMachine
from aoeb.noise import Detector

det = Detector(ron_e=0.5, qe=0.2, excess_noise=True)       # EMCCD: readout_level 0.5, F2 = 2
m = ErrorBudgetMachine('/raid1/.../RISTRETTO_vis', diameter=8.2, L0=25., wind=12., detector=det)
r = m.budget(rMod=3, n_subap=48, seeing=0.9, flux=1e7, fs=1500., latency_s=500e-6,
             controller='leaky', scenario='standard', wavelength_nm=1650.)
r.totals()                       # lag, noise (+ noise_photon, noise_ron), alias, xtalk, fitting, total [nm^2], total_rms [nm]
r.gain, r.ff, r.og, r.residual   # per mode

modes, totals = m.scan(seeings, rMods, n_subaps, fluxes, freqs, scenarios=('standard', 'multistage'),
                       wavelengths=(750., 1650.), store_modes=list(range(150)))
```
From a configuration file (section 5): `python -m aoeb.scripts.run scan study.yml wavelengths.sensing_nm=[750,1650]`.
`scan(..., on_missing='skip')` records configurations with missing products in `machine.skipped` instead of aborting; `scenario` and
`controller` are validated; a missing product raises an error that names the generator step that creates it.

* `flux`: photons/s at the detector (before QE). `latency_s`: latency from the end of the exposure to the command applied, any
  real value (fractional frames allowed); the frame of exposure window + DM hold is part of the model, never add it.
* One physical model (ASSUMPTIONS A11): continuous exposure averaging, zero-order-hold DM, exact loop gain (a pure delay of
  1 + latency frames, no sinc), continuous-time residual. For a SPECULA run use `latency_s = delay × time_step`.
* `fixed_point=True`: optical gains interpolated at the resulting residual of the controlled modes (needs the partial-correction levels).
* `controller='iir'`-like: pass `IirControllers` (per-mode IIR filters in the SPECULA `IirFilterData` convention): the shape is kept and only the
  global gain of each filter is optimised (`result.gain` is the value for `IirFilterData.set_gain`):
  ```python
  from aoeb.filter_optimizers import IirControllers
  ctl = IirControllers.from_iir_filter_data(iir)         # or .from_fits(path), .from_gain_and_ff(gain, ff), IirControllers(num, den)
  r = m.budget(..., controller=ctl)                      # IirControllers(..., optimize_gain=False) evaluates the filters as given
  iir.set_gain(r.gain)
  ```
  Filters must be causal and stable for 0 < x <= 0.98 x_crit (checked); a filter with a zero gain leaves its mode uncontrolled.
  In a config file: `loop.controllers: [int, iir]` with `loop.iir_file`.
* Timing (1200 modes): tables of one loop ≈ 10 ms (integrator) / 50 ms (leaky, 11 ff), reused for all flux / seeing / rMod /
  sensors; then ≈ 0.8 ms (integrator) / 2 ms (leaky) per budget point; 1000 points over 25 loops: 1.2 s / 3.2 s.
* `crosstalk=True`: leak of the residual of the other modes into each measurement (row of R·IM_PC, iterated; incomplete for the
  multistage case, A16).
* `chromatic=ChromaticModel(...)`: adds the shear term and the enabled optional terms (the multiplicative one makes the gains
  optimal for the science residual), see below.

## 3. PSD and PSF of the residual

```python
from aoeb.chromatic import ChromaticModel
from aoeb.config import DEFAULTS                       # the Ristretto layers live in config/defaults.yml (cfg.chromatic() builds it)
from aoeb.psf import PsfEngine                         # separate from the budget: budget.py does not import psf.py
atm = DEFAULTS['atmosphere']
chrom = ChromaticModel(atm['heights'], atm['cn2'], science_nm=1650., zenith_deg=40.,
                       multiplicative=False,                # optional, default off (A24, A26): loop-dependent, gains optimised for it
                       fresnel=False, scintillation=False)  # optional, default off
r = m.budget(rMod=3, n_subap=48, seeing=0.9, flux=1e7, fs=1500., wavelength_nm=750., crosstalk=True, chromatic=chrom)
eng = PsfEngine(m)                   # (or cfg.psf_engine()); caches the KL kernels and the telescope OTF
ps = eng.psd_terms(r)                # lag, noise_photon, noise_ron, alias, xtalk, fitting, chrom_shear, chrom_mult, total, turbulence (open loop)
res = eng.psf_terms(r, 1650.)        # long-exposure PSF of every term alone and of the total, any science wavelength
res.profile('total'); res.contrast_at([2, 5, 10]); res.halo['lag']      # halos (first order) add up
pl.plot_psd_terms(ps); pl.plot_psf_terms(res, rmax=30); pl.plot_psf_image(res, 'total')
```
Conventions and limits: ASSUMPTIONS A15, A24–A28.
TipTop check of the shear term: ASSUMPTIONS section 4 (what to print in TipTop and compare).

## 4. Plots

All functions return the matplotlib figure; nothing is saved. From the scan DataFrames: `plot_budget_terms`, `plot_budget_compare`,
`plot_per_mode`. From a machine: `plot_calibration(machine, 'og' | 'alias' | 'ct' | 'noise' | 'turb' | 'thrp', hue=..., values=...)`,
`plot_modal_variance_check(machine)`; from a budget result: `plot_psd_terms(eng.psd_terms(r))`, `plot_psf_terms(eng.psf_terms(r, 1650.))`,
`plot_psf_image(...)`. Use `fig.savefig(...)` yourself if you want a file.

## 5. Configuration file

`config/defaults.yml` is the single source of the defaults and documents every key (only `paths.root_dir` is required); unknown keys are errors;
floats such as `1e6` or `5e-4` are read as floats. Copy the file and keep the keys you change.
Sections: `paths`, `telescope`, `atmosphere` (one profile for the effective wind and the chromatic terms; `from_specula` reads the
`atmo` section of a SPECULA yml), `sensors`, `detector`, `wavelengths`, `loop`, `scan`, `chromatic`, `options`, `generation`, `output`.

```python
from aoeb.config import Config
cfg = Config.load('study.yml', ['loop.latency_s=0.0008', 'sensors.n_subaps=[48]'])   # overrides: section.key=value (YAML values)
modes, totals = cfg.scan()                  # machine + chromatic model + every loop / scan option from the file
m, gen, chrom = cfg.machine(), cfg.generator(), cfg.chromatic()      # the objects, to use them directly
cfg.dump('results/study_config.yml')        # the resolved configuration, for reproducibility
```
The command line does the same (`scan` also writes `<output.prefix>_totals.csv`, `_modes.csv` and `_config.yml`):
`python -m aoeb.scripts.run scan study.yml`, `... generate study.yml` (= `python -m aoeb.scripts.generate`, section 1), `... show study.yml`.
