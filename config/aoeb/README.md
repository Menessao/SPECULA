# aoeb — error budget for pyramid-WFS AO

Two main pieces: `generation.ErrorBudgetGenerator` runs the SPECULA calibrations and post-processes them;
`budget.ErrorBudgetMachine` reads the products and computes the per-mode budget (assumptions in `ASSUMPTIONS.md`). The budget side
needs only numpy, scipy, pandas, astropy (no SPECULA); matplotlib for plots.

```
aoeb/
  generation.py         ErrorBudgetGenerator           SPECULA runs + post-processing
  budget.py             ErrorBudgetMachine             per-mode budget, fusion of several sensors, scans
  filter_optimizers.py  LoopTables (integrator / leaky), aliasing temporal spectra, evaluate_controllers
  wiener.py             per-mode Wiener filters on pseudo-open-loop measurements (controller 'wiener')
  noise.py              Detector, PupilGeometry, slope noise coefficients
  reconstructors.py     SVD / ML / MAP
  psf.py                PsfEngine: radial PSD of every error term, long-exposure and coronagraphic PSF; KL fitting bound
  chromatic.py          chromatic shear (always) and optional scintillation; TipTop's airRefraction for n - 1
  multiwfs.py           Study: single- versus multi-sensor comparisons ranked by coronagraphic contrast, disk cache
  plotting.py           calibration, PSF and comparison figures (returned, never saved)
  utils.py, naming.py   r0, radial order, KL variances, pupil geometry; the file layout
  temporal.py           exact multi-layer frozen-flow temporal spectra (used by the opt_iir studies)
  config.py             one YAML file for a whole study -> machine / generator / chromatic model / scan
  config/defaults.yml   THE defaults of the study configuration, every key documented
  scripts/              python -m aoeb.scripts.run {scan|generate|show} study.yml;  python -m aoeb.scripts.generate
```

## 1. Generate the products (once per configuration)

```
python -m aoeb.scripts.generate --root /raid1/.../RISTRETTO                                  # all steps, packaged ymls, defaults
python -m aoeb.scripts.generate study.yml wavelengths.sensing_nm=[750,1650] --steps im_dl recs # a study file / overrides, some steps
aoeb-generate --list-steps          # steps: pupdata alias im_dl recs aliases pc
```
`--overwrite` redoes existing products, `-q` logs only warnings. From Python:
```python
from aoeb.generation import ErrorBudgetGenerator, GeneratorConfig
gen = ErrorBudgetGenerator(GeneratorConfig(root_dir='/raid1/.../RISTRETTO', wavelengths_nm=(750., 1650.), science_nm=1650.))
gen.run()                       # or gen.run(['pupdata', 'im_dl', 'recs']); gen.runs keeps every SPECULA call and its overrides
```

**The SPECULA files** are in `aoeb/config` (`paths.specula_dir` to use your own): `ristretto_vis.yml` (base simulation) and, on top
of it, `calib_pupdata.yml`, `calib_im.yml` (diffraction-limited IM), `calib_alias_sn_thrp.yml` (aliasing, slope null, throughput, noise
illumination), `calib_pc.yml` (IM with the N controlled modes perfectly corrected). Overrides are nested dicts written as
`<block>_override` YAML and passed as `specula main.yml extra.yml overrides.yml`; `main.pixel_pupil` / `pixel_pitch` come from the pupil
stop. **Every independent atmospheric realisation is its own SPECULA run with `atmo_random.seed = seed0 + i + 1`**: the aliasing runs
(`n_screens_alias` per case) are pooled, the PC IMs (`n_screens_pc` = 70 runs of one push-pull cycle) are averaged. Every configuration
gets the same seeds, so the aliasing runs of two sensors see the same turbulence frames (the fusion uses this, A39).

Steps: pupil data -> aliasing runs per (wavelength, rMod, n_subap, seeing, N corrected) -> diffraction-limited IM -> SVD reconstructors ->
aliasing modal signal std(R slopes) -> IM with N modes perfectly corrected (correction vector of ones, `save_correction_vector`).
The pyramid and its source share the sensing wavelength (`wavelengths_nm`); the products carry `_wvl{nm}`. The DM objects are
`{dm_tag}_ifunc`, `{dm_tag}_m2c`, `{dm_tag}_kl_inv` in SPECULA and in the budget machine alike.

ML / MAP reconstructors matched to a flux, rate, detector and seeing: `gen.compute_recs_weighted('map', flux, fs, detector, seeing, machine)`
(or `machine.ensure_rec`), used with `machine.budget(..., rec=('map', n_phot))`.

## 2. Budget of one sensor

```python
from aoeb.budget import ErrorBudgetMachine
from aoeb.noise import Detector

m = ErrorBudgetMachine('/raid1/.../RISTRETTO_vis', diameter=8.2, L0=25., wind=12., detector=Detector(ron_e=0.5, qe=0.2))
r = m.budget(rMod=3, n_subap=48, seeing=0.9, flux=1e7, fs=1500., latency_s=500e-6, controller='leaky', wavelength_nm=900.)
r.totals()                       # lag, noise (+ noise_photon, noise_ron), alias, xtalk, fitting, chromatic terms, total: nm rms (QUADRATURE)
r.variances()                    # the same in nm^2 (they add)
r.gain, r.ff, r.og, r.residual   # per mode (per-mode arrays are variances [nm^2])
modes, totals = m.scan(seeings, rMods, n_subaps, fluxes, freqs, controllers=('int', 'leaky'), wavelengths=(900., 1300.))
```

* **Controllers** (`controller`): `'int'` integrator; `'leaky'` gain and leak (`DEFAULT_FF_GRID`); `'wiener'` the optimal order-2 IIR
  filter per mode acting on pseudo-open-loop measurements (`wiener.py`: Wiener-Hopf bound, Steiglitz-McBride fit, refinement on the exact
  cost; `result.filters` = per-mode (b, a), `wiener.to_iir_filter` converts them to the SPECULA `IirFilterData` convention). The gains are
  limited only by stability (98 % of the critical gain). One model (A11): continuous exposure, zero-order-hold DM, latency `latency_s`
  from the end of the exposure to the command (any real value; for a SPECULA run `delay x time_step`).
* **Reconstructors** (`rec`): `('svd',)`, `('ml', n_phot)`, `('map', n_phot)` (ML / MAP: full sx-sy noise covariance, MAP prior from the
  seeing).
* **Frame rate and modes are maxima**: with `optimize_freq` (default) the rates below `fs` on the grid 100 Hz x 2^(k/3) are tried, with
  `optimize_modes` the calibrated mode counts below `N` (`modes_per_subap`); the best is returned (`params['freqUsedInHz']`,
  `params['NmodesUsed']`), with a warning when it is not (fs, N).
* **Fitting** (A15): the bound of the N controlled modes, i.e. the turbulence inside the pupil with its projection on the first N modes
  of the KL basis in use removed perfectly, exactly in expectation (`psf.kl_fitting_psd`); its variance is the budget's fitting term and
  its PSD the PSF's. Cost: ~1 min for the cross kernels and ~20 s per N, once, cached in `<root>/data` (`kl_fitting_cross_*`,
  `kl_fitting_psd_*`).
* **Aliasing**: the calibrated modal std, rescaled to the seeing (variance ∝ seeing^5/3); temporal shape `AliasSpectrum` (A10), or the
  MEASURED per-radial-order shapes where given (`alias_psd`, A10b; the model elsewhere, with a warning).
* `crosstalk=True`: leak of the residual of the other modes into each measurement (rows of R·IM_PC, iterated).
* `chromatic=ChromaticModel(...)`: the chromatic shear (always) and, with `scintillation=True`, the scintillation below k_c (A24–A27).
* **Seeings and wavelengths that were not simulated** are derived (warnings): optical gains and noise coefficients by regression over
  the simulated seeings, aliasing and crosstalk from the nearest one; a wavelength without products from the nearest calibrated one at the
  equivalent seeing s·(λ₀/λ)^(6/5), IMs × λ₀/λ, reconstructors recomputed and saved (A37).
* **Slope normalisation** is measured from the noise-free runs (mean slopes against the mean frame), which also checks the slope ordering.

## 3. Several sensors (fusion, A39)

```python
s1 = dict(rMod=2, n_subap=48, wavelength_nm=900., flux=f1)
s2 = dict(rMod=0, n_subap=48, wavelength_nm=1300., flux=f2)
r = m.budget_fused([s1, s2], seeing=0.9, fs=2000., controller='leaky', chromatic=chrom)    # one loop, fused measurements
r.weights                                      # (K, N) per-mode weights; r.params['weight_i']: turbulence-weighted mean
d = m.budget_dual(dict(s1, N=100, fs=1000.), dict(s2, fs=2000.), 0.9, controller='leaky')  # first 100 modes fused, the rest HO alone
```
The measurements of every mode are combined with static weights summing to 1, by generalised least squares on the per-mode error
covariance of the sensors: noise (independent between sensors) + aliasing (correlation measured from the paired aliasing runs,
`m.alias_correlation`) + chromatic shear of the sensing beams with respect to the science one (`m.shear_covariance`), so that with two
bands the weights extrapolate towards the science wavelength when that pays. The loop is then optimised as for one sensor. Within 0.1–0.2
nm of the optimal causal fusion (frequency-dependent weights) for synchronous sensors (`doc/wfs_fusion.tex`). `budget_dual`: the LO sensor
at its own (slower) rate enters with the same noise PSD at the loop rate, its longer exposure and delay are not modelled (optimistic).

## 4. PSD and PSF of the residual

```python
from aoeb.psf import PsfEngine
eng = PsfEngine(m)                   # (or cfg.psf_engine()); uses the machine's KL kernels and fitting bounds
ps = eng.psd_terms(r)                # lag, noise_photon, noise_ron, alias, xtalk, fitting, chrom_shear, (scintillation), total, turbulence
res = eng.psf_terms(r, 1650.)        # long-exposure PSF of every term alone and of the total, at the science wavelength
res.contrast_at([2, 5, 10], kind='coro') / res.image('total').max()     # raw contrast after a perfect coronagraph, AO-peak normalised
```
`kind='psf'`: long-exposure PSF (peak of the diffraction-limited PSF = 1). `kind='coro'`: **after a perfect coronagraph**, Cavarroc et al.
(2006, A&A 447, 397) eq. (1): for Gaussian phase the long-exposure intensity is exactly PSF − e^{−σ²}·PSF_DL. The image of a term alone is
not its contribution to the total (inside the total its halo is attenuated by the Strehl of the other terms): contrasts of several terms
must be computed jointly. Conventions: A15, A24–A28.

## 5. Comparisons (`multiwfs.Study`) and plots

`Study(machine, engine, seeing, latency_s, controller, area, bandw, colour, cache_dir)` compares single, fused, dual systems against
the magnitude, every choice (binning, frame rates, light split, sampling) ranked by the mean coronagraphic raw contrast in a zone
(relative to the AO PSF peak, as the e2e runs), the candidates with science SR ≥ 10 % first; sweeps are cached on disk under a hash of
every setting (bump `multiwfs.VERSION` after a code change). `config/analyse_errorbudget.ipynb` drives it.

`aoeb.plotting` returns figures and saves nothing: `plot_calibration(machine, product, hue, values, ...)` (optical gain, aliasing,
crosstalk, noise, turbulence per mode, or throughput), `plot_psf_terms(psf_result, kind='psf' | 'coro', ...)` (radial profiles of every
term and the total), and the comparison figures of the notebook (`plot_comparison`, `plot_noise_frame_rate`, `plot_fusion_gains`,
`plot_scan`).

## 6. Configuration file

`config/defaults.yml` is the single source of the defaults and documents every key (only `paths.root_dir` is required); unknown keys are
errors. Sections: `paths`, `telescope`, `atmosphere` (`from_specula` reads the `atmo` section of a SPECULA yml), `sensors`, `detector`,
`wavelengths`, `loop`, `scan`, `chromatic`, `options`, `generation`, `output`.
```python
from aoeb.config import Config
cfg = Config.load('study.yml', ['loop.latency_s=0.0008', 'sensors.n_subaps=[48]'])   # overrides: section.key=value (YAML values)
modes, totals = cfg.scan()                  # machine + chromatic model + every loop / scan option from the file
m, gen, chrom = cfg.machine(), cfg.generator(), cfg.chromatic()
```
The command line does the same: `python -m aoeb.scripts.run scan study.yml` (writes `<output.prefix>_totals.csv`, `_modes.csv`,
`_config.yml`), `... generate study.yml`, `... show study.yml`.
