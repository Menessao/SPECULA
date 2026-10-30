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
r.totals()                       # lag, noise (+ noise_photon, noise_ron), alias, xtalk, fitting, total: nm rms (they add in QUADRATURE)
r.variances()                    # the same terms in nm^2 (they add linearly)
r.gain, r.ff, r.og, r.residual   # per mode (r.residual and the other per-mode arrays are variances [nm^2])

modes, totals = m.scan(seeings, rMods, n_subaps, fluxes, freqs, scenarios=('standard', 'multistage'),
                       wavelengths=(750., 1650.), store_modes=list(range(150)))
```
From a configuration file (section 5): `python -m aoeb.scripts.run scan study.yml wavelengths.sensing_nm=[750,1650]`.
`scan(..., on_missing='skip')` records configurations with missing products in `machine.skipped` instead of aborting; `scenario` and
`controller` are validated; a missing product raises an error that names the generator step that creates it.

**Slope noise normalisation.** PyrSlopec computes s = (A+B−C−D)·n_norm/I_total with n_norm = `PupData.n_subap`. The machine
**measures** n_norm from the products of the same noise-free runs (mean slopes `sn` against the mean frame `avgframe`; correlation > 0.9
required), which also checks the slope ordering the noise propagation assumes; it warns and falls back to the number of valid
subapertures when the fit fails, and warns when the measured value differs from that number by more than 2 %.

**Frame rate and modes are maxima.** `budget(..., fs, N)` (and the `fs` values of a scan) are upper limits: with
`optimize_freq=True` (default; `options.optimize_freq`) the frame rates below `fs` on the fixed grid `freq_min`·2^(k/3) (`freq_min` = 100 Hz)
are tried too, and with `optimize_modes=True` (default) the calibrated mode counts below `N`; the configuration with the lowest total
residual is returned, with a warning when it is not (fs, N) (a scan gives one summary warning instead). The result keeps both:
`params['freqInHz']` / `params['Nmodes']` are the requested maxima (the axes of scans and plots), `freqUsedInHz` / `NmodesUsed` the values
used (also columns of the scan tables). Cost: about 13 frame rates per call below 2 kHz (the loop tables are cached on the shared grid).
Pass `optimize_freq=False` / `optimize_modes=False` to a call, or set them on the machine, for one configuration as given.

**Number of modes.** Per call: `m.budget(..., N=800)`. Per sensor: `modes_per_subap` (machine argument; `sensors.modes_per_subap` in a
configuration file), an int or a list, e.g. `{48: [800, 1000, 1200]}`: the largest is the default and the maximum, the others are the
candidates of `optimize_modes`. Every count needs its products, so list it in the generator's `modes_per_subap` too (it then makes the
reconstructor, the aliasing runs and the IMs of each count); a count without products is skipped with a warning.

**Robustness (modulus margin).** The gains are optimised with the peak rejection max|1/(1+L)| ≤ `loop.max_sensitivity` = 2 (6 dB, the usual
AO design margin); `null` lets them go up to 98 % of the stability limit as before, which is optimistic (the lag error is then about half)
and fragile against optical-gain errors. Note that a small fractional latency raises the stability limit of a pure integrator (the exposure
averages the old and the new command, which damps the Nyquist resonance), so a lower latency does not always reduce the lag error.

**Fitting.** `ErrorBudgetMachine(..., asm=True)` (default; `options.asm` in a configuration file) uses the LBT-ASM law 0.2778 N^−0.9
(D/r0)^5/3 rad²; `asm=False` uses TipTop's fitting: the von Kármán PSD outside the square of corrected frequencies |kx|, |ky| ≤ k_c,
k_c = √(N/π)/D (a DM with N actuators in the pupil), for the variance and for the PSD / PSF (the azimuthal average of the square mask).

**Units.** Every error the tables and `totals()` report is in **nm rms**: the scan tables (`totals` and the `_totals.csv` file: lag, noise,
noise_photon, noise_ron, alias, xtalk, fitting, the chromatic terms, total) and the per-mode table (`modes` / `_modes.csv`: residual, lag,
noise, noise_photon, noise_ron, alias, xtalk, chrom_mult). The terms add in quadrature (total² = Σ term²; noise² = noise_photon² +
noise_ron²); a term whose variance is negative (the signed cross term `chrom_mult`) is shown as −sqrt|v|. `variances()` gives the nm² values;
the arrays of a `BudgetResult` (`r.lag`, `r.residual`, ...) stay variances.

**Seeings that were not simulated.** The products of the seeing rounded to 0.1" are used when that seeing was simulated (the aliasing
is rescaled to the exact seeing, variance ∝ seeing^(5/3)). Otherwise the calibration is estimated from the simulated seeings, with a warning
(once per configuration; it also says when the seeing is outside the simulated range, i.e. extrapolated): optical gains (perfect and
partial correction) and noise coefficients by a quadratic least-squares fit against the simulated seeings (linear with two, constant with
one), aliasing and crosstalk from the nearest simulated seeing (aliasing rescaled as above). The turbulence and fitting terms are always
analytic at the exact seeing.

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
res.contrast_at([2, 5, 10], kind='coro'); res.coro['total']              # after a perfect coronagraph (Cavarroc et al. 2006, eq. 1)
pl.plot_psd_terms(ps); pl.plot_psf_terms(res, rmax=30); pl.plot_psf_image(res, 'total')
```
Conventions and limits: ASSUMPTIONS A15, A24–A28.
TipTop check of the shear term: ASSUMPTIONS section 4 (what to print in TipTop and compare).

## 4. Plots

Everything is in `aoeb.plotting` (needs matplotlib: `pip install -e .[plots]`). Every function **returns** the matplotlib figure and saves
nothing: `fig.savefig('name.png', dpi=150)`; `plt.show()` in an interactive session; `plt.close(fig)` in long loops. Errors are in nm rms.
Single-panel functions accept `ax=` to draw into an existing axes (subplots of your own). All accept `title=`, `xlim=(lo, hi)` and
`ylim=(lo, hi)` (`None` on one side keeps it automatic) and have `logx` / `logy` switches.

### 4.1 Budget plots, from a scan

They take the tables `m.scan(...)` returns, or the CSV files written by `python -m aoeb.scripts.run scan study.yml`
(`<output.prefix>_totals.csv`; `_modes.csv` only with `scan.store_modes`):

```python
import pandas as pd
from aoeb import plotting as pl

totals = pd.read_csv('results/scan_totals.csv')              # or: modes, totals = m.scan(...)
modes = pd.read_csv('results/scan_modes.csv')
one = {'scenario': 'standard', 'controller': 'int', 'rMod': 3, 'seeing': 0.9, 'Nsubaps': 48, 'wavelength_nm': 750}

# 1. every term of ONE configuration versus flux
fig = pl.plot_budget_terms(totals, x='photPerMs', filters={**one, 'freqInHz': 1000},
                           terms=('lag', 'noise_photon', 'noise_ron', 'alias', 'fitting', 'total'))
fig.savefig('terms_vs_flux.png', dpi=150)

# 2. one quantity, a curve per frame rate, a panel per sensor
fig = pl.plot_budget_compare(totals, x='photPerMs', hue='freqInHz', quantity='total', col='Nsubaps',
                             filters={k: v for k, v in one.items() if k != 'Nsubaps'})

# 3. per-mode residual of one configuration, a curve per flux
fig = pl.plot_per_mode(modes, y='residual', hue='photPerMs', filters={**one, 'freqInHz': 1000})
```

| function | draws | options |
|---|---|---|
| `plot_budget_terms(totals, x, filters, terms)` | the error terms [nm rms] of one configuration versus `x` (thick line: total) | `logx=True`, `logy=False`, `xlim`, `ylim`, `ax` |
| `plot_budget_compare(totals, x, hue, quantity, filters, col)` | one column (`quantity`, default `total`) versus `x`, one curve per value of `hue`, one panel per value of `col` | `logx`, `logy`, `xlim`, `ylim` (all panels) |
| `plot_per_mode(modes, y, hue, filters)` | a per-mode column versus `modeId` (errors in nm rms; `gain`, `ff`, `og` as they are) | `logy=True`, `logx=False`, `xlim`, `ylim` |

* `x`, `hue`, `col` are any column: `photPerMs` (photons/ms at the detector), `freqInHz`, `seeing`, `rMod`, `Nsubaps`, `Nmodes`,
  `wavelength_nm`, `scenario`, `controller`, `crosstalk`. `terms` / `quantity` / `y` are the error columns (lag, noise, noise_photon,
  noise_ron, alias, xtalk, fitting, chrom_shear / chrom_mult / chrom_fresnel / scintillation when enabled, total; per mode also residual,
  gain, ff, og).
* `filters` = `{column: value or [values]}`; numbers match within 1e-6 relative. **Pin every column that varies in the table except `x`
  (and `hue`, `col`)**: if two configurations would share an `x` value the function raises an error listing the columns that still vary
  (instead of drawing a zig-zag line).

### 4.2 Calibration checks, from a machine

```python
fig = pl.plot_calibration(m, 'og', hue='seeing', values=[0.5, 0.9, 1.3], rMod=3, n_subap=48)    # optical gain per mode
fig = pl.plot_calibration(m, 'thrp', hue='rMod', values=[0, 1, 2, 3], n_subap=48, seeing=0.9)    # throughput versus rMod
fig = pl.plot_calibration(m, 'noise', hue='flux', values=[1e3, 1e4, 1e5, 1e6], fs=2000, rMod=3, n_subap=48, seeing=0.9,
                          wavelength_nm=750, logx=True, ylim=(1e-4, None))                       # noise per mode, one curve per flux
fig = pl.plot_modal_variance_check(m, rMod=3, n_subap=48, seeing=0.9)
```
* `plot_calibration(machine, product, hue, values, rMod, n_subap, seeing, N, N_corr, flux, fs, wavelength_nm)`: `product` is `'og'`
  (optical gain), `'alias'` (aliasing std), `'ct'` (crosstalk), `'noise'` (noise rms at `flux` photons/s and `fs` Hz), `'turb'`
  (turbulence rms), all per mode in command units, or `'thrp'` (throughput versus the `hue` values); one curve per value of `hue` (`rMod`,
  `n_subap`, `seeing`, `N`, `wavelength_nm`, or `flux` [photons/s] / `fs` [Hz], which only change `'noise'`); any other `hue` is an
  error. `logx`, `logy` (default: log for alias / noise / turb), `xlim`, `ylim`. A seeing that was not simulated is estimated (section 2,
  warning).
* `plot_modal_variance_check(machine, rMod, n_subap, seeing)`: variance of the simulated turbulence modes (aliasing run) over the
  analytic one, per mode, with the 1-sigma statistical band: checks the KL basis, r0 and L0.

### 4.3 PSD and PSF, from a budget result

```python
from aoeb.psf import PsfEngine
eng = PsfEngine(m)                                    # or cfg.psf_engine(); kernels cached on disk
r = m.budget(rMod=3, n_subap=48, seeing=0.9, flux=1e7, fs=1500., wavelength_nm=750.)
fig = pl.plot_psd_terms(eng.psd_terms(r), terms=('lag', 'noise_photon', 'alias', 'fitting'))
res = eng.psf_terms(r, 1650.)                        # science wavelength
fig = pl.plot_psf_terms(res, rmax=30)                # contrast profiles, Strehl in the legend
fig = pl.plot_psf_terms(res, kind='halo', rmax=30, pa_range=(0, 90))
fig = pl.plot_psf_image(res, 'total', half_fov=20)
```
* `plot_psd_terms(psd_set, terms, xaxis='lamD' | 'k', show_open_loop=True, xlim, ylim)`: radial PSD [nm² m²] of every term and the
  total, versus separation [λ/D] or spatial frequency [cycles/m]; dotted curve: open-loop turbulence.
* `plot_psf_terms(psf_result, kind='psf' | 'coro' | 'halo', terms, binsize=0.5, rmax, stat='mean' | 'median' | 'std', pa_range=(deg, deg),
  show_dl=True, show_open_loop=False, ymin=1e-6, logx, xlim, ylim)`: radial raw-contrast profiles of every term alone and of the total;
  `'psf'` (default): long-exposure PSF; `'coro'`: **after a perfect coronagraph**, Cavarroc et al. (2006, A&A 447, 397) eq. (1), pupil
  field Π(√E_C − e^{iφ}) with E_C = e^{−σ²}, whose long-exposure intensity for Gaussian phase is PSF − e^{−σ²}·PSF_DL (checked against eq. (1)
  on random phase screens: within 1 %); `'halo'`: first-order halos, which add up; `pa_range` restricts to a sector.
  * **Spikes.** The zig-zag of the PSF profiles is the Airy-ring structure of the pupil (it is in the diffraction-limited curve too),
    sampled by annuli of 0.5 λ/D while the rings are ~1 λ/D apart. It is physical: padding does not change it. `kind='coro'` removes it
    (the coherent part, which carries the Airy pattern, is what a perfect coronagraph removes); `binsize=1.0` only halves the jumps.
  * **The total below a term.** The image of a term *alone* is not its contribution to the total: in the total, each term's halo is
    attenuated by the Strehl of all the other terms (exp(−σ²_total + C) = Π exp(−σ²_j) · …), the missing light being scattered elsewhere by
    them (mostly by the fitting error beyond the control radius). Where one term dominates, the total can therefore lie below that term's
    own profile (example: two terms with S = 0.61 and 0.90; where the second dominates, total / second-alone = 0.85). The first-order halos
    (`kind='halo'`) ignore that attenuation and add up exactly: use them for a decomposition, the PSF or the coronagraphic image for the
    actual contrast.
* `plot_psf_image(psf_result, name='total', half_fov=20, vmin=1e-7, kind='psf' | 'coro' | 'halo')`: 2-D long-exposure image, log colour
  scale, ±half_fov λ/D. `psf_result.image(name, kind)` and `psf_result.contrast_at(rho, kind=...)` give the same images and profiles.

### 4.4 A batch of figures

```python
import os
import matplotlib
matplotlib.use('Agg')                                 # no display (servers, scripts)
import matplotlib.pyplot as plt
os.makedirs('figs', exist_ok=True)
for fs in sorted(totals['freqInHz'].unique()):
    fig = pl.plot_budget_terms(totals, 'photPerMs', {**one, 'freqInHz': fs}, title=f'{fs:g} Hz')
    fig.savefig(f'figs/terms_{fs:g}Hz.png', dpi=150)
    plt.close(fig)
```

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
