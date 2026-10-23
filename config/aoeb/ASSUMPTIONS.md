# aoeb — assumptions of the semi-analytical error budget

The budget returns, for every KL mode controlled by a pyramid sensor, the residual variance [nm² of OPD]
split in servo lag, noise and aliasing, plus the fitting error. Everything below is what the model
assumes; each row says where it lives and what happens when it fails.

## 1. Model

Per KL mode, modes decoupled. Frequencies in Hz, z⁻¹ = exp(−2πj f/fs).

| Quantity | Definition |
|---|---|
| Controller | C(z) = g/(1 − ff·z⁻¹): integrator (ff = 1) or leaky integrator, g and ff optimised per mode |
| Delay | latency τ = `latency_s` (any real value ≥ 0, readout + RTC + driver, from the end of the exposure to the command applied) + the frame of exposure window + DM hold (T/2 + T/2): B(z) = (1−m)·z⁻⁽ˡ⁺¹⁾ + m·z⁻⁽ˡ⁺²⁾, τ/T = l + m (linear interpolation, the SPECULA `discrete_delay_tf`) |
| Sensor and DM | continuous exposure averaging over one frame (window anchored at its end, response b = sinc(f/fs)·e^(−jπf/fs)) and a zero-order-hold DM keeping each command for one frame, applied τ after the end of the exposure (hold response H = b·e^(−2πjfτ)) |
| Loop gain | exact for this system: Σ over the spectral images of b·H = B (Poisson summation, checked numerically), so L = ρ·C·B = x·Cn·B with Cn = 1/(1 − ff·z⁻¹): a pure delay, NO sinc envelope, valid for fractional τ. x = g·ρ |
| Turbulence → residual | the science image integrates the continuous e(t) = x(t) − u(t): with K = x·Cn/(1+L), P0 = Σₘ Sₘ, P1 = Σₘ \|bₘ\|² Sₘ, W = Σₘ Sₘ Hₘ bₘ (Σₘ \|Hₘ\|² = 1): T_t = ∫[P0 − 2 Re(K·W) + \|K\|² P1], cross term T_c = ∫[P0 − Re(K·W)] (time average over the spectral images, images \|m\| ≤ 3) |
| Noise, aliasing → residual | measurement disturbances injected after the exposure: DM variance \|K\|²/ρ² × their spectrum (white 2σ²/fs for the noise; the turbulence-like aliasing spectrum of cut-off order n_a filtered by the exposure, P1); ρ in the denominator only |
| Turbulence PSD | one-sided, unit variance, plateau up to f_c = 0.3 (n+1)·V/D (f^−2/3 for n = 1), f^−17/3 above (Conan 1995); variance from the numerical von Kármán KL variances |
| Aliasing PSD | same family with the cut-off of radial order n_a = order of the first uncorrected mode; variance = (std of the simulated aliasing modal signal)² |
| Folding | S_fold(f) = Σₖ S(\|f + k·fs\|), \|k\| ≤ 3, applied to turbulence and aliasing (power above Nyquist returns into the band) |
| Noise PSD | white, flat one-sided 2σ²/fs |
| Optimisation | closed-loop integrals tabulated once per loop on a (ff, x) grid; per mode, the cost is a linear combination of the stacked tables, so one float32 matrix product per radial order scans the whole grid; parabolic refinement and an exact "no control" candidate (details: section 7) |
| Stability | x ≤ 0.98·x_crit(ff); x_crit = smallest −1/Re L₀ at the phase crossovers (gain margin only) |
| Units | everything in command units (the DM modal command); residuals converted to nm² of OPD with the mode rms² (rms = 1 for unit-OPD modes) |
| Totals | lag + noise + aliasing + fitting (+ crosstalk and the chromatic terms if requested) added in quadrature; `noise` is also reported split into `noise_photon` (incl. excess noise) and `noise_ron` |

## 2. Inputs and how they are produced

| Input | Definition | Source |
|---|---|---|
| Turbulence variance Vᵢ | ∫Φ_vK(f)·\|FT(P·mᵢ)\|² df / A², unit-rms-normalised modes, zero-padded FFT (pad 8), r0 at 500 nm, scaled by r0^(−5/3); divided by rms² to get command units | `utils.modal_variance_von_karman`, cached in `data/` |
| Noise σ²ᵢ | pixel variance F²·nₖ + ron_e² (independent pixels) propagated through the reconstructor and PyrSlopec's normalisation; linear in 1/N and 1/N² (coefficients cached) | `noise.py`, average frame of the aliasing run |
| Aliasing A_i | std(R slopes)^2, slopes of runs with N modes perfectly corrected, noise-free, independent random atmospheres (one SPECULA run per seed, pooled) | generator steps 2 and 5 |
| Optical gain ρᵢ | diag(R·IM_PC), R from the diffraction-limited IM, IM_PC measured with N modes perfectly corrected; further correction levels → ρ(residual) by interpolation + fixed point | generator step 6 |
| Detector | `ron_e` = input-referred readout noise [e⁻ rms] = CCD `readout_level` (EMCCD gain divided out); F² = 1 + 1/`excess_delta` with excess noise on (2 at the default), else 1; QE × transmission | `noise.Detector` |
| Photons | `flux` = photons/s reaching the detector (before QE); N_det = flux·QE/fs over the whole detector frame | machine |
| Slopes | PyrSlopec default: quad-cell sx = (A+B−C−D)/⟨F⟩, sy = (B+C−A−D)/⟨F⟩, ⟨F⟩ = (counts in the pupils)/n_norm, n_norm = `PupData.n_subap` = pupil diameter in subapertures; `slopes_from_intensity=True` supported | `noise.py` |

## 3. Assumptions

| # | Assumption | If it fails |
|---|---|---|
| A1 | Modes are decoupled: independent loops, diagonal optical gain, diagonal noise, no cross-mode temporal correlation | frozen flow couples modes of the same azimuthal order across time (prediction potential missed); optical-gain crosstalk `ct` is computed but not used |
| A2 | Turbulence temporal spectrum is the Conan-like piecewise power law per radial order, one effective wind V (5/3 moment of the profile), no wind direction | the low-frequency law differs by mode and with wind direction; layers with very different speeds are lumped |
| A3 | Von Kármán statistics, single L0, r0 scaling exact; the pupil of the variance computation is the pupilstop of the KL basis | the KL basis and pupil must correspond (checked: pixel count) |
| A4 | The DM modal command is in nm of OPD (SPECULA); the KL modes are used as stored, whatever their rms: the budget works in command units and converts with rms² | none, the result does not depend on the mode normalisation |
| A5 | Detector noise = Poisson × F² + Gaussian readout, independent between pixels; no dark, sky, CIC, ADC quantisation, pixel non-uniformity; negative pixels are not clamped | low-flux bias from PyrSlopec's clamping at 0 is not modelled |
| A6 | Noise propagation uses the average frame of the noise-free aliasing run (illumination with N modes perfectly corrected) | the real illumination in partial correction is broader (halo) |
| A7 | Slope noise is the diagonal pixel noise plus the sx–sy photon covariance (exact for quad-cell); PyrSlopec's per-frame normalisation fluctuation is neglected | second order, ~1/√N_det |
| A8 | Optical gain is a scalar per mode, measured per sensor (n_subap), modulation radius and sensing wavelength in perfect correction; the dependence on the actual residual comes from partial-correction IMs (later); the same gain applies to atmosphere and DM signals | the gain also depends on the residual the sensor really sees |
| A9 | Aliasing is an additive measurement disturbance: same transfer function as noise (no extra scaling by ρ), independent of the noise and between modes; its std removes a static offset, i.e. the slope null is assumed calibrated | static aliasing offset would add a static residual |
| A10 | Aliasing temporal PSD is turbulence-like with cut-off order = first uncorrected mode | the simulation provides only the variance (independent random phases); the shape is a model |
| A11 | **Exposure, hold, latency.** One physical model: an exposure-averaging WFS, a zero-order-hold DM and a latency of any real number of frames; the loop gain is a pure interpolated delay of 1 + latency frames (no sinc) and the residual is the continuous-time average (section 7, A11). `latency_s` excludes the frame of window + hold | an instantaneous-snapshot SPECULA run looks 0.5-1 % better at <= 1 kHz |
| A12 | Folding: the disturbance spectra are summed over the spectral images |m| ≤ 3 (power above Nyquist returns into the band); the DM hold ripple and the exposure filtering are part of the model (A11) | high orders at the lowest rates (0.3 % of the variance above Nyquist for n = 48 at 100 Hz) |
| A13 | Controllers are first order (integrator / leaky integrator), stable below 98 % of the gain margin; no phase-margin or sensitivity-peak constraint | optimum may peak strongly near the limit |
| A14 | The DM and RTC are ideal beyond the delay: no DM dynamics, saturation, hysteresis | optimistic at high rates |
| A15 | Fitting error: LBT-ASM law 0.2778·N^−0.9·(D/r0)^(5/3) rad² with N = modes controlled, Kolmogorov, independent of L0 and wind. For the PSD / PSF: von Kármán above k_c = √(N/π)/D (rMax = √(4N/π) actuators across, k_c = rMax/(2D); N = 1200 → 39.1 across, 2.38 c/m, 19.5 λ/D), scaled to the LBT variance by default (`fitting_scale='lbt'`; the pure tail is ≈ 28 % higher at N = 1200); power above the grid Nyquist is outside the PSF field | extrapolated beyond the ASM actuator count (N up to 1200 here) |
| A16 | 'multistage': two sensors in PARALLEL after the DM, both measuring the residual after the full correction. Aliasing, average frame and optical gain come from the run with N_ho modes perfectly corrected; each mode is controlled by its own sensor with the full turbulence as input (no fusion of the sensors yet). **Crosstalk is incomplete in this case**: `crosstalk=True` only includes leaks from the N modes this sensor controls; the leak of the modes between N and N_ho (controlled by the other stage, whose residual comes from the other sensor's budget) and of the modes above N_ho is not added (the latter is part of the aliasing) | modes controlled by both sensors need a joint estimator; the crosstalk of the multistage case is underestimated |
| A17 | Reconstructors: SVD by default (flux independent); ML/MAP are matched to one flux, rate, detector and seeing (diagonal Cn) and used at every flux of a scan | a scan with a fixed ML/MAP reconstructor is the "fixed reconstructor" case |
| A18 | Monochromatic sensing at the wavelength of the products, errors in nm OPD. Chromatic terms: A24; not modelled: NCPA, vibrations, wind shake | to be added as extra terms |
| A20 | Wavelengths: the pupilstop, KL basis, pupil data (pupil geometry on the detector) and the turbulence variance do not depend on the sensing wavelength; the pyramid and the source wavelength are set together; QE is a detector parameter | if the pupil geometry changes with wavelength, run the pupil-data step per wavelength |
| A21 | One aliasing simulation per (wavelength, rMod, n_subap, seeing, N) provides aliasing, slope null, throughput and the illumination for the noise | — |
| A22 | **Crosstalk** (`crosstalk=True`): the leak into mode j is the ROW of R·IM_PC (perfect-correction IM), d_j² = Σᵢ≠ⱼ (R·IM_PC)ⱼᵢ² · residualᵢ, weighted by the residual (not turbulence) variance of the other modes; a white measurement disturbance through Hₙ, iterated with damping (≤ 8 passes); the residuals of different modes are uncorrelated and the leaks into different modes are treated as independent. Reported as its own `xtalk` term | the real residual spectra of the controlled modes are not white (they peak near the bandwidth); see A16 for the multistage case |
| A23 | **Anisoplanatism is neglected.** The halo to beat is the star's own residual (on-axis); anisoplanatism only costs the planet some of its own Strehl: (θ/θ₀)^(5/3) rad², with θ₀ ≈ 1.7″ at 500 nm for the Ristretto profile (≈ 7″ at 1.65 µm, ≈ 2.7″ at 750 nm): ≈ 0.9 % at 10 λ/D (1.65 µm), 1.2 % at 10 λ/D (750 nm), ≈ 0.3 % at 5 λ/D; an upper bound (hand estimate) | planet flux/throughput, not the raw contrast; revisit for separations > 10 λ/D |
| A24 | **Chromatic terms.** Shear (TipTop `differentialRefractionPSD`) whenever a `ChromaticModel` is given; multiplicative (TipTop `chromatismPSD`, loop dependent, gains optimised for it), Fresnel OPD and scintillation are optional, default off (A26); the ADC is assumed ideal (section 7, A24) | see section 7 |
| A25 | Refractivity comes from TipTop's `MatharAirRefraction` (Ciddor < 1.3 µm, Mathar ≥ 1.3 µm), no fallback model; default conditions are Paranal-like (11.5 °C, 743 hPa, 10 %). If TipTop's `fourierModel` calls `get_refractive_index(wvl)` with its defaults (15 °C, 1013 hPa, 20 %), its Δn is 1.35× larger (shear 1.35× in rms for 750 → 1650 nm) | site conditions matter at the 35 % level for the shear |
| A26 | **Default state of the optional terms:** multiplicative, Fresnel OPD and scintillation are OFF, i.e. assumed corrected (multiplicative, Fresnel OPD: the DM or the controller can compensate them) or neglected (scintillation: a phase-only DM cannot correct it); also the ADC residual and the water-vapour fluctuations | the raw contrast at 2–10 λ/D is barely affected by the scintillation (sin²(πλhk²) ≲ 0.1 there); enable them to see the size |
| A27 | Chromatic geometry: heights are divided by the airmass cos z (`heights_are_slant=True` if they already include it, as TipTop's `Hs`); one science and one sensing wavelength per model; layers are the 10 Ristretto layers by default | variance ∝ (1/cos z)² for the shear |
| A28 | **PSD / PSF**: the modal PSD is the PSD of the pupil-windowed residual, Σ σ²ᵢ Kᵢ(k) with Kᵢ the azimuthal mean of |FT(P·mᵢ)|²/A (unit-rms modes); the 2-D PSD is the radial one (isotropic; spiders only through the telescope OTF, shear anisotropy lost); PSF = OTF_tel · exp(−[C(0) − C(r)]) with the pupil-averaged covariance C = IFT(PSD)/OTF_tel for windowed terms (first-order halo exact: (2π/λ)²·PSD/A, checked to ≈ 1 %), pad 4 (4 px per λ/D, field n λ/D); terms do not add beyond first order (the PSF of a term is its own exponential; the first-order halos add) | contrast on the spider axes; far field beyond the PSD grid |
| A29 | **Tag resolution.** Products are named at seeing 0.1 arcsec, rMod 0.1, wavelength 1 nm and L0 1 m; requested values are ROUNDED to it in the generator (which simulates the rounded value, so a product is what its name says) and in the machine. For the seeing the machine rescales the aliasing variance to the requested value (Kolmogorov: variance ~ seeing^(5/3)); optical gain and illumination are taken unscaled; the turbulence variance always uses the exact seeing. A seeing whose rounded value was not simulated is estimated from the simulated ones (warning): optical gains and noise coefficients by a quadratic least-squares fit in seeing (linear with two seeings, constant with one), aliasing (rescaled as above) and crosstalk from the nearest simulated seeing. The KL-variance cache is named with the rounded L0 | a 0.04 arcsec seeing step changes the aliasing variance by ~ 4 % before the rescale; L0 25.4 reuses the L0 = 25 cache (~ 1 % on the tilt variance) |
| A30 | **SPECULA runs (as in LBTSynIM).** Overrides are nested dicts written as `<block>_override` YAML and passed as `specula main.yml extra.yml overrides.yml` (argument list, one unique temporary override file per run, `main.root_dir` set to the products root, `main.pixel_pupil` / `main.pixel_pitch` from the pupilstop size (the electric fields of the atmosphere and of the DM must have the same size), the pyramid and its source always given the same wavelength, the science source propagated like the NGS, the DM objects named from one `dm_tag`); every independent atmospheric realisation is its own run with `atmo_random.seed = seed0 + i + 1` (the same seeds for every configuration: paired comparisons between rMod / seeing / sensors) and the results are averaged (PC IMs: 70 runs of ONE push-pull cycle, `total_time = 2 N time_step`, one screen per run) or pooled (aliasing runs); leading all-zero frames (and `discard_frames`, default 0) are dropped; the pooled frames must pass a lag-1 independence check | if SPECULA ignores `seed`, all realisations are identical: the check warns for the aliasing runs, the PC IMs would silently repeat |
| A31 | **Partial correction** (optional, `generation.pc_corrections`). IMs measured with the N controlled modes corrected by a correction vector (as `save_correction_vector`: max_corr at the tip-tilt, linearly down to min_corr at the highest order, a vector of ones = perfect correction), in the SAME yml as the perfect correction (`calib_pc.yml`, `scale_random`). The rms residual of a level is computed analytically, sqrt(sum (1 - c_i)^2 sigma_i^2) over the controlled modes, and `fixed_point` interpolates the optical gain at the rms of the controlled-mode residual (lag + noise + aliasing + crosstalk); the optical gain is assumed to depend on that level, not on its modal distribution | a different residual shape changes og by an amount not tested here |
| A32 | **IIR controllers.** Per-mode filters H(z) = num(z) / den(z) (SPECULA `IirFilterData`: lowest order first, polynomials in z, gain = last numerator coefficient); only the global gain g (x = g rho) is optimised, per mode, by the same tabulated cost as the integrator (shape fixed, one table row per mode). The loop must be stable for 0 < x <= 0.98 x_crit (x_crit from the phase crossovers; closed-loop poles are checked at three gains), which excludes conditionally stable filters; filters that share a shape share the loop response | a conditionally stable filter is rejected with an error |
| A19 | Numerics: log frequency grid 10⁻³ Hz … fs/2 (1000 points, trapezoid), the first decade completed analytically, 300 gain points per ff, parabolic refinement | ≤ 0.16 % against a 10 000-point direct reference |

## 4. To check on the real products

| Check | How |
|---|---|
| `n_norm` | = pupil diameter in subapertures (48 for 48×48), as in `PupData.n_subap` |
| Slope mode | `slopes_from_intensity` must match the PyrSlopec configuration used for the IM and the aliasing run (default: quad-cell); the reconstructor must have 2·n_valid (quad-cell) or 4·n_valid (intensity) columns |
| Detector | `ron_e` = CCD `readout_level`; `excess_noise` and `excess_delta` as in the CCD |
| Modal variance (optional) | `plotting.plot_modal_variance_check`: simulated variance of the `turbmodes` of an aliasing run over the analytical one, ≈ 1 |
| Wavelengths | the pyramid and source wavelengths set by the generator are the sensing wavelength; QE and flux belong to the budget |
| Latency | `latency_s` = latency from the end of the exposure to the command applied (any real value); the frame of window + hold is in the model (A11). For a SPECULA run: `delay` × time_step; expect it to look 0.5–1 % better than the physical model at ≤ 1 kHz (A11) |
| Chromatic vs TipTop | in TipTop print `fm.n_air_wvlRef - fm.n_air_gs`, `fm.ao.tel.zenith_angle`, `fm.strechFactor`, `fm.ao.atm.heights` (is the airmass applied?) and the temperature / pressure / humidity its `get_refractive_index` call uses (Paranal-like defaults here: 11.5 C, 743 hPa, 10 %), and compare with `ChromaticModel.layer_shifts` (the formula itself is tested against TipTop's code) |
| SPECULA end to end | `generate` for one tiny sensor (n_subaps [12], one rMod, one seeing, ~ 20 modes), then `scan`; check that the override keys exist, that the products appear under `root_dir` with the names of `Paths`, that different seeds give different slopes, and that the first frames are not zero |
| Noise scale | one NOISY run with N modes perfectly corrected at two fluxes: std(R slopes) against sqrt(machine noise); a ratio near n_valid / n_subap (~ 38 for 48 x 48) would mean PyrSlopec normalises per subaperture, not by `PupData.n_subap` |
| Unified PC yml | `calib_pc.yml` uses the block names of your partial-correction yml (`modal_analysis_random`, `scale_random`, `dm_random`, `atmo_random`, `seeing_random`, `ef_mode`, `atmo_pc`); `dm_perfect` / `modal_analysis` are only in the aliasing yml; the correction vector is read from `<root>/data`; it multiplies the reconstructed modes that command the DM and SPECULA applies the command as a correction (negative sign), so 1 = corrected and 0 = uncorrected (one function, `save_correction_vector`, for perfect and partial correction; header NDARRAY = 1) |
| Pupil-data / IM ymls | `calib_pupdata.yml` and `calib_im.yml` are rebuilt from the verified pieces (your IM yml of LBTSynIM, the classes of the SPECULA docs); compare with the ones you ran before: PyrPupdataCalibrator inputs and parameters (only `output_tag` is set), `total_time`, `pushpull.amp` (25 nm), unmodulated vs modulated pupil data |
| IIR | `IirFilterData.set_gain(result.gain)` and the `.num` / `.den` layout (checked here against SPECULA's `max_stable_gain` and `from_gain_and_ff` only) |
| Outer scale | the PC-IM yml (`atmo_random.L0`) must equal `telescope.L0` (the aliasing run is overridden) |
| KL ordering | the radial order of each KL mode (or its dominant spatial frequency) against `utils.radial_order` |
| seeing -> r0 | `utils.r0_from_seeing` uses 0.98 lambda/seeing; SPECULA may use another constant (e.g. 0.9759): compare |
| Reconstructor | `svd_reconstructor` uses no truncation (rcond 0); match the reconstructor SPECULA / the instrument really uses |

## 5. Validation performed (synthetic data, SPECULA not available)

| Check | Result |
|---|---|
| Critical gain vs analytic 2·sin(π/(4N−2)) | agrees |
| Delay vs `IirFilterData.discrete_delay_tf` | 10⁻¹⁶ |
| Boxcar vs brute-force average of shifted exponentials | 10⁻¹⁶ |
| Folded PSD integral over [0, fs/2] | 1.0000 |
| Tabulated optimum vs direct bounded minimisation (orders 1–48, 100–5000 Hz, ρ = 1 and 0.4, with and without exposure, leaky) | ≤ 0.16 % |
| Never worse than no control; no negative gains | all rates |
| Pixel noise Monte Carlo of the CCD algorithm + PyrSlopec normalisation + reconstructor | within 2 %, quad-cell and intensity, EMCCD (F² = 2) and CMOS |
| KL variance: zero padding | pad 4 within 1.5 %, pad 8 within 0.6 % of pad 16; without padding the tilt is 4× too small |
| Exact model (continuous window, held DM, latency 0 / 0.3 / 1.5 frames) vs time-domain simulation of that system, residual and cross term | 0.1–0.5 % (one case −1.3 % at 1.4 σ) |
| Window × hold = interpolated delay for any latency; Σ\|H\|² = 1 | 3·10⁻⁴ with ±4000 images |
| Tabulated optimum vs direct bounded minimisation (fractional latencies, integrator and leaky) | ≤ 0.3 % |
| Multiplicative term | η²V for corrected modes (cross term ~ 1 %), (2η+η²)V for uncorrected modes, scales with 1−compensation, gains lower for η < 0 and optimal for the science cost |
| Crosstalk | xtalk/noise ratio = d²/σ² of the iteration, no leak → identical result, enters like noise in the optimiser |
| PSF chain | Airy zero at 1.2 λ/D, Strehl = exp(−σ²) exactly, energy conserved, first-order halo energy = σ², halo = (2π/λ)²·PSD/A (≈ 1 % modal, ≤ 7 % stationary), halos additive, ∝ λ⁻² |
| Fresnel / scintillation formulas vs FFT Fresnel propagation of random screens (weak turbulence): phase Φcos², log-amplitude Φsin², chromatic phase residual Φ(cos_c − cos_s)² | band integrals within 8–10 %, shape correlation > 0.95 |
| Shear vs TipTop | `differentialRefractionPSD` formula as provided, azimuth averaged = 2[1−J0(·)] (rtol 10⁻⁶); `chromatismPSD` factor ((n2−n1)/n2)² = η²; a run of TipTop itself is NOT done (see section 4) |
| Generator | command and override construction, post-processing and file consistency with the machine; SPECULA runs themselves not executed |

## 6. Not included

Sensor fusion, vibrations, optimisation of the IIR poles and zeros (only their global gain is optimised), NCPA, wind-direction dependence, cross-mode correlations, parallel scans / generation,
the ADC (assumed ideal), the multiplicative, Fresnel and scintillation terms unless enabled (A24, A26), water-vapour fluctuations,
anisoplanatism (A23).

## 7. Details

### Optimisation
closed-loop integrals tabulated once per loop on a (ff, x) grid (T_t and T_c per radial order, T_w, T_a: matrix products); the cost of a mode is a linear combination [V, σ²/ρ², A/ρ², 2ηV] of the stacked tables, so the full (ff, x) scan of all modes of one radial order is ONE matrix product; float32 argmin + parabolic refinement along ln x (float64 evaluation; 600 frequency × 300 gain points: the worst error against the direct bounded minimisation is the same ≈ 2·10⁻³ as with 1000 frequency points; a mode whose optimum sits within a few % of the stability limit can be left ≲ 1 % above the full-grid minimum by the table interpolation), plus an exact "no control" candidate; the gain optimises the science residual when the multiplicative chromatic term is enabled (cost + 2ηV·T_c)

### A11 - exposure, hold and latency
An exposure-averaging WFS + zero-order-hold DM + latency τ; the loop gain is the pure interpolated delay of 1 + τ/T frames (exact, fractional τ allowed, no sinc: the textbook "double sinc" envelope is its baseband approximation and makes the loop look more stable), the frame of window + hold is intrinsic and `latency_s` is only the extra latency (never add the frame); the residual is the continuous-time average of x − u including the intra-frame ripple of the held DM. Validated against a time-domain simulation (continuous window, DM switching at fractional sub-frame instants, τ = 0, 0.3, 1.5 frames): 0.1–0.5 %. A SPECULA run with `dt` = time step uses instantaneous snapshots, which is NOT this system: its fractional `delay` is a blend of consecutive commands (a different noise propagation) and its residual is sampled at the steps; against the physical model such a run looks 0.5–1 % better at ≤ 1 kHz and equal at 2 kHz (illustrative modal budget, not your products); use `latency_s = delay × time_step` and time_step ≪ dt for a like-for-like check

If it fails: the sub-frame DM switching is exact for the loop gain and the residual; the aliasing spectrum is assumed to be filtered by the exposure while its variance A comes from instantaneous snapshots

### A24 - chromatic terms
Always on when a `ChromaticModel` is given: `chrom_shear` (TipTop `differentialRefractionPSD`): footprint displacement s_l = (h_l/cos z)·Δn·tan z between the sensing and science beams, PSD Φ·Σw·2[1−J0(2πk s_l)] for k < k_c (azimuthal mean of an anisotropic PSD); the ADC is assumed ideal and does not remove it. Optional, default OFF (= assumed corrected / neglected): `multiplicative=True, compensation=c` (TipTop `chromatismPSD`): the OPD of air scales with n−1, e_c = e_s + η·x_s, η = (N_c − N_s)/N_s·(1 − c), N = n − 1; per mode Var(e_c) − Var(e_s) = η²V·T0 + 2ηV·T_c; it depends on the LOOP (not on the ADC), T_c being the covariance of the sensed residual with the turbulence, and the gains are OPTIMISED for the science residual when it is enabled (η < 0: the loop corrects less); TipTop's term is the η² part; signed per mode. `fresnel=True`: chromatic OPD of the Fresnel propagation, Φ·Σw·[cos(πλ_c h k²) − cos(πλ_s h k²)]² for k < k_c (not in TipTop). `scintillation=True`: log-amplitude at the science wavelength, Φ·Σw·sin²(πλ_c h k²) in OPD-equivalent units over all k up to `k_max` (PSF grid Nyquist), not corrected by the DM (not in TipTop). Fresnel and scintillation are weak-turbulence (Rytov) formulas, validated against an FFT Fresnel propagation of random screens; the scalar budget treats the amplitude like a phase variance and the PSF applies the same exponential formula as for a phase PSD. Water-vapour fluctuations are not modelled

If it fails: uncompensated, the multiplicative term ≈ η²·(in-band turbulence): η ≈ −0.8 % for 750 → 1650 nm, ≈ 23 nm rms for 1200 modes (hand estimate); Fresnel ≈ 1–2 nm rms; scintillation ≈ 20–25 nm OPD-equivalent, mostly above the DM cut-off
