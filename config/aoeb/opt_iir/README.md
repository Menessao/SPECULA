# IIR Wiener predictor for modal AO control

Optimal order-N IIR filter `u = B(z)/A(z) · POL` for one modal loop, compared with a pure
integrator, a leaky integrator and the infinite-order Wiener bound.

## Files

| File | Content |
|---|---|
| `wiener_ao_iir.py` | Filter design: `ModeSpectra` (folded spectra of one mode), `wiener_iir`, `wiener_bound`. Assumptions in its docstring. |
| `ao_tools.py` | Non-plotting helpers: disturbance PSDs (multi-layer von Kármán Zernike, vibrations), closed-loop evaluation, optimal (leaky) integrator, transfer functions, caching, total residual. |
| `iir_study.py` | All comparisons, sweeps and figures, with their configuration. |

Requirements: `numpy`, `scipy`, `matplotlib`.

## Usage

```bash
python iir_study.py compare   # table: INT, LI, IIR vs bound per mode, + optical-gain test
python iir_study.py order     # residual_vs_radial_order.png
python iir_study.py tf        # psd_rtf_ntf.png
python iir_study.py sweep     # residual_total_vs_noise_aliasing.png
python iir_study.py gain      # residual_vs_optical_gain.png (robustness to optical-gain error)
python iir_study.py coeffs    # filter_coefficients.png (optimal LI / IIR2 / IIR3 coefficients)
python iir_study.py           # all of the above
```

Parameters are in three blocks at the top of `iir_study.py`:
- `TURB`: turbulence (multi-layer frozen-flow von Kármán, residuals at `TURB.lam`), sampling
  frequency, `n_max` (used by the aliasing model) and vibrations. Shared by all tasks.
- `ORDER`: per-mode comparison (`compare`, `order`, `tf`): IIR orders, radial orders, noises,
  delays, aliasing level α.
- `SWEEP`: noise and aliasing sweeps of the total residual.
- `GAIN`: radial orders, true/assumed optical-gain ratios, noise and delay of the robustness test.

Aliasing on radial order n has variance α·(n+1)/(n_max+1) times that of one radial-order
n_max+1 mode, with the same temporal shape.

Results of `order` and `sweep` are cached in files named after a hash of the configuration
and of the code, so any change triggers a recomputation and stale results are never reused.
Functions can also be called directly, e.g. `plot_tf(j=2, noise=1e-3, total=2.5, g=0.8)` or
`plot_order(compute_order(g=0.8, g_hat=1.0))`.

Minimal use of the design code:

```python
from wiener_ao_iir import ModeSpectra, wiener_iir
from ao_tools import ZernikePSD, with_vibrations, closed_loop_residual

turb = ZernikePSD(D=8, r0=0.10, L0=25, layers=[(0.6, 7), (0.4, 20)], lam=750e-9)
S = with_vibrations(turb.psd(1), [(20.0, 1e-2, 0.005)])     # tip/tilt + one line
spec = ModeSpectra(S, fs=1000, delay=1.0, noise=1e-3)  # delay = tau; total delay = 1 + tau frames
b, a, info = wiener_iir(spec, order=3)
print(info["residual"], info["bound"], closed_loop_residual(spec, b, a, "pol"))
```

## Vibrations

Set `TURB.vibrations`, keyed by **Noll index** (2 = tip, 3 = tilt, …):

```python
vibrations={2: [(20.0, 1e-2, 0.005)], 3: [(55.0, 3e-3, 0.002)]}
#  j: [(f0 [Hz], variance [rad² @ TURB.lam], damping ζ), ...]
```

Each line is a damped 2nd-order resonance driven by white noise, normalised to the given
variance (FWHM ≈ 2ζf0). A vibrated mode gets the turbulence PSD of its radial order (the
average over that order) plus its lines, and is designed separately. All other modes stay
turbulence only, and vibrations are not aliased. `{}` (default) gives the turbulence-only
results.

Where the vibrated modes show up:
- `compare`: as extra rows `j=…*`.
- `order`: as filled markers next to their radial order.
- `tf`: shows the first vibrated mode.
- `sweep`: each vibrated mode replaces one of the n+1 modes of its order in the total.

Grid requirements (`M` = number of frequency points):
- `fs/M` should not exceed the line width 2ζf0, and `M/fs` must be well above the vibration
  correlation time 1/(2πζf0) (circular causal projection). Check by doubling `M`.
  Per-mode tasks use M = 2^16 for vibrated modes; `SWEEP.M` = 2^14 (16 s at 1 kHz) is
  marginal for ζ ≲ 0.005.
- Lines should lie below fs/2: only the in-band part of the PSD is cell-averaged.
