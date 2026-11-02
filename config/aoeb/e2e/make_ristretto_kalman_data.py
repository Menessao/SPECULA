"""
Build the KalmanFilterData of ristretto_main_kalman.yml: VIS modulated pyramid
at 500 Hz (reference, unbiased) fused with an unmodulated NIR pyramid at 2 kHz
whose static offset (no slope nulls) is estimated as a bias.

Inputs
------
1. AR(2) model of each mode, identified on open-loop modal turbulence sampled at
   the loop rate (0.5 ms), in the units of the modal reconstructor outputs
   (pyr_modes_*.out_modes). An array of shape (n_modes, 3), columns
   [a1, a2, variance], saved as .npy, .npz (first array) or FITS (primary HDU):

       phi[k] = a1 phi[k-1] + a2 phi[k-2] + w[k]

   ``variance`` is the variance of the driving noise w (``--variance driving``,
   default) or of phi itself (``--variance stationary``).

2. The modal measurements of the noise-only run (calib_kalman_noise.yml), one
   DataStore file per sensor (noise_vis.fits, noise_nir.fits), shape
   (n_steps, n_modes). Frames repeated at loop steps without a new readout (500 Hz
   sensor) are dropped. Their temporal variance is R, per frame of each sensor;
   their means are the static offsets on a flat wavefront: the NIR - VIS
   difference initialises the NIR bias.

Output
------
A KalmanFilterData FITS file, to be put where your CalibManager looks for the
KalmanFilterData objects, under the tag used in the YAML
(``kalman_filter_data_object: 'kf_ar2_1300modes_vis_nir'``).

Example
-------
    python make_ristretto_kalman_data.py ar2_1300modes.npy \\
        <store_dir>/<TN>/noise_vis.fits <store_dir>/<TN>/noise_nir.fits \\
        --steps-vis 4 --steps-nir 1 --reference vis \\
        -o <root_dir>/<kalman data dir>/kf_ar2_1300modes_vis_nir.fits
"""
import argparse
import numpy as np
from astropy.io import fits

from config.aoeb.e2e.kalman_filter_data import (
    KalmanFilterData, ar2_driving_noise_variance, ar2_stationary_variance)#, ar2_is_stationary)


def load_array(path):
    if path.endswith('.npy'):
        return np.load(path)
    if path.endswith('.npz'):
        with np.load(path) as f:
            return f[f.files[0]]
    return np.asarray(fits.getdata(path), dtype=float)


def noise_variance_from_run(modes, skip=10):
    """Per-mode measurement-noise variance from a noise-only, open-loop run.

    modes: (n_steps, n_modes) modal measurements. The first `skip` rows are
    dropped, then rows identical to the previous one (no new frame at that
    step, as for a slower sensor) are removed. Returns (variance, mean, n_frames);
    the mean is the static offset of the measurements on a flat wavefront.
    """
    modes = np.asarray(modes, dtype=float)[skip:]
    new_frame = np.r_[True, np.any(np.diff(modes, axis=0) != 0, axis=1)]
    modes = modes[new_frame]
    return modes.var(axis=0, ddof=1), modes.mean(axis=0), modes.shape[0]


def analytic_noise_variance(recmat, flux_per_subap, ron=0.0, excess_factor2=1.0):
    """Cross-check of R, assuming pyramid slopes normalised by the mean flux
    per sub-aperture (each slope is a +/- sum of the 4 pupil pixels), uniform
    pupil illumination and white noise:

        var(slope) = (F^2 * N + 4 * ron^2) / N^2,   N = photo-electrons per
                                                    sub-aperture per frame
        R_j        = var(slope) * sum_s recmat[j, s]^2

    ron must be the effective (gain-referred) read noise in e-, and
    excess_factor2 = 2 for an EMCCD with excess noise (1 without). Check these
    conventions against PyrSlopec/CCD before trusting this over the run.
    """
    recmat = np.asarray(recmat, dtype=float)
    var_slope = (excess_factor2 * flux_per_subap + 4.0 * ron ** 2) / flux_per_subap ** 2
    return var_slope * np.sum(recmat ** 2, axis=1)


def build(ar2, noise_vis, noise_nir, variance='driving', delay=1.5, skip=10,
          steps_vis=4, steps_nir=1, reference='vis', bias_rw=0.0):
    """KalmanFilterData for the VIS + NIR configuration (sensor order: VIS, NIR).

    steps_*: frame time of each sensor in loop steps. reference: the sensor
    without bias; the other one gets a bias per mode, initialised to the offset
    difference measured in the noise run, with an initial variance equal to
    that difference squared plus its noise variance. bias_rw: variance of the
    bias random walk per step, as a fraction of that sensor's noise variance
    (0: constant bias).
    """
    ar2 = np.asarray(ar2, dtype=float)
    if ar2.ndim != 2 or 3 not in ar2.shape:
        raise ValueError(f'AR(2) array must be (n_modes, 3) or (3, n_modes), got {ar2.shape}')
    if ar2.shape[1] != 3:
        ar2 = ar2.T
    a1, a2, var = ar2.T

    if variance == 'stationary':
        driving = ar2_driving_noise_variance(a1, a2, var)
    elif variance == 'driving':
        driving = var
    else:
        raise ValueError("variance must be 'driving' or 'stationary'")
    if reference not in ('vis', 'nir'):
        raise ValueError("reference must be 'vis' or 'nir'")

    names = ('VIS', 'NIR')
    r_list, offsets = [], []
    for name, run in zip(names, (noise_vis, noise_nir)):
        r, mean, n_frames = noise_variance_from_run(run, skip=skip)
        if r.size != a1.size:
            raise ValueError(f'{name} noise run has {r.size} modes, AR(2) model has {a1.size}')
        print(f'{name}: {n_frames} frames in the noise run')
        r_list.append(r)
        offsets.append(mean)

    ref = 0 if reference == 'vis' else 1
    biased = 1 - ref
    offset_diff = offsets[biased] - offsets[ref]
    bias_q, bias_x0, bias_p0 = [None, None], [None, None], [None, None]
    bias_q[biased] = bias_rw * r_list[biased]
    bias_x0[biased] = offset_diff
    bias_p0[biased] = offset_diff ** 2 + r_list[biased]

    data = KalmanFilterData.from_ar2_closed_loop(
        a1, a2, driving, r_list, integration_steps=[steps_vis, steps_nir], delay=delay,
        bias_process_noise_cov=bias_q, initial_bias=bias_x0, initial_bias_covariance=bias_p0)

    stat = ar2_stationary_variance(a1, a2, driving)
    for i, name in enumerate(names):
        r, mean = r_list[i], offsets[i]
        snr = stat / r
        print(f'{name}{" (reference)" if i == ref else " (biased)"}: R median {np.median(r):.3g}, '
              f'turbulence/noise variance ratio median {np.median(snr):.3g} '
              f'(min {snr.min():.3g}, mode {snr.argmin()}); '
              f'static offset rms {np.sqrt(np.mean(mean ** 2)):.3g}')
    print(f'initial bias ({names[biased]} - {names[ref]} offsets): rms {np.sqrt(np.mean(offset_diff ** 2)):.3g}')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ar2', help='AR(2) array file (n_modes, 3): a1, a2, variance')
    p.add_argument('noise_vis', help='DataStore file of pyr_modes_vis.out_modes from the noise run')
    p.add_argument('noise_nir', help='DataStore file of pyr_modes_nir.out_modes from the noise run')
    p.add_argument('-o', '--output', required=True, help='output KalmanFilterData FITS file')
    p.add_argument('--variance', choices=['driving', 'stationary'], default='driving',
                   help='meaning of the third AR(2) column (default: driving-noise variance)')
    p.add_argument('--delay', type=float, default=1.5,
                   help='total plant delay in steps = 1 + kdelay.delay (default 1.5)')
    p.add_argument('--steps-vis', type=int, default=4, help='VIS frame time in loop steps (default 4: 500 Hz)')
    p.add_argument('--steps-nir', type=int, default=1, help='NIR frame time in loop steps (default 1: 2 kHz)')
    p.add_argument('--reference', choices=['vis', 'nir'], default='vis',
                   help='unbiased reference sensor; the other one gets a bias (default vis)')
    p.add_argument('--bias-rw', type=float, default=0.0,
                   help='bias random-walk variance per step, as a fraction of the biased sensor R '
                        '(default 0: constant bias)')
    p.add_argument('--skip', type=int, default=10, help='initial steps of the noise run to drop')
    args = p.parse_args()

    data = build(load_array(args.ar2), load_array(args.noise_vis), load_array(args.noise_nir),
                 variance=args.variance, delay=args.delay, skip=args.skip,
                 steps_vis=args.steps_vis, steps_nir=args.steps_nir,
                 reference=args.reference, bias_rw=args.bias_rw)
    data.save(args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
