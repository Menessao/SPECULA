"""
Build the KalmanFilterData of the Ristretto Kalman simulation from analytic
terms, and save it where SPECULA can load it with
``kalman_filter_data_object: '<tag>'``.

Model (per mode j, see KalmanFilterData.from_ar2_closed_loop):

    phi[k]  = a1_j phi[k-1] + a2_j phi[k-2] + w[k],      var(w) = q_j
    sensor i, refreshed at step k, frame of T_i loop steps:
    z_i[k]  = mean_{t < T_i} (phi[k-t] - d[k-t])  (+ b_i if biased)  + v_i[k]
    var(v_i) = noise_var_i,j + aliasing_i,j       (aliasing treated as white)

d is the modal DM command seen by the WFSs at each step (Kalman command input
'kpred.out_comm:-1'), b_i a per-mode offset (sensor without slope nulls,
relative to the reference sensor).

All variances must be in the squared units of the modal reconstructor outputs
(pyr_modes_*.out_modes), at the loop sampling time.

The filter outputs E[phi[k+1] | k]; an IirFilter with the plant inverse
(save_plant_inverse_iir) and its own fractional delay 0.5 then makes the DM,
d[k+1] = 0.5 c[k] + 0.5 c[k-1], follow it.
TODO: AR(2) + plant inverse is Wiener-optimal only if the turbulence is AR(2);
option: design the IIR from the true modal PSD.
"""
import os

import numpy as np

from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.kalman_filter_data import (
    KalmanFilterData, ar2_driving_noise_variance, ar2_stationary_variance)


def save_kalman_data(tag, root_dir, a1, a2, q, sensors, n_modes=None, q_is='driving',
                     aliasing_model='white', aliasing_ar2=None, command_hold_steps=1,
                     plant_inverse_tag=None, beta=0.99, subdir='filter', overwrite=True):
    """
    Build and save a KalmanFilterData FITS file.

    Parameters
    ----------
    tag : str
        Name used in the YAML: ``kalman_filter_data_object: '<tag>'``.
    root_dir : str
        SPECULA calibration root (main.root_dir in the YAML).
    a1, a2, q : array-like, shape (n_modes,)
        AR(2) coefficients and process variance of each mode, at the loop
        sampling time (output of get_turbulence_coefficients(wind_speed, dt)).
    sensors : list of dict, one per sensor, in the order of
        ``kalman.inputs.in_measurements_list``. Keys:
          'noise_var'   (n_modes,) measurement-noise variance per frame of
                        that sensor (get_noise_propagation(...)**2, with the
                        flux collected in ONE frame of that sensor)
          'aliasing'    (n_modes,) aliasing variance (default 0): per frame of
                        that sensor with aliasing_model='white', single-step
                        (loop rate) with aliasing_model='ar2'
          'frame_steps' int, frame time in loop steps (4 for 500 Hz in a 2 kHz
                        loop; default 1)
          'biased'      bool, estimate a per-mode offset for this sensor
                        (default False; the unbiased sensor is the reference)
          'bias'        (n_modes,) optional, initial estimate of the offset (default 0),
                        e.g. the static measurement of this sensor minus that of the
                        reference, from the mean residual frames (utils.get_static_modes)
          'bias_var'    (n_modes,) or scalar, prior variance of the offset
                        (default: the stationary turbulence variance of the mode)
          'bias_rw'     (n_modes,) or scalar, offset random-walk variance per
                        loop step (default 0: constant offset)
          'photon_noise_var' (n_modes,) optional: the photon part of noise_var
                        (e.g. get_noise_propagation(..., sigma_e=0, flux)**2);
                        the rest of noise_var is the read-out part
          'reference_flux' flux at which noise_var was computed, as PyrSlopec
                        'out_flux' reports it: total pupil flux per frame of
                        that sensor [e-]. With photon_noise_var, the filter
                        rescales the noise with the measured flux
    n_modes : int, optional
        Number of controlled modes (= outputs of the modal reconstructors).
        If the turbulence coefficients cover more modes (e.g. the whole KL
        basis), only the first n_modes are kept; they must be the same modes,
        in the same order, as the reconstructor outputs. Default: len(a1).
    aliasing_model : 'white' or 'ar2'
        'white': aliasing variance added to the measurement noise. Acceptable
        only if the aliasing variance is about 0.1 of the noise or less: aliasing
        is strongly correlated in time, and 'white' then costs up to tens of %.
        'ar2': one aliasing AR(2) process per sensor and mode, kept in the
        state, with the dynamics of the first uncorrected mode N+1 (VIS+NIR
        with bias: 11 states per mode instead of 5).
    aliasing_ar2 : (a1, a2), optional
        AR(2) coefficients of the aliasing (scalars or (n_modes,) arrays).
        Default with aliasing_model='ar2': those of mode N+1, i.e.
        a1[n_modes], a2[n_modes] of the full turbulence arrays.
    command_hold_steps : int
        1 when a command is computed every loop step (a sensor at loop rate).
        For a single sensor slower than the loop, its frame in steps (the
        filter and the command then run at the frame rate).
    plant_inverse_tag : str, optional
        If given, also save the plant-inverse IirFilterData
        (save_plant_inverse_iir, with ``beta``) under this tag, in the same folder.
    q_is : 'driving' or 'stationary'
        Whether q is the variance of the AR(2) driving noise w (default) or of
        phi itself.
    subdir : str
        Folder under root_dir where CalibManager looks for KalmanFilterData
        objects ('filter').

    Returns
    -------
    filename : str
        Path of the saved file.
    """
    a1 = np.atleast_1d(np.asarray(a1, dtype=float))
    a2 = np.atleast_1d(np.asarray(a2, dtype=float))
    q = np.atleast_1d(np.asarray(q, dtype=float))
    for name, v in (('a2', a2), ('q', q)):
        if v.size != a1.size:
            raise ValueError(f'{name} has {v.size} values, a1 has {a1.size}')
    if n_modes is None:
        n_modes = a1.size
    if n_modes > a1.size:
        raise ValueError(f'n_modes = {n_modes}, but the turbulence coefficients cover only {a1.size} modes')
    if aliasing_model == 'ar2' and aliasing_ar2 is None:
        if a1.size <= n_modes:
            raise ValueError('aliasing_model="ar2" needs the coefficients of mode N+1: pass turbulence '
                             'arrays with more than n_modes modes, or aliasing_ar2=(a1, a2)')
        aliasing_ar2 = (a1[n_modes], a2[n_modes])
    elif aliasing_model not in ('white', 'ar2'):
        raise ValueError("aliasing_model must be 'white' or 'ar2'")
    a1, a2, q = a1[:n_modes], a2[:n_modes], q[:n_modes]
    if q_is == 'stationary':
        q = ar2_driving_noise_variance(a1, a2, q)
    elif q_is != 'driving':
        raise ValueError("q_is must be 'driving' or 'stationary'")
    if not sensors:
        raise ValueError('At least one sensor is required')
    if all(s.get('biased', False) for s in sensors):
        raise ValueError('At least one sensor must be unbiased (the reference)')

    def per_mode(value, name, default=None):
        if value is None:
            value = default
        v = np.asarray(value, dtype=float)
        if v.ndim > 0 and v.size != n_modes:
            raise ValueError(f'{name} has {v.size} values, but the model has {n_modes} modes '
                             f'(set n_modes to the number of reconstructed modes)')
        v = np.broadcast_to(v.reshape(-1) if v.ndim else v, (n_modes,)).copy()
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError(f'{name} must be finite and non-negative')
        return v

    phi_var = ar2_stationary_variance(a1, a2, q)
    r_list, steps, alias_list = [], [], []
    photon_list, readout_list, flux_list = [], [], []
    bias_q, bias_p0, bias_x0 = [], [], []
    for i, s in enumerate(sensors):
        noise = per_mode(s['noise_var'], f'sensors[{i}]["noise_var"]')
        alias = per_mode(s.get('aliasing'), f'sensors[{i}]["aliasing"]', default=0.0)
        if aliasing_model == 'white':
            r = noise + alias
            alias_list.append(None)
        else:
            r = noise
            alias_list.append(alias if np.any(alias > 0) else None)
        if np.any(r <= 0):
            raise ValueError(f'sensors[{i}]: noise_var + aliasing must be > 0 for every mode')
        r_list.append(r)
        if s.get('photon_noise_var') is not None:
            photon = per_mode(s['photon_noise_var'], f'sensors[{i}]["photon_noise_var"]')
            readout = noise - photon
            if np.any(readout < -1e-12 * np.abs(noise).max()):
                raise ValueError(f'sensors[{i}]: photon_noise_var larger than noise_var')
            if s.get('reference_flux') is None:
                raise ValueError(f'sensors[{i}]: reference_flux is required with photon_noise_var')
            photon_list.append(photon)
            readout_list.append(np.maximum(readout, 0))
            flux_list.append(float(s['reference_flux']))
        else:
            photon_list.append(None)
            readout_list.append(None)
            flux_list.append(None)
        steps.append(int(s.get('frame_steps', 1)))
        if s.get('biased', False):
            # TODO: recheck gain freezing when a chromatic drift (bias_rw > 0) is added:
            # at bias_rw = 1e-8 the frozen bias gain was up to ~30x its long-run value.
            bias_q.append(per_mode(s.get('bias_rw'), f'sensors[{i}]["bias_rw"]', default=0.0))
            bias_p0.append(per_mode(s.get('bias_var'), f'sensors[{i}]["bias_var"]', default=phi_var))
            b0 = s.get('bias')
            if b0 is not None:
                b0 = np.broadcast_to(np.asarray(b0, dtype=float).reshape(-1)[:n_modes]
                                     if np.ndim(b0) else float(b0), (n_modes,)).copy()
                if not np.all(np.isfinite(b0)):
                    raise ValueError(f'sensors[{i}]["bias"] must be finite')
            bias_x0.append(b0)
        else:
            if s.get('bias') is not None:
                raise ValueError(f'sensors[{i}]: "bias" given, but the sensor is not biased')
            bias_q.append(None)
            bias_p0.append(None)
            bias_x0.append(None)

    data = KalmanFilterData.from_ar2_closed_loop(
        a1, a2, q, r_list, integration_steps=steps, delay=1.0,      # output: E[phi[k+1] | k]
        command_hold_steps=command_hold_steps,
        bias_process_noise_cov=bias_q if any(b is not None for b in bias_q) else None,
        initial_bias=bias_x0 if any(b is not None for b in bias_x0) else None,
        initial_bias_covariance=bias_p0 if any(b is not None for b in bias_p0) else None,
        aliasing_var=alias_list if any(v is not None for v in alias_list) else None,
        aliasing_a1=None if aliasing_ar2 is None else aliasing_ar2[0],
        aliasing_a2=None if aliasing_ar2 is None else aliasing_ar2[1],
        photon_noise_var=photon_list if any(v is not None for v in photon_list) else None,
        readout_noise_var=readout_list if any(v is not None for v in readout_list) else None,
        reference_flux=flux_list if any(v is not None for v in flux_list) else None)

    folder = os.path.join(root_dir, subdir)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, tag + '.fits')
    data.save(filename, overwrite=overwrite)
    if plant_inverse_tag is not None:
        save_plant_inverse_iir(plant_inverse_tag, root_dir, a1.size, beta=beta, subdir=subdir,
                               overwrite=overwrite)
    return filename


def save_plant_inverse_iir(tag, root_dir, n_modes, beta=0.99, subdir='filter', overwrite=True):
    """
    Save, as IirFilterData, the inverse of the plant d[k+1] = 0.5 c[k] + 0.5 c[k-1]
    (IirFilter delay 0.5 plus the one-step DM delay), for every mode:

        H(z) = (1 + beta) z / (z + beta),   i.e. c[k] = (1 + beta) u[k] - beta c[k-1]

    DC gain 1. beta = 1 is the exact inverse (d[k+1] = u[k]), but its Nyquist
    component grows without bound; beta = 0.99 keeps it bounded with almost the
    same residual. Fed with E[phi[k+1] | k] from the Kalman filter.
    YAML: ``iir_filter_data_object: '<tag>'`` in an IirFilter with ``delay: 0.5``.
    """
    iir = IirFilterData(ordnum=np.full(n_modes, 2), ordden=np.full(n_modes, 2),
                        num=np.tile([0.0, 1.0 + beta], (n_modes, 1)),
                        den=np.tile([beta, 1.0], (n_modes, 1)))
    folder = os.path.join(root_dir, subdir)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, tag + '.fits')
    if os.path.exists(filename) and not overwrite:
        raise FileExistsError(filename)
    iir.save(filename)                              # IirFilterData.save always overwrites
    return filename


if __name__ == '__main__':
    # Ristretto example: VIS modulated pyramid at 500 Hz (reference) + unmodulated NIR at 2 kHz (biased).
    # Replace the import with the module that defines your three functions.
    from my_ao_budget import get_turbulence_coefficients, get_aliasing_power, get_noise_propagation

    dt = 0.0005              # [s] loop time step (main.time_step)
    wind_speed = 15.0        # [m/s]
    n_subaps, n_modes = 48, 1300
    # total pupil flux per frame of each sensor [e-], as PyrSlopec 'out_flux' reports it;
    # convert if your noise function uses another flux definition (per sub-aperture, ...)
    flux_vis = 1.0e5         # photo-electrons per 2 ms VIS frame
    flux_nir = 2.5e4         # photo-electrons per 0.5 ms NIR frame
    sigma_e = 0.5            # [e-] RON

    a1, a2, q = get_turbulence_coefficients(wind_speed, dt)
    sensors = [
        dict(noise_var=get_noise_propagation(3.0, n_subaps, n_modes, 900e-9, sigma_e, flux_vis) ** 2,
             photon_noise_var=get_noise_propagation(3.0, n_subaps, n_modes, 900e-9, 0.0, flux_vis) ** 2,
             reference_flux=flux_vis,
             aliasing=get_aliasing_power(3.0, n_subaps, n_modes, 900e-9),   # single-step variance
             frame_steps=4, biased=False),
        dict(noise_var=get_noise_propagation(0.0, n_subaps, n_modes, 1300e-9, sigma_e, flux_nir) ** 2,
             photon_noise_var=get_noise_propagation(0.0, n_subaps, n_modes, 1300e-9, 0.0, flux_nir) ** 2,
             reference_flux=flux_nir,
             aliasing=get_aliasing_power(0.0, n_subaps, n_modes, 1300e-9),
             frame_steps=1, biased=True),
    ]
    root_dir = '/raid1/mmenessini/calibration/WFSW'   # main.root_dir (the one aoeb sets, if it overrides it)
    fn = save_kalman_data('kf_ar2_1300modes_vis_nir', root_dir,
                          a1, a2, q, sensors, n_modes=n_modes, aliasing_model='ar2',
                          plant_inverse_tag='plant_inverse_1300modes')
    print('saved', fn)
