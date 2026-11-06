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
(pyr_modes_*.out_modes), at the loop sampling time. The noise is fixed: rebuild
the data if the flux changes.

Each command c[k] is held on the DM from t_k + tau T to t_k+1 + tau T (fractional delay tau of the IirFilter;
the frame average seen by the sensor is d[k+1] = (1 - tau) c[k] + tau c[k-1]). The mean of phi over that interval
is 1 + tau steps ahead of the last measured frame, so the filter outputs E[phi[k + 1 + tau] | k] and the IirFilter
is the identity (beta = 0): the delay is compensated by the prediction and its lag stays in the residual.
A plant inverse (beta > 0, horizon 1) would cancel tau only in the frame average, with a command ripple
inside the frame that a one-sample-per-step simulation does not see.
TODO: the prediction is Wiener-optimal only if the turbulence is AR(2).
"""
import os

import numpy as np

from specula.data_objects.iir_filter_data import IirFilterData
from specula.data_objects.kalman_filter_data import (
    KalmanFilterData, ar2_driving_noise_variance, ar2_stationary_variance)


def save_kalman_data(tag, root_dir, a1, a2, q, sensors, n_modes=None, q_is='driving', command_hold_steps=1,
                     plant_inverse_tag=None, beta=0.0, subdir='filter', overwrite=True, delay=1.5):
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
          'aliasing'    (n_modes,) aliasing variance per frame of that sensor
                        (default 0), added to the noise as white noise
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
    n_modes : int, optional
        Number of controlled modes (= outputs of the modal reconstructors).
        If the turbulence coefficients cover more modes (e.g. the whole KL
        basis), only the first n_modes are kept; they must be the same modes,
        in the same order, as the reconstructor outputs. Default: len(a1).
    command_hold_steps : int
        1 when a command is computed every loop step (a sensor at loop rate).
        For a single sensor slower than the loop, its frame in steps (the
        filter and the command then run at the frame rate).
    plant_inverse_tag : str, optional
        If given, also save the IirFilterData of kpred (save_plant_inverse_iir, with
        ``beta``; default beta = 0: identity) under this tag, in the same folder.
    q_is : 'driving' or 'stationary'
        Whether q is the variance of the AR(2) driving noise w (default) or of
        phi itself.
    subdir : str
        Folder under root_dir where CalibManager looks for KalmanFilterData
        objects ('filter').
    delay : float
        Prediction horizon of the filter output, in loop steps: 1 + tau, with tau the
        fractional delay of the IirFilter (default 1.5: tau = 0.5).

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
    r_list, steps = [], []
    bias_q, bias_p0, bias_x0 = [], [], []
    for i, s in enumerate(sensors):
        r = per_mode(s['noise_var'], f'sensors[{i}]["noise_var"]') + per_mode(s.get('aliasing'), f'sensors[{i}]["aliasing"]', default=0.0)
        if np.any(r <= 0):
            raise ValueError(f'sensors[{i}]: noise_var + aliasing must be > 0 for every mode')
        r_list.append(r)
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
        a1, a2, q, r_list, integration_steps=steps, delay=delay,    # output: E[phi[k+delay] | k]
        command_hold_steps=command_hold_steps,
        bias_process_noise_cov=bias_q if any(b is not None for b in bias_q) else None,
        initial_bias=bias_x0 if any(b is not None for b in bias_x0) else None,
        initial_bias_covariance=bias_p0 if any(b is not None for b in bias_p0) else None)

    folder = os.path.join(root_dir, subdir)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, tag + '.fits')
    data.save(filename, overwrite=overwrite)
    if plant_inverse_tag is not None:
        save_plant_inverse_iir(plant_inverse_tag, root_dir, a1.size, beta=beta, subdir=subdir,
                               overwrite=overwrite)
    return filename


def save_plant_inverse_iir(tag, root_dir, n_modes, beta=0.0, subdir='filter', overwrite=True):
    """
    Save, as IirFilterData, for every mode:

        H(z) = (1 + beta) z / (z + beta),   i.e. c[k] = (1 + beta) u[k] - beta c[k-1]

    beta = 0 (default): identity, c[k] = u[k], the Ristretto chain.
    beta > 0: inverse of the plant d[k+1] = (1 - tau) c[k] + tau c[k-1] for beta = tau / (1 - tau)
    (beta = 0.99 for tau = 0.5, where the exact inverse is marginally stable), to be fed with E[phi[k+1] | k].
    It only cancels tau in the frame average: its command ripple inside the frame is not simulated.
    YAML: ``iir_filter_data_object: '<tag>'`` in an IirFilter with ``delay: tau``.
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
