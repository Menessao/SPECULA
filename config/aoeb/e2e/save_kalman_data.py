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
'kdelay.out_comm:-1'), b_i a per-mode offset (sensor without slope nulls,
relative to the reference sensor).

All variances must be in the squared units of the modal reconstructor outputs
(pyr_modes_*.out_modes), at the loop sampling time.
"""
import os
import numpy as np

from specula.data_objects.kalman_filter_data import (
    KalmanFilterData, ar2_driving_noise_variance, ar2_stationary_variance)


def save_kalman_data(tag, root_dir, a1, a2, q, sensors, delay=1.5, n_modes=None, q_is='driving',
                     subdir='filter', overwrite=True):
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
          'aliasing'    (n_modes,) aliasing variance per frame (default 0)
          'frame_steps' int, frame time in loop steps (4 for 500 Hz in a 2 kHz
                        loop; default 1)
          'biased'      bool, estimate a per-mode offset for this sensor
                        (default False; the unbiased sensor is the reference)
          'bias_var'    (n_modes,) or scalar, prior variance of the offset
                        (default: the stationary turbulence variance of the mode)
          'bias_rw'     (n_modes,) or scalar, offset random-walk variance per
                        loop step (default 0: constant offset)
    delay : float
        Total plant delay in loop steps: 1 + kdelay.delay in the YAML.
    n_modes : int, optional
        Number of controlled modes (= outputs of the modal reconstructors).
        If the turbulence coefficients cover more modes (e.g. the whole KL
        basis), only the first n_modes are kept; they must be the same modes,
        in the same order, as the reconstructor outputs. Default: len(a1).
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
    bias_q, bias_p0 = [], []
    for i, s in enumerate(sensors):
        noise = per_mode(s['noise_var'], f'sensors[{i}]["noise_var"]')
        alias = per_mode(s.get('aliasing'), f'sensors[{i}]["aliasing"]', default=0.0)
        r = noise + alias
        if np.any(r <= 0):
            raise ValueError(f'sensors[{i}]: noise_var + aliasing must be > 0 for every mode')
        r_list.append(r)
        steps.append(int(s.get('frame_steps', 1)))
        if s.get('biased', False):
            bias_q.append(per_mode(s.get('bias_rw'), f'sensors[{i}]["bias_rw"]', default=0.0))
            bias_p0.append(per_mode(s.get('bias_var'), f'sensors[{i}]["bias_var"]', default=phi_var))
        else:
            bias_q.append(None)
            bias_p0.append(None)

    data = KalmanFilterData.from_ar2_closed_loop(
        a1, a2, q, r_list, integration_steps=steps, delay=delay,
        bias_process_noise_cov=bias_q if any(b is not None for b in bias_q) else None,
        initial_bias_covariance=bias_p0 if any(b is not None for b in bias_p0) else None)

    folder = os.path.join(root_dir, subdir)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, tag + '.fits')
    data.save(filename, overwrite=overwrite)
    return filename

