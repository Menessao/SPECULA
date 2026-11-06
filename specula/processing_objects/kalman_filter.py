import warnings

from specula import cpuArray
from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from specula.data_objects.kalman_filter_data import KalmanFilterData


class KalmanFilter(BaseProcessingObj):
    """
    Multi-sensor, multi-rate linear Kalman filter.

    The model (state transition, observations, noises, known commands, sensor
    biases, output projection, dense or one small filter per mode) is described
    by a :class:`~specula.data_objects.kalman_filter_data.KalmanFilterData`,
    given either as ``kalman_filter_data`` (YAML: ``kalman_filter_data_object``)
    or through the same arguments as KalmanFilterData.

    The filter is triggered whenever one of its inputs is refreshed. Each time:

    1. Prediction: x <- A x + B u (and P <- A P A' + Q), once per elapsed step.
    2. Update with each sensor refreshed at this time (generation_time equal
       to the current time), one after the other (Joseph-form covariance).
       Sensors not refreshed are skipped, so they can run at different rates.
    3. Outputs: state, covariance, biases, and the projection C x.

    Flux scaling of the noise: for the sensors whose model has photon- and
    read-out-noise variances at a reference flux F_ref, the filter receives the
    PyrSlopec 'out_flux' of that sensor (``in_flux_list``), keeps a running
    average F of it over ``flux_averaging_frames`` frames and, every
    ``flux_update_period`` frames of that sensor, sets

        R_i = R_i(model) + photon (F_ref / F - 1) + read-out ((F_ref / F)^2 - 1)

    (slopes normalised by the flux: photon variance ~ 1/F, read-out ~ 1/F^2;
    the rest of R_i, e.g. white aliasing, is kept). Until the first refresh
    the R_i of the model is used. TODO: full spatial model from the per-pixel
    pupil intensities.

    Gain freezing (``gain_period``): with constant noises, the gains converge
    to a pattern that repeats with the sensor schedule (e.g. every 4 steps for
    a 500 Hz sensor in a 2 kHz loop). Once the gains of two consecutive periods
    agree within ``gain_tolerance``, they are frozen and the covariance is no
    longer propagated: each step then costs a few multiply-adds per state. Any
    change (new measurement noise, different sensor schedule) unfreezes the
    filter, which re-converges and freezes again.

    Parameters
    ----------
    state_transition_matrix, observation_matrix, process_noise_cov,
    measurement_noise_cov, command_matrix, command_observation_matrix,
    initial_state, initial_covariance, bias_process_noise_cov, initial_bias,
    initial_bias_covariance, projection_matrix, photon_noise_var,
    readout_noise_var, reference_flux :
        Model arguments, see KalmanFilterData. Not used with kalman_filter_data.
    kalman_filter_data : KalmanFilterData, optional
        The model in one data object.
    flux_averaging_frames : int
        Time constant, in frames of each sensor, of the running average of its flux.
    flux_update_period : int
        Frames of a sensor between two recomputations of its R_i. Each one
        unfreezes the gains, so keep it long with gain freezing (e.g. 100-500).
    gain_period : int, optional
        Period, in steps, of the sensor schedule (lcm of the sensor frame
        times). Enables gain freezing. Default: None (no freezing).
    gain_tolerance : float
        Relative gain change between two periods below which gains are frozen.
    time_step : float [s], optional
        Duration of the step for which A is defined. If given, the prediction is
        repeated for each step elapsed since the last trigger (the initial state
        refers to t = 0). Default: one prediction per trigger.
        Set it whenever the filter may be triggered less often than every step
        (e.g. a single sensor slower than the loop): without it, A is applied
        once per trigger and the model runs too slowly.
    target_device_idx, precision : as for any processing object.

    TODO: optical gain is not modelled (option: correct the input modes or
    estimate a per-sensor gain); modes are independent, cross-mode noise
    covariance is ignored. Flux scaling is valid for global slope normalisation only.

    Inputs
    ------
    in_measurements_list : list of BaseValue
        Measurement vectors, one per sensor (n_modes * m_i values, mode-major).
    in_command : BaseValue
        Known command u (only if the model has B or D). Held between refreshes.
    in_flux_list : list of BaseValue
        Only if the model has flux scaling: the PyrSlopec 'out_flux' of each
        sensor with flux scaling, in sensor order.

    Outputs
    -------
    out_state : (n,) dense, (n_modes * n,) per mode
    out_covariance : (n, n) dense, (n_modes, n, n) per mode
    out_bias, out_bias_covariance : only if the model has biases
    out_projected_state : C x, only if the model has a projection
    """

    def __init__(self,
                 state_transition_matrix=None,
                 observation_matrix=None,
                 process_noise_cov=None,
                 measurement_noise_cov=None,
                 command_matrix=None,
                 command_observation_matrix=None,
                 initial_state=None,
                 initial_covariance=None,
                 bias_process_noise_cov=None,
                 initial_bias=None,
                 initial_bias_covariance=None,
                 projection_matrix=None,
                 photon_noise_var=None,
                 readout_noise_var=None,
                 reference_flux=None,
                 kalman_filter_data: KalmanFilterData = None,
                 flux_averaging_frames: int = 100,
                 flux_update_period: int = 100,
                 gain_period: int = None,
                 gain_tolerance: float = 1e-4,
                 time_step: float = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)

        matrices = dict(state_transition_matrix=state_transition_matrix,
                        observation_matrix=observation_matrix,
                        process_noise_cov=process_noise_cov,
                        measurement_noise_cov=measurement_noise_cov,
                        command_matrix=command_matrix,
                        command_observation_matrix=command_observation_matrix,
                        initial_state=initial_state,
                        initial_covariance=initial_covariance,
                        bias_process_noise_cov=bias_process_noise_cov,
                        initial_bias=initial_bias,
                        initial_bias_covariance=initial_bias_covariance,
                        projection_matrix=projection_matrix,
                        photon_noise_var=photon_noise_var,
                        readout_noise_var=readout_noise_var,
                        reference_flux=reference_flux)
        if kalman_filter_data is None:
            if state_transition_matrix is None:
                raise ValueError('Give kalman_filter_data or the model matrices')
            model = KalmanFilterData(**matrices, target_device_idx=target_device_idx, precision=precision)
        elif any(v is not None for v in matrices.values()):
            raise ValueError('Give either kalman_filter_data or the matrices, not both')
        else:
            model = kalman_filter_data

        # --- model, on this object's device and precision ----------------
        cast = lambda v: None if v is None else self.to_xp(v, dtype=self.dtype)
        self.A, self.Q, self.B, self.C = cast(model.A), cast(model.Q), cast(model.B), cast(model.C)
        self.H = [cast(h) for h in model.H]
        self.R = [cast(r) for r in model.R]
        self.D = [cast(d) for d in model.D]
        self._R_model = [r.copy() for r in self.R]
        self._noise_scaling = {}                 # sensor -> (photon var, read-out var, reference flux)
        for i, scaling in enumerate(model.noise_scaling):
            if scaling is not None:
                photon, readout, f_ref = scaling
                self._noise_scaling[i] = (cast(photon), cast(readout), f_ref)
        self._flux_sensors = sorted(self._noise_scaling)
        self._x0, self._P0 = cast(model.x0), cast(model.P0)
        self.batched = model.batched
        self.n_modes, self.n_sensors = model.n_modes, model.n_sensors
        self.n_states, self.n_bias = model.n_states, model.n_bias
        self.command_lags = model.command_lags
        self._model = model                               # converts new measurement noises

        self._At = self._T(self.A)
        self._Ht = [self._T(h) for h in self.H]
        self._eye = self.xp.eye(self.A.shape[1], dtype=self.dtype)

        # --- options ------------------------------------------------------
        self._dt = None
        if time_step is not None:
            if time_step <= 0:
                raise ValueError('time_step must be positive')
            self._dt = self.seconds_to_t(time_step)
        if gain_period is not None and gain_period < 1:
            raise ValueError('gain_period must be >= 1')
        self.gain_period = gain_period
        self.gain_tolerance = gain_tolerance
        if flux_averaging_frames < 1 or flux_update_period < 1:
            raise ValueError('flux_averaging_frames and flux_update_period must be >= 1')
        self.flux_averaging_frames = flux_averaging_frames
        self.flux_update_period = flux_update_period

        self.reset_states()

        # --- inputs and outputs -------------------------------------------
        self.inputs['in_measurements_list'] = InputList(type=BaseValue)
        if self.B is not None or self.command_lags > 0:
            self.inputs['in_command'] = InputValue(type=BaseValue)
        if self._flux_sensors:
            self.inputs['in_flux_list'] = InputList(type=BaseValue)

        def output(name, value):
            out = BaseValue(value=value, target_device_idx=target_device_idx, precision=precision)
            self.outputs[name] = out
            return out

        n = self.n_states
        self.state = output('out_state', self._state_out())
        self.covariance = output('out_covariance', self._cov_out(slice(0, n)))
        self.bias = self.bias_covariance = self.projected_state = None
        if self.n_bias > 0:
            self.bias = output('out_bias', self._bias_out())
            self.bias_covariance = output('out_bias_covariance', self._cov_out(slice(n, None)))
        if self.C is not None:
            self.projected_state = output('out_projected_state', self._projection_out())

    @classmethod
    def input_names(cls):
        return {
            'in_measurements_list': InputDesc(
                BaseValue, 'Sensor measurement vectors, one per observation matrix; '
                           'the filter updates whenever any of them is refreshed'),
            'in_command': InputDesc(
                BaseValue, 'Known command, required only if the model has a command '
                           'or command observation matrix (optional)'),
            'in_flux_list': InputDesc(
                BaseValue, 'PyrSlopec out_flux of each sensor with flux scaling, required '
                           'only if the model has flux scaling (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_state': OutputDesc(BaseValue, 'Estimated state vector'),
            'out_covariance': OutputDesc(BaseValue, 'Covariance of the estimated state'),
            'out_bias': OutputDesc(BaseValue, 'Estimated sensor biases, only if the model has biases'),
            'out_bias_covariance': OutputDesc(BaseValue, 'Covariance of the estimated sensor biases, '
                                                         'only if the model has biases'),
            'out_projected_state': OutputDesc(BaseValue, 'Projection matrix times the state estimate, '
                                                         'only if the model has a projection'),
        }

    def check_output_names(self):
        """The bias and projection outputs exist only if the model has them: check those of this model."""
        declared = self.output_names()
        self.output_names = lambda: {k: v for k, v in declared.items() if k in self.outputs}
        try:
            super().check_output_names()
        finally:
            del self.output_names

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset_states(self):
        """Back to the initial state and covariance, empty command history, gains unfrozen."""
        xp = self.xp
        self._x = self._x0.copy()
        self._P = self._P0.copy()
        p = 1 if self.B is None else self.B.shape[2]
        self._u = xp.zeros((self.n_modes, p), dtype=self.dtype)                    # last command
        self._u_history = xp.zeros((self.n_modes, self.command_lags), dtype=self.dtype)
        self._last_t = None
        self._step = 0
        self._flux_mean = {i: None for i in self._flux_sensors}       # running mean of each flux
        self._flux_frames = {i: 0 for i in self._flux_sensors}
        self.R = [r.copy() for r in self._R_model]                    # undo flux scaling
        self._warned_nonfinite = False
        self._unfreeze()

    def set_measurement_noise(self, i, value):
        """Replace R_i (formats of measurement_noise_cov); unfreezes the gains.
        The value becomes the new model value: flux scaling starts from it and
        reset_states keeps it."""
        m = self.H[i].shape[1]
        value = cpuArray(value)
        if self.batched and m > 1 and value.ndim == 1 and value.size == self.n_modes * m:
            value = value.reshape(self.n_modes, m)
        cov = self._model.per_mode_covariance(value, m, f'measurement noise of sensor {i}')
        self.R[i] = self.to_xp(cov, dtype=self.dtype)
        self._R_model[i] = self.R[i].copy()
        self._unfreeze()

    def _update_flux(self, i, flux):
        """Running mean of the flux of sensor i; rescale R_i every flux_update_period frames."""
        flux = float(cpuArray(flux).reshape(-1)[0])
        self._flux_frames[i] += 1
        if self._flux_mean[i] is None:
            self._flux_mean[i] = flux
        else:
            alpha = max(1.0 / self._flux_frames[i], 1.0 / self.flux_averaging_frames)
            self._flux_mean[i] += alpha * (flux - self._flux_mean[i])
        # TODO: with frozen gains, each rescaling restarts from the covariance stored at
        # freeze time, so the bias covariance stops advancing.
        if self._flux_frames[i] % self.flux_update_period == 0 and self._flux_mean[i] > 0:
            photon, readout, f_ref = self._noise_scaling[i]
            ratio = f_ref / self._flux_mean[i]
            extra = photon * (ratio - 1) + readout * (ratio ** 2 - 1)            # (n_modes, m_i)
            m = extra.shape[1]
            self.R[i] = self._R_model[i] + extra[:, :, None] * self.xp.eye(m, dtype=self.dtype)
            self._unfreeze()

    # ------------------------------------------------------------------
    # Kalman steps
    # ------------------------------------------------------------------
    def _predict(self, n_steps):
        """x <- A x + B u, command history shifted, P <- A P A' + Q (unless frozen)."""
        xp = self.xp
        for _ in range(n_steps):
            self._x = self._matvec(self.A, self._x)
            if self.B is not None:
                self._x = self._x + self._matvec(self.B, self._u)
            if self.command_lags > 0:
                self._u_history = xp.concatenate([self._u[:, :1], self._u_history[:, :-1]], axis=1)
            if not self._frozen:
                self._P = self.A @ self._P @ self._At + self.Q

    def _update(self, i, z, K=None):
        """Measurement update with sensor i. Computes the gain K unless given (frozen)."""
        xp = self.xp
        H, R, D = self.H[i], self.R[i], self.D[i]
        m = H.shape[1]
        z = xp.asarray(z, dtype=self.dtype).reshape(-1)
        if z.size != self.n_modes * m:
            raise ValueError(f'Measurement {i} has {z.size} elements, expected {self.n_modes * m}')
        z = z.reshape(self.n_modes, m)
        if D is not None:
            z = z - self._matvec(D, self._u_history)               # remove the known commands

        if K is None:
            PHt = self._P @ self._Ht[i]
            S = H @ PHt + R
            K = self._T(xp.linalg.solve(S, self._T(PHt)))           # K = P H' S^-1
            I_KH = self._eye - K @ H
            P = I_KH @ self._P @ self._T(I_KH) + K @ R @ self._T(K)
            self._P = 0.5 * (P + self._T(P))
        self._x = self._x + self._matvec(K, z - self._matvec(H, self._x))
        return K

    # ------------------------------------------------------------------
    # Gain freezing
    # ------------------------------------------------------------------
    def _unfreeze(self):
        self._frozen = False
        self._frozen_gains = None     # phase -> ({sensor: K}, P after the update, steps predicted)
        self._this_period = {}
        self._last_period = None
        self._period_index = None

    def _can_use_frozen(self, phase, fresh, n_steps):
        return (phase in self._frozen_gains and n_steps == self._frozen_gains[phase][2]
                and set(self._frozen_gains[phase][0]) == set(fresh))

    def _record(self, phase, gains, n_steps):
        """Store this step's gains; at the start of a new period, compare with the previous one."""
        period_index = (self._step - 1) // self.gain_period
        if self._this_period and period_index != self._period_index:
            if self._periods_agree(self._last_period, self._this_period):
                self._frozen, self._frozen_gains = True, self._this_period
            self._last_period, self._this_period = self._this_period, {}
        self._period_index = period_index
        if not self._frozen:
            self._this_period[phase] = (gains, self._P.copy(), n_steps)

    def _periods_agree(self, old, new):
        # TODO: the tolerance uses the global max|K| and the change over one period, so
        # slow or low-gain states (e.g. biases) can freeze early. Option: per-mode,
        # per-state comparison, or a periodic Riccati solve. Recheck with a drifting bias.
        if old is None or set(old) != set(new):
            return False
        xp = self.xp
        for phase, (gains, _, n_steps) in new.items():
            old_gains = old[phase][0]
            if set(gains) != set(old_gains) or n_steps != old[phase][2]:
                return False
            for i, K in gains.items():
                scale = float(xp.max(xp.abs(old_gains[i]))) + 1e-30
                if float(xp.max(xp.abs(K - old_gains[i]))) > self.gain_tolerance * scale:
                    return False
        return True

    # ------------------------------------------------------------------
    # Processing object interface
    # ------------------------------------------------------------------
    def trigger_code(self):
        t = self.current_time
        measurements = self.local_inputs['in_measurements_list']
        if len(measurements) != self.n_sensors:
            raise ValueError(f'{self.n_sensors} sensors configured, '
                             f'but {len(measurements)} measurement inputs connected')

        if self._flux_sensors:
            fluxes = self.local_inputs['in_flux_list']
            if len(fluxes) != len(self._flux_sensors):
                raise ValueError(f'{len(self._flux_sensors)} sensors have flux scaling, '
                                 f'but {len(fluxes)} flux inputs connected')
            for i, flux in zip(self._flux_sensors, fluxes):
                if flux.generation_time == t:
                    self._update_flux(i, flux.value)

        if 'in_command' in self.local_inputs:
            command = self.local_inputs['in_command'].value
            if command is not None:
                command = self.xp.asarray(command, dtype=self.dtype).reshape(-1)
                if command.size != self._u.size:
                    raise ValueError(f'Command has {command.size} elements, expected {self._u.size}')
                self._u = command.reshape(self._u.shape)

        n_steps = self._elapsed_steps(t)
        self._step += n_steps
        self._last_t = t
        phase = None if self.gain_period is None else (self._step - 1) % self.gain_period
        fresh = [i for i, meas in enumerate(measurements)
                 if meas.generation_time == t and self._finite(i, meas.value)]

        if self._frozen and not self._can_use_frozen(phase, fresh, n_steps):
            self._unfreeze()                     # schedule changed: re-converge
            # (P holds the covariance of the last frozen step)

        self._predict(n_steps)
        frozen_gains = self._frozen_gains[phase][0] if self._frozen else {}
        gains = {i: self._update(i, measurements[i].value, frozen_gains.get(i)) for i in fresh}

        if self._frozen:
            self._P = self._frozen_gains[phase][1]
        elif self.gain_period is not None:
            self._record(phase, gains, n_steps)

        n = self.n_states
        self.state.value[:] = self._state_out()
        self.covariance.value[:] = self._cov_out(slice(0, n))
        if self.n_bias > 0:
            self.bias.value[:] = self._bias_out()
            self.bias_covariance.value[:] = self._cov_out(slice(n, None))
        if self.C is not None:
            self.projected_state.value[:] = self._projection_out()

    def post_trigger(self):
        super().post_trigger()
        for out in (self.state, self.covariance, self.bias, self.bias_covariance, self.projected_state):
            if out is not None:
                out.generation_time = self.current_time

    # ------------------------------------------------------------------
    # Helpers (arrays have a leading mode axis, of size 1 for a dense filter)
    # ------------------------------------------------------------------
    def _finite(self, i, value):
        """False for a measurement with NaN/inf (e.g. zero-flux frame): it is skipped."""
        if bool(self.xp.all(self.xp.isfinite(self.xp.asarray(value)))):
            return True
        if not self._warned_nonfinite:
            warnings.warn(f'KalmanFilter: non-finite measurement from sensor {i} skipped '
                          f'(further ones are skipped silently)')
            self._warned_nonfinite = True
        return False

    def _elapsed_steps(self, t):
        if self._dt is None:
            return 1
        t_prev = 0 if self._last_t is None else self._last_t
        return max(1, int(round((t - t_prev) / self._dt)))

    def _T(self, M):
        return self.xp.swapaxes(M, -1, -2)

    def _matvec(self, M, v):
        """Per-mode matrix times per-mode vector: (N, a, b) x (N, b) -> (N, a)."""
        return self.xp.einsum('mij,mj->mi', M, v)

    def _state_out(self):
        return self._x[:, :self.n_states].reshape(-1)

    def _bias_out(self):
        return self._x[:, self.n_states:].reshape(-1)

    def _cov_out(self, s):
        P = self._P[:, s, s]
        return P if self.batched else P[0]

    def _projection_out(self):
        return self._matvec(self.C, self._x[:, :self.n_states]).reshape(-1)
