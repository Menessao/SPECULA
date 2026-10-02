from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList


class KalmanFilter(BaseProcessingObj):
    """
    Multi-sensor, multi-rate linear Kalman filter processing object, with
    optional estimation of additive sensor biases.

    State model (per filter step)::

        x[k+1] = A x[k] + B u[k] + w,        w   ~ N(0, Q)
        z_i[k] = H_i x[k] + b_i[k] + v_i,    v_i ~ N(0, R_i)
        b_i[k+1] = b_i[k] + wb_i,            wb_i ~ N(0, Qb_i)

    where ``i`` indexes the sensors and ``b_i`` is the bias of sensor ``i``
    (present only for the sensors for which ``bias_process_noise_cov`` is not
    None; for the other sensors the term is absent). Biases are handled by
    augmenting the state with the bias vectors, so the filter estimates
    ``[x; b]`` jointly and the biases are corrected for in the estimate of x.
    Bias dynamics are a random walk: Qb_i = 0 means a constant bias, a small
    Qb_i lets it drift slowly.

    Measurement noises of different sensors are assumed to be uncorrelated
    (block-diagonal overall R). Under this assumption, processing the sensors
    one after the other is mathematically identical to a single update with the
    stacked measurement vector, and it lets each sensor contribute only when it
    has actually been refreshed.

    The filter is triggered whenever at least one of its inputs is refreshed
    (standard SPECULA readiness rule). At each trigger:

    1. *Prediction*: the state and covariance are propagated with A (and Q, and
       B u if a command matrix is given); biases follow their random walk.
    2. *Update*: for each sensor whose ``generation_time`` equals the current
       time, a Kalman update is applied. Sensors that were not refreshed are
       skipped, so sensors can run at different frequencies.

    Parameters
    ----------
    state_transition_matrix : array-like, shape (n, n)
        State transition matrix A, defined for one filter time step.
    observation_matrix : array or list of arrays
        Observation matrix H. One array of shape (m_i, n) per sensor, given as
        a list/tuple (in the same order as the ``in_measurements_list`` input).
        A single array is accepted for a single-sensor filter.
    process_noise_cov : array-like, shape (n, n)
        Process noise covariance Q.
    measurement_noise_cov : array or list of arrays
        Measurement noise covariance R, one per sensor, matching
        ``observation_matrix``. Each item can be an (m_i, m_i) matrix, an
        (m_i,) vector of variances (diagonal R) or a scalar variance.
    command_matrix : array-like, shape (n, p), optional
        Command-to-state matrix B. If given, the ``in_command`` input becomes
        mandatory and B u is added at every prediction step.
    initial_state : array-like, shape (n,), optional
        Initial state estimate (default: zeros).
    initial_covariance : array-like, shape (n, n), optional
        Initial state covariance (default: identity).
    bias_process_noise_cov : None, or list of per-sensor items, optional
        Enables bias estimation. One item per sensor (a single item, not in a
        list, for a single-sensor filter): None if that sensor has no bias,
        otherwise the covariance Qb_i of the bias random walk, given as an
        (m_i, m_i) matrix, an (m_i,) vector of variances or a scalar variance
        (0 for a constant bias). Each biased sensor has one bias per
        measurement channel (m_i values). Default: no bias estimation.
    initial_bias : None, or list of per-sensor items, optional
        Initial bias estimate (m_i,) for each biased sensor, None for the
        others (default: zeros).
    initial_bias_covariance : None, or list of per-sensor items, optional
        Initial bias covariance for each biased sensor, as for
        ``bias_process_noise_cov`` (default: identity). Choose it to reflect
        how uncertain the biases are at start.
    time_step : float [s], optional
        Duration of the time step for which A is defined. If given, and the
        filter is triggered after several time steps (e.g. no input was
        refreshed in between), the prediction is repeated for each elapsed
        step; the initial state and covariance are taken to refer to t = 0,
        so the first trigger also catches up on the steps elapsed since the
        start. If None (default), exactly one prediction is done per trigger.
    target_device_idx : int, optional
        Target device for computation (-1 for CPU, >=0 for GPU)
    precision : int, optional
        Numerical precision (0 for double, 1 for single)

    Notes
    -----
    Biases must be observable to be estimated: a bias that cannot be told apart
    from the state (e.g. the only sensor, measuring the state directly, with an
    uncertain initial state) just makes the joint covariance grow. Redundant
    sensors, or an accurately known initial state, make them observable.

    Inputs
    ------
    in_measurements_list : list of BaseValue
        Sensor measurement vectors, one per sensor, in the same order as
        ``observation_matrix``.
    in_command : BaseValue
        Command vector u (only if ``command_matrix`` is given). The last
        available value is held between updates.

    Outputs
    -------
    out_state : BaseValue, shape (n,)
        Current estimate of the state (bias-corrected, if biases are estimated).
    out_covariance : BaseValue, shape (n, n)
        Covariance of the state estimate (state block of the joint covariance).
    out_bias : BaseValue, shape (nb,)
        Only if biases are estimated. Bias estimates of the biased sensors,
        concatenated in sensor order (nb = sum of m_i over biased sensors).
    out_bias_covariance : BaseValue, shape (nb, nb)
        Only if biases are estimated. Covariance of the bias estimates.
    """

    def __init__(self,
                 state_transition_matrix,
                 observation_matrix,
                 process_noise_cov,
                 measurement_noise_cov,
                 command_matrix=None,
                 initial_state=None,
                 initial_covariance=None,
                 bias_process_noise_cov=None,
                 initial_bias=None,
                 initial_bias_covariance=None,
                 time_step: float = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        xp = self.xp

        # --- Physical model (before bias augmentation) -------------------
        A = self._as_array(state_transition_matrix, 2, 'state_transition_matrix')
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError(f'state_transition_matrix must be square, got {A.shape}')

        Q = self._as_array(process_noise_cov, 2, 'process_noise_cov')
        if Q.shape != (n, n):
            raise ValueError(f'process_noise_cov must have shape {(n, n)}, got {Q.shape}')

        H_list = self._as_list(observation_matrix)
        R_list = self._as_list(measurement_noise_cov)
        if len(H_list) != len(R_list):
            raise ValueError(f'Got {len(H_list)} observation matrices but '
                             f'{len(R_list)} measurement noise covariances')
        if len(H_list) == 0:
            raise ValueError('At least one sensor is required')

        H = []
        self.R = []
        for i, (h, r) in enumerate(zip(H_list, R_list)):
            h = self._as_array(h, 2, f'observation_matrix[{i}]')
            if h.shape[1] != n:
                raise ValueError(f'observation_matrix[{i}] must have {n} columns, got {h.shape[1]}')
            H.append(h)
            self.R.append(self._make_cov(r, h.shape[0], f'measurement_noise_cov[{i}]'))
        self.n_sensors = len(H)
        sensor_sizes = [h.shape[0] for h in H]

        B = None
        if command_matrix is not None:
            B = self._as_array(command_matrix, 2, 'command_matrix')
            if B.shape[0] != n:
                raise ValueError(f'command_matrix must have {n} rows, got {B.shape[0]}')

        if initial_state is None:
            x0 = xp.zeros(n, dtype=self.dtype)
        else:
            x0 = xp.asarray(initial_state, dtype=self.dtype).ravel()
            if x0.size != n:
                raise ValueError(f'initial_state must have {n} elements, got {x0.size}')
        if initial_covariance is None:
            P0 = xp.eye(n, dtype=self.dtype)
        else:
            P0 = self._as_array(initial_covariance, 2, 'initial_covariance')
            if P0.shape != (n, n):
                raise ValueError(f'initial_covariance must have shape {(n, n)}, got {P0.shape}')

        # --- Sensor biases (state augmentation) --------------------------
        bias_q = self._per_sensor(bias_process_noise_cov, 'bias_process_noise_cov')
        bias_x0 = self._per_sensor(initial_bias, 'initial_bias')
        bias_p0 = self._per_sensor(initial_bias_covariance, 'initial_bias_covariance')

        self.bias_sensors = [i for i, q in enumerate(bias_q) if q is not None]
        self._bias_slices = {}      # sensor index -> slice in the bias vector
        nb = 0
        for i in self.bias_sensors:
            self._bias_slices[i] = slice(nb, nb + sensor_sizes[i])
            nb += sensor_sizes[i]
        for i in range(self.n_sensors):
            if i not in self._bias_slices and (bias_x0[i] is not None or bias_p0[i] is not None):
                raise ValueError(f'Initial bias given for sensor {i}, '
                                 f'but bias_process_noise_cov[{i}] is None (no bias)')

        self.n_states = n
        self.n_bias = nb
        n_aug = n + nb

        self.A = xp.zeros((n_aug, n_aug), dtype=self.dtype)
        self.A[:n, :n] = A
        self.A[n:, n:] = xp.eye(nb, dtype=self.dtype)

        self.Q = xp.zeros((n_aug, n_aug), dtype=self.dtype)
        self.Q[:n, :n] = Q

        self.B = None
        if B is not None:
            self.B = xp.zeros((n_aug, B.shape[1]), dtype=self.dtype)
            self.B[:n] = B

        self._x0 = xp.zeros(n_aug, dtype=self.dtype)
        self._x0[:n] = x0
        self._P0 = xp.zeros((n_aug, n_aug), dtype=self.dtype)
        self._P0[:n, :n] = P0

        self.H = []
        for i, h in enumerate(H):
            h_aug = xp.zeros((sensor_sizes[i], n_aug), dtype=self.dtype)
            h_aug[:, :n] = h
            if i in self._bias_slices:
                s = self._bias_slices[i]
                m = sensor_sizes[i]
                h_aug[:, n + s.start:n + s.stop] = xp.eye(m, dtype=self.dtype)
                self.Q[n + s.start:n + s.stop, n + s.start:n + s.stop] = \
                    self._make_cov(bias_q[i], m, f'bias_process_noise_cov[{i}]')
                if bias_x0[i] is not None:
                    b0 = xp.asarray(bias_x0[i], dtype=self.dtype).ravel()
                    if b0.size != m:
                        raise ValueError(f'initial_bias[{i}] must have {m} elements, got {b0.size}')
                    self._x0[n + s.start:n + s.stop] = b0
                pb0 = xp.eye(m, dtype=self.dtype) if bias_p0[i] is None else \
                    self._make_cov(bias_p0[i], m, f'initial_bias_covariance[{i}]')
                self._P0[n + s.start:n + s.stop, n + s.start:n + s.stop] = pb0
            self.H.append(h_aug)

        self._eye = xp.eye(n_aug, dtype=self.dtype)
        self._x = self._x0.copy()
        self._P = self._P0.copy()

        # --- Timing ------------------------------------------------------
        self._dt = None
        if time_step is not None:
            if time_step <= 0:
                raise ValueError('time_step must be positive')
            self._dt = self.seconds_to_t(time_step)
        self._last_t = None

        # --- Inputs ------------------------------------------------------
        self.inputs['in_measurements_list'] = InputList(type=BaseValue)
        if self.B is not None:
            self.inputs['in_command'] = InputValue(type=BaseValue)

        # --- Outputs (allocated once, refilled at each trigger) ----------
        self.state = BaseValue(value=self._x[:n].copy(),
                               target_device_idx=target_device_idx,
                               precision=precision)
        self.covariance = BaseValue(value=self._P[:n, :n].copy(),
                                    target_device_idx=target_device_idx,
                                    precision=precision)
        self.outputs['out_state'] = self.state
        self.outputs['out_covariance'] = self.covariance

        self.bias = None
        self.bias_covariance = None
        if nb > 0:
            self.bias = BaseValue(value=self._x[n:].copy(),
                                  target_device_idx=target_device_idx,
                                  precision=precision)
            self.bias_covariance = BaseValue(value=self._P[n:, n:].copy(),
                                             target_device_idx=target_device_idx,
                                             precision=precision)
            self.outputs['out_bias'] = self.bias
            self.outputs['out_bias_covariance'] = self.bias_covariance

    # ------------------------------------------------------------------
    # Framework declarations
    # ------------------------------------------------------------------
    @classmethod
    def input_names(cls):
        return {
            'in_measurements_list': InputDesc(
                BaseValue,
                'Sensor measurement vectors, one per observation matrix; '
                'the filter updates whenever any of them is refreshed'),
            'in_command': InputDesc(
                BaseValue,
                'Command vector, required only if command_matrix is given (optional)'),
        }

    @classmethod
    def output_names(cls):
        return {
            'out_state': OutputDesc(BaseValue, 'Estimated state vector'),
            'out_covariance': OutputDesc(BaseValue, 'Covariance of the estimated state'),
            'out_bias': OutputDesc(BaseValue,
                                   'Estimated sensor biases, only if bias_process_noise_cov is given'),
            'out_bias_covariance': OutputDesc(BaseValue,
                                              'Covariance of the estimated sensor biases, '
                                              'only if bias_process_noise_cov is given'),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _as_array(self, value, ndim, name):
        arr = self.xp.asarray(value, dtype=self.dtype)
        if arr.ndim != ndim:
            raise ValueError(f'{name} must be {ndim}-dimensional, got shape {arr.shape}')
        return arr

    @staticmethod
    def _as_list(value):
        """Lists/tuples are one-item-per-sensor; anything else is a single sensor."""
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _per_sensor(self, value, name):
        """One entry per sensor (None allowed); None as a whole means all None."""
        if value is None:
            return [None] * self.n_sensors
        items = self._as_list(value)
        if len(items) != self.n_sensors:
            raise ValueError(f'{name} has {len(items)} items, but there are {self.n_sensors} sensors')
        return items

    def _make_cov(self, c, m, name):
        """Covariance from an (m, m) matrix, an (m,) vector of variances or a scalar."""
        c = self.xp.asarray(c, dtype=self.dtype)
        if c.ndim == 0:
            c = c * self.xp.eye(m, dtype=self.dtype)
        elif c.ndim == 1:
            if c.size != m:
                raise ValueError(f'{name} must have {m} elements, got {c.size}')
            c = self.xp.diag(c)
        if c.shape != (m, m):
            raise ValueError(f'{name} must have shape {(m, m)}, got {c.shape}')
        return c

    # ------------------------------------------------------------------
    # Kalman steps (on the joint state [x; biases])
    # ------------------------------------------------------------------
    def _predict(self, t):
        """Time update: x <- A x + B u, P <- A P A' + Q (repeated for elapsed steps)."""
        n_steps = 1
        if self._dt is not None:
            # the initial state refers to t = 0, so the first trigger also
            # catches up on the steps elapsed since the start of the simulation
            t_prev = 0 if self._last_t is None else self._last_t
            n_steps = max(1, int(round((t - t_prev) / self._dt)))

        bu = None
        if self.B is not None:
            cmd = self.local_inputs['in_command'].value
            if cmd is not None:
                cmd = self.xp.asarray(cmd, dtype=self.dtype).ravel()
                if cmd.size != self.B.shape[1]:
                    raise ValueError(f'Command has {cmd.size} elements, '
                                     f'command_matrix expects {self.B.shape[1]}')
                bu = self.B @ cmd

        for _ in range(n_steps):
            self._x = self.A @ self._x
            if bu is not None:
                self._x = self._x + bu
            self._P = self.A @ self._P @ self.A.T + self.Q

    def _update(self, i, z):
        """Measurement update with sensor i (Joseph-form covariance update)."""
        xp = self.xp
        H, R = self.H[i], self.R[i]

        z = xp.asarray(z, dtype=self.dtype).ravel()
        if z.size != H.shape[0]:
            raise ValueError(f'Measurement {i} has {z.size} elements, '
                             f'observation_matrix[{i}] expects {H.shape[0]}')

        PHt = self._P @ H.T
        S = H @ PHt + R
        K = xp.linalg.solve(S, PHt.T).T            # K = P H' S^-1

        self._x = self._x + K @ (z - H @ self._x)
        i_kh = self._eye - K @ H
        P = i_kh @ self._P @ i_kh.T + K @ R @ K.T
        self._P = 0.5 * (P + P.T)                  # keep P symmetric

    # ------------------------------------------------------------------
    # Processing object interface
    # ------------------------------------------------------------------
    def trigger_code(self):
        t = self.current_time
        measurements = self.local_inputs['in_measurements_list']

        if len(measurements) != self.n_sensors:
            raise ValueError(f'{self.n_sensors} sensors configured, '
                             f'but {len(measurements)} measurement inputs connected')

        self._predict(t)

        for i, meas in enumerate(measurements):
            if meas.generation_time == t:
                self._update(i, meas.value)

        self._last_t = t

        n = self.n_states
        self.state.value[:] = self._x[:n]
        self.covariance.value[:] = self._P[:n, :n]
        if self.n_bias > 0:
            self.bias.value[:] = self._x[n:]
            self.bias_covariance.value[:] = self._P[n:, n:]

    def post_trigger(self):
        super().post_trigger()
        self.state.generation_time = self.current_time
        self.covariance.generation_time = self.current_time
        if self.n_bias > 0:
            self.bias.generation_time = self.current_time
            self.bias_covariance.generation_time = self.current_time

    def reset_states(self):
        """Restore the initial state estimate and covariance."""
        self._x = self._x0.copy()
        self._P = self._P0.copy()
        self._last_t = None
