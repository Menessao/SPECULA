from specula.base_processing_obj import BaseProcessingObj, InputDesc, OutputDesc
from specula.base_value import BaseValue
from specula.connections import InputValue, InputList
from config.aoeb.e2e.kalman_filter_data import KalmanFilterData


class KalmanFilter(BaseProcessingObj):
    """
    Multi-sensor, multi-rate linear Kalman filter processing object, with
    optional estimation of additive sensor biases and an optional output
    projection (e.g. to turn the state estimate into a command).

    State model (per filter step)::

        x[k+1] = A x[k] + B u[k] + w,        w   ~ N(0, Q)
        z_i[k] = H_i x[k] + b_i[k] + v_i,    v_i ~ N(0, R_i)
        b_i[k+1] = b_i[k] + wb_i,            wb_i ~ N(0, Qb_i)

    where ``i`` indexes the sensors and ``b_i`` is the bias of sensor ``i``
    (only for the sensors with a ``bias_process_noise_cov``). Biases are handled
    by augmenting the state with the bias vectors, so the filter estimates
    ``[x; b]`` jointly and the biases are corrected for in the estimate of x.
    Bias dynamics are a random walk: Qb_i = 0 means a constant bias.

    Dense and per-mode ("batched") operation
    ----------------------------------------
    If ``state_transition_matrix`` is 2-D (n, n), the filter is *dense*: one
    full n-dimensional state, with O(n^3) cost per step.

    If it is 3-D (n_modes, n, n), the filter is *batched*: n_modes independent
    filters of state size n (e.g. one per KL mode, with n = 2 for an AR(2)
    turbulence model), all propagated at once. This is equivalent to a dense
    filter with block-diagonal matrices, at O(n_modes n^3) cost, which keeps
    a modal filter with thousands of modes fast. Shapes in batched mode:

    - A: (n_modes, n, n); Q: (n_modes, n, n), or (n_modes, n) for diagonal Q
    - H_i: (n_modes, m_i, n), or (m_i, n) shared by all modes
    - R_i (and bias covariances): scalar, (n_modes,) one variance per mode,
      (n_modes, m_i) diagonal, or (n_modes, m_i, m_i)
    - B: (n_modes, n, p) or (n, p) shared; command input of size n_modes * p
    - initial_state: (n_modes, n) or (n_modes * n,); initial_covariance:
      (n_modes, n, n) or (n, n) shared; initial_bias: (n_modes, m_i)
    - projection_matrix: (n_modes, q, n), or (n_modes, n) for q = 1
    - measurements: vectors of size n_modes * m_i, mode-major (for m_i = 1,
      simply one value per mode, e.g. the output of a modal reconstructor)

    Vector outputs are flattened mode-major (size n_modes * n for the state);
    covariance outputs keep the per-mode blocks (n_modes, n, n).

    Measurement noises of different sensors are assumed to be uncorrelated, so
    the sensors refreshed at a given time are processed one after the other,
    which is identical to a single stacked update and lets each sensor
    contribute only when it has actually been refreshed.

    The filter is triggered whenever at least one of its inputs is refreshed
    (standard SPECULA readiness rule). At each trigger:

    1. *Prediction*: x and P are propagated with A (and Q, and B u if a command
       matrix is given); biases follow their random walk.
    2. *Update*: for each sensor whose ``generation_time`` equals the current
       time, a Kalman update is applied (Joseph-form covariance update).
    3. *Projection* (optional): ``out_projected_state = C x`` on the updated
       estimate.

    Parameters
    ----------
    state_transition_matrix : array, (n, n) or (n_modes, n, n)
        State transition matrix A, defined for one filter time step.
    observation_matrix : array or list of arrays
        Observation matrix H_i of each sensor, as a list/tuple in the same order
        as the ``in_measurements_list`` input (a single array for one sensor).
    process_noise_cov : array
        Process noise covariance Q.
    measurement_noise_cov : array or list of arrays
        Measurement noise covariance R_i of each sensor (matrix, vector of
        variances or scalar; see shapes above).
    command_matrix : array, optional
        Command-to-state matrix B. If given, the ``in_command`` input becomes
        mandatory and B u is added at every prediction step.
    initial_state : array, optional
        Initial state estimate (default: zeros).
    initial_covariance : array, optional
        Initial state covariance (default: identity).
    bias_process_noise_cov : None, or list of per-sensor items, optional
        Enables bias estimation: one item per sensor (a single item for one
        sensor), None for an unbiased sensor, else the covariance Qb_i of its
        bias random walk (0 for a constant bias). Each biased sensor has one
        bias per measurement channel.
    initial_bias : None, or list of per-sensor items, optional
        Initial bias estimate for each biased sensor (default: zeros).
    initial_bias_covariance : None, or list of per-sensor items, optional
        Initial bias covariance for each biased sensor (default: identity).
    projection_matrix : array, optional
        Matrix C applied to the updated state estimate (physical states only,
        not the biases) to produce ``out_projected_state``, e.g. the
        prediction of the turbulence over the loop delay, used as command.
    kalman_filter_data : KalmanFilterData, optional
        All the matrices above in one data object (YAML:
        ``kalman_filter_data_object: '<tag>'``). Mutually exclusive with the
        explicit matrix arguments.
    time_step : float [s], optional
        Duration of the time step for which A is defined. If given, and the
        filter is triggered after several time steps, the prediction is
        repeated for each elapsed step; the initial state refers to t = 0. If
        None (default), exactly one prediction is done per trigger.
    target_device_idx : int, optional
        Target device for computation (-1 for CPU, >=0 for GPU)
    precision : int, optional
        Numerical precision (0 for double, 1 for single)

    Notes
    -----
    - Biases must be observable to be estimated (redundant sensors, or an
      accurately known initial state); otherwise their covariance just grows.
    - H_i maps the state at the trigger time. A sensor integrating over several
      filter steps (slower sensor with longer exposure) must have this built
      into its H_i (e.g. averaging delayed states kept in x).

    Inputs
    ------
    in_measurements_list : list of BaseValue
        Sensor measurement vectors, one per sensor.
    in_command : BaseValue
        Command vector u (only if ``command_matrix`` is given). The last
        available value is held between updates.

    Outputs
    -------
    out_state : BaseValue
        State estimate (bias-corrected): (n,) dense, (n_modes * n,) batched.
    out_covariance : BaseValue
        State covariance: (n, n) dense, (n_modes, n, n) batched.
    out_bias, out_bias_covariance : BaseValue
        Only if biases are estimated: bias estimates of the biased sensors,
        concatenated in sensor order, and their covariance.
    out_projected_state : BaseValue
        Only if ``projection_matrix`` is given: C x, (q,) dense,
        (n_modes * q,) batched.
    """

    def __init__(self,
                 state_transition_matrix=None,
                 observation_matrix=None,
                 process_noise_cov=None,
                 measurement_noise_cov=None,
                 command_matrix=None,
                 initial_state=None,
                 initial_covariance=None,
                 bias_process_noise_cov=None,
                 initial_bias=None,
                 initial_bias_covariance=None,
                 projection_matrix=None,
                 kalman_filter_data: KalmanFilterData = None,
                 time_step: float = None,
                 target_device_idx: int = None,
                 precision: int = None):

        super().__init__(target_device_idx=target_device_idx, precision=precision)
        xp = self.xp

        if kalman_filter_data is not None:
            explicit = (state_transition_matrix, observation_matrix, process_noise_cov,
                        measurement_noise_cov, command_matrix, initial_state, initial_covariance,
                        bias_process_noise_cov, initial_bias, initial_bias_covariance,
                        projection_matrix)
            if any(v is not None for v in explicit):
                raise ValueError('Give either kalman_filter_data or the matrices, not both')
            kw = kalman_filter_data.kalman_kwargs()
            state_transition_matrix = kw['state_transition_matrix']
            observation_matrix = kw['observation_matrix']
            process_noise_cov = kw['process_noise_cov']
            measurement_noise_cov = kw['measurement_noise_cov']
            command_matrix = kw['command_matrix']
            initial_state = kw['initial_state']
            initial_covariance = kw['initial_covariance']
            bias_process_noise_cov = kw['bias_process_noise_cov']
            initial_bias = kw['initial_bias']
            initial_bias_covariance = kw['initial_bias_covariance']
            projection_matrix = kw['projection_matrix']

        for name, value in (('state_transition_matrix', state_transition_matrix),
                            ('observation_matrix', observation_matrix),
                            ('process_noise_cov', process_noise_cov),
                            ('measurement_noise_cov', measurement_noise_cov)):
            if value is None:
                raise ValueError(f'{name} is required (or give kalman_filter_data)')

        # --- State transition: decides dense vs batched -------------------
        A = xp.asarray(state_transition_matrix, dtype=self.dtype)
        if A.ndim == 2:
            self.batched = False
            A = A[None]
        elif A.ndim == 3:
            self.batched = True
        else:
            raise ValueError(f'state_transition_matrix must be 2-D (dense) or 3-D (batched), '
                             f'got shape {A.shape}')
        N, n, n2 = A.shape
        if n != n2:
            raise ValueError(f'state_transition_matrix must be square, got {A.shape}')
        self.n_modes = N

        # --- Process noise ----------------------------------------------
        Q = xp.asarray(process_noise_cov, dtype=self.dtype)
        if self.batched and Q.ndim == 2 and Q.shape == (N, n):
            Q = Q[:, :, None] * xp.eye(n, dtype=self.dtype)
        else:
            Q = self._matrix(Q, n, n, 'process_noise_cov', shared_ok=False)

        # --- Sensors ------------------------------------------------------
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
            h = self._matrix(h, None, n, f'observation_matrix[{i}]')
            H.append(h)
            self.R.append(self._cov(r, h.shape[1], f'measurement_noise_cov[{i}]'))
        self.n_sensors = len(H)
        sensor_sizes = [h.shape[1] for h in H]

        B = None
        if command_matrix is not None:
            B = self._matrix(command_matrix, n, None, 'command_matrix')

        x0 = xp.zeros((N, n), dtype=self.dtype) if initial_state is None else \
            self._vector(initial_state, n, 'initial_state')
        P0 = xp.broadcast_to(xp.eye(n, dtype=self.dtype), (N, n, n)) if initial_covariance is None else \
            self._matrix(initial_covariance, n, n, 'initial_covariance')

        self.C = None
        if projection_matrix is not None:
            C = xp.asarray(projection_matrix, dtype=self.dtype)
            if self.batched and C.ndim == 2:
                C = C[:, None, :]                    # (n_modes, n): one output per mode
            self.C = self._matrix(C, None, n, 'projection_matrix', shared_ok=False)

        # --- Sensor biases (state augmentation) --------------------------
        bias_q = self._per_sensor(bias_process_noise_cov, 'bias_process_noise_cov')
        bias_x0 = self._per_sensor(initial_bias, 'initial_bias')
        bias_p0 = self._per_sensor(initial_bias_covariance, 'initial_bias_covariance')

        self.bias_sensors = [i for i, q in enumerate(bias_q) if q is not None]
        self._bias_slices = {}      # sensor index -> slice in the (per-mode) bias vector
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
        na = n + nb

        self.A = xp.zeros((N, na, na), dtype=self.dtype)
        self.A[:, :n, :n] = A
        self.A[:, n:, n:] = xp.eye(nb, dtype=self.dtype)

        self.Q = xp.zeros((N, na, na), dtype=self.dtype)
        self.Q[:, :n, :n] = Q

        self.B = None
        if B is not None:
            self.B = xp.zeros((N, na, B.shape[2]), dtype=self.dtype)
            self.B[:, :n] = B

        self._x0 = xp.zeros((N, na), dtype=self.dtype)
        self._x0[:, :n] = x0
        self._P0 = xp.zeros((N, na, na), dtype=self.dtype)
        self._P0[:, :n, :n] = P0

        self.H = []
        for i, h in enumerate(H):
            m = sensor_sizes[i]
            h_aug = xp.zeros((N, m, na), dtype=self.dtype)
            h_aug[:, :, :n] = h
            if i in self._bias_slices:
                s = slice(n + self._bias_slices[i].start, n + self._bias_slices[i].stop)
                h_aug[:, :, s] = xp.eye(m, dtype=self.dtype)
                self.Q[:, s, s] = self._cov(bias_q[i], m, f'bias_process_noise_cov[{i}]')
                if bias_x0[i] is not None:
                    self._x0[:, s] = self._vector(bias_x0[i], m, f'initial_bias[{i}]')
                self._P0[:, s, s] = xp.eye(m, dtype=self.dtype) if bias_p0[i] is None else \
                    self._cov(bias_p0[i], m, f'initial_bias_covariance[{i}]')
            self.H.append(h_aug)

        self._At = xp.swapaxes(self.A, -1, -2).copy()
        self._Ht = [xp.swapaxes(h, -1, -2).copy() for h in self.H]
        self._eye = xp.eye(na, dtype=self.dtype)
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
        def new_value(v):
            return BaseValue(value=v.copy(), target_device_idx=target_device_idx, precision=precision)

        self.state = new_value(self._state_out())
        self.covariance = new_value(self._cov_out(slice(0, n)))
        self.outputs['out_state'] = self.state
        self.outputs['out_covariance'] = self.covariance

        self.bias = None
        self.bias_covariance = None
        if nb > 0:
            self.bias = new_value(self._bias_out())
            self.bias_covariance = new_value(self._cov_out(slice(n, na)))
            self.outputs['out_bias'] = self.bias
            self.outputs['out_bias_covariance'] = self.bias_covariance

        self.projected_state = None
        if self.C is not None:
            self.projected_state = new_value(self._projection_out())
            self.outputs['out_projected_state'] = self.projected_state

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
            'out_projected_state': OutputDesc(BaseValue,
                                              'projection_matrix times the state estimate, '
                                              'only if projection_matrix is given'),
        }

    # ------------------------------------------------------------------
    # Helpers (all internal arrays have a leading mode axis, of size 1 if dense)
    # ------------------------------------------------------------------
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

    def _matrix(self, value, rows, cols, name, shared_ok=True):
        """Per-mode matrix -> (n_modes, rows, cols). rows/cols None means any size."""
        xp, N = self.xp, self.n_modes
        m = xp.asarray(value, dtype=self.dtype)
        if self.batched:
            if m.ndim == 2 and shared_ok:
                m = xp.broadcast_to(m, (N,) + m.shape).copy()
            elif m.ndim != 3 or m.shape[0] != N:
                raise ValueError(f'{name} must have shape ({N}, rows, cols)'
                                 f'{" or (rows, cols)" if shared_ok else ""}, got {m.shape}')
        else:
            if m.ndim != 2:
                raise ValueError(f'{name} must be 2-dimensional, got shape {m.shape}')
            m = m[None]
        if (rows is not None and m.shape[1] != rows) or (cols is not None and m.shape[2] != cols):
            raise ValueError(f'{name} must have shape '
                             f'{(rows if rows is not None else "any", cols if cols is not None else "any")}'
                             f' per mode, got {m.shape[1:]}')
        return m

    def _vector(self, value, size, name):
        """Per-mode vector -> (n_modes, size)."""
        N = self.n_modes
        v = self.xp.asarray(value, dtype=self.dtype)
        if v.size != N * size:
            raise ValueError(f'{name} must have {N * size} elements'
                             f'{f" ({N} modes x {size})" if self.batched else ""}, got {v.size}')
        return v.reshape(N, size)

    def _cov(self, c, m, name):
        """Covariance -> (n_modes, m, m), from a scalar, variances or full matrices."""
        xp, N = self.xp, self.n_modes
        c = xp.asarray(c, dtype=self.dtype)
        eye = xp.eye(m, dtype=self.dtype)
        if c.ndim == 0:
            return xp.broadcast_to(c * eye, (N, m, m)).copy()
        if not self.batched:
            if c.ndim == 1:
                if c.size != m:
                    raise ValueError(f'{name} must have {m} elements, got {c.size}')
                c = xp.diag(c)
            if c.shape != (m, m):
                raise ValueError(f'{name} must have shape {(m, m)}, got {c.shape}')
            return c[None]
        if c.ndim == 1:
            if c.size != N:
                raise ValueError(f'{name} must have one variance per mode ({N}), got {c.size}')
            return c[:, None, None] * eye
        if c.ndim == 2:
            if c.shape != (N, m):
                raise ValueError(f'{name} must have shape {(N, m)}, got {c.shape}')
            return c[:, :, None] * eye
        if c.shape != (N, m, m):
            raise ValueError(f'{name} must have shape {(N, m, m)}, got {c.shape}')
        return c

    def _state_out(self):
        return self._x[:, :self.n_states].reshape(-1)

    def _bias_out(self):
        return self._x[:, self.n_states:].reshape(-1)

    def _cov_out(self, s):
        P = self._P[:, s, s]
        return P if self.batched else P[0]

    def _projection_out(self):
        x = self._x[:, :self.n_states]
        return self.xp.einsum('mqi,mi->mq', self.C, x).reshape(-1)

    # ------------------------------------------------------------------
    # Kalman steps (on the joint state [x; biases], all modes at once)
    # ------------------------------------------------------------------
    def _predict(self, t):
        """Time update: x <- A x + B u, P <- A P A' + Q (repeated for elapsed steps)."""
        xp = self.xp
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
                p = self.B.shape[2]
                cmd = xp.asarray(cmd, dtype=self.dtype).reshape(-1)
                if cmd.size != self.n_modes * p:
                    raise ValueError(f'Command has {cmd.size} elements, '
                                     f'command_matrix expects {self.n_modes * p}')
                bu = xp.einsum('mip,mp->mi', self.B, cmd.reshape(self.n_modes, p))

        for _ in range(n_steps):
            self._x = xp.einsum('mij,mj->mi', self.A, self._x)
            if bu is not None:
                self._x = self._x + bu
            self._P = self.A @ self._P @ self._At + self.Q

    def _update(self, i, z):
        """Measurement update with sensor i (Joseph-form covariance update)."""
        xp = self.xp
        H, Ht, R = self.H[i], self._Ht[i], self.R[i]
        m = H.shape[1]

        z = xp.asarray(z, dtype=self.dtype).reshape(-1)
        if z.size != self.n_modes * m:
            raise ValueError(f'Measurement {i} has {z.size} elements, '
                             f'observation_matrix[{i}] expects {self.n_modes * m}')
        z = z.reshape(self.n_modes, m)

        PHt = self._P @ Ht                                          # (N, na, m)
        S = H @ PHt + R                                             # (N, m, m)
        K = xp.swapaxes(xp.linalg.solve(S, xp.swapaxes(PHt, -1, -2)), -1, -2)   # P H' S^-1

        innovation = z - xp.einsum('mij,mj->mi', H, self._x)
        self._x = self._x + xp.einsum('mij,mj->mi', K, innovation)
        i_kh = self._eye - K @ H
        P = i_kh @ self._P @ xp.swapaxes(i_kh, -1, -2) + K @ R @ xp.swapaxes(K, -1, -2)
        self._P = 0.5 * (P + xp.swapaxes(P, -1, -2))              # keep P symmetric

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
        self.state.value[:] = self._state_out()
        self.covariance.value[:] = self._cov_out(slice(0, n))
        if self.n_bias > 0:
            self.bias.value[:] = self._bias_out()
            self.bias_covariance.value[:] = self._cov_out(slice(n, n + self.n_bias))
        if self.C is not None:
            self.projected_state.value[:] = self._projection_out()

    def post_trigger(self):
        super().post_trigger()
        for out in (self.state, self.covariance, self.bias, self.bias_covariance,
                    self.projected_state):
            if out is not None:
                out.generation_time = self.current_time

    def reset_states(self):
        """Restore the initial state estimate and covariance."""
        self._x = self._x0.copy()
        self._P = self._P0.copy()
        self._last_t = None
