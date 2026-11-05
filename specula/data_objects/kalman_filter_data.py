import numpy as np
from astropy.io import fits

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class KalmanFilterData(BaseDataObj):
    """
    Linear state-space model used by
    :class:`~specula.processing_objects.kalman_filter.KalmanFilter`.
    In YAML: ``kalman_filter_data_object: '<tag>'`` (calibration folder 'filter').

    Model, for one filter step::

        x[k+1] = A x[k] + B u[k] + w              w   ~ N(0, Q)
        z_i[k] = H_i x[k] + D_i U[k] + v_i        v_i ~ N(0, R_i)

    - i indexes the sensors (order of the KalmanFilter ``in_measurements_list``).
    - u[k] is a known command; U[k] = [u[k], u[k-1], ..., u[k-L+1]] its recent
      history. B (command matrix) makes the command act on the state;
      D_i (command observation matrix) makes it appear directly in the
      measurements, e.g. the DM shapes seen by a WFS during its frame. Known
      commands need no state, which keeps the covariance small.
    - Optional sensor biases: sensor i gets z_i += b_i, with
      b_i[k+1] = b_i[k] + wb_i, wb_i ~ N(0, Qb_i). Biases are appended to the
      state (one per measurement channel) and estimated with it.
    - Optional projection C: the filter outputs C x (e.g. the command to send).
    - Optional flux scaling of the measurement noise: with the photon- and
      read-out-noise parts of R_i at a reference flux F_ref, the filter rescales
      R_i with the measured flux F (PyrSlopec 'out_flux', global normalisation):
      photon part x F_ref / F, read-out part x (F_ref / F)^2, rest unchanged.
      TODO: full spatial model from the per-pixel pupil intensities.

    Dense or per-mode
    -----------------
    A of shape (n, n) describes one dense filter. A of shape (n_modes, n, n)
    describes n_modes independent small filters (e.g. one per KL mode), all
    run at once. Accepted per-mode shapes:

    - A, Q: (n_modes, n, n); Q also (n_modes, n) for a diagonal Q
    - H_i: (n_modes, m_i, n), or (m_i, n) shared by all modes
    - R_i, Qb_i, initial bias covariances: scalar, (n_modes,) one variance per
      mode, (n_modes, m_i) diagonal, or (n_modes, m_i, m_i)
    - B: (n_modes, n, p) or (n, p) shared; command input of n_modes * p values
    - D_i: (n_modes, m_i, L) or (m_i, L) shared; one command value per mode
    - initial_state: (n_modes, n) or (n_modes * n,); initial_covariance:
      (n_modes, n, n) or (n, n) shared; initial_bias: (n_modes, m_i)
    - projection C: (n_modes, q, n), or (n_modes, n) for q = 1
    - photon_noise_var_i, readout_noise_var_i: variances, (n_modes * m_i,) or
      (n_modes, m_i); reference_flux_i: scalar [e- per frame of that sensor]

    Attributes
    ----------
    The constructor arguments are kept as given (numpy arrays, what save()
    writes). The model actually used by the filter is built from them, on the
    target device, with a leading mode axis (size 1 for a dense filter) and the
    biases appended to the state:

    A, Q, P0 (n_modes, n_aug, n_aug); x0 (n_modes, n_aug); B (n_modes, n_aug, p)
    or None; H[i] (n_modes, m_i, n_aug); R[i] (n_modes, m_i, m_i); D[i]
    (n_modes, m_i, L) or None; C (n_modes, q, n) or None; noise_scaling[i]
    (photon (n_modes, m_i), read-out (n_modes, m_i), F_ref) or None; with
    n_aug = n_states + n_bias.

    FITS layout (VERSION 1): primary header with N_SENS; one image extension per
    given argument (A, Q, B, X0, P0, PROJ; per sensor H_<i>, R_<i>, D_<i>,
    QB_<i>, BX0_<i>, BP0_<i>, PN_<i>, RN_<i>, RF_<i>); scalars are 1-element
    arrays with SCALAR = T.
    """

    _single = {'state_transition_matrix': 'A',
               'process_noise_cov': 'Q',
               'command_matrix': 'B',
               'initial_state': 'X0',
               'initial_covariance': 'P0',
               'projection_matrix': 'PROJ'}
    _per_sensor = {'observation_matrix': 'H',
                   'measurement_noise_cov': 'R',
                   'command_observation_matrix': 'D',
                   'bias_process_noise_cov': 'QB',
                   'initial_bias': 'BX0',
                   'initial_bias_covariance': 'BP0',
                   'photon_noise_var': 'PN',
                   'readout_noise_var': 'RN',
                   'reference_flux': 'RF'}

    def __init__(self,
                 state_transition_matrix,
                 observation_matrix,
                 process_noise_cov,
                 measurement_noise_cov,
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
                 target_device_idx: int = None,
                 precision: int = None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        given = dict(state_transition_matrix=state_transition_matrix,
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
        for name in ('state_transition_matrix', 'observation_matrix',
                     'process_noise_cov', 'measurement_noise_cov'):
            if given[name] is None:
                raise ValueError(f'{name} is required')

        # Arguments as given (CPU), one item per sensor for the per-sensor ones
        self.n_sensors = len(_as_list(observation_matrix))
        for name in self._single:
            setattr(self, name, _as_numpy(given[name]))
        for name in self._per_sensor:
            items = given[name]
            if items is not None:
                items = _as_list(items)
                if len(items) != self.n_sensors:
                    raise ValueError(f'{name} has {len(items)} items, but there are {self.n_sensors} sensors')
                items = [_as_numpy(v) for v in items]
            setattr(self, name, items)

        self._build_model()

    def kalman_kwargs(self):
        """The constructor arguments, e.g. to build a KalmanFilter directly."""
        return {name: getattr(self, name) for name in list(self._single) + list(self._per_sensor)}

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------
    def _build_model(self):
        """Per-mode model with the biases appended to the state (numpy, then device)."""
        A = self.state_transition_matrix
        if A.ndim not in (2, 3):
            raise ValueError(f'state_transition_matrix must be 2-D (dense) or 3-D (per mode), got shape {A.shape}')
        self.batched = A.ndim == 3
        A = A if self.batched else A[None]
        N, n, n2 = A.shape
        if n != n2:
            raise ValueError(f'state_transition_matrix must be square, got {A.shape}')
        self.n_modes, self.n_states = N, n

        Q = self.process_noise_cov
        if self.batched and Q.shape == (N, n):
            Q = Q[:, :, None] * np.eye(n)
        else:
            Q = self._matrix(Q, n, n, 'process_noise_cov', shared_ok=False)

        H = [self._matrix(h, None, n, f'observation_matrix[{i}]')
             for i, h in enumerate(self.observation_matrix)]
        sizes = [h.shape[1] for h in H]
        R = [self.per_mode_covariance(r, sizes[i], f'measurement_noise_cov[{i}]')
             for i, r in enumerate(self.measurement_noise_cov)]

        B = None if self.command_matrix is None else \
            self._matrix(self.command_matrix, n, None, 'command_matrix')
        D = [None] * self.n_sensors
        if self.command_observation_matrix is not None:
            D = [None if d is None else self._matrix(d, sizes[i], None, f'command_observation_matrix[{i}]')
                 for i, d in enumerate(self.command_observation_matrix)]
            lags = {d.shape[2] for d in D if d is not None}
            if len(lags) > 1:
                raise ValueError('all command observation matrices must cover the same number of command lags')
            if B is not None and B.shape[2] != 1:
                raise ValueError('command_observation_matrix needs one command value per mode (p = 1)')
        self.command_lags = max((d.shape[2] for d in D if d is not None), default=0)

        x0 = np.zeros((N, n)) if self.initial_state is None else \
            self._vector(self.initial_state, n, 'initial_state')
        P0 = np.broadcast_to(np.eye(n), (N, n, n)) if self.initial_covariance is None else \
            self._matrix(self.initial_covariance, n, n, 'initial_covariance')

        C = self.projection_matrix
        if C is not None:
            if self.batched and C.ndim == 2:
                C = C[:, None, :]                          # one output per mode
            C = self._matrix(C, None, n, 'projection_matrix', shared_ok=False)

        # --- biases: one random-walk state per channel of each biased sensor
        bias_q = self.bias_process_noise_cov or [None] * self.n_sensors
        bias_x0 = self.initial_bias or [None] * self.n_sensors
        bias_p0 = self.initial_bias_covariance or [None] * self.n_sensors
        self.bias_sensors = [i for i, q in enumerate(bias_q) if q is not None]
        for i in range(self.n_sensors):
            if bias_q[i] is None and (bias_x0[i] is not None or bias_p0[i] is not None):
                raise ValueError(f'Initial bias given for sensor {i}, '
                                 f'but bias_process_noise_cov[{i}] is None (no bias)')
        self.n_bias = sum(sizes[i] for i in self.bias_sensors)
        na = n + self.n_bias

        def grow(M, rows=True, cols=True):
            """Pad the state dimension(s) of M with zeros for the bias states."""
            pad = [(0, 0)] * M.ndim
            if rows:
                pad[1] = (0, na - n)
            if cols:
                pad[-1] = (0, na - n)
            return np.pad(M, pad)

        A, Q, P0 = grow(A), grow(Q), grow(P0)
        A[:, n:, n:] = np.eye(self.n_bias)
        x0 = grow(x0, rows=False)
        B = None if B is None else grow(B, cols=False)
        H = [grow(h, rows=False) for h in H]
        start = n
        for i in self.bias_sensors:
            m = sizes[i]
            s = slice(start, start + m)
            H[i][:, :, s] = np.eye(m)
            Q[:, s, s] = self.per_mode_covariance(bias_q[i], m, f'bias_process_noise_cov[{i}]')
            if bias_x0[i] is not None:
                x0[:, s] = self._vector(bias_x0[i], m, f'initial_bias[{i}]')
            P0[:, s, s] = np.eye(m) if bias_p0[i] is None else \
                self.per_mode_covariance(bias_p0[i], m, f'initial_bias_covariance[{i}]')
            start += m

        to_device = lambda v: None if v is None else self.to_xp(v, dtype=self.dtype)
        self.A, self.Q, self.B, self.C = to_device(A), to_device(Q), to_device(B), to_device(C)
        self.x0, self.P0 = to_device(x0), to_device(P0)
        self.H = [to_device(h) for h in H]
        self.R = [to_device(r) for r in R]
        self.D = [to_device(d) for d in D]

        # --- flux scaling of the measurement noise
        self.noise_scaling = [None] * self.n_sensors
        photon = self.photon_noise_var or [None] * self.n_sensors
        readout = self.readout_noise_var or [None] * self.n_sensors
        f_ref = self.reference_flux or [None] * self.n_sensors
        for i in range(self.n_sensors):
            given = [v is not None for v in (photon[i], readout[i], f_ref[i])]
            if not any(given):
                continue
            if not all(given):
                raise ValueError(f'sensor {i}: photon_noise_var, readout_noise_var and reference_flux '
                                 f'must be given together')
            if float(f_ref[i]) <= 0:
                raise ValueError(f'reference_flux[{i}] must be positive')
            self.noise_scaling[i] = (to_device(self._vector(photon[i], sizes[i], f'photon_noise_var[{i}]')),
                                     to_device(self._vector(readout[i], sizes[i], f'readout_noise_var[{i}]')),
                                     float(f_ref[i]))

    def _matrix(self, value, rows, cols, name, shared_ok=True):
        """Per-mode matrix -> (n_modes, rows, cols); rows/cols None mean any size."""
        N = self.n_modes
        m = np.asarray(value, dtype=float)
        if not self.batched:
            if m.ndim != 2:
                raise ValueError(f'{name} must be 2-dimensional, got shape {m.shape}')
            m = m[None]
        elif m.ndim == 2 and shared_ok:
            m = np.broadcast_to(m, (N,) + m.shape)
        elif m.ndim != 3 or m.shape[0] != N:
            raise ValueError(f'{name} must have shape ({N}, rows, cols)'
                             f'{" or (rows, cols)" if shared_ok else ""}, got {m.shape}')
        if (rows is not None and m.shape[1] != rows) or (cols is not None and m.shape[2] != cols):
            raise ValueError(f'{name} must have {rows if rows is not None else "any"} x '
                             f'{cols if cols is not None else "any"} elements per mode, got {m.shape[1:]}')
        return np.array(m)

    def _vector(self, value, size, name):
        """Per-mode vector -> (n_modes, size)."""
        v = np.asarray(value, dtype=float)
        if v.size != self.n_modes * size:
            raise ValueError(f'{name} must have {self.n_modes * size} elements, got {v.size}')
        return v.reshape(self.n_modes, size)

    def per_mode_covariance(self, value, m, name='covariance'):
        """Covariance -> (n_modes, m, m), from a scalar, variances or matrices."""
        N = self.n_modes
        c = np.asarray(value, dtype=float)
        if c.ndim == 0:
            return np.broadcast_to(c * np.eye(m), (N, m, m)).copy()
        if not self.batched:                       # (m,) variances or (m, m)
            c = np.diag(c) if c.ndim == 1 and c.size == m else c
            if c.shape != (m, m):
                raise ValueError(f'{name} must have {m} variances or shape {(m, m)}, got {c.shape}')
            return c[None]
        if c.ndim == 1 and c.size == N:            # one variance per mode
            return c[:, None, None] * np.eye(m)
        if c.ndim == 2 and c.shape == (N, m):      # per-mode diagonal
            return c[:, :, None] * np.eye(m)
        if c.shape != (N, m, m):
            raise ValueError(f'{name} must be a scalar or have shape ({N},), {(N, m)} or {(N, m, m)}, '
                             f'got {c.shape}')
        return c

    # ------------------------------------------------------------------
    # AR(2) turbulence models
    # ------------------------------------------------------------------
    @classmethod
    def from_ar2(cls, a1, a2, driving_noise_var, measurement_noise_var, delay=1.5,
                 initial_covariance='stationary', target_device_idx=None, precision=None):
        """
        Pseudo-open-loop model: per mode, AR(2) turbulence

            phi[k] = a1 phi[k-1] + a2 phi[k-2] + w[k],   var(w) = driving_noise_var

        measured directly by each sensor: z_i[k] = phi[k] + n_i[k]. State
        [phi[k], phi[k-1]]; the projection is the prediction of phi ``delay``
        steps ahead (the command for a ``delay``-step plant).

        a1, a2, driving_noise_var: (n_modes,) arrays. measurement_noise_var:
        (n_modes,) array, or a list of them (one per sensor).
        initial_covariance='stationary' starts from the AR(2) stationary
        covariance (None: identity).
        """
        a1, a2, sw2 = _check_ar2(a1, a2, driving_noise_var, delay)
        A, Q = _ar2_companion(a1, a2, sw2, 2)
        R = _check_noise_list(measurement_noise_var, a1.size)
        H = [np.tile([[1.0, 0.0]], (a1.size, 1, 1)) for _ in R]
        P0 = stationary_covariance(A, Q) if initial_covariance == 'stationary' else None
        return cls(state_transition_matrix=A, observation_matrix=H, process_noise_cov=Q,
                   measurement_noise_cov=R, initial_covariance=P0,
                   projection_matrix=prediction_projection(A, delay),
                   target_device_idx=target_device_idx, precision=precision)

    @classmethod
    def from_ar2_closed_loop(cls, a1, a2, driving_noise_var, measurement_noise_var,
                             integration_steps=1, measurement_delay=0, delay=1.5, command_hold_steps=1,
                             bias_process_noise_cov=None, initial_bias=None,
                             initial_bias_covariance=None, aliasing_var=None,
                             aliasing_a1=None, aliasing_a2=None, photon_noise_var=None,
                             readout_noise_var=None, reference_flux=None,
                             target_device_idx=None, precision=None):
        """
        Closed-loop model: the filter receives the raw modal measurements and,
        as command input, the modal DM command d[k] seen by the sensors at
        step k (in the Ristretto YAML, 'kpred.out_comm:-1'). Per mode::

            phi[k]   = a1 phi[k-1] + a2 phi[k-2] + w[k]                 (turbulence)
            alpha_i  : same form, coefficients aliasing_a1/a2           (aliasing, optional)
            z_i[k]   = mean_{j = D_i .. D_i+T_i-1} (phi - d + alpha_i)[k-j]
                       (+ b_i if biased) + noise

        T_i = integration_steps[i] is the frame of sensor i in loop steps (4 for
        500 Hz in a 2 kHz loop) and D_i = measurement_delay[i] an extra readout
        delay in steps. The DM commands are known, so they enter through the
        command observation matrices, not the state.

        State per mode: [phi lags | alpha_0 lags | alpha_1 lags | ...] (+ biases,
        appended by the constructor). The projection is the prediction of phi
        ``delay`` steps ahead: 1 (E[phi[k+1] | k]) for the Ristretto chain, where
        the plant-inverse IirFilter compensates the 1.5-step plant.
        With ``command_hold_steps`` T > 1 (filter and command at a frame rate
        T times slower than the loop, single slow sensor), the projection is the
        mean of the 1..T-step predictions, i.e. phi over the next frame.

        Parameters
        ----------
        a1, a2, driving_noise_var : (n_modes,) arrays
            AR(2) turbulence model at the loop rate.
        measurement_noise_var : list of (n_modes,) arrays
            Noise variance of each sensor, per frame of that sensor.
        integration_steps, measurement_delay : int or list of int
        delay : float
            Prediction horizon of the projection, in steps (ignored if command_hold_steps > 1).
        command_hold_steps : int
            Steps for which each command is held (1: a new command every step).
        bias_process_noise_cov, initial_bias, initial_bias_covariance :
            Per-sensor bias settings (None for unbiased sensors, e.g. the reference).
        aliasing_var : None or list of per-sensor items
            Single-step (loop-rate) aliasing variance of each sensor, (n_modes,)
            or scalar, None for a sensor without aliasing states. Do not also
            include it in measurement_noise_var.
        aliasing_a1, aliasing_a2 : float or (n_modes,)
            AR(2) coefficients of the aliasing, e.g. those of mode N+1.
            TODO: the aliasing blocks are independent per sensor; a shared state
            if the correlation between sensors proves significant.
        photon_noise_var, readout_noise_var, reference_flux : None or lists
            Flux scaling of the measurement noise (see the class docstring);
            None for the sensors whose noise is kept fixed.
        """
        a1, a2, sw2 = _check_ar2(a1, a2, driving_noise_var, delay)
        N = a1.size
        R = _check_noise_list(measurement_noise_var, N)
        n_sensors = len(R)
        steps = _per_sensor_ints(integration_steps, n_sensors, 'integration_steps')
        delays = _per_sensor_ints(measurement_delay, n_sensors, 'measurement_delay')
        if min(steps) < 1 or min(delays) < 0:
            raise ValueError('integration_steps must be >= 1 and measurement_delay >= 0')
        windows = [slice(d, d + t) for t, d in zip(steps, delays)]   # lags seen by each sensor
        L = max(2, max(w.stop for w in windows))

        # turbulence block, then one aliasing block per sensor that has one
        blocks = [_ar2_companion(a1, a2, sw2, L)]
        alias = [None] * n_sensors if aliasing_var is None else _as_list(aliasing_var)
        if len(alias) != n_sensors:
            raise ValueError(f'aliasing_var has {len(alias)} items, but there are {n_sensors} sensors')
        alias_offset = {}
        if any(v is not None for v in alias):
            if aliasing_a1 is None or aliasing_a2 is None:
                raise ValueError('aliasing_a1 and aliasing_a2 are required with aliasing_var')
            aa1 = np.broadcast_to(np.asarray(aliasing_a1, dtype=float), (N,))
            aa2 = np.broadcast_to(np.asarray(aliasing_a2, dtype=float), (N,))
            if not np.all(ar2_is_stationary(aa1, aa2)):
                raise ValueError('aliasing AR(2) coefficients are not stationary')
            offset = L
            for i, var in enumerate(alias):
                if var is None:
                    continue
                var = np.broadcast_to(np.asarray(var, dtype=float), (N,))
                if np.any(var < 0) or not np.all(np.isfinite(var)):
                    raise ValueError(f'aliasing_var[{i}] must be finite and non-negative')
                n_lags = max(2, windows[i].stop)
                blocks.append(_ar2_companion(aa1, aa2, ar2_driving_noise_variance(aa1, aa2, var), n_lags))
                alias_offset[i] = offset
                offset += n_lags

        n = sum(Ab.shape[1] for Ab, _ in blocks)
        A, Q, P0 = (np.zeros((N, n, n)) for _ in range(3))
        start = 0
        for Ab, Qb in blocks:
            s = slice(start, start + Ab.shape[1])
            A[:, s, s], Q[:, s, s], P0[:, s, s] = Ab, Qb, stationary_covariance(Ab, Qb)
            start = s.stop

        H, Dm = [], []
        for i, (t, w) in enumerate(zip(steps, windows)):
            h = np.zeros((1, n))
            h[0, w] = 1.0 / t                                         # turbulence, frame-averaged
            if i in alias_offset:
                o = alias_offset[i]
                h[0, o + w.start:o + w.stop] = 1.0 / t                # aliasing, frame-averaged
            d = np.zeros((1, L))
            d[0, w] = -1.0 / t                                        # minus the DM commands seen
            H.append(h)
            Dm.append(d)

        C = np.zeros((N, n))
        if command_hold_steps > 1:
            # TODO: commands at every loop step would gain ~3% variance at R=1, ~0.5% at R=25
            C[:, :L] = np.mean([prediction_projection(blocks[0][0], float(j))
                                for j in range(1, command_hold_steps + 1)], axis=0)
        else:
            C[:, :L] = prediction_projection(blocks[0][0], delay)

        return cls(state_transition_matrix=A, observation_matrix=H, process_noise_cov=Q,
                   measurement_noise_cov=R, command_observation_matrix=Dm, initial_covariance=P0,
                   bias_process_noise_cov=bias_process_noise_cov, initial_bias=initial_bias,
                   initial_bias_covariance=initial_bias_covariance, projection_matrix=C,
                   photon_noise_var=photon_noise_var, readout_noise_var=readout_noise_var,
                   reference_flux=reference_flux,
                   target_device_idx=target_device_idx, precision=precision)

    # ------------------------------------------------------------------
    # FITS I/O and BaseDataObj interface
    # ------------------------------------------------------------------
    def save(self, filename, overwrite=True):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['N_SENS'] = self.n_sensors
        hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])

        def append(value, name):
            value = cpuArray(value)
            hdu = fits.ImageHDU(data=np.atleast_1d(value), name=name)
            hdu.header['SCALAR'] = value.ndim == 0
            hdul.append(hdu)

        for name, ext in self._single.items():
            if getattr(self, name) is not None:
                append(getattr(self, name), ext)
        for name, ext in self._per_sensor.items():
            for i, value in enumerate(getattr(self, name) or []):
                if value is not None:
                    append(value, f'{ext}_{i}')
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def restore(filename, target_device_idx=None):
        # pylint: disable=no-member # members of HDUList[i] are created dynamically by pyfits
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            # files written before VERSION was added carry OBJ_TYPE only (same layout as version 1)
            version = hdr.get('VERSION', 1 if hdr.get('OBJ_TYPE') == 'KalmanFilterData' else None)
            if version != 1:
                raise ValueError(f"Error: unknown version {version} in file {filename}")
            n_sensors = int(hdr['N_SENS'])
            data = {}
            for hdu in hdul[1:]:
                value = np.array(hdu.data)
                data[hdu.name] = value.reshape(()) if hdu.header.get('SCALAR', False) else value

        fields = {name: data.get(ext) for name, ext in KalmanFilterData._single.items()}
        for name, ext in KalmanFilterData._per_sensor.items():
            items = [data.get(f'{ext}_{i}') for i in range(n_sensors)]
            fields[name] = items if any(v is not None for v in items) else None
        return KalmanFilterData(**fields, target_device_idx=target_device_idx)

    def get_fits_header(self):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_header(hdr):
        # TODO
        raise NotImplementedError()

    def get_value(self):
        # TODO
        raise NotImplementedError()

    def set_value(self, v):
        # TODO
        raise NotImplementedError()


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def _as_list(value):
    """Lists/tuples hold one item per sensor; anything else is a single sensor."""
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _as_numpy(value):
    return None if value is None else np.asarray(cpuArray(value), dtype=float)


def _per_sensor_ints(value, n_sensors, name):
    items = _as_list(value) if isinstance(value, (list, tuple)) else [value] * n_sensors
    if len(items) != n_sensors:
        raise ValueError(f'{name} has {len(items)} items, but there are {n_sensors} sensors')
    return [int(v) for v in items]


def _check_ar2(a1, a2, driving_noise_var, delay):
    a1, a2, sw2 = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (a1, a2, driving_noise_var))
    if not a1.size == a2.size == sw2.size:
        raise ValueError('a1, a2 and driving_noise_var must have the same length')
    bad = np.where(~ar2_is_stationary(a1, a2))[0]
    if bad.size:
        raise ValueError(f'AR(2) coefficients are not stationary for modes {bad[:20].tolist()}'
                         f'{" ..." if bad.size > 20 else ""}')
    if np.any(sw2 < 0):
        raise ValueError('driving_noise_var must be non-negative')
    if delay < 0:
        raise ValueError('delay must be non-negative')
    return a1, a2, sw2


def _check_noise_list(measurement_noise_var, n_modes):
    R = [np.atleast_1d(np.asarray(r, dtype=float)) for r in _as_list(measurement_noise_var)]
    for i, r in enumerate(R):
        if r.size != n_modes:
            raise ValueError(f'measurement_noise_var[{i}] has {r.size} values, expected {n_modes}')
    return R


def _ar2_companion(a1, a2, driving_noise_var, n_lags):
    """A, Q of an AR(2) process keeping n_lags >= 2 lags: state [y[k], ..., y[k-n_lags+1]]."""
    N = np.size(a1)
    A = np.zeros((N, n_lags, n_lags))
    A[:, 0, 0] = a1
    A[:, 0, 1] = a2
    for j in range(1, n_lags):
        A[:, j, j - 1] = 1.0
    Q = np.zeros((N, n_lags, n_lags))
    Q[:, 0, 0] = driving_noise_var
    return A, Q


# ----------------------------------------------------------------------
# AR(2) helpers
# ----------------------------------------------------------------------
def ar2_is_stationary(a1, a2):
    """True where phi[k] = a1 phi[k-1] + a2 phi[k-2] + w has its poles inside the unit circle."""
    a1, a2 = np.asarray(a1, dtype=float), np.asarray(a2, dtype=float)
    return (np.abs(a2) < 1) & (a1 + a2 < 1) & (a2 - a1 < 1)


def ar2_stationary_variance(a1, a2, driving_noise_var):
    """Variance of phi for an AR(2) process with driving-noise variance sigma_w^2."""
    a1, a2, s = (np.asarray(v, dtype=float) for v in (a1, a2, driving_noise_var))
    return s * (1 - a2) / ((1 + a2) * ((1 - a2) ** 2 - a1 ** 2))


def ar2_driving_noise_variance(a1, a2, stationary_var):
    """Inverse of :func:`ar2_stationary_variance`: sigma_w^2 from the variance of phi."""
    a1, a2, v = (np.asarray(x, dtype=float) for x in (a1, a2, stationary_var))
    return v * (1 + a2) * ((1 - a2) ** 2 - a1 ** 2) / (1 - a2)


def stationary_covariance(A, Q):
    """Solution P of P = A P A' + Q for each mode, A and Q of shape (n_modes, n, n)."""
    A, Q = np.asarray(A, dtype=float), np.asarray(Q, dtype=float)
    n_modes, n, _ = A.shape
    kron = np.einsum('mik,mjl->mijkl', A, A).reshape(n_modes, n * n, n * n)
    vecP = np.linalg.solve(np.eye(n * n) - kron, Q.reshape(n_modes, n * n, 1))
    P = vecP.reshape(n_modes, n, n)
    return 0.5 * (P + np.swapaxes(P, 1, 2))


def prediction_projection(A, delay):
    """Row(s) giving the prediction of the first state component `delay` steps
    ahead (linear interpolation for fractional delays): shape (n_modes, n)."""
    A = np.asarray(A, dtype=float)
    n_modes, n, _ = A.shape
    lo = int(np.floor(delay))
    frac = delay - lo
    e1 = np.zeros((n_modes, 1, n))
    e1[:, 0, 0] = 1.0
    row_lo = e1 @ np.linalg.matrix_power(A, lo)
    row = row_lo if frac == 0 else (1 - frac) * row_lo + frac * (row_lo @ A)
    return row[:, 0, :]
