import numpy as np


class KalmanFilterData:
    """
    Container for the matrices of a :class:`KalmanFilter`, with FITS save/restore,
    so that a filter can be configured from a YAML file with
    ``kalman_filter_data_object: '<tag>'``.

    Fields have the same names, shapes and meaning as the KalmanFilter arguments
    (dense or per-mode "batched" shapes). The per-sensor fields
    (``observation_matrix``, ``measurement_noise_cov``, ``bias_process_noise_cov``,
    ``initial_bias``, ``initial_bias_covariance``) are lists with one item per
    sensor, in the order of the ``in_measurements_list`` input; the bias items
    may be None.

    FITS layout: the primary header holds ``N_SENS``; each non-None field is an
    image extension (``A``, ``Q``, ``B``, ``X0``, ``P0``, ``PROJ``; per-sensor:
    ``H_<i>``, ``R_<i>``, ``QB_<i>``, ``BX0_<i>``, ``BP0_<i>``). Scalars are
    stored as 1-element arrays flagged with ``SCALAR = T``.
    """

    _single = {'state_transition_matrix': 'A',
               'process_noise_cov': 'Q',
               'command_matrix': 'B',
               'initial_state': 'X0',
               'initial_covariance': 'P0',
               'projection_matrix': 'PROJ'}
    _per_sensor = {'observation_matrix': 'H',
                   'measurement_noise_cov': 'R',
                   'bias_process_noise_cov': 'QB',
                   'initial_bias': 'BX0',
                   'initial_bias_covariance': 'BP0'}

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
                 projection_matrix=None):

        as_np = lambda v: None if v is None else np.asarray(v, dtype=float)
        as_list = lambda v: list(v) if isinstance(v, (list, tuple)) else [v]

        self.state_transition_matrix = as_np(state_transition_matrix)
        self.process_noise_cov = as_np(process_noise_cov)
        self.command_matrix = as_np(command_matrix)
        self.initial_state = as_np(initial_state)
        self.initial_covariance = as_np(initial_covariance)
        self.projection_matrix = as_np(projection_matrix)

        self.observation_matrix = [as_np(h) for h in as_list(observation_matrix)]
        self.measurement_noise_cov = [as_np(r) for r in as_list(measurement_noise_cov)]
        self.n_sensors = len(self.observation_matrix)
        if len(self.measurement_noise_cov) != self.n_sensors:
            raise ValueError(f'Got {self.n_sensors} observation matrices but '
                             f'{len(self.measurement_noise_cov)} measurement noise covariances')

        def per_sensor(v, name):
            if v is None:
                return None
            items = as_list(v)
            if len(items) != self.n_sensors:
                raise ValueError(f'{name} has {len(items)} items, but there are {self.n_sensors} sensors')
            return [as_np(x) for x in items]

        self.bias_process_noise_cov = per_sensor(bias_process_noise_cov, 'bias_process_noise_cov')
        self.initial_bias = per_sensor(initial_bias, 'initial_bias')
        self.initial_bias_covariance = per_sensor(initial_bias_covariance, 'initial_bias_covariance')

    # ------------------------------------------------------------------
    def kalman_kwargs(self):
        """Keyword arguments for KalmanFilter (all matrix fields)."""
        return {name: getattr(self, name) for name in list(self._single) + list(self._per_sensor)}

    # ------------------------------------------------------------------
    # AR(2) turbulence model
    # ------------------------------------------------------------------
    @classmethod
    def from_ar2(cls, a1, a2, driving_noise_var, measurement_noise_var, delay=1.5,
                 initial_covariance='stationary'):
        """
        Per-mode (batched) Kalman model for modal turbulence following, for each
        mode j, the AR(2) process

            phi[k] = a1_j phi[k-1] + a2_j phi[k-2] + w[k],   w ~ N(0, driving_noise_var_j)

        measured directly (pseudo-open-loop measurements) by one or more sensors:
        z_i[k] = phi[k] + n_i[k], n_i ~ N(0, measurement_noise_var[i]_j).

        The per-mode state is [phi[k], phi[k-1]]. The projection matrix gives,
        from the a-posteriori estimate at step k, the prediction of phi at
        k + ``delay`` (linear interpolation between the integer-step
        predictions): for a 1.5-step loop delay it is the average of the one-
        and two-step-ahead predictions, which is the command to send to the
        delay chain.

        Parameters
        ----------
        a1, a2 : array-like, shape (n_modes,)
            AR(2) coefficients, identified at the loop sampling time.
        driving_noise_var : array-like, shape (n_modes,)
            Variance of the AR(2) driving (innovation) noise w, in the squared
            units of the modal measurements. If you saved the variance of phi
            itself, convert it with :func:`ar2_driving_noise_variance`.
        measurement_noise_var : array (n_modes,) or list of such arrays
            Measurement-noise variance of each mode, one array per sensor.
        delay : float
            Total loop delay in steps between the measurement and the command
            becoming effective (1 + the ``delay`` parameter of the IirFilter/
            Integrator that models the fractional part).
        initial_covariance : 'stationary' or None
            'stationary' starts from the stationary covariance of the AR(2)
            process (recommended: the first estimates are then not over-confident).
        """
        a1 = np.atleast_1d(np.asarray(a1, dtype=float))
        a2 = np.atleast_1d(np.asarray(a2, dtype=float))
        sw2 = np.atleast_1d(np.asarray(driving_noise_var, dtype=float))
        n_modes = a1.size
        if a2.size != n_modes or sw2.size != n_modes:
            raise ValueError('a1, a2 and driving_noise_var must have the same length')
        bad = np.where(~ar2_is_stationary(a1, a2))[0]
        if bad.size:
            raise ValueError(f'AR(2) coefficients are not stationary for modes {bad[:20].tolist()}'
                             f'{" ..." if bad.size > 20 else ""}')
        if np.any(sw2 < 0):
            raise ValueError('driving_noise_var must be non-negative')
        if delay < 0:
            raise ValueError('delay must be non-negative')

        A = np.zeros((n_modes, 2, 2))
        A[:, 0, 0] = a1
        A[:, 0, 1] = a2
        A[:, 1, 0] = 1.0
        Q = np.zeros((n_modes, 2, 2))
        Q[:, 0, 0] = sw2

        r_list = measurement_noise_var if isinstance(measurement_noise_var, (list, tuple)) \
            else [measurement_noise_var]
        R = []
        for i, r in enumerate(r_list):
            r = np.atleast_1d(np.asarray(r, dtype=float))
            if r.size != n_modes:
                raise ValueError(f'measurement_noise_var[{i}] has {r.size} values, expected {n_modes}')
            R.append(r)
        H = [np.tile(np.array([[1.0, 0.0]]), (n_modes, 1, 1)) for _ in R]

        P0 = stationary_covariance(A, Q) if initial_covariance == 'stationary' else None

        return cls(state_transition_matrix=A, observation_matrix=H, process_noise_cov=Q,
                   measurement_noise_cov=R, initial_covariance=P0,
                   projection_matrix=prediction_projection(A, delay))

    @classmethod
    def from_ar2_closed_loop(cls, a1, a2, driving_noise_var, measurement_noise_var,
                             integration_steps=1, measurement_delay=0, delay=1.5,
                             bias_process_noise_cov=None, initial_bias=None,
                             initial_bias_covariance=None):
        """
        Per-mode AR(2) model fed directly with the closed-loop modal measurements
        and the DM command, for sensors that may integrate over several loop
        steps (slower frame rate) and carry a bias.

        State of each mode (L = number of lags kept)::

            [phi_k, ..., phi_{k-L+1}, d_k, ..., d_{k-L+1}]

        where phi is the turbulence (AR(2)) and d the modal DM command seen by
        the sensors at each step. d is a known input: the KalmanFilter command
        input must be the command seen at the current step (in the Ristretto
        YAML, 'kdelay.out_comm:-1'), which is shifted into d_k at each
        prediction. Sensor i, when refreshed at step k, measures

            z_i[k] = mean over j = D_i .. D_i + T_i - 1 of (phi_{k-j} - d_{k-j})
                     (+ b_i if sensor i is biased) + noise

        with T_i = integration_steps[i] (frame time in loop steps) and
        D_i = measurement_delay[i]. The command projection is the prediction of
        phi ``delay`` steps ahead, as in :meth:`from_ar2`.

        Parameters
        ----------
        a1, a2, driving_noise_var : array-like, shape (n_modes,)
            AR(2) model of each mode (see :meth:`from_ar2`).
        measurement_noise_var : list of arrays (n_modes,)
            Noise variance of each sensor, per frame of that sensor.
        integration_steps : int or list of int
            Frame time of each sensor in loop steps (e.g. 4 for 500 Hz in a
            2 kHz loop).
        measurement_delay : int or list of int
            Extra delay, in steps, between the end of a frame and the step at
            which its measurement is available (0 if the measurement comes out
            at the last integrated step).
        delay : float
            Total plant delay in steps (1 + the fractional delay of the
            pass-through filter).
        bias_process_noise_cov, initial_bias, initial_bias_covariance :
            Per-sensor bias settings, as for KalmanFilter (None for unbiased
            sensors, e.g. the reference one).
        """
        a1 = np.atleast_1d(np.asarray(a1, dtype=float))
        a2 = np.atleast_1d(np.asarray(a2, dtype=float))
        sw2 = np.atleast_1d(np.asarray(driving_noise_var, dtype=float))
        n_modes = a1.size
        if a2.size != n_modes or sw2.size != n_modes:
            raise ValueError('a1, a2 and driving_noise_var must have the same length')
        bad = np.where(~ar2_is_stationary(a1, a2))[0]
        if bad.size:
            raise ValueError(f'AR(2) coefficients are not stationary for modes {bad[:20].tolist()}'
                             f'{" ..." if bad.size > 20 else ""}')
        if np.any(sw2 < 0):
            raise ValueError('driving_noise_var must be non-negative')
        if delay < 0:
            raise ValueError('delay must be non-negative')

        r_list = list(measurement_noise_var) if isinstance(measurement_noise_var, (list, tuple)) \
            else [measurement_noise_var]
        n_sensors = len(r_list)

        def per_sensor_int(v, name):
            items = list(v) if isinstance(v, (list, tuple)) else [v] * n_sensors
            if len(items) != n_sensors:
                raise ValueError(f'{name} has {len(items)} items, but there are {n_sensors} sensors')
            items = [int(x) for x in items]
            return items

        steps = per_sensor_int(integration_steps, 'integration_steps')
        delays = per_sensor_int(measurement_delay, 'measurement_delay')
        if any(t < 1 for t in steps) or any(d < 0 for d in delays):
            raise ValueError('integration_steps must be >= 1 and measurement_delay >= 0')
        L = max(2, max(t + d for t, d in zip(steps, delays)))
        ns = 2 * L

        A_phi = np.zeros((n_modes, L, L))
        A_phi[:, 0, 0] = a1
        A_phi[:, 0, 1] = a2
        for j in range(1, L):
            A_phi[:, j, j - 1] = 1.0
        Q_phi = np.zeros((n_modes, L, L))
        Q_phi[:, 0, 0] = sw2

        A = np.zeros((n_modes, ns, ns))
        A[:, :L, :L] = A_phi
        for j in range(1, L):
            A[:, L + j, L + j - 1] = 1.0          # d_k is set by the command input
        Q = np.zeros((n_modes, ns, ns))
        Q[:, :L, :L] = Q_phi
        B = np.zeros((ns, 1))
        B[L, 0] = 1.0

        H, R = [], []
        for i, (r, t, d) in enumerate(zip(r_list, steps, delays)):
            r = np.atleast_1d(np.asarray(r, dtype=float))
            if r.size != n_modes:
                raise ValueError(f'measurement_noise_var[{i}] has {r.size} values, expected {n_modes}')
            h = np.zeros((1, ns))
            h[0, d:d + t] = 1.0 / t
            h[0, L + d:L + d + t] = -1.0 / t
            H.append(h)
            R.append(r)

        P0 = np.zeros((n_modes, ns, ns))
        P0[:, :L, :L] = stationary_covariance(A_phi, Q_phi)    # DM commands are known exactly

        C = np.zeros((n_modes, ns))
        C[:, :L] = prediction_projection(A_phi, delay)

        return cls(state_transition_matrix=A, observation_matrix=H, process_noise_cov=Q,
                   measurement_noise_cov=R, command_matrix=B, initial_covariance=P0,
                   bias_process_noise_cov=bias_process_noise_cov, initial_bias=initial_bias,
                   initial_bias_covariance=initial_bias_covariance, projection_matrix=C)

    # ------------------------------------------------------------------
    # FITS I/O
    # ------------------------------------------------------------------
    def save(self, filename, overwrite=True):
        from astropy.io import fits

        def hdu(value, name):
            value = np.asarray(value, dtype=float)
            h = fits.ImageHDU(np.atleast_1d(value), name=name)
            h.header['SCALAR'] = value.ndim == 0
            return h

        primary = fits.PrimaryHDU()
        primary.header['OBJ_TYPE'] = 'KalmanFilterData'
        primary.header['N_SENS'] = self.n_sensors
        hdus = [primary]
        for attr, ext in self._single.items():
            value = getattr(self, attr)
            if value is not None:
                hdus.append(hdu(value, ext))
        for attr, ext in self._per_sensor.items():
            items = getattr(self, attr)
            for i, value in enumerate(items or []):
                if value is not None:
                    hdus.append(hdu(value, f'{ext}_{i}'))
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)

    @classmethod
    def restore(cls, filename, target_device_idx=None, **kwargs):
        """Load from FITS. Arrays stay on the CPU (the KalmanFilter moves them)."""
        from astropy.io import fits

        with fits.open(filename) as hdul:
            n_sensors = int(hdul[0].header['N_SENS'])
            data = {}
            for h in hdul[1:]:
                value = np.array(h.data, dtype=float)
                if h.header.get('SCALAR', False):
                    value = value.reshape(())
                data[h.name] = value

        fields = {attr: data.get(ext) for attr, ext in cls._single.items()}
        for attr, ext in cls._per_sensor.items():
            items = [data.get(f'{ext}_{i}') for i in range(n_sensors)]
            fields[attr] = items if any(v is not None for v in items) else None
        return cls(**fields)


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
