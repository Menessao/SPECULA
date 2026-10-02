import specula
specula.init(0, precision=0)  # Default target device, double precision

import unittest
import numpy as np
from scipy.linalg import block_diag, solve_discrete_are

from specula import cpuArray
from specula.base_value import BaseValue
from specula.processing_objects.kalman_filter import KalmanFilter
from test.specula_testlib import cpu_and_gpu


# ----------------------------------------------------------------------
# Wikipedia "Kalman filter" practical example: a truck on frictionless,
# straight rails, buffeted by random uncontrolled forces. Its position is
# measured every dt seconds with an imprecise sensor (e.g. GPS).
#
#   state          x = [position, velocity]
#   transition     F = [[1, dt], [0, 1]]
#   process noise  Q = G G^T sigma_a^2,   G = [dt^2/2, dt]^T
#                  (random acceleration a_k ~ N(0, sigma_a^2))
#   observation    H = [1, 0]
#   meas. noise    R = [sigma_z^2]
#
# The article also mentions a second, smoother-but-drifting source of
# information (dead reckoning from wheel revolutions), which is used here as
# a second sensor measuring the velocity, and a known control input, which
# is used here as a commanded acceleration.
# ----------------------------------------------------------------------

def truck_model(dt=1.0, sigma_a=1.0, sigma_z=1.0):
    F = np.array([[1.0, dt], [0.0, 1.0]])
    G = np.array([[0.5 * dt ** 2], [dt]])
    H = np.array([[1.0, 0.0]])
    Q = sigma_a ** 2 * (G @ G.T)
    R = np.array([[sigma_z ** 2]])
    return F, G, H, Q, R


def simulate_truck(F, G, H, sigma_a, sigma_z, n_steps, rng, B=None, u=None):
    """Truth x_k = F x_{k-1} + G a_k (+ B u_k), measurement z_k = H x_k + v_k."""
    x = np.zeros(2)
    xs, zs = [], []
    for k in range(n_steps):
        x = F @ x + (G[:, 0] * rng.normal(0.0, sigma_a))
        if B is not None:
            x = x + B @ u[k]
        xs.append(x.copy())
        zs.append(H @ x + rng.normal(0.0, sigma_z, size=H.shape[0]))
    return np.array(xs), np.array(zs)


class ReferenceKalmanFilter:
    """Plain-numpy textbook Kalman filter (Wikipedia equations, explicit inverse)."""

    def __init__(self, F, Q, x0, P0, B=None):
        self.F, self.Q, self.B = F, Q, B
        self.x, self.P = x0.copy(), P0.copy()

    def predict(self, u=None):
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, H, R):
        y = z - H @ self.x                                  # innovation
        S = H @ self.P @ H.T + R                            # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)                 # optimal Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P


class KalmanRig:
    """Connects a KalmanFilter to plain BaseValue inputs and steps it
    following the SPECULA life cycle (check_ready, prepare_trigger,
    trigger_code, post_trigger)."""

    def __init__(self, kf, sensor_sizes, xp, target_device_idx, command_size=None):
        self.kf = kf
        self.xp = xp
        self.meas = [BaseValue(value=xp.zeros(m), target_device_idx=target_device_idx)
                     for m in sensor_sizes]
        kf.inputs['in_measurements_list'].set(self.meas)
        self.cmd = None
        if command_size is not None:
            self.cmd = BaseValue(value=xp.zeros(command_size),
                                 target_device_idx=target_device_idx)
            kf.inputs['in_command'].set(self.cmd)
        kf.setup()

    def step(self, t, fresh=None, command=None):
        """fresh: {sensor index: measurement vector} of the sensors refreshed at time t."""
        for i, z in (fresh or {}).items():
            self.meas[i].value[:] = self.xp.asarray(z)
            self.meas[i].generation_time = t
        if command is not None:
            self.cmd.value[:] = self.xp.asarray(command)
            self.cmd.generation_time = t
        self.kf.check_ready(t)
        self.kf.prepare_trigger(t)
        self.kf.trigger_code()
        self.kf.post_trigger()
        return (cpuArray(self.kf.outputs['out_state'].value).copy(),
                cpuArray(self.kf.outputs['out_covariance'].value).copy())


class TestKalmanFilter(unittest.TestCase):

    def _kf(self, target_device_idx, *args, **kwargs):
        return KalmanFilter(*args, target_device_idx=target_device_idx, precision=0, **kwargs)

    # ------------------------------------------------------------------
    # Single sensor: the Wikipedia truck example
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_truck_matches_reference_filter(self, target_device_idx, xp):
        """Every step of the truck example must agree with a textbook implementation."""
        dt, sigma_a, sigma_z = 1.0, 0.2, 2.0
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        P0 = np.diag([4.0, 1.0])  # initial position/velocity not known perfectly
        _, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 200, np.random.default_rng(1))

        kf = self._kf(target_device_idx, F, H, Q, R, initial_covariance=P0)
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        ref = ReferenceKalmanFilter(F, Q, np.zeros(2), P0)
        T = kf.seconds_to_t(dt)

        for k, z in enumerate(zs, start=1):
            x, P = rig.step(k * T, {0: z})
            ref.predict()
            ref.update(z, H, R)
            np.testing.assert_allclose(x, ref.x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P, ref.P, rtol=1e-8, atol=1e-10)

    @cpu_and_gpu
    def test_covariance_converges_to_riccati_solution(self, target_device_idx, xp):
        """With a perfectly known start (P0 = 0, as in the article) the covariance
        converges to the solution of the discrete algebraic Riccati equation."""
        dt = 1.0
        F, G, H, Q, R = truck_model(dt, sigma_a=1.0, sigma_z=1.0)

        kf = self._kf(target_device_idx, F, H, Q, R, initial_covariance=np.zeros((2, 2)))
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        T = kf.seconds_to_t(dt)
        for k in range(1, 61):
            _, P = rig.step(k * T, {0: [0.0]})

        P_prior = solve_discrete_are(F.T, H.T, Q, R)         # asymptotic P_{k|k-1}
        K = P_prior @ H.T @ np.linalg.inv(H @ P_prior @ H.T + R)
        P_post = (np.eye(2) - K @ H) @ P_prior               # asymptotic P_{k|k}
        np.testing.assert_allclose(P, P_post, rtol=1e-6)

    @cpu_and_gpu
    def test_covariance_does_not_depend_on_measurements(self, target_device_idx, xp):
        """Gain and covariance evolve independently of the measured values."""
        F, G, H, Q, R = truck_model(1.0, 0.5, 1.0)
        results = []
        for seed in (0, 1):
            kf = self._kf(target_device_idx, F, H, Q, R)
            rig = KalmanRig(kf, [1], xp, target_device_idx)
            T = kf.seconds_to_t(1.0)
            rng = np.random.default_rng(seed)
            for k in range(1, 31):
                _, P = rig.step(k * T, {0: rng.normal(0.0, 100.0, size=1)})
            results.append(P)
        np.testing.assert_allclose(results[0], results[1], rtol=1e-12)

    @cpu_and_gpu
    def test_filtering_beats_raw_measurements(self, target_device_idx, xp):
        """The filtered position is closer to the truth than the raw sensor reading,
        and the velocity (never measured) is recovered."""
        dt, sigma_a, sigma_z = 1.0, 0.1, 1.0
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        truth, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 500, np.random.default_rng(7))

        kf = self._kf(target_device_idx, F, H, Q, R)
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        T = kf.seconds_to_t(dt)
        est = np.array([rig.step(k * T, {0: z})[0] for k, z in enumerate(zs, start=1)])

        skip = 50  # let the filter converge
        raw_err = np.sqrt(np.mean((zs[skip:, 0] - truth[skip:, 0]) ** 2))
        pos_err = np.sqrt(np.mean((est[skip:, 0] - truth[skip:, 0]) ** 2))
        vel_err = np.sqrt(np.mean((est[skip:, 1] - truth[skip:, 1]) ** 2))
        naive_vel_err = np.sqrt(np.mean(
            ((np.diff(zs[:, 0]) / dt)[skip:] - truth[skip + 1:, 1]) ** 2))

        self.assertLess(pos_err, 0.6 * raw_err)
        self.assertLess(vel_err, 0.5 * naive_vel_err)

    @cpu_and_gpu
    def test_covariance_stays_symmetric_and_positive_definite(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model(1.0, 1e-3, 1.0)  # small Q stresses numerical stability
        kf = self._kf(target_device_idx, F, H, Q, R)
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        T = kf.seconds_to_t(1.0)
        rng = np.random.default_rng(3)
        for k in range(1, 2001):
            _, P = rig.step(k * T, {0: rng.normal(size=1)})
        np.testing.assert_allclose(P, P.T, rtol=0, atol=0)
        self.assertTrue(np.all(np.linalg.eigvalsh(P) > 0))

    @cpu_and_gpu
    def test_trigger_from_command_only_is_a_pure_prediction(self, target_device_idx, xp):
        """SPECULA only triggers an object when one of its inputs is refreshed. If
        that input is the command and no sensor is refreshed, the filter does a
        pure prediction step: x <- F x + B u, P <- F P F' + Q."""
        dt = 1.0
        F, G, H, Q, R = truck_model(dt)
        x0 = np.array([5.0, 1.0])
        u = np.array([0.5])
        kf = self._kf(target_device_idx, F, H, Q, R, command_matrix=G,
                      initial_state=x0, initial_covariance=np.zeros((2, 2)))
        rig = KalmanRig(kf, [1], xp, target_device_idx, command_size=1)
        T = kf.seconds_to_t(dt)

        x, P = rig.step(T, command=u)             # sensor never refreshed
        np.testing.assert_allclose(x, F @ x0 + G @ u)
        np.testing.assert_allclose(P, Q)          # F 0 F' + Q

    # ------------------------------------------------------------------
    # Multiple sensors, possibly at different rates
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_stale_sensor_is_ignored(self, target_device_idx, xp):
        """A sensor whose generation_time is not the current time must not update the filter."""
        dt, sigma_a, sigma_z = 1.0, 0.3, 1.0
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[0.25]])
        _, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 50, np.random.default_rng(2))

        kf2 = self._kf(target_device_idx, F, [H, Hv], Q, [R, Rv])
        kf1 = self._kf(target_device_idx, F, H, Q, R)
        rig2 = KalmanRig(kf2, [1, 1], xp, target_device_idx)
        rig1 = KalmanRig(kf1, [1], xp, target_device_idx)
        rig2.meas[1].value[:] = 1e6              # garbage, but never refreshed
        T = kf1.seconds_to_t(dt)

        for k, z in enumerate(zs, start=1):
            x2, P2 = rig2.step(k * T, {0: z})
            x1, P1 = rig1.step(k * T, {0: z})
            np.testing.assert_allclose(x2, x1, rtol=1e-12)
            np.testing.assert_allclose(P2, P1, rtol=1e-12)

    @cpu_and_gpu
    def test_simultaneous_sensors_equal_stacked_update(self, target_device_idx, xp):
        """Sequential updates with independent sensors equal one update with
        stacked H and block-diagonal R."""
        dt, sigma_a, sigma_z, sigma_v = 1.0, 0.3, 1.0, 0.5
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        truth, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 100, np.random.default_rng(4))
        vs = truth[:, 1] + np.random.default_rng(5).normal(0.0, sigma_v, size=len(truth))

        kf_multi = self._kf(target_device_idx, F, [H, Hv], Q, [R, Rv])
        kf_stack = self._kf(target_device_idx, F, np.vstack([H, Hv]), Q, block_diag(R, Rv))
        rig_multi = KalmanRig(kf_multi, [1, 1], xp, target_device_idx)
        rig_stack = KalmanRig(kf_stack, [2], xp, target_device_idx)
        T = kf_multi.seconds_to_t(dt)

        for k in range(1, 101):
            z, v = zs[k - 1], vs[k - 1]
            xm, Pm = rig_multi.step(k * T, {0: z, 1: [v]})
            xs, Ps = rig_stack.step(k * T, {0: [z[0], v]})
            np.testing.assert_allclose(xm, xs, rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(Pm, Ps, rtol=1e-9, atol=1e-11)

    @cpu_and_gpu
    def test_multirate_sensors(self, target_device_idx, xp):
        """Fast position sensor every step, slow velocity sensor every 5th step."""
        dt, sigma_a, sigma_z, sigma_v = 1.0, 0.3, 1.0, 0.5
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        truth, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 100, np.random.default_rng(6))
        vs = truth[:, 1] + np.random.default_rng(8).normal(0.0, sigma_v, size=len(truth))

        kf = self._kf(target_device_idx, F, [H, Hv], Q, [R, Rv])
        rig = KalmanRig(kf, [1, 1], xp, target_device_idx)
        ref = ReferenceKalmanFilter(F, Q, np.zeros(2), np.eye(2))
        T = kf.seconds_to_t(dt)

        for k in range(1, 101):
            fresh = {0: zs[k - 1]}
            ref.predict()
            ref.update(zs[k - 1], H, R)
            if k % 5 == 0:
                fresh[1] = [vs[k - 1]]
                ref.update(np.array([vs[k - 1]]), Hv, Rv)
            x, P = rig.step(k * T, fresh)
            np.testing.assert_allclose(x, ref.x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P, ref.P, rtol=1e-8, atol=1e-10)

    @cpu_and_gpu
    def test_time_step_catches_up_skipped_steps(self, target_device_idx, xp):
        """Two slow sensors (every 3rd and 4th step): the filter is only triggered
        when one of them is refreshed, but with time_step it still predicts over
        every elapsed step."""
        dt, sigma_a, sigma_z, sigma_v = 1.0, 0.3, 1.0, 0.5
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        truth, zs = simulate_truck(F, G, H, sigma_a, sigma_z, 120, np.random.default_rng(9))
        vs = truth[:, 1] + np.random.default_rng(10).normal(0.0, sigma_v, size=len(truth))

        kf = self._kf(target_device_idx, F, [H, Hv], Q, [R, Rv], time_step=dt)
        rig = KalmanRig(kf, [1, 1], xp, target_device_idx)
        ref = ReferenceKalmanFilter(F, Q, np.zeros(2), np.eye(2))
        T = kf.seconds_to_t(dt)

        n_compared = 0
        for k in range(1, 121):
            ref.predict()
            fresh = {}
            if k % 3 == 0:
                fresh[0] = zs[k - 1]
                ref.update(zs[k - 1], H, R)
            if k % 4 == 0:
                fresh[1] = [vs[k - 1]]
                ref.update(np.array([vs[k - 1]]), Hv, Rv)
            if not fresh:
                continue                          # nothing refreshed: not triggered
            x, P = rig.step(k * T, fresh)
            np.testing.assert_allclose(x, ref.x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P, ref.P, rtol=1e-8, atol=1e-10)
            n_compared += 1
        self.assertGreater(n_compared, 50)

    @cpu_and_gpu
    def test_without_time_step_one_prediction_per_trigger(self, target_device_idx, xp):
        """Default behaviour: A is applied once per trigger, whatever the elapsed time."""
        dt = 1.0
        F, G, H, Q, R = truck_model(dt)
        x0 = np.array([0.0, 1.0])
        kf = self._kf(target_device_idx, F, H, Q, R,
                      initial_state=x0, initial_covariance=np.zeros((2, 2)))
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        ref = ReferenceKalmanFilter(F, Q, x0, np.zeros((2, 2)))
        T = kf.seconds_to_t(dt)

        # first trigger after "5 steps", second one 15 steps later: one prediction each
        for k, z in ((5, 3.0), (20, 7.0)):
            x, P = rig.step(k * T, {0: [z]})
            ref.predict()
            ref.update(np.array([z]), H, R)
            np.testing.assert_allclose(x, ref.x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P, ref.P, rtol=1e-8, atol=1e-10)

    # ------------------------------------------------------------------
    # Optional command-to-state matrix
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_command_input_only_exists_with_command_matrix(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model()
        kf_no_cmd = self._kf(target_device_idx, F, H, Q, R)
        kf_cmd = self._kf(target_device_idx, F, H, Q, R, command_matrix=G)
        self.assertIn('in_measurements_list', kf_no_cmd.inputs)
        self.assertNotIn('in_command', kf_no_cmd.inputs)
        self.assertIn('in_command', kf_cmd.inputs)

    @cpu_and_gpu
    def test_command_matrix_matches_reference_and_improves_tracking(self, target_device_idx, xp):
        """Truck driven by a known commanded acceleration u_k (control-input model B)."""
        dt, sigma_a, sigma_z = 1.0, 0.1, 1.0
        F, G, H, Q, R = truck_model(dt, sigma_a, sigma_z)
        B = G.copy()                              # command = acceleration
        n = 200
        u = 2.0 * np.sin(0.15 * np.arange(1, n + 1))[:, None]
        truth, zs = simulate_truck(F, G, H, sigma_a, sigma_z, n,
                                   np.random.default_rng(11), B=B, u=u)

        kf_b = self._kf(target_device_idx, F, H, Q, R, command_matrix=B)
        kf_0 = self._kf(target_device_idx, F, H, Q, R)
        rig_b = KalmanRig(kf_b, [1], xp, target_device_idx, command_size=1)
        rig_0 = KalmanRig(kf_0, [1], xp, target_device_idx)
        ref = ReferenceKalmanFilter(F, Q, np.zeros(2), np.eye(2), B=B)
        T = kf_b.seconds_to_t(dt)

        est_b, est_0 = [], []
        for k in range(1, n + 1):
            x_b, P_b = rig_b.step(k * T, {0: zs[k - 1]}, command=u[k - 1])
            x_0, _ = rig_0.step(k * T, {0: zs[k - 1]})
            ref.predict(u[k - 1])
            ref.update(zs[k - 1], H, R)
            np.testing.assert_allclose(x_b, ref.x, rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P_b, ref.P, rtol=1e-8, atol=1e-10)
            est_b.append(x_b)
            est_0.append(x_0)

        err_b = np.sqrt(np.mean((np.array(est_b)[20:] - truth[20:]) ** 2))
        err_0 = np.sqrt(np.mean((np.array(est_0)[20:] - truth[20:]) ** 2))
        self.assertLess(err_b, 0.5 * err_0)

    # ------------------------------------------------------------------
    # Interface: outputs, reset, measurement noise formats
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_outputs_are_time_stamped(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model()
        kf = self._kf(target_device_idx, F, H, Q, R)
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        t = 3 * kf.seconds_to_t(1.0)
        rig.step(t, {0: [1.0]})
        self.assertEqual(kf.outputs['out_state'].generation_time, t)
        self.assertEqual(kf.outputs['out_covariance'].generation_time, t)
        self.assertEqual(cpuArray(kf.outputs['out_state'].value).shape, (2,))
        self.assertEqual(cpuArray(kf.outputs['out_covariance'].value).shape, (2, 2))

    @cpu_and_gpu
    def test_reset_states_replays_identically(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model(1.0, 0.3, 1.0)
        _, zs = simulate_truck(F, G, H, 0.3, 1.0, 10, np.random.default_rng(12))
        kf = self._kf(target_device_idx, F, H, Q, R, time_step=1.0)
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        T = kf.seconds_to_t(1.0)

        first = [rig.step(k * T, {0: z})[0] for k, z in enumerate(zs, start=1)]
        kf.reset_states()
        second = [rig.step(k * T, {0: z})[0] for k, z in enumerate(zs, start=1)]
        np.testing.assert_allclose(np.array(first), np.array(second), rtol=1e-12)

    @cpu_and_gpu
    def test_measurement_noise_formats_are_equivalent(self, target_device_idx, xp):
        """R can be given as a matrix, a vector of variances, or a scalar variance."""
        F, G, H, Q, _ = truck_model()
        H2 = np.eye(2)
        _, zs = simulate_truck(F, G, H, 0.3, 1.0, 20, np.random.default_rng(13))
        outs = []
        for R in (np.diag([0.5, 2.0]), np.array([0.5, 2.0])):
            kf = self._kf(target_device_idx, F, H2, Q, R)
            rig = KalmanRig(kf, [2], xp, target_device_idx)
            T = kf.seconds_to_t(1.0)
            outs.append([rig.step(k * T, {0: [z[0], 0.0]})[0] for k, z in enumerate(zs, start=1)])
        np.testing.assert_allclose(outs[0], outs[1], rtol=1e-12)

        kf_s = self._kf(target_device_idx, F, np.eye(2), Q, 0.5)
        kf_m = self._kf(target_device_idx, F, np.eye(2), Q, 0.5 * np.eye(2))
        for kf in (kf_s, kf_m):
            rig = KalmanRig(kf, [2], xp, target_device_idx)
        np.testing.assert_allclose(cpuArray(kf_s.R[0]), cpuArray(kf_m.R[0]))

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_init_validation(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model()
        make = lambda **kw: self._kf(target_device_idx,
                                     kw.get('F', F), kw.get('H', H), kw.get('Q', Q),
                                     kw.get('R', R), command_matrix=kw.get('B'))

        with self.assertRaises(ValueError):
            make(F=np.ones((2, 3)))                         # A not square
        with self.assertRaises(ValueError):
            make(Q=np.eye(3))                               # Q wrong size
        with self.assertRaises(ValueError):
            make(H=np.ones((1, 3)))                         # H wrong number of columns
        with self.assertRaises(ValueError):
            make(R=np.eye(2))                               # R does not match H rows
        with self.assertRaises(ValueError):
            make(H=[H, H], R=[R])                           # sensors count mismatch
        with self.assertRaises(ValueError):
            make(B=np.ones((3, 1)))                         # B wrong number of rows

    @cpu_and_gpu
    def test_runtime_validation(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model()

        # Wrong number of connected sensors
        kf = self._kf(target_device_idx, F, [H, H], Q, [R, R])
        rig = KalmanRig(kf, [1], xp, target_device_idx)
        with self.assertRaises(ValueError):
            rig.step(kf.seconds_to_t(1.0), {0: [0.0]})

        # Wrong measurement size
        kf = self._kf(target_device_idx, F, H, Q, R)
        rig = KalmanRig(kf, [2], xp, target_device_idx)
        with self.assertRaises(ValueError):
            rig.step(kf.seconds_to_t(1.0), {0: [0.0, 0.0]})



# ----------------------------------------------------------------------
# Sensor biases
#
# The article's two sources of information for the truck are a noisy but
# unbiased GPS (position) and a smooth dead-reckoning estimate (velocity from
# wheel revolutions) that "drifts over time as small errors accumulate". Here
# the GPS can also carry a constant offset and the dead reckoning a velocity
# bias:
#
#   z_pos = x + b_pos + v_pos,      z_vel = v + b_vel + v_vel
#
# With bias_process_noise_cov the filter estimates the biases jointly with
# the state [x, v, b...]. The initial position is known exactly (P0 = 0, as
# in the article), which, together with the redundancy of the two sensors,
# makes both biases observable.
# ----------------------------------------------------------------------

def joint_model(F, Q, H_list, bias_q, x0, P0, bias_x0=None, bias_p0=None):
    """Explicit joint [x; biases] model, written independently of KalmanFilter.

    bias_q[i] is None (sensor i unbiased) or the variance(s) of the bias random
    walk of sensor i (scalar or one per channel); bias_p0[i] the initial bias
    variance(s) (default 1) and bias_x0[i] the initial bias (default 0).
    """
    n = F.shape[0]
    sizes = [H.shape[0] for H in H_list]
    nb = sum(m for m, q in zip(sizes, bias_q) if q is not None)

    F_j = np.eye(n + nb)
    F_j[:n, :n] = F
    Q_j = np.zeros((n + nb, n + nb))
    Q_j[:n, :n] = Q
    P_j = np.zeros((n + nb, n + nb))
    P_j[:n, :n] = P0
    x_j = np.zeros(n + nb)
    x_j[:n] = x0

    H_j = []
    start = n
    for i, H in enumerate(H_list):
        m = sizes[i]
        Hi = np.zeros((m, n + nb))
        Hi[:, :n] = H
        if bias_q[i] is not None:
            sl = slice(start, start + m)
            Hi[:, sl] = np.eye(m)
            Q_j[sl, sl] = np.diag(np.broadcast_to(np.asarray(bias_q[i], float), (m,)))
            p0 = 1.0 if bias_p0 is None or bias_p0[i] is None else bias_p0[i]
            P_j[sl, sl] = np.diag(np.broadcast_to(np.asarray(p0, float), (m,)))
            if bias_x0 is not None and bias_x0[i] is not None:
                x_j[sl] = bias_x0[i]
            start += m
        H_j.append(Hi)
    return F_j, Q_j, H_j, x_j, P_j, nb


def bias_scenario(seed, n, sigma_a, sigma_z, sigma_v, bias_pos=0.0, bias_vel=0.0):
    """Truck truth plus position and velocity measurements, with additive biases."""
    F, G, H, Q, R = truck_model(1.0, sigma_a, sigma_z)
    rng = np.random.default_rng(seed)
    truth, _ = simulate_truck(F, G, H, sigma_a, sigma_z, n, rng)
    z_pos = truth[:, 0] + rng.normal(0.0, sigma_z, n) + bias_pos
    z_vel = truth[:, 1] + rng.normal(0.0, sigma_v, n) + bias_vel
    return truth, z_pos, z_vel


class TestKalmanFilterSensorBias(unittest.TestCase):

    def _run(self, target_device_idx, xp, sigmas, z_pos, z_vel, bias_q,
             pos_period=1, bias_x0=None, bias_p0=None):
        """Truck filter with position (every pos_period steps) and velocity
        (every step) sensors. Returns state, bias and bias-variance histories
        (the last two are empty if bias_q is None)."""
        sigma_a, sigma_z, sigma_v = sigmas
        F, G, H, Q, R = truck_model(1.0, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        kf = KalmanFilter(F, [H, Hv], Q, [R, Rv],
                          initial_covariance=np.zeros((2, 2)),
                          bias_process_noise_cov=bias_q, initial_bias=bias_x0,
                          initial_bias_covariance=bias_p0,
                          target_device_idx=target_device_idx, precision=0)
        rig = KalmanRig(kf, [1, 1], xp, target_device_idx)

        est, bias, bias_var = [], [], []
        for k in range(1, len(z_vel) + 1):
            fresh = {1: [z_vel[k - 1]]}
            if k % pos_period == 0:
                fresh[0] = [z_pos[k - 1]]
            x, _ = rig.step(k, fresh)
            est.append(x)
            if bias_q is not None:
                bias.append(cpuArray(kf.outputs['out_bias'].value).copy())
                bias_var.append(np.diag(cpuArray(kf.outputs['out_bias_covariance'].value)))
        return np.array(est), np.array(bias), np.array(bias_var)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_bias_outputs_exist_only_when_requested(self, target_device_idx, xp):
        F, G, H, Q, R = truck_model()
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[0.25]])
        kw = dict(target_device_idx=target_device_idx, precision=0)

        kf0 = KalmanFilter(F, [H, Hv], Q, [R, Rv], **kw)
        self.assertNotIn('out_bias', kf0.outputs)
        self.assertNotIn('out_bias_covariance', kf0.outputs)

        kf1 = KalmanFilter(F, [H, Hv], Q, [R, Rv], bias_process_noise_cov=[None, 1e-6], **kw)
        kf2 = KalmanFilter(F, [H, Hv], Q, [R, Rv], bias_process_noise_cov=[1e-4, 1e-6], **kw)
        for kf, nb in ((kf1, 1), (kf2, 2)):
            rig = KalmanRig(kf, [1, 1], xp, target_device_idx)
            rig.step(7, {0: [0.5], 1: [0.1]})
            self.assertEqual(cpuArray(kf.outputs['out_bias'].value).shape, (nb,))
            self.assertEqual(cpuArray(kf.outputs['out_bias_covariance'].value).shape, (nb, nb))
            # the state outputs keep the physical size
            self.assertEqual(cpuArray(kf.outputs['out_state'].value).shape, (2,))
            self.assertEqual(cpuArray(kf.outputs['out_covariance'].value).shape, (2, 2))
            self.assertEqual(kf.outputs['out_bias'].generation_time, 7)
            self.assertEqual(kf.outputs['out_bias_covariance'].generation_time, 7)

    # ------------------------------------------------------------------
    # Equivalence with an explicitly augmented filter
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_matches_explicit_joint_filter(self, target_device_idx, xp):
        """One or both sensors biased, position every 10th step and velocity
        every step: every output matches a hand-built [x; b] filter."""
        sigma_a, sigma_z, sigma_v, n = 0.1, 1.0, 0.5, 120
        F, G, H, Q, R = truck_model(1.0, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        _, z_pos, z_vel = bias_scenario(1, n, sigma_a, sigma_z, sigma_v, bias_pos=1.5, bias_vel=-0.3)
        P0 = np.diag([4.0, 1.0])

        cases = {'velocity sensor only': [None, 1e-6],
                 'position sensor only': [1e-4, None],
                 'both sensors': [1e-4, 1e-6]}
        for name, bias_q in cases.items():
            with self.subTest(name):
                bias_x0 = [None if q is None else np.array([0.1]) for q in bias_q]
                bias_p0 = [None if q is None else 2.0 for q in bias_q]
                kf = KalmanFilter(F, [H, Hv], Q, [R, Rv], initial_covariance=P0,
                                  bias_process_noise_cov=bias_q, initial_bias=bias_x0,
                                  initial_bias_covariance=bias_p0,
                                  target_device_idx=target_device_idx, precision=0)
                rig = KalmanRig(kf, [1, 1], xp, target_device_idx)

                F_j, Q_j, H_j, x_j, P_j, nb = joint_model(
                    F, Q, [H, Hv], bias_q, np.zeros(2), P0, bias_x0, bias_p0)
                ref = ReferenceKalmanFilter(F_j, Q_j, x_j, P_j)

                for k in range(1, n + 1):
                    fresh = {1: [z_vel[k - 1]]}
                    ref.predict()
                    if k % 10 == 0:
                        fresh[0] = [z_pos[k - 1]]
                        ref.update(np.array([z_pos[k - 1]]), H_j[0], R)
                    ref.update(np.array([z_vel[k - 1]]), H_j[1], Rv)

                    x, P = rig.step(k, fresh)
                    b = cpuArray(kf.outputs['out_bias'].value)
                    Pb = cpuArray(kf.outputs['out_bias_covariance'].value)
                    np.testing.assert_allclose(x, ref.x[:2], rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(P, ref.P[:2, :2], rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(b, ref.x[2:], rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(Pb, ref.P[2:, 2:], rtol=1e-8, atol=1e-10)

    @cpu_and_gpu
    def test_multichannel_sensor_bias(self, target_device_idx, xp):
        """A single 2-channel sensor [position, velocity]: one bias per channel,
        with per-channel noise and initial-uncertainty vectors."""
        sigma_a, sigma_z, sigma_v, n = 0.1, 1.0, 0.5, 80
        F, G, H, Q, R = truck_model(1.0, sigma_a, sigma_z)
        H2, R2 = np.eye(2), np.diag([sigma_z ** 2, sigma_v ** 2])
        _, z_pos, z_vel = bias_scenario(2, n, sigma_a, sigma_z, sigma_v, bias_pos=2.0, bias_vel=0.2)
        bias_q, bias_x0, bias_p0 = [np.array([1e-4, 1e-6])], [np.array([0.2, -0.1])], [np.array([2.0, 0.5])]

        kf = KalmanFilter(F, H2, Q, R2, bias_process_noise_cov=bias_q, initial_bias=bias_x0,
                          initial_bias_covariance=bias_p0,
                          target_device_idx=target_device_idx, precision=0)
        rig = KalmanRig(kf, [2], xp, target_device_idx)
        F_j, Q_j, H_j, x_j, P_j, nb = joint_model(F, Q, [H2], bias_q, np.zeros(2), np.eye(2),
                                                   bias_x0, bias_p0)
        self.assertEqual(nb, 2)
        ref = ReferenceKalmanFilter(F_j, Q_j, x_j, P_j)

        for k in range(n):
            z = np.array([z_pos[k], z_vel[k]])
            x, P = rig.step(k + 1, {0: z})
            ref.predict()
            ref.update(z, H_j[0], R2)
            np.testing.assert_allclose(x, ref.x[:2], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(cpuArray(kf.outputs['out_bias'].value), ref.x[2:],
                                       rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(cpuArray(kf.outputs['out_bias_covariance'].value),
                                       ref.P[2:, 2:], rtol=1e-8, atol=1e-10)

    @cpu_and_gpu
    def test_bias_with_command_and_skipped_steps(self, target_device_idx, xp):
        """Biases together with the command input and time_step catch-up
        (command held between triggers)."""
        sigma_a, sigma_z, sigma_v, n = 0.1, 1.0, 0.5, 120
        F, G, H, Q, R = truck_model(1.0, sigma_a, sigma_z)
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigma_v ** 2]])
        u = np.array([0.3])
        _, z_pos, z_vel = bias_scenario(3, n, sigma_a, sigma_z, sigma_v, bias_vel=0.2)
        bias_q = [None, 1e-6]

        kf = KalmanFilter(F, [H, Hv], Q, [R, Rv], command_matrix=G, time_step=1.0,
                          initial_covariance=np.zeros((2, 2)), bias_process_noise_cov=bias_q,
                          target_device_idx=target_device_idx, precision=0)
        rig = KalmanRig(kf, [1, 1], xp, target_device_idx, command_size=1)
        T = kf.seconds_to_t(1.0)

        F_j, Q_j, H_j, x_j, P_j, nb = joint_model(F, Q, [H, Hv], bias_q, np.zeros(2), np.zeros((2, 2)))
        ref = ReferenceKalmanFilter(F_j, Q_j, x_j, P_j, B=np.vstack([G, np.zeros((nb, 1))]))

        n_compared = 0
        for k in range(1, n + 1):
            ref.predict(u)
            fresh = {}
            if k % 4 == 0:
                fresh[0] = [z_pos[k - 1]]
                ref.update(np.array([z_pos[k - 1]]), H_j[0], R)
            if k % 6 == 0:
                fresh[1] = [z_vel[k - 1]]
                ref.update(np.array([z_vel[k - 1]]), H_j[1], Rv)
            if not fresh:
                continue                           # nothing refreshed: not triggered
            x, P = rig.step(k * T, fresh, command=u)
            np.testing.assert_allclose(x, ref.x[:2], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(P, ref.P[:2, :2], rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(cpuArray(kf.outputs['out_bias'].value), ref.x[2:],
                                       rtol=1e-8, atol=1e-10)
            n_compared += 1
        self.assertGreater(n_compared, 30)

    # ------------------------------------------------------------------
    # The biases are actually estimated and corrected
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_dead_reckoning_velocity_bias_is_corrected(self, target_device_idx, xp):
        """Slow noisy GPS (every 10th step) + fast precise dead reckoning with a
        velocity bias. Ignoring the bias makes the position drift away between
        GPS fixes; estimating it removes the drift."""
        sigmas, n, b_vel = (0.05, 3.0, 0.05), 600, 0.2
        truth, z_pos, z_vel = bias_scenario(1, n, *sigmas, bias_vel=b_vel)

        est, bias, bias_var = self._run(target_device_idx, xp, sigmas, z_pos, z_vel,
                                        bias_q=[None, 0.0], pos_period=10)
        est_ign, _, _ = self._run(target_device_idx, xp, sigmas, z_pos, z_vel,
                                  bias_q=None, pos_period=10)

        # the estimated bias agrees with the true one within its own uncertainty
        self.assertLess(abs(bias[-1, 0] - b_vel), 4.0 * np.sqrt(bias_var[-1, 0]))
        self.assertLess(abs(bias[-1, 0] - b_vel), 0.02)

        rms = lambda e: np.sqrt(np.mean((e[100:, 0] - truth[100:, 0]) ** 2))
        self.assertLess(rms(est), 0.2 * rms(est_ign))
        self.assertLess(rms(est), sigmas[1])          # better than the raw GPS noise

    @cpu_and_gpu
    def test_position_sensor_bias_is_corrected(self, target_device_idx, xp):
        """GPS with a constant offset, accurate initial position, unbiased velocity sensor."""
        sigmas, n, b_pos = (0.1, 1.0, 0.5), 600, 3.0
        truth, z_pos, z_vel = bias_scenario(1, n, *sigmas, bias_pos=b_pos)

        # the initial bias uncertainty (10 m rms) must cover the true 3 m offset
        est, bias, bias_var = self._run(target_device_idx, xp, sigmas, z_pos, z_vel,
                                        bias_q=[0.0, None], bias_p0=[100.0, None])
        est_ign, _, _ = self._run(target_device_idx, xp, sigmas, z_pos, z_vel, bias_q=None)

        self.assertLess(abs(bias[-1, 0] - b_pos), 4.0 * np.sqrt(bias_var[-1, 0]))
        rms = lambda e: np.sqrt(np.mean((e[100:, 0] - truth[100:, 0]) ** 2))
        self.assertLess(rms(est), 0.5 * rms(est_ign))

    @cpu_and_gpu
    def test_both_sensor_biases_are_corrected(self, target_device_idx, xp):
        sigmas, n, b_pos, b_vel = (0.1, 1.0, 0.5), 600, 3.0, 0.2
        truth, z_pos, z_vel = bias_scenario(1, n, *sigmas, bias_pos=b_pos, bias_vel=b_vel)

        est, bias, bias_var = self._run(target_device_idx, xp, sigmas, z_pos, z_vel,
                                        bias_q=[0.0, 0.0], bias_p0=[100.0, 1.0])
        est_ign, _, _ = self._run(target_device_idx, xp, sigmas, z_pos, z_vel, bias_q=None)

        for j, true_bias in enumerate((b_pos, b_vel)):
            self.assertLess(abs(bias[-1, j] - true_bias), 4.0 * np.sqrt(bias_var[-1, j]),
                            msg=f'bias {j} not within 4 sigma of the truth')
        rms = lambda e: np.sqrt(np.mean((e[100:, 0] - truth[100:, 0]) ** 2))
        self.assertLess(rms(est), 0.5 * rms(est_ign))

    @cpu_and_gpu
    def test_constant_bias_uncertainty_never_increases(self, target_device_idx, xp):
        """A constant bias (Qb = 0) is a fixed unknown: more data can only
        reduce its variance. With a random-walk term the variance levels off higher."""
        sigmas, n = (0.1, 1.0, 0.5), 200
        _, z_pos, z_vel = bias_scenario(4, n, *sigmas, bias_pos=1.0, bias_vel=0.1)

        _, _, var_const = self._run(target_device_idx, xp, sigmas, z_pos, z_vel, bias_q=[0.0, 0.0])
        _, _, var_walk = self._run(target_device_idx, xp, sigmas, z_pos, z_vel, bias_q=[1e-4, 1e-6])

        self.assertTrue(np.all(np.diff(var_const, axis=0) <= 1e-12))
        self.assertTrue(np.all(var_walk[-1] > var_const[-1]))

    @cpu_and_gpu
    def test_random_walk_term_tracks_drifting_bias(self, target_device_idx, xp):
        """Dead-reckoning bias that drifts linearly: a bias random-walk term lets the
        filter follow it, a constant-bias model does not."""
        sigmas, n = (0.05, 3.0, 0.05), 600
        truth, z_pos, z_vel = bias_scenario(1, n, *sigmas)
        true_bias = 0.2 + 0.0005 * np.arange(1, n + 1)
        z_vel = z_vel + true_bias

        results = {}
        for qb in (0.0, 1e-6):
            est, bias, _ = self._run(target_device_idx, xp, sigmas, z_pos, z_vel,
                                     bias_q=[None, qb], pos_period=10)
            bias_err = np.sqrt(np.mean((bias[100:, 0] - true_bias[100:]) ** 2))
            pos_err = np.sqrt(np.mean((est[100:, 0] - truth[100:, 0]) ** 2))
            results[qb] = (bias_err, pos_err)

        self.assertLess(results[1e-6][0], 0.8 * results[0.0][0])
        self.assertLess(results[1e-6][1], 0.7 * results[0.0][1])

    # ------------------------------------------------------------------
    # Housekeeping and validation
    # ------------------------------------------------------------------
    @cpu_and_gpu
    def test_reset_states_restores_initial_biases(self, target_device_idx, xp):
        sigmas, n = (0.1, 1.0, 0.5), 30
        _, z_pos, z_vel = bias_scenario(5, n, *sigmas, bias_pos=1.0, bias_vel=0.1)
        F, G, H, Q, R = truck_model(1.0, sigmas[0], sigmas[1])
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[sigmas[2] ** 2]])
        kf = KalmanFilter(F, [H, Hv], Q, [R, Rv], bias_process_noise_cov=[1e-4, 1e-6],
                          initial_bias=[[0.5], [0.05]],
                          target_device_idx=target_device_idx, precision=0)
        rig = KalmanRig(kf, [1, 1], xp, target_device_idx)

        def run(t0):
            out = []
            for k in range(n):
                rig.step(t0 + k + 1, {0: [z_pos[k]], 1: [z_vel[k]]})
                out.append(np.concatenate([cpuArray(kf.outputs['out_state'].value),
                                           cpuArray(kf.outputs['out_bias'].value)]))
            return np.array(out)

        first = run(0)
        kf.reset_states()
        np.testing.assert_allclose(cpuArray(kf._x[2:]), [0.5, 0.05])
        second = run(n)
        np.testing.assert_allclose(first, second, rtol=1e-12)

    def test_bias_argument_validation(self):
        F, G, H, Q, R = truck_model()
        Hv, Rv = np.array([[0.0, 1.0]]), np.array([[0.25]])

        def make(**kw):
            return KalmanFilter(F, [H, Hv], Q, [R, Rv], target_device_idx=-1, precision=0, **kw)

        make(bias_process_noise_cov=[1e-4, 1e-6])      # sanity: valid configuration
        with self.assertRaises(ValueError):            # one item for two sensors
            make(bias_process_noise_cov=[1e-6])
        with self.assertRaises(ValueError):            # wrong number of variances
            make(bias_process_noise_cov=[None, np.ones(3)])
        with self.assertRaises(ValueError):            # initial bias for an unbiased sensor
            make(bias_process_noise_cov=[1e-4, None], initial_bias=[None, [0.1]])
        with self.assertRaises(ValueError):            # initial bias covariance for an unbiased sensor
            make(bias_process_noise_cov=[1e-4, None], initial_bias_covariance=[None, 1.0])
        with self.assertRaises(ValueError):            # wrong initial bias size
            make(bias_process_noise_cov=[1e-4, None], initial_bias=[[0.0, 0.0], None])
        with self.assertRaises(ValueError):            # wrong initial bias covariance shape
            make(bias_process_noise_cov=[1e-4, None],
                 initial_bias_covariance=[np.ones((2, 2)), None])
        with self.assertRaises(ValueError):            # initial bias without bias estimation
            make(initial_bias=[[0.1], [0.1]])


if __name__ == '__main__':
    unittest.main()
