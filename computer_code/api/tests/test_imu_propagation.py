import math
import unittest

from imu_calibration import CorrectedIMUVectors
from imu_propagation import BoundedIMUAttitudePropagator


_DIGEST = "a" * 64


def vector_sample(
    sequence,
    timestamp_ns,
    gyro=(0.0, 0.0, 0.0),
    *,
    imu_id="imu-1",
    clock_domain="imu-clock",
    revision="cal-v1",
    digest=_DIGEST,
):
    return CorrectedIMUVectors(
        imu_id=imu_id,
        sequence_number=sequence,
        sensor_timestamp_ns=timestamp_ns,
        clock_domain=clock_domain,
        calibration_revision=revision,
        calibration_sha256=digest,
        gyro_body_rad_s=gyro,
        specific_force_body_m_s2=(0.0, 0.0, 9.81),
    )


def propagator(**overrides):
    options = {
        "imu_id": "imu-1",
        "clock_domain": "imu-clock",
        "calibration_revision": "cal-v1",
        "calibration_sha256": _DIGEST,
        "max_sample_gap_ns": 20_000_000,
        "max_step_ns": 5_000_000,
        "max_substeps": 64,
    }
    options.update(overrides)
    return BoundedIMUAttitudePropagator(**options)


class BoundedIMUAttitudePropagationTests(unittest.TestCase):
    def test_first_sample_only_establishes_baseline(self):
        tracker = propagator()
        state = tracker.advance(vector_sample(0, 1_000_000_000, (0.0, 0.0, 4.0)))
        self.assertEqual(state.status, "INITIALIZED")
        self.assertEqual(state.orientation_world_from_body_xyzw, (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(state.integrated_interval_ns, 0)
        self.assertEqual(state.substeps, 0)
        self.assertIsNone(state.global_position_m)
        self.assertIsNone(state.global_velocity_m_s)
        self.assertFalse(state.global_position_observable)

    def test_constant_rate_about_each_axis_matches_analytic_quaternion(self):
        cases = [
            ((1.0, 0.0, 0.0), (math.sin(0.5), 0.0, 0.0, math.cos(0.5))),
            ((0.0, -2.0, 0.0), (0.0, -math.sin(1.0), 0.0, math.cos(1.0))),
            ((0.0, 0.0, 0.5), (0.0, 0.0, math.sin(0.25), math.cos(0.25))),
        ]
        for rate, expected in cases:
            with self.subTest(rate=rate):
                tracker = propagator(max_sample_gap_ns=20_000_000, max_step_ns=5_000_000, max_substeps=8)
                tracker.advance(vector_sample(0, 0, rate))
                state = None
                for seq in range(1, 101):
                    state = tracker.advance(vector_sample(seq, seq * 10_000_000, rate))
                self.assertEqual(state.status, "IMU_PROPAGATED")
                for actual, target in zip(state.orientation_world_from_body_xyzw, expected):
                    self.assertAlmostEqual(actual, target, places=6)
                self.assertEqual(state.sensor_timestamp_ns, 1_000_000_000)

    def test_trapezoidal_rate_and_bounded_substeps_are_reported(self):
        tracker = propagator()
        tracker.advance(vector_sample(0, 100, (0.0, 0.0, 0.0)))
        state = tracker.advance(vector_sample(1, 20_000_100, (0.0, 0.0, 2.0)))
        self.assertEqual(state.integrated_interval_ns, 20_000_000)
        self.assertEqual(state.substeps, 4)
        self.assertAlmostEqual(state.orientation_world_from_body_xyzw[2], math.sin(0.01), places=7)
        self.assertAlmostEqual(state.orientation_world_from_body_xyzw[3], math.cos(0.01), places=7)

    def test_long_run_quaternion_remains_normalized(self):
        tracker = propagator(max_sample_gap_ns=10_000_000, max_step_ns=2_000_000, max_substeps=8)
        tracker.advance(vector_sample(0, 0, (0.2, -0.3, 0.4)))
        state = None
        for seq in range(1, 1001):
            state = tracker.advance(vector_sample(seq, seq * 1_000_000, (0.2, -0.3, 0.4)))
        norm = math.sqrt(sum(x * x for x in state.orientation_world_from_body_xyzw))
        self.assertAlmostEqual(norm, 1.0, places=12)

    def test_rejects_duplicate_reordered_and_regressed_samples_without_state_change(self):
        tracker = propagator()
        tracker.advance(vector_sample(4, 1_000_000))
        accepted = tracker.advance(vector_sample(5, 2_000_000, (0.0, 0.0, 1.0)))
        for invalid, message in (
            (vector_sample(5, 3_000_000), "duplicate or out of order"),
            (vector_sample(4, 4_000_000), "duplicate or out of order"),
            (vector_sample(6, 2_000_000), "duplicate or regressed"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    tracker.advance(invalid)
                self.assertEqual(tracker.state, accepted)

    def test_rejects_excessive_gap_and_substep_capacity_without_advancing(self):
        tracker = propagator()
        tracker.advance(vector_sample(0, 0))
        accepted = tracker.advance(vector_sample(1, 10_000_000, (0.0, 0.0, 1.0)))
        with self.assertRaisesRegex(ValueError, "sample gap exceeds"):
            tracker.advance(vector_sample(2, 40_000_001, (0.0, 0.0, 1.0)))
        self.assertEqual(tracker.state, accepted)

        limited = propagator(max_sample_gap_ns=20_000_000, max_step_ns=1_000_000, max_substeps=2)
        baseline = limited.advance(vector_sample(0, 0))
        with self.assertRaisesRegex(ValueError, "substep capacity"):
            limited.advance(vector_sample(1, 3_000_000, (0.0, 0.0, 1.0)))
        self.assertEqual(limited.state, baseline)

    def test_rejects_identity_clock_and_calibration_changes_transactionally(self):
        tracker = propagator()
        tracker.advance(vector_sample(0, 1_000_000))
        accepted = tracker.advance(vector_sample(1, 2_000_000))
        invalid_samples = [
            (vector_sample(2, 3_000_000, imu_id="imu-2"), "device ID mismatch"),
            (vector_sample(2, 3_000_000, clock_domain="host-clock"), "clock-domain mismatch"),
            (vector_sample(2, 3_000_000, revision="cal-v2"), "revision mismatch"),
            (vector_sample(2, 3_000_000, digest="b" * 64), "digest mismatch"),
        ]
        for item, message in invalid_samples:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    tracker.advance(item)
                self.assertEqual(tracker.state, accepted)

    def test_invalid_public_quaternion_and_bounds_are_rejected(self):
        for quaternion in ((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, float("nan"), 1.0), (0.0, 0.0, 1.0)):
            with self.subTest(quaternion=quaternion):
                with self.assertRaises(ValueError):
                    propagator(initial_orientation_world_from_body_xyzw=quaternion)
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            propagator(max_sample_gap_ns=1, max_step_ns=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            propagator(max_substeps=0)

    def test_reset_clears_time_baseline_and_restores_initial_attitude(self):
        initial = (0.0, 0.0, math.sin(0.25), math.cos(0.25))
        tracker = propagator(initial_orientation_world_from_body_xyzw=initial)
        tracker.advance(vector_sample(0, 10))
        tracker.advance(vector_sample(1, 1_000_010, (0.0, 0.0, 1.0)))
        tracker.reset()
        self.assertIsNone(tracker.state)
        state = tracker.advance(vector_sample(0, 50, (0.0, 0.0, 3.0)))
        self.assertEqual(state.status, "INITIALIZED")
        self.assertAlmostEqual(state.orientation_world_from_body_xyzw[2], math.sin(0.25))
        self.assertAlmostEqual(state.orientation_world_from_body_xyzw[3], math.cos(0.25))


if __name__ == "__main__":
    unittest.main()
