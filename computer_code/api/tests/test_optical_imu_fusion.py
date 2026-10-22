import math
import unittest

from imu_contract import ClockAnchor, fit_clock_mapping
from imu_propagation import BoundedIMUAttitudePropagator
from optical_imu_fusion import OpticalAttitudeMeasurement, OpticalIMUAttitudeFusion

IMU_SHA = "a" * 64
PATTERN_SHA = "b" * 64
OPTICAL_CAL_SHA = "c" * 64


def mapping():
    return fit_clock_mapping(
        [
            ClockAnchor(1_000_000_000, 1_000_001_000, event_id="sync-1"),
            ClockAnchor(2_000_000_000, 2_000_001_000, event_id="sync-2"),
            ClockAnchor(3_000_000_000, 3_000_001_000, event_id="sync-3"),
        ],
        source_clock_domain="camera-clock",
        target_clock_domain="imu-clock",
        max_drift_ppm=10,
        max_residual_ns=10,
        max_extrapolation_ns=20_000_000,
        min_anchor_span_ns=100_000_000,
    )


def setup_pipeline(noise=(0.01, 0.01, 0.01), dropout=30_000_000):
    clock = mapping()
    propagator = BoundedIMUAttitudePropagator(
        imu_id="imu-1", clock_domain="imu-clock", calibration_revision="cal-v1",
        calibration_sha256=IMU_SHA, max_sample_gap_ns=20_000_000,
        max_step_ns=5_000_000, max_substeps=8,
    )
    fusion = OpticalIMUAttitudeFusion(
        body_id="board-A", imu_id="imu-1", imu_clock_domain="imu-clock",
        imu_calibration_revision="cal-v1", imu_calibration_sha256=IMU_SHA,
        optical_pattern_sha256=PATTERN_SHA, optical_calibration_sha256=OPTICAL_CAL_SHA,
        clock_mapping=clock, gyro_noise_density_rad_s_sqrt_hz=noise,
        optical_variance_rad2=0.0025, max_dropout_ns=dropout,
    )
    states = []
    for seq in range(6):
        states.append(propagator.advance(type_sample(seq, 1_000_001_000 + seq*10_000_000)))
    return fusion, states


def type_sample(seq, ts, gyro=(0.0, 0.0, 1.0)):
    from imu_calibration import CorrectedIMUVectors
    return CorrectedIMUVectors(
        imu_id="imu-1", sequence_number=seq, sensor_timestamp_ns=ts,
        clock_domain="imu-clock", calibration_revision="cal-v1",
        calibration_sha256=IMU_SHA, gyro_body_rad_s=gyro,
        specific_force_body_m_s2=(0.0, 0.0, 9.81),
    )


def optical(source_ts, frame_id, q=(0.0, 0.0, 0.0, 1.0), quality="MEASURED"):
    return OpticalAttitudeMeasurement(
        body_id="board-A", timestamp_ns=source_ts, clock_domain="camera-clock",
        quaternion_world_from_body_xyzw=q, frame_id=frame_id,
        pattern_sha256=PATTERN_SHA, calibration_sha256=OPTICAL_CAL_SHA,
        quality=quality,
    )


class OpticalIMUAttitudeFusionTests(unittest.TestCase):
    def test_seed_requires_qualified_mapping_and_emits_tracking_state(self):
        fusion, states = setup_pipeline()
        self.assertEqual(fusion.ingest_imu(states[0]).status, "LOST")
        state = fusion.ingest_imu(states[1])
        state = fusion.ingest_optical(optical(1_010_000_000, 10))
        self.assertEqual(state.status, "TRACKING")
        self.assertEqual(state.last_optical_frame_id, 10)
        self.assertIsNotNone(state.quaternion_world_from_body_xyzw)
        self.assertIsNone(state.global_position_m)
        self.assertFalse(state.global_position_observable)

    def test_imu_dropout_propagates_orientation_and_increases_variance(self):
        fusion, states = setup_pipeline(noise=(0.1, 0.1, 0.1))
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        seeded = fusion.ingest_optical(optical(1_010_000_000, 10))
        propagated = fusion.ingest_imu(states[2])
        self.assertEqual(propagated.status, "DEGRADED")
        self.assertGreater(propagated.attitude_variance_rad2, seeded.attitude_variance_rad2)
        self.assertAlmostEqual(math.sqrt(sum(x*x for x in propagated.quaternion_world_from_body_xyzw)), 1.0, places=12)

    def test_optical_update_corrects_toward_measurement_and_reduces_variance(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        first = fusion.ingest_optical(optical(1_010_000_000, 10))
        fusion.ingest_imu(states[2])
        before = fusion.state
        measured_angle = 0.04
        corrected = fusion.ingest_optical(optical(1_020_000_000, 11, (0.0, 0.0, math.sin(measured_angle/2), math.cos(measured_angle/2))))
        self.assertLess(abs(corrected.quaternion_world_from_body_xyzw[2] - math.sin(measured_angle/2)),
                        abs(before.quaternion_world_from_body_xyzw[2] - math.sin(measured_angle/2)))
        self.assertLess(corrected.attitude_variance_rad2, before.attitude_variance_rad2)
        self.assertEqual(corrected.last_optical_frame_id, 11)

    def test_rejects_wrong_identity_stale_order_and_large_innovation_without_mutation(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        accepted = fusion.ingest_optical(optical(1_010_000_000, 10))
        with self.assertRaisesRegex(ValueError, "out of order"):
            fusion.ingest_optical(optical(1_010_000_000, 10))
        with self.assertRaisesRegex(ValueError, "body mismatch"):
            fusion.ingest_optical(OpticalAttitudeMeasurement("other", 1_020_000_000, "camera-clock",
                (0,0,0,1), 11, PATTERN_SHA, OPTICAL_CAL_SHA))
        fusion.ingest_imu(states[2])
        before = fusion.state
        with self.assertRaisesRegex(ValueError, "innovation exceeds"):
            fusion.ingest_optical(optical(1_020_000_000, 11, (0,0,1,0)))
        self.assertEqual(fusion.state, before)
        self.assertEqual(accepted.last_optical_frame_id, 10)

    def test_dropout_expiry_hides_orientation_and_requires_explicit_reseed(self):
        fusion, states = setup_pipeline(dropout=15_000_000)
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        fusion.ingest_optical(optical(1_010_000_000, 10))
        lost = fusion.ingest_imu(states[4])
        self.assertEqual(lost.status, "LOST")
        self.assertIsNone(lost.quaternion_world_from_body_xyzw)
        with self.assertRaisesRegex(ValueError, "explicit reseed"):
            fusion.ingest_optical(optical(1_040_000_000, 11))
        recovered = fusion.ingest_optical(optical(1_040_000_000, 11), allow_reseed_after_lost=True)
        self.assertEqual(recovered.status, "TRACKING")
        self.assertIsNotNone(recovered.quaternion_world_from_body_xyzw)

    def test_rejects_mapping_domain_and_stale_pose_quality(self):
        fusion, states = setup_pipeline()
        fusion.ingest_imu(states[0])
        fusion.ingest_imu(states[1])
        bad = OpticalAttitudeMeasurement("board-A", 1_010_000_000, "other-clock", (0,0,0,1), 10, PATTERN_SHA, OPTICAL_CAL_SHA)
        with self.assertRaisesRegex(ValueError, "clock mapping source"):
            fusion.ingest_optical(bad)
        with self.assertRaisesRegex(ValueError, "must be current MEASURED"):
            optical(1_010_000_000, 10, quality="INVALID")


if __name__ == "__main__":
    unittest.main()
