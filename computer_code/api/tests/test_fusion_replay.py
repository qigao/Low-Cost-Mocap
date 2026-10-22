import math
import unittest
from dataclasses import replace

from imu_calibration import CorrectedIMUVectors
from imu_contract import ClockAnchor, fit_clock_mapping
from imu_propagation import BoundedIMUAttitudePropagator
from optical_imu_fusion import OpticalAttitudeMeasurement, OpticalIMUFusionState
from fusion_replay import (
    FusionReplayConfig,
    FusionReplayEvent,
    ReferenceOrientation,
    evaluate_orientation_error,
    replay_fusion_events,
)

IMU_SHA = "a" * 64
PATTERN_SHA = "b" * 64
OPTICAL_SHA = "c" * 64
REF_SHA = "d" * 64


def config(noise=(0.01, 0.01, 0.01)):
    clock = fit_clock_mapping(
        [
            ClockAnchor(1_000_000_000, 1_000_001_000, event_id="sync-a"),
            ClockAnchor(2_000_000_000, 2_000_001_000, event_id="sync-b"),
            ClockAnchor(3_000_000_000, 3_000_001_000, event_id="sync-c"),
        ],
        source_clock_domain="camera-clock", target_clock_domain="imu-clock",
        max_drift_ppm=10, max_residual_ns=10,
        max_extrapolation_ns=20_000_000, min_anchor_span_ns=100_000_000,
    )
    return FusionReplayConfig(
        body_id="board-A", imu_id="imu-1", imu_clock_domain="imu-clock",
        imu_calibration_revision="cal-v1", imu_calibration_sha256=IMU_SHA,
        optical_pattern_sha256=PATTERN_SHA, optical_calibration_sha256=OPTICAL_SHA,
        clock_mapping=clock, gyro_noise_density_rad_s_sqrt_hz=noise,
        optical_variance_rad2=0.0025, max_dropout_ns=50_000_000,
    )


def events():
    cfg = config()
    prop = BoundedIMUAttitudePropagator(
        imu_id="imu-1", clock_domain="imu-clock", calibration_revision="cal-v1",
        calibration_sha256=IMU_SHA, max_sample_gap_ns=20_000_000,
        max_step_ns=5_000_000, max_substeps=8,
    )
    samples = []
    for seq, timestamp in enumerate((1_001_001_000, 1_011_001_000, 1_021_001_000)):
        raw = CorrectedIMUVectors(
            imu_id="imu-1", sequence_number=seq, sensor_timestamp_ns=timestamp,
            clock_domain="imu-clock", calibration_revision="cal-v1",
            calibration_sha256=IMU_SHA, gyro_body_rad_s=(0.0, 0.0, 0.5),
            specific_force_body_m_s2=(0.0, 0.0, 9.81),
        )
        samples.append(prop.advance(raw))
    optical = OpticalAttitudeMeasurement(
        body_id="board-A", timestamp_ns=1_010_000_000,
        clock_domain="camera-clock", quaternion_world_from_body_xyzw=(0,0,0,1),
        frame_id=10, pattern_sha256=PATTERN_SHA, calibration_sha256=OPTICAL_SHA,
    )
    return cfg, [
        FusionReplayEvent(0, "imu", samples[0]),
        FusionReplayEvent(1, "imu", samples[1]),
        FusionReplayEvent(2, "optical", optical),
        FusionReplayEvent(3, "imu", samples[2]),
    ]


def fusion_state(status, timestamp, q=None, optical_ts=None, frame_id=None, variance=None, source=None):
    if status == "LOST":
        q = None
        variance = None
        source = "LOST"
    elif source is None:
        source = "OPTICAL_CORRECTED" if status == "TRACKING" else "IMU_PROPAGATED"
    return OpticalIMUFusionState(
        body_id="board-A", imu_id="imu-1", timestamp_ns=timestamp, clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=q, attitude_variance_rad2=variance, status=status, source=source,
        last_optical_timestamp_ns=optical_ts, last_optical_frame_id=frame_id,
        optical_age_ns=None if optical_ts is None else timestamp-optical_ts,
        clock_mapping_sha256="e"*64, imu_calibration_sha256=IMU_SHA,
        optical_pattern_sha256=PATTERN_SHA, optical_calibration_sha256=OPTICAL_SHA,
    )


class FusionReplayTests(unittest.TestCase):
    def test_replay_is_deterministic_and_emits_stable_digests(self):
        cfg, trace = events()
        first = replay_fusion_events(cfg, trace)
        second = replay_fusion_events(cfg, trace)
        self.assertEqual(first.input_sha256, second.input_sha256)
        self.assertEqual(first.output_sha256, second.output_sha256)
        self.assertEqual(first.config_sha256, second.config_sha256)
        self.assertEqual(first.states, second.states)
        self.assertEqual(first.event_count, 4)
        self.assertEqual(first.output_count, 4)
        self.assertFalse(first.hardware_provenance_verified)

    def test_input_config_and_event_changes_change_digest(self):
        cfg, trace = events()
        baseline = replay_fusion_events(cfg, trace)
        changed_config = replay_fusion_events(replace(cfg, gyro_noise_density_rad_s_sqrt_hz=(0.02,0.02,0.02)), trace)
        changed_event = list(trace)
        changed_event[2] = FusionReplayEvent(2, "optical", OpticalAttitudeMeasurement(
            "board-A", 1_010_000_000, "camera-clock",
            (0,0,math.sin(0.02),math.cos(0.02)), 10, PATTERN_SHA, OPTICAL_SHA,
        ))
        changed = replay_fusion_events(cfg, changed_event)
        self.assertNotEqual(baseline.config_sha256, changed_config.config_sha256)
        self.assertNotEqual(baseline.input_sha256, changed.input_sha256)

    def test_rejects_noncontiguous_order_empty_trace_and_capacity_overflow(self):
        cfg, trace = events()
        with self.assertRaisesRegex(ValueError, "contiguous"):
            replay_fusion_events(cfg, [trace[0], replace(trace[1], event_index=3)])
        with self.assertRaisesRegex(ValueError, "at least one event"):
            replay_fusion_events(cfg, [])
        with self.assertRaisesRegex(ValueError, "event capacity"):
            replay_fusion_events(cfg, trace, max_events=3)
        with self.assertRaisesRegex(ValueError, "input byte capacity"):
            replay_fusion_events(cfg, trace, max_input_bytes=10)
        with self.assertRaisesRegex(ValueError, "max_input_bytes"):
            replay_fusion_events(cfg, trace, max_input_bytes=True)

    def test_reference_metrics_use_geodesic_error_and_quaternion_sign_equivalence(self):
        identity = fusion_state("TRACKING", 100, (0,0,0,1), 100, 1, 0.01)
        same_negated = ReferenceOrientation(100, "imu-clock", (0,0,0,-1), REF_SHA)
        report = evaluate_orientation_error([identity], [same_negated], clock_domain="imu-clock", max_match_delta_ns=0)
        self.assertEqual(report.matched_count, 1)
        self.assertAlmostEqual(report.rmse_rad, 0.0, places=7)
        self.assertFalse(report.hardware_ground_truth_verified)

        angle = 0.2
        ref = ReferenceOrientation(100, "imu-clock", (0,0,math.sin(angle/2),math.cos(angle/2)), REF_SHA)
        report = evaluate_orientation_error([identity], [ref], clock_domain="imu-clock", max_match_delta_ns=0)
        self.assertAlmostEqual(report.rmse_rad, angle, places=6)
        self.assertAlmostEqual(report.max_rad, angle, places=6)
        self.assertAlmostEqual(report.rmse_deg, math.degrees(angle), places=5)
        self.assertAlmostEqual(report.max_deg, math.degrees(angle), places=5)

    def test_reference_report_counts_degraded_and_lost_states(self):
        tracking = fusion_state("TRACKING", 100, (0,0,0,1), 100, 1, 0.01)
        degraded = fusion_state("DEGRADED", 200, (0,0,0,1), 100, 1, 0.02)
        lost = fusion_state("LOST", 300, optical_ts=100, frame_id=1)
        refs = [
            ReferenceOrientation(100, "imu-clock", (0,0,0,1), REF_SHA),
            ReferenceOrientation(200, "imu-clock", (0,0,0,1), REF_SHA),
        ]
        report = evaluate_orientation_error([tracking,degraded,lost], refs, clock_domain="imu-clock", max_match_delta_ns=0)
        self.assertEqual((report.tracking_count, report.degraded_count, report.lost_count), (1,1,1))
        self.assertEqual(report.matched_count, 2)
        self.assertEqual(report.unmatched_state_count, 1)
        self.assertEqual(report.unmatched_reference_count, 0)

    def test_rejects_duplicate_or_mixed_reference_clock_and_zero_matches(self):
        state = fusion_state("TRACKING", 100, (0,0,0,1), 100, 1, 0.01)
        duplicate = [
            ReferenceOrientation(100, "imu-clock", (0,0,0,1), REF_SHA),
            ReferenceOrientation(100, "imu-clock", (0,0,0,1), REF_SHA),
        ]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            evaluate_orientation_error([state], duplicate, clock_domain="imu-clock")
        mixed = [ReferenceOrientation(100, "camera-clock", (0,0,0,1), REF_SHA)]
        with self.assertRaisesRegex(ValueError, "mixed clock domain"):
            evaluate_orientation_error([state], mixed, clock_domain="imu-clock")
        far = [ReferenceOrientation(1_000, "imu-clock", (0,0,0,1), REF_SHA)]
        with self.assertRaisesRegex(ValueError, "no publishable"):
            evaluate_orientation_error([state], far, clock_domain="imu-clock", max_match_delta_ns=0)

    def test_rejects_mixed_fusion_identity_or_calibration_revisions(self):
        first = fusion_state("TRACKING", 100, (0,0,0,1), 100, 1, 0.01)
        second = OpticalIMUFusionState(
            body_id="another-board", imu_id="imu-1", timestamp_ns=200, clock_domain="imu-clock",
            quaternion_world_from_body_xyzw=(0,0,0,1), attitude_variance_rad2=0.01,
            status="TRACKING", source="OPTICAL_CORRECTED", last_optical_timestamp_ns=200,
            last_optical_frame_id=2, optical_age_ns=0, clock_mapping_sha256="e"*64,
            imu_calibration_sha256=IMU_SHA, optical_pattern_sha256=PATTERN_SHA,
            optical_calibration_sha256=OPTICAL_SHA,
        )
        refs = [
            ReferenceOrientation(100, "imu-clock", (0,0,0,1), REF_SHA),
            ReferenceOrientation(200, "imu-clock", (0,0,0,1), REF_SHA),
        ]
        with self.assertRaisesRegex(ValueError, "mixed body/IMU/calibration"):
            evaluate_orientation_error([first, second], refs, clock_domain="imu-clock", max_match_delta_ns=0)

    def test_rejects_invalid_reference_quaternion_and_changed_dataset_digest(self):
        with self.assertRaisesRegex(ValueError, "nonzero norm"):
            ReferenceOrientation(100, "imu-clock", (0,0,0,0), REF_SHA)
        refs = [
            ReferenceOrientation(100, "imu-clock", (0,0,0,1), REF_SHA),
            ReferenceOrientation(200, "imu-clock", (0,0,0,1), "f"*64),
        ]
        states = [
            fusion_state("TRACKING", 100, (0,0,0,1), 100, 1, 0.01),
            fusion_state("TRACKING", 200, (0,0,0,1), 200, 2, 0.01),
        ]
        with self.assertRaisesRegex(ValueError, "digest changed"):
            evaluate_orientation_error(states, refs, clock_domain="imu-clock", max_match_delta_ns=0)


if __name__ == "__main__":
    unittest.main()
