import math
import unittest

from fusion_replay import (
    OpticalOnlyOrientationEstimate,
    ReferenceOrientation,
    compare_optical_only_vs_fused,
)
from optical_imu_fusion import OpticalIMUFusionState

PATTERN = "b" * 64
OPTICAL_CAL = "c" * 64
IMU_CAL = "a" * 64
MAPPING = "d" * 64
REFERENCE = "e" * 64


def qz(angle):
    return (0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2))


def optical(ts, q, status="TRACKING"):
    return OpticalOnlyOrientationEstimate(
        body_id="board-A", timestamp_ns=ts, clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=q if status == "TRACKING" else None,
        status=status, pattern_sha256=PATTERN, optical_calibration_sha256=OPTICAL_CAL,
    )


def fused(ts, q, status="TRACKING", optical_ts=None, frame_id=None):
    if status == "LOST":
        qv, variance, source = None, None, "LOST"
    else:
        qv, variance = q, 0.01
        source = "OPTICAL_CORRECTED" if status == "TRACKING" else "IMU_PROPAGATED"
    if optical_ts is None and status != "LOST":
        optical_ts, frame_id = ts, ts
    return OpticalIMUFusionState(
        body_id="board-A", imu_id="imu-1", timestamp_ns=ts, clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=qv, attitude_variance_rad2=variance,
        status=status, source=source, last_optical_timestamp_ns=optical_ts,
        last_optical_frame_id=frame_id,
        optical_age_ns=None if optical_ts is None else ts - optical_ts,
        clock_mapping_sha256=MAPPING, imu_calibration_sha256=IMU_CAL,
        optical_pattern_sha256=PATTERN, optical_calibration_sha256=OPTICAL_CAL,
    )


def refs():
    return [
        ReferenceOrientation(ts, "imu-clock", (0, 0, 0, 1), REFERENCE)
        for ts in (100, 200, 300)
    ]


class OpticalVsFusedReportTests(unittest.TestCase):
    def test_compares_both_series_to_same_reference_and_counts_invalid_optical(self):
        optical_series = [
            optical(100, (0, 0, 0, 1)),
            optical(200, None, "INVALID"),
            optical(300, qz(0.2)),
        ]
        fused_series = [
            fused(100, (0, 0, 0, 1)),
            fused(200, qz(0.05), "DEGRADED", 100, 100),
            fused(300, qz(0.15)),
        ]
        report = compare_optical_only_vs_fused(
            optical_series, fused_series, refs(), clock_domain="imu-clock",
            max_match_delta_ns=0,
        )
        self.assertEqual(report.reference_sha256, REFERENCE)
        self.assertEqual(report.optical_only.matched_count, 2)
        self.assertEqual(report.optical_only.invalid_count, 1)
        self.assertEqual(report.optical_only.unmatched_sample_count, 1)
        self.assertEqual(report.fused.matched_count, 3)
        self.assertEqual(report.fused.degraded_count, 1)
        self.assertEqual(report.paired_reference_count, 2)
        self.assertAlmostEqual(report.fused.rmse_rad, math.sqrt((0.0**2+0.05**2+0.15**2)/3), places=6)
        self.assertEqual(len(report.report_sha256), 64)
        self.assertFalse(report.hardware_ground_truth_verified)

    def test_report_is_deterministic_and_digest_changes_with_inputs(self):
        optical_series = [optical(100, (0,0,0,1)), optical(200, qz(0.1))]
        fused_series = [fused(100, (0,0,0,1)), fused(200, qz(0.05))]
        baseline = compare_optical_only_vs_fused(optical_series, fused_series, refs()[:2],
            clock_domain="imu-clock", max_match_delta_ns=0)
        repeat = compare_optical_only_vs_fused(optical_series, fused_series, refs()[:2],
            clock_domain="imu-clock", max_match_delta_ns=0)
        changed = compare_optical_only_vs_fused(
            optical_series, [fused(100, (0,0,0,1)), fused(200, qz(0.08))], refs()[:2],
            clock_domain="imu-clock", max_match_delta_ns=0,
        )
        self.assertEqual(baseline.report_sha256, repeat.report_sha256)
        self.assertEqual(baseline.input_sha256, repeat.input_sha256)
        self.assertNotEqual(baseline.report_sha256, changed.report_sha256)
        self.assertNotEqual(baseline.input_sha256, changed.input_sha256)

    def test_sign_equivalent_quaternions_have_zero_error(self):
        report = compare_optical_only_vs_fused(
            [optical(100, (0,0,0,-1))],
            [fused(100, (0,0,0,1))],
            refs()[:1], clock_domain="imu-clock", max_match_delta_ns=0,
        )
        self.assertAlmostEqual(report.optical_only.rmse_rad, 0.0, places=7)
        self.assertAlmostEqual(report.fused.rmse_rad, 0.0, places=7)
        self.assertAlmostEqual(report.paired_rmse_improvement_rad, 0.0, places=7)

    def test_rejects_mixed_identity_revision_clock_and_duplicate_timestamps(self):
        valid_optical = [optical(100, (0,0,0,1)), optical(200, (0,0,0,1))]
        valid_fused = [fused(100, (0,0,0,1)), fused(200, (0,0,0,1))]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            compare_optical_only_vs_fused(
                [valid_optical[0], optical(100, (0,0,0,1))], valid_fused, refs()[:2],
                clock_domain="imu-clock",
            )
        with self.assertRaisesRegex(ValueError, "mixed body/pattern/calibration"):
            compare_optical_only_vs_fused(
                [valid_optical[0], OpticalOnlyOrientationEstimate(
                    "board-B", 200, "imu-clock", (0,0,0,1), "TRACKING", PATTERN, OPTICAL_CAL
                )], valid_fused, refs()[:2], clock_domain="imu-clock",
            )
        mixed_clock = [ReferenceOrientation(100, "other-clock", (0,0,0,1), REFERENCE)]
        with self.assertRaisesRegex(ValueError, "mixed clock domain"):
            compare_optical_only_vs_fused(valid_optical[:1], valid_fused[:1], mixed_clock, clock_domain="imu-clock")

    def test_rejects_revision_mismatch_no_matches_and_input_capacity(self):
        bad_optical = [
            optical(100, (0,0,0,1)),
            OpticalOnlyOrientationEstimate("board-A", 200, "imu-clock", (0,0,0,1),
                "TRACKING", "f"*64, OPTICAL_CAL),
        ]
        with self.assertRaisesRegex(ValueError, "mixed body/pattern/calibration"):
            compare_optical_only_vs_fused(bad_optical, [fused(100,(0,0,0,1)),fused(200,(0,0,0,1))],
                refs()[:2], clock_domain="imu-clock")
        far_reference = [ReferenceOrientation(1_000, "imu-clock", (0,0,0,1), REFERENCE)]
        with self.assertRaisesRegex(ValueError, "both optical-only and fused"):
            compare_optical_only_vs_fused([optical(100,(0,0,0,1))],
                [fused(100,(0,0,0,1))], far_reference, clock_domain="imu-clock", max_match_delta_ns=0)
        with self.assertRaisesRegex(ValueError, "input byte capacity"):
            compare_optical_only_vs_fused([optical(100,(0,0,0,1))],
                [fused(100,(0,0,0,1))], refs()[:1], clock_domain="imu-clock", max_report_bytes=1)


if __name__ == "__main__":
    unittest.main()
