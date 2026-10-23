import unittest
from dataclasses import replace

from localization_accuracy_report import (
    PositionReference,
    ProcessingLatencySample,
    evaluate_localization_accuracy,
)
from test_outdoor_localization_replay import build_replay


DATASET_SHA = "f" * 64


def sample_report_inputs():
    owner, _calibration, _camera_clock, _rtk_clock, _rtk_cal = build_replay()
    base = owner.state
    first = replace(
        base,
        timestamp_ns=1_000,
        position_world_from_body_m=(0.0, 0.0, 0.0),
        velocity_world_from_body_m_s=(0.0, 0.0, 0.0),
        last_position_update_timestamp_ns=1_000,
        position_age_ns=0,
        status="TRACKING",
        source="POSITION_SEEDED",
    )
    propagated = replace(
        first,
        timestamp_ns=2_000,
        position_world_from_body_m=(1.0, 0.0, 0.0),
        last_position_update_timestamp_ns=1_000,
        position_age_ns=1_000,
        status="DEGRADED",
        source="IMU_PROPAGATED",
    )
    lost = replace(
        propagated,
        timestamp_ns=3_000,
        last_position_update_timestamp_ns=1_000,
        position_age_ns=2_000,
        position_world_from_body_m=None,
        velocity_world_from_body_m_s=None,
        covariance_p_v_bias=None,
        status="LOST",
        source="LOST",
    )
    reacquired = replace(
        propagated,
        timestamp_ns=4_000,
        position_world_from_body_m=(3.1, 0.0, 0.0),
        last_position_update_timestamp_ns=4_000,
        position_age_ns=0,
        status="TRACKING",
        source="POSITION_CORRECTED",
    )
    states = (first, propagated, lost, reacquired)
    references = tuple(
        PositionReference(
            timestamp_ns=timestamp,
            clock_domain="imu-clock",
            position_world_from_body_m=position,
            dataset_sha256=DATASET_SHA,
        )
        for timestamp, position in (
            (1_000, (0.1, 0.0, 0.0)),
            (2_000, (1.1, 0.0, 0.0)),
            (3_000, (2.1, 0.0, 0.0)),
            (4_000, (3.0, 0.0, 0.0)),
        )
    )
    latency = tuple(
        ProcessingLatencySample(
            event_id=f"frame-{index}",
            clock_domain="host-monotonic",
            ingress_monotonic_ns=index * 100,
            egress_monotonic_ns=index * 100 + duration,
        )
        for index, duration in enumerate((10, 20, 30, 40), start=1)
    )
    return states, references, latency


class LocalizationAccuracyReportTests(unittest.TestCase):
    def test_report_counts_lost_states_and_reports_drift_reacquisition_latency(self):
        states, references, latency = sample_report_inputs()
        report = evaluate_localization_accuracy(
            states, references, max_match_delta_ns=0, latency_samples=latency
        )
        self.assertEqual(report["qualification"], "reference_metrics_only")
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["matched_count"], 3)
        self.assertEqual(report["unmatched_state_count"], 1)
        self.assertEqual(report["unmatched_reference_count"], 1)
        self.assertEqual(report["status_counts"], {
            "TRACKING": 2, "DEGRADED": 1, "LOST": 1,
        })
        self.assertAlmostEqual(report["position_error_m"]["rmse"], 0.1)
        self.assertAlmostEqual(report["imu_propagated_position_error_m"]["max"], 0.1)
        self.assertAlmostEqual(report["reacquisition_position_error_m"]["max"], 0.1)
        self.assertEqual(report["latency"]["p50_ns"], 20)
        self.assertEqual(report["latency"]["p95_ns"], 40)
        self.assertEqual(report["latency"]["p99_ns"], 40)
        self.assertFalse(report["reference_independence_verified"])
        self.assertFalse(report["hardware_ground_truth_verified"])
        self.assertFalse(report["hardware_provenance_verified"])

    def test_report_digest_is_stable_and_changes_when_reference_changes(self):
        states, references, latency = sample_report_inputs()
        first = evaluate_localization_accuracy(
            states, references, max_match_delta_ns=0, latency_samples=latency
        )
        second = evaluate_localization_accuracy(
            states, references, max_match_delta_ns=0, latency_samples=latency
        )
        self.assertEqual(first, second)
        changed = list(references)
        changed[0] = replace(
            changed[0], position_world_from_body_m=(0.2, 0.0, 0.0)
        )
        third = evaluate_localization_accuracy(
            states, tuple(changed), max_match_delta_ns=0, latency_samples=latency
        )
        self.assertNotEqual(first["input_sha256"], third["input_sha256"])
        self.assertNotEqual(first["report_sha256"], third["report_sha256"])

    def test_rejects_mixed_reference_datasets_clocks_and_reversed_latency(self):
        states, references, latency = sample_report_inputs()
        mixed_digest = list(references)
        mixed_digest[1] = replace(mixed_digest[1], dataset_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "mix dataset digests"):
            evaluate_localization_accuracy(
                states, tuple(mixed_digest), max_match_delta_ns=0
            )
        mixed_clock = list(references)
        mixed_clock[1] = replace(mixed_clock[1], clock_domain="other-clock")
        with self.assertRaisesRegex(ValueError, "clock domain"):
            evaluate_localization_accuracy(
                states, tuple(mixed_clock), max_match_delta_ns=0
            )
        with self.assertRaisesRegex(ValueError, "must not precede"):
            ProcessingLatencySample("bad", "host-monotonic", 20, 10)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            evaluate_localization_accuracy(
                states, tuple(reversed(references)), max_match_delta_ns=0
            )

    def test_rejects_mixed_state_calibration_and_unbounded_sample_count(self):
        states, references, _latency = sample_report_inputs()
        changed_state = replace(states[-1], imu_calibration_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "mix body, frame, clock or calibration"):
            evaluate_localization_accuracy(
                states[:-1] + (changed_state,), references, max_match_delta_ns=0
            )
        with self.assertRaisesRegex(ValueError, "capacity"):
            evaluate_localization_accuracy(
                states, references, max_match_delta_ns=0, max_samples=3
            )


if __name__ == "__main__":
    unittest.main()
