import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path

from fusion_ab_report_cli import (
    MAX_INPUT_BYTES,
    evaluate_bytes,
    evaluate_document,
    main,
)


PATTERN = "b" * 64
OPTICAL_CAL = "c" * 64
IMU_CAL = "a" * 64
MAPPING = "d" * 64
REFERENCE = "e" * 64


def document():
    optical = {
        "body_id": "board-A", "timestamp_ns": 100, "clock_domain": "imu-clock",
        "quaternion_world_from_body_xyzw": [0, 0, 0, 1], "status": "TRACKING",
        "pattern_sha256": PATTERN, "optical_calibration_sha256": OPTICAL_CAL,
    }
    fused = {
        "body_id": "board-A", "imu_id": "imu-1", "timestamp_ns": 100,
        "clock_domain": "imu-clock", "quaternion_world_from_body_xyzw": [0, 0, 0, 1],
        "attitude_variance_rad2": 0.01, "status": "TRACKING", "source": "OPTICAL_CORRECTED",
        "last_optical_timestamp_ns": 100, "last_optical_frame_id": 1, "optical_age_ns": 0,
        "clock_mapping_sha256": MAPPING, "imu_calibration_sha256": IMU_CAL,
        "optical_pattern_sha256": PATTERN, "optical_calibration_sha256": OPTICAL_CAL,
        "global_position_m": None, "global_velocity_m_s": None,
        "global_position_observable": False,
    }
    reference = {
        "timestamp_ns": 100, "clock_domain": "imu-clock",
        "quaternion_world_from_body_xyzw": [0, 0, 0, 1],
        "reference_sha256": REFERENCE,
    }
    return {
        "schema_version": 1, "clock_domain": "imu-clock", "max_match_delta_ns": 0,
        "optical_only": [optical], "fused": [fused], "references": [reference],
    }


class FusionABReportCLITests(unittest.TestCase):
    def test_valid_report_has_raw_and_semantic_digests_and_no_hardware_claim(self):
        raw = json.dumps(document(), sort_keys=True, separators=(",", ":")).encode()
        report = evaluate_bytes(raw)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["report_kind"], "optical_vs_fused_attitude")
        self.assertEqual(report["optical_only"]["matched_count"], 1)
        self.assertEqual(report["fused"]["matched_count"], 1)
        self.assertEqual(report["paired_reference_count"], 1)
        self.assertFalse(report["hardware_ground_truth_verified"])
        self.assertEqual(len(report["recording_sha256"]), 64)
        self.assertEqual(len(report["report_sha256"]), 64)

    def test_report_is_deterministic_for_identical_input_bytes(self):
        raw = json.dumps(document(), sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(evaluate_bytes(raw), evaluate_bytes(raw))

    def test_rejects_duplicate_keys_nonfinite_and_input_capacity(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            evaluate_bytes(b'{"schema_version":1,"schema_version":1}')
        raw = json.dumps(document()).replace('"max_match_delta_ns": 0', '"max_match_delta_ns": NaN').encode()
        with self.assertRaisesRegex(ValueError, "nonfinite JSON constant"):
            evaluate_bytes(raw)
        with self.assertRaisesRegex(ValueError, "input byte capacity"):
            from fusion_ab_report_cli import parse_input_bytes
            parse_input_bytes(b" " * (MAX_INPUT_BYTES + 1))

    def test_rejects_unknown_missing_and_bool_schema_fields(self):
        item = document()
        item["extra"] = 1
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            evaluate_document(item)
        item = document()
        item["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "integer 1"):
            evaluate_document(item)
        item = document()
        del item["fused"][0]["clock_mapping_sha256"]
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            evaluate_document(item)

    def test_rejects_mixed_clock_revision_and_no_match(self):
        item = document()
        item["references"][0]["clock_domain"] = "camera-clock"
        with self.assertRaisesRegex(ValueError, "mixed clock domain"):
            evaluate_document(item)
        item = document()
        item["optical_only"][0]["pattern_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "pattern/calibration revisions differ"):
            evaluate_document(item)
        item = document()
        item["references"][0]["timestamp_ns"] = 10_000
        with self.assertRaisesRegex(ValueError, "both optical-only and fused"):
            evaluate_document(item)

    def test_cli_failure_writes_no_partial_json_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text('{"schema_version":true}', encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main([str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("rejected", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
