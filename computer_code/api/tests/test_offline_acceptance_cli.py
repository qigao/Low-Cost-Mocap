"""Run the offline replay JSON contract without any camera/flight hardware."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from offline_acceptance_cli import evaluate_document, main


class OfflineRecordingCliTests(unittest.TestCase):
    def sample(self):
        patterns = {
            "10": [0, 0, 0, 0, 1, 1, 1, 1],
            "11": [0, 0, 1, 1, 0, 0, 1, 1],
            "12": [0, 1, 0, 1, 0, 1, 0, 1],
            "13": [0, 1, 1, 0, 1, 0, 0, 1],
        }
        return {
            "schema_version": 1,
            "trigger": {
                "expected_camera_ids": [2, 5],
                "max_exposure_skew_ns": 20,
                "recordings": [
                    {
                        "reference_trigger_id": 300,
                        "camera_frames": [
                            {
                                "camera_id": 2, "trigger_id": 300,
                                "exposure_timestamp_ns": 1_000_000,
                                "clock_domain": "source-declared-clock",
                            },
                            {
                                "camera_id": 5, "trigger_id": 300,
                                "exposure_timestamp_ns": 1_000_005,
                                "clock_domain": "source-declared-clock",
                            },
                        ],
                    },
                    {
                        "reference_trigger_id": 301,
                        "camera_frames": [
                            {
                                "camera_id": 2, "trigger_id": 301,
                                "exposure_timestamp_ns": 1_010_000,
                                "clock_domain": "source-declared-clock",
                            },
                            {
                                "camera_id": 5, "trigger_id": 302,
                                "exposure_timestamp_ns": 1_010_003,
                                "clock_domain": "source-declared-clock",
                            },
                        ],
                    },
                ],
            },
            "temporal": {
                "codebook": patterns,
                "max_bit_errors": 1,
                "min_valid_samples": 6,
                "trials": [{
                    "reference_marker_id": 13,
                    "trigger_epoch": 500,
                    "samples": [
                        {"trigger_id": 500 + i, "illuminated": bit}
                        for i, bit in enumerate(patterns["13"])
                    ],
                }],
            },
        }

    def test_json_report_counts_failed_frames_without_claiming_hardware_success(self):
        summary = evaluate_document(self.sample())
        self.assertEqual(summary["qualification"], "software_replay_only")
        self.assertFalse(summary["hardware_provenance_verified"])
        self.assertEqual(summary["trigger"]["admitted_batches"], 1)
        self.assertEqual(summary["trigger"]["rejected_batches"], 1)
        self.assertEqual(summary["temporal"]["correct_accepts"], 1)
        self.assertEqual(summary["temporal"]["wrong_accepts"], 0)
        self.assertGreater(
            summary["temporal"]["wrong_accept_one_sided_upper_95"], 0.0
        )
        self.assertTrue(summary["temporal"]["bound_assumes_independent_trials"])
        self.assertFalse(summary["temporal"]["trial_independence_verified"])
        self.assertEqual(len(summary["temporal"]["statistical_limitations"]), 2)
        self.assertIn(
            "cluster-aware analysis",
            summary["temporal"]["statistical_limitations"][1],
        )

    def test_cli_reports_input_sha256_from_actual_bytes(self):
        payload = json.dumps(self.sample()).encode("utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic-capture.json"
            path.write_bytes(payload)
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                return_code = main([str(path)])
        self.assertEqual(return_code, 0)
        parsed = json.loads(stream.getvalue())
        self.assertEqual(parsed["input_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse(parsed["hardware_provenance_verified"])

    def test_missing_schema_or_fake_codebook_marker_fails_closed(self):
        doc = self.sample()
        doc["schema_version"] = True
        with self.assertRaisesRegex(ValueError, "schema_version"):
            evaluate_document(doc)
        doc = self.sample()
        doc["temporal"]["codebook"]["01"] = doc["temporal"]["codebook"].pop("10")
        with self.assertRaisesRegex(ValueError, "canonical"):
            evaluate_document(doc)

    def test_invalid_trigger_record_fields_fail_closed(self):
        doc = self.sample()
        doc["trigger"]["recordings"][0]["camera_frames"][0]["received_at"] = 50
        with self.assertRaisesRegex(ValueError, "incomplete or unknown"):
            evaluate_document(doc)

    def test_unknown_fields_fail_closed_at_every_document_section(self):
        doc = self.sample()
        doc["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            evaluate_document(doc)

        doc = self.sample()
        doc["trigger"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            evaluate_document(doc)

        doc = self.sample()
        doc["trigger"]["recordings"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            evaluate_document(doc)

        doc = self.sample()
        doc["temporal"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            evaluate_document(doc)

        doc = self.sample()
        doc["temporal"]["trials"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown unexpected"):
            evaluate_document(doc)

    def test_cli_rejects_duplicate_json_keys_without_partial_stdout(self):
        payload = b'{"schema_version":1,"schema_version":1,"temporal":{"codebook":{},"trials":[]}}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_bytes(payload)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as raised:
                    main([str(path)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("duplicate JSON object key", stderr.getvalue())

    def test_cli_rejects_oversized_input_before_parsing_or_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.json"
            path.write_bytes(b" " * 17)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("offline_acceptance_cli.MAX_RECORDING_BYTES", 16):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main([str(path)])
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("byte limit", stderr.getvalue())

    def test_fabricated_hardware_label_is_never_certification(self):
        doc = self.sample()
        for row in doc["trigger"]["recordings"]:
            for frame in row["camera_frames"]:
                frame["clock_domain"] = "HW_VERIFIED_TRUST_ME"
        summary = evaluate_document(doc)
        self.assertFalse(summary["hardware_provenance_verified"])
        self.assertFalse(summary["trigger"]["hardware_provenance_verified"])


if __name__ == "__main__":
    unittest.main()
