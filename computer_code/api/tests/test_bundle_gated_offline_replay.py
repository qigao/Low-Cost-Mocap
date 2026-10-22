"""Bundle-gated offline replay never runs on unverified input bytes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bundle_gated_offline_replay import main as combined_cli
from bundle_gated_offline_replay import run_bundle_gated_offline_replay


class BundleGatedOfflineReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "capture"
        self.root.mkdir()
        self.manifest = self.build_bundle()
        self.write_manifest()

    def build_bundle(self, *, include_replay=True, replay_raw=None):
        base_files = {
            "raw_camera_frames": ("frames/raw.bin", b"raw-frame-data"),
            "reference_trigger_log": ("reference/triggers.log", b"reference-trigger-data"),
            "independent_marker_labels": ("labels/labels.jsonl", b"independent-label-data"),
            "device_metadata": ("metadata/devices.json", b"device-metadata"),
            "led_controller_log": ("controller/output.log", b"controller-output-log"),
            "pattern_config": ("config/pattern.json", b"pattern-config-bytes"),
            "codebook": ("config/codebook.json", b"codebook-bytes"),
            "controller_config": ("config/controller.json", b"controller-config-bytes"),
            "calibration_config": ("config/calibration.json", b"calibration-config-bytes"),
        }
        replay_document = {
            "schema_version": 1,
            "trigger": {
                "expected_camera_ids": [2, 5],
                "max_exposure_skew_ns": 20,
                "recordings": [{
                    "reference_trigger_id": 10,
                    "camera_frames": [
                        {"camera_id": 2, "trigger_id": 10, "exposure_timestamp_ns": 100000, "clock_domain": "declared-reference"},
                        {"camera_id": 5, "trigger_id": 10, "exposure_timestamp_ns": 100006, "clock_domain": "declared-reference"},
                    ],
                }],
            },
        }
        if replay_raw is None:
            replay_raw = json.dumps(
                replay_document, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        if include_replay:
            base_files["offline_replay_input"] = ("replay/offline-input.json", replay_raw)

        artifacts = []
        hashes = {}
        for role, (relative, content) in base_files.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            hashes[role] = digest
            artifacts.append({
                "role": role, "path": relative,
                "size_bytes": len(content), "sha256": digest,
                "media_type": "application/json" if relative.endswith(".json") else "application/octet-stream",
            })
        return {
            "schema_version": 1,
            "bundle_id": "combined-replay-fixture",
            "capture": {
                "started_at_utc": "2026-10-22T01:00:00Z",
                "ended_at_utc": "2026-10-22T01:01:00Z",
                "camera_ids": [2, 5],
                "clock_domain": "declared-reference",
                "timestamp_semantics": "hardware_trigger",
                "timestamp_unit": "ns",
                "reference_source": "external-test-reference",
                "independent_reference_declared": True,
            },
            "software": {
                "python_version": "3.13.0",
                "host_os": "Linux",
                "host_os_version": "6.8",
                "application_commit_sha": "b" * 40,
                "pseyepy_version": "0.6.0",
                "led_controller_model": "fixture-controller",
                "led_controller_firmware_version": "1.2.3",
                "devices": [
                    {"camera_id": camera, "model": "PS3Eye", "firmware_version": "1.0",
                     "driver_version": "2.0", "frame_width": 640, "frame_height": 480,
                     "nominal_fps": 75.0}
                    for camera in (2, 5)
                ],
            },
            "revisions": {
                "pattern_sha256": hashlib.sha256(b"semantic-pattern-revision").hexdigest(),
                "pattern_config_artifact_sha256": hashes["pattern_config"],
                "codebook_sha256": hashes["codebook"],
                "controller_config_sha256": hashes["controller_config"],
                "calibration_sha256": hashes["calibration_config"],
            },
            "acceptance_limits": {
                "predeclared_at_utc": "2026-10-22T00:30:00Z",
                "max_exposure_skew_ns": 20,
                "max_reprojection_rms_px": 1.0,
                "max_marker_id_switches": 0,
                "min_labeled_frames": 100,
                "min_labeled_tracks": 4,
            },
            "artifacts": artifacts,
        }

    def write_manifest(self):
        (self.root / "capture-bundle.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_valid_bundle_gates_and_runs_existing_trigger_replay(self):
        report = run_bundle_gated_offline_replay(self.root)
        self.assertTrue(report["bundle_integrity_valid"], report["failures"])
        self.assertEqual(report["offline_replay_status"], "completed_software_replay")
        replay = report["offline_replay_report"]["trigger"]
        self.assertEqual(replay["total_batches"], 1)
        self.assertEqual(replay["admitted_batches"], 1)
        self.assertEqual(replay["rejected_batches"], 0)
        self.assertEqual(
            report["offline_replay_input_sha256"],
            self.manifest["artifacts"][-1]["sha256"],
        )
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])
        self.assertFalse(report["offline_replay_report"]["hardware_provenance_verified"])

    def test_tampered_artifact_skips_replay_entirely(self):
        artifact = next(x for x in self.manifest["artifacts"] if x["role"] == "raw_camera_frames")
        target = self.root / artifact["path"]
        target.write_bytes(b"tampered-frame-data")
        with patch("bundle_gated_offline_replay.evaluate_document", side_effect=AssertionError("must not replay")):
            report = run_bundle_gated_offline_replay(self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertEqual(report["offline_replay_status"], "skipped_bundle_integrity_invalid")
        self.assertIsNone(report["offline_replay_report"])

    def test_duplicate_key_and_unknown_replay_fields_fail_after_integrity(self):
        for raw, message in (
            (b'{"schema_version":1,"schema_version":1,"trigger":{}}', "duplicate JSON"),
            (b'{"schema_version":1,"trigger":{},"hardware_verified":true}', "unknown fields"),
        ):
            with self.subTest(message=message):
                self.manifest = self.build_bundle(replay_raw=raw)
                self.write_manifest()
                report = run_bundle_gated_offline_replay(self.root)
                self.assertTrue(report["bundle_integrity_valid"], report["failures"])
                self.assertEqual(report["offline_replay_status"], "invalid_replay_input")
                self.assertIn(message, report["failures"][0])
                self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])

    def test_oversized_replay_input_fails_before_json_parse(self):
        raw = b" " * 256
        self.manifest = self.build_bundle(replay_raw=raw)
        self.write_manifest()
        report = run_bundle_gated_offline_replay(
            self.root, max_replay_input_bytes=128
        )
        self.assertTrue(report["bundle_integrity_valid"], report["failures"])
        self.assertEqual(report["offline_replay_status"], "invalid_replay_input")
        self.assertIn("replay byte cap", report["failures"][0])

    def test_bundle_mutation_during_replay_invalidates_combined_report(self):
        raw_artifact = next(
            x for x in self.manifest["artifacts"]
            if x["role"] == "raw_camera_frames"
        )
        path = self.root / raw_artifact["path"]
        original = path.read_bytes()
        from offline_acceptance_cli import evaluate_document as real_evaluate

        def mutate_then_evaluate(document):
            path.write_bytes(b"mutated during replay")
            return real_evaluate(document)

        with patch("bundle_gated_offline_replay.evaluate_document", side_effect=mutate_then_evaluate):
            report = run_bundle_gated_offline_replay(self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertEqual(report["offline_replay_status"], "bundle_changed_during_replay")
        self.assertTrue(any("changed while replay" in x for x in report["failures"]))
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])
        path.write_bytes(original)

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink test requires OS support")
    def test_root_symlink_is_rejected_before_replay(self):
        alias = self.root.with_name(self.root.name + "-alias")
        alias.symlink_to(self.root, target_is_directory=True)
        with patch(
            "bundle_gated_offline_replay.evaluate_document",
            side_effect=AssertionError("replay must not run"),
        ):
            report = run_bundle_gated_offline_replay(alias)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertEqual(report["offline_replay_status"], "invalid_bundle_manifest")
        self.assertIsNone(report["offline_replay_report"])
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])

    @unittest.skipUnless(os.name == "posix", "directory symlink replacement test requires POSIX")
    def test_bundle_root_replacement_during_replay_invalidates_integrity(self):
        moved_root = self.root.with_name(self.root.name + "-original")
        from offline_acceptance_cli import evaluate_document as real_evaluate

        def replace_root_then_evaluate(document):
            self.root.rename(moved_root)
            self.root.symlink_to(moved_root, target_is_directory=True)
            return real_evaluate(document)

        with patch(
            "bundle_gated_offline_replay.evaluate_document",
            side_effect=replace_root_then_evaluate,
        ):
            report = run_bundle_gated_offline_replay(self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertFalse(report["bundle_report"]["bundle_integrity_valid"])
        self.assertEqual(report["offline_replay_status"], "bundle_changed_during_replay")
        self.assertTrue(any("final bundle revalidation failed" in x for x in report["failures"]))
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])

    def test_missing_replay_role_reports_no_replay_without_claiming_success(self):
        self.manifest = self.build_bundle(include_replay=False)
        self.write_manifest()
        report = run_bundle_gated_offline_replay(self.root)
        self.assertTrue(report["bundle_integrity_valid"], report["failures"])
        self.assertEqual(report["offline_replay_status"], "missing_or_duplicate_replay_input")
        self.assertIsNone(report["offline_replay_report"])

    def test_cli_emits_combined_report_and_nonzero_on_bundle_failure(self):
        valid_out = Path(self.temp.name) / "valid-report.json"
        with patch("builtins.print", side_effect=lambda *args, **kwargs: valid_out.write_text(args[0])):
            code = combined_cli([str(self.root)])
        self.assertEqual(code, 0)
        parsed = json.loads(valid_out.read_text())
        self.assertEqual(parsed["offline_replay_status"], "completed_software_replay")
        self.assertFalse(parsed["declarations_not_verified"]["hardware_provenance_verified"])

        artifact = next(x for x in self.manifest["artifacts"] if x["role"] == "raw_camera_frames")
        (self.root / artifact["path"]).write_bytes(b"changed")
        bad_out = Path(self.temp.name) / "bad-report.json"
        with patch("builtins.print", side_effect=lambda *args, **kwargs: bad_out.write_text(args[0])):
            code = combined_cli([str(self.root)])
        self.assertEqual(code, 1)
        bad = json.loads(bad_out.read_text())
        self.assertEqual(bad["offline_replay_status"], "skipped_bundle_integrity_invalid")
        self.assertFalse(bad["declarations_not_verified"]["hardware_provenance_verified"])


if __name__ == "__main__":
    unittest.main()
