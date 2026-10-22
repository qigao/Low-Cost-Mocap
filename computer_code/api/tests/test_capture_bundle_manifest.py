"""Strict real-rig capture bundle manifest and path-integrity conformance."""
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from io import StringIO
import unittest
from unittest.mock import patch

from capture_bundle_cli import main as cli_main
from capture_bundle_manifest import (
    MANIFEST_NAME,
    load_manifest_bytes,
    validate_capture_bundle,
    validate_capture_bundle_directory,
)


ROLES = (
    "raw_camera_frames",
    "reference_trigger_log",
    "independent_marker_labels",
    "device_metadata",
    "led_controller_log",
    "pattern_config",
    "codebook",
    "controller_config",
    "calibration_config",
)


class CaptureBundleManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        self.manifest = self.make_bundle()

    def make_bundle(self):
        artifacts = []
        hashes = {}
        contents = {
            "raw_camera_frames": b"synthetic-raw-frame-payload",
            "reference_trigger_log": b"independent-reference-trigger-log",
            "independent_marker_labels": b"independently-reviewed-marker-labels",
            "device_metadata": b"camera-driver-version-record",
            "led_controller_log": b"controller-timecode-output-log",
            "pattern_config": b"exact-pattern-registry-config",
            "codebook": b"immutable-codebook-config",
            "controller_config": b"immutable-led-controller-config",
            "calibration_config": b"camera-intrinsics-extrinsics-and-pattern-config",
        }
        paths = {
            "raw_camera_frames": "frames/camera-bundle.bin",
            "reference_trigger_log": "reference/triggers.jsonl",
            "independent_marker_labels": "labels/marker-labels.jsonl",
            "device_metadata": "metadata/devices.json",
            "led_controller_log": "controller/output.log",
            "pattern_config": "config/pattern-registry.json",
            "codebook": "config/codebook.json",
            "controller_config": "config/controller.json",
            "calibration_config": "config/calibration.json",
        }
        for role in ROLES:
            path = self.root / paths[role]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents[role])
            digest = hashlib.sha256(contents[role]).hexdigest()
            hashes[role] = digest
            artifacts.append({
                "role": role,
                "path": paths[role],
                "size_bytes": len(contents[role]),
                "sha256": digest,
                "media_type": "application/octet-stream",
            })
        return {
            "schema_version": 1,
            "bundle_id": "rig-session-20261022-001",
            "capture": {
                "started_at_utc": "2026-10-22T01:00:00Z",
                "ended_at_utc": "2026-10-22T01:10:00Z",
                "camera_ids": [2, 5, 9],
                "clock_domain": "declared-independent-reference-clock",
                "timestamp_semantics": "sensor_exposure",
                "timestamp_unit": "ns",
                "reference_source": "external-logic-analyzer",
                "independent_reference_declared": True,
            },
            "software": {
                "python_version": "3.13.0",
                "host_os": "Linux",
                "host_os_version": "6.8.0",
                "application_commit_sha": "a" * 40,
                "pseyepy_version": "0.6.0",
                "led_controller_model": "test-controller-v2",
                "led_controller_firmware_version": "2.4.1",
                "devices": [
                    {
                        "camera_id": camera,
                        "model": "PS3Eye",
                        "firmware_version": "1.0.0",
                        "driver_version": "2.1.3",
                        "frame_width": 640,
                        "frame_height": 480,
                        "nominal_fps": 75.0,
                    }
                    for camera in (2, 5, 9)
                ],
            },
            "revisions": {
                "pattern_sha256": hashlib.sha256(b"canonical-pattern-registry").hexdigest(),
                "pattern_config_artifact_sha256": hashes["pattern_config"],
                "codebook_sha256": hashes["codebook"],
                "controller_config_sha256": hashes["controller_config"],
                "calibration_sha256": hashes["calibration_config"],
            },
            "acceptance_limits": {
                "predeclared_at_utc": "2026-10-22T00:30:00Z",
                "max_exposure_skew_ns": 100_000,
                "max_reprojection_rms_px": 1.25,
                "max_marker_id_switches": 0,
                "min_labeled_frames": 10_000,
                "min_labeled_tracks": 12,
            },
            "artifacts": artifacts,
        }

    def test_valid_bundle_reports_all_artifacts_and_explicit_noncertification(self):
        report = validate_capture_bundle(self.manifest, self.root)
        self.assertTrue(report["bundle_integrity_valid"], report["failures"])
        self.assertEqual(report["qualification"], "capture_bundle_integrity_only")
        self.assertEqual(report["artifact_count"], len(ROLES))
        self.assertEqual(report["required_roles_missing"], [])
        self.assertTrue(all(x["status"] == "ok" for x in report["artifact_checks"]))
        self.assertEqual(
            report["declared_capture"]["timestamp_semantics"], "sensor_exposure"
        )
        self.assertTrue(report["declared_capture"]["acceptance_predeclared_before_capture"])
        for key in (
            "hardware_provenance_verified", "device_authenticity_verified",
            "independent_reference_verified", "timestamp_semantics_verified",
            "physical_metric_scale_verified",
        ):
            self.assertFalse(report["declarations_not_verified"][key])
        self.assertEqual(len(report["manifest_sha256"]), 64)
        self.assertGreater(report["total_actual_bytes_hashed"], 0)

    def test_canonical_manifest_sha_does_not_depend_on_key_insertion_order(self):
        first = validate_capture_bundle(self.manifest, self.root)
        reordered = dict(reversed(list(self.manifest.items())))
        second = validate_capture_bundle(reordered, self.root)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

    def test_changed_file_missing_file_and_wrong_declared_size_are_reported(self):
        changed = deepcopy(self.manifest)
        path = self.root / changed["artifacts"][0]["path"]
        content = path.read_bytes()
        path.write_bytes(b"X" + content[1:])
        report = validate_capture_bundle(changed, self.root)
        entry = next(x for x in report["artifact_checks"] if x["role"] == "raw_camera_frames")
        self.assertEqual(entry["status"], "mismatch")
        self.assertIn("SHA-256 mismatch", entry["errors"])
        self.assertFalse(report["bundle_integrity_valid"])

        path.write_bytes(content)
        missing = deepcopy(self.manifest)
        missing["artifacts"][0]["path"] = "frames/missing.bin"
        report = validate_capture_bundle(missing, self.root)
        self.assertEqual(report["artifact_checks"][0]["status"], "missing")
        self.assertFalse(report["bundle_integrity_valid"])

        wrong_size = deepcopy(self.manifest)
        wrong_size["artifacts"][0]["size_bytes"] += 1
        report = validate_capture_bundle(wrong_size, self.root)
        self.assertEqual(report["artifact_checks"][0]["status"], "mismatch")
        self.assertIn("size mismatch", report["artifact_checks"][0]["errors"])

    def test_path_traversal_absolute_drive_and_backslash_fail_without_reading(self):
        for unsafe in ("../outside.bin", "/etc/passwd", "C:/Windows/system.ini", "frames\\..\\outside.bin"):
            with self.subTest(path=unsafe):
                candidate = deepcopy(self.manifest)
                candidate["artifacts"][0]["path"] = unsafe
                report = validate_capture_bundle(candidate, self.root)
                self.assertFalse(report["bundle_integrity_valid"])
                self.assertEqual(
                    report["artifact_checks"][0]["status"], "unsafe_or_unreadable"
                )

    def test_symlink_escape_and_duplicate_path_are_rejected(self):
        outside = Path(self.temp.name) / "outside.bin"
        outside.write_bytes(b"not-inside-bundle")
        link = self.root / "frames" / "external.bin"
        link.symlink_to(outside)
        candidate = deepcopy(self.manifest)
        candidate["artifacts"][0]["path"] = "frames/external.bin"
        report = validate_capture_bundle(candidate, self.root)
        self.assertEqual(report["artifact_checks"][0]["status"], "unsafe_or_unreadable")
        self.assertIn("symlink", report["artifact_checks"][0]["errors"][0])

        duplicate = deepcopy(self.manifest)
        duplicate["artifacts"][1]["path"] = duplicate["artifacts"][0]["path"]
        report = validate_capture_bundle(duplicate, self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertEqual(report["artifact_checks"][1]["status"], "duplicate_path")

        case_collision = deepcopy(self.manifest)
        case_collision["artifacts"][1]["path"] = case_collision["artifacts"][0]["path"].upper()
        report = validate_capture_bundle(case_collision, self.root)
        self.assertEqual(report["artifact_checks"][1]["status"], "duplicate_path")

    def test_missing_required_role_and_unlinked_revision_do_not_pass(self):
        missing = deepcopy(self.manifest)
        missing["artifacts"] = [
            x for x in missing["artifacts"] if x["role"] != "reference_trigger_log"
        ]
        report = validate_capture_bundle(missing, self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertEqual(report["required_roles_missing"], ["reference_trigger_log"])

        mismatch = deepcopy(self.manifest)
        mismatch["revisions"]["codebook_sha256"] = "f" * 64
        report = validate_capture_bundle(mismatch, self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertTrue(any("codebook artifact SHA" in x for x in report["failures"]))

    def test_limits_predeclared_after_capture_and_no_independent_reference_fail(self):
        late = deepcopy(self.manifest)
        late["acceptance_limits"]["predeclared_at_utc"] = "2026-10-22T01:01:00Z"
        report = validate_capture_bundle(late, self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertFalse(report["declared_capture"]["acceptance_predeclared_before_capture"])

        no_reference = deepcopy(self.manifest)
        no_reference["capture"]["independent_reference_declared"] = False
        report = validate_capture_bundle(no_reference, self.root)
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertFalse(report["declarations_not_verified"]["independent_reference_verified"])

    def test_unknown_fields_bad_digest_duplicate_roles_and_wrong_schema_reject(self):
        variants = []
        extra = deepcopy(self.manifest)
        extra["hardware_verified"] = True
        variants.append((extra, "unknown hardware_verified"))
        nested = deepcopy(self.manifest)
        nested["capture"]["hardware_verified"] = True
        variants.append((nested, "unknown hardware_verified"))
        bad_semantics = deepcopy(self.manifest)
        bad_semantics["capture"]["timestamp_semantics"] = []
        variants.append((bad_semantics, "timestamp_semantics"))
        huge_rate = deepcopy(self.manifest)
        huge_rate["software"]["devices"][0]["nominal_fps"] = 10**1000
        variants.append((huge_rate, "nominal_fps"))
        huge_rms = deepcopy(self.manifest)
        huge_rms["acceptance_limits"]["max_reprojection_rms_px"] = 10**1000
        variants.append((huge_rms, "max_reprojection_rms_px"))
        digest = deepcopy(self.manifest)
        digest["artifacts"][0]["sha256"] = "A" * 64
        variants.append((digest, "lowercase hexadecimal SHA-256"))
        schema = deepcopy(self.manifest)
        schema["schema_version"] = True
        variants.append((schema, "schema_version"))
        duplicate_role = deepcopy(self.manifest)
        duplicate_role["artifacts"][1]["role"] = duplicate_role["artifacts"][0]["role"]
        variants.append((duplicate_role, "artifact roles must be unique"))
        for candidate, expected in variants:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    validate_capture_bundle(candidate, self.root)

    def test_artifact_and_total_byte_caps_reject_before_hashing(self):
        with self.assertRaisesRegex(ValueError, "1..7 entries"):
            validate_capture_bundle(self.manifest, self.root, max_artifacts=7)
        with self.assertRaisesRegex(ValueError, "total byte cap"):
            validate_capture_bundle(self.manifest, self.root, max_total_bytes=20)

    def test_manifest_loader_rejects_duplicate_keys_nonfinite_and_oversized(self):
        for raw, expected in (
            (b'{"schema_version":1,"schema_version":1}', "duplicate JSON"),
            (b'{"value":NaN}', "nonfinite JSON"),
        ):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, expected):
                load_manifest_bytes(raw)
        with self.assertRaisesRegex(ValueError, "no larger than"):
            load_manifest_bytes(b"x" * 1_000_001)

    def test_directory_manifest_is_opened_without_following_symlinks(self):
        manifest_path = self.root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        report = validate_capture_bundle_directory(self.root)
        self.assertTrue(report["bundle_integrity_valid"], report["failures"])

        # A manifest symlink must fail before its target is parsed.
        target = Path(self.temp.name) / "outside-manifest.json"
        target.write_text(json.dumps(self.manifest), encoding="utf-8")
        manifest_path.unlink()
        manifest_path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "without following links"):
            validate_capture_bundle_directory(self.root)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO test requires POSIX")
    def test_directory_manifest_fifo_fails_without_blocking(self):
        manifest_path = self.root / MANIFEST_NAME
        os.mkfifo(manifest_path)
        api_dir = Path(__file__).resolve().parents[1]
        script = (
            "import sys; "
            "from capture_bundle_manifest import validate_capture_bundle_directory; "
            "try:\n validate_capture_bundle_directory(sys.argv[1])\n"
            "except ValueError as exc:\n print(str(exc)); sys.exit(0)\n"
            "sys.exit(1)"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(api_dir)
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root)],
            cwd=api_dir, env=env, capture_output=True, text=True, timeout=3,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("regular file", result.stdout)

    def test_directory_manifest_descriptor_enforces_size_cap(self):
        manifest_path = self.root / MANIFEST_NAME
        manifest_path.write_bytes(b" " * 1_000_001)
        with self.assertRaisesRegex(ValueError, "byte cap"):
            validate_capture_bundle_directory(self.root)

    def test_directory_cli_outputs_machine_report_and_never_certifies_hardware(self):
        (self.root / MANIFEST_NAME).write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )
        output = StringIO()
        with patch("sys.stdout", output):
            code = cli_main([str(self.root)])
        self.assertEqual(code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["bundle_integrity_valid"])
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])
        self.assertIn("manifest_file_sha256", report)

        malicious = deepcopy(self.manifest)
        malicious["hardware_verified"] = True
        (self.root / MANIFEST_NAME).write_text(json.dumps(malicious), encoding="utf-8")
        output = StringIO()
        with patch("sys.stdout", output):
            code = cli_main([str(self.root)])
        self.assertEqual(code, 2)
        report = json.loads(output.getvalue())
        self.assertFalse(report["bundle_integrity_valid"])
        self.assertFalse(report["declarations_not_verified"]["hardware_provenance_verified"])


if __name__ == "__main__":
    unittest.main()
