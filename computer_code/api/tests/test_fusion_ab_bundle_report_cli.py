import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fusion_ab_bundle_report_cli import evaluate_bundle
from capture_bundle_manifest import MANIFEST_NAME, validate_capture_bundle_and_read_role


def ab_document():
    pattern, optical_cal, imu_cal, mapping, reference = (
        "b"*64, "c"*64, "a"*64, "d"*64, "e"*64
    )
    optical = {
        "body_id":"board-A", "timestamp_ns":100, "clock_domain":"imu-clock",
        "quaternion_world_from_body_xyzw":[0,0,0,1], "status":"TRACKING",
        "pattern_sha256":pattern, "optical_calibration_sha256":optical_cal,
    }
    fused = {
        "body_id":"board-A", "imu_id":"imu-1", "timestamp_ns":100,
        "clock_domain":"imu-clock", "quaternion_world_from_body_xyzw":[0,0,0,1],
        "attitude_variance_rad2":0.01, "status":"TRACKING", "source":"OPTICAL_CORRECTED",
        "last_optical_timestamp_ns":100, "last_optical_frame_id":1, "optical_age_ns":0,
        "clock_mapping_sha256":mapping, "imu_calibration_sha256":imu_cal,
        "optical_pattern_sha256":pattern, "optical_calibration_sha256":optical_cal,
        "global_position_m":None, "global_velocity_m_s":None, "global_position_observable":False,
    }
    ref = {"timestamp_ns":100,"clock_domain":"imu-clock","quaternion_world_from_body_xyzw":[0,0,0,1],"reference_sha256":reference}
    return {"schema_version":1,"clock_domain":"imu-clock","max_match_delta_ns":0,
            "optical_only":[optical],"fused":[fused],"references":[ref]}


def make_bundle(root, *, include_fusion=True):
    root.mkdir(parents=True, exist_ok=True)
    raw_ab = json.dumps(ab_document(), sort_keys=True, separators=(",", ":")).encode()
    payloads = {
        "raw_camera_frames": ("frames/raw.bin", b"camera-raw-bytes"),
        "reference_trigger_log": ("reference/triggers.jsonl", b"trigger-reference"),
        "independent_marker_labels": ("labels/markers.jsonl", b"independent-labels"),
        "device_metadata": ("metadata/devices.json", b"device-metadata"),
        "led_controller_log": ("controller/led.log", b"controller-log"),
        "pattern_config": ("config/pattern.json", b"pattern-config"),
        "codebook": ("config/codebook.json", b"codebook"),
        "controller_config": ("config/controller.json", b"controller-config"),
        "calibration_config": ("config/calibration.json", b"calibration-config"),
    }
    if include_fusion:
        payloads["fusion_ab_input"] = ("reports/fusion-ab-input.json", raw_ab)
    artifacts = []
    digests = {}
    for role, (rel, raw) in payloads.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        digests[role] = digest
        artifacts.append({"role":role,"path":rel,"size_bytes":len(raw),"sha256":digest,"media_type":"application/json" if rel.endswith(".json") or rel.endswith(".jsonl") else "application/octet-stream"})
    manifest = {
        "schema_version":1, "bundle_id":"rig-session-ab-001",
        "capture":{
            "started_at_utc":"2026-10-22T01:00:00Z","ended_at_utc":"2026-10-22T01:10:00Z",
            "camera_ids":[0,1],"clock_domain":"declared-reference-clock",
            "timestamp_semantics":"sensor_exposure","timestamp_unit":"ns",
            "reference_source":"external-reference-logger","independent_reference_declared":True,
        },
        "software":{
            "python_version":"3.13.0","host_os":"Linux","host_os_version":"6.8",
            "application_commit_sha":"a"*40,"pseyepy_version":"0.6.0",
            "led_controller_model":"controller-v1","led_controller_firmware_version":"1.0",
            "devices":[{"camera_id":0,"model":"camera-A","firmware_version":"1","driver_version":"1","frame_width":640,"frame_height":480,"nominal_fps":60.0},
                       {"camera_id":1,"model":"camera-B","firmware_version":"1","driver_version":"1","frame_width":640,"frame_height":480,"nominal_fps":60.0}],
        },
        "revisions":{
            "pattern_sha256":hashlib.sha256(b"pattern").hexdigest(),
            "pattern_config_artifact_sha256":digests["pattern_config"],
            "codebook_sha256":digests["codebook"],
            "controller_config_sha256":digests["controller_config"],
            "calibration_sha256":digests["calibration_config"],
        },
        "acceptance_limits":{
            "predeclared_at_utc":"2026-10-22T00:30:00Z","max_exposure_skew_ns":100000,
            "max_reprojection_rms_px":1.0,"max_marker_id_switches":0,
            "min_labeled_frames":100,"min_labeled_tracks":1,
        },
        "artifacts":artifacts,
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


class FusionABBundleReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.manifest = make_bundle(self.root)

    def test_valid_bundle_binds_report_to_manifest_and_exact_artifact(self):
        report = evaluate_bundle(self.root)
        artifact = next(a for a in self.manifest["artifacts"] if a["role"] == "fusion_ab_input")
        self.assertTrue(report["bundle_integrity_valid"])
        self.assertEqual(report["bundle_id"], self.manifest["bundle_id"])
        self.assertEqual(report["fusion_ab_artifact_path"], artifact["path"])
        self.assertEqual(report["fusion_ab_artifact_sha256"], artifact["sha256"])
        self.assertEqual(report["recording_sha256"], artifact["sha256"])
        self.assertEqual(len(report["manifest_sha256"]), 64)
        self.assertEqual(len(report["manifest_file_sha256"]), 64)
        self.assertEqual(len(report["bundle_bound_report_sha256"]), 64)
        self.assertFalse(report["hardware_provenance_verified"])
        self.assertFalse(report["hardware_ground_truth_verified"])

    def test_missing_role_and_tampered_bytes_fail_closed(self):
        root2 = Path(self.temp.name) / "no-fusion"
        make_bundle(root2, include_fusion=False)
        with self.assertRaisesRegex(ValueError, "exactly one fusion_ab_input"):
            evaluate_bundle(root2)
        item = next(a for a in self.manifest["artifacts"] if a["role"] == "fusion_ab_input")
        (self.root / item["path"]).write_bytes(b'{"tampered":true}')
        with self.assertRaisesRegex(ValueError, "integrity validation failed"):
            evaluate_bundle(self.root)

    def test_safe_reader_rejects_symlink_artifact_and_byte_mutation(self):
        artifact = next(a for a in self.manifest["artifacts"] if a["role"] == "fusion_ab_input")
        target = self.root / artifact["path"]
        raw = target.read_bytes()
        outside = Path(self.temp.name) / "outside.json"
        outside.write_bytes(raw)
        target.unlink()
        target.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "integrity validation failed"):
            validate_capture_bundle_and_read_role(self.root, role="fusion_ab_input")

    def test_changed_manifest_is_bound_into_report_digest(self):
        first = evaluate_bundle(self.root)
        manifest_path = self.root / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["bundle_id"] = "rig-session-ab-002"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        second = evaluate_bundle(self.root)
        self.assertNotEqual(first["manifest_sha256"], second["manifest_sha256"])
        self.assertNotEqual(first["bundle_bound_report_sha256"], second["bundle_bound_report_sha256"])


if __name__ == "__main__":
    unittest.main()
