import copy
import unittest
from pathlib import Path

from board_patterns import BoardPatternRegistry
from fused_attitude_stream import (
    fused_attitude_state_to_dict,
    validate_fused_attitude_payload,
)
from optical_imu_fusion import OpticalIMUFusionState
from pose_stream_broker import PoseStreamConsumer, _envelope
from pose_stream_recording import BoundedPoseStreamJournal


PATTERN = "b" * 64
OPTICAL_CAL = "c" * 64
IMU_CAL = "a" * 64
MAPPING = "d" * 64
CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"


def state(*, status="TRACKING", timestamp=10, optical_timestamp=10, frame_id=1,
          q=(0.0, 0.0, 0.0, 1.0), variance=0.002, source=None):
    if source is None:
        source = {"TRACKING": "OPTICAL_CORRECTED", "DEGRADED": "IMU_PROPAGATED", "LOST": "LOST"}[status]
    lost = status == "LOST"
    return OpticalIMUFusionState(
        body_id="board_A",
        imu_id="imu-1",
        timestamp_ns=timestamp,
        clock_domain="imu-clock",
        quaternion_world_from_body_xyzw=None if lost else q,
        attitude_variance_rad2=None if lost else variance,
        status=status,
        source=source,
        last_optical_timestamp_ns=optical_timestamp,
        last_optical_frame_id=frame_id,
        optical_age_ns=None if optical_timestamp is None else timestamp-optical_timestamp,
        clock_mapping_sha256=MAPPING,
        imu_calibration_sha256=IMU_CAL,
        optical_pattern_sha256=PATTERN,
        optical_calibration_sha256=OPTICAL_CAL,
    )


def validate(payload, previous_timestamp_ns=None):
    return validate_fused_attitude_payload(
        payload,
        known_body_ids={"board_A"},
        expected_pattern_sha256=PATTERN,
        expected_optical_calibration_sha256=OPTICAL_CAL,
        previous_timestamp_ns=previous_timestamp_ns,
    )


class FusedAttitudeStreamTests(unittest.TestCase):
    def test_tracking_degraded_and_lost_roundtrip(self):
        for status in ("TRACKING", "DEGRADED", "LOST"):
            with self.subTest(status=status):
                payload = fused_attitude_state_to_dict(state(status=status, timestamp=20, optical_timestamp=10))
                accepted = validate(payload)
                self.assertEqual(accepted["status"], status)
                self.assertFalse(accepted["global_position_observable"])
                self.assertIsNone(accepted["global_position_m"])
                if status == "LOST":
                    self.assertIsNone(accepted["quaternion_world_from_body_xyzw"])
                    self.assertIsNone(accepted["attitude_variance_rad2"])
                else:
                    self.assertEqual(len(accepted["quaternion_world_from_body_xyzw"]), 4)

    def test_rejects_unknown_fields_versions_bool_schema_and_unknown_body(self):
        payload = fused_attitude_state_to_dict(state())
        for mutation, message in (
            (lambda p: p.update(extra=1), "fields mismatch"),
            (lambda p: p.update(schema_version=True), "unsupported fused-attitude schema"),
            (lambda p: p.update(body_id="unregistered"), "unknown body"),
        ):
            item = copy.deepcopy(payload)
            mutation(item)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate(item)

    def test_rejects_bad_quaternion_variance_and_lost_pose(self):
        payload = fused_attitude_state_to_dict(state())
        cases = [
            ({"quaternion_world_from_body_xyzw": [0, 0, float("nan"), 1]}, "finite numeric"),
            ({"quaternion_world_from_body_xyzw": [0, 0, 0, 0]}, "normalized"),
            ({"attitude_variance_rad2": -0.1}, "nonnegative"),
            ({"status": "LOST", "source": "LOST"}, "LOST fused attitude"),
        ]
        for update, message in cases:
            item = copy.deepcopy(payload)
            item.update(update)
            if item["status"] == "LOST":
                item["quaternion_world_from_body_xyzw"] = [0, 0, 0, 1]
                item["attitude_variance_rad2"] = 0.1
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate(item)

    def test_rejects_status_source_mismatch_and_publishable_state_without_optical_seed(self):
        payload = fused_attitude_state_to_dict(state())
        payload["source"] = "IMU_PROPAGATED"
        with self.assertRaisesRegex(ValueError, "TRACKING requires"):
            validate(payload)
        payload = fused_attitude_state_to_dict(state(status="DEGRADED", timestamp=20, optical_timestamp=10))
        payload["source"] = "OPTICAL_CORRECTED"
        with self.assertRaisesRegex(ValueError, "DEGRADED requires"):
            validate(payload)
        payload = fused_attitude_state_to_dict(state(status="LOST", timestamp=20, optical_timestamp=None, frame_id=None))
        payload["status"] = "DEGRADED"
        payload["source"] = "IMU_PROPAGATED"
        payload["quaternion_world_from_body_xyzw"] = [0, 0, 0, 1]
        payload["attitude_variance_rad2"] = 0.1
        with self.assertRaisesRegex(ValueError, "requires an optical seed"):
            validate(payload)

    def test_rejects_digest_age_position_and_timestamp_regression(self):
        payload = fused_attitude_state_to_dict(state(timestamp=20, optical_timestamp=10))
        cases = [
            ({"optical_age_ns": 8}, "age is inconsistent"),
            ({"optical_pattern_sha256": "e" * 64}, "pattern revision"),
            ({"global_position_observable": True}, "cannot claim global position"),
            ({"global_position_m": [0, 0, 0]}, "cannot publish global position"),
        ]
        for update, message in cases:
            item = copy.deepcopy(payload)
            item.update(update)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    validate(item)
        with self.assertRaisesRegex(ValueError, "stale or regressed"):
            validate(payload, previous_timestamp_ns=21)

    def test_consumer_requires_asset_first_and_enforces_sequence_and_revision(self):
        registry = BoardPatternRegistry.from_json_file(CONFIG)
        consumer = PoseStreamConsumer()
        asset = {
            "protocol": "low_cost_mocap_pose", "schema_version": 1,
            "kind": "asset_description", "pattern_sha256": registry.config_sha256,
            "hardware_provenance_verified": False,
            "model_coordinates_are_declared_not_metrologically_verified": True,
            "period_triggers": registry.period_triggers,
            "max_bit_errors": registry.max_bit_errors,
            "pose_phase": registry.pose_phase,
            "boards": [
                {
                    "board_id": board_id,
                    "type": board.board_type,
                    "min_visible_markers": board.min_visible_markers,
                    "body_from_board": {
                        "rotation": board.rotation_body_from_board.tolist(),
                        "translation_m": board.translation_body_from_board_m.tolist(),
                    },
                    "markers": [
                        {"marker_id": marker, "xyz_m": [float(x) for x in board.marker_xyz_m[marker]],
                         "code": "".join(str(bit) for bit in registry.codebook.patterns[marker])}
                        for marker in sorted(board.marker_xyz_m)
                    ],
                }
                for board_id, board in sorted(registry.boards.items())
            ],
        }
        consumer.consume(_envelope(asset, session_id=1, sequence=0))
        payload_state = state()
        payload_state = OpticalIMUFusionState(
            body_id=next(iter(registry.boards)), imu_id=payload_state.imu_id,
            timestamp_ns=payload_state.timestamp_ns, clock_domain=payload_state.clock_domain,
            quaternion_world_from_body_xyzw=payload_state.quaternion_world_from_body_xyzw,
            attitude_variance_rad2=payload_state.attitude_variance_rad2,
            status=payload_state.status, source=payload_state.source,
            last_optical_timestamp_ns=payload_state.last_optical_timestamp_ns,
            last_optical_frame_id=payload_state.last_optical_frame_id,
            optical_age_ns=payload_state.optical_age_ns,
            clock_mapping_sha256=MAPPING, imu_calibration_sha256=IMU_CAL,
            optical_pattern_sha256=registry.config_sha256,
            optical_calibration_sha256=OPTICAL_CAL,
        )
        payload = fused_attitude_state_to_dict(payload_state)
        envelope = _envelope(payload, session_id=1, sequence=1)
        accepted = consumer.consume(envelope)
        self.assertEqual(accepted["kind"], "fused_attitude_state")
        with self.assertRaisesRegex(ValueError, "repeated or stale"):
            consumer.consume(envelope)
        changed = copy.deepcopy(payload)
        changed["clock_mapping_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "clock mapping changed"):
            consumer.consume(_envelope(changed, session_id=1, sequence=2))

    def test_journal_roundtrips_fused_payload_with_hash_chain(self):
        registry = BoardPatternRegistry.from_json_file(CONFIG)
        from pose_stream import describe_board_assets
        journal = BoundedPoseStreamJournal()
        journal.append(_envelope(describe_board_assets(registry), session_id=1, sequence=0))
        fused = state()
        fused = OpticalIMUFusionState(
            body_id=next(iter(registry.boards)), imu_id=fused.imu_id,
            timestamp_ns=fused.timestamp_ns, clock_domain=fused.clock_domain,
            quaternion_world_from_body_xyzw=fused.quaternion_world_from_body_xyzw,
            attitude_variance_rad2=fused.attitude_variance_rad2, status=fused.status,
            source=fused.source, last_optical_timestamp_ns=fused.last_optical_timestamp_ns,
            last_optical_frame_id=fused.last_optical_frame_id, optical_age_ns=fused.optical_age_ns,
            clock_mapping_sha256=MAPPING, imu_calibration_sha256=IMU_CAL,
            optical_pattern_sha256=registry.config_sha256, optical_calibration_sha256=OPTICAL_CAL,
        )
        journal.append(_envelope(fused_attitude_state_to_dict(fused), session_id=1, sequence=1))
        replayed = BoundedPoseStreamJournal.from_jsonl(journal.jsonl())
        self.assertEqual(list(replayed.iter_envelopes()), list(journal.iter_envelopes()))
        self.assertEqual(replayed.statistics.head_sha256, journal.statistics.head_sha256)


if __name__ == "__main__":
    unittest.main()
