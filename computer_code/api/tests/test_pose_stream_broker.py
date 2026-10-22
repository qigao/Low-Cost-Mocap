"""Independent bounded pose publisher/consumer conformance without a network."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import BoardOutcome, BoardPose
from identity_locked_board_tracker import BoardFrameStatus, LockedBoardFrame
from pose_stream import pose_frame_to_dict
from pose_stream_broker import (
    BoundedPoseStreamPublisher, PoseStreamConsumer, flush_to_socketio,
)


CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"


class BoundedPoseStreamTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        self.publisher = BoundedPoseStreamPublisher(
            self.registry, max_subscribers=2, max_pending_frames=2,
        )

    def frame(self, trigger, *, calibration="a" * 64, clock="rig-clock",
              measured=True):
        timestamp = 1_000_000_000 + trigger * 10_000
        body = self.registry.boards["board_A"]
        board_t = np.array([0.3 + trigger * 0.001, 0.1, 2.0])
        position_body = board_t - body.translation_body_from_board_m
        pose = BoardPose(
            board_id="board_A", trigger_id=trigger, marker_ids=(100, 101, 102),
            rotation_world_from_board=np.eye(3),
            translation_world_from_board=board_t,
            rotation_world_from_body=np.eye(3),
            translation_world_from_body=position_body,
            rigid_rms_error=0.001, reprojection_max_rms_px=0.03,
        )
        a = BoardOutcome(
            "board_A", "recognized" if measured else "insufficient_markers",
            (100, 101, 102) if measured else (100, 101),
            (103,), pose if measured else None,
        )
        b = BoardOutcome("board_B", "not_visible", (), (), None)
        return LockedBoardFrame(
            trigger_id=trigger,
            camera_ids=(2, 5),
            clock_domain=clock,
            exposure_skew_ns=10,
            latest_exposure_timestamp_ns=timestamp,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256=calibration,
            boards={
                "board_A": BoardFrameStatus(
                    "board_A", "TRACKING" if measured else "DISCOVER",
                    "MEASURED" if measured else "INVALID",
                    trigger if measured else None,
                    0 if measured else None,
                    timestamp if measured else None,
                    0 if measured else None,
                    a,
                ),
                "board_B": BoardFrameStatus(
                    "board_B", "DISCOVER", "INVALID",
                    None, None, None, None, b,
                ),
            },
            active_locks=(),
            newly_confirmed_marker_ids=(),
            invalidated_marker_ids=(),
            measured_marker_ids=(100, 101, 102) if measured else (),
        )

    def test_assets_before_frames_bounded_drops_and_independent_consumer(self):
        self.publisher.subscribe("viewerA")
        for trigger in range(1, 6):
            self.assertEqual(self.publisher.publish(self.frame(trigger)), trigger)
        stats = self.publisher.subscriber_statistics("viewerA")
        self.assertEqual(stats.pending_frames, 2)
        self.assertEqual(stats.dropped_frames, 3)
        self.assertTrue(stats.pending_asset)
        consumer = PoseStreamConsumer()
        packet = self.publisher.peek_next("viewerA")
        self.assertEqual(packet["sequence"], 0)
        self.assertEqual(packet["payload"]["kind"], "asset_description")
        self.assertEqual(consumer.consume(packet)["kind"], "asset_description")
        self.publisher.acknowledge("viewerA", packet["sequence"])

        received = []
        while (packet := self.publisher.peek_next("viewerA")) is not None:
            received.append(consumer.consume(packet))
            self.publisher.acknowledge("viewerA", packet["sequence"])
        self.assertEqual([p["frame_id"] for p in received], [4, 5])
        self.assertEqual(consumer.last_frame_id, 5)
        self.assertEqual(consumer.pattern_sha256, self.registry.config_sha256)
        self.assertIsNone(received[-1]["rigid_bodies"][1]["position_world_from_body"])
        self.assertEqual(self.publisher.subscriber_statistics("viewerA").pending_frames, 0)

    def test_source_stale_frame_calibration_change_and_clock_mix_fail_closed(self):
        self.publisher.subscribe("viewerA")
        self.publisher.publish(self.frame(10))
        for invalid in (
            self.frame(10),
            self.frame(9),
            self.frame(11, calibration="b" * 64),
            self.frame(11, clock="host-clock"),
        ):
            with self.subTest(invalid=repr(invalid)[:100]):
                with self.assertRaises(ValueError):
                    self.publisher.publish(invalid)
                self.assertEqual(
                    self.publisher.subscriber_statistics("viewerA").pending_frames, 1
                )
        self.publisher.publish(self.frame(11))
        self.assertEqual(self.publisher.subscriber_statistics("viewerA").pending_frames, 2)

    def test_source_frame_must_be_really_measured_or_explicitly_invalid(self):
        self.publisher.subscribe("viewerA")
        bogus = self.frame(1)
        fake = dict(bogus.boards)
        fresh = fake["board_A"]
        fake["board_A"] = replace(
            fresh,
            measurement_age_triggers=1,
        )
        # The independent producer must reject inconsistent source
        # trigger age *before* considering any pose to be fresh MEASURED.
        with self.assertRaisesRegex(ValueError, "measurement trigger age is inconsistent"):
            self.publisher.publish(replace(bogus, boards=fake))
        self.assertEqual(self.publisher.subscriber_statistics("viewerA").pending_frames, 0)
        self.publisher.publish(self.frame(1, measured=False))
        packet = self.publisher.peek_next("viewerA")
        self.publisher.acknowledge("viewerA", packet["sequence"])
        payload = self.publisher.peek_next("viewerA")["payload"]
        self.assertEqual(payload["rigid_bodies"][0]["evidence_kind"], "INVALID")
        self.assertIsNone(payload["rigid_bodies"][0]["position_world_from_board"])

    def test_callback_failure_does_not_ack_and_success_delivers_static_assets_first(self):
        self.publisher.subscribe("viewerA")
        self.publisher.publish(self.frame(1))
        seen = []
        def emit(event, packet, subscriber):
            seen.append((event, packet["payload"]["kind"], subscriber))
            if packet["payload"]["kind"] == "pose_frame" and len(seen) == 2:
                return False
            return None
        with self.assertRaisesRegex(ValueError, "did not acknowledge"):
            self.publisher.flush_with_callback("viewerA", emit)
        self.assertEqual(self.publisher.subscriber_statistics("viewerA").pending_frames, 1)
        self.assertFalse(self.publisher.subscriber_statistics("viewerA").pending_asset)
        self.assertEqual(self.publisher.flush_with_callback("viewerA", emit), 1)
        self.assertEqual(seen, [
            ("mocap:pose-stream", "asset_description", "viewerA"),
            ("mocap:pose-stream", "pose_frame", "viewerA"),
            ("mocap:pose-stream", "pose_frame", "viewerA"),
        ])
        self.assertIsNone(self.publisher.peek_next("viewerA"))

    def test_socketio_callback_adapter_uses_targeted_room_without_io_imports(self):
        self.publisher.subscribe("viewerA")
        self.publisher.publish(self.frame(1))
        called = []
        def fake_socketio_emit(event, message, *, to):
            called.append((event, message["sequence"], to))
            return None

        self.assertEqual(
            flush_to_socketio(
                self.publisher, "viewerA", fake_socketio_emit,
                max_messages=4,
            ), 2,
        )
        self.assertEqual(called, [
            ("mocap:pose-stream", 0, "viewerA"),
            ("mocap:pose-stream", 1, "viewerA"),
        ])
        self.assertIsNone(self.publisher.peek_next("viewerA"))

    def test_full_asset_envelope_byte_bound_and_consumer_non_json_fail_closed(self):
        from pose_stream_broker import _envelope
        from pose_stream import describe_board_assets
        asset = describe_board_assets(self.registry)
        sample = _envelope(asset, session_id=1, sequence=0)
        import json
        size = len(json.dumps(
            sample, sort_keys=True, separators=(",", ":")
        ).encode("utf8"))
        with self.assertRaisesRegex(ValueError, "asset description exceeds"):
            BoundedPoseStreamPublisher(
                self.registry, max_message_bytes=size - 1
            )
        self.publisher.subscribe("viewerA")
        consumer = PoseStreamConsumer()
        candidate = self.publisher.peek_next("viewerA")
        candidate["payload"]["bad_set"] = {1, 2}
        with self.assertRaisesRegex(ValueError, "non-JSON"):
            consumer.consume(candidate)
        self.assertIsNone(consumer.session_id)

    def test_reconnect_resubscribe_needs_explicit_fresh_asset(self):
        self.publisher.subscribe("viewerA")
        consumer = PoseStreamConsumer()
        packet = self.publisher.peek_next("viewerA")
        consumer.consume(packet)
        self.publisher.acknowledge("viewerA", packet["sequence"])
        self.publisher.publish(self.frame(1))
        packet = self.publisher.peek_next("viewerA")
        consumer.consume(packet)
        self.publisher.acknowledge("viewerA", packet["sequence"])

        self.publisher.unsubscribe("viewerA")
        self.publisher.subscribe("viewerA")
        fresh = self.publisher.peek_next("viewerA")
        with self.assertRaisesRegex(ValueError, "sequence"):
            consumer.consume(fresh)
        consumer.reset_for_reconnect()
        consumer.consume(fresh)
        self.assertEqual(consumer.last_sequence, 0)
        self.publisher.acknowledge("viewerA", 0)
        self.publisher.publish(self.frame(2))
        consumer.consume(self.publisher.peek_next("viewerA"))
        self.assertEqual(consumer.last_frame_id, 2)

    def test_session_restart_enforces_new_assets_and_allows_timebase_reset(self):
        self.publisher.subscribe("viewerA")
        consumer = PoseStreamConsumer()
        consumer.consume(self.publisher.peek_next("viewerA"))
        self.publisher.acknowledge("viewerA", 0)
        self.publisher.publish(self.frame(20))
        consumer.consume(self.publisher.peek_next("viewerA"))
        self.publisher.acknowledge("viewerA", 1)

        self.publisher.restart_session()
        asset = self.publisher.peek_next("viewerA")
        self.assertEqual(asset["session_id"], 2)
        self.assertEqual(asset["sequence"], 0)
        consumer.consume(asset)
        self.publisher.acknowledge("viewerA", 0)
        self.publisher.publish(self.frame(1, calibration="b"*64))
        packet = self.publisher.peek_next("viewerA")
        consumer.consume(packet)
        self.assertEqual(consumer.last_frame_id, 1)
        self.assertEqual(consumer.calibration_sha256, "b"*64)

    def test_replacement_pattern_revision_forces_new_assets_and_frames(self):
        self.publisher.subscribe("viewerA")
        self.publisher.publish(self.frame(1))
        baseline = self.publisher.subscriber_statistics("viewerA")
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        raw["boards"]["board_B"]["markers"]["107"]["xyz_m"][2] += 0.004
        changed = BoardPatternRegistry.from_mapping(raw)
        self.assertNotEqual(changed.config_sha256, self.registry.config_sha256)
        self.publisher.replace_registry(changed)
        stats = self.publisher.subscriber_statistics("viewerA")
        self.assertEqual(stats.pending_frames, 0)
        self.assertTrue(stats.pending_asset)
        self.assertEqual(stats.session_id, baseline.session_id + 1)
        payload = self.publisher.peek_next("viewerA")["payload"]
        self.assertEqual(payload["pattern_sha256"], changed.config_sha256)
        with self.assertRaisesRegex(ValueError, "pattern revision"):
            self.publisher.publish(self.frame(2))

    def test_subscriber_bounded_capacity_unsubscribe_and_packet_mutation_isolation(self):
        self.publisher.subscribe("viewerA")
        self.publisher.subscribe("viewerB")
        with self.assertRaisesRegex(ValueError, "capacity"):
            self.publisher.subscribe("viewerC")
        with self.assertRaisesRegex(ValueError, "already connected"):
            self.publisher.subscribe("viewerA")
        first = self.publisher.peek_next("viewerA")
        first["payload"]["boards"].clear()
        second = self.publisher.peek_next("viewerA")
        self.assertEqual(len(second["payload"]["boards"]), 2)
        self.publisher.unsubscribe("viewerA")
        self.publisher.subscribe("viewerC")
        self.assertEqual(self.publisher.subscriber_ids, ("viewerB", "viewerC"))


class AdversarialConsumerTests(BoundedPoseStreamTests):
    def _packets(self):
        self.publisher.subscribe("viewerA")
        self.publisher.publish(self.frame(1))
        asset = self.publisher.peek_next("viewerA")
        self.publisher.acknowledge("viewerA", 0)
        frame = self.publisher.peek_next("viewerA")
        return asset, frame

    def test_rejects_unknown_boards_wrong_quaternion_and_stale_pose_atomically(self):
        asset, packet = self._packets()
        for variant in ("unknown", "unit", "stale", "duplicate", "raw", "mount", "minimum"):
            consumer = PoseStreamConsumer()
            consumer.consume(asset)
            invalid = deepcopy(packet)
            body = invalid["payload"]["rigid_bodies"][0]
            if variant == "unknown":
                body["board_id"] = "foreign-board"
            elif variant == "unit":
                body["quaternion_world_from_body_xyzw"] = [0., 0., 1., 1.]
            elif variant == "stale":
                body["measurement_age_triggers"] = 1
            elif variant == "duplicate":
                body["visible_marker_ids"] = [100, 100]
            elif variant == "raw":
                invalid["payload"]["hardware_provenance_verified"] = True
            elif variant == "mount":
                body["position_world_from_body"][0] += 0.7
            elif variant == "minimum":
                body["visible_marker_ids"] = [100, 101]
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                consumer.consume(invalid)
            self.assertEqual(consumer.last_sequence, 0)
            self.assertIsNone(consumer.last_frame_id)
            correct = consumer.consume(packet)
            self.assertEqual(correct["frame_id"], 1)

    def test_new_session_must_start_with_assets_not_pose(self):
        asset, frame = self._packets()
        consumer = PoseStreamConsumer()
        with self.assertRaisesRegex(ValueError, "fresh asset"):
            consumer.consume(frame)
        consumer.consume(asset)
        switched = deepcopy(frame)
        switched["session_id"] = 2
        switched["sequence"] = 0
        with self.assertRaisesRegex(ValueError, "fresh asset"):
            consumer.consume(switched)

    def test_nonmonotonic_replay_frame_source_fails_even_if_wire_sequence_advances(self):
        asset, frame = self._packets()
        consumer = PoseStreamConsumer()
        consumer.consume(asset)
        consumer.consume(frame)
        repeated = deepcopy(frame)
        repeated["sequence"] += 1
        with self.assertRaisesRegex(ValueError, "stale"):
            consumer.consume(repeated)
        self.assertEqual(consumer.last_sequence, 1)


if __name__ == "__main__":
    unittest.main()
