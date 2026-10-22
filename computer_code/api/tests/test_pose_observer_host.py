"""Test the actual Flask-SocketIO test client without any camera/serial IO."""
from dataclasses import replace
import sys
import unittest
from pathlib import Path

import numpy as np

from board_patterns import BoardPatternRegistry
from board_recognizer import BoardOutcome, BoardPose
from identity_locked_board_tracker import BoardFrameStatus, LockedBoardFrame
from pose_observer_host import PoseObserverHost, POSE_OBSERVER_NAMESPACE
from pose_stream_broker import PoseStreamConsumer


CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"
SECRET = "local-test-viewer-secret-not-for-hardware"


class PoseObserverHostTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        self.clock = 100_000_000
        self.host = PoseObserverHost(
            self.registry,
            viewer_secret=SECRET,
            max_subscribers=2,
            max_pending_frames=2,
            max_emit_per_dispatch=4,
            min_poll_interval_ms=50,
            monotonic_ns=lambda: self.clock,
        )

    def connect(self, *, token=SECRET, origin=None):
        kwargs = {
            "namespace": POSE_OBSERVER_NAMESPACE,
            "auth": {"token": token} if token is not None else None,
        }
        if origin is not None:
            kwargs["headers"] = {"Origin": origin}
        return self.host.socketio.test_client(self.host.app, **kwargs)

    def frame(self, trigger, *, measured=True, calibration="a"*64,
              clock="synthetic-clock"):
        timestamp = 5_000_000 + 1000*trigger
        board = self.registry.boards["board_A"]
        t = np.array([0.25 + trigger*0.001, -0.05, 1.8])
        pos_body = t - board.translation_body_from_board_m
        pose = BoardPose(
            board_id="board_A", trigger_id=trigger,
            marker_ids=(100,101,102),
            rotation_world_from_board=np.eye(3),
            translation_world_from_board=t,
            rotation_world_from_body=np.eye(3),
            translation_world_from_body=pos_body,
            rigid_rms_error=0.0005,
            reprojection_max_rms_px=0.04,
        )
        outcome_a = BoardOutcome(
            "board_A", "recognized" if measured else "not_visible",
            (100,101,102) if measured else (), (),
            pose if measured else None,
        )
        outcome_b = BoardOutcome("board_B", "not_visible", (), (), None)
        return LockedBoardFrame(
            trigger_id=trigger, camera_ids=(2, 5),
            clock_domain=clock, exposure_skew_ns=12,
            latest_exposure_timestamp_ns=timestamp,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256=calibration,
            boards={
                "board_A": BoardFrameStatus(
                    "board_A",
                    "TRACKING" if measured else "DISCOVER",
                    "MEASURED" if measured else "INVALID",
                    trigger if measured else None,
                    0 if measured else None,
                    timestamp if measured else None,
                    0 if measured else None,
                    outcome_a,
                ),
                "board_B": BoardFrameStatus(
                    "board_B", "DISCOVER", "INVALID",
                    None, None, None, None,
                    outcome_b,
                ),
            },
            active_locks=(), newly_confirmed_marker_ids=(),
            invalidated_marker_ids=(),
            measured_marker_ids=(100,101,102) if measured else (),
        )

    def poll(self, client):
        # Keep synthetic host time monotonic so independent test polls do not
        # accidentally exercise the rate-limit branch.
        self.clock += 60_000_000
        response = client.emit(
            "pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, callback=True
        )
        if isinstance(response, list):
            response = response[0] if response else None
        self.assertIsInstance(response, dict)
        return response

    def messages(self, client):
        response = self.poll(client)
        self.assertTrue(response.get("ok"), response)
        self.assertIsInstance(response.get("messages"), list)
        return response["messages"]

    def test_no_anonymous_or_wrong_token_connection(self):
        for token in (None, "", "wrong", "x"*700):
            with self.subTest(token=repr(token)[:40]):
                client = self.connect(token=token)
                self.assertFalse(client.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
                self.assertEqual(self.host.subscriber_count, 0)
        health = self.host.app.test_client().get("/pose-observe/healthz")
        self.assertEqual(health.status_code, 200)
        self.assertFalse(health.get_json()["hardware_provenance_verified"])
        self.assertFalse(health.get_json()["live_hardware_input_enabled"])
        self.assertNotIn("serial", self.host.app.import_name)

    def test_asset_first_then_actual_measured_and_invalid_frames(self):
        client = self.connect()
        self.assertTrue(client.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        first = self.messages(client)
        self.assertEqual([msg["payload"]["kind"] for msg in first],
                         ["asset_description"])
        independent = PoseStreamConsumer()
        independent.consume(first[0])
        self.assertEqual(independent.pattern_sha256, self.registry.config_sha256)
        self.assertEqual(self.host.publish_frame(self.frame(101)), 101)
        second = self.messages(client)
        self.assertEqual(len(second), 1)
        measured = independent.consume(second[0])
        self.assertEqual(measured["frame_id"], 101)
        self.assertEqual(measured["rigid_bodies"][0]["evidence_kind"], "MEASURED")
        self.assertIsNotNone(measured["rigid_bodies"][0]["position_world_from_body"])
        self.assertFalse(measured["hardware_provenance_verified"])
        self.host.publish_frame(self.frame(102, measured=False))
        third = self.messages(client)
        self.assertEqual(third[0]["payload"]["frame_id"], 102)
        missing = independent.consume(third[0])
        self.assertEqual(missing["rigid_bodies"][0]["evidence_kind"], "INVALID")
        self.assertIsNone(missing["rigid_bodies"][0]["position_world_from_body"])
        self.assertEqual(self.host.subscriber_count, 1)
        client.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        self.assertEqual(self.host.subscriber_count, 0)

    def test_subscriber_limit_disconnect_then_reconnect_needs_fresh_assets(self):
        a, b = self.connect(), self.connect()
        self.assertTrue(a.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        self.assertTrue(b.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        third = self.connect()
        self.assertFalse(third.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        self.assertEqual(self.host.subscriber_count, 2)
        self.messages(a)
        self.messages(b)
        self.host.publish_frame(self.frame(111))
        self.assertEqual(len(self.messages(a)), 1)
        self.assertEqual(len(self.messages(b)), 1)
        a.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        self.assertEqual(self.host.subscriber_count, 1)
        reopened = self.connect()
        self.assertTrue(reopened.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        self.assertEqual(
            [e["payload"]["kind"] for e in self.messages(reopened)],
            ["asset_description"],
        )
        # No previous MEASURED pose is replayed merely because the observer
        # has subscribed again. Only a genuinely new source frame can arrive.
        self.assertEqual(self.messages(reopened), [])
        self.host.publish_frame(self.frame(112))
        packet = self.messages(reopened)
        self.assertEqual(len(packet), 1)
        self.assertEqual(packet[0]["payload"]["frame_id"], 112)
        b.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        reopened.disconnect(namespace=POSE_OBSERVER_NAMESPACE)

    def test_slow_viewer_drops_old_frames_not_assets_or_other_subscribers(self):
        first = self.connect()
        second = self.connect()
        self.assertEqual(len(self.messages(first)), 1)
        self.assertEqual(len(self.messages(second)), 1)
        # Synthetic trusted producer fills the broker without dispatching.
        # A slow subscriber must remain bounded; no background worker,
        # hardware source or failed transport assumption is involved.
        for trigger in range(1, 8):
            self.host.publisher.publish(self.frame(trigger))
        current_stats = [
            self.host.publisher.subscriber_statistics(key)
            for key in self.host.publisher.subscriber_ids
        ]
        self.assertEqual(len(current_stats), 2)
        self.assertTrue(all(stats.dropped_frames == 5 for stats in current_stats))
        self.assertTrue(all(stats.pending_frames == 2 for stats in current_stats))
        reply = self.poll(first)
        self.assertEqual(reply["sent"], 2)
        received = reply["messages"]
        self.assertEqual(
            [packet["payload"]["frame_id"] for packet in received],
            [6, 7],
        )
        # The other subscriber still holds its own independent 2 frames.
        self.assertEqual(
            sorted(
                self.host.publisher.subscriber_statistics(key).pending_frames
                for key in self.host.publisher.subscriber_ids
            ),
            [0, 2],
        )
        self.assertEqual(
            sorted(
                self.host.publisher.subscriber_statistics(key).pending_frames
                for key in self.host.publisher.subscriber_ids
            ),
            [0, 2],
        )
        self.assertEqual(
            [p["payload"]["frame_id"] for p in self.messages(second)], [6, 7]
        )
        first.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        second.disconnect(namespace=POSE_OBSERVER_NAMESPACE)

    def test_source_restart_sends_new_assets_and_resets_clock(self):
        client = self.connect()
        first = self.messages(client)
        independent = PoseStreamConsumer()
        independent.consume(first[0])
        self.host.publish_frame(self.frame(100))
        independent.consume(self.messages(client)[0])
        self.host.restart_source_session()
        restarted = self.messages(client)
        self.assertEqual(len(restarted), 1)
        self.assertEqual(restarted[0]["payload"]["kind"], "asset_description")
        self.assertGreater(restarted[0]["session_id"], first[0]["session_id"])
        independent.consume(restarted[0])
        self.host.publish_frame(self.frame(2, calibration="b"*64))
        new_frame = self.messages(client)
        self.assertEqual(new_frame[0]["payload"]["frame_id"], 2)
        independent.consume(new_frame[0])
        self.assertEqual(independent.calibration_sha256, "b"*64)

    def test_pattern_revision_reconnect_barrier_and_source_mismatch(self):
        client = self.connect()
        self.messages(client)
        raw = {
            "version": 1,
            "period_triggers": self.registry.period_triggers,
            "max_bit_errors": self.registry.max_bit_errors,
            "pose_phase": self.registry.pose_phase,
            "boards": {},
        }
        for board_id, board in self.registry.boards.items():
            raw["boards"][board_id] = {
                "type": board.board_type,
                "min_visible_markers": board.min_visible_markers,
                "body_from_board": {
                    "rotation": board.rotation_body_from_board.tolist(),
                    "translation_m": board.translation_body_from_board_m.tolist(),
                },
                "markers": {
                    str(marker): {
                        "xyz_m": list(xyz),
                        "code": "".join(str(bit) for bit in self.registry.codebook.patterns[marker]),
                    } for marker, xyz in board.marker_xyz_m.items()
                },
            }
        raw["boards"]["board_B"]["markers"]["107"]["xyz_m"][2] += 0.004
        changed = BoardPatternRegistry.from_mapping(raw)
        self.host.replace_pattern_registry(changed)
        messages = self.messages(client)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["payload"]["pattern_sha256"], changed.config_sha256)
        with self.assertRaisesRegex(ValueError, "pattern revision"):
            self.host.publish_frame(self.frame(103))
        client.disconnect(namespace=POSE_OBSERVER_NAMESPACE)

    def test_host_poll_rate_limits_and_never_registers_control_handlers(self):
        client = self.connect()
        self.messages(client)  # Initial asset delivery is a poll response.
        self.clock += 60_000_000
        first = client.emit("pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, callback=True)
        self.assertTrue(first)
        self.assertEqual(first[0]["ok"] if isinstance(first,list) else first["ok"], True)
        second = client.emit("pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, callback=True)
        status = second[0] if isinstance(second,list) else second
        self.assertFalse(status["ok"])
        self.assertEqual(status["error"], "rate-limited")
        self.clock += 60_000_000
        third = client.emit("pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, callback=True)
        status = third[0] if isinstance(third,list) else third
        self.assertTrue(status["ok"])
        # Flask-SocketIO keeps decorator registrations in a list; the
        # actual namespace->event mapping lives in python-socketio.Server.
        declared = set(self.host.socketio.server.handlers.get(POSE_OBSERVER_NAMESPACE, {}))
        self.assertEqual(declared, {"connect", "disconnect", "pose:poll"})
        self.assertNotIn("arm-drone", declared)
        self.assertNotIn("set-drone-setpoint", declared)

    def test_urlsafe_transport_sid_maps_to_valid_broker_key_without_room_change(self):
        # Socket.IO transport SIDs may begin with '-' or '_', even though
        # our bounded publisher expects a letter/digit in its own ID.
        for sid in ("_weird-client", "-also-legal", "ABC_012"):
            key = self.host._subscriber_key(sid)
            self.assertTrue(key.startswith("viewer-"))
            self.assertEqual(len(key), 47)
            self.assertRegex(key, r"^[a-z][a-z0-9-]+$")
            self.assertEqual(key, self.host._subscriber_key(sid))
        client = self.connect()
        self.assertTrue(client.is_connected(namespace=POSE_OBSERVER_NAMESPACE))
        self.assertEqual(self.host.subscriber_count, 1)
        self.assertEqual(len(self.messages(client)), 1)
        self.host.publish_frame(self.frame(1))
        self.assertEqual(
            [m["payload"]["frame_id"] for m in self.messages(client)],
            [1],
        )
        client.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        self.assertEqual(self.host.subscriber_count, 0)

    def test_explicit_origin_validation_and_secret_minimum(self):
        for invalid in (("*",), ("https://evil.test/path",), ("http://*",)):
            with self.subTest(origin=invalid):
                with self.assertRaises(ValueError):
                    PoseObserverHost(
                        self.registry, viewer_secret=SECRET,
                        allowed_origins=invalid,
                    )
        with self.assertRaisesRegex(ValueError, "viewer secret"):
            PoseObserverHost(self.registry, viewer_secret="short")


if __name__ == "__main__":
    unittest.main()
