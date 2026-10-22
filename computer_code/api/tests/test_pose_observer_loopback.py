"""Actual local HTTP/Socket.IO long-polling transport with no hardware IO.

Unlike SocketIO.test_client this starts a real ephemeral 127.0.0.1 TCP
listener and a python-socketio client. It never imports the legacy index
module and does not use the PS3Eye, serial port or any drone control route.
"""
import threading
import time
import unittest
from pathlib import Path

import requests
import socketio
from urllib.request import build_opener, ProxyHandler
from werkzeug.serving import make_server

from board_patterns import BoardPatternRegistry
from board_recognizer import BoardOutcome
from identity_locked_board_tracker import BoardFrameStatus, LockedBoardFrame
from pose_observer_host import PoseObserverHost, POSE_OBSERVER_NAMESPACE
from pose_stream_broker import PoseStreamConsumer


CONFIG = Path(__file__).resolve().parents[1] / "board-patterns.continuous.example.json"
SECRET = "test-only-loopback-readonly-viewer-token"


class LoopbackObserverTransportTests(unittest.TestCase):
    def setUp(self):
        self.registry = BoardPatternRegistry.from_json_file(CONFIG)
        self.host = PoseObserverHost(
            self.registry, viewer_secret=SECRET,
            max_subscribers=2, max_pending_frames=2,
            min_poll_interval_ms=1,
        )
        self.server = make_server(
            "127.0.0.1", 0, self.host.app, threaded=True
        )
        self.port = self.server.server_port
        self.runner = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.runner.start()
        self.clients = []

    def tearDown(self):
        for client in self.clients:
            if client.connected:
                client.disconnect()
        self.server.shutdown()
        self.runner.join(timeout=5)
        self.server.server_close()

    def client(self, secret=SECRET):
        session = requests.Session()
        session.trust_env = False  # localhost tests must bypass CI HTTP_PROXY
        client = socketio.Client(
            reconnection=False, logger=False, engineio_logger=False,
            http_session=session,
        )
        self.clients.append(client)
        client.connect(
            "http://127.0.0.1:" + str(self.port),
            auth={"token": secret},
            namespaces=[POSE_OBSERVER_NAMESPACE],
            transports=["polling"],
            wait_timeout=5,
        )
        return client

    def invalid_frame(self, trigger):
        stamp = 3_000_000 + trigger * 1000
        outcomes = {
            board_id: BoardFrameStatus(
                board_id=board_id,
                state="DISCOVER",
                evidence_kind="INVALID",
                measurement_trigger_id=None,
                measurement_age_triggers=None,
                last_measurement_exposure_timestamp_ns=None,
                measurement_age_ns=None,
                outcome=BoardOutcome(board_id, "not_visible", (), (), None),
            )
            for board_id in self.registry.boards
        }
        return LockedBoardFrame(
            trigger_id=trigger, camera_ids=(2, 5),
            clock_domain="software-synthetic-clock",
            exposure_skew_ns=4,
            latest_exposure_timestamp_ns=stamp,
            pattern_sha256=self.registry.config_sha256,
            calibration_sha256="a"*64,
            boards=outcomes,
            active_locks=(), newly_confirmed_marker_ids=(),
            invalidated_marker_ids=(), measured_marker_ids=(),
        )

    def test_real_local_transport_asset_then_invalid_pose_and_latency(self):
        session = requests.Session()
        session.trust_env = False
        client = socketio.Client(
            reconnection=False, logger=False, engineio_logger=False,
            http_session=session,
        )
        self.clients.append(client)
        origin = "http://127.0.0.1:" + str(self.port)
        with build_opener(ProxyHandler({})).open(
            origin + "/pose-observe/healthz", timeout=4
        ) as response:
            self.assertEqual(response.status, 200)
        client.connect(
            origin,
            headers={"Origin": "http://127.0.0.1:5173"},
            auth={"token": SECRET},
            namespaces=[POSE_OBSERVER_NAMESPACE],
            transports=["polling"],
            wait_timeout=5,
        )
        # Dynamic data is delivered only inside the bounded poll response,
        # never by unbounded asynchronous Socket.IO emits.
        asset_reply = client.call(
            "pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, timeout=5
        )
        self.assertTrue(asset_reply["ok"])
        self.assertEqual(len(asset_reply["messages"]), 1)
        asset_packet = asset_reply["messages"][0]
        self.assertEqual(asset_packet["payload"]["kind"], "asset_description")
        consumer = PoseStreamConsumer()
        consumer.consume(asset_packet)
        software_round_trip_ns = []
        for trigger in range(10, 15):
            started_ns = time.monotonic_ns()
            self.host.publish_frame(self.invalid_frame(trigger))
            time.sleep(0.003)  # obey the explicit 1 ms minimum poll interval
            frame_reply = client.call(
                "pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, timeout=5
            )
            received_ns = time.monotonic_ns()
            self.assertTrue(frame_reply["ok"])
            self.assertEqual(len(frame_reply["messages"]), 1)
            packet = frame_reply["messages"][0]
            elapsed = received_ns - started_ns
            self.assertGreaterEqual(elapsed, 0)
            self.assertLess(elapsed, 5_000_000_000)  # generous CI-only bound
            software_round_trip_ns.append(elapsed)
            frame = consumer.consume(packet)
            self.assertEqual(frame["frame_id"], trigger)
            # The hardware-source timestamp is supplied data, not derived
            # from the software publish/receive clock used for this metric.
            self.assertEqual(
                frame["latest_exposure_timestamp_ns"], 3_000_000 + trigger * 1000
            )
            self.assertFalse(frame["hardware_provenance_verified"])
            self.assertTrue(all(
                body["position_world_from_body"] is None
                and body["evidence_kind"] == "INVALID"
                for body in frame["rigid_bodies"]
            ))
        self.assertEqual(len(software_round_trip_ns), 5)
        self.assertLess(max(software_round_trip_ns), 5_000_000_000)
        # Disconnect the authenticated observer namespace explicitly. With
        # polling transport, a default/global disconnect can return before the
        # namespace disconnect packet is serviced by the threaded test server.
        client.disconnect(namespace=POSE_OBSERVER_NAMESPACE)
        # Wait for server-side ownership cleanup, not just local client state.
        deadline = time.monotonic() + 5.0
        while self.host.subscriber_count and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(self.host.subscriber_count, 0)

    def test_wrong_origin_cannot_connect_with_valid_viewer_token(self):
        session = requests.Session()
        session.trust_env = False
        client = socketio.Client(
            reconnection=False, logger=False, engineio_logger=False,
            http_session=session,
        )
        self.clients.append(client)
        with self.assertRaises(socketio.exceptions.ConnectionError):
            client.connect(
                "http://127.0.0.1:" + str(self.port),
                headers={"Origin": "https://untrusted.example"},
                auth={"token": SECRET},
                namespaces=[POSE_OBSERVER_NAMESPACE],
                transports=["polling"],
                wait_timeout=3,
            )
        self.assertEqual(self.host.subscriber_count, 0)

    def test_wrong_token_cannot_connect_to_real_local_listener(self):
        session = requests.Session()
        session.trust_env = False
        client = socketio.Client(
            reconnection=False, logger=False, engineio_logger=False,
            http_session=session,
        )
        self.clients.append(client)
        with self.assertRaises(socketio.exceptions.ConnectionError):
            client.connect(
                "http://127.0.0.1:" + str(self.port),
                headers={"Origin": "http://127.0.0.1:5173"},
                auth={"token": "bad-token"},
                namespaces=[POSE_OBSERVER_NAMESPACE],
                transports=["polling"],
                wait_timeout=3,
            )
        self.assertEqual(self.host.subscriber_count, 0)


if __name__ == "__main__":
    unittest.main()
