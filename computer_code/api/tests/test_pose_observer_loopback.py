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
        received = []
        asset_seen = threading.Event()
        frame_seen = threading.Event()
        session = requests.Session()
        session.trust_env = False
        client = socketio.Client(
            reconnection=False, logger=False, engineio_logger=False,
            http_session=session,
        )
        self.clients.append(client)

        @client.on("mocap:pose-stream", namespace=POSE_OBSERVER_NAMESPACE)
        def on_data(packet):
            received.append((time.monotonic_ns(), packet))
            kind = packet["payload"]["kind"]
            if kind == "asset_description":
                asset_seen.set()
            elif kind == "pose_frame":
                frame_seen.set()

        origin = "http://127.0.0.1:" + str(self.port)
        with build_opener(ProxyHandler({})).open(
            origin + "/pose-observe/healthz", timeout=4
        ) as response:
            self.assertEqual(response.status, 200)
        client.connect(
            origin,
            auth={"token": SECRET},
            namespaces=[POSE_OBSERVER_NAMESPACE],
            transports=["polling"],
            wait_timeout=5,
        )
        self.assertTrue(asset_seen.wait(5), "asset was not delivered via HTTP polling")
        self.assertEqual(received[0][1]["payload"]["kind"], "asset_description")
        started_ns = time.monotonic_ns()
        self.host.publish_frame(self.invalid_frame(10))
        client.emit("pose:poll", {}, namespace=POSE_OBSERVER_NAMESPACE, callback=True)
        self.assertTrue(frame_seen.wait(5), "pose frame was not delivered via HTTP polling")
        self.assertEqual(len(received), 2)
        elapsed = received[1][0] - started_ns
        self.assertGreaterEqual(elapsed, 0)
        self.assertLess(elapsed, 5_000_000_000)  # generous CI-only bound
        consumer = PoseStreamConsumer()
        consumer.consume(received[0][1])
        frame = consumer.consume(received[1][1])
        self.assertEqual(frame["frame_id"], 10)
        self.assertFalse(frame["hardware_provenance_verified"])
        self.assertTrue(all(
            body["position_world_from_body"] is None
            and body["evidence_kind"] == "INVALID"
            for body in frame["rigid_bodies"]
        ))
        client.disconnect()
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
                auth={"token": "bad-token"},
                namespaces=[POSE_OBSERVER_NAMESPACE],
                transports=["polling"],
                wait_timeout=3,
            )
        self.assertEqual(self.host.subscriber_count, 0)


if __name__ == "__main__":
    unittest.main()
