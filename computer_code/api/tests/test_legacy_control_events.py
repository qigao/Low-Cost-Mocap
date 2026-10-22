import threading
import unittest
from types import SimpleNamespace

from flask import Flask
from flask_socketio import SocketIO

from legacy_control_events import loopback_only_event, loopback_only_http_route, register_legacy_control_handlers


class RecordingSerial:
    def __init__(self):
        self.writes = []

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)


class FakeCameraState:
    def __init__(self):
        self.drone_armed = [False, False]
        self.to_world_coords_matrix = None
        self.calls = []

    def start_locating_objects(self):
        self.calls.append(("locate", "start"))

    def stop_locating_objects(self):
        self.calls.append(("locate", "stop"))

    def start_trangulating_points(self, camera_poses):
        self.calls.append(("triangulate", "start", camera_poses))

    def stop_trangulating_points(self):
        self.calls.append(("triangulate", "stop"))


class LegacyControlEventIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.socketio = SocketIO(
            self.app, async_mode="threading", logger=False, engineio_logger=False
        )
        self.serial = RecordingSerial()
        self.state = FakeCameraState()
        self.enabled = None
        self.initialized = True
        self.registry = register_legacy_control_handlers(
            self.socketio,
            serial_port=self.serial,
            serial_lock=threading.Lock(),
            get_camera_state=lambda: self.state,
            is_camera_initialized=lambda: self.initialized,
            num_objects=2,
            sleep=lambda _seconds: None,
            enabled_getter=lambda: self.enabled,
        )
        self.clients = []

        @self.socketio.on("protected-camera-config")
        @loopback_only_event(self.socketio)
        def protected_camera_config(data):
            self.state.calls.append(("camera-config", data))

    def tearDown(self):
        for client in self.clients:
            if client.is_connected():
                client.disconnect()

    def make_client(self, remote_addr):
        flask_client = self.app.test_client()
        flask_client.environ_base["REMOTE_ADDR"] = remote_addr
        client = self.socketio.test_client(self.app, flask_test_client=flask_client)
        self.clients.append(client)
        return client

    @staticmethod
    def command_payloads():
        return [
            ("arm-drone", {"droneArmed": [True, False], "count": 1, "currentDroneIndex": 0}),
            ("set-drone-pid", {"dronePID": [0.0] * 17, "droneIndex": 0}),
            ("set-drone-setpoint", {"droneSetpoint": [1.0, 2.0, 3.0], "droneIndex": 0}),
            ("set-drone-trim", {"droneTrim": [0, 1, 2, 3], "droneIndex": 0}),
            ("locate-objects", {"startOrStop": "start"}),
            ("triangulate-points", {
                "startOrStop": "start",
                "cameraPoses": [{"R": [], "t": []}],
                "toWorldCoordsMatrix": [],
            }),
        ]

    def rejection_events(self, client):
        return [event for event in client.get_received() if event["name"] == "legacy-control-rejected"]

    def test_remote_client_cannot_write_or_enable_indirect_tracking_even_with_opt_in(self):
        self.enabled = "1"
        client = self.make_client("192.168.1.20")
        for event, payload in self.command_payloads():
            with self.subTest(event=event):
                client.emit(event, payload)
        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.state.calls, [])
        self.assertEqual(self.state.drone_armed, [False, False])
        self.assertEqual(len(self.rejection_events(client)), 6)

    def test_loopback_is_default_denied_for_direct_and_indirect_control_events(self):
        client = self.make_client("127.0.0.1")
        for event, payload in self.command_payloads():
            with self.subTest(event=event):
                client.emit(event, payload)
        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.state.calls, [])
        self.assertEqual(len(self.rejection_events(client)), 6)

    def test_explicit_loopback_opt_in_allows_valid_commands_and_tracking_controls(self):
        self.enabled = "1"
        client = self.make_client("127.0.0.1")
        for event, payload in self.command_payloads():
            client.emit(event, payload)
        self.assertEqual(len(self.serial.writes), 5)
        self.assertEqual(self.state.drone_armed, [True, False])
        self.assertEqual(self.state.calls[0], ("locate", "start"))
        self.assertEqual(self.state.calls[1], ("triangulate", "start", [{"R": [], "t": []}]))
        self.assertEqual(self.state.to_world_coords_matrix, [])
        self.assertEqual(self.rejection_events(client), [])

    def test_invalid_payloads_and_replayed_arm_counter_do_not_write(self):
        self.enabled = "1"
        client = self.make_client("127.0.0.1")
        arm = {"droneArmed": [True, False], "count": 7, "currentDroneIndex": 0}
        client.emit("arm-drone", arm)
        self.assertEqual(len(self.serial.writes), 2)
        client.emit("arm-drone", arm)
        client.emit("set-drone-pid", {"dronePID": [float("nan")] * 17, "droneIndex": 0})
        client.emit("set-drone-setpoint", {"droneSetpoint": [0.0, 1000.0, 0.0], "droneIndex": 0})
        client.emit("set-drone-trim", {"droneTrim": [0, 1, 2], "droneIndex": 0})
        client.emit("locate-objects", {"startOrStop": "launch"})
        client.emit("triangulate-points", {"startOrStop": "start", "cameraPoses": []})
        self.assertEqual(len(self.serial.writes), 2)
        self.assertEqual(self.state.calls, [])
        self.assertGreaterEqual(len(self.rejection_events(client)), 5)

    def test_arm_does_not_write_before_camera_state_is_initialized(self):
        self.enabled = "1"
        self.initialized = False
        client = self.make_client("127.0.0.1")
        client.emit("arm-drone", {"droneArmed": [True, False], "count": 1, "currentDroneIndex": 0})
        self.assertEqual(self.serial.writes, [])
        self.assertEqual(self.state.drone_armed, [False, False])

    def test_http_camera_stream_guard_rejects_remote_before_side_effect(self):
        app = Flask("camera-stream-guard-test")
        calls = []

        @app.route("/camera-stream")
        @loopback_only_http_route
        def camera_stream():
            calls.append("camera-init")
            return "stream-body"

        remote = app.test_client()
        remote.environ_base["REMOTE_ADDR"] = "192.168.1.20"
        response = remote.get("/camera-stream")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(calls, [])

        local = app.test_client()
        local.environ_base["REMOTE_ADDR"] = "127.0.0.1"
        response = local.get("/camera-stream")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), "stream-body")
        self.assertEqual(calls, ["camera-init"])
    def test_loopback_only_camera_event_rejects_remote_peer(self):
        remote = self.make_client("192.168.1.20")
        remote.emit("protected-camera-config", {"exposure": 100})
        self.assertEqual(self.state.calls, [])
        self.assertEqual(len(self.rejection_events(remote)), 1)

        local = self.make_client("127.0.0.1")
        local.emit("protected-camera-config", {"exposure": 100})
        self.assertEqual(self.state.calls, [("camera-config", {"exposure": 100})])
        self.assertEqual(self.rejection_events(local), [])
    def test_disconnect_releases_bounded_arm_counter_slot(self):
        self.enabled = "1"
        client = self.make_client("127.0.0.1")
        client.emit("arm-drone", {"droneArmed": [True, False], "count": 1, "currentDroneIndex": 0})
        self.assertEqual(len(self.registry), 1)
        client.disconnect()
        self.assertEqual(len(self.registry), 0)


if __name__ == "__main__":
    unittest.main()
