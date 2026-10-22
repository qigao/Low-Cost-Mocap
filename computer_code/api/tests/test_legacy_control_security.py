import unittest

from legacy_control_security import (
    ArmCounterRegistry,
    DEFAULT_ALLOWED_ORIGINS,
    configured_origins,
    legacy_control_peer_allowed,
    is_loopback_peer,
    validate_arm_payload,
    validate_pid_payload,
    validate_setpoint_payload,
    validate_trim_payload,
)


class LegacyControlSecurityTests(unittest.TestCase):
    def test_loopback_peer_policy_handles_remote_and_mapped_addresses(self):
        self.assertTrue(is_loopback_peer("127.0.0.1"))
        self.assertTrue(is_loopback_peer("::1"))
        self.assertTrue(is_loopback_peer("::ffff:127.0.0.1"))
        self.assertFalse(is_loopback_peer("192.168.1.20"))
        self.assertFalse(is_loopback_peer(None))
        self.assertFalse(is_loopback_peer("localhost"))

    def test_camera_stream_guards_before_initializing_hardware(self):
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "index.py").read_text(encoding="utf-8")
        start = source.index('@app.route("/api/camera-stream")')
        end = source.index('@app.route("/api/trajectory-planning")', start + 1)
        handler = source[start:end]
        self.assertIn("is_loopback_peer(request.remote_addr)", handler)
        self.assertLess(handler.index("is_loopback_peer(request.remote_addr)"), handler.index("Cameras.instance()"))
    def test_control_is_disabled_without_explicit_opt_in(self):
        for address in ("127.0.0.1", "::1", "localhost", "192.168.1.20", None):
            with self.subTest(address=address):
                self.assertFalse(legacy_control_peer_allowed(address, None))
                self.assertFalse(legacy_control_peer_allowed(address, "0"))

    def test_control_opt_in_still_requires_loopback_peer(self):
        self.assertTrue(legacy_control_peer_allowed("127.0.0.1", "1"))
        self.assertTrue(legacy_control_peer_allowed("::1", "1"))
        self.assertFalse(legacy_control_peer_allowed("192.168.1.20", "1"))
        self.assertFalse(legacy_control_peer_allowed("localhost", "1"))
        self.assertFalse(legacy_control_peer_allowed(None, "1"))

    def test_default_origins_are_explicit_local_origins(self):
        self.assertEqual(configured_origins(None), list(DEFAULT_ALLOWED_ORIGINS))
        self.assertNotIn("*", configured_origins(None))

    def test_explicit_origin_allowlist_is_normalized_and_deduplicated(self):
        self.assertEqual(
            configured_origins("https://mocap.example, http://localhost:3000,https://mocap.example"),
            ["https://mocap.example", "http://localhost:3000"],
        )

    def test_legacy_serial_handlers_and_server_defaults_use_guard(self):
        from pathlib import Path

        api_dir = Path(__file__).resolve().parents[1]
        index_source = (api_dir / "index.py").read_text(encoding="utf-8")
        event_source = (api_dir / "legacy_control_events.py").read_text(encoding="utf-8")
        for event in (
            "arm-drone", "set-drone-pid", "set-drone-setpoint", "set-drone-trim",
            "locate-objects", "triangulate-points",
        ):
            with self.subTest(event=event):
                start = event_source.index(f'@socketio.on("{event}")')
                next_handler = event_source.find("\n    @socketio.on(", start + 1)
                handler = event_source[start: next_handler if next_handler >= 0 else len(event_source)]
                self.assertIn("_legacy_control_authorized()", handler)
        self.assertIn('os.environ.get("MOCAP_BIND_HOST", "127.0.0.1")', index_source)
        self.assertIn("debug=False, use_reloader=False", index_source)
        self.assertNotIn("cors_allowed_origins='*'", index_source)
        self.assertIn("register_legacy_control_handlers(", index_source)
        self.assertIn("arm_counter_registry.accept(request.sid, count)", event_source)
        self.assertIn('@socketio.on("disconnect")', event_source)
        stateful_events = {
            "acquire-floor": "acquire_floor",
            "set-origin": "set_origin",
            "update-camera-settings": "change_camera_settings",
            "capture-points": "capture_points",
            "calculate-camera-pose": "calculate_camera_pose",
            "determine-scale": "determine_scale",
        }
        for event, function_name in stateful_events.items():
            with self.subTest(event=event):
                start = index_source.index(f'@socketio.on("{event}")')
                end = index_source.index(f"def {function_name}(", start)
                self.assertIn("@loopback_only_event(socketio)", index_source[start:end])

    def test_arm_payload_requires_exact_boolean_vector_and_metadata(self):
        self.assertEqual(
            validate_arm_payload({
                "droneArmed": [False, True],
                "count": 7,
                "currentDroneIndex": 1,
            }),
            ([False, True], 7, 1),
        )
        invalid = [
            {"droneArmed": [0, 1], "count": 0, "currentDroneIndex": 0},
            {"droneArmed": [True], "count": 0, "currentDroneIndex": 0},
            {"droneArmed": [True, False], "count": True, "currentDroneIndex": 0},
            {"droneArmed": [True, False], "count": 9007199254740992, "currentDroneIndex": 0},
            {"droneArmed": [True, False], "count": 0, "currentDroneIndex": 2},
            {"droneArmed": [True, False], "count": 0, "currentDroneIndex": 0, "extra": 1},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_arm_payload(payload)

    def test_arm_counter_registry_rejects_replays_and_bounds_clients(self):
        registry = ArmCounterRegistry(max_clients=2)
        self.assertTrue(registry.accept("sid-a", 0))
        self.assertTrue(registry.accept("sid-a", 1))
        self.assertFalse(registry.accept("sid-a", 1))
        self.assertFalse(registry.accept("sid-a", 0))
        self.assertFalse(registry.accept("sid-a", True))
        self.assertTrue(registry.accept("sid-b", 5))
        self.assertFalse(registry.accept("sid-c", 0))
        self.assertEqual(len(registry), 2)
        registry.discard("sid-a")
        self.assertTrue(registry.accept("sid-c", 0))
        self.assertEqual(len(registry), 2)

    def test_arm_counter_registry_rejects_invalid_client_ids(self):
        registry = ArmCounterRegistry()
        for client_id in (None, "", 3, True):
            with self.subTest(client_id=client_id):
                self.assertFalse(registry.accept(client_id, 0))
        with self.assertRaises(ValueError):
            ArmCounterRegistry(max_clients=0)

    def test_pid_payload_rejects_bad_shape_nonfinite_and_out_of_range_values(self):
        valid = ["1", "0", "0", "1.5", "0", "0", "0.3", "0.1", "0.05", "0.2", "0.03", "0.05", "0.3", "0.1", "0.05", "28", "-0.035"]
        index, values = validate_pid_payload({"dronePID": valid, "droneIndex": 0})
        self.assertEqual(index, 0)
        self.assertEqual(len(values), 17)
        for values_in in (
            valid[:-1],
            valid[:-1] + ["nan"],
            valid[:-1] + ["inf"],
            valid[:-1] + ["1001"],
            valid[:-1] + [True],
        ):
            with self.subTest(values=values_in):
                with self.assertRaises(ValueError):
                    validate_pid_payload({"dronePID": values_in, "droneIndex": 0})
        with self.assertRaises(ValueError):
            validate_pid_payload({"dronePID": valid, "droneIndex": True})

    def test_setpoint_payload_requires_three_finite_bounded_coordinates(self):
        self.assertEqual(
            validate_setpoint_payload({"droneSetpoint": ["0", "1.25", "-2"], "droneIndex": 1}),
            (1, [0.0, 1.25, -2.0]),
        )
        for coordinates in ([0, 1], [0, 1, float("nan")], [0, 1, float("inf")], [0, 1, 100.01], [True, 1, 2]):
            with self.subTest(coordinates=coordinates):
                with self.assertRaises(ValueError):
                    validate_setpoint_payload({"droneSetpoint": coordinates, "droneIndex": 0})

    def test_trim_payload_requires_four_bounded_integers(self):
        self.assertEqual(
            validate_trim_payload({"droneTrim": ["0", "1", "-2", "3"], "droneIndex": 0}),
            (0, [0, 1, -2, 3]),
        )
        for trims in ([0, 1, 2], [0, 1, 2, 1.5], [0, 1, 2, 1001], [0, 1, 2, float("nan")], [0, 1, 2, True]):
            with self.subTest(trims=trims):
                with self.assertRaises(ValueError):
                    validate_trim_payload({"droneTrim": trims, "droneIndex": 0})

    def test_handlers_validate_payload_before_writing_serial(self):
        from pathlib import Path

        event_source = (Path(__file__).resolve().parents[1] / "legacy_control_events.py").read_text(encoding="utf-8")
        checks = {
            "arm-drone": "validate_arm_payload",
            "set-drone-pid": "validate_pid_payload",
            "set-drone-setpoint": "validate_setpoint_payload",
            "set-drone-trim": "validate_trim_payload",
        }
        for event, validator in checks.items():
            with self.subTest(event=event):
                start = event_source.index(f'@socketio.on("{event}")')
                next_handler = event_source.find("\n    @socketio.on(", start + 1)
                handler = event_source[start: next_handler if next_handler >= 0 else len(event_source)]
                self.assertIn(validator, handler)
                self.assertLess(handler.index(validator), handler.index("serial_port.write"))

    def test_wildcard_and_non_origin_urls_are_rejected(self):
        for value in (
            "*",
            "",
            "https://example.test/path",
            "https://example.test/?q=1",
            "ftp://example.test",
            "https://user:password@example.test",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    configured_origins(value)


if __name__ == "__main__":
    unittest.main()
