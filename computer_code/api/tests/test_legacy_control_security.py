import unittest

from legacy_control_security import (
    DEFAULT_ALLOWED_ORIGINS,
    configured_origins,
    legacy_control_peer_allowed,
    validate_arm_payload,
    validate_pid_payload,
    validate_setpoint_payload,
    validate_trim_payload,
)


class LegacyControlSecurityTests(unittest.TestCase):
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

        index_source = (Path(__file__).resolve().parents[1] / "index.py").read_text(encoding="utf-8")
        for event in ("arm-drone", "set-drone-pid", "set-drone-setpoint", "set-drone-trim"):
            with self.subTest(event=event):
                start = index_source.index(f'@socketio.on("{event}")')
                next_handler = index_source.find("\n@socketio.on(", start + 1)
                handler = index_source[start: next_handler if next_handler >= 0 else len(index_source)]
                self.assertIn("_legacy_control_authorized()", handler)
        self.assertIn('os.environ.get("MOCAP_BIND_HOST", "127.0.0.1")', index_source)
        self.assertIn("debug=False, use_reloader=False", index_source)
        self.assertNotIn("cors_allowed_origins='*'", index_source)

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
            {"droneArmed": [True, False], "count": 0, "currentDroneIndex": 2},
            {"droneArmed": [True, False], "count": 0, "currentDroneIndex": 0, "extra": 1},
        ]
        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_arm_payload(payload)

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

        index_source = (Path(__file__).resolve().parents[1] / "index.py").read_text(encoding="utf-8")
        checks = {
            "arm-drone": "validate_arm_payload",
            "set-drone-pid": "validate_pid_payload",
            "set-drone-setpoint": "validate_setpoint_payload",
            "set-drone-trim": "validate_trim_payload",
        }
        for event, validator in checks.items():
            with self.subTest(event=event):
                start = index_source.index(f'@socketio.on("{event}")')
                next_handler = index_source.find("\\n@socketio.on(", start + 1)
                handler = index_source[start: next_handler if next_handler >= 0 else len(index_source)]
                self.assertIn(validator, handler)
                self.assertLess(handler.index(validator), handler.index("ser.write"))

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
