import unittest

from legacy_control_security import (
    DEFAULT_ALLOWED_ORIGINS,
    configured_origins,
    legacy_control_peer_allowed,
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
