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
