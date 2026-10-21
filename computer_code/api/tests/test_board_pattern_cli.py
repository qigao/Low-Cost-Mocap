"""Offline board-pattern utility tests; no controller or camera device needed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from board_pattern_cli import main, describe_patterns
from board_patterns import BoardPatternRegistry


EXAMPLE = Path(__file__).resolve().parents[1] / "board-patterns.example.json"


class BoardPatternCliTests(unittest.TestCase):
    def test_full_manifest_and_fingerprint_match_registry(self):
        registry = BoardPatternRegistry.from_json_file(EXAMPLE)
        expected = describe_patterns(registry)
        self.assertFalse(expected["hardware_provenance_verified"])
        self.assertEqual(len(expected["config_sha256"]), 64)
        self.assertEqual(set(expected["boards"]), {"board_A", "board_B"})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([str(EXAMPLE)]), 0)
        self.assertEqual(json.loads(output.getvalue()), expected)

    def test_controller_states_and_config_revision_are_explicit(self):
        registry = BoardPatternRegistry.from_json_file(EXAMPLE)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main([
                str(EXAMPLE), "--board", "board_B",
                "--trigger", "1015", "--epoch", "1000",
                "--expect-sha256", registry.config_sha256,
            ]), 0)
        data = json.loads(output.getvalue())
        self.assertEqual(data["selected_trigger"], 1015)
        self.assertEqual(set(data["led_states"]), {"104", "105", "106", "107"})
        self.assertTrue(all(data["led_states"].values()))
        self.assertEqual(data["config_sha256"], registry.config_sha256)
        self.assertEqual(data["qualification"], "software_config_only")

    def test_fingerprint_is_deterministic_for_equivalent_config_order(self):
        raw = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        registry = BoardPatternRegistry.from_mapping(raw)
        raw["boards"] = dict(reversed(list(raw["boards"].items())))
        for board in raw["boards"].values():
            board["markers"] = dict(reversed(list(board["markers"].items())))
        same = BoardPatternRegistry.from_mapping(raw)
        self.assertEqual(same.config_sha256, registry.config_sha256)
        raw["boards"]["board_A"]["markers"]["100"]["xyz_m"][0] += 0.002
        new = BoardPatternRegistry.from_mapping(raw)
        self.assertNotEqual(new.config_sha256, registry.config_sha256)

    def test_mismatched_revision_or_partial_cli_state_request_fails_closed(self):
        for arguments in (
            ["--board", "board_A"],
            ["--board", "board_A", "--trigger", "1015"],
            ["--expect-sha256", "0" * 64],
            ["--board", "missing", "--trigger", "1015", "--epoch", "1000"],
            ["--board", "board_A", "--trigger", "999", "--epoch", "1000"],
        ):
            with self.subTest(arguments=arguments):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        main([str(EXAMPLE), *arguments])
                self.assertEqual(raised.exception.code, 2)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "invalid.json"
            target.write_text('{"version": 1, "version": 1}', encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main([str(target)])
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
