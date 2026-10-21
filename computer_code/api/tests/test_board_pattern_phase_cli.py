"""Check phase availability and strict CLI rejection of unsuitable patterns."""
import contextlib
import io
import json
from pathlib import Path
import unittest

from board_pattern_cli import main


API = Path(__file__).resolve().parents[1]
OLD = API / "board-patterns.example.json"
NEW = API / "board-patterns.continuous.example.json"


def execute(argv: list[str]) -> dict:
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(argv)
    if code != 0:
        raise AssertionError("board_pattern_cli returned nonzero without SystemExit")
    return json.loads(output.getvalue())


class BoardPatternPhaseCliTests(unittest.TestCase):
    def test_legacy_config_exposes_low_visibility_without_masking_it(self):
        report = execute([str(OLD), "--phase-report"])
        coverage = report["observability"]
        self.assertEqual(coverage["all_boards_observable_phases"], [15])
        self.assertFalse(coverage["every_phase_nominally_observable"])
        self.assertFalse(coverage["hardware_visibility_verified"])
        self.assertEqual(coverage["min_global_hamming_distance"], 8)

    def test_continuous_candidate_passes_phase_by_phase_design_qualification(self):
        report = execute([
            str(NEW), "--require-every-phase", "--board", "board_A",
            "--epoch", "1000", "--trigger", "1000",
        ])
        coverage = report["observability"]
        self.assertTrue(coverage["every_phase_nominally_observable"])
        self.assertEqual(len(coverage["all_boards_observable_phases"]), 16)
        self.assertEqual(coverage["min_global_hamming_distance"], 4)
        self.assertEqual(
            sum(report["led_states"].values()), 3
        )
        self.assertEqual(
            coverage["boards"]["board_B"]["worst_illuminated_count"], 3
        )
        self.assertEqual(report["qualification"], "software_config_only")

    def test_strict_flag_rejects_original_codebook(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as ctx:
                main([str(OLD), "--require-every-phase"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("does not support nominal rigid pose", output.getvalue())

    def test_plain_cli_remains_backward_compatible(self):
        report = execute([str(OLD)])
        self.assertNotIn("observability", report)
        self.assertEqual(report["period_triggers"], 16)


if __name__ == "__main__":
    unittest.main()
