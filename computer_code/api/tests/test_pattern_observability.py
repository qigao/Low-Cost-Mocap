"""Nominal continuous LED visibility is a separate codebook acceptance gate."""
import copy
import json
from pathlib import Path
import unittest

from board_patterns import BoardPatternRegistry
from pattern_observability import (
    analyze_pattern_observability,
    observability_to_dict,
    require_every_phase_observable,
)


API = Path(__file__).resolve().parents[1]
LEGACY = API / "board-patterns.example.json"
CONTINUOUS = API / "board-patterns.continuous.example.json"


class PatternObservabilityTests(unittest.TestCase):
    def test_existing_balanced_codes_only_offer_all_board_pose_at_phase_15(self):
        registry = BoardPatternRegistry.from_json_file(LEGACY)
        summary = analyze_pattern_observability(registry)
        self.assertEqual(summary.all_boards_observable_phases, (15,))
        self.assertFalse(summary.every_phase_nominally_observable)
        self.assertEqual(summary.min_global_hamming_distance, 8)
        self.assertFalse(summary.hardware_visibility_verified)
        self.assertEqual(summary.boards["board_A"].worst_illuminated_count, 1)
        self.assertEqual(summary.boards["board_B"].worst_illuminated_count, 1)
        with self.assertRaisesRegex(ValueError, "every trigger phase"):
            require_every_phase_observable(registry)

    def test_high_visibility_codes_support_three_noncollinear_leds_every_phase(self):
        registry = BoardPatternRegistry.from_json_file(CONTINUOUS)
        report = require_every_phase_observable(registry)
        self.assertTrue(report.every_phase_nominally_observable)
        self.assertEqual(report.all_boards_observable_phases, tuple(range(16)))
        self.assertEqual(report.min_global_hamming_distance, 4)
        for board in report.boards.values():
            self.assertEqual(board.missing_pose_phases, ())
            self.assertEqual(board.worst_illuminated_count, 3)
            self.assertEqual(
                [x.illuminated_count for x in board.phases],
                [3] * 12 + [4] * 4,
            )
            self.assertTrue(all(x.geometric_rank >= 2 for x in board.phases))
        self.assertEqual(
            max(report.max_consecutive_dark_phases_by_marker.values()), 2
        )
        self.assertFalse(report.hardware_visibility_verified)
        info = observability_to_dict(report)
        self.assertEqual(info["all_boards_observable_phases"], list(range(16)))
        self.assertFalse(info["hardware_visibility_verified"])

    def test_registry_rejects_collinear_three_marker_subset_before_phase_analysis(self):
        config = json.loads(CONTINUOUS.read_text(encoding="utf-8"))
        # The full cloud remains non-collinear thanks to raised LED 103, but
        # an occlusion could leave only the collinear 100/101/102 triple.
        config["boards"]["board_A"]["markers"]["102"]["xyz_m"] = [0.28, 0, 0]
        with self.assertRaisesRegex(ValueError, "degenerate 3-marker visible subset"):
            BoardPatternRegistry.from_mapping(config)

    def test_code_phase_with_too_few_lit_markers_is_not_pose_observable(self):
        config = json.loads(CONTINUOUS.read_text(encoding="utf-8"))
        # Remove one of the three otherwise-lit markers at phase 3. The
        # codebook remains valid, but the board has only two illuminated LEDs.
        code = config["boards"]["board_A"]["markers"]["100"]["code"]
        config["boards"]["board_A"]["markers"]["100"]["code"] = (
            code[:3] + "0" + code[4:]
        )
        registry = BoardPatternRegistry.from_mapping(config)
        report = analyze_pattern_observability(registry)
        phase3 = report.boards["board_A"].phases[3]
        self.assertEqual(phase3.illuminated_count, 2)
        self.assertFalse(phase3.supports_nominal_rigid_fit)
        self.assertIn(3, report.boards["board_A"].missing_pose_phases)
        self.assertNotIn(3, report.all_boards_observable_phases)
        with self.assertRaisesRegex(ValueError, "every trigger phase"):
            require_every_phase_observable(registry)

    def test_missing_rois_and_real_hardware_remain_independent_gates(self):
        registry = BoardPatternRegistry.from_json_file(CONTINUOUS)
        report = analyze_pattern_observability(registry)
        self.assertTrue(report.every_phase_nominally_observable)
        # This is a code schedule guarantee *only*: it does not attest
        # physical visibility, camera timestamps, shutter or scale.
        self.assertFalse(report.hardware_visibility_verified)
        for board in report.boards.values():
            self.assertEqual(len(board.phases), registry.period_triggers)
            for phase in board.phases:
                self.assertLessEqual(
                    phase.illuminated_count,
                    len(registry.boards[board.board_id].marker_xyz_m),
                )
        with self.assertRaisesRegex(ValueError, "BoardPatternRegistry"):
            analyze_pattern_observability(None)
        with self.assertRaisesRegex(ValueError, "PatternObservability"):
            observability_to_dict(None)


if __name__ == "__main__":
    unittest.main()
