"""Validate board configurations and emit reproducible LED pattern tables.

Examples (software-only, no LED hardware or exposure synchronization):
  python board_pattern_cli.py board-patterns.example.json
  python board_pattern_cli.py board-patterns.example.json --board board_A --epoch 1000 --trigger 1015
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from board_patterns import BoardPatternRegistry


def describe_patterns(registry: BoardPatternRegistry) -> dict:
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    return {
        "schema_version": 1,
        "qualification": "software_config_only",
        "hardware_provenance_verified": False,
        "config_sha256": registry.config_sha256,
        "period_triggers": registry.period_triggers,
        "max_bit_errors": registry.max_bit_errors,
        "pose_phase": registry.pose_phase,
        "boards": {
            board_id: {
                "type": board.board_type,
                "min_visible_markers": board.min_visible_markers,
                "markers": {
                    str(marker): {
                        "xyz_m": list(board.marker_xyz_m[marker]),
                        "code": "".join(map(str, registry.codebook.patterns[marker])),
                    }
                    for marker in board.marker_xyz_m
                },
            }
            for board_id, board in registry.boards.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Version-1 board-pattern JSON")
    parser.add_argument("--board", help="Board to generate one trigger's LED states")
    parser.add_argument("--trigger", type=int, help="Actual common hardware trigger ID")
    parser.add_argument("--epoch", type=int, help="Known shared LED code epoch")
    parser.add_argument("--expect-sha256", help="Reject if expected pattern revision differs")
    args = parser.parse_args(argv)
    flags = (args.board is not None, args.trigger is not None, args.epoch is not None)
    if any(flags) and not all(flags):
        parser.error("--board, --trigger and --epoch must be supplied together")
    try:
        registry = BoardPatternRegistry.from_json_file(args.config)
        if args.expect_sha256 is not None and args.expect_sha256 != registry.config_sha256:
            raise ValueError("board pattern configuration SHA-256 mismatch")
        output = describe_patterns(registry)
        if all(flags):
            output["selected_trigger"] = args.trigger
            output["selected_board"] = args.board
            output["led_states"] = {
                str(marker): state
                for marker, state in registry.led_states_at_trigger(
                    args.board, args.trigger, trigger_epoch=args.epoch
                ).items()
            }
    except (ValueError, TypeError, OSError) as exc:
        parser.error(f"invalid board pattern: {exc}")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
