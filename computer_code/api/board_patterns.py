"""Data-driven, immutable board geometry and global temporal LED codes.

Positions are board-local metres. An actual metric camera/world calibration
must be established separately; this registry never certifies one.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

import numpy as np

from temporal_pattern import TemporalCodebook, _nonnegative_int


_BOARD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_MARKER_KEY = re.compile(r"(?:0|[1-9][0-9]*)\Z")
_MAX_PERIOD = 256
_MAX_BOARDS = 32
_MAX_MARKERS = 128

# Dimensionless twice-area / longest-edge-squared threshold. This rejects
# nearly collinear three-marker subsets independently of board scale.
_MIN_VISIBLE_TRIANGLE_QUALITY = 1e-6


def _validate_three_marker_subsets(cloud: np.ndarray, board_id: str) -> None:
    """Require every possible three-marker observation to constrain a pose."""
    count = int(cloud.shape[0])
    for first_index in range(count - 2):
        vectors = cloud[first_index + 1:] - cloud[first_index]
        first_squared = np.einsum("ij,ij->i", vectors, vectors)
        for second_index in range(vectors.shape[0] - 1):
            remaining = vectors[second_index + 1:]
            cross = np.cross(vectors[second_index], remaining)
            twice_area = np.linalg.norm(cross, axis=1)
            second_squared = np.einsum("ij,ij->i", remaining, remaining)
            third_vectors = remaining - vectors[second_index]
            third_squared = np.einsum("ij,ij->i", third_vectors, third_vectors)
            longest_edge_squared = np.maximum(
                np.maximum(first_squared[second_index], second_squared),
                third_squared,
            )
            quality = np.divide(
                twice_area,
                longest_edge_squared,
                out=np.zeros_like(twice_area),
                where=longest_edge_squared > 0.0,
            )
            if (
                np.any(longest_edge_squared <= 0.0)
                or not np.isfinite(quality).all()
                or np.any(quality <= _MIN_VISIBLE_TRIANGLE_QUALITY)
            ):
                raise ValueError(
                    f"board {board_id} has a degenerate 3-marker visible subset"
                )




def _strict_object(
    raw: object, name: str, required: set[str], optional: set[str] = frozenset()
) -> Mapping:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be an object")
    keys = set(raw)
    if keys - required - optional or required - keys:
        raise ValueError(f"{name} has missing or unknown fields")
    return raw


def _real_vector(raw: object, count: int, name: str) -> np.ndarray:
    if not isinstance(raw, (list, tuple)) or len(raw) != count:
        raise ValueError(f"{name} must contain exactly {count} real coordinates")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        for value in raw
    ):
        raise ValueError(f"{name} must contain real numeric coordinates")
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must contain representable real coordinates") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real coordinates")
    return values


def _body_transform(raw: object | None) -> tuple[np.ndarray, np.ndarray]:
    if raw is None:
        rotation = np.eye(3, dtype=np.float64)
        translation = np.zeros(3, dtype=np.float64)
    else:
        data = _strict_object(raw, "body_from_board", {"rotation", "translation_m"})
        rows = data["rotation"]
        if not isinstance(rows, (list, tuple)) or len(rows) != 3:
            raise ValueError("body_from_board.rotation must be a real 3x3 matrix")
        rotation = np.stack([
            _real_vector(row, 3, "body_from_board.rotation") for row in rows
        ])
        translation = _real_vector(
            data["translation_m"], 3, "body_from_board.translation_m"
        )
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0)
        ):
            raise ValueError("body_from_board.rotation must be a proper rigid rotation")
    rotation.setflags(write=False)
    translation.setflags(write=False)
    return rotation, translation


def _marker_id(raw: object) -> int:
    if not isinstance(raw, str) or _MARKER_KEY.fullmatch(raw) is None:
        raise ValueError("marker keys must be canonical nonnegative integer strings")
    return _nonnegative_int(int(raw), "marker_id")


@dataclass(frozen=True)
class BoardPattern:
    board_id: str
    board_type: str
    marker_xyz_m: Mapping[int, tuple[float, float, float]]
    min_visible_markers: int
    rotation_body_from_board: np.ndarray
    translation_body_from_board_m: np.ndarray


@dataclass(frozen=True)
class BoardPatternRegistry:
    """One shared, validated codebook across all configured boards."""
    boards: Mapping[str, BoardPattern]
    marker_to_board: Mapping[int, str]
    codebook: TemporalCodebook
    period_triggers: int
    max_bit_errors: int
    pose_phase: int | None
    config_sha256: str

    @classmethod
    def from_mapping(cls, raw: object) -> "BoardPatternRegistry":
        root = _strict_object(
            raw, "board pattern registry",
            {"version", "period_triggers", "max_bit_errors", "boards"},
            {"pose_phase"},
        )
        if type(root["version"]) is not int or root["version"] != 1:
            raise ValueError("board pattern version must be integer 1")
        period = _nonnegative_int(root["period_triggers"], "period_triggers")
        if not 4 <= period <= _MAX_PERIOD:
            raise ValueError("period_triggers must be between 4 and 256")
        errors = _nonnegative_int(root["max_bit_errors"], "max_bit_errors")
        if errors >= period:
            raise ValueError("max_bit_errors must be smaller than period_triggers")
        pose_phase = root.get("pose_phase")
        if pose_phase is not None:
            pose_phase = _nonnegative_int(pose_phase, "pose_phase")
            if pose_phase >= period:
                raise ValueError("pose_phase must be inside the code period")
        raw_boards = root["boards"]
        if not isinstance(raw_boards, Mapping) or not 1 <= len(raw_boards) <= _MAX_BOARDS:
            raise ValueError("boards must contain 1..32 board definitions")

        boards: dict[str, BoardPattern] = {}
        owners: dict[int, str] = {}
        codes: dict[int, tuple[int, ...]] = {}
        for board_id, raw_board in raw_boards.items():
            if not isinstance(board_id, str) or _BOARD_ID.fullmatch(board_id) is None:
                raise ValueError("board_id must be a safe nonempty identifier")
            data = _strict_object(
                raw_board, f"board {board_id}", {"markers"},
                {"type", "min_visible_markers", "body_from_board"},
            )
            board_type = data.get("type", "rigid-led-board")
            if not isinstance(board_type, str) or _BOARD_ID.fullmatch(board_type) is None:
                raise ValueError(f"board {board_id} has invalid type")
            raw_markers = data["markers"]
            if not isinstance(raw_markers, Mapping) or not 3 <= len(raw_markers) <= _MAX_MARKERS:
                raise ValueError(f"board {board_id} needs 3..128 markers")
            min_visible = _nonnegative_int(
                data.get("min_visible_markers", 3), "min_visible_markers"
            )
            if not 3 <= min_visible <= len(raw_markers):
                raise ValueError(f"board {board_id} min_visible_markers must be 3..marker count")

            xyz_map: dict[int, tuple[float, float, float]] = {}
            for raw_id, marker_config in raw_markers.items():
                marker = _marker_id(raw_id)
                if marker in owners:
                    raise ValueError("duplicate global marker_id across boards")
                entry = _strict_object(
                    marker_config, f"marker {marker}", {"xyz_m", "code"}
                )
                xyz = _real_vector(entry["xyz_m"], 3, f"marker {marker}.xyz_m")
                binary = entry["code"]
                if (
                    not isinstance(binary, str) or len(binary) != period
                    or any(bit not in "01" for bit in binary)
                ):
                    raise ValueError(f"marker {marker} code must have {period} binary phases")
                if pose_phase is not None and binary[pose_phase] != "1":
                    raise ValueError(f"marker {marker} must be ON at shared pose_phase")
                xyz_map[marker] = tuple(float(v) for v in xyz)
                codes[marker] = tuple(int(bit) for bit in binary)
                owners[marker] = board_id
                if len(owners) > _MAX_MARKERS:
                    raise ValueError("total marker capacity exceeded")

            cloud = np.asarray(list(xyz_map.values()), dtype=np.float64)
            centered = cloud - np.mean(cloud, axis=0)
            if not np.isfinite(centered).all():
                raise ValueError(f"board {board_id} geometry is unrepresentable")
            singular = np.linalg.svd(centered, compute_uv=False)
            if singular[0] == 0 or singular[1] <= 1e-10 * singular[0]:
                raise ValueError(f"board {board_id} marker geometry is collinear or degenerate")
            if min_visible == 3:
                # The full point cloud may be non-collinear while one
                # three-marker subset is still unusable after occlusion.
                _validate_three_marker_subsets(cloud, board_id)
            R, t = _body_transform(data.get("body_from_board"))
            boards[board_id] = BoardPattern(
                board_id, board_type, MappingProxyType(xyz_map),
                min_visible, R, t,
            )
        codebook = TemporalCodebook(codes, max_bit_errors=errors)
        # Hash validated *semantic* content, not JSON whitespace/order. A
        # controller can echo this fingerprint when a pattern is provisioned;
        # the fingerprint alone is not hardware authenticity or time sync.
        manifest = {
            "version": 1,
            "period_triggers": period,
            "max_bit_errors": errors,
            "pose_phase": pose_phase,
            "boards": {
                board_id: {
                    "type": board.board_type,
                    "min_visible_markers": board.min_visible_markers,
                    "body_from_board": {
                        "rotation": board.rotation_body_from_board.tolist(),
                        "translation_m": board.translation_body_from_board_m.tolist(),
                    },
                    "markers": {
                        str(marker): {
                            "xyz_m": board.marker_xyz_m[marker],
                            "code": "".join(map(str, codes[marker])),
                        }
                        for marker in board.marker_xyz_m
                    },
                }
                for board_id, board in boards.items()
            },
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return cls(
            MappingProxyType(boards), MappingProxyType(owners),
            codebook, period, errors, pose_phase, fingerprint,
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BoardPatternRegistry":
        """Reject repeated JSON keys before decoding any pattern."""
        def distinct_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key {key}")
                result[key] = value
            return result

        def nonfinite(name):
            raise ValueError(f"invalid nonfinite JSON constant {name}")

        data = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=distinct_keys,
            parse_constant=nonfinite,
        )
        return cls.from_mapping(data)

    def board_for_marker(self, marker_id: int) -> BoardPattern:
        marker = _nonnegative_int(marker_id, "marker_id")
        try:
            return self.boards[self.marker_to_board[marker]]
        except KeyError as exc:
            raise ValueError(f"unknown global marker_id {marker}") from exc

    def led_states_at_trigger(
        self, board_id: str, trigger_id: int, *, trigger_epoch: int
    ) -> Mapping[int, bool]:
        """Compute controller bits; does not create a hardware trigger."""
        if board_id not in self.boards:
            raise ValueError("unknown board_id")
        trigger = _nonnegative_int(trigger_id, "trigger_id")
        epoch = _nonnegative_int(trigger_epoch, "trigger_epoch")
        if trigger < epoch:
            raise ValueError("LED trigger precedes shared code epoch")
        phase = (trigger - epoch) % self.period_triggers
        return MappingProxyType({
            marker: bool(self.codebook.patterns[marker][phase])
            for marker in self.boards[board_id].marker_xyz_m
        })
