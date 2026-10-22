"""Nominal every-trigger LED availability and rigid-geometry quality gates.

This analyzes *designed* LED illumination, not actually detected spots.
Measured 6DoF still requires real synchronized frame observations, current
identity association, calibrated cameras and minimum 3D reconstruction gates.
A codebook's Hamming distance alone cannot establish per-frame observability.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from board_patterns import BoardPatternRegistry


@dataclass(frozen=True)
class PhaseObservability:
    phase: int
    illuminated_marker_ids: tuple[int, ...]
    illuminated_count: int
    geometric_rank: int
    supports_nominal_rigid_fit: bool


@dataclass(frozen=True)
class BoardObservability:
    board_id: str
    min_required_markers: int
    worst_illuminated_count: int
    missing_pose_phases: tuple[int, ...]
    phases: tuple[PhaseObservability, ...]


@dataclass(frozen=True)
class PatternObservability:
    """Diagnostic coverage of configured phase-by-phase LED geometry."""
    period_triggers: int
    min_global_hamming_distance: int
    guaranteed_erasure_budget: int
    max_consecutive_dark_phases_by_marker: Mapping[int, int]
    boards: Mapping[str, BoardObservability]
    all_boards_observable_phases: tuple[int, ...]
    every_phase_nominally_observable: bool
    hardware_visibility_verified: bool = False


def _spatial_rank(model_xyz: object) -> int:
    xyz = np.asarray(model_xyz, dtype=np.float64)
    if len(xyz) < 2:
        return 0
    centered = xyz - xyz.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular[0] <= 0.0 or not np.isfinite(singular).all():
        return 0
    # Same rank criterion used by rigid_pattern.fit_rigid_pattern.
    return int(np.count_nonzero(singular > singular[0] * 1e-10))


def _max_dark_span_cyclic(code: tuple[int, ...]) -> int:
    if not any(code):
        return len(code)
    longest = 0
    current = 0
    for bit in code + code:
        if bit:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return min(longest, len(code))


def analyze_pattern_observability(
    registry: BoardPatternRegistry,
) -> PatternObservability:
    """Evaluate every board at *every* trigger phase with model geometry.

    Failures are reported, not silently fixed by past-state interpolation.
    A board passes a phase only if at least its configured minimum LEDs are
    illuminated and their model points are not collinear.
    """
    if not isinstance(registry, BoardPatternRegistry):
        raise ValueError("registry must be BoardPatternRegistry")
    period = registry.period_triggers
    patterns = registry.codebook.patterns
    marker_ids = tuple(sorted(patterns))
    if len(marker_ids) < 2:
        raise ValueError("pattern requires multiple LED identities")
    min_distance = min(
        sum(x != y for x, y in zip(patterns[left], patterns[right]))
        for i, left in enumerate(marker_ids)
        for right in marker_ids[i + 1:]
    )
    dark_spans = MappingProxyType({
        marker: _max_dark_span_cyclic(tuple(patterns[marker]))
        for marker in marker_ids
    })
    boards = {}
    for board_id, board in registry.boards.items():
        observations = []
        for phase in range(period):
            illuminated = tuple(
                marker for marker in board.marker_xyz_m
                if patterns[marker][phase] == 1
            )
            rank = _spatial_rank(
                [board.marker_xyz_m[marker] for marker in illuminated]
            )
            valid = len(illuminated) >= board.min_visible_markers and rank >= 2
            observations.append(PhaseObservability(
                phase=phase,
                illuminated_marker_ids=illuminated,
                illuminated_count=len(illuminated),
                geometric_rank=rank,
                supports_nominal_rigid_fit=valid,
            ))
        failures = tuple(
            row.phase for row in observations if not row.supports_nominal_rigid_fit
        )
        boards[board_id] = BoardObservability(
            board_id=board_id,
            min_required_markers=board.min_visible_markers,
            worst_illuminated_count=min(x.illuminated_count for x in observations),
            missing_pose_phases=failures,
            phases=tuple(observations),
        )
    all_full_phases = tuple(
        phase for phase in range(period)
        if all(boards[board_id].phases[phase].supports_nominal_rigid_fit
               for board_id in boards)
    )
    return PatternObservability(
        period_triggers=period,
        min_global_hamming_distance=min_distance,
        guaranteed_erasure_budget=registry.codebook.guaranteed_erasure_budget,
        max_consecutive_dark_phases_by_marker=dark_spans,
        boards=MappingProxyType(boards),
        all_boards_observable_phases=all_full_phases,
        every_phase_nominally_observable=len(all_full_phases) == period,
    )


def require_every_phase_observable(
    registry: BoardPatternRegistry,
) -> PatternObservability:
    """Reject a pattern that can never yield current-frame 6DoF on some phases."""
    report = analyze_pattern_observability(registry)
    if not report.every_phase_nominally_observable:
        shortfalls = ", ".join(
            f"{board_id}:{list(row.missing_pose_phases)}"
            for board_id, row in report.boards.items()
            if row.missing_pose_phases
        )
        raise ValueError(
            "board pattern does not support nominal rigid pose on every "
            f"trigger phase ({shortfalls})"
        )
    return report


def observability_to_dict(report: PatternObservability) -> dict:
    """Stable JSON-compatible summary with explicit synthetic-only provenance."""
    if not isinstance(report, PatternObservability):
        raise ValueError("report must be PatternObservability")
    return {
        "period_triggers": report.period_triggers,
        "min_global_hamming_distance": report.min_global_hamming_distance,
        "guaranteed_erasure_budget": report.guaranteed_erasure_budget,
        "max_consecutive_dark_phases_by_marker": {
            str(marker): span for marker, span
            in report.max_consecutive_dark_phases_by_marker.items()
        },
        "all_boards_observable_phases": list(report.all_boards_observable_phases),
        "every_phase_nominally_observable": report.every_phase_nominally_observable,
        "hardware_visibility_verified": False,
        "boards": {
            board_id: {
                "min_required_markers": row.min_required_markers,
                "worst_illuminated_count": row.worst_illuminated_count,
                "missing_pose_phases": list(row.missing_pose_phases),
                "phases": [
                    {
                        "phase": phase.phase,
                        "illuminated_marker_ids": list(
                            phase.illuminated_marker_ids
                        ),
                        "illuminated_count": phase.illuminated_count,
                        "geometric_rank": phase.geometric_rank,
                        "supports_nominal_rigid_fit":
                            phase.supports_nominal_rigid_fit,
                    }
                    for phase in row.phases
                ],
            }
            for board_id, row in report.boards.items()
        },
    }
