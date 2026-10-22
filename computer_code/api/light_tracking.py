"""Bounded fail-closed per-camera light tracks between trigger exposures.

Track IDs are local association handles only, NOT physical marker identities.
Temporal codes must be decoded independently after hardware trigger admission.
Absence of a bright detection is NOT an illuminated=0 observation; predicted
ROIs require an independent motion/brightness quality gate before sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry import _observation_array
from temporal_pattern import _nonnegative_int


def _positive_float(value: object, name: str, *, zero_allowed: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative real scalar") from exc
    if not np.isfinite(result) or result < 0 or (result == 0 and not zero_allowed):
        raise ValueError(f"{name} must be a finite nonnegative real scalar")
    return result


@dataclass(frozen=True)
class TrackEvidence:
    track_id: int
    camera_id: int
    trigger_id: int
    predicted_xy: tuple[float, float]
    observed_xy: tuple[float, float] | None
    missed_triggers: int


@dataclass
class _TrackState:
    last_xy: np.ndarray
    velocity_per_trigger: np.ndarray
    last_observed_trigger: int
    observation_count: int


class BoundedLightTracker:
    """Transactionally update camera-local tracks with strict uniqueness.

    Each frame must already have an admissible shared hardware trigger.
    The assignment maximizes gated matches then minimizes predicted distance.
    Close alternative assignments, duplicate detections, invalid values, and
    capacity exhaustion raise ValueError without changing internal state.
    """

    def __init__(
        self,
        *,
        camera_id: int,
        max_tracks: int = 16,
        max_motion_px_per_trigger: float = 8.0,
        max_missing_triggers: int = 4,
        ambiguity_margin_fraction: float = 0.05,
        minimum_detection_separation_px: float = 0.5,
        allow_trigger_gaps: bool = False,
    ) -> None:
        self.camera_id = _nonnegative_int(camera_id, "camera_id")
        self.max_tracks = _nonnegative_int(max_tracks, "max_tracks")
        self.max_missing_triggers = _nonnegative_int(
            max_missing_triggers, "max_missing_triggers"
        )
        self.max_motion = _positive_float(
            max_motion_px_per_trigger, "max_motion_px_per_trigger"
        )
        self.ambiguity_margin = _positive_float(
            ambiguity_margin_fraction, "ambiguity_margin_fraction", zero_allowed=True
        )
        self.min_separation = _positive_float(
            minimum_detection_separation_px,
            "minimum_detection_separation_px", zero_allowed=True,
        )
        if self.max_tracks < 1:
            raise ValueError("max_tracks must be at least one")
        if self.ambiguity_margin >= 1:
            raise ValueError("ambiguity_margin_fraction must be less than one")
        if not isinstance(allow_trigger_gaps, (bool, np.bool_)):
            raise ValueError("allow_trigger_gaps must be boolean")
        self.allow_trigger_gaps = bool(allow_trigger_gaps)
        self._tracks: dict[int, _TrackState] = {}
        self._previous_trigger: int | None = None
        self._next_track_id = 0

    @property
    def last_trigger_id(self) -> int | None:
        return self._previous_trigger

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def staged_copy(self) -> "BoundedLightTracker":
        """Clone bounded numerical state for atomic multi-camera admission.

        No image buffers or hardware handles are retained. The coordinator
        publishes this clone only after *all* camera and geometry gates pass.
        """
        clone = BoundedLightTracker(
            camera_id=self.camera_id,
            max_tracks=self.max_tracks,
            max_motion_px_per_trigger=self.max_motion,
            max_missing_triggers=self.max_missing_triggers,
            ambiguity_margin_fraction=self.ambiguity_margin,
            minimum_detection_separation_px=self.min_separation,
            allow_trigger_gaps=self.allow_trigger_gaps,
        )
        clone._tracks = {
            ident: _TrackState(
                last_xy=state.last_xy.copy(),
                velocity_per_trigger=state.velocity_per_trigger.copy(),
                last_observed_trigger=state.last_observed_trigger,
                observation_count=state.observation_count,
            )
            for ident, state in self._tracks.items()
        }
        clone._previous_trigger = self._previous_trigger
        clone._next_track_id = self._next_track_id
        return clone

    def update(
        self,
        trigger_id: int,
        detections_xy: Sequence[Any],
        *,
        forbidden_track_ids: Sequence[int] = (),
    ) -> tuple[TrackEvidence, ...]:
        """Associate centroids to local tracks, preserving off-frame erasures.

        The caller must validate actual camera trigger metadata before this
        method. It cannot infer physical exposure synchronization from frames.
        """
        trigger = _nonnegative_int(trigger_id, "trigger_id")
        if self._previous_trigger is not None:
            if trigger <= self._previous_trigger:
                raise ValueError("stale or repeated tracker trigger_id")
            if not self.allow_trigger_gaps and trigger != self._previous_trigger + 1:
                raise ValueError("missing tracker trigger frame")

        forbidden = tuple(
            _nonnegative_int(ident, "forbidden track ID")
            for ident in forbidden_track_ids
        )
        if len(set(forbidden)) != len(forbidden):
            raise ValueError("forbidden track IDs must be unique")
        forbidden_set = set(forbidden)
        positions = [
            _observation_array(xy, self.camera_id) for xy in detections_xy
        ]
        if len(positions) > self.max_tracks:
            raise ValueError("detected light capacity exceeded")
        for i, first in enumerate(positions):
            for second in positions[i + 1:]:
                if np.linalg.norm(first - second) <= self.min_separation:
                    raise ValueError("duplicate or overlapping light detections")

        # Work on copies. Every failure below leaves committed tracks and
        # last-trigger bookkeeping unchanged.
        staged = {
            ident: _TrackState(
                state.last_xy.copy(), state.velocity_per_trigger.copy(),
                state.last_observed_trigger, state.observation_count,
            )
            for ident, state in self._tracks.items()
            if trigger - state.last_observed_trigger <= self.max_missing_triggers + 1
        }
        identifiers = sorted(staged)
        predictions = []
        gates = []
        for ident in identifiers:
            state = staged[ident]
            gap = trigger - state.last_observed_trigger
            predicted = state.last_xy + gap * state.velocity_per_trigger
            if not np.isfinite(predicted).all():
                raise ValueError("light-track prediction is not finite")
            predictions.append(predicted)
            gates.append(self.max_motion * gap)
        track_count = len(identifiers)
        detection_count = len(positions)
        penalty = float(track_count + 1)
        invalid = 2.0 * penalty
        costs = np.full(
            (track_count, detection_count + track_count),
            penalty, dtype=np.float64,
        )
        for row, (predicted, gate) in enumerate(zip(predictions, gates)):
            # A validated temporal code saying OFF forbids the track from
            # consuming any currently bright detection. Otherwise one LED
            # can steal another's global identity during an OFF phase.
            if identifiers[row] in forbidden_set:
                continue
            for col, xy in enumerate(positions):
                distance = float(np.linalg.norm(xy - predicted))
                if np.isfinite(distance) and distance <= gate:
                    costs[row, col] = distance / gate
                else:
                    costs[row, col] = invalid
        rows, cols = linear_sum_assignment(costs)
        selected = [
            (int(row), int(col))
            for row, col in zip(rows, cols)
            if col < detection_count and costs[row, col] < penalty
        ]
        baseline_count = len(selected)
        baseline_cost = float(costs[rows, cols].sum())

        # A seemingly unique Hungarian output can still be an arbitrary
        # tie when lights cross. Forbid each selected real edge in turn and
        # reject any similarly good solution with the same match cardinality.
        for row, col in selected:
            alternative = costs.copy()
            alternative[row, col] = invalid
            alt_rows, alt_cols = linear_sum_assignment(alternative)
            alt_count = sum(
                int(c < detection_count and alternative[r, c] < penalty)
                for r, c in zip(alt_rows, alt_cols)
            )
            if (
                alt_count == baseline_count
                and float(alternative[alt_rows, alt_cols].sum())
                <= baseline_cost + self.ambiguity_margin * max(1, baseline_count)
            ):
                raise ValueError("ambiguous light-track association")

        matched_cols = {col for _, col in selected}
        matched_rows = {row for row, _ in selected}
        births = [col for col in range(detection_count) if col not in matched_cols]
        # A track can be reacquired after up to max_missing_triggers OFF
        # frames, but expires if the next frame also lacks a detection.
        expiring_rows = {
            row for row, ident in enumerate(identifiers)
            if row not in matched_rows
            and trigger - staged[ident].last_observed_trigger > self.max_missing_triggers
        }
        if len(staged) - len(expiring_rows) + len(births) > self.max_tracks:
            raise ValueError("active light-track capacity exceeded")
        # Expired tracks must neither consume capacity nor block a new
        # physical light detected near their stale extrapolation.
        for col in births:
            for row, predicted in enumerate(predictions):
                if row in expiring_rows:
                    continue
                if np.linalg.norm(positions[col] - predicted) <= self.min_separation:
                    raise ValueError("ambiguous new light-track birth")

        evidence = {}
        for row, col in selected:
            ident = identifiers[row]
            state = staged[ident]
            gap = trigger - state.last_observed_trigger
            observed = positions[col]
            evidence[ident] = TrackEvidence(
                ident, self.camera_id, trigger,
                tuple(predictions[row].tolist()), tuple(observed.tolist()), 0,
            )
            state.velocity_per_trigger = (observed - state.last_xy) / gap
            state.last_xy = observed.copy()
            state.last_observed_trigger = trigger
            state.observation_count += 1
        for row, ident in enumerate(identifiers):
            if row in matched_rows:
                continue
            state = staged[ident]
            missed = trigger - state.last_observed_trigger
            if row in expiring_rows:
                del staged[ident]
                continue
            evidence[ident] = TrackEvidence(
                ident, self.camera_id, trigger,
                tuple(predictions[row].tolist()), None, missed,
            )
        next_id = self._next_track_id
        for col in births:
            observed = positions[col]
            ident = next_id
            next_id += 1
            staged[ident] = _TrackState(
                observed.copy(), np.zeros(2), trigger, 1
            )
            evidence[ident] = TrackEvidence(
                ident, self.camera_id, trigger,
                tuple(observed.tolist()), tuple(observed.tolist()), 0,
            )

        self._tracks = staged
        self._previous_trigger = trigger
        self._next_track_id = next_id
        return tuple(evidence[ident] for ident in sorted(evidence))
