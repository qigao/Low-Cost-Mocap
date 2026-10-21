"""Hardware-neutral trigger synchronization and temporal marker identities.

These are algorithm acceptance primitives, not a hardware time-source adapter.
Exposure timestamps MUST come from one comparable hardware clock domain;
arrival times or independent camera clocks cannot establish synchronization.
Temporal codes require an explicit trigger epoch and never infer phase by
shifting observations until a marker happens to match.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from geometry import _observation_array

_MAX_ID = (1 << 63) - 1


def _nonnegative_int(value: object, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
        or value > _MAX_ID
    ):
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return int(value)


@dataclass(frozen=True)
class FrameStamp:
    camera_id: int
    trigger_id: int
    exposure_timestamp_ns: int
    clock_domain: str


@dataclass(frozen=True)
class SynchronizedBatch:
    trigger_id: int
    camera_ids: tuple[int, ...]
    clock_domain: str
    exposure_skew_ns: int


def validate_trigger_batch(
    frames: Sequence[FrameStamp],
    expected_camera_ids: Sequence[int],
    *,
    max_exposure_skew_ns: int,
    previous_trigger_id: int | None = None,
    allow_trigger_gaps: bool = False,
) -> SynchronizedBatch:
    """Fail closed on missing, duplicate, stale or mixed-trigger camera frames.

    A clock-domain label is not a clock conversion. The hardware adapter is
    responsible for ensuring all exposure timestamps really share that clock.
    This function never synthesizes trigger IDs or substitutes arrival times.
    """
    cameras = tuple(
        _nonnegative_int(camera, "expected camera_id") for camera in expected_camera_ids
    )
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("expected_camera_ids must be nonempty and unique")
    skew_limit = _nonnegative_int(max_exposure_skew_ns, "max_exposure_skew_ns")
    if not isinstance(allow_trigger_gaps, (bool, np.bool_)):
        raise ValueError("allow_trigger_gaps must be boolean")
    if len(frames) != len(cameras):
        raise ValueError("synchronized batch has missing or extra camera frames")

    triggers: set[int] = set()
    domains: set[str] = set()
    seen_cameras: set[int] = set()
    timestamps = []
    for frame in frames:
        if not isinstance(frame, FrameStamp):
            raise ValueError("frames must contain FrameStamp records")
        camera = _nonnegative_int(frame.camera_id, "camera_id")
        trigger = _nonnegative_int(frame.trigger_id, "trigger_id")
        timestamp = _nonnegative_int(frame.exposure_timestamp_ns, "exposure_timestamp_ns")
        if not isinstance(frame.clock_domain, str) or not frame.clock_domain.strip():
            raise ValueError("clock_domain must be a nonempty shared clock name")
        if camera in seen_cameras:
            raise ValueError("duplicate camera frame in synchronized batch")
        seen_cameras.add(camera)
        triggers.add(trigger)
        domains.add(frame.clock_domain)
        timestamps.append(timestamp)
    if seen_cameras != set(cameras):
        raise ValueError("synchronized batch has missing or unexpected camera_ids")
    if len(triggers) != 1:
        raise ValueError("camera frames have different trigger_ids")
    if len(domains) != 1:
        raise ValueError("camera exposure timestamps use different clock domains")
    skew = max(timestamps) - min(timestamps)
    if skew > skew_limit:
        raise ValueError("camera exposure timestamp skew exceeds configured limit")

    current = next(iter(triggers))
    if previous_trigger_id is not None:
        previous = _nonnegative_int(previous_trigger_id, "previous_trigger_id")
        if current <= previous:
            raise ValueError("stale or repeated trigger_id")
        if not allow_trigger_gaps and current != previous + 1:
            raise ValueError("trigger gap detected")
    return SynchronizedBatch(current, cameras, next(iter(domains)), skew)


@dataclass(frozen=True)
class TemporalCodebook:
    """Periodic per-marker on/off codes with bounded Hamming error correction.

    Code distance greater than twice max_bit_errors is required before an
    erasure-free period can uniquely correct that many corrupted observations.
    Erasures still reduce distance and must pass runtime ambiguity checks.
    """
    patterns: Mapping[int, Sequence[int]]
    max_bit_errors: int = 1
    period: int = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.patterns, Mapping) or len(self.patterns) < 2:
            raise ValueError("codebook must contain at least two marker patterns")
        errors = _nonnegative_int(self.max_bit_errors, "max_bit_errors")
        normalized: dict[int, tuple[int, ...]] = {}
        period = None
        for raw_marker, raw_bits in self.patterns.items():
            marker = _nonnegative_int(raw_marker, "marker_id")
            if marker in normalized:
                raise ValueError("duplicate marker_id in codebook")
            if not isinstance(raw_bits, Sequence) and not isinstance(raw_bits, np.ndarray):
                raise ValueError("each marker pattern must be a binary sequence")
            bits = []
            for bit in raw_bits:
                if not isinstance(bit, (bool, np.bool_, int, np.integer)) or int(bit) not in (0, 1):
                    raise ValueError("marker patterns must contain only binary bits")
                bits.append(int(bit))
            if period is None:
                period = len(bits)
            if len(bits) != period or len(bits) < 4:
                raise ValueError("marker patterns must share a period of at least 4 triggers")
            if 0 not in bits or 1 not in bits:
                raise ValueError("marker pattern must have both illuminated and dark phases")
            normalized[marker] = tuple(bits)
        ids = list(normalized)
        for i, first in enumerate(ids):
            for second in ids[i + 1:]:
                distance = sum(
                    a != b for a, b in zip(normalized[first], normalized[second])
                )
                if distance <= 2 * errors:
                    raise ValueError("temporal codebook Hamming distance is insufficient")
        object.__setattr__(self, "patterns", MappingProxyType(normalized))
        object.__setattr__(self, "period", int(period))


@dataclass(frozen=True)
class LightSample:
    trigger_id: int
    illuminated: int | bool | None


@dataclass(frozen=True)
class DecodedMarker:
    marker_id: int
    cycle_index: int
    valid_samples: int
    erased_samples: int
    bit_errors: int
    competing_error_margin: int


def decode_temporal_marker(
    samples: Sequence[LightSample],
    codebook: TemporalCodebook,
    *,
    trigger_epoch: int,
    min_valid_samples: int = 4,
    required_margin: int = 1,
) -> DecodedMarker:
    """Decode one tracked light using the shared trigger phase and erasures.

    This does not track image positions and must not fuse samples from two
    physical lights. Samples must all come from one complete or partial cycle;
    missing triggers are explicit erasures. A tie or weak runner-up margin
    rejects the candidate rather than assigning an arbitrary marker ID.
    """
    if not isinstance(codebook, TemporalCodebook):
        raise ValueError("codebook must be TemporalCodebook")
    epoch = _nonnegative_int(trigger_epoch, "trigger_epoch")
    min_valid = _nonnegative_int(min_valid_samples, "min_valid_samples")
    margin = _nonnegative_int(required_margin, "required_margin")
    if min_valid < 1 or min_valid > codebook.period:
        raise ValueError("min_valid_samples must be within the code period")
    if margin < 1:
        raise ValueError("required_margin must be positive")
    if not samples:
        raise ValueError("temporal marker has no trigger samples")
    seen: set[int] = set()
    cycle_indices: set[int] = set()
    observed: dict[int, int] = {}
    for sample in samples:
        if not isinstance(sample, LightSample):
            raise ValueError("samples must contain LightSample records")
        trigger = _nonnegative_int(sample.trigger_id, "trigger_id")
        if trigger < epoch:
            raise ValueError("trigger precedes temporal code epoch")
        if trigger in seen:
            raise ValueError("duplicate temporal trigger sample")
        seen.add(trigger)
        cycle, phase = divmod(trigger - epoch, codebook.period)
        cycle_indices.add(cycle)
        bit = sample.illuminated
        if bit is None:
            continue
        if not isinstance(bit, (bool, np.bool_, int, np.integer)) or int(bit) not in (0, 1):
            raise ValueError("illumination must be a binary bit or an erasure")
        observed[phase] = int(bit)
    if len(cycle_indices) != 1:
        raise ValueError("temporal samples span different code cycles")
    if len(observed) < min_valid:
        raise ValueError("insufficient non-erased temporal observations")

    candidates = []
    for marker, pattern in codebook.patterns.items():
        errors = sum(pattern[phase] != bit for phase, bit in observed.items())
        candidates.append((errors, marker))
    candidates.sort()
    best, marker = candidates[0]
    difference = candidates[1][0] - best
    if best > codebook.max_bit_errors:
        raise ValueError("no temporal marker matches the allowed bit-error budget")
    # With erasures, the originally separated codewords can become close.
    # A smaller error count alone does NOT identify the source: any code
    # within the declared bit-flip budget could have produced the samples.
    # Require exactly one candidate inside that budget even if the winner
    # itself has zero errors and the runner-up is just one error away.
    if candidates[1][0] <= codebook.max_bit_errors or difference < margin:
        raise ValueError("ambiguous temporal marker code after erasures")
    return DecodedMarker(
        marker_id=marker,
        cycle_index=next(iter(cycle_indices)),
        valid_samples=len(observed),
        # Both explicit None entries and omitted trigger phases are
        # missing measurements in the actual code period.
        erased_samples=codebook.period - len(observed),
        bit_errors=best,
        competing_error_margin=difference,
    )


@dataclass(frozen=True)
class IdentifiedDetection:
    camera_id: int
    marker_id: int
    xy: Any


@dataclass(frozen=True)
class StampedIdentifiedDetection:
    """A decoded marker location bound to its originating frame stamp."""
    frame_stamp: FrameStamp
    marker_id: int
    xy: Any


@dataclass(frozen=True)
class GroupedMarkers:
    observations_by_marker_id: Mapping[int, tuple[np.ndarray | None, ...]]
    excluded_marker_ids: tuple[int, ...]


def group_decoded_observations(
    detections: Sequence[IdentifiedDetection],
    *,
    camera_count: int,
    min_views: int = 2,
    allowed_marker_ids: Sequence[int] | None = None,
) -> GroupedMarkers:
    """Group positively decoded IDs without detector-order assumptions.

    Any duplicate (camera, marker) identity fails closed; identities with
    fewer than min_views are reported as excluded, never silently triangulated.

    This low-level helper assumes frames already align. Moving targets must
    use group_synchronized_detections to enforce same-trigger admission.
    """
    cameras = _nonnegative_int(camera_count, "camera_count")
    views = _nonnegative_int(min_views, "min_views")
    if cameras < 2 or views < 2 or views > cameras:
        raise ValueError("camera_count and min_views must admit at least two views")
    allowed = None
    if allowed_marker_ids is not None:
        identities = [_nonnegative_int(v, "allowed marker_id") for v in allowed_marker_ids]
        if len(identities) != len(set(identities)):
            raise ValueError("allowed marker IDs must be unique")
        allowed = set(identities)

    grouped: dict[int, list[np.ndarray | None]] = {}
    for detection in detections:
        if not isinstance(detection, IdentifiedDetection):
            raise ValueError("detections must be IdentifiedDetection records")
        camera = _nonnegative_int(detection.camera_id, "camera_id")
        marker = _nonnegative_int(detection.marker_id, "marker_id")
        if camera >= cameras:
            raise ValueError("camera_id exceeds camera_count")
        if allowed is not None and marker not in allowed:
            raise ValueError("decoded marker_id is not in the allowed codebook")
        values = grouped.setdefault(marker, [None] * cameras)
        if values[camera] is not None:
            raise ValueError("duplicate decoded marker identity in one camera")
        values[camera] = _observation_array(detection.xy, camera)
    admitted = {}
    excluded = []
    for marker in sorted(grouped):
        values = grouped[marker]
        if sum(xy is not None for xy in values) < views:
            excluded.append(marker)
        else:
            admitted[marker] = tuple(values)
    return GroupedMarkers(MappingProxyType(admitted), tuple(excluded))


def group_synchronized_detections(
    detections: Sequence[StampedIdentifiedDetection],
    frames: Sequence[FrameStamp],
    expected_camera_ids: Sequence[int],
    *,
    max_exposure_skew_ns: int,
    previous_trigger_id: int | None = None,
    allow_trigger_gaps: bool = False,
    min_views: int = 2,
    allowed_marker_ids: Sequence[int] | None = None,
) -> tuple[SynchronizedBatch, GroupedMarkers]:
    """Bind decoded 2D marker locations to one admitted exposure batch.

    It rejects observations borrowed from another trigger, camera, timestamp
    or clock domain, preserving expected_camera_ids order even when those
    physical IDs are not consecutive. A separate trusted camera adapter must
    still prove real hardware exposure provenance.
    """
    batch = validate_trigger_batch(
        frames, expected_camera_ids,
        max_exposure_skew_ns=max_exposure_skew_ns,
        previous_trigger_id=previous_trigger_id,
        allow_trigger_gaps=allow_trigger_gaps,
    )
    frame_by_camera = {frame.camera_id: frame for frame in frames}
    camera_position = {
        camera_id: offset for offset, camera_id in enumerate(batch.camera_ids)
    }
    aligned = []
    for detection in detections:
        if not isinstance(detection, StampedIdentifiedDetection):
            raise ValueError("detections must contain StampedIdentifiedDetection records")
        stamp = detection.frame_stamp
        if (
            not isinstance(stamp, FrameStamp)
            or stamp.camera_id not in frame_by_camera
            or stamp != frame_by_camera[stamp.camera_id]
        ):
            raise ValueError(
                "marker observation frame does not belong to synchronized trigger batch"
            )
        aligned.append(IdentifiedDetection(
            camera_id=camera_position[stamp.camera_id],
            marker_id=detection.marker_id,
            xy=detection.xy,
        ))
    grouped = group_decoded_observations(
        aligned, camera_count=len(batch.camera_ids), min_views=min_views,
        allowed_marker_ids=allowed_marker_ids,
    )
    return batch, grouped
