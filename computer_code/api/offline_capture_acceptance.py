"""Offline replay statistics for independently labeled capture recordings.

These metrics never certify actual exposure hardware provenance. They quantify
software admission, known-marker confusion and sampling uncertainty so a
separate real-rig acceptance process can report evidence without silently
turning green synthetic CI into a hardware claim.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, sqrt
from statistics import NormalDist
from types import MappingProxyType
from typing import Mapping

from temporal_pattern import (
    FrameStamp,
    LightSample,
    TemporalCodebook,
    _nonnegative_int,
    decode_temporal_marker,
    validate_trigger_batch,
)


@dataclass(frozen=True)
class RecordedFrameBatch:
    """Frame stamps plus independently logged *reference* trigger ID."""
    reference_trigger_id: int
    camera_frames: tuple[FrameStamp, ...]


@dataclass(frozen=True)
class TriggerReplayReport:
    total_batches: int
    admitted_batches: int
    rejected_batches: int
    reference_gaps: int
    maximum_declared_skew_ns: int | None
    admitted_exposure_skew_ns_percentiles: Mapping[str, int]
    comparable_exposure_skew_batch_count: int
    comparable_exposure_skew_ns_percentiles: Mapping[str, int]
    rejection_reasons: Mapping[str, int]
    hardware_provenance_verified: bool = False


def _nearest_rank_skew_percentiles(values: Sequence[int]) -> dict[str, int]:
    """Summarize admitted exposure skews using the documented nearest-rank rule."""
    if not values:
        return {}
    ordered = sorted(values)

    def rank(fraction: float) -> int:
        return ordered[max(0, ceil(fraction * len(ordered)) - 1)]

    return {
        "p50": rank(0.50),
        "p95": rank(0.95),
        "p99": rank(0.99),
        "max": ordered[-1],
    }


def _comparable_declared_skew_ns(
    frames: object, expected_camera_ids: Sequence[int]
) -> int | None:
    """Measure skew when per-camera timestamps are structurally comparable.

    This diagnostic deliberately does not apply the skew threshold or require
    the independent reference trigger to match. It still requires exactly one
    valid frame per expected camera, one internal trigger ID and one clock
    domain. It summarizes declarations, not verified physical exposure times.
    """
    if not isinstance(frames, Sequence) or len(frames) != len(expected_camera_ids):
        return None
    if not frames or any(not isinstance(frame, FrameStamp) for frame in frames):
        return None
    try:
        camera_ids = tuple(_nonnegative_int(frame.camera_id, "camera_id") for frame in frames)
        trigger_ids = tuple(_nonnegative_int(frame.trigger_id, "trigger_id") for frame in frames)
        timestamps = tuple(
            _nonnegative_int(frame.exposure_timestamp_ns, "exposure_timestamp_ns")
            for frame in frames
        )
    except (TypeError, ValueError):
        return None
    if len(set(camera_ids)) != len(camera_ids) or set(camera_ids) != set(expected_camera_ids):
        return None
    if len(set(trigger_ids)) != 1:
        return None
    domains = tuple(frame.clock_domain for frame in frames)
    if any(not isinstance(domain, str) or not domain.strip() for domain in domains):
        return None
    if len(set(domains)) != 1:
        return None
    return max(timestamps) - min(timestamps)


def replay_trigger_batches(
    recordings: Sequence[RecordedFrameBatch],
    *,
    expected_camera_ids: Sequence[int],
    max_exposure_skew_ns: int,
    allow_reference_gaps: bool = False,
) -> TriggerReplayReport:
    """Apply trigger consistency gates to an ordered offline batch stream.

    The reference_trigger_id must come from an independent logging device
    when used with real hardware. A caller-supplied label is NOT proof.
    Invalid or out-of-order records cannot be silently marked admitted.
    """
    if not recordings:
        raise ValueError("recordings must be nonempty")
    if not isinstance(allow_reference_gaps, bool):
        raise ValueError("allow_reference_gaps must be boolean")
    expected = tuple(_nonnegative_int(i, "expected camera_id") for i in expected_camera_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError("expected_camera_ids must be nonempty and unique")
    max_skew = _nonnegative_int(max_exposure_skew_ns, "max_exposure_skew_ns")
    rejected = Counter()
    admitted = 0
    gaps = 0
    maximum = None
    admitted_skews: list[int] = []
    comparable_skews: list[int] = []
    previous = None
    for record in recordings:
        if not isinstance(record, RecordedFrameBatch):
            raise ValueError("recordings must contain RecordedFrameBatch entries")
        comparable_skew = _comparable_declared_skew_ns(
            record.camera_frames, expected
        )
        if comparable_skew is not None:
            comparable_skews.append(comparable_skew)
        reference = _nonnegative_int(record.reference_trigger_id, "reference_trigger_id")
        if previous is not None and reference <= previous:
            rejected["reference_not_monotonic"] += 1
            continue
        if previous is not None and reference > previous + 1:
            gaps += reference - previous - 1
            if not allow_reference_gaps:
                rejected["reference_trigger_gap"] += 1
                previous = reference
                continue
        previous = reference
        try:
            batch = validate_trigger_batch(
                record.camera_frames, expected,
                max_exposure_skew_ns=max_skew,
            )
        except ValueError as exc:
            rejected[str(exc)] += 1
            continue
        if batch.trigger_id != reference:
            rejected["independent_reference_trigger_mismatch"] += 1
            continue
        admitted += 1
        admitted_skews.append(batch.exposure_skew_ns)
        if maximum is None or batch.exposure_skew_ns > maximum:
            maximum = batch.exposure_skew_ns

    return TriggerReplayReport(
        total_batches=len(recordings),
        admitted_batches=admitted,
        rejected_batches=len(recordings) - admitted,
        reference_gaps=gaps,
        maximum_declared_skew_ns=maximum,
        admitted_exposure_skew_ns_percentiles=MappingProxyType(
            _nearest_rank_skew_percentiles(admitted_skews)
        ),
        comparable_exposure_skew_batch_count=len(comparable_skews),
        comparable_exposure_skew_ns_percentiles=MappingProxyType(
            _nearest_rank_skew_percentiles(comparable_skews)
        ),
        rejection_reasons=MappingProxyType(dict(sorted(rejected.items()))),
        hardware_provenance_verified=False,
    )


@dataclass(frozen=True)
class LabeledTemporalTrial:
    """Independent label, code epoch and one camera-local bit sequence."""
    reference_marker_id: int
    trigger_epoch: int
    samples: tuple[LightSample, ...]


@dataclass(frozen=True)
class TemporalReplayReport:
    trial_count: int
    correct_accepts: int
    wrong_accepts: int
    rejected: int
    acceptance_fraction: float
    false_accept_fraction: float
    conditional_false_accept_fraction: float | None
    wrong_accept_one_sided_upper_95: float
    rejection_reasons: Mapping[str, int]
    hardware_provenance_verified: bool = False


def _wilson_upper_95(successes: int, trials: int) -> float:
    """One-sided Wilson score bound; independent Bernoulli trials assumed."""
    if trials < 1:
        raise ValueError("need at least one trial")
    z = NormalDist().inv_cdf(0.95)
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    spread = (z / denominator) * sqrt(
        p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)
    )
    return min(1.0, center + spread)


def replay_labeled_temporal_trials(
    trials: Sequence[LabeledTemporalTrial],
    codebook: TemporalCodebook,
    *,
    min_valid_samples: int = 4,
) -> TemporalReplayReport:
    """Decode independently labeled records and count false accepted IDs.

    Malformed source data fails the *entire* replay, never gets laundered into
    an innocent decoder rejection. A zero observed error count retains a
    positive 95% upper bound, conditional on independent trials.
    """
    if not trials:
        raise ValueError("temporal trials must be nonempty")
    if not isinstance(codebook, TemporalCodebook):
        raise ValueError("codebook must be TemporalCodebook")
    required_samples = _nonnegative_int(min_valid_samples, "min_valid_samples")
    if not 1 <= required_samples <= codebook.period:
        raise ValueError("min_valid_samples must be within the codebook period")
    wrong = 0
    correct = 0
    rejection = Counter()
    for trial in trials:
        if not isinstance(trial, LabeledTemporalTrial):
            raise ValueError("trials must contain LabeledTemporalTrial entries")
        expected = _nonnegative_int(trial.reference_marker_id, "reference_marker_id")
        epoch = _nonnegative_int(trial.trigger_epoch, "trigger_epoch")
        if expected not in codebook.patterns:
            raise ValueError("independent reference ID is missing from codebook")
        # Validate trial structure and values before handling legitimate
        # decoder rejections for ambiguity, insufficient data and bit errors.
        if not trial.samples or len(trial.samples) > codebook.period:
            raise ValueError("trial must contain 1..period light samples")
        seen_triggers = set()
        cycles = set()
        for sample in trial.samples:
            if not isinstance(sample, LightSample):
                raise ValueError("trial must contain LightSample entries")
            trigger = _nonnegative_int(sample.trigger_id, "trial trigger_id")
            if trigger < epoch or trigger in seen_triggers:
                raise ValueError("trial trigger IDs must be unique and after epoch")
            seen_triggers.add(trigger)
            cycles.add((trigger - epoch) // codebook.period)
            if sample.illuminated is not None and (
                isinstance(sample.illuminated, bool) is False
                and not isinstance(sample.illuminated, int)
            ):
                # NumPy integer/bool scalars are handled by decoder below;
                # malformed non-integral values should never count as a
                # legitimate optical-marker rejection.
                import numpy as np
                if not isinstance(sample.illuminated, (np.integer, np.bool_)):
                    raise ValueError("trial light values must be binary or erased")
            if sample.illuminated is not None and int(sample.illuminated) not in (0, 1):
                raise ValueError("trial light values must be binary or erased")
        if len(cycles) != 1:
            raise ValueError("trial samples span multiple code cycles")

        try:
            decoded = decode_temporal_marker(
                trial.samples, codebook,
                trigger_epoch=epoch, min_valid_samples=required_samples,
            )
        except ValueError as exc:
            rejection[str(exc)] += 1
            continue
        if decoded.marker_id == expected:
            correct += 1
        else:
            wrong += 1
    total = len(trials)
    accepted = correct + wrong
    return TemporalReplayReport(
        trial_count=total,
        correct_accepts=correct,
        wrong_accepts=wrong,
        rejected=total - accepted,
        acceptance_fraction=accepted / total,
        false_accept_fraction=wrong / total,
        conditional_false_accept_fraction=(wrong / accepted if accepted else None),
        wrong_accept_one_sided_upper_95=_wilson_upper_95(wrong, total),
        rejection_reasons=MappingProxyType(dict(sorted(rejection.items()))),
        hardware_provenance_verified=False,
    )
