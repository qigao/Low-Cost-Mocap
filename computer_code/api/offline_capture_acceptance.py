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
from math import sqrt
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
    rejection_reasons: Mapping[str, int]
    hardware_provenance_verified: bool = False


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
    previous = None
    for record in recordings:
        if not isinstance(record, RecordedFrameBatch):
            raise ValueError("recordings must contain RecordedFrameBatch entries")
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
        if maximum is None or batch.exposure_skew_ns > maximum:
            maximum = batch.exposure_skew_ns

    return TriggerReplayReport(
        total_batches=len(recordings),
        admitted_batches=admitted,
        rejected_batches=len(recordings) - admitted,
        reference_gaps=gaps,
        maximum_declared_skew_ns=maximum,
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
