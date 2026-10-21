"""Admission boundary from camera-local light history to a decoded 2D marker.

A marker ID is never supplied independently of its temporal evidence.
An actual ON-frame centroid from that same trace is mandatory to emit
a 2D detection. Exposure metadata provenance remains a hardware adapter
obligation, and this module is not connected to live flight control.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from geometry import _observation_array
from light_intensity import (
    BinaryIntensityCalibration,
    StampedTrackROI,
    quantize_stamped_track_trace,
)
from temporal_pattern import (
    DecodedMarker,
    FrameStamp,
    StampedIdentifiedDetection,
    TemporalCodebook,
    _nonnegative_int,
    decode_temporal_marker,
)


@dataclass(frozen=True)
class AdmittedMarker:
    """Decoded marker plus its local track and cycle audit metadata."""
    detection: StampedIdentifiedDetection
    camera_id: int
    local_track_id: int
    decoded_cycle_index: int
    valid_samples: int
    erased_samples: int
    bit_errors: int


def admit_track_marker_at_frame(
    trace: Sequence[StampedTrackROI],
    codebook: TemporalCodebook,
    calibration: BinaryIntensityCalibration,
    *,
    camera_id: int,
    local_track_id: int,
    target_frame: FrameStamp,
    trigger_epoch: int,
    min_valid_samples: int = 4,
    required_margin: int = 1,
) -> AdmittedMarker:
    """Decode one local track and produce a frame-anchored ON detection.

    Every trace item must refer to this camera/track, and the target frame
    must be the final item of the *same* code cycle. The final sample must
    itself have a positively observed centroid and a calibrated ON bit:
    a known-dark ROI or predicted-only position cannot be triangulated.
    No historical marker identity can be attached to the next cycle's frame.
    """
    if not isinstance(codebook, TemporalCodebook):
        raise ValueError("codebook must be TemporalCodebook")
    if not isinstance(calibration, BinaryIntensityCalibration):
        raise ValueError("calibration must be BinaryIntensityCalibration")
    camera = _nonnegative_int(camera_id, "camera_id")
    track = _nonnegative_int(local_track_id, "local_track_id")
    epoch = _nonnegative_int(trigger_epoch, "trigger_epoch")
    if not isinstance(target_frame, FrameStamp):
        raise ValueError("target_frame must be FrameStamp")
    if not trace or not isinstance(trace[-1], StampedTrackROI):
        raise ValueError("trace must end in a StampedTrackROI")
    latest = trace[-1]
    if latest.frame_stamp != target_frame:
        raise ValueError("target frame must match latest trace exposure stamp")

    samples = quantize_stamped_track_trace(
        trace, calibration, camera_id=camera,
        track_id=track, max_samples=codebook.period,
    )
    decoded: DecodedMarker = decode_temporal_marker(
        samples, codebook, trigger_epoch=epoch,
        min_valid_samples=min_valid_samples,
        required_margin=required_margin,
    )
    trigger = _nonnegative_int(target_frame.trigger_id, "target_frame.trigger_id")
    if trigger < epoch or (trigger - epoch) // codebook.period != decoded.cycle_index:
        raise ValueError("decoded marker cycle does not match target frame")
    evidence = latest.track_evidence
    if evidence.observed_xy is None:
        raise ValueError("target frame requires an actually observed ON marker")
    if samples[-1].illuminated != 1:
        raise ValueError("target frame requires a calibrated ON bit")
    xy = _observation_array(evidence.observed_xy, camera)
    detection = StampedIdentifiedDetection(
        target_frame, decoded.marker_id, tuple(float(v) for v in xy)
    )
    return AdmittedMarker(
        detection=detection,
        camera_id=camera,
        local_track_id=track,
        decoded_cycle_index=decoded.cycle_index,
        valid_samples=decoded.valid_samples,
        erased_samples=decoded.erased_samples,
        bit_errors=decoded.bit_errors,
    )
