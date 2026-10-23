"""Provider-neutral detector boundary for fiducial tag observations.

Detector backends may identify tags and image corners, but they do not own
camera/frame/timing/calibration/map provenance. The host injects that metadata
through DetectorFrameContext after the camera frame has passed its own gates.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

from fiducial_contract import FiducialObservation


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _text(value: object, name: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
        raise ValueError(
            f"{name} must be a nonempty string of at most {limit} characters"
        )
    return value.strip()


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


@dataclass(frozen=True)
class DetectorFrameContext:
    """Trusted host metadata for one detector invocation.

    Values here are declarations from an already validated camera/capture layer;
    this class does not independently prove hardware timing or calibration.
    """

    camera_id: int
    camera_serial: str
    frame_id: int
    trigger_id: int
    exposure_timestamp_ns: int
    clock_domain: str
    calibration_sha256: str
    map_revision_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "camera_id",
            "frame_id",
            "trigger_id",
            "exposure_timestamp_ns",
        ):
            _nonnegative_int(getattr(self, name), name)
        object.__setattr__(
            self, "camera_serial", _text(self.camera_serial, "camera_serial")
        )
        object.__setattr__(
            self, "clock_domain", _text(self.clock_domain, "clock_domain")
        )
        _digest(self.calibration_sha256, "calibration_sha256")
        _digest(self.map_revision_sha256, "map_revision_sha256")


@dataclass(frozen=True)
class RawFiducialDetection:
    """Detector-owned data only; no capture provenance is accepted here."""

    tag_family: str
    tag_id: int
    image_corners_px: object
    corrected_bit_errors: int
    corner_rms_px: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tag_family", _text(self.tag_family, "tag_family", 96)
        )
        _nonnegative_int(self.tag_id, "tag_id")
        _nonnegative_int(self.corrected_bit_errors, "corrected_bit_errors")
        object.__setattr__(
            self,
            "corner_rms_px",
            _nonnegative_real(self.corner_rms_px, "corner_rms_px"),
        )


class FiducialDetectorAdapter(Protocol):
    """AprilTag/ArUco-compatible provider boundary.

    The image payload is intentionally opaque to the core contract. A concrete
    adapter owns any backend-specific image conversion and must return only
    RawFiducialDetection records.
    """

    backend_name: str
    backend_revision: str

    def detect(self, image: object) -> Sequence[RawFiducialDetection]: ...


@dataclass(frozen=True)
class FiducialDetectionBatch:
    """Canonical normalized output for one detector invocation."""

    backend_name: str
    backend_revision: str
    context: DetectorFrameContext
    observations: tuple[FiducialObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_name", _text(self.backend_name, "backend_name", 128)
        )
        object.__setattr__(
            self,
            "backend_revision",
            _text(self.backend_revision, "backend_revision", 128),
        )
        if not isinstance(self.context, DetectorFrameContext):
            raise ValueError("context must be DetectorFrameContext")
        if not isinstance(self.observations, tuple):
            raise ValueError("observations must be a tuple")
        if any(not isinstance(obs, FiducialObservation) for obs in self.observations):
            raise ValueError("observations must contain FiducialObservation records")

        keys = [(obs.tag_family, obs.tag_id) for obs in self.observations]
        if len(set(keys)) != len(keys):
            raise ValueError("detector frame contains duplicate tag family/ID")
        if keys != sorted(keys):
            raise ValueError("observations must use canonical family/ID ordering")

        for obs in self.observations:
            if (
                obs.camera_id != self.context.camera_id
                or obs.camera_serial != self.context.camera_serial
                or obs.frame_id != self.context.frame_id
                or obs.trigger_id != self.context.trigger_id
                or obs.exposure_timestamp_ns
                != self.context.exposure_timestamp_ns
                or obs.clock_domain != self.context.clock_domain
                or obs.calibration_sha256 != self.context.calibration_sha256
                or obs.map_revision_sha256 != self.context.map_revision_sha256
            ):
                raise ValueError(
                    "observation metadata does not match detector frame context"
                )


def normalize_detector_output(
    adapter: FiducialDetectorAdapter,
    image: object,
    context: DetectorFrameContext,
    *,
    allowed_families: Sequence[str] | None = None,
) -> FiducialDetectionBatch:
    """Run one backend and bind validated host provenance to its detections.

    No fallback backend is attempted. Backend exceptions propagate so the
    caller can fail the frame explicitly instead of silently changing detector
    semantics.
    """

    if not isinstance(context, DetectorFrameContext):
        raise ValueError("context must be DetectorFrameContext")

    backend_name = _text(getattr(adapter, "backend_name", None), "backend_name", 128)
    backend_revision = _text(
        getattr(adapter, "backend_revision", None), "backend_revision", 128
    )
    detect = getattr(adapter, "detect", None)
    if not callable(detect):
        raise ValueError("adapter must provide callable detect(image)")

    family_filter: set[str] | None = None
    if allowed_families is not None:
        if (
            not isinstance(allowed_families, Sequence)
            or isinstance(allowed_families, (str, bytes))
            or not allowed_families
        ):
            raise ValueError("allowed_families must be a nonempty sequence")
        family_filter = {
            _text(family, "allowed tag family", 96) for family in allowed_families
        }

    raw = detect(image)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("detector output must be a sequence")

    observations = []
    seen: set[tuple[str, int]] = set()
    for detection in raw:
        if not isinstance(detection, RawFiducialDetection):
            raise ValueError(
                "detector output must contain RawFiducialDetection records"
            )
        if family_filter is not None and detection.tag_family not in family_filter:
            raise ValueError("detector returned a tag family outside the active profile")
        key = (detection.tag_family, detection.tag_id)
        if key in seen:
            raise ValueError("detector frame contains duplicate tag family/ID")
        seen.add(key)
        observations.append(
            FiducialObservation(
                tag_family=detection.tag_family,
                tag_id=detection.tag_id,
                camera_id=context.camera_id,
                camera_serial=context.camera_serial,
                frame_id=context.frame_id,
                trigger_id=context.trigger_id,
                exposure_timestamp_ns=context.exposure_timestamp_ns,
                clock_domain=context.clock_domain,
                image_corners_px=detection.image_corners_px,
                corrected_bit_errors=detection.corrected_bit_errors,
                corner_rms_px=detection.corner_rms_px,
                calibration_sha256=context.calibration_sha256,
                map_revision_sha256=context.map_revision_sha256,
            )
        )

    ordered = tuple(sorted(observations, key=lambda obs: (obs.tag_family, obs.tag_id)))
    return FiducialDetectionBatch(
        backend_name=backend_name,
        backend_revision=backend_revision,
        context=context,
        observations=ordered,
    )
