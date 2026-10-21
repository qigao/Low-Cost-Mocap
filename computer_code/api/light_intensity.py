"""Background-normalized raw-frame light evidence for temporal marker IDs.

This module never interprets pseyepy software dequeue times as exposure
timestamps. Inputs must be raw, 2D grayscale images with a separately
established local physical-light track position for each frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from geometry import _observation_array
from temporal_pattern import LightSample, _nonnegative_int


def _finite_real(value: object, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


@dataclass(frozen=True)
class ROIContrast:
    foreground_median: float
    background_median: float
    contrast: float
    foreground_pixels: int
    background_pixels: int


def sample_light_roi(
    image: object,
    center_xy: object,
    *,
    foreground_radius_px: int = 2,
    background_radius_px: int = 6,
) -> ROIContrast:
    """Measure local LED contrast without using a global brightness threshold.

    The foreground disk and surrounding annulus are disjoint. This suppresses
    global exposure/ambient offsets, but not motion mismatch or a second LED
    contaminating the ROI; those require independent tracking/quality gates.
    Reject ROI borders and malformed inputs rather than sampling arbitrary
    clipped pixels with changed spatial support.
    """
    raw = np.asarray(image)
    if raw.ndim != 2 or raw.dtype.kind not in "iuf":
        raise ValueError("image must be a finite 2D numeric grayscale array")
    if not np.isfinite(raw).all():
        raise ValueError("image must contain finite grayscale values")
    front = _nonnegative_int(foreground_radius_px, "foreground_radius_px")
    outer = _nonnegative_int(background_radius_px, "background_radius_px")
    if front < 1 or outer <= front:
        raise ValueError("background radius must exceed positive foreground radius")

    center = _observation_array(center_xy, 0)
    cx, cy = float(center[0]), float(center[1])
    height, width = raw.shape
    if (
        cx - outer < 0 or cy - outer < 0
        or cx + outer >= width or cy + outer >= height
    ):
        raise ValueError("marker ROI falls outside the image boundary")

    x0 = max(0, int(np.floor(cx - outer)))
    x1 = min(width, int(np.ceil(cx + outer)) + 1)
    y0 = max(0, int(np.floor(cy - outer)))
    y1 = min(height, int(np.ceil(cy + outer)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    foreground = distance_sq <= front * front
    background = (distance_sq > front * front) & (distance_sq <= outer * outer)
    if not foreground.any() or not background.any():
        raise ValueError("marker ROI has insufficient foreground or background pixels")

    values = raw[y0:y1, x0:x1]
    foreground_median = float(np.median(values[foreground]))
    background_median = float(np.median(values[background]))
    contrast = foreground_median - background_median
    if not np.isfinite(contrast):
        raise ValueError("marker ROI contrast is not finite")
    return ROIContrast(
        foreground_median, background_median, contrast,
        int(np.count_nonzero(foreground)),
        int(np.count_nonzero(background)),
    )


@dataclass(frozen=True)
class BinaryIntensityCalibration:
    """Known local dark/on reference contrasts for one camera and gain setting.

    A dead zone around the midpoint yields an explicit erasure when a bit
    cannot be distinguished confidently. References must be established on
    measured exposures, not inferred by trying arbitrary marker code phases.
    """
    dark_contrast: float
    illuminated_contrast: float
    minimum_contrast_separation: float = 10.0
    dead_zone_fraction: float = 0.2

    def __post_init__(self) -> None:
        dark = _finite_real(self.dark_contrast, "dark_contrast")
        bright = _finite_real(self.illuminated_contrast, "illuminated_contrast")
        separation = _finite_real(
            self.minimum_contrast_separation, "minimum_contrast_separation"
        )
        dead_zone = _finite_real(self.dead_zone_fraction, "dead_zone_fraction")
        if (
            separation <= 0 or not bright > dark
            or not np.isfinite(bright - dark)
            or bright - dark < separation
        ):
            raise ValueError("illumination references have insufficient contrast separation")
        if not 0 < dead_zone < 0.5:
            raise ValueError("dead_zone_fraction must lie strictly between 0 and 0.5")
        object.__setattr__(self, "dark_contrast", dark)
        object.__setattr__(self, "illuminated_contrast", bright)
        object.__setattr__(self, "minimum_contrast_separation", separation)
        object.__setattr__(self, "dead_zone_fraction", dead_zone)

    def classify(self, contrast: float | None) -> int | None:
        """Return 0, 1 or None (uncertain/erasure), never force a midpoint bit."""
        if contrast is None:
            return None
        value = _finite_real(contrast, "light contrast")
        middle = 0.5 * self.dark_contrast + 0.5 * self.illuminated_contrast
        margin = self.dead_zone_fraction * (
            self.illuminated_contrast - self.dark_contrast
        )
        if value <= middle - margin:
            return 0
        if value >= middle + margin:
            return 1
        return None


def quantize_roi_track(
    measured: Sequence[tuple[int, ROIContrast | None]],
    calibration: BinaryIntensityCalibration,
) -> tuple[LightSample, ...]:
    """Convert one tracked LED's ROI measurements into timestamped bits.

    A missing ROI is an erasure, not a known OFF phase. A matching code epoch
    and actual hardware trigger IDs remain required by the temporal decoder.
    """
    if not isinstance(calibration, BinaryIntensityCalibration):
        raise ValueError("calibration must be BinaryIntensityCalibration")
    samples = []
    for entry in measured:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("ROI samples must be (trigger_id, ROIContrast|None) pairs")
        trigger = _nonnegative_int(entry[0], "trigger_id")
        roi = entry[1]
        if roi is not None and not isinstance(roi, ROIContrast):
            raise ValueError("ROI evidence must be ROIContrast or None")
        samples.append(
            LightSample(trigger, calibration.classify(None if roi is None else roi.contrast))
        )
    return tuple(samples)
