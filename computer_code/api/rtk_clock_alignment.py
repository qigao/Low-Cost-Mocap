"""Explicit alignment of RTK solution epochs to the camera/trigger clock."""
from __future__ import annotations

from dataclasses import dataclass

from imu_contract import ClockMapping
from outdoor_sensor_contract import RTKSample


@dataclass(frozen=True)
class ClockAlignedRTKSample:
    """RTK sample plus a mapped timestamp; source evidence is never overwritten."""

    sample: RTKSample
    mapped_timestamp_ns: int
    target_clock_domain: str
    clock_mapping_sha256: str
    hardware_time_alignment_verified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.sample, RTKSample):
            raise ValueError("sample must be RTKSample")
        if isinstance(self.mapped_timestamp_ns, bool) or not isinstance(self.mapped_timestamp_ns, int) or self.mapped_timestamp_ns < 0:
            raise ValueError("mapped_timestamp_ns must be a nonnegative integer")
        if not isinstance(self.target_clock_domain, str) or not self.target_clock_domain.strip():
            raise ValueError("target_clock_domain must be nonempty")
        target_domain = self.target_clock_domain.strip()
        if target_domain == self.sample.clock_domain:
            raise ValueError("aligned target clock must differ from source clock")
        if (
            not isinstance(self.clock_mapping_sha256, str)
            or len(self.clock_mapping_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.clock_mapping_sha256)
        ):
            raise ValueError("clock_mapping_sha256 must be lowercase SHA-256")
        if self.hardware_time_alignment_verified is not False:
            raise ValueError("hardware time alignment cannot be asserted by software mapping alone")
        object.__setattr__(self, "target_clock_domain", target_domain)


def align_rtk_sample_to_camera_clock(
    sample: RTKSample,
    mapping: ClockMapping,
    *,
    expected_camera_clock_domain: str,
) -> ClockAlignedRTKSample:
    """Map receiver solution epoch to the camera clock using an explicit affine fit.

    The mapping must be built from externally measured synchronization anchors
    and bounded by the fit's residual, drift and extrapolation limits. This
    function does not validate physical provenance of those anchors and never
    substitutes host receive time for the RTK solution epoch.
    """
    if not isinstance(sample, RTKSample):
        raise ValueError("sample must be RTKSample")
    if not isinstance(mapping, ClockMapping):
        raise ValueError("mapping must be ClockMapping")
    if not isinstance(expected_camera_clock_domain, str) or not expected_camera_clock_domain.strip():
        raise ValueError("expected_camera_clock_domain must be nonempty")
    target = expected_camera_clock_domain.strip()
    if mapping.source_clock_domain != sample.clock_domain:
        raise ValueError("clock mapping source domain does not match RTK solution clock")
    if mapping.target_clock_domain != target:
        raise ValueError("clock mapping target domain does not match camera clock")
    mapped = mapping.map_timestamp(sample.solution_timestamp_ns)
    return ClockAlignedRTKSample(
        sample=sample,
        mapped_timestamp_ns=mapped,
        target_clock_domain=target,
        clock_mapping_sha256=mapping.mapping_sha256,
        hardware_time_alignment_verified=False,
    )
