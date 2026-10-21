# Continuous identity-locked rigid-body tracking — software prototype

This is an isolated, bounded software service. It **does not** connect the
PS3Eye/pseyepy backend, validate physical exposure/LED-controller timing, or
enable drone serial/flight-control output.

## Why another tracking layer is needed

[The original balanced codes](board-patterns.example.json) show three or more
active points on both boards only once per 16-trigger period. The
[high-visibility design-only candidate](board-patterns.continuous.example.json)
nominally lights at least three noncollinear LEDs on both boards at **each**
trigger. The earlier admission function, however, decodes only the current
full code cycle: a valid marker ID cannot simply be reused on every exposure
unless its physical camera-local light track is kept unambiguous and its
identity is revalidated periodically.

`identity_locked_board_tracker.IdentityLockedBoardTracker` adds this bounded
identity-lock software stage. It consumes *current* `StampedTrackROI` evidence
from the existing `BoundedLightTracker` and `sample_tracked_light_roi`
algorithms, and is initially seeded by `AdmittedMarker` records obtained
from `admit_track_marker_at_frame()` with the current registered codebook.

## Single-batch API

```python
from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import IdentityLockedBoardTracker

registry = BoardPatternRegistry.from_json_file(
    "board-patterns.continuous.example.json"
)
tracker = IdentityLockedBoardTracker(
    registry,
    camera_ids,                         # stable physical IDs, e.g. (2, 5, 9)
    camera_poses_by_id,                 # calibrated world-to-camera R/t
    intrinsics_by_id,                   # camera-specific calibrated K
    brightness_calibration_by_camera_id,
    trigger_epoch=hardware_led_epoch,
    max_exposure_skew_ns=measured_exposure_skew_limit,
    max_identity_age_triggers=32,       # policy, not a measured safety limit
    max_degraded_triggers=2,
)

# At the first correctly decoded full code cycle, seed/revalidate the lock:
first = tracker.advance(
    physical_frame_stamps,              # verified exposure-associated metadata
    current_stamped_track_roi_evidence, # current camera-local LED observations
    revalidations=cycle_admitted_markers,
    controller_config_sha256=registry.config_sha256,
)

# Subsequent hardware triggers can report measured 6DoF *if* the LED positions
# are really observed, the codebook says they are ON, and geometry passes.
next_frame = tracker.advance(
    next_physical_frame_stamps, next_current_track_roi_evidence
)
for board_id, status in next_frame.boards.items():
    if status.evidence_kind == "MEASURED":
        assert status.state == "TRACKING"
        current_pose = status.outcome.pose
    else:
        assert status.evidence_kind == "INVALID"
        assert status.outcome.pose is None  # old pose is NEVER reported as new
```

## Contracts and failure modes

- A lock is `(physical camera_id, local_track_id) -> global marker_id`,
  seeded/revalidated only by a matching current `AdmittedMarker` and actual
  ON ROI centroid. A competing marker on the same camera and local track, or
  two tracks claiming the same marker, fails closed.
- `config_sha256` is carried with the lock. The optional
  `controller_config_sha256` must match before any frame updates. A matching
  string is only a **software revision check**, not hardware authenticity.
- Each current exposure must use a monotonically increasing trigger number
  and declared shared clock domain, with monotonic per-camera timestamps and
  bounded cross-camera skew. All input metadata must be independently proven
  by an actual hardware source; pseyepy host dequeue timestamps are not valid.
- A code phase saying **OFF** cannot emit a triangulable marker. A calibrated
  LED contrast contradicting the registered phase revokes that local lock.
  Qualified predicted/dark ROIs never become a measured point. Only a
  currently observed, calibrated ON centroid is routed into triangulation.
- A lock expires after `max_identity_age_triggers` unless its same local
  track is revalidated against a new code cycle. Expired/contradictory
  identities are not re-associated automatically with another LED ID.
- Reconstructed markers are partitioned by board *before* triangulation.
  A bad marker on board B does not discard board A's current valid
  measurement. Each board still passes its own rigid-layout fit.
- `TRACKING/MEASURED`: fresh same-exposure rigid pose.
  `DEGRADED/INVALID`: temporarily missing measurement within configured age.
  `LOST/INVALID`: missing for too long.
  `DISCOVER/INVALID`: never accepted a board pose.
  Predicted 6DoF output is **not implemented**. The status field never
  labels a held/forecast pose as optically measured.
- The service retains only bounded code identity metadata and last-success
  trigger IDs, not the prior image/3D pose. Unhandled global validation
  failures do not advance tracker state.

## Acceptance

Synthetic tests include two boards, noncontiguous camera IDs, distinct
intrinsics, per-trigger movement, full-code-cycle initialization and second-
cycle revalidation, observed-ON-only association, a bad-code-bit lock revocation,
one-board-only geometry failure, timeout/occlusion/LOST transitions, cross-
clock and duplicate frame rollback, and controller revision mismatch.

These are not real-world performance or flight-safety results. Outstanding
[SW identity tracking #10](https://github.com/qigao/Low-Cost-Mocap/issues/10)
and independent hardware [HW-01..HW-06](../../docs/HARDWARE_VALIDATION.md)
remain relevant until the actual capture source, motion accuracy, relock,
measurement age and control latency are independently qualified.
