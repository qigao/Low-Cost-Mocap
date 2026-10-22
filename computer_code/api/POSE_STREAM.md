# Software-only PoseFrame and synchronized local LED tracking

This is a hardware-neutral integration surface, NOT a NatNet implementation
or a real-time flight-control transport. No PS3Eye camera, LED timing source,
sensor-exposure provenance or serial drone output is configured by it.

## What is connected

The software input to `SynchronizedLocalLightPipeline.process` is:

- One trusted `FrameStamp` per **physical camera ID**, all from one
  independently verified sensor exposure trigger.
- One raw 2D grayscale image per camera in the validated 2D intrinsic
  coordinate system and one 2D bright-centroid sequence from the detector.
- Optionally, a set of externally qualified local IDs for sampling a
  predicted dark ROI. Without positive **independent motion qualification**,
  missing points produce erasures (`None`), not a fabricated OFF bit.
- Optionally, the LED controller's pattern `config_sha256` for software
  revision equality; a matching string is not proof of hardware authenticity.

A detector overlap or an ambiguous global assignment is a **special
fail-closed state transition**, not a generic malformed input. On the
accepted trigger, the affected physical camera retires its old local IDs,
releases the corresponding marker identity locks, removes its ROI history,
and enters persistent camera quarantine. Subsequent positions cannot
inherit those IDs by proximity. Only a full *new* temporal code cycle
can admit new global LED IDs, after which quarantine may clear. Other
cameras may still emit measured poses if the remaining independent
current-frame geometry is sufficient. If there are fewer than two valid
camera views, the entire affected board is INVALID and contains no pose.
Malformed frame arrays, capacity violations, invalid revision/clock
metadata and other non-assignment errors still roll the whole frame back.

The adapter calls the existing `BoundedLightTracker.update` to associate
local bright points, stores only bounded stamped ROI contrast metadata
(no raw image persistence), and at the end of a code period proposes
`LocalTrackCycle` histories. The existing
`IdentityLockedBoardTracker.advance` independently re-decodes those cycles
against the immutable registry before accepting new marker IDs.

**Important:** all local trackers are staged first. If *any* camera or
downstream same-trigger/identity/geometry gate rejects the batch, no staged
local-tracker state or history is published. A locked LED scheduled OFF
cannot steal another LED's bright detection. When an actual local tracker
retires, its old ID lock is revoked; a newborn track still needs an
independently decoded code cycle before it gets a global marker ID.

This is a software join of two existing validated modules, not the same as
integrating the legacy camera's streaming hardware transport.

## Example API

```python
from board_patterns import BoardPatternRegistry
from identity_locked_board_tracker import IdentityLockedBoardTracker
from synchronized_light_pipeline import SynchronizedLocalLightPipeline
from pose_stream import describe_board_assets, pose_frame_to_json

registry = BoardPatternRegistry.from_json_file(
    "board-patterns.continuous.example.json"
)
identity = IdentityLockedBoardTracker(
    registry, camera_ids, camera_poses_by_id, intrinsics_by_id,
    calibrated_on_off_brightness_by_camera_id,
    trigger_epoch=verified_led_epoch,
    max_exposure_skew_ns=measured_exposure_skew_limit,
)
pipeline = SynchronizedLocalLightPipeline(
    identity, max_tracks_per_camera=len(registry.marker_to_board),
)

static_description = describe_board_assets(registry)

# Run this exactly once for each externally verified common-trigger batch.
current = pipeline.process(
    frame_stamps,                        # [FrameStamp ...]
    grayscale_images_by_camera_id,       # {camera_id: raw_2d_array}
    bright_centroids_by_camera_id,       # {camera_id: [(x, y) ...]}
    controller_config_sha256=registry.config_sha256,
)
message_json = pose_frame_to_json(current, registry)
```

At first, local bright tracks have no *global* LED identity and no board
measurement is published. After a code cycle is admitted, 6DoF can update
on later same-exposure measurements, even between code-cycle boundaries,
subject to current actually visible noncollinear LED geometry.

## Wire-schema contract, version 1

Asset descriptions (send only after config changes) and per-frame data
have the same fixed `protocol="low_cost_mocap_pose"` and
`schema_version=1`, but different `kind`:

### AssetDescription: `kind="asset_description"`

Pattern registry SHA-256, period, global code error budget, list of boards,
per-board rigid mounting transform, stable marker IDs, declared model-local
metre coordinates and temporal codewords. No pose measurements.

### PoseFrame: `kind="pose_frame"`

Frame trigger ID, declared common clock domain, latest input exposure
timestamp (ns), inter-camera skew (ns), physical camera IDs, pattern SHA-256,
camera calibration SHA-256 and a sorted list of per-board outcomes.

Each board reports:

- `tracking_status`: TRACKING / DEGRADED / LOST / DISCOVER
- `evidence_kind`: MEASURED or INVALID (no prediction output in v1)
- `quarantined_camera_ids` at frame level: physical camera IDs whose
  local assignments were ambiguous and whose old temporal identity locks
  have been revoked. Their current bright pixels are excluded from measured
  3D geometry until a full new code cycle revalidates an identity.
- `visible_marker_ids` / `excluded_marker_ids`
- `measurement_trigger_id` / `measurement_age_triggers`
- `last_measurement_exposure_timestamp_ns` / `measurement_age_ns`
- `position_world_from_body` and `quaternion_world_from_body_xyzw`
- `position_world_from_board` and `quaternion_world_from_board_xyzw`
- `quality`: rigid-fit RMS (reconstruction world units) and max reprojection RMS (pixels)
- `metric_scale_validated=false`

**Only** a current same-trigger, zero-age TRACKING/MEASURED pose may have
non-null position/quaternion/quality. All INVALID records contain null
values for them; a past measurement timestamp/age may still be available
for diagnostic purposes. A consumer must not treat INVALID as a usable pose.

World position units are labelled `uncalibrated_world` until the actual
camera array's metric scale is independently verified; model-local metre
labels alone cannot certify camera/world scale. Both messages expose
`hardware_provenance_verified=false`, always. A fake source string cannot
turn them into hardware evidence.

The serializer checks proper rotations, finite numbers, board-ID ownership,
current-trigger freshness and that body poses agree with the registered
board-to-body mounting transform. Quaternion order is **XY Z W**, explicitly
`[qx, qy, qz, qw]` (not WXYZ), with nonnegative W canonicalization where
possible. JSON uses finite numeric values only and stable key ordering.

## Explicit non-goals and next acceptance

- This is **not** NatNet/UDP transport, an activated Flask Socket.IO event
  or a new path to drone arming. A future network publisher must preserve
  this schema and expose a bounded backpressure/drop policy.
- Existing `helpers.Cameras._camera_read` discards pseyepy dequeue timing
  and still uses legacy `locate_objects`. It has *not* been switched.
- Synthetic tests exercise real local-tracker assignment/retirement,
  camera-specific raw-image sampling, staggered LED ON/OFF phases,
  dropout/re-lock, and multi-board pose serialization. They do not establish
  hardware sensor-exposure sync, motion blur, optical cross-talk, actual
  confidence bounds, millisecond latency, or flight safety.
- The remaining real-hardware acceptance stays in [HW-01..HW-06](../../docs/HARDWARE_VALIDATION.md).
  Software Issue [#10](https://github.com/qigao/Low-Cost-Mocap/issues/10)
  tracks the standalone integration and conformance work.
