# RTK + IMU + Visual Tags: Indoor Map and Outdoor Anchor Profiles

**Status:** architecture proposal plus provider-neutral tag-map/observation contracts. No camera/tag detector or outdoor hardware is qualified by this document.

## Decision

Use one timestamped, quality-aware state-estimation architecture with separate environment profiles:

- **Indoor dense-map profile:** standardized visual tags are fixed landmarks in a surveyed/calibrated indoor map. IMU propagates state between tag observations. RTK is normally absent or treated as unavailable.
- **Outdoor sparse-anchor profile:** RTK provides georeferenced position and its original FIXED/FLOAT/DGNSS/SINGLE/INVALID quality state; IMU provides high-rate propagation; fixed visual tags at selected waypoints provide local pose corrections, map association, and relocalization checks.
- **Shared estimator:** consume only observations with explicit acquisition time, clock domain, camera calibration/map revision, and quality. Preserve TRACKING, DEGRADED and LOST states. Do not fabricate a full 6DoF pose from RTK position alone.

The initial implementation remains a **moving camera observing fixed environmental tags** (inside-out localization). The existing Outside-In mode—fixed external cameras observing a tag/marker constellation on the moving body—is a distinct profile. Environmental anchors do not replace body-mounted markers in that topology.

## Map and observation contracts

The new provider-neutral module is `computer_code/api/fiducial_contract.py`.

- `TagAnchor`: family/ID, map frame, physical size, map-from-tag pose, uncertainty, and reference method.
- `FiducialTagMap`: `indoor_dense` or `outdoor_sparse_anchors`, a single map frame, unique family/ID pairs, immutable lookup, and deterministic SHA-256 revision digest.
- `FiducialObservation`: family/ID, camera serial, frame/trigger IDs, sensor-exposure timestamp and clock domain, ordered image corners, corrected code-bit count, corner RMS, camera-calibration digest, and tag-map digest.
- `validate_fiducial_observation`: fail-closed checks for frame/time/camera identity, active map and calibration revisions, known tag, code errors and corner quality.

Outdoor anchor maps require surveyed tag poses. Indoor maps may use surveyed or calibrated tag poses. A hash establishes revision identity, not that a survey or timestamp is physically trustworthy. The camera adapter must source timestamps from documented exposure metadata, not packet arrival time.

## Estimation and fusion

For a detected tag, estimate camera-from-tag from its known physical size, ordered corners, and calibrated camera intrinsics. Combine multiple visible tags using robust reprojection-error gating. Reject planar-pose ambiguities and poor geometry using multi-frame consistency, viewing angle, tag pixel area, and covariance; never trust one detector's pose without checks.

For a moving camera observing fixed tags:

1. Resolve the tag observation against the active map revision.
2. Estimate camera pose in the map frame from the known tag pose and camera-from-tag measurement.
3. Convert camera pose to body pose using a calibrated camera-to-body extrinsic.
4. Fuse timestamp-aligned visual pose with IMU propagation and, outdoors, RTK position after converting WGS-84 coordinates into the configured local ENU frame.
5. Apply the calibrated GNSS antenna-to-body lever arm using the estimated orientation.
6. Gate innovations by covariance and source quality. RTK FIXED does not mean perfect; FLOAT/invalid/stale corrections must lower or remove the global-position update. A tag relocalization that conflicts with RTK must be diagnosed, not silently averaged.

A single fixed planar tag can have pose ambiguity and poor depth/orientation observability at oblique views. Prefer several tags with spatial spread and varying orientations at critical anchors; keep tag size and map survey uncertainty explicit.

## Placement strategy

### Indoor

- Standardize one family and documented physical print size per deployment.
- Use unique IDs, non-coplanar or multi-wall placement where feasible, and overlapping visibility between neighboring areas.
- Survey/calibrate tag poses into a single map frame; version the map whenever a tag moves.
- Place tags near entrances, corridor junctions, room transitions and other locations where relocalization is valuable.

### Outdoor

- Do not try to cover the entire field with tags. Put surveyed anchors at entrances, turns, loading areas, narrow passages, covered sections, RTK multipath zones and repeatable start/finish locations.
- Tie anchor poses to the same site ENU frame as RTK using an independently surveyed control network.
- Use tags for local pose correction and consistency checks; RTK remains the main global-position observation between anchors when its quality is acceptable.
- At a tag-poor stretch, allow IMU propagation only within a bounded uncertainty/time budget. If RTK also degrades, report DEGRADED/LOST rather than claiming global pose is still accurate.

## Validation plan

1. Unit-test map uniqueness, digest stability, map-frame consistency, corner geometry, stale frame/trigger, clock-domain mismatch, unknown tags, wrong map/calibration revision, code errors and corner RMS.
2. Replay synthetic camera trajectories with tag dropouts, wrong IDs, tag-map revision changes, planar ambiguity, moving/incorrectly surveyed tags, RTK FIXED-to-FLOAT transitions, clock offsets/drift, and IMU gaps.
3. Hardware-in-loop test actual cameras with measured exposure timing; evaluate tag detection rate, corner/reprojection error, relocalization jump, RTK-vs-tag innovation, position/orientation error, latency and correct TRACKING/DEGRADED/LOST behavior.
4. Independently survey outdoor anchor coordinates and the RTK reference point. Validate antenna phase-centre, camera-to-body and antenna-to-body lever arms.

The initial contract and its unit tests are software groundwork only. They do not implement AprilTag/ArUco detection, PnP/pose estimation, the fusion update, or any hardware qualification.
