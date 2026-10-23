# RTK + IMU + Visual Tags: Indoor Map and Outdoor Anchor Profiles

**Status:** architecture proposal plus provider-neutral tag-map/observation contracts. No camera/tag detector or outdoor hardware is qualified by this document.

## Decision

Use one timestamped, quality-aware state-estimation architecture with separate environment profiles:

- **Indoor dense-map profile:** standardized visual tags are fixed landmarks in a surveyed/calibrated indoor map. IMU propagates state between tag observations. RTK is normally absent or treated as unavailable.
- **Outdoor sparse-anchor profile:** RTK provides georeferenced position and its original FIXED/FLOAT/DGNSS/SINGLE/INVALID quality state; IMU provides high-rate propagation; fixed visual tags at selected waypoints provide local pose corrections, map association, and relocalization checks.
- **Shared estimator:** consume only observations with explicit acquisition time, clock domain, camera calibration/map revision, and quality. Preserve TRACKING, DEGRADED and LOST states. Do not fabricate a full 6DoF pose from RTK position alone.

The initial implementation remains a **moving camera observing fixed environmental tags** (inside-out localization). The existing Outside-In mode—fixed external cameras observing a tag/marker constellation on the moving body—is a distinct profile. Environmental anchors do not replace body-mounted markers in that topology.

## Sparse Tags + SLAM: avoid dense tag coverage

The tag-map profiles above are anchor maps, not a requirement to cover every square metre with printed tags. The preferred extension is visual-inertial SLAM with optional sparse fiducial anchors:

SLAM pose observations carry an explicit position unit (`meters` or `map_units`). Do not label an arbitrary-scale monocular map as metres. A pose labelled `meters` must carry `metric_scale_qualified=true` plus a non-`none` independent `scale_reference`; arbitrary-scale poses use `map_units`, `metric_scale_qualified=false` and `scale_reference=none`. `MapAlignment` records the local-to-world similarity transform, but `scale_m_per_source_unit` remains `None` until independently qualified—using `1.0` as an unknown-scale placeholder is forbidden. `LOST` and `INVALID` are pose-free `LocalizationStatus` transitions; they must never carry the last pose forward as if it were a current measurement.

- **Indoor:** visual SLAM/VIO tracks natural visual features and IMU motion between anchors. Standardized Tags at entrances, corridor junctions, repetitive/texture-poor areas and relocalization points constrain map drift and provide repeatable map-frame references. Monocular-only SLAM may have unknown metric scale; stereo, depth, a calibrated baseline or independently surveyed anchors must qualify scale before metric RTK/tag fusion.
- **Outdoor:** visual-inertial odometry/SLAM supplies local motion and map continuity; RTK supplies global position when its solution and correction age pass quality gates; sparse surveyed Tags at selected waypoints provide local pose/map corrections and checks. Under sunlight, low texture, motion blur, foliage or occlusion, SLAM quality can degrade, so the estimator must propagate uncertainty and report DEGRADED/LOST rather than assume the map remains accurate.
- **Loop closure:** a loop closure can correct accumulated map drift, but must be checked for false data association and must not silently change the active map revision or world frame. Downstream consumers need an explicit map-alignment revision and a controlled pose correction/relocalization event.
- **One estimator, distinct topologies:** in **inside-out**, a moving camera observes natural features and fixed environmental Tags; SLAM estimates the camera/body pose. In **outside-in**, fixed cameras observe a moving body's rigid Tag/marker constellation; this remains a Mocap problem. SLAM can help map/align the fixed camera network or support moving-camera subprofiles, but environmental SLAM alone does not recover a moving body's pose from static external cameras. Keep the sensor topology and tracked frame explicit in every observation.
- **No tag carpet:** add Tags where they improve observability, map alignment or recovery. Do not infer that sparse Tags can guarantee localization through arbitrary textureless/featureless corridors or long outdoor stretches; the bounded unanchored travel distance must be measured for each sensor and environment.

## Map and observation contracts

The provider-neutral localization module is `computer_code/api/localization_contract.py`.

- `MapPoseObservation` carries only a current usable TRACKING/DEGRADED pose, explicit topology/frame/time/revisions, uncertainty, and fail-closed metric-scale qualification.
- `LocalizationStatus` carries pose-free TRACKING/DEGRADED/LOST/INVALID state transitions and an optional last-pose timestamp for diagnostics; it has no position/quaternion fields.
- `MapAlignment` carries local-to-world similarity alignment. Its numeric scale is absent until an independent scale reference qualifies it.

The new provider-neutral module is `computer_code/api/fiducial_contract.py`.

- `TagAnchor`: family/ID, map frame, physical size, map-from-tag pose, uncertainty, and reference method.
- `FiducialTagMap`: `indoor_dense` or `outdoor_sparse_anchors`, a single map frame, unique family/ID pairs, immutable lookup, and deterministic SHA-256 revision digest.
- `FiducialObservation`: family/ID, camera serial, frame/trigger IDs, sensor-exposure timestamp and clock domain, ordered image corners, corrected code-bit count, corner RMS, camera-calibration digest, and tag-map digest.
- `validate_fiducial_observation`: fail-closed checks for frame/time/camera identity, active map and calibration revisions, known tag, code errors and corner quality.

Detector backends are isolated behind `computer_code/api/fiducial_detector_adapter.py`. A backend receives an opaque image payload and may return only tag family/ID, ordered image corners, corrected-bit count and corner quality. Camera identity, frame/trigger IDs, exposure timestamp/clock domain, calibration digest and active map digest come from a host-owned `DetectorFrameContext` and are injected only during normalization. Backend exceptions are not silently retried with a different detector, and duplicate tag identities or families outside the active profile fail the frame.

Outdoor anchor maps require surveyed tag poses. Indoor maps may use surveyed or calibrated tag poses. A hash establishes revision identity, not that a survey or timestamp is physically trustworthy. The camera adapter must source timestamps from documented exposure metadata, not packet arrival time.

## Estimation and fusion

For a detected tag, estimate camera-from-tag from its known physical size, ordered corners, and calibrated camera intrinsics. The single-tag solver is `computer_code/api/fiducial_pose.py`: tag coordinates are right-handed (+X right, +Y up, +Z out of the printed front), and detector corners must correspond to decoded top-left, top-right, bottom-right, bottom-left in that frame. The solver supports no-distortion and Brown5 calibration, initializes from a planar homography, refines reprojection error, and evaluates a second deterministic planar-pose seed. It rejects insufficient image area, non-positive depth, back-facing/excessively oblique views, excessive reprojection residual, and competing positive-depth planar solutions whose residual margin is too small. Its result is only `camera_from_tag`; it does not yet claim a map/body pose.

Combine multiple visible tags with `computer_code/api/fiducial_multitag_pose.py`. Each accepted single-tag `camera_from_tag` measurement is independently composed with the anchor's `map_from_tag` transform to form a `map_from_camera` hypothesis. Consensus is performed in pose space using bounded translation/orientation disagreement: a minority inconsistent tag can be excluded, while distinct equal-support pose clusters fail closed. The aggregate retains explicit inlier/excluded tag IDs and residual bounds. This output is still a camera pose; camera-to-body composition and the final `MapPoseObservation` belong to the next layer.

For a moving camera observing fixed tags:

1. Resolve the tag observation against the active map revision.
2. Estimate camera pose in the map frame from the known tag pose and camera-from-tag measurement.
3. Convert camera pose to body pose using `computer_code/api/fiducial_localization.py`. `CameraBodyExtrinsic` explicitly defines body-from-camera mounting; `FiducialUncertaintyModel` is a versioned software model that combines declared base solve uncertainty, anchor-map uncertainty and multi-tag pose residuals. The final `MapPoseObservation.calibration_sha256` hashes the camera calibration revision, camera/body extrinsic revision and uncertainty-model revision together. A two-tag or high-residual accepted consensus is reported as DEGRADED rather than silently promoted to TRACKING. Calibrated indoor maps use `calibrated_tag_map` as the explicit metric-scale reference; fully surveyed inlier anchors use `surveyed_tags`.
4. Fuse timestamp-aligned visual pose with IMU propagation and, outdoors, RTK position after converting WGS-84 coordinates into the configured local ENU frame. The first T4 vertical slice is `computer_code/api/localization_imu_fusion.py`: `MapPoseObservation` now carries an explicit source sequence number, localization orientation uncertainty is converted into a per-measurement attitude variance, and the existing bounded IMU attitude-fusion state machine is reused through a provider-neutral wrapper. TRACKING poses may correct attitude; DEGRADED poses are rejected by default and require an explicit opt-in with variance inflation. This slice still publishes no fused global position or velocity.
   Translational propagation is `translational_propagation.py`: it requires a TRACKING metric seed, explicit initial velocity and residual accelerometer-bias uncertainty, an explicit gravity vector in the same map frame, corrected specific force and a same-timestamp fused attitude. The conditioned estimator is `translational_eskf.py`, with a full 9×9 position/velocity/bias covariance and transactional absolute-position updates. Tag and RTK both enter through `MetricPositionMeasurement`; delayed arrivals use `delayed_position_replay.py`, bounded by retained frame/time windows; exact-frame updates and interior acquisition epochs are replayed by splitting propagation between real IMU samples without inventing source sequence numbers. Every provider must be explicitly pinned by a `PositionSourceContract` containing source, calibration and clock-mapping revisions.
5. Apply the calibrated GNSS antenna-to-body lever arm using the estimated orientation. The RTK adapter is `computer_code/api/rtk_position_measurement.py`: `RTKPositionCalibration` binds receiver/body identity, site ENU origin, antenna lever arm, receiver-reported accuracy semantics and quality thresholds into deterministic source/calibration revisions. The fused attitude state carries an explicit `map_frame_id`; RTK lever-arm correction is accepted only when that frame matches the ENU/map frame and the mapped RTK solution timestamp exactly matches the current attitude timestamp.
6. Gate innovations by covariance and source quality. Only RTK FIXED solutions that pass correction-age, horizontal/vertical-accuracy and satellite-count gates become `MetricPositionMeasurement` updates; FLOAT/invalid/stale solutions do not enter the position filter. Receiver-reported accuracy is converted to per-axis standard deviation only through explicit configured scale factors, and lever-arm orientation uncertainty is added to the position uncertainty. RTK uses its own RTK-to-estimator clock-mapping revision and its own source sequence domain. A tag relocalization that conflicts with RTK is rejected by the common position NIS gate rather than silently averaged. The ESKF must be constructed with `position_source_contract_from_rtk(calibration, mapping, expected_target_clock_domain=...)`; this binds the RTK source/map revision, receiver/lever-arm/accuracy calibration digest, and the exact RTK clock mapping used to produce measurements.
   For position/velocity/bias cross-correction, `translational_eskf.py` maintains a full 9×9 covariance for p/v/residual accelerometer bias while conditioning on the separately fused attitude. IMU propagation creates cross-covariance; Tag or RTK position innovations update p, v and residual bias through the Kalman gain. Covariance uses a Joseph-form measurement update and bounded residual-bias correction. `PositionSourceContract` pins the accepted source revision, receiver/map calibration and clock-mapping revision for the entire estimator session; unconfigured RTK or a changed digest is rejected. This is not yet a joint attitude/translation INS: attitude error and gyro bias remain outside this translational covariance, and the translational estimator is not a joint attitude/translation INS. Delayed position updates between retained IMU timestamps are supported by exact-epoch interval splitting, but hardware timing and fusion qualification remain separate gates.
   The application-facing closure is `localization_state_stream.py`: a same-timestamp attitude state and translational ESKF state are combined only after body/IMU/map/clock/revision invariants match. TRACKING/DEGRADED are derived from both component states; if either component is LOST the unified full localization state becomes LOST and hides all dynamic pose/velocity/bias/covariance fields. This is an additive localization protocol and does not reinterpret the existing optical PoseStream v1 payloads.
   Delayed absolute-position arrival is handled by `delayed_position_replay.py`, not by replacing acquisition time with host arrival time. A delayed Tag or RTK measurement must fall within the retained IMU/attitude time window. Exact-frame measurements are applied at the frame; interior measurements split the enclosing interval at their acquisition epoch while retaining the two real bracketing IMU sequence numbers. The translational ESKF rewinds to an explicit configuration-bound checkpoint, inserts the measurement in canonical per-source order, and replays to the latest state transactionally. A delayed update can recover a currently LOST translation only when a valid earlier checkpoint and measurement remain inside the window; otherwise explicit full-state reseed is still required.

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

The tag-map/observation contracts, detector adapter, calibrated single-tag pose solver, multi-tag consensus, camera-to-body localization, attitude fusion, translational ESKF and bounded delayed-position replay are software implementations with deterministic tests. They do not establish physical camera exposure provenance, surveyed metric accuracy, real-hardware detector performance, or outdoor hardware qualification.
