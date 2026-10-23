# General Optical–IMU Motion-Capture Architecture

**Status:** architecture and implementation plan; software-only until hardware evidence gates pass.  
**Tracking topology:** fixed external cameras observing active LED/IR markers (**Outside-In**).  
**Primary objective:** reusable motion-capture and rigid-body state estimation, not drone flight control.

## 1. Product definition

Low-Cost-Mocap is to become a general motion-capture engine that turns synchronized optical observations and optional body-mounted IMU measurements into timestamped, quality-qualified rigid-body state. Drone control, robotics, animation, biomechanics, and visualization are downstream consumers. None is allowed to define the core estimator's types or lifecycle.

The current repository contains temporal LED codebooks, identity-locked tracking, physical-camera-ID keyed calibration, multiview geometry, rigid-pattern fitting, PoseStream, bounded recording/replay, capture-bundle verification, a read-only observer, a typed IMU sample contract, affine clock-mapping primitive and immutable IMU calibration contract. These are software primitives, not an end-to-end hardware-qualified system. In particular, the existing `FrameStamp` and clock-sync anchors are caller-supplied declarations, not proof that a physical sensor used that clock or trigger.

## 2. System boundary

```text
Camera / LED detector ─┐
LED trigger/controller ├─> typed sensor adapters ─> clock alignment
Body-mounted IMU ──────┘                              │
                                                      v
Immutable pattern + calibration ─> optical measurements + IMU samples
                                                      │
                                                      v
                                     bounded optical/IMU state estimator
                                                      │
                                                      v
                         MotionState / PoseFrame (quality + provenance)
                               ┌──────────────────────┼────────────────┐
                               v                      v                v
                         PoseStream/replay       Observer/UI     control consumer
                                                                   (separate)
```

Adapters own device APIs and vendor-specific metadata. The core owns validation, clock/coordinate semantics, optical geometry, inertial propagation, fusion, quality state and replay determinism. A consumer cannot feed control commands back into the estimator.

## 3. Canonical contracts and conventions

### Coordinates

- `world`: capture-volume frame; its origin/orientation is explicit in calibration metadata.
- `camera`: existing geometry uses world-to-camera extrinsics, `x_camera = R_camera_from_world @ x_world + t_camera_from_world`.
- `board`: marker coordinates in metres as declared by the immutable pattern registry; the declaration alone does not prove metrological accuracy.
- `body`: rigid-body frame. Existing PoseStream reports `position_world_from_body` and `quaternion_world_from_body_xyzw`.
- `imu`: sensor-native frame, with a calibrated rigid transform `T_body_from_imu`. Never silently treat the IMU axes as body axes.

All transforms must state source and destination frames in the name or type. Rotation matrices must be proper rotations; quaternions must declare component order. SI units are used at adapter boundaries: seconds/nanoseconds for time, radians/s for angular rate, m/s² for specific force, metres for calibrated translation.

### Sensor sample envelope

Every camera, LED-controller, and IMU observation should carry:
- stable physical `device_id` and `sequence_number`;
- sensor acquisition timestamp and declared `clock_domain`;
- status/validity and explicit saturation/overrun/missing-data flags;
- calibration and configuration revision IDs;
- host receive time as separate diagnostics only, never substituted for acquisition time.

For IMU samples, additionally include gyro xyz, accelerometer xyz, units, scale/bias calibration revision and sensor status. If the device cannot provide acquisition time, the adapter must mark timing as unqualified; it cannot invent a hardware timestamp from dequeue time.

### Output state

The core state record should include:
- body ID and acquisition time / clock domain;
- world-from-body pose, velocity only when observable, and estimator uncertainty/covariance;
- source flags (optical, IMU propagation, fused correction), age of last optical correction, and calibration/pattern revisions;
- explicit state `TRACKING`, `DEGRADED`, `LOST`, or `INVALID`, with reason codes;
- metric-scale and hardware-provenance flags that remain false until independent qualification.

A predicted state must never be labeled as a fresh optical measurement. Existing PoseStream v1 is optical-only and has strict MEASURED/INVALID semantics. Add fusion output as a new version or a separate versioned message kind; do not silently reinterpret v1 fields.

## 4. Estimation strategy

### Optical measurement path

Reuse the current modules in this order:
1. Validate synchronized exposure batch and clock domain.
2. Detect and track local bright LEDs with bounded local IDs.
3. Decode time codes and admit global marker IDs only from validated code history. The codebook exposes its minimum pairwise Hamming distance and guaranteed erasure budget `d_min - 2t - 1` for `t` correctable bit errors; this bound assumes a known trigger epoch and phase. Each successful decode reports whether the actual erased-sample count stayed within that guaranteed budget. A unique decode beyond the bound may still be valid for that particular trace, but it is outside the codebook's worst-case guarantee and should be treated as degraded evidence by safety-critical consumers. The production marker-admission boundary rejects over-budget traces by default; only explicit diagnostic calls may inspect such a trace without creating an identity lock. Active lock metadata is published with the pose frame so downstream consumers can audit the per-camera Track, decoded ID, bit errors, erasures, and codebook guarantee.
4. Associate cross-camera observations, triangulate, reject ambiguous/outlier geometry, and fit registered rigid-body geometry.
5. Produce an optical pose measurement with timestamp, residuals, support, and calibration revision.

When a board declares `min_visible_markers=3`, the registry checks every three-marker subset using a scale-normalized triangle-quality threshold. A board whose full point cloud is non-collinear can still be rejected if an occlusion leaves a collinear or nearly collinear triple. Track continuity proposes camera-local associations; the temporal code must admit global marker IDs; rigid geometry validates pose observability and rejects ambiguity. Geometry never overrides an unresolved temporal identity.

The pattern-observability report also exposes codeword duty cycle and global LED load
per trigger phase. The current balanced example uses 8 ON phases per marker
(duty cycle 0.5), minimum Hamming distance 8 and a five-erasure worst-case
budget, but its configured geometry supports a nominal pose for both boards
only at phase 15. The high-visibility example uses 13 ON phases per marker
(duty cycle 13/16), minimum distance 4 and a one-erasure budget; its model
geometry supports a nominal pose at every phase. These are a software design
trade-off in synthetic configurations, not measured hardware performance.

For a predicted dark ROI, the pipeline accepts only a `DarkROIQualification`
record bound to the current camera-local Track, trigger, registered marker ID,
pattern SHA-256, and an external expected image position. The residual from
the Track prediction must remain below the configured pixel limit, and the
registered marker must be OFF at that exact trigger phase. A verifier ID and
reference-evidence digest are retained for audit; the digest is not a signature
and does not make an untrusted verifier trustworthy. If the external motion
model or reference is unavailable, leave the sample as an erasure.

No proximity-only reassignment, inferred trigger phase, stale pose reuse, or silent metric-scale assumption is allowed.

### IMU path

1. Validate monotonic acquisition timestamps, units, finite values, saturation, sequence gaps and maximum integration interval.
2. Apply calibrated sensor bias/scale and `T_body_from_imu`.
3. Propagate attitude and velocity using bounded gyro/accelerometer integration or preintegration, with explicit bias/noise models. The first translational primitive is `translational_propagation.py`: a TRACKING metric localization pose is mapped into the IMU clock, while initial velocity and residual accelerometer bias remain explicit inputs because one position sample cannot observe them. Gravity is an explicit vector bound to the same map frame. Corrected specific force is rotated by a same-timestamp fused attitude and integrated with bounded steps and growing uncertainty.
4. Reset/reinitialize on clock reset, invalid calibration, saturation or excessive gaps rather than integrating across unknown time. Translational dead reckoning has a separate bounded horizon; after expiry position/velocity are hidden and an explicit position reseed is required. A nonzero IMU-to-body lever arm is rejected in this first primitive because rotational lever-arm acceleration compensation is not yet modeled.

IMU translation drifts when acceleration is integrated twice. It cannot provide indefinitely stable global position without optical/other independent constraints. Optical measurements correct accumulated drift; IMU samples provide high-rate propagation between optical updates. Gravity does not by itself make yaw or global position observable.
The first position correction layer uses a common `MetricPositionMeasurement` and a current-time diagonal innovation update. It reduces position uncertainty and resets the bounded dead-reckoning age, but deliberately leaves velocity and residual accelerometer bias unchanged because the current primitive does not carry position–velocity–bias cross-covariance. Measurements outside the current estimator timestamp fail closed instead of rewinding state. A configurable normalized-innovation-squared gate rejects inconsistent position updates atomically.
The next estimator layer is `translational_eskf.py`, a conditioned 9-state error-state filter over position, velocity and residual accelerometer bias with a full 9×9 covariance. Fused attitude remains an external input rather than part of this covariance. IMU propagation creates p/v/bias cross-covariance using the calibrated accelerometer noise density and bias random walk; current-time metric position updates use a Joseph-form covariance update so position innovations can legitimately correct velocity and residual bias. Bias corrections are bounded. This is still not a full 15-state INS because attitude error/gyro bias are outside the state and delayed measurements are not replayed.

### Fusion choice

Do not select an EKF or factor graph by fashion. First specify state, frames, process/measurement equations, noise units, initialization, observability and failure transitions. The first vertical slice should fuse optical orientation/pose with IMU gyro propagation during a short optical dropout, with covariance/quality and bounded dropout duration. Then add accelerometer-based velocity/position propagation and bias estimation, only with independent ground-truth benchmarks. Keep optical-only output as the baseline for every A/B test.

## 5. Required failure behavior

- Unknown, mixed or stale clock mappings: reject fusion update.
- Out-of-order sample: reject or route to a bounded reorder window explicitly; never silently rewind estimator state.
- IMU saturation, nonfinite samples, calibration mismatch or gap above limit: stop propagation and transition to DEGRADED/INVALID.
- Ambiguous marker identity, insufficient parallax/views or failed rigid fit: no new optical measurement.
- Optical dropout: permit bounded IMU propagation only with explicit predicted/degraded status and uncertainty growth; on expiry, output LOST/INVALID with no pose usable as a fresh measurement.
- Reacquisition: require optical geometry and identity gates; apply a bounded innovation/consistency test, and reject a discontinuous correction instead of hiding a jump.
- Capacity exhaustion: fail closed; never drop an arbitrary middle sample and continue as though the sequence were complete.

## 6. Implementation milestones

| ID | Work item | Exit criterion | Status |
|---|---|---|---|
| A1 | Canonical frame, units, clock and state contracts | Typed contracts and conversion tests; no ambiguous transform naming | **Merged** — PR #34 |
| A2 | IMU sample schema and adapter boundary | Schema validation for IDs, sequence, timestamp, units, saturation and calibration revision | **Merged software contract** — PR #36 |
| A3 | Clock alignment service | Tests for offset, drift, reset, skew, out-of-order data, stale mapping and unknown clock | **Merged software primitive** — PR #36; hardware clock qualification remains open |
| A4 | IMU/body extrinsics and noise/bias calibration | Immutable revisioned calibration and reproducible calibration fixture | **Merged software contract** — PR #39 |
| A5 | Bounded inertial propagation | Static, constant angular rate, bias, sample-gap, saturation and reset tests | **Merged gyro-only attitude primitive** — PR #43; global position/velocity and optical fusion remain unavailable |
| A6 | Optical/IMU fusion vertical slice | Short optical dropout: orientation propagation, uncertainty growth, optical reacquisition correction; optical-only A/B report | **Software-only attitude fusion primitive added in PR #45**; independent-reference A/B qualification and hardware timing remain open |
| A7 | Versioned fused state output | New schema does not weaken PoseStream v1; consumers can distinguish measured and predicted state | **Merged** — PR #47; additive `fused_attitude_state` validated by independent consumer and journal |
| A8 | Deterministic capture/replay | Same bundle + same config reproduces state sequence within declared tolerances | **Merged software harness** — PR #49; bounded typed-event replay with config/input/output digests; hardware capture qualification remains open |
| A9 | Benchmark/qualification | Independent reference, position/orientation error, drift, reacquisition jump, latency percentiles and invalid-output rate | **Software orientation reporting present** — PR #49 single-series metrics, PR #51 paired optical-only/fused comparison, PR #52 strict offline CLI; physical reference and latency qualification remain open |
| A10 | Real hardware adapters | Only after HW-01–HW-05 supply independently verifiable evidence; HW-06 remains separate | Blocked on hardware evidence |

## 7. Validation matrix

Synthetic tests are necessary but never close hardware issues. The software matrix should cover:
- static body, constant angular velocity, multi-axis rotation and bounded translation;
- LED code aliasing, local-track ambiguity, camera dropout and rigid-body degeneracy;
- IMU offset/drift, clock reset, sample loss/reorder, saturation, bias and scale error;
- short/long optical occlusion and reacquisition with an independently declared acceptance limit;
- incorrect IMU extrinsic, changed calibration revision and mixed clock domain;
- deterministic replay, capacity limits, and stale/predicted state serialization.

Report optical-only and fused metrics separately. Do not tune thresholds on the held-out acceptance sequence. Publish position error, orientation geodesic error, drift vs. dropout duration, reacquisition discontinuity, end-to-end latency p50/p95/p99, and false-valid output count.

## 8. Repo module mapping

| Existing module | Intended role |
|---|---|
| `board_patterns.py`, `temporal_pattern.py`, `board_recognizer.py` | Pattern definition, temporal identity and board recognition |
| `synchronized_light_pipeline.py`, `light_tracking.py`, `light_intensity.py` | Camera-local tracking and evidence admission |
| `multiview_association.py`, `multiview_consensus.py`, `geometry.py`, `spatiotemporal_reconstruction.py`, `rigid_pattern.py` | Optical reconstruction and rigid-body measurement |
| `calibration_geometry.py`, `bundle_geometry.py` | Calibration geometry; metric scale remains a separately qualified claim |
| `identity_locked_board_tracker.py` | Current-frame identity-locked optical state |
| `imu_contract.py` | Typed SI-unit IMU samples, stream-order/gap guard and affine clock mapping from declared synchronization anchors |
| `imu_calibration.py` | Immutable IMU-to-body extrinsics, bias/noise/correction calibration and bias-corrected vector transforms |
| `imu_propagation.py` | Bounded gyro-only attitude propagation; no global position/velocity or optical fusion |
| `translational_propagation.py` | Bounded accelerometer position/velocity dead reckoning from an explicit metric seed, gravity model and fused attitude; residual accel bias state/uncertainty retained; nonzero IMU lever arm rejected until compensation exists |
| `position_measurement.py` | Provider-neutral clock-aligned metric position correction boundary for localization and RTK; current implementation adapts MapPoseObservation and carries diagonal position uncertainty/source revisions |
| `rtk_position_measurement.py` | Qualified RTK FIXED -> site ENU -> body-origin adapter with explicit receiver/ENU/lever-arm calibration, accuracy semantics, clock mapping and fused-attitude frame checks |
| `translational_eskf.py` | Conditioned 9-state position/velocity/residual-accel-bias error-state estimator with full covariance, bounded IMU propagation and NIS/Joseph-form absolute-position updates |
| `optical_imu_fusion.py` | Bounded optical attitude correction over interpolated gyro history, isotropic uncertainty, explicit TRACKING/DEGRADED/LOST states; no position/velocity |
| `fused_attitude_stream.py` | Additive strict fused-attitude payload and state serializer; existing optical PoseStream v1 remains unchanged |
| `fusion_replay.py` | Bounded deterministic typed-event replay, canonical digests and quaternion geodesic reference metrics |
| `fusion_ab_report_cli.py`, `fusion_ab_bundle_report_cli.py`, `docs/FUSION_AB_REPORT.md` | Strict versioned optical-vs-fused report; bundle-gated mode binds output to verified capture-bundle/artifact hashes without asserting hardware authenticity |
| `pose_stream.py`, `pose_stream_broker.py`, `pose_stream_recording.py` | Versioned output, bounded distribution and deterministic recording/replay |
| `capture_bundle_manifest.py`, `bundle_gated_offline_replay.py` | Evidence integrity and gated offline qualification |
| `index.py`, legacy serial handlers | Legacy app/controller adapter only; not core state estimation |

The IMU sample contract, clock-mapping primitive, calibration contract, bounded gyro-only attitude propagation, software-only optical attitude fusion, versioned fused-attitude output, and deterministic replay/reference-metric primitives are present. The paired optical-only-versus-fused report is tracked in Issue #50. There is still no hardware-qualified clock/trigger adapter, accelerometer-based inertial position/velocity propagation, full ESKF/factor-graph estimator or independently qualified optical-IMU fusion.

## 9. Hardware gates and scope

Keep HW-01 (#2) exposure/trigger provenance, HW-02 (#3) calibration/metric scale, HW-03 (#4) physical optical time code, HW-04 (#5) moving-marker/occlusion safety, HW-05 (#6) independent labeled recordings and HW-06 (#7) latency/control safety open until their own evidence is accepted.

This plan does not switch production camera capture, enable live flight, implement NatNet, or certify a controller. Legacy control hardening is a separate security track; it must not block or dictate the reusable estimator API.

## References

- Hicks et al., *Sensor Fusion for Enhancing Motion Capture: Integrating Optical and Inertial Motion Capture Systems* (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12349364/
- OptiTrack Motive IMU sensor fusion documentation: https://docs.optitrack.com/v3.2/motive/imu-sensor-fusion
- *Fusion of unsynchronized optical tracker and inertial sensor in EKF*: https://www.dfki.de/fileadmin/user_upload/import/9141_UnsynchronizedVisualInertial.pdf


## 8. Outdoor RTK + active-NIR extension

The indoor Outside-In optical pipeline remains the baseline. An outdoor profile may add RTK GNSS as a global-position observation and active NIR as the optical marker channel, with optional IMU propagation. The proposed operating modes, wired camera-sync daemon, sensor timestamp requirements, RTK lever-arm handling, sunlight/optical constraints and hardware acceptance plan are specified in [Outdoor RTK + Active-NIR Architecture](OUTDOOR_RTK_NIR_ARCHITECTURE.md) and tracked by [issue #74](https://github.com/qigao/Low-Cost-Mocap/issues/74).

The provider-neutral contract in `computer_code/api/outdoor_sensor_contract.py` validates camera-frame metadata, synchronized batches and RTK solution status. It does not include a vendor camera SDK or claim that PTP state proves physical exposure alignment. HW-01 remains open until an independent trigger/exposure reference and real hardware recording qualify the adapter. A single RTK antenna is only a position observation; orientation must come from optical geometry, a qualified IMU fusion path or a separately qualified multi-antenna heading solution.
