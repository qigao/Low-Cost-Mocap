# Algorithm-first: trigger synchronization and temporal marker IDs

This Draft PR provides **software contracts and synthetic conformance**, NOT
proof that the physical capture rig meets exposure sync or identification
requirements.

## Current source boundary

- In helpers.py, Cameras._camera_read calls self.cameras.read() and discards
  its second return value. Upstream pseyepy.Camera.read returns **software**
  timestamps in seconds, derived from ps3eye_grab_frame microseconds. The
  underlying PS3Eye queue calls gettimeofday when a buffered frame is read,
  **not at sensor exposure**; no hardware trigger ID is returned by this API.
  Source: https://github.com/bensondaled/pseyepy/blob/master/pseyepy/cameras.pyx#L312-L319
  https://github.com/bensondaled/pseyepy/blob/master/pseyepy/src/ps3eye_capi.cpp#L174-L190
  https://github.com/bensondaled/pseyepy/blob/master/pseyepy/src/ps3eye.cpp#L519-L525
  Confirm the deployed pseyepy fork/revision separately; do not treat these
  dequeue timestamps as proof of synchronized camera exposure.
- The _find_dot detector produces only a 2D contour centroid, or [None, None].
  It does not provide a stable physical marker track or illumination levels
  through an LED off phase.
- Pure epipolar geometry proposes candidate matches, but cannot distinguish
  two physically identical bright lights in a single exposure.

## Algorithm primitives

- temporal_pattern.FrameStamp records a camera ID, a hardware trigger ID, an
  exposure timestamp in nanoseconds and an explicit common clock domain.
  validate_trigger_batch fails closed on missing, duplicated, stale, gapped
  or mixed-trigger frames, clock-domain mismatches and excessive skew.
- TemporalCodebook stores each marker's periodic binary on/off sequence.
  Pairwise Hamming distance must exceed twice the intended bit-error budget,
  and codes must actually blink. decode_temporal_marker uses the explicitly
  shared trigger_epoch, supports erasures and limited bit errors and refuses
  ambiguous results. Every alternative marker with an error count
  within the full configured bit-error budget blocks decoding, even if the
  apparent winner has fewer errors; partial erasures cannot silently weaken
  this uniqueness gate. Missing trigger phases count as erasures,
  including omitted records with no explicit None. Exhaustive one-bit-error
  and erasure-mask tests require all accepted IDs to match the true source.
  It does not search for a convenient phase shift.
- light_tracking.BoundedLightTracker uses bounded camera-local predicted
  motion, globally one-to-one assignment and explicit alternative-assignment
  checks. Near-crossing ambiguity, duplicate detections and exhausted track
  capacity fail closed without mutating the previous accepted track state.
  A local track ID is never automatically a physical marker ID. Missing
  bright detections are marked as absent evidence, NOT known dark phases.
  The sample_tracked_light_roi helper explicitly returns an erasure for
  unmatched tracks by default. Sampling a predicted dark ROI requires an
  affirmative predicted_roi_qualified flag from a separate measured motion
  uncertainty / light-tracker quality gate, not from this tracker alone.
- light_intensity.quantize_stamped_track_trace binds every ROI measurement
  to its FrameStamp and camera-local TrackEvidence, rejecting borrowed
  camera IDs, track IDs, trigger numbers, clock domains and stale capture
  timestamps. Unqualified predicted-only ROIs cannot be converted to
  known-OFF samples. This checks metadata consistency only; the hardware
  adapter must independently attest real sensor-exposure provenance.
- light_intensity.sample_light_roi extracts foreground-minus-annulus
  contrast from raw grayscale frames at an independently tracked light
  position. BinaryIntensityCalibration applies per-camera ON/OFF references
  and a dead zone; uncertain samples stay erasures before temporal decoding.
  This requires a separate stable light tracker and reference calibration.
- marker_admission.admit_track_marker_at_frame is the **trusted software
  admission path** between a camera-local tracked light and a decoded 2D
  marker. It derives marker_id from that one trace's calibrated bit history,
  ties it to the last matching exposure stamp in the same decoding cycle,
  and requires an actually observed ON-frame centroid. An old ID from an
  earlier cycle, a borrowed track or a predicted-only dark position cannot
  become a new 3D point. The lower-level grouping API still accepts IDs
  directly for standalone tests and is not a hardware trust source.
- spatiotemporal_reconstruction maps the validated, synchronized 2D
  marker tuple to **physical camera IDs** before triangulation, never
  dense array indexes that break for camera IDs like 2/5/9. It reports
  physical camera IDs for per-view residuals and depths, rejects mismatched
  calibration mappings and fails the entire reconstruction on any admitted
  marker violating geometry gates. A rigid-pattern fit uses original marker
  IDs and never claims measured metric scale from synthetic camera baselines.
- group_synchronized_detections requires decoded 2D marker observations
  to reference the exact FrameStamp of an admitted shared-trigger batch.
  Borrowed frames from different triggers, clocks, or timestamps fail closed
  before triangulation. Camera IDs may be noncontiguous and remain ordered
  as declared in the admitted batch. This is a software consistency guard,
  not proof of hardware timestamp provenance.
- group_decoded_observations associates positively decoded marker IDs across
  cameras, preserving absent views and rejecting duplicate IDs in one view.
  geometry.triangulate_point and rigid_pattern.fit_rigid_pattern can then
  validate those matches against a fixed 3D marker layout.
- An additional controlled synthetic raw-image test links camera-local
  LED tracking, background-corrected ROIs, temporal bits, decoded global
  marker IDs, triangulation and noncoplanar rigid-pattern recovery. It uses
  static markers, so predicted dark ROIs are provably stationary in the test.
  That assumption is NOT established for flying markers.
- The synthetic end-to-end test includes 3 distinct calibrated intrinsics,
  independent detector permutations, a bit flip, erasures, and a marker
  missing entirely from one camera.

## Offline recorded-evidence replay (preparation, not hardware acceptance)

- offline_capture_acceptance.replay_trigger_batches compares each recorded
  multi-camera stamp set with a separately logged reference trigger ID,
  returning admitted/rejected counts, reference gaps, declared timestamp
  skew and named rejection reasons. Clock-domain names and trigger fields
  remain untrusted software claims unless an independently verified hardware
  capture pipeline supplies them.
- replay_labeled_temporal_trials compares decoded marker IDs with external
  ground-truth labels, counting correct accepts, false accepts and
  rejections separately. A zero observed false-ID count still produces
  a **nonzero one-sided 95% Wilson bound**, conditional on independent trials.
  Correlated video frames do not constitute independent binomial trials;
  report the physical capture design and independent labeling protocol.
- Neither helper can declare a real rig hardware-qualified; its report
  hardware_provenance_verified field is always false. Supply actual recorded
  evidence, timebase provenance, shutter timings and calibration certificates
  separately before requesting a hardware or production gate.

## Hardware and timing gates NOT YET VERIFIED

1. Obtain **exposure-associated hardware trigger IDs** from every camera
   and a code epoch shared with the LED controller. Do not derive trigger IDs
   from Python frame counters or receiver arrival timestamps.
2. Verify the timestamps have a physically comparable clock source. The
   software clock_domain name is a declared invariant, NOT clock evidence.
   Measure exposure skew, jitter, dropped and duplicated triggers.
3. Implement tracking of each local marker through illuminated and dark
   phases and a calibrated intensity-to-bit/erasure decision rule. The
   current contour-centroid detector cannot yet do this.
4. Validate code period/frequencies against actual FPS, shutter integration,
   rolling/global exposure, LED switching latency, aliasing and crosstalk.
   An ambiguous code is an unresolved identity, never a geometry fallback.
5. Measure hardware-in-the-loop scale, geometric residuals, marker identity
   confusion, occlusion recovery and total motion-to-control latency before
   enabling live-flight control.

The algorithm interfaces are deliberately NOT connected to the current
camera/serial flight-control path. There is no implicit synchronization
fallback and no evidence of hardware acceptance.
