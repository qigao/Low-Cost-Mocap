# Algorithm-first: trigger synchronization and temporal marker IDs

This Draft PR provides **software contracts and synthetic conformance**, NOT
proof that the physical capture rig meets exposure sync or identification
requirements.

## Current source boundary

- In helpers.py, Cameras._camera_read calls self.cameras.read() and discards
  the second return value. Whether this contains exposure or trigger metadata
  must be verified against the actual camera SDK and device.
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
  ambiguous results. It does not search for a convenient phase shift.
- group_decoded_observations associates positively decoded marker IDs across
  cameras, preserving absent views and rejecting duplicate IDs in one view.
  geometry.triangulate_point and rigid_pattern.fit_rigid_pattern can then
  validate those matches against a fixed 3D marker layout.
- The synthetic end-to-end test includes 3 distinct calibrated intrinsics,
  independent detector permutations, a bit flip, erasures, and a marker
  missing entirely from one camera.

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
