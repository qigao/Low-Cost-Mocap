# Board Pattern Registry: multiple LED boards (software qualification)

Each physical board is identified by the **global marker IDs** decoded from
the LED timecodes, and oriented by fitting that board's **measured local 3D
layout** to reconstructed 3D marker positions. Board A and board B never
share a global marker ID. The service does not guess the board ID by distance,
relative camera index, or contour detection order.

## Configure each board

Load [board-patterns.example.json](board-patterns.example.json):

- `version` must be `1`. `boards` maps stable `board_id` strings to configurations.
- `markers` is a map from *canonical decimal strings* to marker records.
  A record has `xyz_m` (board-local physical position) and `code`, one binary
  digit per shared exposure trigger.
- `period_triggers`, `max_bit_errors` and optional `pose_phase` are **global**,
  including across different board types. Timecodes are globally checked for
  Hamming distance, not merely for uniqueness inside one board.
- `min_visible_markers` is per board and must be at least 3. For real moving
  rigs, plan for 4+ visible non-collinear markers; more LEDs improve redundancy.
- A configured `pose_phase` must illuminate *every* LED of *every* board.
  In the example: 16 phases, 50% duty per LED, minimum pairwise Hamming
  distance 8 and common all-ON phase 15. This permits multiple real
  same-frame observations for fitting after code identification.
- Optional `body_from_board` contains a proper 3x3 `rotation` and
  `translation_m` defining the board-local -> body-local transform:
  `p_body = R_body_from_board @ p_board + t_body_from_board`.
  The recognizer exports both world-from-board and world-from-body poses.

The JSON loader rejects unknown fields, repeated JSON keys, non-canonical
IDs, duplicate LEDs across boards, invalid codes/coordinates, mirrored
body frames, and collinear model geometry. Registry mappings are read-only.
All validated configuration values are hashed into one stable
`registry.config_sha256`. A controller can report its provisioned revision
for comparison, but matching strings cannot themselves prove hardware
authenticity or exposure synchronization.

## Assign the LED controller states

From `computer_code/api`:

```python
from board_patterns import BoardPatternRegistry

registry = BoardPatternRegistry.from_json_file("board-patterns.example.json")

# trigger_id and trigger_epoch must come from the actual common hardware source.
led_states = registry.led_states_at_trigger(
    "board_A", trigger_id=1015, trigger_epoch=1000
)
# {100: True, 101: True, 102: True, 103: True}
```

The configuration CLI can validate the entire codebook and print a
read-only table with the registry SHA-256, marker ID assignments and
one trigger's desired LED states:

```sh
cd computer_code/api
python board_pattern_cli.py board-patterns.example.json
python board_pattern_cli.py board-patterns.example.json \
  --board board_A --epoch 1000 --trigger 1015
```

Use `--expect-sha256 <revision>` to reject a different deployed pattern.
This outputs **desired bits**, NOT firmware programming, a hardware transport,
or proof of camera sync. The LED controller and cameras must share an
independently validated trigger/epoch
([HW-01](https://github.com/qigao/Low-Cost-Mocap/issues/2) and
[HW-03](https://github.com/qigao/Low-Cost-Mocap/issues/4)).

## Server-side multi-board recognition

1. Obtain hardware-proven exposure `FrameStamp` records for the cameras.
2. Track each local LED through ON/OFF frames and record calibrated, stamped
   ROI evidence and explicit erasures as `LocalTrackCycle` records. A
   predicted dark ROI requires a separately measured motion-quality gate.
3. Prefer `recognize_tracked_boards()`: it decodes **directly from these local
   traces using the active registry codebook**. A caller cannot supply a
   marker ID or reuse an old decoder's result as a new board identity.
   It checks local-track uniqueness and can compare a controller-reported
   `config_sha256` against the server registry.
4. It then validates common-trigger frames and pose-phase alignment, groups
   LEDs by the owning board, triangulates using physical camera ID
   calibration, and independently fits each board's rigid layout.
5. Each board returns `recognized`, `insufficient_markers`,
   `geometry_rejected` or `not_visible`, with `pose=None` on failure. A
   missing board never inherits the last valid frame's pose.

```python
from board_patterns import BoardPatternRegistry
from board_recognizer import recognize_tracked_boards

registry = BoardPatternRegistry.from_json_file("board-patterns.example.json")

report = recognize_tracked_boards(
    track_cycles,                   # LocalTrackCycle from calibrated ROI tracks
    frames,                         # one FrameStamp per physical camera
    expected_camera_ids,            # e.g. [2, 5, 9]
    camera_poses_by_id,
    intrinsics_by_id,
    contrast_calibration_by_camera_id,
    registry,
    trigger_epoch=controller_epoch,
    max_exposure_skew_ns=measured_skew_limit,
    controller_config_sha256=controller_reported_pattern_sha256,
)
for board_id, outcome in report.boards.items():
    if outcome.pose is not None:
        board_position = outcome.pose.translation_world_from_board
        body_rotation = outcome.pose.rotation_world_from_body
    else:
        # Handle the failed/missing board explicitly; no stale pose.
        pass
```

The lower-level `recognize_admitted_boards()` remains available for
standalone conformance tests and callers with already verified
`AdmittedMarker` instances. New service integrations should use the
preferred trace-to-ID entry point to avoid cross-configuration ID reuse.

When called **only** on the common pose phase, `allow_trigger_gaps=True` must
be explicitly selected for successive pose-phase frames; all intervening
hardware triggers must still be verified by an external capture monitor.
It is not a fallback to inferred synchronization.

## Per-phase rigid-body visibility: a separate design gate

The original [balanced example](board-patterns.example.json) uses 50% LED
duty cycles; only phase 15 contains three or more illuminated noncollinear
LEDs on **both** boards. Identifying a marker is not the same as obtaining a
current-frame 6DoF observation. At nominal 90 FPS a rigid-body pose produced
only once per 16-phase cycle has an optical measurement cadence of **5.625
Hz**, not 90 Hz.

[board-patterns.continuous.example.json](board-patterns.continuous.example.json)
is an **alternative software-only candidate**: four LEDs per board with
81.25% nominal duty each. Three are illuminated in phases 0..11, all four
in phases 12..15, and every active subset is noncollinear. The eight
timecodes have minimum *global* Hamming distance 4, supporting a declared
single-bit corruption budget with strictly reduced erasure tolerance
compared with the original distance-8 pattern. Maximum cyclic dark run
is two triggers. The candidate omits \`pose_phase\` to avoid artificially
restricting the recognizer to one phase; the final four phases remain ON.

Use the non-hardware phase audit:

\`\`\`sh
cd computer_code/api

# Reports why the original 50%-duty pattern does NOT cover every phase.
python board_pattern_cli.py board-patterns.example.json --phase-report

# Intentionally fails for the original pattern.
python board_pattern_cli.py board-patterns.example.json --require-every-phase

# Design-only per-phase geometry checks pass for the alternative schedule.
python board_pattern_cli.py board-patterns.continuous.example.json --require-every-phase
\`\`\`

**Scope of this gate:** it proves illumination *planning* and noncollinear
model geometry only. It does not prove LED contrast, shutter/phase alignment,
camera viewing angle, marker recognition in the presence of erasures or a
physical world's known metric scale. A 81.25% duty cycle may increase
controller power and heating and must be evaluated on the physical board.

Crucially, current \`admit_track_marker_at_frame()\` still decodes from
bounded samples **within one coding cycle**. Running a nominally visible
code at every trigger does not automatically preserve the physical marker
ID across cycles, and the existing recognizer does not yet measure a new
Pose on every camera exposure. A separate **identity-locked continuous
tracker** is required, with real current-frame observations, periodic
code revalidation, prediction explicitly labelled as non-measured, and
hard stale/invalid states.

## Limits / hardware gates

- The inspected upstream pseyepy `Camera.read` timestamps host dequeue, not
  sensor exposure. Matching string clock domains or synthetic trigger IDs
  cannot establish real exposure synchronization.
- At nominal 90 FPS, a 16-trigger code spans about **178 ms** before initial
  complete-cycle identification. Continuous 6DoF at every exposure would
  additionally require a validated persistent ID lock, motion tracking, and
  freshness/revalidation policy; **that runtime feature is not implemented**.
- The existing `helpers.locate_objects` and real serial flight-control
  pipeline are deliberately **unchanged**. No production switch is enabled.
- Board coordinates use labelled metres, but the output flag
  `metric_scale_validated=False` remains until independently measured camera
  world scale, spatial calibration, latency and flight-control safety pass
  [HW-01–HW-06](../../docs/HARDWARE_VALIDATION.md).
