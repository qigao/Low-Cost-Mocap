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
IDs, duplicate LEDs across boards, invalid codes/coordinates, a mirrored
body frame, and collinear model geometry. Registry mappings are read-only.

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

This produces **desired bits**, NOT a hardware transport or proof of camera
sync. The LED controller and cameras must share an independently validated
hardware trigger/epoch ([HW-01](https://github.com/qigao/Low-Cost-Mocap/issues/2)
and [HW-03](https://github.com/qigao/Low-Cost-Mocap/issues/4)).

## Server-side multi-board recognition

1. Obtain a hardware-proven exposure `FrameStamp` for every physical camera.
2. Maintain each LED's local track, sample calibrated raw ROI contrast, and
   preserve erasures. `admit_track_marker_at_frame()` converts each proven
   local track into a `StampedIdentifiedDetection` with an actual ON-frame
   centroid and a globally decoded marker ID from that same code cycle.
3. Call `recognize_admitted_boards()`. It validates common-trigger frames,
   the common pose phase and local-track uniqueness; then groups LEDs by
   their registered board IDs, triangulates positions with calibration
   selected by physical camera ID, and fits each board separately.
4. Each board reports one of `recognized`, `insufficient_markers`,
   `geometry_rejected` or `not_visible`. All unsuccessful outcomes have
   `pose=None`. No last-good pose is silently used for a new frame.

```python
from board_patterns import BoardPatternRegistry
from board_recognizer import recognize_admitted_boards

registry = BoardPatternRegistry.from_json_file("board-patterns.example.json")

report = recognize_admitted_boards(
    admissions,           # admitted markers from local temporal traces
    frames,               # validated frame stamps, one per physical camera
    expected_camera_ids,  # e.g. [2, 5, 9]
    camera_poses_by_id,
    intrinsics_by_id,
    registry,
    trigger_epoch=controller_epoch,
    max_exposure_skew_ns=measured_skew_limit,
)
for board_id, outcome in report.boards.items():
    if outcome.pose is not None:
        current_board_position = outcome.pose.translation_world_from_board
        current_body_rotation = outcome.pose.rotation_world_from_body
```

When called **only** on the common pose phase, `allow_trigger_gaps=True` must
be explicitly selected for successive pose-phase frames; all intervening
hardware triggers must still be verified by an external capture monitor.
It is not a fallback to inferred synchronization.

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
