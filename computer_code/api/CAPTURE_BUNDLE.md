# Capture bundle manifest v1 — integrity, not hardware certification

The capture-bundle validator prepares a reviewable evidence package for
HW-05. It is deliberately separate from
`offline_acceptance_cli.py`, which replays trigger and temporal-label records
and computes statistical summaries.

## Layout

A bundle directory contains one fixed name, `capture-bundle.json`, plus the
files it references using canonical relative POSIX paths. Version 1 requires
these artifact roles:

| Role | Intended evidence |
|---|---|
| `raw_camera_frames` | Original frames or a lossless, documented raw-frame container |
| `reference_trigger_log` | Separately recorded trigger/exposure reference |
| `independent_marker_labels` | Independently reviewed physical marker IDs and track labels |
| `device_metadata` | Device identity, model, dimensions and capture metadata |
| `led_controller_log` | Controller outputs, trigger epoch and logged timecode |
| `pattern_config` | Exact pattern-registry/configuration bytes; SHA must match its revision |
| `codebook` | Exact codebook bytes used during capture |
| `controller_config` | Exact controller configuration bytes |
| `calibration_config` | Camera calibration and geometry configuration bytes |
| `supplemental` | Optional bounded supporting files; this role may repeat |

The manifest also records physical camera IDs, a declared clock domain,
timestamp semantics and units, software/device/driver/pseyepy versions,
pattern/codebook/controller/calibration SHA-256 revisions, and explicit
acceptance limits. The timestamp semantics must be one of
`sensor_exposure`, `hardware_trigger`, `driver_dequeue`,
`host_monotonic` or `unknown`. The value is a **declaration**, not proof
that a timestamp actually came from sensor exposure.

The acceptance limit declaration has a UTC timestamp. The validator requires
that this timestamp precede the capture start, but cannot independently prove
when the limits were written. A self-reported declaration is not a signature.

## Run the validator

From the API directory:

```sh
python capture_bundle_cli.py /path/to/capture-bundle > bundle-report.json
echo $?
```

Exit status:

- `0`: schema is valid and every required artifact byte count and SHA-256
  matches the declaration, required roles are present, revisions link to
  the declared codebook/controller/calibration artifacts, the reference is
  declared independent, and the acceptance limits are timestamped before
  capture.
- `1`: the manifest is structurally parseable, but one or more required
  files, hashes, sizes, revision links or evidence declarations failed.
- `2`: manifest parsing/schema, root path or resource-limit validation failed.

The JSON report contains a canonical manifest SHA-256, a raw manifest-file
SHA-256, per-artifact status/actual size/actual SHA-256, failure reasons,
declared source clock/timestamp semantics, revision IDs and predeclared
acceptance limits.

Optional resource limits can be changed explicitly:

```sh
python capture_bundle_cli.py /path/to/capture-bundle \
  --max-artifacts 32 \
  --max-artifact-bytes 8589934592 \
  --max-total-bytes 34359738368
```

Defaults cap the manifest at 1 MB, artifacts at 32 entries, each artifact at
8 GiB, and total declared/hashed artifact bytes at 32 GiB. Hashing is streamed
in 1 MiB chunks; raw video/frame data is never loaded wholesale into memory.

## Fail-closed path and schema rules

- Exact schema version 1 and exact allowed fields at every level; unknown
  fields and duplicate JSON object keys are rejected.
- SHA-256 must be 64 lowercase hexadecimal characters. Declared size must be
  positive and within the configured cap.
- No absolute path, Windows drive path, backslash, parent traversal,
  noncanonical component, duplicate/case-colliding path, or symlink in an
  artifact path. Symlink escape and non-regular files are rejected.
- Missing, tampered, wrong-size, duplicate or unreadable artifacts make
  `bundle_integrity_valid` false. The validator reports these per artifact
  and exits nonzero.
- Pattern, codebook, controller and calibration revision IDs must match
  the hashes of their corresponding declared artifact bytes. A matching hash
  proves only byte equality with the submitted file, not that the file was
  actually used by the physical rig.
- A manifest cannot add a `hardware_verified` field; unknown fields reject.
  Even when every artifact passes, the report always contains
  `hardware_provenance_verified=false`,
  `device_authenticity_verified=false`,
  `independent_reference_verified=false`,
  `timestamp_semantics_verified=false`, and
  `physical_metric_scale_verified=false`.

## Next: offline acceptance

A valid bundle report means **only** that the listed bytes match the manifest.
It does not imply the labels are correct, a trigger logger is independent, a
camera driver reports true sensor-exposure time, or the physical calibration
is metrically correct. Extract/convert the independently reviewed records
into the existing versioned input expected by `offline_acceptance_cli.py`,
then review that software replay report separately. Keep hardware
qualification [HW-05](../../../../issues/6) and the real exposure, optical,
calibration and end-to-end safety gates open until physical measurements
and independent evidence have been reviewed.
