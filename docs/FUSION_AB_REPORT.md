# Optical-only vs fused attitude report CLI

The paired A/B report is a software evaluation tool. It compares the optical-only and fused orientation series against the same reference dataset and timestamp matching policy. It does not authenticate the source hardware or prove that the reference is independent.

## Run

From `computer_code/api`:

```sh
python fusion_ab_report_cli.py capture-ab.json > fusion-ab-report.json
```

Exit code is zero only when the complete input is validated and the complete output report is serialized. Invalid input returns nonzero and writes no partial JSON report to stdout.

Limits:
- Input JSON: 16,000,000 bytes maximum.
- Output JSON: 16,000,000 bytes maximum.
- Schema version must be integer `1`; unknown/missing keys, duplicate JSON keys and nonfinite numbers are rejected.
- Output always includes `hardware_ground_truth_verified: false`.

## Input schema v1

Top-level keys are exact:

- `schema_version`: integer `1`
- `clock_domain`: one common acquisition-time clock domain for all series
- `max_match_delta_ns`: nonnegative integer matching tolerance
- `optical_only`: ordered optical-only observations
- `fused`: ordered fused attitude states
- `references`: one ordered reference dataset with a stable SHA-256 identifier

Optical-only records have exact fields:

```json
{
  "body_id": "board-A",
  "timestamp_ns": 1000000000,
  "clock_domain": "capture-clock",
  "quaternion_world_from_body_xyzw": [0, 0, 0, 1],
  "status": "TRACKING",
  "pattern_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "optical_calibration_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
}
```

For optical-only `INVALID` or `LOST` records, quaternion must be `null`. A `TRACKING` record must have a finite nonzero quaternion; it is normalized during typed parsing.

Fused records must contain the complete `OpticalIMUFusionState` contract, including `body_id`, `imu_id`, timestamp and clock domain, quaternion and variance, status/source, last optical timestamp/frame/age, clock mapping and calibration/pattern digests, and explicit `global_position_m: null`, `global_velocity_m_s: null`, `global_position_observable: false`.

Reference records have exact fields:

```json
{
  "timestamp_ns": 1000000000,
  "clock_domain": "capture-clock",
  "quaternion_world_from_body_xyzw": [0, 0, 0, 1],
  "reference_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
```

All reference records must share one dataset digest and have strictly increasing timestamps. The digest identifies the declared dataset; it is not proof of reference independence.

## Output interpretation

The report contains:
- the raw recording SHA-256 and semantic input/report SHA-256 values;
- separate optical-only and fused match coverage, unmatched counts, and TRACKING/DEGRADED/INVALID/LOST counts;
- RMSE, median, P95 and maximum quaternion geodesic error in radians and degrees;
- per-reference errors with `null` for unmatched observations;
- paired RMSE and paired improvement, computed only where both series match the same reference timestamp;
- `hardware_ground_truth_verified: false`.

Do not compare only RMSE. Review match coverage and invalid/lost counts alongside error statistics. A series with missing or invalid optical observations must not be made to look better by silently dropping those outcomes. Synthetic references qualify software behavior only; attach raw, independently sourced hardware recordings and the declared acceptance criteria to HW-05 before making a hardware accuracy claim.


## Bundle-bound report for evidence review

For an input stored as the `fusion_ab_input` artifact inside a capture bundle, use:

```sh
python fusion_ab_bundle_report_cli.py /path/to/capture-bundle > bundle-bound-report.json
```

The bundle must contain all required capture-bundle roles and exactly one `fusion_ab_input` artifact. The command verifies every declared artifact, securely re-reads the selected role relative to an anchored bundle-root descriptor, and checks its exact size and SHA-256 again before parsing. The output binds the result to the bundle ID, canonical manifest digest, raw manifest-file digest and A/B artifact path/hash.

This verifies integrity relative to the supplied manifest. It does not prove that the manifest is truthful, devices are authentic, timestamps are physical exposure times, or reference labels are independent. All hardware-provenance and hardware-ground-truth flags remain false.
