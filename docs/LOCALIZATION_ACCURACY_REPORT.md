# Localization accuracy and latency report

```sh
cd computer_code/api
python localization_accuracy_report_cli.py localization-acceptance.json > localization-acceptance-report.json
```

The CLI replays a versioned IMU/attitude/position event stream, matches the resulting estimator timeline to a separately supplied position-reference dataset, and summarizes processing latency measured with a single monotonic host clock. It is a software report tool, not a hardware certification tool.

## Input schema v1

The root object must contain exactly:

- `schema_version`: integer `1`
- `config`: a serialized `LocalizationReplayConfig` object (including seed, IMU calibration, gravity model and configured position-source contracts)
- `events`: ordered `LocalizationReplayEvent` objects, each with contiguous `event_index` starting at zero; supported kinds are `imu` and `position`
- `references`: strictly increasing position references
- `max_match_delta_ns`: nonnegative timestamp matching tolerance
- optional `latency_samples`: monotonic-clock ingress/egress samples

A reference record has the exact typed fields:

```json
{
  "timestamp_ns": 1000000000,
  "clock_domain": "imu-clock",
  "position_world_from_body_m": [1.0, 2.0, 3.0],
  "dataset_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
}
```

All references must use the estimator's clock domain and the same dataset digest. The digest identifies the declared dataset; it does not establish who collected it or whether it is independent.

A latency record has these fields:

```json
{
  "event_id": "frame-0001",
  "clock_domain": "host-monotonic",
  "ingress_monotonic_ns": 1000000,
  "egress_monotonic_ns": 1042000
}
```

Latency samples must use one monotonic clock domain, have unique event IDs, and satisfy egress >= ingress. Do not subtract host-monotonic timestamps from sensor timestamps to estimate latency.

The complete `config` and `events` objects are the serialized typed records used by `localization_replay_cli.py`; unknown fields, duplicate JSON keys, non-finite values, invalid derived revision digests and inconsistent event ordering are rejected. Input and output are bounded (64 MB defaults); event/sample capacities are bounded by the underlying replay and metrics APIs.

## Output metrics

The report includes replay config/input/output digests and:

- position error RMSE, median, P95 and maximum in metres;
- error for states whose source is `IMU_PROPAGATED`, useful for evaluating dead-reckoning drift;
- error at a state that follows a `LOST` state, useful for checking reacquisition quality;
- sample/reference match counts, unmatched counts, and TRACKING/DEGRADED/LOST counts;
- processing latency P50/P95/P99/maximum in nanoseconds;
- per-reference position errors, with `null` for unmatched references;
- deterministic input/report digests.

P95/P99 use nearest-rank empirical percentiles. Position matching is one-to-one, chronological, and constrained by `max_match_delta_ns`. The CLI collapses repeated post-update outputs at the same estimator timestamp, retaining the final state at that time, before computing reference metrics. This avoids counting multiple delayed corrections at one current timestamp as independent trajectory samples.

## Qualification boundary

Every report sets these flags to false:

- `reference_independence_verified`
- `hardware_ground_truth_verified`
- `hardware_provenance_verified`

A SHA-256 digest protects the identity of supplied bytes/semantic inputs, not their truth. Synthetic references can validate the report implementation, but they cannot establish physical accuracy. Issue [#33](https://github.com/qigao/Low-Cost-Mocap/issues/33) A9 and hardware evidence gates [#2–#7](https://github.com/qigao/Low-Cost-Mocap/issues/2) remain open until independently surveyed reference data, raw sensor recordings, declared acceptance limits and reproducible reports are available.
