# Hardware validation backlog after PR #1

Merged software baseline: [PR #1](https://github.com/qigao/Low-Cost-Mocap/pull/1), squash commit `71567d64118f05210c667e0e13cedd3318376527`.

**Tracking status (2026-10-22):** GitHub Issues is enabled. Six independent hardware-qualification Issues (#2–#7) were created from the corresponding committed templates. Each remains open until real captured evidence meets the defined acceptance criteria. These are not reopened software CI failures.

| Gate | Open GitHub Issue | Reusable issue template |
| --- | --- | --- |
| HW-01 — Exposure and trigger synchronization | [#2](https://github.com/qigao/Low-Cost-Mocap/issues/2) | [Template](../.github/ISSUE_TEMPLATE/HW-01-exposure-trigger-sync.md) |
| HW-02 — Camera calibration and metric scale | [#3](https://github.com/qigao/Low-Cost-Mocap/issues/3) | [Template](../.github/ISSUE_TEMPLATE/HW-02-calibration-metric-scale.md) |
| HW-03 — LED codes, epoch and optical aliasing | [#4](https://github.com/qigao/Low-Cost-Mocap/issues/4) | [Template](../.github/ISSUE_TEMPLATE/HW-03-optical-timecode-led.md) |
| HW-04 — Moving LED tracking and occlusion safety | [#5](https://github.com/qigao/Low-Cost-Mocap/issues/5) | [Template](../.github/ISSUE_TEMPLATE/HW-04-moving-led-tracks.md) |
| HW-05 — Independent labeled hardware recordings | [#6](https://github.com/qigao/Low-Cost-Mocap/issues/6) | [Template](../.github/ISSUE_TEMPLATE/HW-05-independent-recordings.md) |
| HW-06 — Latency and fail-closed control boundary | [#7](https://github.com/qigao/Low-Cost-Mocap/issues/7) | [Template](../.github/ISSUE_TEMPLATE/HW-06-e2e-latency-safety.md) |

**Dependency:** HW-06 requires independent acceptance of HW-01–HW-05 before any live-flight authorization. Hardware-only work should attach raw data provenance, independently measured references, reproducible reports, and predeclared limits directly to the relevant issue.

## Merge and acceptance boundary

- PR #1 software functionality is merged to `main`. This does **not** certify real PS3Eye sensor-exposure synchronization, physical metric scale, optical timecode reliability, live moving-light tracking, hardware-labeled datasets, or the flight-control deadline.
- Current pseyepy frames have a host dequeue timestamp, not a proven sensor exposure trigger. Do not infer a common clock or trigger ID from Python polling or receive times.
- Each gate requires independent captured evidence, a specific reproducible setup, raw artifact checksums, predefined measurable acceptance criteria, and explicit fail-closed results. No production or live-flight switch is authorized by this backlog.
- For reproducible offline summaries, run `cd computer_code/api && python offline_acceptance_cli.py recordings.json > offline-report.json` on externally verified recordings. The CLI intentionally sets `hardware_provenance_verified=false`; a passing software replay never flips this status by itself.

These are **hardware-only follow-ups**, not reopened software CI failures.
