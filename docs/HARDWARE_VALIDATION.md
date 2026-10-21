# Hardware validation backlog after PR #1

Merged software baseline: [PR #1](https://github.com/qigao/Low-Cost-Mocap/pull/1), squash commit `71567d64118f05210c667e0e13cedd3318376527`.

**Tracking status (2026-10-21):** GitHub Issues is disabled for this repository (`has_issues=false` / HTTP 410). The six entries below are **issue-ready GitHub templates**, not existing numbered Issues. They are retained in the repository so hardware work is not lost or confused with the merged software code. A repository administrator must enable **Settings → General → Features → Issues**; then create one GitHub Issue per template from **Issues → New issue**.

| Gate | Issue-ready template | Status |
| --- | --- | --- |
| HW-01 | [[HW-01] Verify real sensor-exposure synchronization and trigger provenance](../.github/ISSUE_TEMPLATE/HW-01-exposure-trigger-sync.md) | Open hardware gate; Issue not yet created |
| HW-02 | [[HW-02] Qualify physical multi-camera calibration and metric scale](../.github/ISSUE_TEMPLATE/HW-02-calibration-metric-scale.md) | Open hardware gate; Issue not yet created |
| HW-03 | [[HW-03] Validate physical LED temporal codes, exposure aliasing and code epoch](../.github/ISSUE_TEMPLATE/HW-03-optical-timecode-led.md) | Open hardware gate; Issue not yet created |
| HW-04 | [[HW-04] Measure local LED tracking and motion/occlusion ID-switch safety](../.github/ISSUE_TEMPLATE/HW-04-moving-led-tracks.md) | Open hardware gate; Issue not yet created |
| HW-05 | [[HW-05] Build independent labeled hardware replay dataset and acceptance report](../.github/ISSUE_TEMPLATE/HW-05-independent-recordings.md) | Open hardware gate; Issue not yet created |
| HW-06 | [[HW-06] Qualify end-to-end pose latency and fail-closed flight-control interface](../.github/ISSUE_TEMPLATE/HW-06-e2e-latency-safety.md) | Open hardware gate; Issue not yet created |

## Merge and acceptance boundary

- PR #1 software functionality is merged to `main`. This does **not** certify real PS3Eye sensor-exposure synchronization, physical metric scale, optical timecode reliability, live moving-light tracking, hardware-labeled datasets, or the flight-control deadline.
- Current pseyepy frames have a host dequeue timestamp, not a proven sensor exposure trigger. Do not infer a common clock or trigger ID from Python polling or receive times.
- Each gate requires independent captured evidence, a specific reproducible setup, raw artifact checksums, predefined measurable acceptance criteria, and explicit fail-closed results. No production or live-flight switch is authorized by this backlog.
- For reproducible offline summaries, run `cd computer_code/api && python offline_acceptance_cli.py recordings.json > offline-report.json` on externally verified recordings. The CLI intentionally sets `hardware_provenance_verified=false`; a passing software replay never flips this status by itself.

These are **hardware-only follow-ups**, not reopened software CI failures.
