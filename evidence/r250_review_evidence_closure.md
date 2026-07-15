# R250 Review Evidence Closure

Status: `LOCAL_EVIDENCE_COMPLETE_TV_RUNNER_READY_AWAITING_HASHED_BUNDLE`

## C2 Eligibility

- C2 all-source support covers 66,139/157,192 rows (42.1%).
- Mean source slots: 14.72 eligible vs. 18.69 ineligible.
- Mean test rows per user at the row level: 75.25 eligible vs. 101.61 ineligible.
- C0 all-source dependence is positive in both groups for every seed, but is larger on the C2-eligible population; the C0-C2 reversal itself remains identified only on C2 support.

## Latest-Slot Validity

- Every row has at least 1 non-padding source slot.
- C2 latest-slot coverage is 55.4%; every eligible row changes exactly one slot, with zero bad changed-slot rows.
- The eligibility and changed-slot structure is identical across all three checkpoints.

## TV Baseline

- Observed exact-ID occurrence TV is 0.111 under the definition recorded above.
- A random-derangement reference is not recoverable from summary statistics alone. It requires the raw source-ID occurrence vector from the hash-bound processed bundle; the hash-checking random-reference runner is ready, and no paper claim is authorized from an invented baseline.

## C0 All-Source Margins by C2 Support

| Seed | Eligible | Ineligible | Difference |
|---:|---:|---:|---:|
| 20230601 | 0.0054 [0.0042, 0.0067] | 0.0022 [0.0015, 0.0030] | 0.0032 [0.0017, 0.0047] |
| 20230602 | 0.0020 [0.0008, 0.0032] | 0.0018 [0.0012, 0.0025] | 0.0002 [-0.0012, 0.0015] |
| 20230603 | 0.0032 [0.0020, 0.0044] | 0.0015 [0.0008, 0.0022] | 0.0017 [0.0004, 0.0031] |
