# Evidence Tables

The authoritative vocabulary is
`paper/figures/source_data/r242_control_contract.json`: latest-slot versus
all-source intervention scope; C0 neutral-mask, C1 profile-prioritized donor,
C2 category/profile-matched donor, and C3 same-user temporal donor. Historical
raw evidence retains its original field keys for hash and replay compatibility;
those keys are provenance identifiers, not alternative reader-facing terms.

`unisar_corrected_multiseed_summary.csv` reports the corrected latest-slot
dependence and swap diagnostics for three retained checkpoints on one
official-processed surface. `unisar_source_scope_summary.csv` and
`unisar_mechanism_contrasts.csv` retain the R233 C0-versus-C1
control calibration. `unisar_r242_control_ladder_summary.csv` and
`unisar_r242_control_reversal_contrasts.csv` report the same-row C0--C2
control ladder. `unisar_dual_estimand_summary.csv` and its JSON counterpart
retain the earlier neutral-mask result as numerical provenance.
`sesrec_corrected_bootstrap.json` reports the
finite-neutral SESRec latest-slot dependence and matched-swap intervals. The two
`common_surface_*` files report six SESRec/UniSAR rows under identical rows,
candidates, donors, and metrics.

On the registered checkpoint surfaces, no latest-slot intervention establishes
positive dependence. For UniSAR, the all-source C0 neutral-mask control produces a positive
RR reduction on 66,139 C2-eligible rows, but the C2 category/profile-matched
observed-support donor does not. All three paired C0--C2
97.5% intervals are positive. The active claim is therefore control
sensitivity, not accumulated-source use. Matched-swap rows remain diagnostics
because latest-slot dependence does not clear its gate.
The package validator enforces these distinctions.

The common-surface experiment is a controlled surface check. It is not an
official leaderboard reproduction, native-user personalization result, or
architecture-only comparison. Its value is to show that the focal-source
readout depends jointly on model and evaluation surface.

`kuaisar_swap_ladder_metrics.csv` reports the KuaiSAR win/tie/loss and
half-tie components used to interpret its swap-invariant rows. The Qilin
counterpart is retained in `qilin_ten_bucket_multiplicity.json`.
`kuaisar_gate_learning_curve.csv` retains the 500--10,000-row G4 learning
curve, and `positive_control_dose_curve.csv` retains every injected dose and
its recovered headroom. `qilin_g2_null_robustness.json` retains the pooled and
per-seed G2 null summaries rather than only the paper-facing thresholds.

`gensar_persrc_calibration.json` is a released-code positive calibration on
GenSAR's native source-item prediction task. It shows that the audit detects
explicit current-query identity and separates it from retained source-history
perturbations. The record also fixes the pre-readout checkpoint selection and
the official validation-sample filtering contract. It is not recommendation-side
`SeqRecWithSrcHis` focal evidence. `gensar_seqrec_boundary.json` records that
the available recommendation-side checkpoint failed its registered native
quality gate and is therefore excluded from claim-bearing intervention
evidence.

`r243_full_chain_planted_control.json` reports a three-dose by three-seed
calibration on fixed Qilin rows. Dose one clears calibrated opportunity,
dependence, and matched specificity for every seed; the separate downstream
recovery probe is not established. It is a protocol calibration, not a
real-system result.

`r250_c2_eligibility_balance.csv`, `r250_c0_margin_by_eligibility.csv`, and
`r250_latest_slot_validity.json` test whether the control-reversal support is a
trivial population artifact and whether the latest-slot intervention actually
changes one non-padding slot. C0 response remains positive in both eligible and
non-eligible populations; every latest-slot-eligible row changes exactly one
slot. Exact C0--C2 reversal remains scoped to common support.

`r259_c2_slot_stratified_control.csv` and its JSON counterpart localize the
same-row all-source C0--C2 contrast by source-slot count. The contrast is
positive across all 12 seed-by-stratum estimates for 4--30 slots, with 11
97.5% intervals excluding zero; the 1--3-slot strata are unresolved. These
stratum-level intervals are descriptive and are not multiplicity-adjusted. This
supports source-history-length heterogeneity, not a slot-count-invariant
mechanism or focal-dependence claim.

`r261_rank_change_by_slot_stratum.csv` and its JSON counterpart verify that
the C0/C2 mean contrasts aggregate discrete half-tie target-rank changes rather
than floating-point jitter. C0 changes are slightly more likely to worsen the
target rank and have a median absolute shift of two positions; C2 changes are
approximately direction-balanced with a median absolute shift of one. They
also record the C2 donor-map safeguards: a
one-to-one observed-support derangement, no fixed points, and no donor reuse.
This does not upgrade C2 from a behavioral-category sensitivity control to a
lexical-semantic match.

`unifiedssr_interface_boundary.json` records why the released recommendation
and query-conditioned search heads do not provide one native scorer for all
five audit views. It is interface provenance, not a focal-transfer result.
