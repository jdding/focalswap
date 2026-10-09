# FocalSwap Reviewer Artifact

Reviewer access: https://anonymous.4open.science/r/focalswap-5D6D/

This repository accompanies the anonymous WSDM submission **Do
Search-Augmented Recommenders Use the Latest Search Episode?** The paper asks a
narrower question than ordinary
ranking evaluation: when search signals help a recommender, can the gain be
attributed to the immediately preceding search episode?

FocalSwap audits that question by keeping the target, candidate set, history,
and scorer fixed while intervening on the focal search input. It reports
ranking utility, target-rank response, signed contribution, matched focal
specificity, and downstream routing as separate claims.

## What Reviewers Can Verify

The package supports three paper-level conclusions:

1. **The protocol is capable of a positive focal readout.** A planted control
   clears the attribution gates at its highest dose. A released GenSAR
   source-item task also recovers current-query identity. These are protocol
   calibrations, not recommendation-side positive evidence.
2. **Behavioral, lexical, and checkpoint evidence do not collapse into one
   verdict.** KuaiSAR and Qilin provide bounded surface-level results, while the
   checkpoints trained using released SESRec and UniSAR implementations test
   the latest-slot intervention directly.
3. **Signed search contribution is control-sensitive.** On the same 66,139 UniSAR rows
   and three retained checkpoints, an all-source neutral mask yields positive
   RR contributions, whereas an observed-support donor matched by category and
   source profile does not. Neutral masks change 59.7--61.7% of target ranks
   and matched donors change 44.1--44.8%, with different effect shape and
   directional balance. Positive latest-slot contribution is not established.

The repository lets reviewers trace these statements to compact source tables,
inspect the corresponding implementations, and run a data-free consistency
check. It intentionally does not redistribute third-party datasets,
checkpoints, model repositories, raw user logs, or the submission PDF.

## Quick Validation

**Requirement:** Python 3.10 or newer. From the repository root, run:

```bash
bash scripts/run_smoke.sh
```

The command uses only the Python standard library and normally finishes in
under five seconds. It verifies the package manifest, paper-facing source-data
contracts, claim-sensitive evidence invariants, Python syntax, and local import
closure.

Expected output:

```text
FocalSwap reviewer artifact validation passed
```

No installation, network access, private data, checkpoint, or GPU is required.
The exact checks are listed in
[`ARTIFACT_QUICKSTART.md`](ARTIFACT_QUICKSTART.md).

## Recommended Five-Minute Review Path

1. Open
   [`paper/figures/table_figure_manifest.json`](paper/figures/table_figure_manifest.json)
   to map every paper figure and table to its packaged source data and claim.
2. Read
   [`r242_control_contract.json`](paper/figures/source_data/r242_control_contract.json)
   for the canonical latest-slot versus all-source intervention scopes and the
   C0--C3 controls.
3. Inspect the audit-surface scale table
   [`table2_audit_surface_scale.csv`](paper/figures/source_data/table2_audit_surface_scale.csv)
   and the checkpoint summary
   [`table3_checkpoint_summary.csv`](paper/figures/source_data/table3_checkpoint_summary.csv).
4. Inspect the Figure 4 source
   [`fig4_control_sensitivity.csv`](paper/figures/source_data/fig4_control_sensitivity.csv).
5. Run `bash scripts/run_smoke.sh` to verify that the packaged evidence still
   satisfies the paper's registered contracts.

## Evidence-to-Code Map

| Paper-facing question | Primary evidence | Main implementation |
| --- | --- | --- |
| How are fixed rows and controlled views constructed? | `paper/figures/source_data/kuaisar_g1_contract.json`, `paper/figures/source_data/r242_control_contract.json` | `tools/controlled_view_ranking.py`, `tools/focal_source_views.py` |
| What do the behavior and lexical surfaces show? | `paper/figures/source_data/table2_audit_surface_scale.csv`, `paper/figures/source_data/table2_cross_regime_gate_readout.csv`, `paper/figures/source_data/fig4_swap_specificity.csv`, `evidence/kuaisar_*`, `evidence/qilin_*` | `tools/calibration_audit.py`, `tools/swap_ladder_category_broken.py`, `tools/row_level_power_bootstrap.py` |
| What do SESRec and UniSAR checkpoints show under a latest-slot intervention? | `paper/figures/source_data/table4_checkpoint_anchor_readouts.csv`, `evidence/sesrec_corrected_bootstrap.json`, `evidence/unisar_corrected_multiseed_summary.csv` | `tools/sesrec_five_view_readout.py`, `tools/unisar_five_view_readout.py`, `tools/unisar_source_scope_localization.py` |
| How do response frequency, effect shape, and signed contribution differ by control? | `paper/figures/source_data/fig4_control_sensitivity.csv`, `paper/figures/source_data/table3_rank_movement.csv`, `evidence/unisar_r242_control_ladder_summary.csv`, `evidence/unisar_r242_control_reversal_contrasts.csv`, `evidence/r261_rank_change_*`, `evidence/r264_response_effect_shape.*` | `tools/control_ladder_analysis.py`, `tools/control_reversal_analysis.py`, `tools/rank_change_diagnostics.py`, `tools/response_effect_shape.py` |
| Can the protocol return a positive focal readout? | `evidence/r243_full_chain_planted_control.json`, `paper/figures/source_data/r243_full_chain_planted_control.csv` | `tools/full_chain_planted_control.py`, `tools/summarize_planted_control.py` |
| Is there released-code positive calibration outside the recommendation-side claim? | `evidence/gensar_persrc_calibration.json`, `evidence/gensar_seqrec_boundary.json` | `tools/gensar_persrc_readout_analysis.py` |
| What happens on a common SESRec/UniSAR surface? | `evidence/common_surface_summary.csv`, `evidence/common_candidate_contract.json`, `evidence/common_surface_integrity.md` | `tools/common_surface_preflight.py`, `tools/unisar_common_surface_readout.py`, `tools/common_surface_summary.py` |
| Which methods do not expose a valid same-scorer audit interface? | `evidence/unifiedssr_interface_boundary.json`, `evidence/runtime_boundary_dependence_bootstrap.csv` | `tools/official_adapter.py` |

The planted control is a protocol calibration, not a real-system result. The
GenSAR record is a current-query calibration on its native source-item
prediction task, not recommendation-side `SeqRecWithSrcHis` focal evidence.
The common-surface experiment is a controlled comparison, not an official
leaderboard reproduction.

## Reproduction Levels

### Level 1: Data-Free Validation

```bash
bash scripts/run_smoke.sh
```

No network access, private data, checkpoint, GPU, or third-party Python package
is required.

### Level 2: Figure Regeneration

Figures 2 and 4 can be regenerated directly from the packaged source CSVs:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install "matplotlib>=3.7" "numpy>=1.24"
python3 paper/figures/make_fig2_main_evidence.py
python3 paper/figures/make_fig4_control_sensitivity.py
python3 paper/figures/make_fig4_swap_specificity.py
```

The scripts write PDF and PNG outputs under `paper/figures/`. To import every
packaged analysis module, install `requirements.txt` and run:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/check_tool_imports.py
```

### Level 3: Full Checkpoint Readouts

The `tools/` directory contains the audit, adapter, bootstrap, and readout
implementations used for the registered surfaces. Full executions require the
corresponding third-party dataset, released repository, processed-data
contract, and checkpoint. These assets are not redistributed here. After
installing the optional dependencies, each command-line entry point documents
its inputs with `--help`.

## Repository Layout

- `evidence/`: sanitized aggregate evidence, bootstrap summaries, control
  diagnostics, positive calibrations, and interface-boundary records.
- `paper/figures/source_data/`: authoritative CSV/JSON sources for paper
  figures, tables, and numeric claims.
- `paper/sections/`: reviewer-visible LaTeX table and figure-panel sources; the
  manuscript and submission PDF are intentionally absent.
- `paper/figures/`: deterministic plotting scripts and the editable Figure 1
  source.
- `tools/`: audit and released-checkpoint readout implementations.
- `scripts/`: package validation and import checks.
- `PACKAGE_MANIFEST.json`: SHA-256 hash and byte-size ledger for every packaged
  file.
- `ANONYMIZATION_REPORT.md`, `REPRO_CHECK.json`: release-time anonymity and
  smoke-test records.

## Data and Release Boundary

The artifact does not redistribute third-party datasets, released-model
repositories, checkpoints, or row-level user logs. Full checkpoint reruns
therefore require obtaining the datasets and released code from their original
providers. The included derived tables are sufficient to verify the numbers,
ordered gate logic, seed variation, intervention-control sensitivity,
current-query calibration, reverse-source diagnostics, and common-surface
claim boundary reported by the paper.

Some evidence files retain identifiers such as `r242`, `r243`, or `r250`.
These immutable provenance labels bind derived tables to audited runs; they are
not methods, metrics, or alternative claim vocabularies. Reader-facing
terminology is fixed by `r242_control_contract.json`.

## Artifact Integrity

`PACKAGE_MANIFEST.json` is the terminal file ledger. The smoke test replays its
hashes and validates the claim-sensitive invariants. `ANONYMIZATION_REPORT.md`
records that the reviewer artifact passed the strict identity/secret scan, and
`REPRO_CHECK.json` records the release-time smoke result.
