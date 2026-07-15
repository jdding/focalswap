# FocalSwap Reviewer Artifact

This is the anonymous code and evidence artifact for the paper **Do
Search-Augmented Recommenders Use the Focal Search? A Transfer-Validity
Audit**. FocalSwap holds the target, candidate set, history, and scorer fixed,
then changes only the focal search input. The audit separates ranking gain,
search-input dependence, matched focal specificity, and recovery instead of
treating them as one claim.

The repository is designed for two reviewer tasks:

1. trace paper-facing numbers and claim boundaries to compact source tables;
2. inspect and smoke-test the audit/readout implementation without private data.

It is not a redistributed copy of third-party datasets or model repositories,
and it is not a one-command replacement for obtaining their released code.

## 60-Second Validation

**Requirement:** Python 3.10 or newer. From the repository root, run:

```bash
bash scripts/run_smoke.sh
```

The command uses only the Python standard library and normally finishes in
under five seconds. It checks the paper source-data manifest, evidence
invariants, corrected intervention scope, positive controls, Python syntax, and
local import closure.

Expected output:

```text
FocalSwap reviewer artifact validation passed
```

For the exact checks performed, see
[`ARTIFACT_QUICKSTART.md`](ARTIFACT_QUICKSTART.md).

## Recommended Review Path

1. Read
   [`paper/figures/source_data/r242_control_contract.json`](paper/figures/source_data/r242_control_contract.json)
   for the canonical latest-slot/all-source and C0--C3 terminology.
2. Open
   [`paper/figures/table_figure_manifest.json`](paper/figures/table_figure_manifest.json)
   to map each paper figure/table to its source data and claim.
3. Inspect the cross-regime sources
   [`table2_cross_regime_gate_readout.csv`](paper/figures/source_data/table2_cross_regime_gate_readout.csv)
   and
   [`table3_matched_swap_readout.csv`](paper/figures/source_data/table3_matched_swap_readout.csv).
4. Inspect
   [`table4_checkpoint_anchor_readouts.csv`](paper/figures/source_data/table4_checkpoint_anchor_readouts.csv)
   together with the control-sensitivity and latest-slot evidence described
   below.
5. Run `bash scripts/run_smoke.sh` to replay the package-level consistency
   checks.

## Evidence-to-Code Map

| Paper-facing question | Primary evidence | Main implementation |
| --- | --- | --- |
| How are fixed rows and controlled views constructed? | `paper/figures/source_data/kuaisar_g1_contract.json`, `paper/figures/source_data/r242_control_contract.json` | `tools/controlled_view_ranking.py`, `tools/focal_source_views.py` |
| What do the behavior and lexical surfaces show? | `paper/figures/source_data/table2_cross_regime_gate_readout.csv`, `paper/figures/source_data/table3_matched_swap_readout.csv`, `evidence/kuaisar_*`, `evidence/qilin_*` | `tools/calibration_audit.py`, `tools/swap_ladder_category_broken.py`, `tools/row_level_power_bootstrap.py` |
| What do released recommendation checkpoints show under a latest-slot intervention? | `paper/figures/source_data/table4_checkpoint_anchor_readouts.csv`, `evidence/sesrec_corrected_bootstrap.json`, `evidence/unisar_corrected_multiseed_summary.csv` | `tools/sesrec_five_view_readout.py`, `tools/unisar_five_view_readout.py`, `tools/unisar_source_scope_localization.py` |
| Why does dependence change with the control? | `paper/figures/source_data/fig4_control_sensitivity.csv`, `evidence/unisar_r242_control_ladder_summary.csv`, `evidence/unisar_r242_control_reversal_contrasts.csv`, `evidence/r250_*` | `tools/control_ladder_analysis.py`, `tools/control_reversal_analysis.py`, `tools/source_scope_analysis.py` |
| Can the protocol return a positive focal readout? | `evidence/r243_full_chain_planted_control.json`, `paper/figures/source_data/r243_full_chain_planted_control.csv` | `tools/full_chain_planted_control.py`, `tools/summarize_planted_control.py` |
| Is there released-code positive calibration outside the recommendation-side claim? | `evidence/gensar_persrc_calibration.json` | `tools/gensar_persrc_readout_analysis.py` |
| What happens on a common SESRec/UniSAR surface? | `evidence/common_surface_summary.csv`, `evidence/common_candidate_contract.json`, `evidence/common_surface_integrity.md` | `tools/common_surface_preflight.py`, `tools/unisar_common_surface_readout.py`, `tools/common_surface_summary.py` |
| Which methods do not expose a valid same-scorer audit interface? | `evidence/unifiedssr_interface_boundary.json`, `evidence/runtime_boundary_dependence_bootstrap.csv` | `tools/official_adapter.py` |

The planted control is a protocol calibration, not a real-system result.
Likewise, the GenSAR record is a positive current-query calibration on its
native source-item prediction surface; it is not recommendation-side
`SeqRecWithSrcHis` focal evidence. The common-surface experiment is a
controlled comparison, not an official leaderboard reproduction.

## Reproduction Levels

### Level 1: Data-Free Package Validation

```bash
bash scripts/run_smoke.sh
```

No network access, private data, checkpoint, GPU, or third-party Python package
is required.

### Level 2: Paper Figure Regeneration

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
contract, and checkpoint. These assets are not redistributed here. After the
optional dependencies are installed, each command-line entry point documents
its inputs with `--help`.

## Repository Layout

- `evidence/`: sanitized aggregate evidence, bootstrap summaries, control
  diagnostics, positive calibrations, and interface-boundary records.
- `paper/figures/source_data/`: authoritative CSV/JSON sources for paper
  figures, tables, and numeric claims.
- `paper/sections/`: reviewer-visible LaTeX table sources; the manuscript and
  submission PDF are intentionally absent.
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
These are immutable experiment-provenance identifiers used to bind derived
tables to audited runs; they are not extra methods, metrics, or competing claim
vocabularies. Reader-facing terminology is fixed by
`r242_control_contract.json`.

## Artifact Integrity

`PACKAGE_MANIFEST.json` is the terminal file ledger. The smoke test replays its
hashes and validates the claim-sensitive invariants. `ANONYMIZATION_REPORT.md`
records that the reviewer artifact passed the strict identity/secret scan, and
`REPRO_CHECK.json` records the release-time smoke result.
