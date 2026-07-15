# FocalSwap Reviewer Artifact

This anonymous artifact supports the paper's fixed-row audit of focal-search
transfer in search-augmented recommendation. It contains paper-facing source
data, deterministic consistency checks, corrected checkpoint-dependence
evidence, and the analysis/readout scripts used by the registered audit
surfaces.

## Quick Check

Requirements: Python 3.10 or newer. The smoke check uses only the standard
library and normally finishes in under five seconds. It validates evidence,
compiles every packaged Python tool, and checks that every local import resolves
inside the cloned repository or extracted artifact.

```bash
bash scripts/run_smoke.sh
```

Expected output:

```text
FocalSwap reviewer artifact validation passed
```

## Contents

- `evidence/`: corrected latest-slot checks, UniSAR dependence-control
  calibration, a full-chain planted control, GenSAR current-query calibration,
  eligibility/slot-validity closure, historical neutral-mask provenance, and
  common-surface tables.
- `paper/figures/source_data/`: source CSVs for the paper figures and tables.
  `r242_control_contract.json` is the canonical terminology contract for
  latest-slot/all-source scope and C0--C3 controls.
- `paper/sections/`: paper table sources.
- `paper/figures/`: deterministic figure-generation scripts.
- `tools/`: audit and checkpoint-readout implementations.
- `scripts/`: package validation.

## Optional Figure Regeneration

```bash
python3 -m pip install -r requirements.txt
python3 scripts/check_tool_imports.py
python3 paper/figures/make_fig2_main_evidence.py
python3 paper/figures/make_fig4_control_sensitivity.py
python3 paper/figures/make_fig4_swap_specificity.py
```

## Evidence Boundary

The artifact does not redistribute third-party datasets, released-model
repositories, checkpoints, or row-level user logs. Full checkpoint reruns
therefore require obtaining the datasets and released code from their original
providers. The included derived tables are sufficient to verify the numbers,
ordered gate logic, seed variation, intervention-control sensitivity,
current-query calibration, reverse-source diagnostics, and common-surface
claim boundary reported by the paper. Tool imports and syntax are checked by the data-free smoke; full tool
execution requires the optional Python dependencies and, where applicable, the
omitted third-party data and checkpoints.
