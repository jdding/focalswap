# Reproduction guide

After installing the framework with `python3 -m pip install -e .`, run these
commands from the repository root with Python 3.10 or newer.

## Verify the reported results

```sh
bash scripts/run_smoke.sh
mkdir -p outputs
python3 tools/donor_mapping_summary.py \
  --input evidence/donor_mapping_readouts.csv \
  --output-tex outputs/table3_sensitivity.tex \
  --output-summary outputs/table3_sensitivity.csv
```

The smoke check validates packaged file hashes, numeric invariants in the
tables, and imports in the audit tools. The second command reconstructs
Table 3(c) from the 30 aggregate scope-by-assignment readouts.

## Regenerate Figures 2–4

```sh
python3 -m pip install matplotlib numpy
python3 paper/figures/make_fig2_main_evidence.py
python3 paper/figures/make_fig4_swap_specificity.py
python3 paper/figures/make_fig4_control_sensitivity.py
```

Figures are written next to their scripts. The filenames reflect the order
in which the figures were developed. See the
[figure and table index](paper/figures/table_figure_manifest.json) for the
published numbering.

## Run the model readouts

For checkpoint experiments, obtain the original model implementations and
datasets and prepare compatible checkpoints. The audit holds each model's
test candidates and target fixed. SESRec and UniSAR use their own native
candidate pools, so their Full-input RR values describe different setups.

Install the optional analysis and model dependencies with
`python3 -m pip install -r requirements.txt`. Each entry point lists its
arguments with `--help`.

| Analysis | Entry point | Inputs |
|---|---|---|
| SESRec rank readout | `tools/sesrec_five_view_readout.py` | SESRec checkout, processed data, checkpoint |
| UniSAR rank readout | `tools/unisar_five_view_readout.py` | UniSAR checkout, processed data, checkpoint |
| UniSAR mask and replacement comparison | `tools/unisar_semantic_control_ladder.py` | UniSAR runtime, checkpoint and fixed rows with donor mappings |
| Bootstrap the donor assignments | `tools/donor_mapping_analysis.py` | Matching and reference manifests, rank exports and their integrity records |

For a fixed-row walkthrough without a checkpoint, run
`python3 examples/audit_rankings.py` with the core framework installed.

### Supply native inputs

For the UniSAR source-control ladder, point the adapter to the original model
checkout, its processed inputs and a trained checkpoint. Set the environment
variables below to your local inputs before running the command.

```sh
python3 tools/unisar_semantic_control_ladder.py \
  --official-root "$UNISAR_REPO" \
  --processed-data-root "$UNISAR_DATA" \
  --checkpoint "$UNISAR_CHECKPOINT" \
  --work-dir outputs/unisar-work --reports-dir outputs/unisar \
  --split test --row-limit 0 --max-eval-rows 0 \
  --expected-split-rows 157192 --seed 20230601 --device cpu
```

The paper's native UniSAR runs use upstream commit
`1a14a246066cd1aa6ee789c57ff2c7499f3d7379` and seeds 20230601–20230603.
[evidence/checkpoint_input_reference.json](evidence/checkpoint_input_reference.json)
records checkpoint and export hashes. The source-control adapter checks this
fixed experiment contract. To audit a different model, dataset, or candidate
construction, use the generic API described in [the integration guide](docs/integration.md).

The SESRec adapter accepts an episode JSONL and item metadata alongside its
model checkout and checkpoint.

```sh
python3 tools/sesrec_five_view_readout.py \
  --official-root "$SESREC_REPO" \
  --examples-path "$SESREC_EPISODES" \
  --item-metadata "$ITEM_METADATA" \
  --checkpoint "$SESREC_CHECKPOINT" \
  --reports-dir outputs/sesrec --split test \
  --row-limit 0 --max-eval-rows 0 --eval-candidates-only \
  --finite-neutral-controls --device cpu
```

Inspect the adapter's `read_jsonl` and `build_rows` functions before preparing
episode inputs. Original metrics and denominators are recorded in
`evidence/sesrec_corrected_bootstrap.json`. Third-party code, raw interaction
data, checkpoints and user-level exports must be obtained or recreated
separately. The aggregate tables support the checks and plotting commands above.
