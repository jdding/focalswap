# FocalSwap

**A counterfactual audit framework for search-augmented recommenders.**

FocalSwap asks how the latest search episode and accumulated search state
contribute to a recommendation. It freezes the target, candidates, pre-search
history, and scorer, then compares the original search input with neutral masks
and matched observed searches.

The framework accompanies **Do Search-Augmented Recommenders Use the Latest
Search Episode?**, accepted at **WSDM 2027**, by Jiandong Ding, Huawei Technologies
Co., Ltd.

## Install and run

Use Python 3.10 or newer. The core framework runs on CPU with NumPy.

```sh
git clone https://github.com/jdding/focalswap.git
cd focalswap
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/custom_model.py
```

This example loads a synthetic dataset, calls a deterministic recommender,
constructs matched donors, and audits both the latest search slot and all search
inputs. It writes scores, a donor map, and an aggregate report under
`outputs/custom_model/`.

Replay the exported scores through the installed CLI.

```sh
focalswap outputs/custom_model/readouts.jsonl \
  --output outputs/custom_model/replay.json \
  --bootstrap-iterations 1000 --family-size 2
```

The CLI and Python API produce the same estimates and intervals for the same
settings. The example uses 1,000 bootstrap draws to keep the walkthrough quick.
The default is 10,000 draws.

## Use your dataset and model

The [data and model integration guide](docs/integration.md) defines the row
format, scoring callback, and search interventions. Start with
[the model example](examples/custom_model.py) and replace its `predict(row)`
function with your model's scorer. Return one score per candidate in the given
order, with larger scores ranking higher.

```python
from focalswap import analyze, collect_scores

readouts = collect_scores(
    rows=my_rows,
    scorer=my_model_score,
    interventions={
        "latest_slot": {"C0": mask_latest, "C2": replace_latest},
        "all_source": {"C0": mask_all, "C2": replace_all},
    },
)
report = analyze(readouts, seed=2026, bootstrap_iterations=10000, family_size=2)
```

Intervention functions return a replacement search list. Returning `None` marks
that row ineligible for the control. The collector changes only the search
field and checks that scoring leaves the supplied row intact. Freeze your model
and preprocessing before collecting scores. For neural scorers, use evaluation
mode and disable gradient tracking.

You can also export scores or target ranks from an existing evaluation pipeline
and run `focalswap readouts.jsonl --output audit.json`. This path works with models
and data formats that stay outside the Python package.

## Read the audit

Every scope reports the evaluated rows, eligible users, coverage, Full-input RR,
and each control's RR. It also reports

- **Signed contribution**, the mean original RR minus control RR.
- **Rank response**, the fraction of rows whose target rank changes.
- **Replacement preference**, the fraction of within-row wins plus half of ties.
- **Paired control contrasts**, contribution differences on the same eligible rows.

Candidate-score ties use midpoint ranks. Intervals resample whole users while
retaining the row-weighted mean. `--family-size N` applies a Bonferroni adjustment
for a declared family of N tests. Reports record the confidence level, seed,
resampling settings, and coverage. Comparisons with fewer than two users retain
point estimates and mark their intervals unavailable.

Neutral masking and observed-search replacements answer different questions.
Choose matching fields from pre-outcome inputs, inspect donor coverage, and
interpret each estimate under its named control. A confidence interval crossing
zero leaves the direction unresolved. See the integration guide for the exact
report fields and donor construction rules.

## Reproduce the paper

On 66,139 identical rows across three UniSAR checkpoints, all-source masking
yields positive RR contributions while matched replacements yield smaller
estimates. The adjusted paired contrasts are positive, including under five
alternative donor assignments per checkpoint. The paper also reports SESRec,
GenSAR calibration, planted controls, and KuaiSAR/Qilin proxy analyses.

The [paper reproduction guide](ARTIFACT_QUICKSTART.md) covers aggregate checks,
Table 3(c), Figures 2–4, and the original SESRec/UniSAR adapters. These adapters
preserve the paper's model-specific preprocessing and experimental contracts.

```sh
bash scripts/run_smoke.sh
python -m unittest discover -s tests -v
```

[The evidence guide](evidence/README.md) describes the included aggregate results.
Original checkpoint experiments additionally need the datasets, model code, and
compatible weights described in the reproduction guide. All data in the generic
integration example are synthetic.

## Cite

The paper is accepted at WSDM 2027. The citation will be updated with the
publisher DOI and final pagination when available.

```bibtex
@inproceedings{ding2027focalswap,
  author = {Ding, Jiandong},
  title = {Do Search-Augmented Recommenders Use the Latest Search Episode?},
  booktitle = {Proceedings of the 20th ACM International Conference on Web Search and Data Mining},
  year = {2027}
}
```

Machine-readable citation information is in [CITATION.cff](CITATION.cff).

## Contents and license

`focalswap/` contains the reusable API and CLI. `examples/` and `docs/` describe
integration. `tools/`, `evidence/`, and `paper/` support the paper experiments.
The package manifest records file hashes for the release.

Code is available under [the MIT license](LICENSE). Third-party datasets and
model implementations retain their providers' terms.
