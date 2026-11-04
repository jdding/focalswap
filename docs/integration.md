# Connect your data and recommender

## Prepare fixed recommendation rows

Each model input is a JSON-compatible object. Identifiers are strings. Each row
has one observed target among an ordered list of distinct candidates.

```json
{
  "row_id": "recommendation-1",
  "user_id": "user-1",
  "target_id": "camera",
  "candidate_ids": ["camera", "lens", "novel"],
  "history": ["photo"],
  "search": [{"query": "photo"}, {"query": "camera"}],
  "match": {"category": "photo", "length_bucket": "two-episodes"}
}
```

`history` contains information available before the focal search. `search` is a
chronological list, with the latest episode last. You control the contents of
these fields, including any tokenization, session boundaries, and model-specific
IDs. Extra fields can carry frozen context needed by your scorer. Convert tensor
inputs and NumPy arrays to lists when using the collector.

Build the candidates and observed target once. The collector keeps those fields
and all context outside `search` fixed. Use separate audits for different models
or checkpoints, so `full` always denotes one frozen scorer.

## Implement scoring and interventions

`scorer(row)` returns a Python list of finite scores aligned with
`candidate_ids`. Higher scores rank better. A PyTorch adapter can convert its
output with `scores.detach().cpu().tolist()`. Keep the model in evaluation mode
and use `torch.no_grad()` during inference.

An intervention receives a copy of the original row and returns its replacement
`search` list. Latest-slot interventions keep earlier episodes unchanged.
All-source interventions change the complete list. A neutral input must be
compatible with the model's vocabulary and input structure.

The collector makes a fresh copy for each scoring call and rejects callbacks
that mutate it. It scores Full once per row and reuses that result across scopes.
For large evaluations, collect readouts in batches and concatenate them before
analysis so each user is resampled as a complete cluster.

The [complete example](../examples/custom_model.py) includes all four functions
and a scorer. Run it after installation.

```sh
python examples/custom_model.py --data examples/custom_dataset.jsonl \
  --output-dir outputs/custom_model
```

## Select matched donors

`match_donors(rows, fields=("category", "length_bucket"), seed=2026)` groups rows
by exact values in their `match` dictionaries. It sorts identifiers, shuffles
within each bucket, then uses a cyclic assignment. Each donor is used at most
once. The helper excludes singleton buckets, identical search inputs, and
same-user pairs by default. Excluded rows map to `None`.

This single assignment can leave rows unmatched even when a different assignment
would provide donors. Inspect coverage, and use your own donor map when your data
require different constraints. Define matching fields before observing the
recommendation outcome. Buckets should describe the selected input's category
and profile. The paper's UniSAR adapter matches source slots with its original
category/profile buckets; the generic helper matches whole row-level search
inputs.

Supply donor searches only from the permitted data split and time window. The
helper reads matching metadata, user IDs, and search inputs. Dataset-specific
rules for time, exposure, consent, and model vocabulary belong in your adapter.
The example writes its donor map locally for inspection.

## Analyze exported scores or ranks

A JSONL readout has the row identifiers, target and candidates, a scope name, and
a `views` object. The model callback and the CLI share this representation.

```json
{"row_id":"r1","user_id":"u1","target_id":"b","candidate_ids":["a","b","c"],"scope":"latest_slot","views":{"full":[0.3,0.9,0.1],"C0":[0.8,0.5,0.1],"C2":null}}
```

A view may instead contain a target rank, such as `"C0": {"rank": 2.5}`.
Ranks must be integer or half-integer values within the candidate list. Score
arrays use midpoint ranks for exact ties. Keep this same rank convention when
exporting ranks from your model. Explicit candidate IDs document the frozen
slate; rank-only exports rely on the producing adapter to preserve it.

Every scope/row pair must be unique. Row IDs may recur across scopes with the
same user, target, candidate order, and Full view. Multiple targets in an event
can be audited as separate fixed-target rows sharing one user ID.

```sh
focalswap outputs/custom_model/readouts.jsonl \
  --output outputs/custom_model/replay.json \
  --bootstrap-iterations 1000 --seed 2026 --family-size 2
```

Use `--compare C0 C2` to name a paired contribution contrast. Repeat the option
for other control pairs. The default pair is C0 and C2. A pair is evaluated on
rows eligible for both controls, with its own coverage denominator and user
count. The paired object includes both contributions recomputed on that subset.

## Interpret the report

`scopes` maps scope names to `total_rows`, `controls`, and `paired` summaries.
Each summary gives `rows`, `user_clusters`, `coverage`, and `metrics`. A metric
has `estimate`, `ci_low`, and `ci_high` fields.

- `signed_contribution` is mean RR(Full) minus RR(control).
- `rank_response` is the fraction with different target ranks.
- `replacement_preference` is wins plus half of ties, divided by eligible rows.
- `absolute_rank_shift` is the mean absolute target-rank change.
- `contribution_difference` is the left control's contribution minus the right
  control's contribution on the common rows.

Intervals resample whole users with replacement and retain the row-weighted
mean. The seed and number of draws are recorded. An interval requires at least
two eligible users. Each named control uses its own eligible rows. Paired
comparisons always use the intersection.

`confidence=0.95, family_size=2` produces 97.5% intervals. Declare the family that
matches the inference you intend to make; the package does not infer a test
family from the number of scopes, controls, metrics, or model runs. Directional
RR evidence, preference over replacements, and ranking response remain separate
measurements. Routing opportunity and recovery analyses from the paper have
their own calibrated nulls in the original experiment tools.

The CLI report contains only aggregate statistics and a SHA-256 digest of the
input file. Score exports and donor maps contain row/user identifiers. Keep them
in your own data environment and share aggregate results when appropriate.
