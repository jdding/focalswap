#!/usr/bin/env python3
"""Random within-stratum reference for the R242 C2 occurrence-TV statistic.

The paper does not use the observed TV value as matching-quality evidence until
this script can bind it to the hash-checked UniSAR processed test bundle and a
random donor reference under the same C2 eligibility strata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unisar_five_view_readout import load_pickle
from unisar_semantic_control_ladder import (
    build_category_donor_map,
    frequency_bucket,
    item_count_bucket,
    nonzero_count,
    source_category_sets,
)


EXPECTED_REC_TEST_SHA256 = "ead6a99fda7c7a4068facedf87a3e4717bff38aa15ad766cd2a0396020fdcbed"
EXPECTED_ROWS = 157_192
EXPECTED_TV = 0.11082421037637191


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iterations", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--intervention-seed", type=int, default=20260712)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_root(path: Path) -> Path:
    candidate = path / "KuaiSAR"
    return candidate if candidate.is_dir() else path


def collect_counts(rec_df: pd.DataFrame, user_vocab: dict[int, dict[str, Any]]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for row in rec_df.itertuples(index=False):
        user_id = int(row.user_id)
        source_count = int(row.src_session_his)
        source_ids = user_vocab[user_id]["src_session_his"][:source_count]
        counts.update(int(value) for value in source_ids if int(value) > 0)
    return counts


def c2_groups(
    src_map: dict[str, np.ndarray],
    item_categories: np.ndarray,
    observed_counts: Counter[int],
) -> list[np.ndarray]:
    valid_ids = sorted(int(value) for value in observed_counts)
    categories = source_category_sets(src_map, item_categories)
    profile_exact: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    profiles: dict[int, tuple[str, str, str]] = {}
    for source_id in valid_ids:
        item_bucket = item_count_bucket(nonzero_count(src_map["pos_items"][source_id]))
        query_bucket = item_count_bucket(nonzero_count(src_map["keyword"][source_id]))
        count_bucket = frequency_bucket(observed_counts[source_id])
        profiles[source_id] = (item_bucket, query_bucket, count_bucket)
        for category in categories[source_id]:
            profile_exact[(category, item_bucket, query_bucket, count_bucket)].append(source_id)

    assigned: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    for source_id in valid_ids:
        if not categories[source_id]:
            continue
        item_bucket, query_bucket, count_bucket = profiles[source_id]
        canonical = sorted(
            categories[source_id],
            key=lambda category: (
                -len(profile_exact[(category, item_bucket, query_bucket, count_bucket)]),
                category,
            ),
        )[0]
        assigned[(canonical, item_bucket, query_bucket, count_bucket)].append(source_id)
    return [np.asarray(sorted(set(values)), dtype=np.int64) for _, values in sorted(assigned.items()) if len(set(values)) >= 2]


def tv_from_mapping(
    mapping: np.ndarray, eligible_ids: np.ndarray, observed_counts: Counter[int]
) -> float:
    original = np.asarray([observed_counts[int(value)] for value in eligible_ids], dtype=np.float64)
    positions = {int(value): index for index, value in enumerate(eligible_ids)}
    induced = np.zeros_like(original)
    for source_id, count in zip(eligible_ids, original):
        induced[positions[int(mapping[int(source_id)])]] += count
    return float(0.5 * np.abs(original - induced).sum() / original.sum())


def random_cycle_tv(
    groups: list[np.ndarray], observed_counts: Counter[int], rng: np.random.Generator
) -> float:
    numerator = 0.0
    denominator = 0.0
    for group in groups:
        counts = np.asarray([observed_counts[int(value)] for value in group], dtype=np.float64)
        order = rng.permutation(len(group))
        offset = int(rng.integers(1, len(group)))
        induced = np.empty_like(counts)
        induced[order[(np.arange(len(group)) + offset) % len(group)]] = counts[order]
        numerator += float(np.abs(counts - induced).sum())
        denominator += float(counts.sum())
    return 0.5 * numerator / denominator


def main() -> None:
    args = parse_args()
    root = source_root(Path(args.processed_data_root))
    rec_path = root / "dataset" / "rec_test.pkl"
    if sha256_file(rec_path) != EXPECTED_REC_TEST_SHA256:
        raise RuntimeError("rec_test.pkl hash does not match the audited R242 input")
    rec_df = pd.read_pickle(rec_path).reset_index(drop=True)
    if len(rec_df) != EXPECTED_ROWS:
        raise RuntimeError("processed test row count changed")
    user_vocab = load_pickle(root / "vocab" / "user_vocab.pkl")
    src_map = load_pickle(root / "vocab" / "src_session_vocab_np.pkl")
    item_map = load_pickle(root / "vocab" / "item_vocab_np.pkl")
    item_categories = np.asarray(item_map["first_level_category_id"], dtype=np.int64)
    observed_counts = collect_counts(rec_df, user_vocab)
    if sum(observed_counts.values()) != 2_675_881:
        raise RuntimeError("observed source-slot occurrence count changed")

    groups = c2_groups(src_map, item_categories, observed_counts)
    eligible_ids = np.concatenate(groups)
    neutral_id = int(np.asarray(src_map["keyword"]).shape[0])
    observed_mapping, summary = build_category_donor_map(
        src_map, item_categories, observed_counts, neutral_id, args.intervention_seed
    )
    observed_tv = tv_from_mapping(observed_mapping.numpy(), eligible_ids, observed_counts)
    if not np.isclose(observed_tv, EXPECTED_TV, atol=1e-12, rtol=0):
        raise RuntimeError("reconstructed C2 TV does not match the audited value")

    rng = np.random.default_rng(args.seed)
    baseline = np.asarray(
        [random_cycle_tv(groups, observed_counts, rng) for _ in range(args.iterations)],
        dtype=np.float64,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "r250_tv_random_cycle_baseline.csv", baseline, delimiter=",", header="tv", comments="")
    payload = {
        "schema": "focalswap.r250.tv_random_baseline.v1",
        "status": "COMPLETE",
        "definition": "half the L1 distance between original and donor-induced exact-ID occurrence distributions on C2-eligible support",
        "reference": "random single-cycle derangement independently sampled within every exact C2 category/profile stratum",
        "iterations": args.iterations,
        "seed": args.seed,
        "observed_tv": observed_tv,
        "baseline_mean": float(baseline.mean()),
        "baseline_median": float(np.median(baseline)),
        "baseline_p05": float(np.quantile(baseline, 0.05)),
        "baseline_p95": float(np.quantile(baseline, 0.95)),
        "observed_lower_tail_fraction": float((1 + np.sum(baseline <= observed_tv)) / (args.iterations + 1)),
        "eligible_source_ids": int(len(eligible_ids)),
        "eligible_source_occurrences": int(summary["eligible_source_slot_occurrences"]),
        "strata": int(len(groups)),
        "rec_test_sha256": EXPECTED_REC_TEST_SHA256,
    }
    (output_dir / "r250_tv_random_baseline.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = [
        "# R250 C2 TV Random Reference",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Observed TV: {observed_tv:.4f}",
        f"- Random within-stratum cycle baseline: median {payload['baseline_median']:.4f}, "
        f"5th--95th percentile [{payload['baseline_p05']:.4f}, {payload['baseline_p95']:.4f}].",
        f"- Lower-tail fraction: {payload['observed_lower_tail_fraction']:.4f}.",
        f"- Support: {len(eligible_ids):,} source IDs in {len(groups):,} exact C2 strata.",
    ]
    (output_dir / "r250_tv_random_baseline.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
