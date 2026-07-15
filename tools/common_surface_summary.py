#!/usr/bin/env python3
"""Aggregate six R225 common-surface checkpoint readouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("sesrec", "unisar")
SEEDS = (20_230_601, 20_230_602, 20_230_603)
LEVELS = (
    "l1_nuisance",
    "l2_same_category",
    "category_broken_source",
    "same_category_source",
)
EXPECTED_ROWS = 3_249
EXPECTED_SURFACE = "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED"
EXPECTED_CANDIDATE_SHA256 = "d7b3b9c9f83da8b4d67074951e1bc9f2a3f4e0816804a1cbaafad87029e8a004"
EXPECTED_SWAP_SHA256 = "1d34a0c5cbc9a9c0e6bf30838c49d753478ea14c6e679475d0c69d8cab13cf0c"
EXPECTED_COMMITS = {
    "sesrec": "de9c0ae10066704d2e97783ce2e6985a4dea508e",
    "unisar": "1a14a246066cd1aa6ee789c57ff2c7499f3d7379",
}
EXPECTED_SCORERS = {
    "sesrec": "SESRec.parse_input_test->his2feats->interest_contrast->inter_pred",
    "unisar": "UniSAR.rec_predict",
}
RANDOM_NDCG10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, 11)) / 100.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-prefix", default="r225_common_surface_summary")
    parser.add_argument("--bootstrap-iters", type=int, default=5_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_710)
    return parser.parse_args()


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def cluster_ci(
    rows: list[dict[str, str]],
    value_fn: Any,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        value = value_fn(row)
        if value is None:
            continue
        user = str(row["user_id"])
        sums[user] += float(value)
        counts[user] += 1
    users = sorted(counts)
    if not users:
        return {"n": 0, "users": 0, "mean": math.nan, "ci_low": math.nan, "ci_high": math.nan}
    user_sums = np.asarray([sums[user] for user in users], dtype=np.float64)
    user_counts = np.asarray([counts[user] for user in users], dtype=np.float64)
    point = float(user_sums.sum() / user_counts.sum())
    rng = np.random.default_rng(seed)
    values = np.empty(bootstrap_iters, dtype=np.float64)
    for index in range(bootstrap_iters):
        sample = rng.integers(0, len(users), size=len(users))
        values[index] = user_sums[sample].sum() / user_counts[sample].sum()
    return {
        "n": int(user_counts.sum()),
        "users": len(users),
        "mean": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
    }


def half_tie_value(row: dict[str, str], level: str) -> float | None:
    if int(row[f"{level}_matched"]) <= 0:
        return None
    delta = float(row[f"{level}_score_delta"])
    return 1.0 if delta > 0.0 else 0.0 if delta < 0.0 else 0.5


def alignment_digest(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    fields = ["row_index", "example_id", "user_id", "target_item", "candidate_count"]
    fields.extend(f"{level}_swap_row_index" for level in LEVELS)
    for row in rows:
        digest.update(("|".join(row[field] for field in fields) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    expected_pairs = {(method, seed) for method in METHODS for seed in SEEDS}
    actual_pairs = {(str(row["method"]), int(row["seed"])) for row in manifest}
    if actual_pairs != expected_pairs or len(manifest) != 6:
        raise ValueError(f"manifest pairs={sorted(actual_pairs)} expected={sorted(expected_pairs)}")

    results: list[dict[str, Any]] = []
    alignment_hashes: set[str] = set()
    for run_index, item in enumerate(sorted(manifest, key=lambda row: (row["method"], row["seed"]))):
        method = str(item["method"])
        seed = int(item["seed"])
        artifact_hashes = item.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict):
            raise ValueError(f"{method}/{seed} manifest lacks artifact hashes")
        artifact_paths = {
            "checkpoint": resolve(item["checkpoint"]),
            "readout_json": resolve(item["readout_json"]),
            "scores_csv": resolve(item["scores_csv"]),
            "train_log": resolve(item["train_log"]),
        }
        for field, path in artifact_paths.items():
            expected_hash = artifact_hashes.get(field)
            if not isinstance(expected_hash, str) or file_sha256(path) != expected_hash:
                raise ValueError(f"{method}/{seed} {field} hash mismatch")
        payload = json.loads(artifact_paths["readout_json"].read_text(encoding="utf-8"))
        rows = read_csv(artifact_paths["scores_csv"])
        if len(rows) != EXPECTED_ROWS:
            raise ValueError(f"{method}/{seed} rows={len(rows)} expected={EXPECTED_ROWS}")
        expected_status = (
            "done" if method == "sesrec" else "PASS_R225_UNISAR_CHECKPOINT_READOUT"
        )
        if payload.get("status") != expected_status:
            raise ValueError(f"{method}/{seed} status={payload.get('status')} expected={expected_status}")
        if payload.get("mode") != "checkpoint_readout":
            raise ValueError(f"{method}/{seed} is not a checkpoint readout")
        if payload.get("surface_id") != EXPECTED_SURFACE:
            raise ValueError(f"{method}/{seed} surface mismatch")
        if payload.get("official_commit") != EXPECTED_COMMITS[method]:
            raise ValueError(f"{method}/{seed} official commit mismatch")
        if payload.get("specificity_basis") != "target_score":
            raise ValueError(f"{method}/{seed} specificity basis is not target_score")
        if payload.get("candidate_contract") != "canonical_raw_item_ledger":
            raise ValueError(f"{method}/{seed} candidate contract mismatch")
        if payload.get("candidate_ledger_sha256") != EXPECTED_CANDIDATE_SHA256:
            raise ValueError(f"{method}/{seed} candidate ledger hash mismatch")
        if payload.get("swap_ledger_sha256") != EXPECTED_SWAP_SHA256:
            raise ValueError(f"{method}/{seed} swap ledger hash mismatch")
        if payload.get("candidate_tensors_frozen") is not True:
            raise ValueError(f"{method}/{seed} candidate tensors not frozen")
        if payload.get("finite_all_scores") is not True:
            raise ValueError(f"{method}/{seed} contains nonfinite scores")
        if method == "sesrec" and payload.get("finite_neutral_controls") is not True:
            raise ValueError(f"{method}/{seed} did not use finite neutral controls")
        if payload.get("same_native_scorer") != EXPECTED_SCORERS[method]:
            raise ValueError(f"{method}/{seed} scorer path mismatch")
        if int(payload.get("metrics", {}).get("rows", -1)) != EXPECTED_ROWS:
            raise ValueError(f"{method}/{seed} readout payload row count mismatch")
        alignment_hash = alignment_digest(rows)
        alignment_hashes.add(alignment_hash)
        dependence = cluster_ci(
            rows,
            lambda row: float(row["full_sar_rr"]) - float(row["search_masked_rr"]),
            args.bootstrap_iters,
            args.bootstrap_seed + 1009 * run_index,
        )
        ndcg = float(np.mean([float(row["full_sar_ndcg10"]) for row in rows]))
        base: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "rows": len(rows),
            "alignment_sha256": alignment_hash,
            "full_ndcg10": ndcg,
            "full_rr": float(np.mean([float(row["full_sar_rr"]) for row in rows])),
            "masked_rr": float(np.mean([float(row["search_masked_rr"]) for row in rows])),
            "dependence_delta": dependence["mean"],
            "dependence_ci_low": dependence["ci_low"],
            "dependence_ci_high": dependence["ci_high"],
            "user_clusters": dependence["users"],
        }
        for level_index, level in enumerate(LEVELS):
            estimate = cluster_ci(
                rows,
                lambda row, level=level: half_tie_value(row, level),
                args.bootstrap_iters,
                args.bootstrap_seed + 1009 * run_index + 97 * (level_index + 1),
            )
            base[f"{level}_n"] = estimate["n"]
            base[f"{level}_half_tie"] = estimate["mean"]
            base[f"{level}_ci_low"] = estimate["ci_low"]
            base[f"{level}_ci_high"] = estimate["ci_high"]
        results.append(base)

    method_quality: dict[str, Any] = {}
    quality_pass = True
    for method in METHODS:
        values = [row["full_ndcg10"] for row in results if row["method"] == method]
        method_pass = all(value > RANDOM_NDCG10 for value in values) and float(np.mean(values)) > 0.050
        method_quality[method] = {
            "seed_values": values,
            "mean": float(np.mean(values)),
            "random_expectation": RANDOM_NDCG10,
            "pass": method_pass,
        }
        quality_pass = quality_pass and method_pass
    alignment_pass = len(alignment_hashes) == 1
    status = (
        "PASS_R225_COMMON_SURFACE_EVIDENCE"
        if alignment_pass and quality_pass
        else "CLOSED_R225_CONTROLLED_PROVENANCE_ONLY"
    )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.reports_dir / f"{args.report_prefix}.csv"
    json_path = args.reports_dir / f"{args.report_prefix}.json"
    md_path = args.reports_dir / f"{args.report_prefix}.md"
    write_csv(csv_path, results)
    payload = {
        "schema": "focalswap.r225.common_surface_summary.v1",
        "status": status,
        "surface_id": EXPECTED_SURFACE,
        "manifest": str(args.manifest),
        "alignment_pass": alignment_pass,
        "alignment_sha256": next(iter(alignment_hashes)) if alignment_pass else None,
        "quality_gate_pass": quality_pass,
        "quality_gate": {
            "per_seed_min": RANDOM_NDCG10,
            "per_model_mean_min": 0.050,
            "methods": method_quality,
        },
        "bootstrap_iters": args.bootstrap_iters,
        "results": results,
        "claim_boundary": (
            "Controlled fixed-surface competence and model-family corroboration under "
            "row-scoped episode identities only; not native user-personalization performance, "
            "an official-dataset or leaderboard reproduction, or architecture-isolating evidence. "
            "The ID vocabulary is transductive with respect to held-out context IDs, while targets "
            "remain row-held-out and train-seen by the registered eligibility rule."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# R225 Common-Surface Summary",
        "",
        f"Status: `{status}`",
        "",
        f"- Cross-model row/candidate/swap alignment: `{'PASS' if alignment_pass else 'FAIL'}`",
        f"- Predeclared quality gate: `{'PASS' if quality_pass else 'FAIL'}`",
        f"- Random NDCG@10 expectation: `{RANDOM_NDCG10:.6f}`",
        "",
        "| Method | Seed | NDCG@10 | Full RR | Masked RR | Dependence delta [95% CI] | Category-broken half-tie [95% CI] | Same-category half-tie [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        lines.append(
            f"| {row['method']} | {row['seed']} | {row['full_ndcg10']:.3f} | "
            f"{row['full_rr']:.3f} | {row['masked_rr']:.3f} | "
            f"{row['dependence_delta']:+.3f} [{row['dependence_ci_low']:+.3f}, {row['dependence_ci_high']:+.3f}] | "
            f"{row['category_broken_source_half_tie']:.3f} [{row['category_broken_source_ci_low']:.3f}, {row['category_broken_source_ci_high']:.3f}] | "
            f"{row['same_category_source_half_tie']:.3f} [{row['same_category_source_ci_low']:.3f}, {row['same_category_source_ci_high']:.3f}] |"
        )
    lines.extend(["", "## Claim Boundary", "", payload["claim_boundary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
