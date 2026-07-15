#!/usr/bin/env python3
"""Analyze R242 on common eligible rows with user-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (20230601, 20230602, 20230603)
EXPECTED_ROWS = 157192
EXPECTED_USERS = 4852
SCOPE_COLUMNS = {
    "latest_1": {
        "eligible": "category_latest_1_eligible",
        "C0_neutral": "neutral_latest_1_rr",
        "C1_profile_prioritized_valid": "valid_latest_1_rr",
        "C2_category_valid": "category_latest_1_rr",
        "C3_same_user_temporal": "same_user_latest_1_rr",
        "C3_eligible": "same_user_latest_1_eligible",
    },
    "all": {
        "eligible": "category_all_eligible",
        "C0_neutral": "neutral_all_rr",
        "C1_profile_prioritized_valid": "valid_all_rr",
        "C2_category_valid": "category_all_rr",
        "C3_same_user_temporal": "same_user_all_rr",
        "C3_eligible": "same_user_all_eligible",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r242-dir", required=True)
    parser.add_argument("--r233-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    parser.add_argument("--min-category-coverage", type=float, default=0.05)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_bootstrap(
    values: np.ndarray,
    users: np.ndarray,
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float, int]:
    if len(values) == 0:
        raise ValueError("cluster bootstrap requires at least one row")
    user_values, user_codes = np.unique(users, return_inverse=True)
    sums = np.bincount(user_codes, weights=values, minlength=len(user_values)).astype(np.float64)
    counts = np.bincount(user_codes, minlength=len(user_values)).astype(np.float64)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    probabilities = np.full(len(user_values), 1.0 / len(user_values))
    for start in range(0, iterations, 256):
        stop = min(start + 256, iterations)
        weights = rng.multinomial(len(user_values), probabilities, size=stop - start)
        draws[start:stop] = (weights @ sums) / (weights @ counts)
    low, high = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(low), float(high), len(user_values)


def basename_hashes(values: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path, digest in values.items():
        name = Path(path).name
        if name in output and output[name] != digest:
            raise RuntimeError(f"ambiguous provenance basename {name}")
        output[name] = str(digest)
    return output


def load_seed(
    r242_dir: Path, r233_dir: Path, seed: int, reference: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    r242_csv = r242_dir / f"r242_unisar_seed_{seed}_control_scores.csv"
    r242_json = r242_dir / f"r242_unisar_seed_{seed}_integrity.json"
    r233_csv = r233_dir / f"r233_unisar_seed_{seed}_scope_scores.csv"
    r233_json = r233_dir / f"r233_unisar_seed_{seed}_integrity.json"
    for path in (r242_csv, r242_json, r233_csv, r233_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    integrity = json.loads(r242_json.read_text())
    r233_integrity = json.loads(r233_json.read_text())
    expected = reference["seeds"][str(seed)]
    if integrity.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError(f"R242 integrity failed for seed {seed}")
    if integrity.get("output_csv_sha256") != sha256_file(r242_csv):
        raise RuntimeError(f"R242 score hash mismatch for seed {seed}")
    if int(r233_integrity.get("seed", -1)) != seed:
        raise RuntimeError(f"R233 seed mismatch for seed {seed}")
    if r233_integrity.get("output_csv_sha256") != sha256_file(r233_csv):
        raise RuntimeError(f"R233 score hash mismatch for seed {seed}")
    if sha256_file(r233_csv) != expected["r233_score_sha256"]:
        raise RuntimeError(f"R233 score differs from frozen reference for seed {seed}")
    if sha256_file(r233_json) != expected["r233_integrity_sha256"]:
        raise RuntimeError(f"R233 integrity differs from frozen reference for seed {seed}")
    if r233_integrity.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError(f"R233 integrity failed for seed {seed}")
    if r233_integrity.get("checkpoint_sha256") != integrity.get("checkpoint_sha256"):
        raise RuntimeError(f"R233/R242 checkpoint mismatch for seed {seed}")
    if integrity.get("checkpoint_sha256") != expected["checkpoint_sha256"]:
        raise RuntimeError(f"checkpoint differs from frozen reference for seed {seed}")
    r233_lineage = r233_integrity.get("execution_lineage", {})
    r242_lineage = integrity.get("execution_lineage", {})
    if r233_lineage.get("official_git_commit") != r242_lineage.get("official_git_commit"):
        raise RuntimeError(f"R233/R242 official commit mismatch for seed {seed}")
    if r242_lineage.get("official_git_commit") != reference["official_unisar_commit"]:
        raise RuntimeError(f"official commit differs from frozen reference for seed {seed}")
    if basename_hashes(r233_lineage.get("processed_input_sha256", {})) != basename_hashes(
        r242_lineage.get("processed_input_sha256", {})
    ):
        raise RuntimeError(f"R233/R242 processed-input mismatch for seed {seed}")
    donor = r233_integrity.get("donor_map", {})
    if donor.get("bijection") is not True or donor.get("future_label_or_score_used") is not False:
        raise RuntimeError(f"R233 donor lineage failed for seed {seed}")
    category = integrity.get("category_donor_map", {})
    if (
        category.get("bijection_on_eligible_support") is not True
        or int(category.get("max_donor_reuse", -1)) > 1
        or float(category.get("induced_occurrence_total_variation_on_eligible_support", 1.0)) > 0.15
    ):
        raise RuntimeError(f"R242 category-control admissibility failed for seed {seed}")
    r242 = pd.read_csv(r242_csv)
    r233 = pd.read_csv(r233_csv)
    if len(r242) != EXPECTED_ROWS or len(r233) != EXPECTED_ROWS:
        raise RuntimeError(f"row count mismatch for seed {seed}")
    keys = ["official_row_index", "user_id"]
    if not r242[keys].equals(r233[keys]):
        raise RuntimeError(f"row identity mismatch for seed {seed}")
    full_diff = np.abs(r242["full_rr"].to_numpy() - r233["full_rr"].to_numpy())
    if int(np.count_nonzero(full_diff > 1e-12)):
        raise RuntimeError(f"Full RR does not reproduce R233 for seed {seed}")
    keep = keys + ["source_slot_count", "full_rr"] + [
        value
        for columns in SCOPE_COLUMNS.values()
        for key, value in columns.items()
        if key in {"eligible", "C2_category_valid", "C3_same_user_temporal", "C3_eligible"}
    ]
    output = r242[list(dict.fromkeys(keep))].copy()
    for scope, columns in SCOPE_COLUMNS.items():
        output[columns["C0_neutral"]] = r233[columns["C0_neutral"]]
        output[columns["C1_profile_prioritized_valid"]] = r233[
            columns["C1_profile_prioritized_valid"]
        ]
    return output, integrity, r233_integrity


def interval_status(low: float, high: float) -> str:
    if low > 0:
        return "positive"
    if high < 0:
        return "negative"
    return "unresolved"


def main() -> None:
    args = parse_args()
    r242_dir = Path(args.r242_dir)
    r233_dir = Path(args.r233_dir)
    reference_path = Path(args.reference_manifest)
    reference = json.loads(reference_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    lineage: dict[str, Any] = {}
    for seed in EXPECTED_SEEDS:
        frame, integrity, r233_integrity = load_seed(r242_dir, r233_dir, seed, reference)
        lineage[str(seed)] = {
            "r242_score_sha256": integrity["output_csv_sha256"],
            "checkpoint_sha256": integrity["checkpoint_sha256"],
            "category_donor_map_sha256": integrity["category_donor_map"]["donor_map_sha256"],
            "r233_score_sha256": r233_integrity["output_csv_sha256"],
            "r233_integrity_sha256": sha256_file(
                r233_dir / f"r233_unisar_seed_{seed}_integrity.json"
            ),
            "r233_donor_route_counts": r233_integrity["donor_map"]["route_counts"],
        }
        for scope_position, (scope, columns) in enumerate(SCOPE_COLUMNS.items()):
            common = frame[columns["eligible"]].to_numpy(dtype=bool)
            coverage = float(common.mean())
            user_coverage = float(frame.loc[common, "user_id"].nunique() / EXPECTED_USERS)
            if min(coverage, user_coverage) < args.min_category_coverage:
                raise RuntimeError(
                    f"C2 row/user coverage {coverage:.6f}/{user_coverage:.6f} "
                    f"below gate for {scope}, seed {seed}"
                )
            full = frame["full_rr"].to_numpy(dtype=np.float64)
            users = frame["user_id"].to_numpy()
            for control_position, control in enumerate(
                ("C0_neutral", "C1_profile_prioritized_valid", "C2_category_valid")
            ):
                controlled = frame[columns[control]].to_numpy(dtype=np.float64)
                primary = control == "C2_category_valid"
                alpha = 0.025 if primary else 0.05
                point, low, high, user_count = cluster_bootstrap(
                    (full - controlled)[common],
                    users[common],
                    args.bootstrap_iters,
                    args.bootstrap_seed + seed + 100 * scope_position + control_position,
                    alpha=alpha,
                )
                rows.append(
                    {
                        "seed": seed,
                        "scope": scope,
                        "control": control,
                        "rows": int(common.sum()),
                        "coverage": coverage,
                        "eligible_user_coverage": user_coverage,
                        "user_clusters": user_count,
                        "mean_source_slots_eligible": float(
                            frame.loc[common, "source_slot_count"].mean()
                        ),
                        "mean_source_slots_ineligible": float(
                            frame.loc[~common, "source_slot_count"].mean()
                        ) if bool((~common).any()) else None,
                        "delta_rr": point,
                        "ci_low": low,
                        "ci_high": high,
                        "status": interval_status(low, high),
                        "comparison_population": "C2_common_support",
                        "ci_level": 0.975 if primary else 0.95,
                        "promotion_endpoint": primary,
                    }
                )
            c3 = frame[columns["C3_eligible"]].to_numpy(dtype=bool)
            c3_controlled = frame[columns["C3_same_user_temporal"]].to_numpy(dtype=np.float64)
            if bool(c3.any()):
                point, low, high, user_count = cluster_bootstrap(
                    (full - c3_controlled)[c3],
                    users[c3],
                    args.bootstrap_iters,
                    args.bootstrap_seed + seed + 100 * scope_position + 99,
                )
                status = interval_status(low, high)
            else:
                point = low = high = float("nan")
                user_count = 0
                status = "not_estimable"
            rows.append(
                {
                    "seed": seed,
                    "scope": scope,
                    "control": "C3_same_user_temporal",
                    "rows": int(c3.sum()),
                    "coverage": float(c3.mean()),
                    "eligible_user_coverage": float(
                        frame.loc[c3, "user_id"].nunique() / EXPECTED_USERS
                    ),
                    "user_clusters": user_count,
                    "mean_source_slots_eligible": float(
                        frame.loc[c3, "source_slot_count"].mean()
                    ) if bool(c3.any()) else None,
                    "mean_source_slots_ineligible": float(
                        frame.loc[~c3, "source_slot_count"].mean()
                    ) if bool((~c3).any()) else None,
                    "delta_rr": point,
                    "ci_low": low,
                    "ci_high": high,
                    "status": status,
                    "comparison_population": "C3_strict_support",
                    "ci_level": 0.95,
                    "promotion_endpoint": False,
                }
            )
    summary = pd.DataFrame(rows)
    csv_path = output_dir / "r242_control_ladder_summary.csv"
    json_path = output_dir / "r242_control_ladder_summary.json"
    report_path = output_dir / "r242_control_ladder_report.md"
    summary.to_csv(csv_path, index=False)
    consistency = []
    for (scope, control), group in summary.groupby(["scope", "control"], sort=False):
        statuses = group["status"].tolist()
        consistency.append(
            {
                "scope": scope,
                "control": control,
                "statuses": statuses,
                "positive_seeds": statuses.count("positive"),
                "negative_seeds": statuses.count("negative"),
                "unresolved_seeds": statuses.count("unresolved"),
                "mean_delta_rr": float(group["delta_rr"].mean()),
                "min_coverage": float(group["coverage"].min()),
                "min_eligible_user_coverage": float(group["eligible_user_coverage"].min()),
                "promotion_status": (
                    "PROMOTE_ELIGIBLE_POPULATION"
                    if control == "C2_category_valid"
                    and statuses == ["positive", "positive", "positive"]
                    else "NO_PROMOTION"
                ),
            }
        )
    payload = {
        "schema": "focalswap.r242.control_ladder_summary.v1",
        "bootstrap": {
            "iterations": args.bootstrap_iters,
            "unit": "user_cluster",
            "seed_base": args.bootstrap_seed,
        },
        "lineage": lineage,
        "reference_manifest_sha256": sha256_file(reference_path),
        "rows": rows,
        "consistency": consistency,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# R242 Category-Controlled Source-Dependence Ladder",
        "",
        "All C0-C2 estimates use the same C2-eligible rows within each scope.",
        "C3 uses its own strict same-user temporal support and is reported separately.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "No model or architecture claim is authorized by this report alone.",
    ]
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "PASS", "rows": len(summary), "output": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
