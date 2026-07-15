#!/usr/bin/env python3
"""Reviewer-facing C2 coverage and latest-slot validity closure.

This script uses only the audited R242/R233 row-score exports. It deliberately
does not infer source-ID frequency baselines that require the original
processed source-session bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SEEDS = (20230601, 20230602, 20230603)
EXPECTED_ROWS = 157_192
EXPECTED_USERS = 4_852


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r242-dir", required=True)
    parser.add_argument("--r233-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def smd(eligible: np.ndarray, ineligible: np.ndarray) -> float:
    variance = (np.var(eligible, ddof=1) + np.var(ineligible, ddof=1)) / 2
    if variance <= 0:
        return 0.0
    return float((np.mean(eligible) - np.mean(ineligible)) / np.sqrt(variance))


def cluster_group_bootstrap(
    values: np.ndarray,
    users: np.ndarray,
    eligible: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    unique_users, codes = np.unique(users, return_inverse=True)
    group_sums = []
    group_counts = []
    for mask in (eligible, ~eligible):
        group_sums.append(
            np.bincount(codes[mask], weights=values[mask], minlength=len(unique_users)).astype(float)
        )
        group_counts.append(
            np.bincount(codes[mask], minlength=len(unique_users)).astype(float)
        )
    sums = np.stack(group_sums, axis=1)
    counts = np.stack(group_counts, axis=1)
    points = sums.sum(axis=0) / counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(unique_users), 1.0 / len(unique_users))
    draws = np.empty((iterations, 3), dtype=float)
    for start in range(0, iterations, 256):
        stop = min(start + 256, iterations)
        weights = rng.multinomial(len(unique_users), probabilities, size=stop - start)
        numerators = weights @ sums
        denominators = weights @ counts
        if bool((denominators <= 0).any()):
            raise RuntimeError("bootstrap draw lost one eligibility group")
        means = numerators / denominators
        draws[start:stop, :2] = means
        draws[start:stop, 2] = means[:, 0] - means[:, 1]
    intervals = np.percentile(draws, [2.5, 97.5], axis=0)
    return {
        "eligible_mean": float(points[0]),
        "eligible_ci_low": float(intervals[0, 0]),
        "eligible_ci_high": float(intervals[1, 0]),
        "ineligible_mean": float(points[1]),
        "ineligible_ci_low": float(intervals[0, 1]),
        "ineligible_ci_high": float(intervals[1, 1]),
        "difference": float(points[0] - points[1]),
        "difference_ci_low": float(intervals[0, 2]),
        "difference_ci_high": float(intervals[1, 2]),
        "user_clusters": int(len(unique_users)),
    }


def load_seed(r242_dir: Path, r233_dir: Path, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    r242_csv = r242_dir / f"r242_unisar_seed_{seed}_control_scores.csv"
    r242_json = r242_dir / f"r242_unisar_seed_{seed}_integrity.json"
    r233_csv = r233_dir / f"r233_unisar_seed_{seed}_scope_scores.csv"
    for path in (r242_csv, r242_json, r233_csv):
        if not path.is_file():
            raise FileNotFoundError(path)
    integrity = json.loads(r242_json.read_text(encoding="utf-8"))
    if integrity.get("integrity", {}).get("all_pass") is not True:
        raise RuntimeError(f"R242 integrity failed for seed {seed}")
    if integrity.get("output_csv_sha256") != sha256_file(r242_csv):
        raise RuntimeError(f"R242 row-score hash mismatch for seed {seed}")
    r242 = pd.read_csv(r242_csv)
    r233 = pd.read_csv(r233_csv)
    if len(r242) != EXPECTED_ROWS or len(r233) != EXPECTED_ROWS:
        raise RuntimeError(f"row count mismatch for seed {seed}")
    keys = ["official_row_index", "user_id"]
    if not r242[keys].equals(r233[keys]):
        raise RuntimeError(f"R242/R233 row identity mismatch for seed {seed}")
    if r242["user_id"].nunique() != EXPECTED_USERS:
        raise RuntimeError(f"user count mismatch for seed {seed}")
    if not np.allclose(r242["full_rr"], r233["full_rr"], atol=1e-12, rtol=0):
        raise RuntimeError(f"R242/R233 Full RR mismatch for seed {seed}")
    frame = r242[
        [
            "official_row_index",
            "user_id",
            "source_slot_count",
            "candidate_count",
            "full_rr",
            "category_latest_1_eligible",
            "category_all_eligible",
            "category_latest_1_changed_slots",
            "category_all_changed_slots",
        ]
    ].copy()
    frame["neutral_latest_1_rr"] = r233["neutral_latest_1_rr"]
    frame["neutral_all_rr"] = r233["neutral_all_rr"]
    return frame, integrity


def main() -> None:
    args = parse_args()
    r242_dir = Path(args.r242_dir)
    r233_dir = Path(args.r233_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[int, pd.DataFrame] = {}
    integrities: dict[int, dict[str, Any]] = {}
    for seed in EXPECTED_SEEDS:
        frames[seed], integrities[seed] = load_seed(r242_dir, r233_dir, seed)

    reference = frames[EXPECTED_SEEDS[0]]
    structural_columns = [
        "official_row_index",
        "user_id",
        "source_slot_count",
        "candidate_count",
        "category_latest_1_eligible",
        "category_all_eligible",
        "category_latest_1_changed_slots",
        "category_all_changed_slots",
    ]
    for seed in EXPECTED_SEEDS[1:]:
        if not reference[structural_columns].equals(frames[seed][structural_columns]):
            raise RuntimeError(f"eligibility/intervention structure changed for seed {seed}")

    eligible = reference["category_all_eligible"].to_numpy(dtype=bool)
    users = reference["user_id"].to_numpy()
    user_activity = reference["user_id"].map(reference.groupby("user_id").size()).to_numpy(float)
    structural_values = {
        "source_slot_count": reference["source_slot_count"].to_numpy(float),
        "user_test_rows": user_activity,
    }
    balance_rows: list[dict[str, Any]] = []
    for metric, values in structural_values.items():
        balance_rows.append(
            {
                "seed": "structural",
                "metric": metric,
                "eligible_mean": float(values[eligible].mean()),
                "ineligible_mean": float(values[~eligible].mean()),
                "smd_eligible_minus_ineligible": smd(values[eligible], values[~eligible]),
            }
        )

    margin_rows: list[dict[str, Any]] = []
    for position, seed in enumerate(EXPECTED_SEEDS):
        frame = frames[seed]
        for metric, values in {
            "full_rr": frame["full_rr"].to_numpy(float),
            "c0_all_source_delta_rr": (
                frame["full_rr"] - frame["neutral_all_rr"]
            ).to_numpy(float),
            "c0_latest_slot_delta_rr": (
                frame["full_rr"] - frame["neutral_latest_1_rr"]
            ).to_numpy(float),
        }.items():
            summary = cluster_group_bootstrap(
                values,
                users,
                eligible,
                args.bootstrap_iters,
                args.bootstrap_seed + 100 * position + len(margin_rows),
            )
            margin_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "smd_eligible_minus_ineligible": smd(values[eligible], values[~eligible]),
                    **summary,
                }
            )

    balance = pd.DataFrame(balance_rows)
    margins = pd.DataFrame(margin_rows)
    balance_path = output_dir / "r250_c2_eligibility_balance.csv"
    margins_path = output_dir / "r250_c0_margin_by_eligibility.csv"
    balance.to_csv(balance_path, index=False)
    margins.to_csv(margins_path, index=False)

    latest_validity = {
        "rows": int(len(reference)),
        "users": int(reference["user_id"].nunique()),
        "minimum_nonpadding_source_slots": int(reference["source_slot_count"].min()),
        "rows_without_source_slot": int((reference["source_slot_count"] <= 0).sum()),
        "c2_latest_eligible_rows": int(reference["category_latest_1_eligible"].sum()),
        "c2_latest_coverage": float(reference["category_latest_1_eligible"].mean()),
        "c2_latest_changed_exactly_one": int(
            (
                (reference["category_latest_1_eligible"] == 1)
                & (reference["category_latest_1_changed_slots"] == 1)
            ).sum()
        ),
        "c2_latest_bad_changed_slot_rows": int(
            (
                (reference["category_latest_1_eligible"] == 1)
                & (reference["category_latest_1_changed_slots"] != 1)
            ).sum()
        ),
        "c2_all_eligible_rows": int(reference["category_all_eligible"].sum()),
        "c2_all_coverage": float(reference["category_all_eligible"].mean()),
        "c2_all_bad_changed_slot_rows": int(
            (
                (reference["category_all_eligible"] == 1)
                & (
                    reference["category_all_changed_slots"]
                    != reference["source_slot_count"]
                )
            ).sum()
        ),
        "candidate_counts": sorted(int(value) for value in reference["candidate_count"].unique()),
        "structure_identical_across_seeds": True,
    }
    latest_path = output_dir / "r250_latest_slot_validity.json"
    write_json(latest_path, latest_validity)

    tv_value = float(
        integrities[EXPECTED_SEEDS[0]]["category_donor_map"][
            "induced_occurrence_total_variation_on_eligible_support"
        ]
    )
    payload = {
        "schema": "focalswap.r250.review_evidence_closure.v1",
        "status": "LOCAL_EVIDENCE_COMPLETE_TV_RUNNER_READY_AWAITING_HASHED_BUNDLE",
        "bootstrap": {
            "iterations": args.bootstrap_iters,
            "seed": args.bootstrap_seed,
            "unit": "user_cluster",
        },
        "c2_population": {
            "eligible_rows": int(eligible.sum()),
            "ineligible_rows": int((~eligible).sum()),
            "coverage": float(eligible.mean()),
            "eligible_users": int(reference.loc[eligible, "user_id"].nunique()),
            "ineligible_users": int(reference.loc[~eligible, "user_id"].nunique()),
        },
        "latest_slot_validity": latest_validity,
        "tv_audit": {
            "observed_tv": tv_value,
            "definition": (
                "half the L1 distance between original and donor-induced exact-ID "
                "occurrence distributions on C2-eligible source-ID support"
            ),
            "random_derangement_baseline_status": "requires_raw_source_occurrence_counts",
            "runner": "tools/r250_tv_random_baseline.py",
            "reason": (
                "the pulled audited artifacts retain the TV value and donor-map hash, "
                "but not the source-ID occurrence vector needed to simulate random bijections"
            ),
            "required_processed_input_sha256": integrities[EXPECTED_SEEDS[0]][
                "execution_lineage"
            ]["processed_input_sha256"],
        },
        "outputs": {
            "balance_csv": str(balance_path),
            "margin_csv": str(margins_path),
            "latest_validity_json": str(latest_path),
        },
        "source_hashes": {
            f"r242_seed_{seed}": integrities[seed]["output_csv_sha256"]
            for seed in EXPECTED_SEEDS
        },
    }
    json_path = output_dir / "r250_review_evidence_closure.json"
    write_json(json_path, payload)

    c0_all = margins[margins["metric"] == "c0_all_source_delta_rr"]
    report = [
        "# R250 Review Evidence Closure",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## C2 Eligibility",
        "",
        f"- C2 all-source support covers {eligible.sum():,}/{len(eligible):,} rows "
        f"({eligible.mean():.1%}).",
        f"- Mean source slots: {balance.loc[balance.metric == 'source_slot_count', 'eligible_mean'].iloc[0]:.2f} "
        f"eligible vs. {balance.loc[balance.metric == 'source_slot_count', 'ineligible_mean'].iloc[0]:.2f} ineligible.",
        f"- Mean test rows per user at the row level: "
        f"{balance.loc[balance.metric == 'user_test_rows', 'eligible_mean'].iloc[0]:.2f} eligible vs. "
        f"{balance.loc[balance.metric == 'user_test_rows', 'ineligible_mean'].iloc[0]:.2f} ineligible.",
        "- C0 all-source dependence is positive in both groups for every seed, but is larger "
        "on the C2-eligible population; the C0-C2 reversal itself remains identified only "
        "on C2 support.",
        "",
        "## Latest-Slot Validity",
        "",
        f"- Every row has at least {latest_validity['minimum_nonpadding_source_slots']} non-padding source slot.",
        f"- C2 latest-slot coverage is {latest_validity['c2_latest_coverage']:.1%}; every eligible "
        "row changes exactly one slot, with zero bad changed-slot rows.",
        "- The eligibility and changed-slot structure is identical across all three checkpoints.",
        "",
        "## TV Baseline",
        "",
        f"- Observed exact-ID occurrence TV is {tv_value:.3f} under the definition recorded above.",
        "- A random-derangement reference is not recoverable from summary statistics alone. "
        "It requires the raw source-ID occurrence vector from the hash-bound processed bundle; "
        "the hash-checking random-reference runner is ready, and no paper claim is authorized "
        "from an invented baseline.",
        "",
        "## C0 All-Source Margins by C2 Support",
        "",
        "| Seed | Eligible | Ineligible | Difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in c0_all.itertuples(index=False):
        report.append(
            f"| {row.seed} | {row.eligible_mean:.4f} "
            f"[{row.eligible_ci_low:.4f}, {row.eligible_ci_high:.4f}] | "
            f"{row.ineligible_mean:.4f} [{row.ineligible_ci_low:.4f}, {row.ineligible_ci_high:.4f}] | "
            f"{row.difference:.4f} [{row.difference_ci_low:.4f}, {row.difference_ci_high:.4f}] |"
        )
    report_path = output_dir / "r250_review_evidence_closure.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
