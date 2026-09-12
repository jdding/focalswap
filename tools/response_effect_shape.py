#!/usr/bin/env python3
"""Summarize fixed-row response and effect shape for R242 C0/C2 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20230601, 20230602, 20230603)
DEFAULT_R242 = ROOT / "reports" / "r242_semantic_control_ladder_20260712" / "remote_artifacts"
DEFAULT_R233 = ROOT / "reports" / "r233_source_scope_localization_20260711"
DEFAULT_R261 = ROOT / "reports" / "r261_rank_change_c2_donor_diagnostics_20260812" / "r261_rank_change_c2_donor_diagnostics.json"
DEFAULT_OUTPUT = ROOT / "reports" / "r264_response_effect_shape_20260912"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r242-dir", default=str(DEFAULT_R242))
    parser.add_argument("--r233-dir", default=str(DEFAULT_R233))
    parser.add_argument("--r261-integrity", default=str(DEFAULT_R261))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rr_to_rank(values: np.ndarray) -> np.ndarray:
    ranks = np.rint(2.0 / values) / 2.0
    if np.max(np.abs(values - 1.0 / ranks)) > 1e-12:
        raise RuntimeError("RR values are not on the registered half-tie rank grid")
    if np.any((ranks < 1.0) | (ranks > 100.0)):
        raise RuntimeError("rank outside fixed 100-candidate contract")
    return ranks


def higher_quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q, method="higher"))


def summarize_control(
    seed: int,
    control: str,
    users: np.ndarray,
    full_rr: np.ndarray,
    control_rr: np.ndarray,
) -> dict[str, Any]:
    full_rank = rr_to_rank(full_rr)
    control_rank = rr_to_rank(control_rr)
    shift = control_rank - full_rank
    absolute_shift = np.abs(shift)
    changed = absolute_shift > 0
    changed_shifts = absolute_shift[changed]
    changed_rr = np.abs(full_rr[changed] - control_rr[changed])
    delta_rr = full_rr - control_rr
    user_values, user_codes = np.unique(users, return_inverse=True)
    user_sums = np.bincount(user_codes, weights=delta_rr, minlength=len(user_values))
    user_counts = np.bincount(user_codes, minlength=len(user_values))
    user_means = user_sums / user_counts
    leave_one_user_out = (delta_rr.sum() - user_sums) / (len(delta_rr) - user_counts)
    point = float(delta_rr.mean())
    if changed_shifts.size == 0:
        raise RuntimeError(f"no changed rows for {seed} {control}")
    return {
        "model_seed": seed,
        "control": control,
        "rows": int(len(full_rr)),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "worse_given_changed_fraction": float(np.mean(shift[changed] > 0)),
        "signed_delta_rr": point,
        "mean_absolute_rank_shift_changed": float(np.mean(changed_shifts)),
        "p50_absolute_rank_shift_changed": higher_quantile(changed_shifts, 0.50),
        "p75_absolute_rank_shift_changed": higher_quantile(changed_shifts, 0.75),
        "p90_absolute_rank_shift_changed": higher_quantile(changed_shifts, 0.90),
        "p95_absolute_rank_shift_changed": higher_quantile(changed_shifts, 0.95),
        "p99_absolute_rank_shift_changed": higher_quantile(changed_shifts, 0.99),
        "changed_at_least_5_positions_fraction": float(np.mean(changed_shifts >= 5.0)),
        "changed_at_least_10_positions_fraction": float(np.mean(changed_shifts >= 10.0)),
        "p50_absolute_rr_shift_changed": higher_quantile(changed_rr, 0.50),
        "p90_absolute_rr_shift_changed": higher_quantile(changed_rr, 0.90),
        "p95_absolute_rr_shift_changed": higher_quantile(changed_rr, 0.95),
        "top5_hit_flip_fraction": float(np.mean((full_rank <= 5) != (control_rank <= 5))),
        "top10_hit_flip_fraction": float(np.mean((full_rank <= 10) != (control_rank <= 10))),
        "top20_hit_flip_fraction": float(np.mean((full_rank <= 20) != (control_rank <= 20))),
        "positive_user_mean_fraction": float(np.mean(user_means > 0)),
        "max_user_row_fraction": float(user_counts.max() / len(delta_rr)),
        "max_absolute_leave_one_user_out_change": float(np.max(np.abs(leave_one_user_out - point))),
    }


def main() -> None:
    args = parse_args()
    r242_dir = Path(args.r242_dir)
    r233_dir = Path(args.r233_dir)
    r261_path = Path(args.r261_integrity)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    r261 = json.loads(r261_path.read_text(encoding="utf-8"))
    expected_hashes = r261["source_sha256"]
    rows: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}

    for seed in SEEDS:
        r242_path = r242_dir / f"r242_unisar_seed_{seed}_control_scores.csv"
        r233_path = r233_dir / f"r233_unisar_seed_{seed}_scope_scores.csv"
        for key, path in ((f"r242_seed_{seed}", r242_path), (f"r233_seed_{seed}", r233_path)):
            current_hash = sha256_file(path)
            if current_hash != expected_hashes[key]:
                raise RuntimeError(f"source hash mismatch for {key}")
            source_hashes[key] = current_hash

        r242 = pd.read_csv(
            r242_path,
            usecols=["official_row_index", "user_id", "category_all_eligible", "category_all_rr", "full_rr"],
        )
        r233 = pd.read_csv(
            r233_path,
            usecols=["official_row_index", "full_rr", "neutral_all_rr"],
        )
        frame = r242.merge(r233, on="official_row_index", suffixes=("_r242", "_r233"), validate="one_to_one")
        if len(frame) != 157_192:
            raise RuntimeError(f"unexpected row count for seed {seed}")
        if not np.allclose(frame["full_rr_r242"], frame["full_rr_r233"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"Full RR mismatch between R233 and R242 for seed {seed}")
        eligible = frame["category_all_eligible"].to_numpy(bool)
        if int(eligible.sum()) != 66_139:
            raise RuntimeError(f"C2 common-support row count changed for seed {seed}")
        full_rr = frame.loc[eligible, "full_rr_r242"].to_numpy(float)
        users = frame.loc[eligible, "user_id"].to_numpy()
        rows.append(
            summarize_control(
                seed,
                "C0_neutral_mask",
                users,
                full_rr,
                frame.loc[eligible, "neutral_all_rr"].to_numpy(float),
            )
        )
        rows.append(
            summarize_control(
                seed,
                "C2_observed_support",
                users,
                full_rr,
                frame.loc[eligible, "category_all_rr"].to_numpy(float),
            )
        )

    result = pd.DataFrame(rows)
    csv_path = output_dir / "r264_response_effect_shape.csv"
    result.to_csv(csv_path, index=False)

    ranges: dict[str, dict[str, list[float]]] = {}
    range_columns = [
        "changed_fraction",
        "worse_given_changed_fraction",
        "signed_delta_rr",
        "p50_absolute_rank_shift_changed",
        "p90_absolute_rank_shift_changed",
        "p95_absolute_rank_shift_changed",
        "changed_at_least_5_positions_fraction",
        "changed_at_least_10_positions_fraction",
        "top5_hit_flip_fraction",
        "top10_hit_flip_fraction",
        "top20_hit_flip_fraction",
        "positive_user_mean_fraction",
        "max_absolute_leave_one_user_out_change",
    ]
    for control, frame in result.groupby("control", sort=False):
        ranges[control] = {
            column: [float(frame[column].min()), float(frame[column].max())]
            for column in range_columns
        }

    payload = {
        "schema": "focalswap.r264.response_effect_shape.v1",
        "status": "COMPLETE",
        "population": {
            "rows_per_checkpoint": 66_139,
            "candidate_count": 100,
            "model_seeds": list(SEEDS),
            "support": "C2 all-source common support",
        },
        "quantile_method": "higher",
        "ranges_across_checkpoints": ranges,
        "source_sha256": source_hashes,
        "outputs": {"csv": str(csv_path)},
        "interpretation": (
            "Response rate and effect shape are descriptive diagnostics on the registered common support. "
            "They explain the signed RR means but do not add a new promotion endpoint."
        ),
    }
    json_path = output_dir / "r264_response_effect_shape.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# R264 Response and Effect-Shape Analysis",
        "",
        "Status: `COMPLETE`",
        "",
        "All summaries use the 66,139-row C2 all-source common support. Quantiles use the higher observed value and are descriptive diagnostics, not new promotion endpoints.",
        "",
        "| Seed | Control | Rank changed | Worse among changed | Signed delta RR | Abs. rank shift p50 / p90 / p95 | Top-10 flip | Max LOO change |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.model_seed} | {row.control} | {row.changed_fraction:.1%} | "
            f"{row.worse_given_changed_fraction:.1%} | {row.signed_delta_rr:+.6f} | "
            f"{row.p50_absolute_rank_shift_changed:.1f} / {row.p90_absolute_rank_shift_changed:.1f} / "
            f"{row.p95_absolute_rank_shift_changed:.1f} | "
            f"{row.top10_hit_flip_fraction:.1%} | "
            f"{row.max_absolute_leave_one_user_out_change:.6f} |"
        )
    lines.extend([
        "",
        "The analysis reuses the exact R242/R233 row files and requires their hashes to match the completed R261 integrity record. C0 and C2 can therefore be compared as two response distributions on identical rows, candidates, targets, and checkpoints.",
    ])
    (output_dir / "R264_RESPONSE_EFFECT_SHAPE_20260912.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
