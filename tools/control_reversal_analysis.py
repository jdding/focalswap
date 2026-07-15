#!/usr/bin/env python3
"""Secondary paired control-reversal contrasts for completed R242 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from control_ladder_analysis import (
    EXPECTED_SEEDS,
    SCOPE_COLUMNS,
    cluster_bootstrap,
    interval_status,
    load_seed,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r242-dir", required=True)
    parser.add_argument("--r233-dir", required=True)
    parser.add_argument("--reference-manifest", required=True)
    parser.add_argument("--primary-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    r242_dir = Path(args.r242_dir)
    r233_dir = Path(args.r233_dir)
    reference_path = Path(args.reference_manifest)
    reference = json.loads(reference_path.read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in EXPECTED_SEEDS:
        frame, _, _ = load_seed(r242_dir, r233_dir, seed, reference)
        for scope_position, (scope, columns) in enumerate(SCOPE_COLUMNS.items()):
            common = frame[columns["eligible"]].astype(bool).to_numpy()
            users = frame.loc[common, "user_id"].to_numpy()
            c0 = frame.loc[common, columns["C0_neutral"]].to_numpy()
            c1 = frame.loc[common, columns["C1_profile_prioritized_valid"]].to_numpy()
            c2 = frame.loc[common, columns["C2_category_valid"]].to_numpy()
            for contrast_position, (name, values, ci_level) in enumerate(
                (
                    ("C0_minus_C1", c1 - c0, 0.95),
                    ("C0_minus_C2", c2 - c0, 0.975),
                )
            ):
                alpha = 1.0 - ci_level
                point, low, high, user_count = cluster_bootstrap(
                    values,
                    users,
                    args.bootstrap_iters,
                    args.bootstrap_seed + seed + 100 * scope_position + contrast_position,
                    alpha=alpha,
                )
                rows.append(
                    {
                        "seed": seed,
                        "scope": scope,
                        "contrast": name,
                        "rows": int(common.sum()),
                        "user_clusters": user_count,
                        "paired_delta_rr": point,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_level": ci_level,
                        "status": interval_status(low, high),
                        "interpretation": "positive means neutral masking yields a larger Full-minus-control margin",
                    }
                )
    contrasts = pd.DataFrame(rows)
    contrast_csv = output_dir / "r242_control_reversal_contrasts.csv"
    contrast_json = output_dir / "r242_control_reversal_contrasts.json"
    contrasts.to_csv(contrast_csv, index=False)
    primary = pd.read_csv(args.primary_summary)
    bounds = primary.loc[
        primary["promotion_endpoint"].astype(str).str.lower().eq("true"),
        [
            "seed",
            "scope",
            "rows",
            "coverage",
            "eligible_user_coverage",
            "delta_rr",
            "ci_low",
            "ci_high",
            "ci_level",
            "status",
        ],
    ].copy()
    bounds = bounds.rename(columns={"ci_high": "adjusted_positive_effect_upper_bound_rr"})
    bounds_path = output_dir / "r242_primary_positive_effect_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    payload = {
        "schema": "focalswap.r242.control_reversal.v1",
        "contrast_rows": rows,
        "primary_positive_effect_bounds": bounds.to_dict(orient="records"),
        "source_hashes": {
            "primary_summary": sha256_file(Path(args.primary_summary)),
            "reference_manifest": sha256_file(reference_path),
        },
        "claim_ceiling": (
            "Secondary paired mechanism contrast only; it supports control sensitivity, "
            "not beneficial source use or architecture attribution."
        ),
    }
    contrast_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "contrast_rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
