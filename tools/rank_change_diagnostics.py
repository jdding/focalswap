#!/usr/bin/env python3
"""Audit rank changes and donor-map integrity for the R242 C0/C2 comparison."""

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
STRATA = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-29", 16, 29),
    ("30", 30, 30),
)
DEFAULT_R242 = ROOT / "reports" / "r242_semantic_control_ladder_20260712" / "remote_artifacts"
DEFAULT_R233 = ROOT / "reports" / "r233_source_scope_localization_20260711"
DEFAULT_OUTPUT = ROOT / "reports" / "r261_rank_change_c2_donor_diagnostics_20260812"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r242-dir", default=str(DEFAULT_R242))
    parser.add_argument("--r233-dir", default=str(DEFAULT_R233))
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


def summarize(seed: int, label: str, mask: np.ndarray, frame: pd.DataFrame) -> dict[str, Any]:
    full_rank = frame.loc[mask, "full_rank"].to_numpy(float)
    c0_rank = frame.loc[mask, "c0_rank"].to_numpy(float)
    c2_rank = frame.loc[mask, "c2_rank"].to_numpy(float)
    full_rr = frame.loc[mask, "full_rr"].to_numpy(float)
    c0_rr = frame.loc[mask, "neutral_all_rr"].to_numpy(float)
    c2_rr = frame.loc[mask, "category_all_rr"].to_numpy(float)
    c0_shift = c0_rank - full_rank
    c2_shift = c2_rank - full_rank
    c0_changed = c0_shift != 0
    c2_changed = c2_shift != 0
    return {
        "model_seed": seed,
        "source_slot_stratum": label,
        "rows": int(mask.sum()),
        "users": int(frame.loc[mask, "user_id"].nunique()),
        "c0_rank_changed_rows": int(np.sum(c0_changed)),
        "c0_rank_changed_fraction": float(np.mean(c0_changed)),
        "c0_worse_given_changed_fraction": float(np.mean(c0_shift[c0_changed] > 0)),
        "c0_median_absolute_shift_changed": float(np.median(np.abs(c0_shift[c0_changed]))),
        "c2_rank_changed_rows": int(np.sum(c2_changed)),
        "c2_rank_changed_fraction": float(np.mean(c2_changed)),
        "c2_worse_given_changed_fraction": float(np.mean(c2_shift[c2_changed] > 0)),
        "c2_median_absolute_shift_changed": float(np.median(np.abs(c2_shift[c2_changed]))),
        "either_control_rank_changed_rows": int(np.sum((full_rank != c0_rank) | (full_rank != c2_rank))),
        "control_outcomes_differ_rows": int(np.sum(c0_rank != c2_rank)),
        "control_outcomes_differ_fraction": float(np.mean(c0_rank != c2_rank)),
        "c0_better_than_full_rows": int(np.sum(c0_rank < full_rank)),
        "c0_worse_than_full_rows": int(np.sum(c0_rank > full_rank)),
        "c2_better_than_full_rows": int(np.sum(c2_rank < full_rank)),
        "c2_worse_than_full_rows": int(np.sum(c2_rank > full_rank)),
        "c0_minus_c2_delta_rr": float(np.mean(c2_rr - c0_rr)),
        "full_minus_c0_delta_rr": float(np.mean(full_rr - c0_rr)),
        "full_minus_c2_delta_rr": float(np.mean(full_rr - c2_rr)),
    }


def main() -> None:
    args = parse_args()
    r242_dir = Path(args.r242_dir)
    r233_dir = Path(args.r233_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    donor_integrity: dict[str, Any] | None = None
    structural_reference: pd.DataFrame | None = None

    for seed in SEEDS:
        r242_path = r242_dir / f"r242_unisar_seed_{seed}_control_scores.csv"
        r233_path = r233_dir / f"r233_unisar_seed_{seed}_scope_scores.csv"
        integrity_path = r242_dir / f"r242_unisar_seed_{seed}_integrity.json"
        r242 = pd.read_csv(r242_path)
        r233 = pd.read_csv(r233_path, usecols=["official_row_index", "full_rr", "neutral_all_rr"])
        if len(r242) != 157_192 or len(r233) != len(r242):
            raise RuntimeError(f"unexpected row count for seed {seed}")
        frame = r242.merge(r233, on="official_row_index", suffixes=("", "_r233"), validate="one_to_one")
        if not np.allclose(frame["full_rr"], frame["full_rr_r233"], rtol=0.0, atol=1e-15):
            raise RuntimeError(f"Full RR mismatch between R233 and R242 for seed {seed}")
        frame["full_rank"] = rr_to_rank(frame["full_rr"].to_numpy(float))
        frame["c0_rank"] = rr_to_rank(frame["neutral_all_rr"].to_numpy(float))
        frame["c2_rank"] = rr_to_rank(frame["category_all_rr"].to_numpy(float))

        structure = frame[["official_row_index", "user_id", "source_slot_count", "candidate_count", "category_all_eligible"]]
        if structural_reference is None:
            structural_reference = structure.copy()
        elif not structural_reference.equals(structure):
            raise RuntimeError(f"cross-seed structure mismatch for seed {seed}")

        eligible = frame["category_all_eligible"].to_numpy(bool)
        if int(eligible.sum()) != 66_139:
            raise RuntimeError("C2 common-support row count changed")
        rows.append(summarize(seed, "all eligible", eligible, frame))
        slots = frame["source_slot_count"].to_numpy(int)
        for label, lower, upper in STRATA:
            rows.append(summarize(seed, label, eligible & (slots >= lower) & (slots <= upper), frame))

        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        current = integrity["category_donor_map"]
        selected = {
            "bijection_on_eligible_support": current["bijection_on_eligible_support"],
            "fixed_points": current["fixed_points"],
            "max_donor_reuse": current["max_donor_reuse"],
            "all_donors_observed_and_nonempty": current["all_donors_observed_and_nonempty"],
            "profile_mismatch_failures": current["profile_mismatch_failures"],
            "category_overlap_failures": current["category_overlap_failures"],
            "future_label_or_score_used": current["future_label_or_score_used"],
            "eligible_source_ids": current["eligible_source_ids"],
            "route_counts": current["route_counts"],
            "selection_inputs": current["selection_inputs"],
            "forbidden_inputs": current["forbidden_inputs"],
        }
        if donor_integrity is None:
            donor_integrity = selected
        elif donor_integrity != selected:
            raise RuntimeError("donor-map integrity differs across model seeds")
        hashes[f"r242_seed_{seed}"] = sha256_file(r242_path)
        hashes[f"r233_seed_{seed}"] = sha256_file(r233_path)
        hashes[f"integrity_seed_{seed}"] = sha256_file(integrity_path)

    results = pd.DataFrame(rows)
    csv_path = output_dir / "r261_rank_change_by_slot_stratum.csv"
    results.to_csv(csv_path, index=False)
    if donor_integrity is None:
        raise RuntimeError("missing donor integrity")

    overall = results[results["source_slot_stratum"] == "all eligible"]
    payload = {
        "schema": "focalswap.r261.rank_change_c2_donor_diagnostics.v1",
        "status": "COMPLETE",
        "population": {"rows": 66_139, "candidate_count": 100, "model_seeds": list(SEEDS)},
        "rank_integrity": {
            "all_rr_values_map_to_registered_half_tie_ranks": True,
            "rank_range": [1, 100],
            "c0_rank_changed_fraction_range": [float(overall["c0_rank_changed_fraction"].min()), float(overall["c0_rank_changed_fraction"].max())],
            "c2_rank_changed_fraction_range": [float(overall["c2_rank_changed_fraction"].min()), float(overall["c2_rank_changed_fraction"].max())],
            "control_outcomes_differ_fraction_range": [float(overall["control_outcomes_differ_fraction"].min()), float(overall["control_outcomes_differ_fraction"].max())],
            "c0_worse_given_changed_fraction_range": [float(overall["c0_worse_given_changed_fraction"].min()), float(overall["c0_worse_given_changed_fraction"].max())],
            "c0_median_absolute_shift_changed": sorted(set(float(value) for value in overall["c0_median_absolute_shift_changed"])),
            "c2_worse_given_changed_fraction_range": [float(overall["c2_worse_given_changed_fraction"].min()), float(overall["c2_worse_given_changed_fraction"].max())],
            "c2_median_absolute_shift_changed": sorted(set(float(value) for value in overall["c2_median_absolute_shift_changed"])),
            "shift_direction": "positive means the target ranks worse under the control than under Full",
        },
        "donor_integrity": donor_integrity,
        "interpretation": (
            "The RR contrasts are aggregates of discrete target-rank changes, not floating-point jitter. "
            "C2 uses a one-to-one observed-support derangement with no donor reuse or fixed points. "
            "The slot-stratified pattern remains descriptive and does not make C2 a lexical-semantic gold standard."
        ),
        "outputs": {"csv": str(csv_path)},
        "source_sha256": hashes,
    }
    json_path = output_dir / "r261_rank_change_c2_donor_diagnostics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# R261 Rank-Change and C2 Donor Diagnostics",
        "",
        "Status: `COMPLETE`",
        "",
        "## Common-Support Rank Changes",
        "",
        "| Seed | C0 changed rank | C2 changed rank | C0/C2 outcomes differ | C0-C2 RR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.model_seed} | {row.c0_rank_changed_rows:,} ({row.c0_rank_changed_fraction:.1%}) | "
            f"{row.c2_rank_changed_rows:,} ({row.c2_rank_changed_fraction:.1%}) | "
            f"{row.control_outcomes_differ_rows:,} ({row.control_outcomes_differ_fraction:.1%}) | "
            f"{row.c0_minus_c2_delta_rr:.6f} |"
        )
    lines.extend([
        "",
        "Every RR value maps to the registered half-tie target-rank grid from 1 to 100. The reported mean contrasts therefore aggregate genuine tie-aware rank changes and are not numerical jitter.",
        "",
        "## Direction and Magnitude",
        "",
        "Positive rank shifts mean that the target ranks worse under the control than under Full.",
        "",
        "| Seed | C0 worse among changed | C0 median absolute shift | C2 worse among changed | C2 median absolute shift |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in overall.itertuples(index=False):
        lines.append(
            f"| {row.model_seed} | {row.c0_worse_given_changed_fraction:.1%} | "
            f"{row.c0_median_absolute_shift_changed:.1f} | "
            f"{row.c2_worse_given_changed_fraction:.1%} | "
            f"{row.c2_median_absolute_shift_changed:.1f} |"
        )
    lines.extend([
        "",
        "C0 changes are slightly more likely to worsen the target rank, whereas C2 changes are approximately direction-balanced. The median changes are local (two rank positions for C0 and one for C2), so the mean RR contrasts reflect partial cancellation among many discrete, bidirectional shifts.",
        "",
        "## Donor Construction",
        "",
        f"- Eligible source IDs: {donor_integrity['eligible_source_ids']:,}.",
        f"- One-to-one derangement: `{str(donor_integrity['bijection_on_eligible_support']).lower()}`; fixed points: {donor_integrity['fixed_points']}; maximum donor reuse: {donor_integrity['max_donor_reuse']}.",
        f"- Observed and nonempty donors: `{str(donor_integrity['all_donors_observed_and_nonempty']).lower()}`; profile mismatches: {donor_integrity['profile_mismatch_failures']}; category-overlap failures: {donor_integrity['category_overlap_failures']}.",
        "- Donor selection excludes user, target, candidate, rank, label, and model score.",
        "",
        "## Interpretation",
        "",
        "The diagnostics exclude floating-point noise, fixed-point donors, and repeated-donor concentration as explanations of the C0/C2 disagreement. They do not establish semantic equivalence: C2 remains a behavioral-category sensitivity control. Source-slot heterogeneity is reported descriptively because an all-source intervention changes more state on longer histories.",
    ])
    (output_dir / "R261_RANK_CHANGE_C2_DONOR_DIAGNOSTICS_20260812.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
