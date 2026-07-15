#!/usr/bin/env python3
"""R200G row-level bootstrap and power curves for FocalSwap.

This extends R200A's CI-derived approximation with row-level bootstrap
estimates. Qilin uses user-cluster bootstrap over the strong BM25 row table.
KuaiSAR uses row bootstrap over the R174 swap-ladder detail table because that
export does not carry user ids.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QILIN_ROWS = ROOT / "qilin_sarcheck/work/sar_qp3_qilin_dataset_audit_strong_bm25_rows.csv"
KUAI_ROWS = ROOT / "reports/r174_swap_ladder_detail.csv"
OUT_JSON = ROOT / "reports/r200g_row_level_power_bootstrap_20260620.json"
OUT_MD = ROOT / "reports/r200g_row_level_power_bootstrap_20260620.md"
OUT_CSV = ROOT / "reports/r200g_power_curve_20260620.csv"

SEED = 20260620
BOOT_REPS = 5000
Z_975 = 1.959963984540054
Z_80 = 0.8416212335729143
Z_90 = 1.2815515655446004
TARGET_EFFECTS = [0.005, 0.010, 0.020, 0.032, 0.050, 0.084]
N_GRID = [100, 200, 373, 500, 983, 2580, 6135, 10312, 16496]


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def positive_power(effect: float, se: float, z_alpha: float = Z_975) -> float:
    if se <= 0:
        return 1.0 if effect > 0 else 0.0
    return normal_cdf(effect / se - z_alpha)


def mde(se: float, z_power: float = Z_80) -> float:
    return (Z_975 + z_power) * se


def half_tie_from_diff(diff: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(diff, dtype=float)
    return (arr > 0).astype(float) + 0.5 * (arr == 0).astype(float)


def summarize_boot(name: str, n: int, coverage: float, metric: str, values: np.ndarray, boots: np.ndarray) -> dict:
    estimate = float(np.mean(values))
    se = float(np.std(boots, ddof=1))
    ci_low, ci_high = np.quantile(boots, [0.025, 0.975])
    return {
        "surface_bucket": name,
        "n": int(n),
        "coverage": float(coverage),
        "metric": metric,
        "estimate": estimate,
        "bootstrap_se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mde80": float(mde(se, Z_80)),
        "mde90": float(mde(se, Z_90)),
        "power_at_0.010": float(positive_power(0.010, se)),
        "power_at_0.084": float(positive_power(0.084, se)),
    }


def row_bootstrap(values: np.ndarray, rng: np.random.Generator, reps: int = BOOT_REPS) -> np.ndarray:
    n = len(values)
    idx = rng.integers(0, n, size=(reps, n))
    return values[idx].mean(axis=1)


def cluster_bootstrap(df: pd.DataFrame, value_col: str, rng: np.random.Generator, reps: int = BOOT_REPS) -> np.ndarray:
    users = np.array(sorted(df["user_idx"].unique()))
    grouped = {u: df.loc[df["user_idx"] == u, value_col].to_numpy(dtype=float) for u in users}
    out = np.empty(reps, dtype=float)
    for i in range(reps):
        sampled = rng.choice(users, size=len(users), replace=True)
        total = 0.0
        count = 0
        for u in sampled:
            vals = grouped[u]
            total += float(vals.sum())
            count += int(len(vals))
        out[i] = total / count if count else float("nan")
    return out[~np.isnan(out)]


def qilin_bucket_specs() -> list[dict]:
    return [
        {"bucket": "same_user/all", "label": "Qilin same-user all", "alt": "same_user_rr", "mask": lambda d: d["same_user_swap_found"].astype(bool)},
        {"bucket": "time_near/all", "label": "Qilin time-near all", "alt": "time_near_rr", "mask": lambda d: pd.Series(True, index=d.index)},
        {"bucket": "time_near/<=300s", "label": "Qilin time <= 300s", "alt": "time_near_rr", "mask": lambda d: d["time_near_abs_delta_seconds"] <= 300},
        {"bucket": "time_near/<=120s", "label": "Qilin time <= 120s", "alt": "time_near_rr", "mask": lambda d: d["time_near_abs_delta_seconds"] <= 120},
        {"bucket": "time_near/<=60s", "label": "Qilin time <= 60s", "alt": "time_near_rr", "mask": lambda d: d["time_near_abs_delta_seconds"] <= 60},
        {"bucket": "time_near/<=30s", "label": "Qilin time <= 30s", "alt": "time_near_rr", "mask": lambda d: d["time_near_abs_delta_seconds"] <= 30},
        {"bucket": "tax_near/share>=1", "label": "Qilin tax overlap >= 1", "alt": "tax_near_rr", "mask": lambda d: d["tax_near_shared_tax_count"] >= 1},
        {"bucket": "tax_near/share>=2", "label": "Qilin tax overlap >= 2", "alt": "tax_near_rr", "mask": lambda d: d["tax_near_shared_tax_count"] >= 2},
        {"bucket": "tax_near/share>=3", "label": "Qilin tax overlap >= 3", "alt": "tax_near_rr", "mask": lambda d: d["tax_near_shared_tax_count"] >= 3},
        {"bucket": "tax_near/share>=4", "label": "Qilin tax overlap >= 4", "alt": "tax_near_rr", "mask": lambda d: d["tax_near_shared_tax_count"] >= 4},
    ]


def compute_qilin(rng: np.random.Generator) -> list[dict]:
    df = pd.read_csv(QILIN_ROWS)
    total = len(df)
    summaries: list[dict] = []
    for spec in qilin_bucket_specs():
        sub = df.loc[spec["mask"](df)].copy()
        matched = half_tie_from_diff(sub["focal_rr"] - sub[spec["alt"]])
        placebo = half_tie_from_diff(sub["focal_rr"] - sub["random_rr"])
        sub["matched_minus_placebo"] = matched - placebo
        sub["matched_half_tie"] = matched
        summaries.append(
            summarize_boot(
                spec["label"],
                len(sub),
                len(sub) / total,
                "matched_minus_placebo_delta",
                sub["matched_minus_placebo"].to_numpy(dtype=float),
                cluster_bootstrap(sub, "matched_minus_placebo", rng),
            )
        )
        summaries.append(
            summarize_boot(
                spec["label"],
                len(sub),
                len(sub) / total,
                "matched_half_tie_lift_vs_0.5",
                sub["matched_half_tie"].to_numpy(dtype=float) - 0.5,
                cluster_bootstrap(sub, "matched_half_tie", rng) - 0.5,
            )
        )
    return summaries


def compute_kuaisar(rng: np.random.Generator) -> list[dict]:
    df = pd.read_csv(KUAI_ROWS)
    total = df["example_id"].nunique()
    summaries: list[dict] = []
    for level in ["source_broken", "nuisance_broken", "same_category_source"]:
        sub = df.loc[df["level"] == level].copy()
        ht = half_tie_from_diff(sub["true_search_rr"] - sub["swap_rr"])
        sub["half_tie_lift"] = ht - 0.5
        sub["mean_delta"] = sub["true_search_rr"] - sub["swap_rr"]
        label = f"KuaiSAR {level.replace('_', '-')}"
        summaries.append(
            summarize_boot(
                label,
                len(sub),
                len(sub) / total,
                "matched_half_tie_lift_vs_0.5",
                sub["half_tie_lift"].to_numpy(dtype=float),
                row_bootstrap(sub["half_tie_lift"].to_numpy(dtype=float), rng),
            )
        )
        summaries.append(
            summarize_boot(
                label,
                len(sub),
                len(sub) / total,
                "mean_true_minus_swap_rr",
                sub["mean_delta"].to_numpy(dtype=float),
                row_bootstrap(sub["mean_delta"].to_numpy(dtype=float), rng),
            )
        )
    return summaries


def make_power_curve(summary_rows: list[dict]) -> pd.DataFrame:
    anchors = {
        "Qilin broad tax>=1 delta": next(r for r in summary_rows if r["surface_bucket"] == "Qilin tax overlap >= 1" and r["metric"] == "matched_minus_placebo_delta"),
        "Qilin local tax>=4 delta": next(r for r in summary_rows if r["surface_bucket"] == "Qilin tax overlap >= 4" and r["metric"] == "matched_minus_placebo_delta"),
        "KuaiSAR same-category/source half-tie lift": next(r for r in summary_rows if r["surface_bucket"] == "KuaiSAR same-category-source" and r["metric"] == "matched_half_tie_lift_vs_0.5"),
    }
    records = []
    for name, row in anchors.items():
        base_n = row["n"]
        base_se = row["bootstrap_se"]
        for n in N_GRID:
            se_n = base_se * math.sqrt(base_n / n)
            for effect in TARGET_EFFECTS:
                records.append({
                    "anchor": name,
                    "base_n": base_n,
                    "n": n,
                    "target_effect": effect,
                    "scaled_se": se_n,
                    "mde80": mde(se_n, Z_80),
                    "mde90": mde(se_n, Z_90),
                    "power_alpha05_two_sided_positive": positive_power(effect, se_n),
                })
    return pd.DataFrame.from_records(records)


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    rng = np.random.default_rng(SEED)
    summaries = compute_qilin(rng) + compute_kuaisar(rng)
    curve = make_power_curve(summaries)
    curve.to_csv(OUT_CSV, index=False)

    payload = {
        "status": "R200G_ROW_LEVEL_BOOTSTRAP_POWER_DONE",
        "seed": SEED,
        "bootstrap_reps": BOOT_REPS,
        "inputs": [str(QILIN_ROWS.relative_to(ROOT)), str(KUAI_ROWS.relative_to(ROOT))],
        "summary_rows": summaries,
        "power_curve_csv": str(OUT_CSV.relative_to(ROOT)),
        "interpretation": {
            "broad_qilin_tax1_delta_mde80": next(r for r in summaries if r["surface_bucket"] == "Qilin tax overlap >= 1" and r["metric"] == "matched_minus_placebo_delta")["mde80"],
            "broad_qilin_power_for_plus_0010": next(r for r in summaries if r["surface_bucket"] == "Qilin tax overlap >= 1" and r["metric"] == "matched_minus_placebo_delta")["power_at_0.010"],
            "broad_qilin_power_for_plus_0084": next(r for r in summaries if r["surface_bucket"] == "Qilin tax overlap >= 1" and r["metric"] == "matched_minus_placebo_delta")["power_at_0.084"],
            "claim_consequence": "Row-level bootstrap supports the R200A branch: broad Qilin has high power for a tax>=4-scale effect but low power for +0.010 weak broad effects.",
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    key_rows = [
        r for r in summaries
        if (r["surface_bucket"], r["metric"]) in {
            ("Qilin tax overlap >= 1", "matched_minus_placebo_delta"),
            ("Qilin tax overlap >= 3", "matched_minus_placebo_delta"),
            ("Qilin tax overlap >= 4", "matched_minus_placebo_delta"),
            ("KuaiSAR source-broken", "matched_half_tie_lift_vs_0.5"),
            ("KuaiSAR same-category-source", "matched_half_tie_lift_vs_0.5"),
        }
    ]
    lines = [
        "# R200G Row-Level Bootstrap Power Curve",
        "",
        "Date: 2026-06-20",
        "",
        "Status: `R200G_ROW_LEVEL_BOOTSTRAP_POWER_DONE`",
        "",
        "This report extends R200A's CI-derived MDE approximation with row-level",
        "bootstrap. Qilin uses user-cluster bootstrap on the strong BM25 row table;",
        "KuaiSAR uses row bootstrap on the R174 swap-ladder detail export because",
        "that file does not contain user identifiers.",
        "",
        "## Key Bootstrap Rows",
        "",
        "| Surface / bucket | Metric | N | Cov. | Estimate | 95% boot CI | MDE80 | Power +0.010 | Power +0.084 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in key_rows:
        lines.append(
            f"| {row['surface_bucket']} | {row['metric']} | {row['n']} | "
            f"{fmt(row['coverage'])} | {row['estimate']:+.3f} | "
            f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}] | "
            f"{row['mde80']:.3f} | {row['power_at_0.010']:.2f} | {row['power_at_0.084']:.2f} |"
        )

    broad = next(r for r in summaries if r["surface_bucket"] == "Qilin tax overlap >= 1" and r["metric"] == "matched_minus_placebo_delta")
    tax4 = next(r for r in summaries if r["surface_bucket"] == "Qilin tax overlap >= 4" and r["metric"] == "matched_minus_placebo_delta")
    weak = next(r for r in summaries if r["surface_bucket"] == "KuaiSAR same-category-source" and r["metric"] == "matched_half_tie_lift_vs_0.5")
    curve_broad = curve[(curve["anchor"] == "Qilin broad tax>=1 delta") & (curve["target_effect"].isin([0.010, 0.084])) & (curve["n"].isin([373, 983, 2580, 6135, 10312, 16496]))]
    lines.extend([
        "",
        "## Broad-Qilin Power Curve",
        "",
        "| N | Target effect | MDE80 | Power |",
        "|---:|---:|---:|---:|",
    ])
    for _, row in curve_broad.iterrows():
        lines.append(
            f"| {int(row['n'])} | {row['target_effect']:+.3f} | "
            f"{row['mde80']:.3f} | {row['power_alpha05_two_sided_positive']:.2f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Broad Qilin tax>=1 delta is {broad['estimate']:+.3f} with row-level MDE80 {broad['mde80']:.3f}.",
        f"- At broad Qilin N, power is {broad['power_at_0.010']:.2f} for a +0.010 effect and {broad['power_at_0.084']:.2f} for a +0.084 effect.",
        f"- The local Qilin tax>=4 pocket remains positive ({tax4['estimate']:+.3f}) but low coverage ({tax4['coverage']:.3f}).",
        f"- The KuaiSAR same-category/source half-tie lift remains weak positive ({weak['estimate']:+.3f}).",
        "- The reviewer-facing conclusion should remain sensitivity-aware: the audit can reject a large wide-coverage tax>=4-scale lexical effect, but it cannot reject weak broad focal-transfer effects of about +0.010.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{OUT_JSON.relative_to(ROOT)}`",
        f"- Power curve CSV: `{OUT_CSV.relative_to(ROOT)}`",
        "",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
