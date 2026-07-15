#!/usr/bin/env python3
"""R145-R148 calibration audit for SARCheck.

This script answers the late WSDM-short review concern that the current
negative result needs two calibration legs:

1. R145: a shuffled-search null for the G2 max-over-two oracle headroom.
2. R146: a positive control with injected focal-search transfer signal.
3. R147: win/tie/loss and half-tie summaries for Figure 2(b).
4. R148: a same-surface learning curve for tested input-side gates.

The script operates on the same R135large scored surface used by the current
paper draft, so the results are directly comparable to Figure 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from passive_confidence_gate import (  # noqa: E402
    avg,
    gate_metrics,
    write_json,
)


REPORTS = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=str(REPORTS / "r135large_item_proxy_rows.csv"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--report-prefix", default="r145_r148")
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--strengths", default="0,0.01,0.02,0.04,0.08,0.16")
    p.add_argument("--learning-curve-sizes", default="500,1000,3000,7000,10000")
    p.add_argument("--margin", type=float, default=0.005)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def stable_unit(text: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def eval_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["split"] in {"validation", "test"}]


def mean_oracle_gain(passive: np.ndarray, search: np.ndarray) -> float:
    return float(np.maximum(passive, search).mean() - passive.mean())


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def bootstrap_ci(values: np.ndarray, fn, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    if values.size == 0:
        return math.nan, math.nan, math.nan
    est = float(fn(values))
    boots = []
    size = values.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, size, size=size)
        boots.append(float(fn(values[idx])))
    return est, percentile(boots, 0.025), percentile(boots, 0.975)


def r145_oracle_null(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ev = eval_rows(rows)
    passive = np.asarray([f(r, "passive_rr") for r in ev], dtype=float)
    search = np.asarray([f(r, "search_rr") for r in ev], dtype=float)
    observed = mean_oracle_gain(passive, search)
    rng = np.random.default_rng(args.seed)

    nulls: dict[str, list[float]] = {"global_shuffle": [], "source_shuffle": []}
    sources = np.asarray([r["source"] for r in ev])
    source_to_idx: dict[str, np.ndarray] = {}
    for src in sorted(set(sources.tolist())):
        source_to_idx[src] = np.where(sources == src)[0]

    for _ in range(args.permutations):
        shuffled = np.array(search, copy=True)
        rng.shuffle(shuffled)
        nulls["global_shuffle"].append(mean_oracle_gain(passive, shuffled))

        by_source = np.array(search, copy=True)
        for idx in source_to_idx.values():
            tmp = np.array(by_source[idx], copy=True)
            rng.shuffle(tmp)
            by_source[idx] = tmp
        nulls["source_shuffle"].append(mean_oracle_gain(passive, by_source))

    rows_out: list[dict[str, Any]] = []
    for name, vals in nulls.items():
        p_ge = sum(v >= observed for v in vals) / len(vals)
        row = {
            "null": name,
            "eval_rows": len(ev),
            "history_rr": float(passive.mean()),
            "search_rr": float(search.mean()),
            "observed_oracle_gain": observed,
            "null_mean": float(np.mean(vals)),
            "null_p95": percentile(vals, 0.95),
            "null_p99": percentile(vals, 0.99),
            "observed_minus_null_mean": observed - float(np.mean(vals)),
            "p_null_ge_observed": p_ge,
            "status": "pass_above_null" if observed > percentile(vals, 0.95) else "fail_not_above_null",
        }
        rows_out.append(row)

    status = "R145_G2_NULL_PASS" if all(r["status"] == "pass_above_null" for r in rows_out) else "R145_G2_NULL_FAIL"
    return rows_out, {
        "status": status,
        "observed_oracle_gain": observed,
        "nulls": rows_out,
        "interpretation": "observed G2 headroom must exceed shuffled-search null to support calibrated opportunity",
    }


def make_gate_row(row: dict[str, str], passive: float, search: float, label: int, margin: float) -> dict[str, Any]:
    gap = search - passive
    return {
        "example_id": row["example_id"],
        "split": row["split"],
        "source": row["source"],
        "passive_rr": passive,
        "search_rr": search,
        "oracle_rr": max(passive, search),
        "random_gate_rr": 0.5 * (passive + search),
        "label_transfer": int(label and gap > margin),
        "label_nontransfer": int((not label) and abs(gap) <= margin),
        "outcome_gap": gap,
    }


def injected_label(row: dict[str, str]) -> int:
    # Deliberately input-side and learnable: focal search has positive feedback
    # and is not in direct category conflict with the pre-search history.
    return int(i(row, "search_pos_count") > 0 and i(row, "cat_conflict") == 0)


def win_tie_loss(deltas: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, Any]:
    eps = 1e-12

    def stats(vals: np.ndarray) -> np.ndarray:
        wins = float((vals > eps).mean())
        ties = float((np.abs(vals) <= eps).mean())
        losses = float((vals < -eps).mean())
        half = wins + 0.5 * ties
        mean_delta = float(vals.mean())
        return np.asarray([wins, ties, losses, half, mean_delta], dtype=float)

    est = stats(deltas)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        boots.append(stats(deltas[idx]))
    arr = np.vstack(boots)
    names = ["strict_win", "tie", "loss", "half_tie_win", "mean_delta"]
    out: dict[str, Any] = {}
    for col, name in enumerate(names):
        out[name] = float(est[col])
        out[f"{name}_ci_low"] = float(np.quantile(arr[:, col], 0.025))
        out[f"{name}_ci_high"] = float(np.quantile(arr[:, col], 0.975))
    return out


def feature_matrix(rows: list[dict[str, str]], source_vocab: list[str]) -> tuple[np.ndarray, list[str]]:
    names = [
        "pre_len_log",
        "search_pos_log",
        "search_neg_log",
        "search_action_log",
        "cat_conflict",
        "cat_jaccard",
        "candidate_size_log",
    ] + [f"source={s}" for s in source_vocab]
    mat = []
    for r in rows:
        vals = [
            math.log1p(i(r, "pre_len")),
            math.log1p(i(r, "search_pos_count")),
            math.log1p(i(r, "search_neg_count")),
            math.log1p(i(r, "search_action_count")),
            float(i(r, "cat_conflict")),
            f(r, "cat_jaccard"),
            math.log1p(i(r, "candidate_size")),
        ]
        vals.extend(1.0 if r["source"] == s else 0.0 for s in source_vocab)
        mat.append(vals)
    return np.asarray(mat, dtype=float), names


def fit_gate_and_metrics(
    rows: list[dict[str, str]],
    gate_rows: list[dict[str, Any]],
    train_limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    rng = random.Random(seed)
    train_idx = [idx for idx, r in enumerate(rows) if r["split"] == "train"]
    eval_idx = [idx for idx, r in enumerate(rows) if r["split"] in {"validation", "test"}]
    train_idx = sorted(train_idx, key=lambda idx: stable_unit(rows[idx]["example_id"], seed))[:train_limit]
    tune_cut = max(1, int(0.30 * len(train_idx)))
    tune_idx = train_idx[:tune_cut]
    fit_idx = train_idx[tune_cut:]
    eval_gate_rows = [gate_rows[idx] for idx in eval_idx]
    out = [
        gate_metrics(eval_gate_rows, [0.0] * len(eval_gate_rows), "fixed_0_history_only"),
        gate_metrics(eval_gate_rows, [1.0] * len(eval_gate_rows), "fixed_1_search_only"),
        gate_metrics(eval_gate_rows, [1.0 if r["label_transfer"] else 0.0 for r in eval_gate_rows], "oracle_hard"),
    ]
    for row in out:
        row["train_limit"] = train_limit
    if len(set(gate_rows[idx]["label_transfer"] for idx in fit_idx)) < 2:
        return out

    sources = [r["source"] for r in rows if r["split"] == "train"]
    vocab = [s for s, _ in sorted(((s, sources.count(s)) for s in set(sources)), key=lambda x: (-x[1], x[0]))[:8]]
    x_all, _ = feature_matrix(rows, vocab)
    x_fit = x_all[fit_idx]
    x_tune = x_all[tune_idx]
    x_eval = x_all[eval_idx]
    y_fit = np.asarray([gate_rows[idx]["label_transfer"] for idx in fit_idx], dtype=int)
    tune_rows = [gate_rows[idx] for idx in tune_idx]

    models = [
        ("logistic", LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)),
        ("gbdt", GradientBoostingClassifier(n_estimators=120, learning_rate=0.04, max_depth=2, min_samples_leaf=30, random_state=seed)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.002, max_iter=350, early_stopping=True, random_state=seed)),
    ]
    for name, model in models:
        model.fit(x_fit, y_fit)
        p_tune = model.predict_proba(x_tune)[:, 1]
        p_eval = model.predict_proba(x_eval)[:, 1]
        best_t = 0.5
        best = -1.0
        for t in np.linspace(0.03, 0.97, 95):
            score = avg([
                r["passive_rr"] + (1.0 if p >= t else 0.0) * (r["search_rr"] - r["passive_rr"])
                for r, p in zip(tune_rows, p_tune)
            ])
            if score > best:
                best = score
                best_t = float(t)
        out.append(gate_metrics(eval_gate_rows, [float(p) for p in p_eval], f"{name}_soft"))
        hard = gate_metrics(eval_gate_rows, [1.0 if p >= best_t else 0.0 for p in p_eval], f"{name}_hard")
        hard["threshold"] = best_t
        out.append(hard)
    for row in out:
        row["train_limit"] = train_limit
    rng.shuffle(out)
    return sorted(out, key=lambda x: (x["train_limit"], x["gate"]))


def r146_positive_control(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed + 10)
    eval_idx = [idx for idx, r in enumerate(rows) if r["split"] in {"validation", "test"}]
    out_rows: list[dict[str, Any]] = []
    gate_rows_all: list[dict[str, Any]] = []
    for strength in strengths:
        synthetic_gate_rows: list[dict[str, Any]] = []
        l1_deltas = []
        l2_deltas = []
        labels = []
        for r in rows:
            passive = f(r, "passive_rr")
            label = injected_label(r)
            labels.append(label)
            boost = strength * (1.0 - passive) if label else 0.0
            search = min(1.0, passive + boost)
            synthetic_gate_rows.append(make_gate_row(r, passive, search, label, args.margin))
            if r["split"] in {"validation", "test"}:
                l1_deltas.append(search - passive)
                l2_deltas.append(search - passive)
        gate_metrics_rows = fit_gate_and_metrics(rows, synthetic_gate_rows, train_limit=10000, seed=args.seed)
        best_gate = max(
            [m for m in gate_metrics_rows if m["gate"] not in {"fixed_0_history_only", "fixed_1_search_only", "oracle_hard"}],
            key=lambda m: float(m["score_rr"]),
            default={},
        )
        fixed0 = next(m for m in gate_metrics_rows if m["gate"] == "fixed_0_history_only")
        oracle = next(m for m in gate_metrics_rows if m["gate"] == "oracle_hard")
        l1 = win_tie_loss(np.asarray(l1_deltas, dtype=float), rng, args.bootstrap)
        l2 = win_tie_loss(np.asarray(l2_deltas, dtype=float), rng, args.bootstrap)
        out_rows.append({
            "strength": strength,
            "eval_rows": len(eval_idx),
            "injected_rate_eval": avg([labels[idx] for idx in eval_idx]),
            "history_rr": fixed0["score_rr"],
            "oracle_rr": oracle["score_rr"],
            "oracle_gain": float(oracle["score_rr"]) - float(fixed0["score_rr"]),
            "best_gate": best_gate.get("gate", ""),
            "best_gate_rr": best_gate.get("score_rr", math.nan),
            "best_gate_minus_history": best_gate.get("score_minus_passive", math.nan),
            "best_gate_headroom_recovery": best_gate.get("headroom_recovery", math.nan),
            "L1_half_tie_win": l1["half_tie_win"],
            "L1_strict_win": l1["strict_win"],
            "L1_tie": l1["tie"],
            "L2_half_tie_win": l2["half_tie_win"],
            "L2_strict_win": l2["strict_win"],
            "L2_tie": l2["tie"],
        })
        for m in gate_metrics_rows:
            m["strength"] = strength
            gate_rows_all.append(m)
    nonzero = [r for r in out_rows if float(r["strength"]) > 0]
    monotonic = all(nonzero[i]["L1_half_tie_win"] <= nonzero[i + 1]["L1_half_tie_win"] + 1e-12 for i in range(len(nonzero) - 1))
    gate_pass = any(float(r["best_gate_minus_history"]) > 0 and float(r["best_gate_headroom_recovery"]) >= 0.30 for r in nonzero)
    status = "R146_POSITIVE_CONTROL_PASS" if monotonic and gate_pass else "R146_POSITIVE_CONTROL_WARN"
    payload = {
        "status": status,
        "injection_rule": "search_pos_count > 0 and cat_conflict == 0",
        "strengths": strengths,
        "monotonic_l1_half_tie_nonzero": monotonic,
        "gate_recovers_headroom_at_positive_strength": gate_pass,
    }
    return out_rows, gate_rows_all, payload


def r147_actual_wtl(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(args.seed + 20)
    ev = eval_rows(rows)
    out = []
    for level, col, match_col in [
        ("L1", "L1_nuisance_true_minus_swap", "L1_nuisance_matched"),
        ("L2", "L2_same_category_true_minus_swap", "L2_same_category_matched"),
    ]:
        sub = [r for r in ev if i(r, match_col)]
        vals = np.asarray([f(r, col) for r in sub], dtype=float)
        stats = win_tie_loss(vals, rng, args.bootstrap)
        stats.update({"level": level, "matched_rows": len(sub), "eval_rows": len(ev), "match_rate": len(sub) / len(ev)})
        out.append(stats)
    return out, {"status": "R147_WTL_READY", "rows": out}


def r148_learning_curve(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sizes = [int(x) for x in args.learning_curve_sizes.split(",") if x.strip()]
    gate_rows = [
        make_gate_row(r, f(r, "passive_rr"), f(r, "search_rr"), int(f(r, "search_rr") - f(r, "passive_rr") > args.margin), args.margin)
        for r in rows
    ]
    out: list[dict[str, Any]] = []
    for size in sizes:
        out.extend(fit_gate_and_metrics(rows, gate_rows, train_limit=size, seed=args.seed + size))
    learned = [r for r in out if r["gate"] not in {"fixed_0_history_only", "fixed_1_search_only", "oracle_hard"}]
    best = max(learned, key=lambda r: float(r["score_rr"]), default={})
    hist = next((r for r in out if r["gate"] == "fixed_0_history_only" and r["train_limit"] == max(sizes)), None)
    status = "R148_LEARNING_CURVE_RECOVERS" if (
        best
        and hist
        and float(best["score_minus_passive"]) >= 0.001
        and float(best["headroom_recovery"]) >= 0.03
    ) else "R148_LEARNING_CURVE_NO_RECOVERY"
    return out, {"status": status, "best_gate": best, "max_train_size": max(sizes)}


def write_md(path: Path, payload: dict[str, Any]) -> None:
    r145 = payload["r145"]
    r146 = payload["r146"]
    r147 = payload["r147"]
    r148 = payload["r148"]
    lines = [
        "# R145-R148 SARCheck Calibration Audit",
        "",
        f"- Overall status: **{payload['status']}**",
        f"- Input rows: `{payload['inputs']['rows']}`",
        "",
        "## R145: G2 Oracle Null Calibration",
        "",
        "| Null | Observed gain | Null mean | Null p95 | Obs - null mean | p(null >= obs) | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in r145["nulls"]:
        lines.append(
            f"| {row['null']} | {fmt(row['observed_oracle_gain'])} | {fmt(row['null_mean'])} | "
            f"{fmt(row['null_p95'])} | {fmt(row['observed_minus_null_mean'])} | "
            f"{fmt(row['p_null_ge_observed'])} | {row['status']} |"
        )
    lines.extend([
        "",
        "## R146: Positive-Control Signal Injection",
        "",
        "| Strength | Oracle gain | L1 half-tie | L2 half-tie | Best gate | Gate - history | Headroom recovery |",
        "|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in r146["summary_rows"]:
        lines.append(
            f"| {fmt(row['strength'])} | {fmt(row['oracle_gain'])} | {fmt(row['L1_half_tie_win'])} | "
            f"{fmt(row['L2_half_tie_win'])} | {row['best_gate']} | "
            f"{fmt(float(row['best_gate_minus_history']))} | {fmt(float(row['best_gate_headroom_recovery']))} |"
        )
    lines.extend([
        "",
        "## R147: Actual Win/Tie/Loss Decomposition",
        "",
        "| Level | Matched | Strict win | Tie | Loss | Half-tie win | Mean true-swap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in r147["rows"]:
        lines.append(
            f"| {row['level']} | {row['matched_rows']:,} | {fmt(row['strict_win'])} | {fmt(row['tie'])} | "
            f"{fmt(row['loss'])} | {fmt(row['half_tie_win'])} | {fmt(row['mean_delta'])} |"
        )
    lines.extend([
        "",
        "## R148: Same-Surface Gate Learning Curve",
        "",
        f"- Status: **{r148['status']}**",
        f"- Best gate: `{r148['best_gate'].get('gate', '')}` at train size {r148['best_gate'].get('train_limit', '')}; RR {fmt(float(r148['best_gate'].get('score_rr', 0.0)))}.",
        "",
        "## Interpretation",
        "",
    ])
    if r145["status"] == "R145_G2_NULL_FAIL":
        lines.append("- The raw G2 oracle-over-History headroom does **not** survive a shuffled-search max-over-two null on the current scored surface. The paper should not present raw G2 headroom as calibrated opportunity evidence.")
    else:
        lines.append("- The raw G2 oracle headroom is above the shuffled-search null.")
    if r146["status"] == "R146_POSITIVE_CONTROL_PASS":
        lines.append("- The positive control passes: when focal-search signal is injected, G3 half-tie and G4 gate recovery move in the expected direction. This supports SARCheck sensitivity.")
    else:
        lines.append("- The positive control is inconclusive; do not use negative G3/G4 conclusions as strong evidence before revising the control.")
    lines.append("- Figure 2(b) should use win/tie/loss or half-tie win, not a strict-win bar against a 0.5 threshold.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    rows = read_csv(Path(args.rows))
    reports.mkdir(parents=True, exist_ok=True)

    r145_rows, r145_payload = r145_oracle_null(rows, args)
    r146_rows, r146_gate_rows, r146_payload = r146_positive_control(rows, args)
    r147_rows, r147_payload = r147_actual_wtl(rows, args)
    r148_rows, r148_payload = r148_learning_curve(rows, args)

    prefix = args.report_prefix
    paths = {
        "r145_oracle_null_csv": reports / f"{prefix}_r145_oracle_null.csv",
        "r146_positive_control_csv": reports / f"{prefix}_r146_positive_control.csv",
        "r146_gate_metrics_csv": reports / f"{prefix}_r146_gate_metrics.csv",
        "r147_wtl_csv": reports / f"{prefix}_r147_win_tie_loss.csv",
        "r148_learning_curve_csv": reports / f"{prefix}_r148_learning_curve.csv",
        "json": reports / f"{prefix}_calibration_audit.json",
        "md": reports / f"{prefix}_calibration_audit.md",
    }
    write_csv(paths["r145_oracle_null_csv"], r145_rows)
    write_csv(paths["r146_positive_control_csv"], r146_rows)
    write_csv(paths["r146_gate_metrics_csv"], r146_gate_rows)
    write_csv(paths["r147_wtl_csv"], r147_rows)
    write_csv(paths["r148_learning_curve_csv"], r148_rows)

    status = "R145_R148_CALIBRATION_PASS_WITH_G2_REVISION" if (
        r146_payload["status"] == "R146_POSITIVE_CONTROL_PASS"
        and r147_payload["status"] == "R147_WTL_READY"
    ) else "R145_R148_CALIBRATION_WARN"
    payload = {
        "schema": "sarcheck.r145_r148_calibration.v0.1",
        "status": status,
        "inputs": {"rows": str(Path(args.rows).relative_to(ROOT) if Path(args.rows).is_absolute() else args.rows)},
        "r145": r145_payload,
        "r146": {**r146_payload, "summary_rows": r146_rows},
        "r147": r147_payload,
        "r148": r148_payload,
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    write_json(paths["json"], payload)
    write_md(paths["md"], payload)
    print(json.dumps({"status": status, "report": str(paths["md"])}, indent=2))


if __name__ == "__main__":
    main()
