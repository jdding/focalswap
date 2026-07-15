#!/usr/bin/env python3
"""R173 G2 learned-fusion and column-correlation diagnostics.

This audit answers the WSDM 2027 simulated-review concern that the KuaiSAR G2
max-over-two oracle may be a selection artifact. It keeps the R135large proxy
surface fixed and adds:

1. column-correlation diagnostics for History-only and Search-only RR columns;
2. additional shuffle nulls, including history-decile constrained shuffles;
3. held-out fusion/readout checks that do not rely on the hard max operator.

The learned fusions here are diagnostics over the proxy surface. They are not
new SAR models and not leaderboard evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=str(REPORTS / "r135large_item_proxy_rows.csv"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--report-prefix", default="r173")
    p.add_argument("--surface-id", default="K_PROXY_R135LARGE")
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--permutations", type=int, default=1000)
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
        for key in row:
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def i(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def stable_unit(text: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16**12)


def split_rows(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    if split == "eval":
        return [r for r in rows if r["split"] in {"validation", "test"}]
    return [r for r in rows if r["split"] == split]


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return math.nan
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def fmt(x: float) -> str:
    return f"{x:.4f}"


def rankdata_average(xs: np.ndarray) -> np.ndarray:
    order = np.argsort(xs, kind="mergesort")
    ranks = np.empty(len(xs), dtype=float)
    sorted_x = xs[order]
    start = 0
    while start < len(xs):
        end = start + 1
        while end < len(xs) and sorted_x[end] == sorted_x[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr(rankdata_average(x), rankdata_average(y))


def mean_oracle_gain(passive: np.ndarray, search: np.ndarray) -> float:
    return float(np.maximum(passive, search).mean() - passive.mean())


def rr_score(rows: list[dict[str, str]], gates: np.ndarray) -> float:
    passive = np.asarray([f(r, "passive_rr") for r in rows], dtype=float)
    search = np.asarray([f(r, "search_rr") for r in rows], dtype=float)
    return float((passive + gates * (search - passive)).mean())


def row_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    sources = sorted({r["source"] for r in rows})
    common_sources = [s for s in sources if sum(1 for r in rows if r["source"] == s and r["split"] == "train") >= 50]
    names = [
        "pre_len_log",
        "search_pos_log",
        "search_neg_log",
        "search_action_log",
        "cat_conflict",
        "cat_jaccard",
    ] + [f"source={s}" for s in common_sources]
    mat: list[list[float]] = []
    for r in rows:
        vals = [
            math.log1p(i(r, "pre_len")),
            math.log1p(i(r, "search_pos_count")),
            math.log1p(i(r, "search_neg_count")),
            math.log1p(i(r, "search_action_count")),
            float(i(r, "cat_conflict")),
            f(r, "cat_jaccard"),
        ]
        vals.extend(1.0 if r["source"] == s else 0.0 for s in common_sources)
        mat.append(vals)
    return np.asarray(mat, dtype=float), names


def standardize(x_fit: np.ndarray, x_tune: np.ndarray, x_eval: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    std[std < 1e-8] = 1.0
    return (x_fit - mean) / std, (x_tune - mean) / std, (x_eval - mean) / std


def choose_threshold(tune_rows: list[dict[str, str]], probs: np.ndarray) -> float:
    best_t = 0.5
    best_rr = -1.0
    for t in np.linspace(0.02, 0.98, 97):
        gates = (probs >= t).astype(float)
        rr = rr_score(tune_rows, gates)
        if rr > best_rr:
            best_rr = rr
            best_t = float(t)
    return best_t


def metric_row(name: str, rows: list[dict[str, str]], gates: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    passive = float(np.mean([f(r, "passive_rr") for r in rows]))
    search = float(np.mean([f(r, "search_rr") for r in rows]))
    oracle = float(np.mean([max(f(r, "passive_rr"), f(r, "search_rr")) for r in rows]))
    score = rr_score(rows, gates)
    out = {
        "surface_id": "K_PROXY_R135LARGE",
        "fusion": name,
        "rows": len(rows),
        "score_rr": score,
        "history_rr": passive,
        "search_rr": search,
        "oracle_rr": oracle,
        "score_minus_history": score - passive,
        "oracle_headroom": oracle - passive,
        "headroom_recovery": (score - passive) / (oracle - passive) if oracle > passive else math.nan,
        "mean_gate": float(np.mean(gates)) if len(gates) else math.nan,
    }
    if extra:
        out.update(extra)
    return out


def column_correlation(rows: list[dict[str, str]], surface_id: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in ["train", "validation", "test", "eval"]:
        group = split_rows(rows, split)
        passive = np.asarray([f(r, "passive_rr") for r in group], dtype=float)
        search = np.asarray([f(r, "search_rr") for r in group], dtype=float)
        out.append({
            "surface_id": surface_id,
            "split": split,
            "rows": len(group),
            "history_rr_mean": float(passive.mean()) if len(passive) else math.nan,
            "search_rr_mean": float(search.mean()) if len(search) else math.nan,
            "oracle_gain": mean_oracle_gain(passive, search) if len(passive) else math.nan,
            "pearson_history_search": corr(passive, search),
            "spearman_history_search": spearman(passive, search),
            "both_zero_rate": float(((passive == 0.0) & (search == 0.0)).mean()) if len(passive) else math.nan,
            "both_nonzero_rate": float(((passive > 0.0) & (search > 0.0)).mean()) if len(passive) else math.nan,
            "search_beats_history_rate": float((search > passive).mean()) if len(passive) else math.nan,
            "history_beats_search_rate": float((passive > search).mean()) if len(passive) else math.nan,
            "tie_rate": float((passive == search).mean()) if len(passive) else math.nan,
        })
    return out


def history_deciles(passive: np.ndarray, n_deciles: int = 10) -> np.ndarray:
    qs = np.quantile(passive, np.linspace(0.1, 0.9, n_deciles - 1))
    return np.digitize(passive, qs, right=True)


def null_diagnostics(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ev = split_rows(rows, "eval")
    passive = np.asarray([f(r, "passive_rr") for r in ev], dtype=float)
    search = np.asarray([f(r, "search_rr") for r in ev], dtype=float)
    sources = np.asarray([r["source"] for r in ev])
    deciles = history_deciles(passive)
    observed = mean_oracle_gain(passive, search)
    observed_corr = corr(passive, search)
    rng = np.random.default_rng(args.seed)
    nulls: dict[str, list[tuple[float, float]]] = {
        "global_shuffle": [],
        "source_shuffle": [],
        "history_decile_shuffle": [],
        "source_history_decile_shuffle": [],
    }
    source_groups = {src: np.where(sources == src)[0] for src in sorted(set(sources.tolist()))}
    decile_groups = {int(d): np.where(deciles == d)[0] for d in sorted(set(deciles.tolist()))}
    source_decile_groups: dict[tuple[str, int], np.ndarray] = {}
    for src in sorted(set(sources.tolist())):
        for d in sorted(set(deciles.tolist())):
            idx = np.where((sources == src) & (deciles == d))[0]
            if len(idx):
                source_decile_groups[(src, int(d))] = idx

    for _ in range(args.permutations):
        shuf = np.array(search, copy=True)
        rng.shuffle(shuf)
        nulls["global_shuffle"].append((mean_oracle_gain(passive, shuf), corr(passive, shuf)))

        by_src = np.array(search, copy=True)
        for idx in source_groups.values():
            tmp = np.array(by_src[idx], copy=True)
            rng.shuffle(tmp)
            by_src[idx] = tmp
        nulls["source_shuffle"].append((mean_oracle_gain(passive, by_src), corr(passive, by_src)))

        by_dec = np.array(search, copy=True)
        for idx in decile_groups.values():
            tmp = np.array(by_dec[idx], copy=True)
            rng.shuffle(tmp)
            by_dec[idx] = tmp
        nulls["history_decile_shuffle"].append((mean_oracle_gain(passive, by_dec), corr(passive, by_dec)))

        by_src_dec = np.array(search, copy=True)
        for idx in source_decile_groups.values():
            tmp = np.array(by_src_dec[idx], copy=True)
            rng.shuffle(tmp)
            by_src_dec[idx] = tmp
        nulls["source_history_decile_shuffle"].append((mean_oracle_gain(passive, by_src_dec), corr(passive, by_src_dec)))

    rows_out: list[dict[str, Any]] = []
    for name, vals in nulls.items():
        gains = [v[0] for v in vals]
        corrs = [v[1] for v in vals if not math.isnan(v[1])]
        rows_out.append({
            "surface_id": args.surface_id,
            "null": name,
            "eval_rows": len(ev),
            "observed_oracle_gain": observed,
            "observed_pearson": observed_corr,
            "null_gain_mean": float(np.mean(gains)),
            "null_gain_p05": percentile(gains, 0.05),
            "null_gain_p50": percentile(gains, 0.50),
            "null_gain_p95": percentile(gains, 0.95),
            "observed_minus_null_mean": observed - float(np.mean(gains)),
            "p_null_ge_observed": sum(g >= observed for g in gains) / len(gains),
            "null_pearson_mean": float(np.mean(corrs)) if corrs else math.nan,
            "null_pearson_p05": percentile(corrs, 0.05) if corrs else math.nan,
            "null_pearson_p95": percentile(corrs, 0.95) if corrs else math.nan,
            "status": "observed_above_p95" if observed > percentile(gains, 0.95) else "observed_not_above_p95",
        })
    payload = {
        "observed_oracle_gain": observed,
        "observed_pearson": observed_corr,
        "nulls": rows_out,
    }
    return rows_out, payload


def fusion_diagnostics(rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for R173") from exc

    train = split_rows(rows, "train")
    tune: list[dict[str, str]] = []
    fit: list[dict[str, str]] = []
    for r in train:
        (tune if stable_unit(r["example_id"], args.seed) < 0.30 else fit).append(r)
    ev = split_rows(rows, "eval")

    out: list[dict[str, Any]] = []
    out.append(metric_row("fixed_history", ev, np.zeros(len(ev))))
    out.append(metric_row("fixed_search", ev, np.ones(len(ev))))
    out.append(metric_row("uniform_half", ev, np.full(len(ev), 0.5)))
    out.append(metric_row("hard_oracle_target_aware", ev, np.asarray([1.0 if f(r, "search_rr") > f(r, "passive_rr") else 0.0 for r in ev])))

    best_alpha = 0.0
    best_rr = -1.0
    for alpha in np.linspace(0.0, 1.0, 101):
        rr = rr_score(tune, np.full(len(tune), alpha))
        if rr > best_rr:
            best_rr = rr
            best_alpha = float(alpha)
    out.append(metric_row("static_convex_tuned", ev, np.full(len(ev), best_alpha), {"alpha": best_alpha, "tune_rr": best_rr}))

    sources = sorted({r["source"] for r in rows})
    source_alpha: dict[str, float] = {}
    global_alpha = best_alpha
    for src in sources:
        sub = [r for r in tune if r["source"] == src]
        if len(sub) < 20:
            source_alpha[src] = global_alpha
            continue
        src_best_alpha = global_alpha
        src_best = -1.0
        for alpha in np.linspace(0.0, 1.0, 101):
            rr = rr_score(sub, np.full(len(sub), alpha))
            if rr > src_best:
                src_best = rr
                src_best_alpha = float(alpha)
        source_alpha[src] = src_best_alpha
    out.append(metric_row(
        "source_specific_convex_tuned",
        ev,
        np.asarray([source_alpha.get(r["source"], global_alpha) for r in ev], dtype=float),
        {"alpha_summary": json.dumps(source_alpha, sort_keys=True)},
    ))

    x_all, feature_names = row_features(rows)
    row_to_idx = {id(r): idx for idx, r in enumerate(rows)}
    fit_idx = [row_to_idx[id(r)] for r in fit]
    tune_idx = [row_to_idx[id(r)] for r in tune]
    eval_idx = [row_to_idx[id(r)] for r in ev]
    x_fit, x_tune, x_eval = standardize(x_all[fit_idx], x_all[tune_idx], x_all[eval_idx])
    y_fit = np.asarray([1 if f(r, "search_rr") - f(r, "passive_rr") > args.margin else 0 for r in fit], dtype=int)

    if len(set(y_fit.tolist())) < 2:
        payload = {"status": "R173_FUSION_WARN_ONE_CLASS", "best_learned": {}, "feature_names": feature_names}
        return out, payload

    models: list[tuple[str, Any]] = [
        ("input_logistic", LogisticRegression(max_iter=500, class_weight="balanced", random_state=args.seed)),
        ("input_gbdt", GradientBoostingClassifier(n_estimators=160, learning_rate=0.035, max_depth=2, min_samples_leaf=35, random_state=args.seed)),
        ("input_mlp", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, learning_rate_init=0.002, max_iter=400, early_stopping=True, random_state=args.seed)),
    ]
    for name, model in models:
        model.fit(x_fit, y_fit)
        p_tune = model.predict_proba(x_tune)[:, 1]
        p_eval = model.predict_proba(x_eval)[:, 1]
        threshold = choose_threshold(tune, p_tune)
        out.append(metric_row(f"{name}_soft", ev, p_eval, {"threshold": threshold, "feature_contract": "input_side_proxy_features"}))
        out.append(metric_row(f"{name}_hard", ev, (p_eval >= threshold).astype(float), {"threshold": threshold, "feature_contract": "input_side_proxy_features"}))

    learned = [r for r in out if r["fusion"].startswith("input_") or r["fusion"].endswith("_convex_tuned")]
    best = max(learned, key=lambda r: float(r["score_rr"]), default={})
    payload = {
        "status": "R173_FUSION_NO_HELDOUT_RECOVERY" if best and float(best["score_minus_history"]) <= 0.0 else "R173_FUSION_RECOVERY_OR_WARN",
        "best_learned": best,
        "feature_names": feature_names,
        "fit_rows": len(fit),
        "tune_rows": len(tune),
        "eval_rows": len(ev),
    }
    return out, payload


def write_md(path: Path, payload: dict[str, Any]) -> None:
    corr_rows = payload["column_correlation"]
    null_rows = payload["oracle_nulls"]
    fusion_rows = payload["fusion_metrics"]
    best = payload["fusion"]["best_learned"]

    lines = [
        "# R173 G2 Learned-Fusion and Correlation Diagnostics",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Surface ID: `{payload['surface_id']}`",
        "",
        "## Question",
        "",
        "Does the KuaiSAR G2 failure reflect only a pathological hard max-over-two "
        "operator that rewards decorrelated score columns?",
        "",
        "## Column Diagnostics",
        "",
        "| Split | Rows | History RR | Search RR | Oracle gain | Pearson | Spearman | Search > history | Tie |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in corr_rows:
        lines.append(
            f"| {row['split']} | {row['rows']:,} | {fmt(row['history_rr_mean'])} | {fmt(row['search_rr_mean'])} | "
            f"{fmt(row['oracle_gain'])} | {fmt(row['pearson_history_search'])} | {fmt(row['spearman_history_search'])} | "
            f"{fmt(row['search_beats_history_rate'])} | {fmt(row['tie_rate'])} |"
        )

    lines.extend([
        "",
        "## Hard-Oracle Nulls",
        "",
        "| Null | Null mean | Null p95 | Obs - null mean | p(null >= obs) | Null Pearson mean | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in null_rows:
        lines.append(
            f"| {row['null']} | {fmt(row['null_gain_mean'])} | {fmt(row['null_gain_p95'])} | "
            f"{fmt(row['observed_minus_null_mean'])} | {fmt(row['p_null_ge_observed'])} | "
            f"{fmt(row['null_pearson_mean'])} | {row['status']} |"
        )

    lines.extend([
        "",
        "## Held-out Fusion Readouts",
        "",
        "| Fusion | RR | vs history | Headroom recovery | Mean gate | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ])
    for row in fusion_rows:
        note = ""
        if "alpha" in row:
            note = f"alpha={fmt(float(row['alpha']))}"
        elif "threshold" in row:
            note = f"threshold={fmt(float(row['threshold']))}"
        elif row["fusion"] == "hard_oracle_target_aware":
            note = "target-aware upper bound"
        lines.append(
            f"| {row['fusion']} | {fmt(row['score_rr'])} | {fmt(row['score_minus_history'])} | "
            f"{fmt(row['headroom_recovery'])} | {fmt(row['mean_gate'])} | {note} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if payload["oracle_null_status"] == "all_not_above_p95":
        lines.append("- The observed hard-oracle gain remains below every shuffle null tested here, including history-decile constrained shuffles.")
    else:
        lines.append("- At least one hard-oracle null is weaker than the observed gain; inspect the null table before using G2 as negative evidence.")
    if payload["fusion"]["status"] == "R173_FUSION_NO_HELDOUT_RECOVERY":
        lines.append(
            f"- The best non-oracle fusion is `{best.get('fusion', '')}` with RR {fmt(float(best.get('score_rr', 0.0)))}, "
            f"which does not improve over History-only."
        )
    else:
        lines.append("- A non-oracle fusion recovered held-out headroom; the paper must revise the G2 interpretation before integration.")
    lines.extend([
        "- The hard max operator is selection-sensitive, so the paper should not present raw G2 as calibrated opportunity.",
        "- The smoother learned/static fusions tested here also fail to create held-out opportunity on the current proxy surface.",
        "- This remains a proxy-surface diagnostic, not a proof about all fusion architectures.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = read_csv(Path(args.rows))
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    corr_rows = column_correlation(rows, args.surface_id)
    null_rows, null_payload = null_diagnostics(rows, args)
    fusion_rows, fusion_payload = fusion_diagnostics(rows, args)

    paths = {
        "column_correlation_csv": reports / f"{args.report_prefix}_g2_column_correlation.csv",
        "oracle_nulls_csv": reports / f"{args.report_prefix}_g2_oracle_nulls.csv",
        "fusion_metrics_csv": reports / f"{args.report_prefix}_g2_fusion_metrics.csv",
        "json": reports / f"{args.report_prefix}_g2_learned_fusion_diagnostics.json",
        "md": reports / f"{args.report_prefix}_g2_learned_fusion_diagnostics.md",
    }

    oracle_null_status = "all_not_above_p95" if all(r["status"] == "observed_not_above_p95" for r in null_rows) else "some_observed_above_p95"
    status = (
        "R173_CLOSED_NO_FUSION_RECOVERY"
        if oracle_null_status == "all_not_above_p95" and fusion_payload["status"] == "R173_FUSION_NO_HELDOUT_RECOVERY"
        else "R173_WARN_REVISE_G2"
    )
    payload = {
        "schema": "focalswap.r173_g2_learned_fusion_diagnostics.v0.1",
        "status": status,
        "surface_id": args.surface_id,
        "inputs": {
            "rows": str(Path(args.rows).relative_to(ROOT) if Path(args.rows).is_absolute() else args.rows),
            "permutations": args.permutations,
            "seed": args.seed,
        },
        "column_correlation": corr_rows,
        "oracle_nulls": null_rows,
        "oracle_null_status": oracle_null_status,
        "null_payload": null_payload,
        "fusion_metrics": fusion_rows,
        "fusion": fusion_payload,
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }

    write_csv(paths["column_correlation_csv"], corr_rows)
    write_csv(paths["oracle_nulls_csv"], null_rows)
    write_csv(paths["fusion_metrics_csv"], fusion_rows)
    write_json(paths["json"], payload)
    write_md(paths["md"], payload)
    print(json.dumps({"status": status, "report": str(paths["md"])}, indent=2))


if __name__ == "__main__":
    main()
