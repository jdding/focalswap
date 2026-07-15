#!/usr/bin/env python3
"""R094r passive-confidence abstention gate audit.

This local CPU audit tests the pivot suggested by R093r: learn whether passive
recommendation state is confident enough to abstain from search transfer.  The
gate is trained only from input-side confidence/context features.  Outcome
scores come from R093r's item-level passive/search/oracle proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories, read_jsonl  # noqa: E402


DATA = ROOT / "data" / "processed"
B6 = DATA / "b6"
REPORTS = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--r093r-rows", default=str(REPORTS / "r093r_item_proxy_rows.csv"))
    p.add_argument("--examples-path", default=str(B6 / "qbridge_examples.jsonl"))
    p.add_argument("--item-metadata", default=str(DATA / "item_metadata.parquet"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--margin", type=float, default=0.005)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=0.002)
    p.add_argument("--recent-k", type=int, default=8)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def frac(n: int, d: int) -> float:
    return n / d if d else 0.0


def fmt(x: float) -> str:
    return f"{x:.4f}"


def list_ints(row: dict[str, Any], key: str) -> list[int]:
    return [int(x) for x in row.get(key, []) if x is not None]


def raw_targets(row: dict[str, Any], max_bridge: int = 3) -> list[int]:
    out: list[int] = []
    if bool(row.get("y_rank_positive", False)) and row.get("y_rank_item") is not None:
        out.append(int(row["y_rank_item"]))
    out.extend(list_ints(row, "y_bridge_items")[:max_bridge])
    seen: set[int] = set()
    uniq: list[int] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def source_vocab(rows: list[dict[str, str]]) -> list[str]:
    counts = Counter(str(r["source"]) for r in rows if r["split"] == "train")
    return [x for x, _ in counts.most_common(8)]


def target_popularity(raw_by_id: dict[str, dict[str, Any]]) -> tuple[Counter[int], float]:
    pop: Counter[int] = Counter()
    for row in raw_by_id.values():
        if str(row.get("split")) != "train":
            continue
        for t in raw_targets(row):
            pop[int(t)] += 1
    max_log = math.log1p(max(pop.values()) if pop else 1)
    return pop, max_log


def cat_stats(items: list[int], item_cat: dict[int, int]) -> tuple[float, float, int]:
    cats = [int(item_cat[x]) for x in items if int(x) in item_cat]
    if not cats:
        return 0.0, 0.0, 0
    counts = Counter(cats)
    concentration = max(counts.values()) / len(cats)
    probs = [c / len(cats) for c in counts.values()]
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    entropy_norm = entropy / math.log(len(counts) + 1e-12) if len(counts) > 1 else 0.0
    return concentration, entropy_norm, len(counts)


def recent_stats(items: list[int], item_cat: dict[int, int], k: int) -> tuple[float, float, float]:
    if not items:
        return 0.0, 0.0, 0.0
    recent = items[-k:]
    earlier = items[:-k]
    earlier_items = set(earlier)
    recent_item_repeat = sum(1 for x in recent if x in earlier_items) / len(recent)
    earlier_cats = {int(item_cat[x]) for x in earlier if int(x) in item_cat}
    recent_cats = [int(item_cat[x]) for x in recent if int(x) in item_cat]
    recent_cat_repeat = sum(1 for c in recent_cats if c in earlier_cats) / len(recent_cats) if recent_cats else 0.0
    all_cats = earlier_cats | set(recent_cats)
    recent_cat_coverage = len(set(recent_cats)) / len(all_cats) if all_cats else 0.0
    return recent_item_repeat, recent_cat_repeat, recent_cat_coverage


def passive_proxy(pre: list[int], pop: Counter[int], max_log: float) -> float:
    if not pre or max_log <= 0:
        return 0.0
    scores = [math.log1p(pop.get(int(x), 0)) / max_log for x in pre]
    return sum(scores) / len(scores)


def build_feature_row(
    r093: dict[str, str],
    raw: dict[str, Any],
    item_cat: dict[int, int],
    pop: Counter[int],
    max_log: float,
    vocab: list[str],
    recent_k: int,
    margin: float,
) -> dict[str, Any]:
    pre = list_ints(raw, "pre_items")
    search_pos = list_ints(raw, "search_pos_items")
    search_neg = list_ints(raw, "search_neg_items")
    concentration, entropy, unique_cats = cat_stats(pre, item_cat)
    recent_item_repeat, recent_cat_repeat, recent_cat_coverage = recent_stats(pre, item_cat, recent_k)
    src = str(r093["source"])
    features: dict[str, float] = {
        "bias": 1.0,
        "history_len_log": math.log1p(len(pre)),
        "history_cat_concentration": concentration,
        "history_cat_entropy": entropy,
        "history_unique_cat_log": math.log1p(unique_cats),
        "recent_item_repeat": recent_item_repeat,
        "recent_cat_repeat": recent_cat_repeat,
        "recent_cat_coverage": recent_cat_coverage,
        "passive_proxy_score": passive_proxy(pre[-recent_k:], pop, max_log),
        "search_action_log": math.log1p(len(search_pos) + len(search_neg)),
        "search_pos_log": math.log1p(len(search_pos)),
        "search_neg_log": math.log1p(len(search_neg)),
        "cat_conflict": float(r093["cat_conflict"]),
        "cat_jaccard": float(r093["cat_jaccard"]),
    }
    for s in vocab:
        features[f"source={s}"] = 1.0 if src == s else 0.0
    passive = float(r093["passive_rr"])
    search = float(r093["search_rr"])
    gap = search - passive
    return {
        "example_id": r093["example_id"],
        "split": r093["split"],
        "source": src,
        "passive_rr": passive,
        "search_rr": search,
        "oracle_rr": float(r093["oracle_rr"]),
        "random_gate_rr": float(r093["random_gate_rr"]),
        "label_transfer": int(gap > margin),
        "label_nontransfer": int((-gap) > margin),
        "outcome_gap": gap,
        "features": features,
    }


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[float(r["features"].get(f, 0.0)) for f in feature_names] for r in rows], dtype=np.float64)
    y = np.array([int(r["label_transfer"]) for r in rows], dtype=np.float64)
    return x, y


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    train_z = (train_x - mean) / std
    eval_z = (eval_x - mean) / std
    # Keep explicit bias unstandardized if present as first column.
    train_z[:, 0] = 1.0
    eval_z[:, 0] = 1.0
    return train_z, eval_z, mean, std


def fit_logistic(x: np.ndarray, y: np.ndarray, epochs: int, lr: float, l2: float) -> np.ndarray:
    w = np.zeros(x.shape[1], dtype=np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pos_weight = neg / pos
    weights = np.where(y > 0.5, pos_weight, 1.0)
    for _ in range(epochs):
        logits = x @ w
        pred = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        grad = (x.T @ ((pred - y) * weights)) / len(y)
        grad += l2 * w
        grad[0] -= l2 * w[0]
        w -= lr * grad
    return w


def probabilities(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -40, 40)))


def score_gate(rows: list[dict[str, Any]], gates: list[float]) -> float:
    vals = []
    for row, g in zip(rows, gates):
        vals.append(row["passive_rr"] + float(g) * (row["search_rr"] - row["passive_rr"]))
    return avg(vals)


def fixed_score(rows: list[dict[str, Any]], gate: float) -> float:
    return score_gate(rows, [gate] * len(rows))


def oracle_score(rows: list[dict[str, Any]]) -> float:
    return avg([max(r["passive_rr"], r["search_rr"]) for r in rows])


def choose_threshold(rows: list[dict[str, Any]], probs: np.ndarray) -> float:
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        gates = [1.0 if p >= t else 0.0 for p in probs]
        s = score_gate(rows, gates)
        if s > best_score:
            best_score = s
            best_t = float(t)
    return best_t


def gate_metrics(rows: list[dict[str, Any]], gates: list[float], name: str) -> dict[str, Any]:
    passive = fixed_score(rows, 0.0)
    search = fixed_score(rows, 1.0)
    oracle = oracle_score(rows)
    score = score_gate(rows, gates)
    headroom = oracle - passive
    transfer_rows = [i for i, r in enumerate(rows) if r["label_transfer"]]
    nontransfer_rows = [i for i, r in enumerate(rows) if r["label_nontransfer"]]
    neutral_rows = [i for i, r in enumerate(rows) if not r["label_transfer"] and not r["label_nontransfer"]]
    return {
        "gate": name,
        "examples": len(rows),
        "score_rr": score,
        "passive_rr": passive,
        "search_rr": search,
        "oracle_rr": oracle,
        "score_minus_passive": score - passive,
        "oracle_headroom": headroom,
        "headroom_recovery": (score - passive) / headroom if headroom > 1e-12 else 0.0,
        "mean_gate": avg([float(g) for g in gates]),
        "fire_rate_transfer": avg([float(gates[i] >= 0.5) for i in transfer_rows]),
        "fire_rate_nontransfer": avg([float(gates[i] >= 0.5) for i in nontransfer_rows]),
        "fire_rate_neutral": avg([float(gates[i] >= 0.5) for i in neutral_rows]),
        "transfer_examples": len(transfer_rows),
        "nontransfer_examples": len(nontransfer_rows),
        "neutral_examples": len(neutral_rows),
    }


def tune_single_feature_rule(train_rows: list[dict[str, Any]], feature: str) -> tuple[str, float, float]:
    values = sorted({float(r["features"].get(feature, 0.0)) for r in train_rows})
    if not values:
        return "<=", 0.0, fixed_score(train_rows, 0.0)
    if len(values) > 40:
        qs = [i / 40 for i in range(1, 40)]
        thresholds = [float(np.quantile(values, q)) for q in qs]
    else:
        thresholds = values
    best = ("<=", thresholds[0], -1.0)
    for direction in ["<=", ">"]:
        for t in thresholds:
            gates = []
            for row in train_rows:
                v = float(row["features"].get(feature, 0.0))
                fire = v <= t if direction == "<=" else v > t
                gates.append(1.0 if fire else 0.0)
            s = score_gate(train_rows, gates)
            if s > best[2]:
                best = (direction, float(t), s)
    return best


def apply_rule(rows: list[dict[str, Any]], feature: str, direction: str, threshold: float) -> list[float]:
    out = []
    for row in rows:
        v = float(row["features"].get(feature, 0.0))
        fire = v <= threshold if direction == "<=" else v > threshold
        out.append(1.0 if fire else 0.0)
    return out


def combined_rule(train_rows: list[dict[str, Any]]) -> tuple[float, int, int, float]:
    best = (0.0, 0, 0, -1.0)
    passive_vals = sorted(float(r["features"]["passive_proxy_score"]) for r in train_rows)
    thresholds = [float(np.quantile(passive_vals, q)) for q in np.linspace(0.1, 0.9, 17)]
    for t in thresholds:
        for min_pos in [0, 1, 2, 3]:
            for min_actions in [1, 3, 6]:
                gates = []
                for row in train_rows:
                    f = row["features"]
                    fire = (
                        float(f["passive_proxy_score"]) <= t
                        and float(f["search_pos_log"]) >= math.log1p(min_pos)
                        and float(f["search_action_log"]) >= math.log1p(min_actions)
                    )
                    gates.append(1.0 if fire else 0.0)
                s = score_gate(train_rows, gates)
                if s > best[3]:
                    best = (float(t), min_pos, min_actions, s)
    return best


def apply_combined(rows: list[dict[str, Any]], rule: tuple[float, int, int, float]) -> list[float]:
    t, min_pos, min_actions, _ = rule
    out = []
    for row in rows:
        f = row["features"]
        fire = (
            float(f["passive_proxy_score"]) <= t
            and float(f["search_pos_log"]) >= math.log1p(min_pos)
            and float(f["search_action_log"]) >= math.log1p(min_actions)
        )
        out.append(1.0 if fire else 0.0)
    return out


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    item_cat = load_item_categories(Path(args.item_metadata))
    r093_rows = read_csv(Path(args.r093r_rows))
    raw = read_jsonl(Path(args.examples_path))
    raw_by_id = {str(r["example_id"]): r for r in raw}
    pop, max_log = target_popularity(raw_by_id)
    vocab = source_vocab(r093_rows)
    rows = [
        build_feature_row(r, raw_by_id[r["example_id"]], item_cat, pop, max_log, vocab, args.recent_k, args.margin)
        for r in r093_rows
        if r["example_id"] in raw_by_id
    ]
    train_rows = [r for r in rows if r["split"] == "train"]
    eval_rows = [r for r in rows if r["split"] in {"validation", "test"}]
    feature_names = list(train_rows[0]["features"].keys())
    x_train, y_train = matrix(train_rows, feature_names)
    x_eval, _ = matrix(eval_rows, feature_names)
    z_train, z_eval, _, _ = standardize(x_train, x_eval)
    w = fit_logistic(z_train, y_train, args.epochs, args.lr, args.l2)
    p_train = probabilities(z_train, w)
    p_eval = probabilities(z_eval, w)
    threshold = choose_threshold(train_rows, p_train)

    train_hard = [1.0 if p >= threshold else 0.0 for p in p_train]
    eval_hard = [1.0 if p >= threshold else 0.0 for p in p_eval]
    train_soft = [float(p) for p in p_train]
    eval_soft = [float(p) for p in p_eval]

    candidate_features = [
        "passive_proxy_score",
        "history_len_log",
        "history_cat_concentration",
        "history_cat_entropy",
        "recent_item_repeat",
        "recent_cat_repeat",
        "recent_cat_coverage",
        "search_action_log",
        "search_pos_log",
        "cat_conflict",
        "cat_jaccard",
    ]
    rule_rows: list[dict[str, Any]] = []
    best_single: tuple[str, str, float, float] | None = None
    for feat in candidate_features:
        direction, thr, train_score = tune_single_feature_rule(train_rows, feat)
        eval_gates = apply_rule(eval_rows, feat, direction, thr)
        m = gate_metrics(eval_rows, eval_gates, f"rule:{feat}{direction}{thr:.4f}")
        rule_rows.append({"feature": feat, "direction": direction, "threshold": thr, "train_score": train_score, **m})
        if best_single is None or float(train_score) > best_single[3]:
            best_single = (feat, direction, thr, float(train_score))

    combo = combined_rule(train_rows)
    combo_eval = apply_combined(eval_rows, combo)

    metrics = [
        gate_metrics(eval_rows, [0.0] * len(eval_rows), "fixed_0_passive"),
        gate_metrics(eval_rows, [1.0] * len(eval_rows), "fixed_1_search"),
        gate_metrics(eval_rows, [0.5] * len(eval_rows), "random_0.5"),
        gate_metrics(eval_rows, [1.0 if r["label_transfer"] else 0.0 for r in eval_rows], "oracle_hard"),
        gate_metrics(eval_rows, eval_soft, "logistic_soft"),
        gate_metrics(eval_rows, eval_hard, f"logistic_hard_t{threshold:.2f}"),
        gate_metrics(eval_rows, combo_eval, f"combined_passive_search_rule_t{combo[0]:.3f}_pos{combo[1]}_act{combo[2]}"),
    ]
    if best_single:
        feat, direction, thr, _ = best_single
        metrics.append(gate_metrics(eval_rows, apply_rule(eval_rows, feat, direction, thr), f"best_single_rule:{feat}{direction}{thr:.4f}"))

    coef_rows = []
    abs_sum = float(np.sum(np.abs(w[1:]))) or 1.0
    for name, coef in zip(feature_names, w):
        coef_rows.append({"feature": name, "coef": float(coef), "abs_coef_share": float(abs(coef) / abs_sum if name != "bias" else 0.0)})
    coef_rows.sort(key=lambda r: abs(float(r["coef"])), reverse=True)

    logistic_eval = next(m for m in metrics if str(m["gate"]).startswith("logistic_hard"))
    passive_eval = next(m for m in metrics if m["gate"] == "fixed_0_passive")
    best_single_eval = next((m for m in metrics if str(m["gate"]).startswith("best_single_rule")), None)
    top_share = max(float(r["abs_coef_share"]) for r in coef_rows if r["feature"] != "bias")
    success = {
        "beats_fixed_0_passive": float(logistic_eval["score_minus_passive"]) > 0.0,
        "recovers_30pct_oracle_headroom": float(logistic_eval["headroom_recovery"]) >= 0.30,
        "recovers_50pct_oracle_headroom": float(logistic_eval["headroom_recovery"]) >= 0.50,
        "fires_more_on_transfer_than_nontransfer": float(logistic_eval["fire_rate_transfer"]) > float(logistic_eval["fire_rate_nontransfer"]),
        "suppresses_nontransfer_below_0.5": float(logistic_eval["fire_rate_nontransfer"]) < 0.50,
        "not_single_feature_dominated": bool(best_single_eval is None or float(logistic_eval["score_rr"]) >= float(best_single_eval["score_rr"]) - 1e-9) and top_share < 0.50,
    }
    status = "R094R_PASS_PASSIVE_CONFIDENCE_GATE" if all(
        [
            success["beats_fixed_0_passive"],
            success["recovers_30pct_oracle_headroom"],
            success["fires_more_on_transfer_than_nontransfer"],
            success["suppresses_nontransfer_below_0.5"],
            success["not_single_feature_dominated"],
        ]
    ) else "R094R_FAIL_OR_WEAK_PASSIVE_CONFIDENCE_GATE"

    write_csv(reports / "passive_confidence_gate_metrics.csv", metrics)
    write_csv(reports / "r094r_passive_confidence_rule_metrics.csv", rule_rows)
    write_csv(reports / "r094r_passive_confidence_coefficients.csv", coef_rows)
    payload = {
        "status": status,
        "schema": "searchcpt.r094r.passive_confidence_gate.v0.1",
        "inputs": {
            "r093r_rows": "reports/r093r_item_proxy_rows.csv",
            "examples_path": "data/processed/b6/qbridge_examples.jsonl",
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "features": feature_names,
            "feature_contract": "input_side_only; excludes post_event_count and post_positive_count because they are post-search outcomes",
            "threshold": threshold,
            "combined_rule": {
                "passive_proxy_threshold": combo[0],
                "min_search_pos": combo[1],
                "min_search_actions": combo[2],
            },
        },
        "success_criteria": success,
        "metrics": metrics,
        "top_coefficients": coef_rows[:10],
        "best_single_rule": best_single_eval,
    }
    write_json(reports / "passive_confidence_gate.json", payload)
    md = [
        "# R094r Passive-Confidence Abstention Gate",
        "",
        f"- Status: **{status}**",
        f"- Train rows: {len(train_rows):,}",
        f"- Eval rows: {len(eval_rows):,}",
        f"- Tuned logistic threshold: {threshold:.2f}",
        "",
        "## Eval Gate Metrics",
        "",
        "| Gate | RR | vs passive | Headroom recovery | Fire transfer | Fire non-transfer | Mean gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        md.append(
            f"| {m['gate']} | {fmt(float(m['score_rr']))} | {fmt(float(m['score_minus_passive']))} | "
            f"{fmt(float(m['headroom_recovery']))} | {fmt(float(m['fire_rate_transfer']))} | "
            f"{fmt(float(m['fire_rate_nontransfer']))} | {fmt(float(m['mean_gate']))} |"
        )
    md.extend(
        [
            "",
            "## Success Criteria",
            "",
            "| Criterion | Pass |",
            "|---|---:|",
        ]
    )
    for k, v in success.items():
        md.append(f"| {k} | {str(bool(v)).lower()} |")
    md.extend(
        [
            "",
            "## Top Coefficients",
            "",
            "| Feature | Coef | Abs share |",
            "|---|---:|---:|",
        ]
    )
    for r in coef_rows[:10]:
        md.append(f"| {r['feature']} | {fmt(float(r['coef']))} | {fmt(float(r['abs_coef_share']))} |")
    md.extend(["", "## Decision", ""])
    if status.startswith("R094R_PASS"):
        md.append("Passive-confidence abstention passes the local oracle-headroom gate. Next step is code review and stronger SAR baselines before any GPU work.")
    else:
        md.append("Passive-confidence abstention does not pass the local gate. Do not train a larger gate or launch GPU from this feature set; revise the feature/proxy route or downgrade the claim.")
    (reports / "passive_confidence_gate.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(reports / "passive_confidence_gate.md")}, indent=2))


if __name__ == "__main__":
    main()
