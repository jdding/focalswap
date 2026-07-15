#!/usr/bin/env python3
"""R093r raw post-rec outcome and abstention-headroom audit.

This is a pre-training-free falsifier for the V6 behavior-level SAR
transfer/abstention route.  Unlike R092, it does not reuse `Y_carry_v2`.
Targets are taken directly from raw post-search recommendation positives, and
the item-level proxy is a train-split transition ranker rather than a
category-overlap score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import (  # noqa: E402
    cats,
    load_item_categories,
    pick_swap,
    qbucket,
    pre_bucket,
    bridge_bucket,
    read_jsonl,
)


DATA = ROOT / "data" / "processed"
B6 = DATA / "b6"
REPORTS = ROOT / "reports"
V6 = DATA / "v6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-path", default=str(B6 / "qbridge_examples.jsonl"))
    p.add_argument("--item-metadata", default=str(DATA / "item_metadata.parquet"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--output-dir", default=str(V6))
    p.add_argument("--report-prefix", default="r093r")
    p.add_argument("--max-examples", type=int, default=0)
    p.add_argument("--score-sample-per-split", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--max-pre-items", type=int, default=12)
    p.add_argument("--max-search-items", type=int, default=12)
    p.add_argument("--max-bridge-targets", type=int, default=3)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--swap-tries", type=int, default=60)
    p.add_argument("--max-swap-rows", type=int, default=30000)
    p.add_argument("--margin", type=float, default=0.005)
    return p.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def frac(num: int, den: int) -> float:
    return num / den if den else 0.0


def fmt(x: float) -> str:
    return f"{x:.4f}"


def list_ints(row: dict[str, Any], key: str) -> list[int]:
    return [int(x) for x in row.get(key, []) if x is not None]


def uniq(xs: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in xs:
        y = int(x)
        if y in seen:
            continue
        seen.add(y)
        out.append(y)
    return out


def stable_score(x: int) -> float:
    digest = hashlib.md5(str(int(x)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def stable_sample(xs: list[int], k: int, salt: str) -> list[int]:
    if len(xs) <= k:
        return list(xs)
    ranked = sorted(xs, key=lambda x: hashlib.md5(f"{salt}:{int(x)}".encode("utf-8")).hexdigest())
    return ranked[:k]


def raw_targets(row: dict[str, Any], max_bridge: int) -> list[int]:
    out: list[int] = []
    if bool(row.get("y_rank_positive", False)) and row.get("y_rank_item") is not None:
        out.append(int(row["y_rank_item"]))
    out.extend(list_ints(row, "y_bridge_items")[:max_bridge])
    return uniq(out)


def search_items(row: dict[str, Any], max_items: int) -> list[int]:
    pos = list_ints(row, "search_pos_items")
    neg = list_ints(row, "search_neg_items")
    # Positive search feedback is stronger.  Keep a bounded amount of negative
    # exposure because KuaiSAR often has no clicked search item.
    return uniq(pos + neg[: max(0, max_items - len(pos))])[:max_items]


def pre_items(row: dict[str, Any], max_items: int) -> list[int]:
    return list_ints(row, "pre_items")[-max_items:]


def categories(items: list[int], item_cat: dict[int, int]) -> set[int]:
    return cats(items, item_cat)


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r.get("split")) == split]


def stratified_sample_rows(rows: list[dict[str, Any]], per_split: int) -> list[dict[str, Any]]:
    if per_split <= 0:
        return rows
    out: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "unknown"))].append(row)
    for split, group in sorted(by_split.items()):
        if len(group) <= per_split:
            out.extend(group)
            continue
        out.extend(
            sorted(
                group,
                key=lambda x: hashlib.md5(str(x.get("example_id", "")).encode("utf-8")).hexdigest(),
            )[:per_split]
        )
    return out


class TransitionRanker:
    def __init__(self) -> None:
        self.pair: Counter[tuple[int, int]] = Counter()
        self.src_total: Counter[int] = Counter()
        self.tgt_pop: Counter[int] = Counter()
        self.global_pop: Counter[int] = Counter()

    def add(self, evidence: list[int], targets: list[int]) -> None:
        ev = uniq(evidence)
        tg = uniq(targets)
        for t in tg:
            self.tgt_pop[int(t)] += 1
            self.global_pop[int(t)] += 1
        for e in ev:
            self.src_total[int(e)] += len(tg)
            for t in tg:
                self.pair[(int(e), int(t))] += 1

    def item_score(self, evidence: list[int], item: int) -> float:
        if not evidence:
            return 0.0
        best = 0.0
        target = int(item)
        for e in uniq(evidence):
            total = self.src_total.get(int(e), 0)
            if total <= 0:
                continue
            c = self.pair.get((int(e), target), 0)
            if c <= 0:
                continue
            # Bounded association.  The scale is small but comparable across
            # passive/search evidence because both use the same train index.
            best = max(best, math.log1p(c) / math.log1p(total + 1.0))
        return best


def build_ranker(train_rows: list[dict[str, Any]], args: argparse.Namespace) -> TransitionRanker:
    ranker = TransitionRanker()
    for row in train_rows:
        targets = raw_targets(row, args.max_bridge_targets)
        if not targets:
            continue
        ranker.add(pre_items(row, args.max_pre_items), targets)
        ranker.add(search_items(row, args.max_search_items), targets)
    return ranker


def candidate_set(row: dict[str, Any], targets: list[int], ranker: TransitionRanker, args: argparse.Namespace) -> list[int]:
    candidates = uniq(
        targets
        + pre_items(row, args.max_pre_items)
        + search_items(row, args.max_search_items)
        + [x for x, _ in ranker.global_pop.most_common(30)]
    )
    if len(candidates) > args.max_candidates:
        keep = set(targets)
        rest = [x for x in candidates if x not in keep]
        candidates = uniq(list(keep) + stable_sample(rest, args.max_candidates - len(keep), str(row.get("example_id"))))
    return candidates


def reciprocal_rank(
    ranker: TransitionRanker,
    evidence: list[int],
    targets: list[int],
    candidates: list[int],
) -> tuple[float, float, float]:
    target_set = set(int(x) for x in targets)
    scored = []
    nonzero = 0
    for c in candidates:
        s = ranker.item_score(evidence, int(c))
        nonzero += int(s > 0.0)
        scored.append((int(c), s))
    scored.sort(key=lambda x: (-x[1], stable_score(x[0])))
    rr = 0.0
    hit10 = 0.0
    for idx, (item, _) in enumerate(scored, start=1):
        if item in target_set:
            rr = 1.0 / idx
            hit10 = 1.0 if idx <= 10 else 0.0
            break
    coverage = nonzero / len(scored) if scored else 0.0
    target_alignment = avg([ranker.item_score(evidence, t) for t in targets])
    return rr, hit10, max(coverage, target_alignment)


def profile(row: dict[str, Any], item_cat: dict[int, int], args: argparse.Namespace) -> dict[str, Any]:
    pre = pre_items(row, args.max_pre_items)
    search = search_items(row, args.max_search_items)
    target = raw_targets(row, args.max_bridge_targets)
    search_cats = categories(search, item_cat)
    pre_cats = categories(pre, item_cat)
    target_cats = categories(target, item_cat)
    overlap = len(pre_cats & search_cats) / len(pre_cats | search_cats) if pre_cats or search_cats else 0.0
    conflict = bool(pre_cats and search_cats and not (pre_cats & search_cats))
    b = {
        "split": str(row.get("split", "unknown")),
        "source": str(row.get("source", "unknown")),
        "search_pos": qbucket(len(list_ints(row, "search_pos_items"))),
        "search_neg": qbucket(len(list_ints(row, "search_neg_items"))),
        "pre": pre_bucket(len(list_ints(row, "pre_items"))),
        "bridge": bridge_bucket(len(list_ints(row, "y_bridge_items"))),
    }
    return {
        "example_id": row.get("example_id"),
        "split": str(row.get("split", "unknown")),
        "source": str(row.get("source", "unknown")),
        "query_id": int(row.get("query_id", -1)),
        "user_id": int(row.get("user_id", -1)),
        "timestamp_ms": int(row.get("timestamp_ms", 0) or 0),
        "pre_items": pre,
        "search_items": search,
        "search_pos_items": list_ints(row, "search_pos_items"),
        "search_neg_items": list_ints(row, "search_neg_items"),
        "raw_targets": target,
        "targets": {
            "Y_carry_full": target,
            "Y_time_sim": target,
            "Y_time": target,
        },
        "search_cats": search_cats,
        "pre_cats": pre_cats,
        "target_cats": target_cats,
        "cat_conflict": conflict,
        "cat_jaccard": overlap,
        "post_event_count": int((row.get("confounders") or {}).get("post_event_count", 0) or 0),
        "post_positive_count": int((row.get("confounders") or {}).get("post_positive_count", 0) or 0),
        "buckets": b,
    }


def build_indices(profiles: list[dict[str, Any]]) -> dict[str, dict[tuple[Any, ...], list[int]]]:
    indices: dict[str, dict[tuple[Any, ...], list[int]]] = {
        "random": defaultdict(list),
        "nuisance": defaultdict(list),
        "same_category": defaultdict(list),
        "high_similarity": defaultdict(list),
    }
    for i, p in enumerate(profiles):
        b = p["buckets"]
        indices["random"][(b["split"],)].append(i)
        indices["nuisance"][(b["split"], b["source"], b["search_pos"], b["search_neg"], b["pre"], b["bridge"])].append(i)
        for c in p["search_cats"]:
            indices["same_category"][(b["split"], c)].append(i)
            indices["high_similarity"][(b["split"], c, b["source"])].append(i)
    return indices


def target_audit(rows: list[dict[str, Any]], item_cat: dict[int, int], args: argparse.Namespace) -> dict[str, Any]:
    split_counts = Counter(str(r.get("split", "unknown")) for r in rows)
    coverage = Counter()
    leaks = Counter()
    sizes: list[int] = []
    cat_rows: list[float] = []
    for row in rows:
        targets = raw_targets(row, args.max_bridge_targets)
        pre = set(pre_items(row, 10_000))
        search = set(search_items(row, 10_000))
        sizes.append(len(targets))
        coverage["has_y_rank_positive"] += int(bool(row.get("y_rank_positive", False)) and row.get("y_rank_item") is not None)
        coverage["has_y_bridge"] += int(bool(list_ints(row, "y_bridge_items")))
        coverage["has_raw_target"] += int(bool(targets))
        coverage["all_targets_known_category"] += int(bool(targets) and all(int(x) in item_cat for x in targets))
        leaks["target_in_pre"] += int(bool(set(targets) & pre))
        leaks["target_in_search"] += int(bool(set(targets) & search))
        pc = categories(list(pre), item_cat)
        sc = categories(list(search), item_cat)
        tc = categories(targets, item_cat)
        if tc:
            cat_rows.append(len(tc & (pc | sc)) / len(tc))
    return {
        "status": "R093R0_PASS_RAW_TARGETS" if coverage["has_raw_target"] > 0 and coverage["all_targets_known_category"] > 0 else "R093R0_FAIL_NO_RAW_TARGET",
        "examples": len(rows),
        "split_counts": dict(split_counts),
        "coverage": {k: {"count": v, "rate": frac(v, len(rows))} for k, v in coverage.items()},
        "avg_raw_target_size": avg([float(x) for x in sizes]),
        "leakage_counts": dict(leaks),
        "mean_target_cat_explained_by_pre_or_search": avg(cat_rows),
    }


def score_rows(
    rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    ranker: TransitionRanker,
    item_cat: dict[int, int],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, p in zip(rows, profiles):
        targets = p["raw_targets"]
        if not targets:
            continue
        candidates = candidate_set(row, targets, ranker, args)
        passive_rr, passive_hit10, passive_cov = reciprocal_rank(ranker, p["pre_items"], targets, candidates)
        search_rr, search_hit10, search_cov = reciprocal_rank(ranker, p["search_items"], targets, candidates)
        # The oracle tests whether a search-heavy transfer policy is ever worse
        # than passive.  Fixed gates interpolate the item-level rank proxy.
        oracle = max(passive_rr, search_rr)
        gap = search_rr - passive_rr
        out.append(
            {
                "example_id": p["example_id"],
                "split": p["split"],
                "source": p["source"],
                "pre_len": len(list_ints(row, "pre_items")),
                "search_pos_count": len(list_ints(row, "search_pos_items")),
                "search_neg_count": len(list_ints(row, "search_neg_items")),
                "search_action_count": len(list_ints(row, "search_pos_items")) + len(list_ints(row, "search_neg_items")),
                "target_size": len(targets),
                "candidate_size": len(candidates),
                "cat_conflict": int(p["cat_conflict"]),
                "cat_jaccard": p["cat_jaccard"],
                "post_event_count": p["post_event_count"],
                "post_positive_count": p["post_positive_count"],
                "passive_rr": passive_rr,
                "search_rr": search_rr,
                "oracle_rr": oracle,
                "random_gate_rr": 0.5 * passive_rr + 0.5 * search_rr,
                "fixed_0.0": passive_rr,
                "fixed_0.2": passive_rr + 0.2 * gap,
                "fixed_0.5": passive_rr + 0.5 * gap,
                "fixed_0.8": passive_rr + 0.8 * gap,
                "fixed_1.0": search_rr,
                "passive_hit10": passive_hit10,
                "search_hit10": search_hit10,
                "passive_nonzero_proxy": passive_cov,
                "search_nonzero_proxy": search_cov,
                "outcome_gap_search_minus_passive": gap,
                "transferable_by_outcome": int(gap > args.margin),
                "nontransfer_by_outcome": int((-gap) > args.margin),
            }
        )
    return out


def add_swap_scores(
    scored: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    profiles: list[dict[str, Any]],
    ranker: TransitionRanker,
    args: argparse.Namespace,
) -> None:
    rng = random.Random(args.seed)
    indices = build_indices(profiles)
    profile_by_id = {p["example_id"]: i for i, p in enumerate(profiles)}
    raw_by_id = {r.get("example_id"): r for r in raw_rows}
    for rec in scored:
        for level in ["L1_nuisance", "L2_same_category"]:
            rec[f"{level}_matched"] = 0
            rec[f"{level}_swap_rr"] = 0.0
            rec[f"{level}_true_minus_swap"] = 0.0
            rec[f"{level}_true_win"] = 0

    eval_rows = [r for r in scored if r["split"] in {"validation", "test"}]
    if args.max_swap_rows and len(eval_rows) > args.max_swap_rows:
        selected_ids = set(
            r["example_id"]
            for r in sorted(
                eval_rows,
                key=lambda x: hashlib.md5(str(x["example_id"]).encode("utf-8")).hexdigest(),
            )[: args.max_swap_rows]
        )
    else:
        selected_ids = {r["example_id"] for r in eval_rows}

    for rec in scored:
        if rec["example_id"] not in selected_ids:
            continue
        i = profile_by_id[rec["example_id"]]
        p = profiles[i]
        row = raw_by_id[rec["example_id"]]
        targets = p["raw_targets"]
        candidates = candidate_set(row, targets, ranker, args)
        for level in ["L1_nuisance", "L2_same_category"]:
            j = pick_swap(i, level, profiles, indices, rng, args.swap_tries)
            if j is None:
                continue
            q = profiles[j]
            swap_rr, _, _ = reciprocal_rank(ranker, q["search_items"], targets, candidates)
            rec[f"{level}_matched"] = 1
            rec[f"{level}_swap_rr"] = swap_rr
            rec[f"{level}_true_minus_swap"] = rec["search_rr"] - swap_rr
            rec[f"{level}_true_win"] = int(rec["search_rr"] > swap_rr)


def bucket_value(row: dict[str, Any], bucket: str) -> str:
    if bucket == "all":
        return "all"
    if bucket == "eval":
        return "eval" if row["split"] in {"validation", "test"} else "train"
    if bucket == "split":
        return str(row["split"])
    if bucket == "source":
        return str(row["source"])
    if bucket == "search_pos":
        return qbucket(int(row["search_pos_count"]))
    if bucket == "search_actions":
        return qbucket(int(row["search_action_count"]))
    if bucket == "pre_len":
        return pre_bucket(int(row["pre_len"]))
    if bucket == "cat_conflict":
        return "conflict" if int(row["cat_conflict"]) else "overlap_or_missing"
    if bucket == "history_strength":
        return "passive_strong" if float(row["passive_rr"]) >= 0.05 else "passive_weak"
    if bucket == "post_size":
        return bridge_bucket(int(row["post_positive_count"]))
    raise ValueError(bucket)


def summarize(rows: list[dict[str, Any]], label: str, value: str, margin: float) -> dict[str, Any]:
    gates = [0.0, 0.2, 0.5, 0.8, 1.0]
    fixed_avgs = {f"fixed_{g:.1f}": avg([float(r[f"fixed_{g:.1f}"]) for r in rows]) for g in gates}
    best_fixed_gate, best_fixed = max(fixed_avgs.items(), key=lambda kv: kv[1])
    passive = avg([float(r["passive_rr"]) for r in rows])
    search = avg([float(r["search_rr"]) for r in rows])
    oracle = avg([float(r["oracle_rr"]) for r in rows])
    random_gate = avg([float(r["random_gate_rr"]) for r in rows])
    l1 = [r for r in rows if int(r.get("L1_nuisance_matched", 0))]
    l2 = [r for r in rows if int(r.get("L2_same_category_matched", 0))]
    return {
        label: value,
        "examples": len(rows),
        "passive_rr": passive,
        "search_rr": search,
        "oracle_rr": oracle,
        "best_fixed_gate": best_fixed_gate,
        "best_fixed_rr": best_fixed,
        "random_gate_rr": random_gate,
        "oracle_minus_best_fixed": oracle - best_fixed,
        "oracle_minus_random": oracle - random_gate,
        "search_minus_passive": search - passive,
        "transferable_rate": frac(sum(int(r["transferable_by_outcome"]) for r in rows), len(rows)),
        "nontransfer_rate": frac(sum(int(r["nontransfer_by_outcome"]) for r in rows), len(rows)),
        "mean_positive_gap": avg([float(r["outcome_gap_search_minus_passive"]) for r in rows if float(r["outcome_gap_search_minus_passive"]) > margin]),
        "mean_negative_gap": avg([-float(r["outcome_gap_search_minus_passive"]) for r in rows if float(r["outcome_gap_search_minus_passive"]) < -margin]),
        "passive_hit10": avg([float(r["passive_hit10"]) for r in rows]),
        "search_hit10": avg([float(r["search_hit10"]) for r in rows]),
        "L1_matched": len(l1),
        "L1_true_swap_win_rate": frac(sum(int(r.get("L1_nuisance_true_win", 0)) for r in l1), len(l1)),
        "L1_mean_true_minus_swap": avg([float(r.get("L1_nuisance_true_minus_swap", 0.0)) for r in l1]),
        "L2_matched": len(l2),
        "L2_true_swap_win_rate": frac(sum(int(r.get("L2_same_category_true_win", 0)) for r in l2), len(l2)),
        "L2_mean_true_minus_swap": avg([float(r.get("L2_same_category_true_minus_swap", 0.0)) for r in l2]),
    }


def stratified(rows: list[dict[str, Any]], margin: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bucket in ["all", "eval", "split", "source", "search_pos", "search_actions", "pre_len", "cat_conflict", "history_strength", "post_size"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[bucket_value(row, bucket)].append(row)
        for value, group in sorted(groups.items()):
            out.append(summarize(group, "bucket", f"{bucket}:{value}", margin))
    return out


def status_from(eval_summary: dict[str, Any]) -> str:
    if not eval_summary:
        return "R093R_FAIL_NO_EVAL_ROWS"
    has_mixed = eval_summary["transferable_rate"] >= 0.05 and eval_summary["nontransfer_rate"] >= 0.05
    has_headroom = eval_summary["oracle_minus_best_fixed"] > 0.005 and eval_summary["oracle_minus_random"] > 0.005
    identifiable = eval_summary["L1_true_swap_win_rate"] >= 0.50
    if has_mixed and has_headroom and identifiable:
        return "R093R_PASS_ORACLE_HEADROOM"
    if eval_summary["nontransfer_rate"] >= 0.05 and not identifiable:
        return "R093R_FAIL_SEARCH_NOT_IDENTIFIABLE"
    if eval_summary["nontransfer_rate"] < 0.05:
        return "R093R_FAIL_WEAK_NONTRANSFER"
    return "R093R_WEAK_ORACLE_HEADROOM"


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    out_dir = Path(args.output_dir)
    reports.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    item_cat = load_item_categories(Path(args.item_metadata))
    raw_rows = read_jsonl(Path(args.examples_path), args.max_examples)
    train_rows = split_rows(raw_rows, "train")
    ranker = build_ranker(train_rows, args)
    score_raw_rows = stratified_sample_rows(raw_rows, args.score_sample_per_split)
    profiles = [profile(row, item_cat, args) for row in score_raw_rows]

    r0 = target_audit(raw_rows, item_cat, args)
    scored = score_rows(score_raw_rows, profiles, ranker, item_cat, args)
    add_swap_scores(scored, score_raw_rows, profiles, ranker, args)
    rows_csv = reports / f"{args.report_prefix}_item_proxy_rows.csv"
    stratified_csv = reports / f"{args.report_prefix}_stratified_oracle.csv"
    json_path = reports / f"{args.report_prefix}_raw_outcome_oracle_audit.json"
    md_path = reports / f"{args.report_prefix}_raw_outcome_oracle_audit.md"
    write_csv(rows_csv, scored)

    strat_rows = stratified(scored, args.margin)
    write_csv(stratified_csv, strat_rows)
    overall = next((r for r in strat_rows if r["bucket"] == "all:all"), {})
    eval_summary = next((r for r in strat_rows if r["bucket"] == "eval:eval"), {})
    train_summary = next((r for r in strat_rows if r["bucket"] == "eval:train"), {})
    status = status_from(eval_summary)
    payload = {
        "status": status,
        "schema": "searchcpt.r093r.raw_outcome_oracle_audit.v0.1",
        "inputs": {
            "examples_path": str(Path(args.examples_path).relative_to(ROOT)),
            "item_metadata": str(Path(args.item_metadata).relative_to(ROOT)),
            "train_rows_for_ranker": len(train_rows),
            "scored_rows": len(score_raw_rows),
            "score_sample_per_split": args.score_sample_per_split,
            "max_pre_items": args.max_pre_items,
            "max_search_items": args.max_search_items,
            "max_bridge_targets": args.max_bridge_targets,
            "max_swap_rows": args.max_swap_rows,
            "swap_tries": args.swap_tries,
            "margin": args.margin,
        },
        "r093r0_raw_target_audit": r0,
        "overall": overall,
        "eval_summary": eval_summary,
        "train_summary": train_summary,
        "outputs": {
            "rows_csv": str(rows_csv.relative_to(ROOT)),
            "stratified_csv": str(stratified_csv.relative_to(ROOT)),
        },
        "criteria": {
            "mixed_transfer_and_nontransfer_eval_ge_0.05": bool(eval_summary) and eval_summary["transferable_rate"] >= 0.05 and eval_summary["nontransfer_rate"] >= 0.05,
            "oracle_minus_best_fixed_eval_gt_0.005": bool(eval_summary) and eval_summary["oracle_minus_best_fixed"] > 0.005,
            "oracle_minus_random_eval_gt_0.005": bool(eval_summary) and eval_summary["oracle_minus_random"] > 0.005,
            "L1_true_swap_win_eval_ge_0.50": bool(eval_summary) and eval_summary["L1_true_swap_win_rate"] >= 0.50,
        },
    }
    write_json(json_path, payload)

    md: list[str] = [
        "# R093r Raw Outcome Oracle Audit",
        "",
        f"- Status: **{status}**",
        f"- Raw examples: {len(raw_rows):,}",
        f"- Train rows for transition ranker: {len(train_rows):,}",
        f"- Scored rows: {len(scored):,}",
        f"- Score sample per split: {args.score_sample_per_split or 'all'}",
        f"- Detail rows: `{rows_csv.relative_to(ROOT)}`",
        f"- Stratified oracle: `{stratified_csv.relative_to(ROOT)}`",
        "",
        "## R093r-0 Raw Post-Rec Targets",
        "",
        f"- Status: **{r0['status']}**",
        f"- Raw target coverage: {fmt(r0['coverage']['has_raw_target']['rate'])}",
        f"- y_rank positive coverage: {fmt(r0['coverage']['has_y_rank_positive']['rate'])}",
        f"- y_bridge coverage: {fmt(r0['coverage']['has_y_bridge']['rate'])}",
        f"- Avg raw target size: {fmt(float(r0['avg_raw_target_size']))}",
        f"- Target-in-pre leakage count: {r0['leakage_counts'].get('target_in_pre', 0):,}",
        f"- Target-in-search leakage count: {r0['leakage_counts'].get('target_in_search', 0):,}",
        "",
        "## R093r-1/2/3 Eval Summary",
        "",
        "| Split | Passive RR | Search RR | Oracle RR | Best fixed | Oracle-best | Transfer | Non-transfer | L1 win | L2 win |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in [("train", train_summary), ("eval", eval_summary), ("all", overall)]:
        if not row:
            continue
        md.append(
            f"| {name} | {fmt(float(row['passive_rr']))} | {fmt(float(row['search_rr']))} | {fmt(float(row['oracle_rr']))} | "
            f"{row['best_fixed_gate']} | {fmt(float(row['oracle_minus_best_fixed']))} | "
            f"{fmt(float(row['transferable_rate']))} | {fmt(float(row['nontransfer_rate']))} | "
            f"{fmt(float(row['L1_true_swap_win_rate']))} | {fmt(float(row['L2_true_swap_win_rate']))} |"
        )
    md.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if status == "R093R_PASS_ORACLE_HEADROOM":
        md.append("R093r passes the pre-training oracle gate. The next step may be a learned gate smoke, after code review.")
    elif status == "R093R_FAIL_SEARCH_NOT_IDENTIFIABLE":
        md.append("R093r finds non-transfer cases, but true search does not beat matched swaps. Do not train a learned gate; first repair search-identifiability or change data/proxy.")
    elif status == "R093R_FAIL_WEAK_NONTRANSFER":
        md.append("R093r does not find enough item-level non-transfer cases on held-out splits. Do not train a learned abstention gate on this KuaiSAR-only proxy.")
    else:
        md.append("R093r remains weak. Do not launch GPU training until the oracle headroom and swap-identifiability gates are both positive.")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
