#!/usr/bin/env python3
"""R174 category-broken and category-matched swap ladder.

This audit addresses the concern that L2 same-category swaps may hide the very
category signal being tested. It keeps the K_PROXY_R135LARGE row/scorer/candidate
contract fixed and adds a ladder:

- random_broken: same split, no shared search category;
- source_broken: same split/source, no shared search category;
- nuisance_broken: current L1-style nuisance match, no shared search category;
- same_category: current L2-style same-category match;
- same_category_source: same category plus source.

The interpretation separates category-level signal from focal-instance
specificity. A win only against category-broken swaps is not evidence that the
specific focal episode is identifiable beyond shared category.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from behavior_oracle_audit import (  # noqa: E402
    build_ranker,
    load_item_categories,
    pre_items,
    profile,
    read_jsonl,
    reciprocal_rank,
    raw_targets,
    search_items,
    stable_sample,
    uniq,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
B6 = DATA / "b6"
REPORTS = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default=str(REPORTS / "r135large_item_proxy_rows.csv"))
    p.add_argument("--examples-path", default=str(B6 / "qbridge_examples.jsonl"))
    p.add_argument("--item-metadata", default=str(DATA / "item_metadata.parquet"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--report-prefix", default="r174")
    p.add_argument("--surface-id", default="K_PROXY_R135LARGE")
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--swap-tries", type=int, default=120)
    p.add_argument("--max-pre-items", type=int, default=12)
    p.add_argument("--max-search-items", type=int, default=12)
    p.add_argument("--max-bridge-targets", type=int, default=3)
    p.add_argument("--max-candidates", type=int, default=80)
    p.add_argument("--bootstrap", type=int, default=1000)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fmt(x: float) -> str:
    return f"{x:.4f}"


def stable_order(text: str, seed: int) -> str:
    return hashlib.md5(f"{seed}:{text}".encode("utf-8")).hexdigest()


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def build_indices(profiles: list[dict[str, Any]]) -> dict[str, dict[tuple[Any, ...], list[int]]]:
    indices: dict[str, dict[tuple[Any, ...], list[int]]] = {
        "split": defaultdict(list),
        "source": defaultdict(list),
        "nuisance": defaultdict(list),
        "category": defaultdict(list),
        "category_source": defaultdict(list),
    }
    for idx, p in enumerate(profiles):
        b = p["buckets"]
        indices["split"][(b["split"],)].append(idx)
        indices["source"][(b["split"], b["source"])].append(idx)
        indices["nuisance"][(b["split"], b["source"], b["search_pos"], b["search_neg"], b["pre"], b["bridge"])].append(idx)
        for c in p["search_cats"]:
            indices["category"][(b["split"], c)].append(idx)
            indices["category_source"][(b["split"], c, b["source"])].append(idx)
    return indices


LEVELS = [
    {
        "level": "random_broken",
        "family": "category_broken",
        "index": "split",
        "match": "same split; no shared search category",
        "overlap": "forbid",
    },
    {
        "level": "source_broken",
        "family": "category_broken",
        "index": "source",
        "match": "same split/source; no shared search category",
        "overlap": "forbid",
    },
    {
        "level": "nuisance_broken",
        "family": "category_broken",
        "index": "nuisance",
        "match": "same split/source/search-count/pre/bridge buckets; no shared search category",
        "overlap": "forbid",
    },
    {
        "level": "same_category",
        "family": "category_matched",
        "index": "category",
        "match": "same split; at least one shared search category",
        "overlap": "require",
    },
    {
        "level": "same_category_source",
        "family": "category_matched",
        "index": "category_source",
        "match": "same split/source; at least one shared search category",
        "overlap": "require",
    },
]


def keys_for(level: dict[str, str], p: dict[str, Any]) -> list[tuple[Any, ...]]:
    b = p["buckets"]
    if level["index"] == "split":
        return [(b["split"],)]
    if level["index"] == "source":
        return [(b["split"], b["source"])]
    if level["index"] == "nuisance":
        return [(b["split"], b["source"], b["search_pos"], b["search_neg"], b["pre"], b["bridge"])]
    if level["index"] == "category":
        return [(b["split"], c) for c in p["search_cats"]]
    if level["index"] == "category_source":
        return [(b["split"], c, b["source"]) for c in p["search_cats"]]
    raise ValueError(level["index"])


def pick_swap(
    i: int,
    level: dict[str, str],
    profiles: list[dict[str, Any]],
    indices: dict[str, dict[tuple[Any, ...], list[int]]],
    rng: random.Random,
    tries: int,
) -> int | None:
    p = profiles[i]
    true_cats = set(p["search_cats"])
    if not true_cats:
        return None
    pools: list[int] = []
    for key in keys_for(level, p):
        pools.extend(indices[level["index"]].get(key, []))
    if not pools:
        return None
    pools = sorted(set(pools), key=lambda j: stable_order(str(profiles[j]["example_id"]), 17))
    target_ids = set(p["raw_targets"])
    for _ in range(tries):
        j = rng.choice(pools)
        if j == i:
            continue
        q = profiles[j]
        if q["split"] != p["split"]:
            continue
        if q["query_id"] == p["query_id"] or q["user_id"] == p["user_id"]:
            continue
        if target_ids & set(q.get("search_pos_items", [])):
            continue
        q_cats = set(q["search_cats"])
        if not q_cats:
            continue
        overlap = bool(true_cats & q_cats)
        if level["overlap"] == "forbid" and overlap:
            continue
        if level["overlap"] == "require" and not overlap:
            continue
        return j
    return None


def wtl_stats(vals: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float]:
    eps = 1e-12

    def stats(x: np.ndarray) -> np.ndarray:
        strict_win = float((x > eps).mean())
        tie = float((np.abs(x) <= eps).mean())
        loss = float((x < -eps).mean())
        half_tie = strict_win + 0.5 * tie
        mean_delta = float(x.mean())
        return np.asarray([strict_win, tie, loss, half_tie, mean_delta], dtype=float)

    if vals.size == 0:
        return {k: math.nan for k in [
            "strict_win", "strict_win_ci_low", "strict_win_ci_high",
            "tie", "tie_ci_low", "tie_ci_high",
            "loss", "loss_ci_low", "loss_ci_high",
            "half_tie", "half_tie_ci_low", "half_tie_ci_high",
            "mean_delta", "mean_delta_ci_low", "mean_delta_ci_high",
        ]}
    est = stats(vals)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, vals.size, size=vals.size)
        boots.append(stats(vals[idx]))
    arr = np.vstack(boots)
    names = ["strict_win", "tie", "loss", "half_tie", "mean_delta"]
    out: dict[str, float] = {}
    for col, name in enumerate(names):
        out[name] = float(est[col])
        out[f"{name}_ci_low"] = float(np.quantile(arr[:, col], 0.025))
        out[f"{name}_ci_high"] = float(np.quantile(arr[:, col], 0.975))
    return out


def overlap_rate(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def fast_candidate_set(
    row: dict[str, Any],
    targets: list[int],
    global_top30: list[int],
    args: argparse.Namespace,
) -> list[int]:
    """Equivalent to R093r candidate_set, with global-pop precomputed once."""
    candidates = uniq(
        targets
        + pre_items(row, args.max_pre_items)
        + search_items(row, args.max_search_items)
        + global_top30
    )
    if len(candidates) > args.max_candidates:
        keep = set(targets)
        rest = [x for x in candidates if x not in keep]
        candidates = uniq(list(keep) + stable_sample(rest, args.max_candidates - len(keep), str(row.get("example_id"))))
    return candidates


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# R174 Swap Ladder and Category-Broken Controls",
        "",
        f"Status: **{payload['status']}**",
        "",
        f"Surface ID: `{payload['surface_id']}`",
        "",
        "## Question",
        "",
        "Does the L2 same-category counterfactual hide focal-search signal by "
        "replacing the focal search with an almost equivalent category signal?",
        "",
        "## Ladder Summary",
        "",
        "| Level | Family | Match | Matched | Coverage | Strict win | Tie | Half-tie | Half-tie 95% CI | Mean true-swap | Category overlap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["summary_rows"]:
        lines.append(
            f"| {row['level']} | {row['family']} | {row['match']} | {row['matched_rows']:,} | "
            f"{fmt(row['match_rate'])} | {fmt(row['strict_win'])} | {fmt(row['tie'])} | "
            f"{fmt(row['half_tie'])} | [{fmt(row['half_tie_ci_low'])}, {fmt(row['half_tie_ci_high'])}] | "
            f"{fmt(row['mean_delta'])} | {fmt(row['mean_category_jaccard'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    broken = [r for r in payload["summary_rows"] if r["family"] == "category_broken"]
    matched = [r for r in payload["summary_rows"] if r["family"] == "category_matched"]
    best_broken = max(broken, key=lambda r: float(r["half_tie"]), default={})
    best_matched = max(matched, key=lambda r: float(r["half_tie"]), default={})
    lines.append(
        f"- Best category-broken half-tie: `{best_broken.get('level', '')}` = {fmt(float(best_broken.get('half_tie', 0.0)))}."
    )
    lines.append(
        f"- Best category-matched half-tie: `{best_matched.get('level', '')}` = {fmt(float(best_matched.get('half_tie', 0.0)))}."
    )
    lines.append("- Category-broken wins, if any, should be read as category-level signal, not focal-instance specificity.")
    lines.append("- Same-category rows remain the stricter focal-instance test because the replacement preserves coarse category evidence.")
    if payload["status"] == "R174_CATEGORY_LADDER_CLOSED_INSTANCE_NOT_ESTABLISHED":
        lines.append("- The ladder supports the current interpretation: category-level evidence can move, but focal-instance specificity is not established under same-category replacement.")
    elif payload["status"] == "R174_CATEGORY_LADDER_WARN_SAME_CATEGORY_POSITIVE":
        lines.append("- Same-category replacement shows a weak positive same-category readout; the paper must revise any absolute negative G3 interpretation before integration.")
    else:
        lines.append("- The ladder is inconclusive because category-broken or category-matched coverage is too low.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    boot_rng = np.random.default_rng(args.seed)
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    scored_rows = read_csv(Path(args.rows))
    eval_scored = [r for r in scored_rows if r["split"] in {"validation", "test"}]
    wanted_ids = {r["example_id"] for r in scored_rows}
    raw_rows_all = read_jsonl(Path(args.examples_path))
    raw_by_id = {str(r["example_id"]): r for r in raw_rows_all if str(r["example_id"]) in wanted_ids}
    train_rows = [r for r in raw_rows_all if str(r.get("split")) == "train"]
    item_cat = load_item_categories(Path(args.item_metadata))
    ranker = build_ranker(train_rows, args)
    global_top30 = [int(x) for x, _ in ranker.global_pop.most_common(30)]

    ordered_scored = [r for r in scored_rows if r["example_id"] in raw_by_id]
    profiles = [profile(raw_by_id[r["example_id"]], item_cat, args) for r in ordered_scored]
    profile_by_id = {p["example_id"]: idx for idx, p in enumerate(profiles)}
    scored_by_id = {r["example_id"]: r for r in ordered_scored}
    indices = build_indices(profiles)

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    eligible_eval = [r for r in eval_scored if r["example_id"] in profile_by_id and profiles[profile_by_id[r["example_id"]]]["search_cats"]]
    prepared: dict[str, dict[str, Any]] = {}
    for scored in eligible_eval:
        p = profiles[profile_by_id[scored["example_id"]]]
        raw = raw_by_id[p["example_id"]]
        targets = p["raw_targets"]
        if not targets:
            continue
        prepared[p["example_id"]] = {
            "targets": targets,
            "candidates": fast_candidate_set(raw, targets, global_top30, args),
            "true_rr": f(scored_by_id[p["example_id"]], "search_rr"),
        }

    for level in LEVELS:
        deltas: list[float] = []
        cat_jaccards: list[float] = []
        matched = 0
        attempted = 0
        for scored in eligible_eval:
            p_idx = profile_by_id[scored["example_id"]]
            p = profiles[p_idx]
            prep = prepared.get(p["example_id"])
            if prep is None:
                continue
            attempted += 1
            j = pick_swap(p_idx, level, profiles, indices, rng, args.swap_tries)
            if j is None:
                continue
            q = profiles[j]
            targets = prep["targets"]
            candidates = prep["candidates"]
            swap_rr, _, _ = reciprocal_rank(ranker, q["search_items"], targets, candidates)
            true_rr = prep["true_rr"]
            delta = true_rr - swap_rr
            matched += 1
            deltas.append(delta)
            cat_j = overlap_rate(set(p["search_cats"]), set(q["search_cats"]))
            cat_jaccards.append(cat_j)
            detail_rows.append({
                "surface_id": args.surface_id,
                "level": level["level"],
                "family": level["family"],
                "example_id": p["example_id"],
                "swap_example_id": q["example_id"],
                "split": p["split"],
                "source": p["source"],
                "true_search_rr": true_rr,
                "swap_rr": swap_rr,
                "true_minus_swap": delta,
                "true_search_cat_count": len(p["search_cats"]),
                "swap_search_cat_count": len(q["search_cats"]),
                "category_jaccard": cat_j,
            })
        vals = np.asarray(deltas, dtype=float)
        stats = wtl_stats(vals, boot_rng, args.bootstrap)
        summary_rows.append({
            "surface_id": args.surface_id,
            "level": level["level"],
            "family": level["family"],
            "match": level["match"],
            "attempted_rows": attempted,
            "matched_rows": matched,
            "match_rate": matched / attempted if attempted else 0.0,
            "mean_category_jaccard": float(np.mean(cat_jaccards)) if cat_jaccards else math.nan,
            **stats,
        })

    best_same = max((r for r in summary_rows if r["family"] == "category_matched"), key=lambda r: float(r["half_tie"]), default={})
    min_coverage = min((r["match_rate"] for r in summary_rows), default=0.0)
    if min_coverage < 0.20:
        status = "R174_CATEGORY_LADDER_WARN_LOW_COVERAGE"
    elif best_same and float(best_same["half_tie_ci_low"]) > 0.500:
        status = "R174_CATEGORY_LADDER_WARN_SAME_CATEGORY_POSITIVE"
    else:
        status = "R174_CATEGORY_LADDER_CLOSED_INSTANCE_NOT_ESTABLISHED"

    paths = {
        "summary_csv": reports / f"{args.report_prefix}_swap_ladder_metrics.csv",
        "detail_csv": reports / f"{args.report_prefix}_swap_ladder_detail.csv",
        "json": reports / f"{args.report_prefix}_swap_ladder_category_broken.json",
        "md": reports / f"{args.report_prefix}_swap_ladder_category_broken.md",
    }
    payload = {
        "schema": "focalswap.swap_ladder_category_broken.v0.1",
        "status": status,
        "surface_id": args.surface_id,
        "inputs": {
            "rows": str(Path(args.rows).relative_to(ROOT) if Path(args.rows).is_absolute() else args.rows),
            "examples_path": str(Path(args.examples_path).relative_to(ROOT) if Path(args.examples_path).is_absolute() else args.examples_path),
            "item_metadata": str(Path(args.item_metadata).relative_to(ROOT) if Path(args.item_metadata).is_absolute() else args.item_metadata),
            "eligible_eval_rows_with_search_category": len(eligible_eval),
            "swap_tries": args.swap_tries,
            "seed": args.seed,
        },
        "summary_rows": summary_rows,
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
    }
    write_csv(paths["summary_csv"], summary_rows)
    write_csv(paths["detail_csv"], detail_rows)
    write_json(paths["json"], payload)
    write_md(paths["md"], payload)
    print(json.dumps({"status": status, "report": str(paths["md"])}, indent=2))


if __name__ == "__main__":
    main()
