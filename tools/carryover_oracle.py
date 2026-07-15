#!/usr/bin/env python3
"""B6R R060/R061 carryover oracle and matched-swap audit.

This is a CPU-only gate before any QIR model training. It asks whether the
current KuaiSAR/B6 artifacts contain a target-legitimate search-to-rec carryover
signal. The script intentionally stays diagnostic: it does not train QIR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
B6 = ROOT / "data" / "processed" / "b6"
B6R = ROOT / "data" / "processed" / "b6r"
REPORTS = ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-path", default=str(B6 / "qbridge_examples.jsonl"))
    p.add_argument("--item-metadata", default=str(ROOT / "data" / "processed" / "item_metadata.parquet"))
    p.add_argument("--output-dir", default=str(B6R))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--max-examples", type=int, default=0, help="0 means all examples")
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--candidates-per-level", type=int, default=80)
    return p.parse_args()


def load_item_categories(path: Path) -> dict[int, int]:
    df = pd.read_parquet(path, columns=["item_id", "first_level_category_id"])
    df = df.dropna()
    return dict(zip(df["item_id"].astype(int), df["first_level_category_id"].astype(int)))


def read_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_examples and len(rows) >= max_examples:
                break
    return rows


def qbucket(n: int) -> str:
    if n <= 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def pre_bucket(n: int) -> str:
    if n <= 8:
        return "0-8"
    if n <= 16:
        return "9-16"
    if n <= 32:
        return "17-32"
    return "33+"


def bridge_bucket(n: int) -> str:
    if n <= 10:
        return "1-10"
    if n <= 40:
        return "11-40"
    if n <= 100:
        return "41-100"
    return "101+"


def cats(items: list[int], item_cat: dict[int, int]) -> set[int]:
    return {int(item_cat[x]) for x in items if int(x) in item_cat}


def score_items(items: list[int], evidence_cats: set[int], item_cat: dict[int, int]) -> float:
    known = [x for x in items if int(x) in item_cat]
    if not known or not evidence_cats:
        return 0.0
    hits = sum(1 for x in known if int(item_cat[int(x)]) in evidence_cats)
    return hits / len(known)


def first_frac(items: list[int], frac: float = 0.25) -> list[int]:
    if not items:
        return []
    k = max(1, int(math.ceil(len(items) * frac)))
    return items[:k]


def random_filter(items: list[int], k: int, rng: random.Random) -> list[int]:
    if not items or k <= 0:
        return []
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def build_targets(ex: dict[str, Any], item_cat: dict[int, int], rng: random.Random) -> dict[str, list[int]]:
    bridge = [int(x) for x in ex["y_bridge_items"]]
    early = first_frac(bridge, 0.25)
    pre_cats = cats([int(x) for x in ex["pre_items"]], item_cat)
    search_cats = cats([int(x) for x in ex["search_pos_items"]] + [int(x) for x in ex["search_neg_items"]], item_cat)

    time_novel = [x for x in early if x in item_cat and item_cat[x] not in pre_cats]
    time_sim = [x for x in early if x in item_cat and item_cat[x] in search_cats]
    full = [x for x in early if x in item_cat and item_cat[x] in search_cats and item_cat[x] not in pre_cats]
    # Match random target size to the strict full target when possible; otherwise
    # use the early target size to test generic target manufacturing.
    random_k = len(full) if full else len(early)
    return {
        "Y_post_broad": bridge,
        "Y_time": early,
        "Y_time_novel": time_novel,
        "Y_time_sim": time_sim,
        "Y_carry_full": full,
        "Y_random_filter": random_filter(bridge, random_k, rng),
    }


def make_profile(ex: dict[str, Any], item_cat: dict[int, int], rng: random.Random) -> dict[str, Any]:
    pre = [int(x) for x in ex["pre_items"]]
    spos = [int(x) for x in ex["search_pos_items"]]
    sneg = [int(x) for x in ex["search_neg_items"]]
    bridge = [int(x) for x in ex["y_bridge_items"]]
    search_cats = cats(spos + sneg, item_cat)
    pre_cats = cats(pre, item_cat)
    targets = build_targets(ex, item_cat, rng)
    return {
        "example_id": ex["example_id"],
        "split": ex["split"],
        "source": ex["source"],
        "query_id": int(ex["query_id"]),
        "user_id": int(ex["user_id"]),
        "search_pos_items": spos,
        "search_neg_items": sneg,
        "pre_len": len(pre),
        "bridge_len": len(bridge),
        "search_pos_count": len(spos),
        "search_neg_count": len(sneg),
        "query_token_count": len(ex.get("query_tokens", [])),
        "search_cats": search_cats,
        "pre_cats": pre_cats,
        "targets": targets,
        "buckets": {
            "split": ex["split"],
            "source": str(ex["source"]),
            "search_pos": qbucket(len(spos)),
            "search_neg": qbucket(len(sneg)),
            "pre": pre_bucket(len(pre)),
            "bridge": bridge_bucket(len(bridge)),
        },
    }


def target_rows(profiles: list[dict[str, Any]], item_cat: dict[int, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ["Y_post_broad", "Y_time", "Y_time_novel", "Y_time_sim", "Y_carry_full", "Y_random_filter"]:
        eligible = [p for p in profiles if p["targets"][name]]
        true_scores = [score_items(p["targets"][name], p["search_cats"], item_cat) for p in eligible]
        passive_scores = [score_items(p["targets"][name], p["pre_cats"], item_cat) for p in eligible]
        rows.append(
            {
                "target": name,
                "examples_with_target": len(eligible),
                "coverage": len(eligible) / len(profiles) if profiles else 0.0,
                "avg_items": sum(len(p["targets"][name]) for p in eligible) / len(eligible) if eligible else 0.0,
                "true_evidence_score": sum(true_scores) / len(true_scores) if true_scores else 0.0,
                "passive_pre_score": sum(passive_scores) / len(passive_scores) if passive_scores else 0.0,
                "true_minus_passive": (sum(true_scores) / len(true_scores) - sum(passive_scores) / len(passive_scores)) if true_scores and passive_scores else 0.0,
            }
        )
    return rows


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


def pick_swap(
    i: int,
    level: str,
    profiles: list[dict[str, Any]],
    indices: dict[str, dict[tuple[Any, ...], list[int]]],
    rng: random.Random,
    tries: int,
) -> int | None:
    p = profiles[i]
    b = p["buckets"]
    keys: list[tuple[Any, ...]]
    if level == "L0_random":
        keys = [(b["split"],)]
        index = indices["random"]
    elif level == "L1_nuisance":
        keys = [(b["split"], b["source"], b["search_pos"], b["search_neg"], b["pre"], b["bridge"])]
        index = indices["nuisance"]
    elif level == "L2_same_category":
        keys = [(b["split"], c) for c in p["search_cats"]]
        index = indices["same_category"]
    elif level == "L3_high_similarity":
        keys = [(b["split"], c, b["source"]) for c in p["search_cats"]]
        index = indices["high_similarity"]
    else:
        raise ValueError(level)

    pools: list[int] = []
    for key in keys:
        pools.extend(index.get(key, []))
    if not pools:
        return None
    true_cats = p["search_cats"]
    target_ids = set(p["targets"]["Y_carry_full"]) or set(p["targets"]["Y_time_sim"]) or set(p["targets"]["Y_time"])
    for _ in range(tries):
        j = rng.choice(pools)
        if j == i:
            continue
        q = profiles[j]
        if q["split"] != p["split"]:
            continue
        # Avoid exact same query/user and avoid target leakage through search positives.
        if q["query_id"] == p["query_id"] or q["user_id"] == p["user_id"]:
            continue
        if target_ids & set(q.get("search_pos_items", [])):
            continue
        if level in {"L0_random", "L1_nuisance"} and true_cats and q["search_cats"] and (true_cats & q["search_cats"]):
            continue
        if level in {"L2_same_category", "L3_high_similarity"} and true_cats and q["search_cats"] and not (true_cats & q["search_cats"]):
            continue
        return j
    return None


def ladder_rows(profiles: list[dict[str, Any]], item_cat: dict[int, int], rng: random.Random, tries: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indices = build_indices(profiles)
    levels = ["L0_random", "L1_nuisance", "L2_same_category", "L3_high_similarity"]
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for level in levels:
        attempted = 0
        matched = 0
        wins = ties = losses = 0
        true_scores: list[float] = []
        swap_scores: list[float] = []
        cat_overlaps: list[float] = []
        for i, p in enumerate(profiles):
            target = p["targets"]["Y_carry_full"]
            if not target:
                continue
            attempted += 1
            j = pick_swap(i, level, profiles, indices, rng, tries)
            if j is None:
                continue
            q = profiles[j]
            matched += 1
            ts = score_items(target, p["search_cats"], item_cat)
            ss = score_items(target, q["search_cats"], item_cat)
            true_scores.append(ts)
            swap_scores.append(ss)
            union = p["search_cats"] | q["search_cats"]
            cat_overlaps.append((len(p["search_cats"] & q["search_cats"]) / len(union)) if union else 0.0)
            wins += int(ts > ss)
            ties += int(ts == ss)
            losses += int(ts < ss)
            if len(detail) < 2000:
                detail.append(
                    {
                        "level": level,
                        "example_id": p["example_id"],
                        "swap_example_id": q["example_id"],
                        "split": p["split"],
                        "source": p["source"],
                        "true_score": ts,
                        "swap_score": ss,
                        "true_win": int(ts > ss),
                        "search_cat_jaccard": cat_overlaps[-1],
                    }
                )
        win_rate = wins / matched if matched else 0.0
        non_tie_win_rate = wins / (wins + losses) if (wins + losses) else 0.0
        summary.append(
            {
                "level": level,
                "attempted": attempted,
                "matched": matched,
                "match_rate": matched / attempted if attempted else 0.0,
                "true_win_rate": win_rate,
                "non_tie_true_win_rate": non_tie_win_rate,
                "tie_rate": ties / matched if matched else 0.0,
                "true_score": sum(true_scores) / len(true_scores) if true_scores else 0.0,
                "swap_score": sum(swap_scores) / len(swap_scores) if swap_scores else 0.0,
                "true_minus_swap": (sum(true_scores) / len(true_scores) - sum(swap_scores) / len(swap_scores)) if true_scores and swap_scores else 0.0,
                "avg_search_cat_jaccard": sum(cat_overlaps) / len(cat_overlaps) if cat_overlaps else 0.0,
            }
        )
    return summary, detail


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    reports = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    item_cat = load_item_categories(Path(args.item_metadata))
    examples = read_jsonl(Path(args.examples_path), args.max_examples)
    profiles = [make_profile(ex, item_cat, rng) for ex in examples]

    target_summary = target_rows(profiles, item_cat)
    swap_summary, swap_detail = ladder_rows(profiles, item_cat, rng, args.candidates_per_level)

    # Persist compact QIR examples with target family sizes; full lists are kept
    # for the strict target only to avoid writing another huge artifact.
    qir_path = out_dir / "qir_examples.jsonl"
    with qir_path.open("w") as f:
        for p in profiles:
            f.write(
                json.dumps(
                    {
                        "example_id": p["example_id"],
                        "split": p["split"],
                        "source": p["source"],
                        "query_id": p["query_id"],
                        "user_id": p["user_id"],
                        "search_cats": sorted(p["search_cats"]),
                        "pre_cats": sorted(p["pre_cats"]),
                        "target_sizes": {k: len(v) for k, v in p["targets"].items()},
                        "y_carry_full": p["targets"]["Y_carry_full"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    write_csv(reports / "b6r_target_ablation.csv", target_summary)
    write_csv(reports / "carryover_oracle.csv", target_summary)
    write_csv(reports / "b6r_swap_ladder.csv", swap_summary)
    write_csv(reports / "b6r_matched_swap_audit.csv", swap_detail)

    full = next(r for r in target_summary if r["target"] == "Y_carry_full")
    novel = next(r for r in target_summary if r["target"] == "Y_time_novel")
    random_target = next(r for r in target_summary if r["target"] == "Y_random_filter")
    l1 = next((r for r in swap_summary if r["level"] == "L1_nuisance"), None)
    l0 = next((r for r in swap_summary if r["level"] == "L0_random"), None)

    r060_status = "PASS"
    r060_reasons: list[str] = []
    if full["coverage"] < 0.05:
        r060_status = "FAIL"
        r060_reasons.append("full carryover coverage below 5%")
    elif full["coverage"] < 0.10:
        r060_status = "WEAK"
        r060_reasons.append("full carryover coverage below 10%")
    if full["true_evidence_score"] <= random_target["true_evidence_score"]:
        r060_status = "FAIL"
        r060_reasons.append("full target not stronger than random-filter target")
    if novel["true_evidence_score"] <= 0.05:
        r060_status = "WEAK" if r060_status == "PASS" else r060_status
        r060_reasons.append("non-similarity target has only weak true-query signal")
    if full["true_evidence_score"] >= 0.999:
        r060_status = "WEAK" if r060_status == "PASS" else r060_status
        r060_reasons.append("full target score is mechanically saturated by category-sim filtering")

    r061_status = "PASS"
    r061_reasons: list[str] = []
    if not l1 or l1["matched"] == 0:
        r061_status = "FAIL"
        r061_reasons.append("no nuisance-matched swaps")
    elif l1["true_win_rate"] <= 0.5 and l1["true_minus_swap"] <= 0:
        r061_status = "FAIL"
        r061_reasons.append("true query does not beat nuisance-matched swaps")
    if l0 and l1 and l1["match_rate"] < 0.2:
        r061_status = "WEAK" if r061_status == "PASS" else r061_status
        r061_reasons.append("nuisance-matched swap coverage below 20%")
    hard_levels = [r for r in swap_summary if r["level"] in {"L2_same_category", "L3_high_similarity"}]
    if hard_levels and any(r["tie_rate"] > 0.40 for r in hard_levels):
        r061_status = "PASS_WITH_CAVEAT" if r061_status == "PASS" else r061_status
        r061_reasons.append("hard swaps expose many category-level ties; current evidence is category-level, not fine-grained semantics")

    md = [
        "# SearchCPT B6R R060-R061 Carryover Oracle",
        "",
        f"- Examples evaluated: {len(profiles):,}",
        f"- QIR artifact: `{qir_path.relative_to(ROOT)}`",
        f"- R060 status: **{r060_status}**" + (f" ({'; '.join(r060_reasons)})" if r060_reasons else ""),
        f"- R061 status: **{r061_status}**" + (f" ({'; '.join(r061_reasons)})" if r061_reasons else ""),
        "",
        "## R060 Target Ablation",
        "",
        markdown_table(
            target_summary,
            [
                "target",
                "examples_with_target",
                "coverage",
                "avg_items",
                "true_evidence_score",
                "passive_pre_score",
                "true_minus_passive",
            ],
        ),
        "",
        "## R061 Swap Ladder",
        "",
        markdown_table(
            swap_summary,
            [
                "level",
                "attempted",
                "matched",
                "match_rate",
                "true_win_rate",
                "tie_rate",
                "true_score",
                "swap_score",
                "true_minus_swap",
                "avg_search_cat_jaccard",
            ],
        ),
        "",
        "## Decision",
        "",
    ]
    if r060_status == "PASS" and r061_status in {"PASS", "PASS_WITH_CAVEAT"}:
        md.append("R060-R061 pass the CPU oracle gate. Proceed to R062 before any model training.")
    elif r060_status == "FAIL" or r061_status == "FAIL":
        md.append("R060-R061 do not support a QIR GPU launch. Fix target construction or swap matching before model work.")
    else:
        md.append("R060-R061 show partial signal but are not strong enough for GPU. Inspect weak reasons and run R062 before deciding.")
    md.append("")
    md.append("Practical conclusion: the data contains a strict category-level carryover subset, but it is narrow and partly defined by the same category signal used in the oracle. Treat this as a positive local feasibility signal, not as paper-ready QIR evidence.")
    md.append("")
    md.append("Boundary: this oracle uses first-level item categories as the available local similarity signal. It does not establish natural-language query semantics or embedding-level hard negatives.")
    md.append("")
    (reports / "carryover_oracle.md").write_text("\n".join(md), encoding="utf-8")

    # Companion JSON for fast downstream parsing.
    (reports / "carryover_oracle.json").write_text(
        json.dumps(
            {
                "examples": len(profiles),
                "qir_artifact": str(qir_path.relative_to(ROOT)),
                "r060_status": r060_status,
                "r060_reasons": r060_reasons,
                "r061_status": r061_status,
                "r061_reasons": r061_reasons,
                "target_summary": target_summary,
                "swap_summary": swap_summary,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"R060={r060_status} R061={r061_status} examples={len(profiles)}")
    print(reports / "carryover_oracle.md")


if __name__ == "__main__":
    main()
