#!/usr/bin/env python3
"""R161 SESRec official checkpoint five-view readout.

This is the SESRec counterpart of R136.  It uses the official SESRec model
definition and an optional official checkpoint, but scores SARCheck five-view
inputs constructed from the same split-complete R158 bundle contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
REPORTS = ROOT / "reports"
EXAMPLES = ROOT / "data/processed/backbone_audit/examples.jsonl"
CONTRACT_ID = "R161_SESREC_OFFICIAL_FIVE_VIEW"
R172_CONTRACT_ID = "R172_CHECKPOINT_REFRESH"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories  # noqa: E402
from transition_backbone import (  # noqa: E402
    build_swap_indices,
    build_swap_profiles,
    pick_swap,
)
from unisar_preflight import (  # noqa: E402
    build_swap_lookups as build_extended_swap_lookups,
)
from official_adapter import (  # noqa: E402
    CompactMaps,
    compact_id,
    compact_item_sequence,
    compact_query_tokens,
    fixed_negs,
    get_git_commit,
    reset_imports,
    restore_imports,
    row_user_key,
    sesrec_item_tensors,
    sesrec_source_id,
    write_csv,
    write_json,
    write_md,
)
from controlled_view_ranking import (  # noqa: E402
    midpoint_rr_from_scores,
    rank_rr_identity_pass,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--examples-path", default=str(EXAMPLES))
    p.add_argument("--item-metadata", default=str(ROOT / "data/processed/item_metadata.parquet"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--row-limit", type=int, default=9000)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--neg-k", type=int, default=99)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-eval-rows", type=int, default=2000)
    p.add_argument("--swap-tries", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--report-prefix", default="r161_sesrec_official")
    p.add_argument(
        "--surface-id",
        default="",
        help="Optional registered evidence-surface identifier written to the readout manifest.",
    )
    p.add_argument("--candidate-expanded", action="store_true")
    p.add_argument("--contract-check-only", action="store_true")
    p.add_argument(
        "--eval-candidates-only",
        action="store_true",
        help=(
            "Build fixed candidate lists only for evaluated rows. "
            "Use this for full-universe readouts where train/validation rows "
            "are needed for compact maps but are not scored."
        ),
    )
    p.add_argument("--progress-every", type=int, default=10)
    p.add_argument(
        "--candidate-ledger",
        default="",
        help="Optional R225 canonical raw-item candidate ledger.",
    )
    p.add_argument(
        "--extended-swap-ladder",
        action="store_true",
        help="Also score category-broken/source and same-category/source swaps.",
    )
    p.add_argument(
        "--score-level-half-tie",
        action="store_true",
        help="Define swap win/tie/loss from target-score differences instead of RR differences.",
    )
    p.add_argument(
        "--swap-ledger",
        default="",
        help="Optional frozen R225 four-level swap-donor ledger.",
    )
    p.add_argument(
        "--finite-neutral-controls",
        action="store_true",
        help="Use non-padding neutral history/source ids for finite masked controls.",
    )
    return p.parse_args()


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def jsonl_row_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else math.nan


def emit_progress(stage: str, **fields: Any) -> None:
    print(json.dumps({"progress": stage, **fields}, sort_keys=True), flush=True)


def frac(num: int, den: int) -> float:
    return float(num / den) if den else math.nan


def split_indices(selected: list[dict[str, Any]], split: str) -> list[int]:
    idx = [i for i, ex in enumerate(selected) if str(ex.get("split", "train")) == split]
    return idx if idx else list(range(len(selected)))


def split_counts(selected: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "test": 0}
    for ex in selected:
        key = str(ex.get("split", ""))
        if key in counts:
            counts[key] += 1
        else:
            counts[key] = counts.get(key, 0) + 1
    return counts


def validate_r172_contract(
    args: argparse.Namespace,
    examples_path: Path,
    selected: list[dict[str, Any]],
    candidates: list[list[int]],
    eval_indices: list[int],
) -> None:
    if not str(args.report_prefix).startswith("r172_"):
        return
    canonical = (ROOT / "data/processed/backbone_audit/examples.jsonl").resolve()
    actual = examples_path.resolve()
    if actual != canonical:
        raise ValueError(
            "R172_CHECKPOINT_REFRESH requires exact K_CHECKPOINT_R172_9K examples path: "
            f"{canonical}"
        )
    if jsonl_row_count(actual) != 9000:
        raise ValueError("R172_CHECKPOINT_REFRESH requires exactly 9000 non-empty JSONL rows")
    if args.row_limit != 9000 or args.max_eval_rows != 3000:
        raise ValueError(
            "R172_CHECKPOINT_REFRESH requires K_CHECKPOINT_R172_9K: "
            "row_limit=9000 and max_eval_rows=3000"
        )
    if args.split != "test":
        raise ValueError("R172_CHECKPOINT_REFRESH requires split=test")
    if args.neg_k != 99:
        raise ValueError("R172_CHECKPOINT_REFRESH requires neg_k=99 for the 100-candidate contract")
    counts = split_counts(selected)
    expected_counts = {"train": 3000, "validation": 3000, "test": 3000}
    if counts != expected_counts:
        raise ValueError(
            "R172_CHECKPOINT_REFRESH requires split counts "
            f"{expected_counts}, got {counts}"
        )
    if len(eval_indices) != 3000:
        raise ValueError("R172_CHECKPOINT_REFRESH requires exactly 3000 evaluated test rows")
    candidate_counts = {len(c) for c in candidates}
    if candidate_counts != {100}:
        raise ValueError(
            "R172_CHECKPOINT_REFRESH requires exactly 100 candidates per row, "
            f"got candidate counts {sorted(candidate_counts)}"
        )


def build_maps(
    selected: list[dict[str, Any]],
    candidate_raw_by_example: dict[str, list[int]] | None = None,
) -> CompactMaps:
    maps = CompactMaps(user={}, item={}, query_token={}, session={})
    for ex in selected:
        compact_id(maps.user, row_user_key(ex))
        for raw_item in ex.get("H_pre", []) + ex.get("E_search_all", []) + ex.get("Y_post_rec", []):
            compact_id(maps.item, int(raw_item))
        if candidate_raw_by_example is not None:
            example_id = str(ex["example_id"])
            if example_id not in candidate_raw_by_example:
                raise ValueError(f"candidate ledger missing example {example_id}")
            for raw_item in candidate_raw_by_example[example_id]:
                compact_id(maps.item, int(raw_item))
        for raw_tok in ex.get("query_tokens", []):
            compact_id(maps.query_token, int(raw_tok))
        compact_id(maps.session, ex["example_id"])
    return maps


def build_candidate_lists(
    selected: list[dict[str, Any]],
    maps: CompactMaps,
    neg_k: int,
    seed: int,
    row_indices: list[int] | None = None,
) -> list[list[int]]:
    rng = random.Random(seed)
    all_items = sorted(maps.item.values())
    neutral_item = (max(all_items) if all_items else 1) + 1
    if row_indices is None:
        row_indices = list(range(len(selected)))
    candidates: list[list[int]] = [[] for _ in selected]
    for row_idx in row_indices:
        ex = selected[row_idx]
        target = maps.item[int(ex["primary_target_item"])]
        search_pool = [maps.item[int(x)] for x in ex.get("E_search_all", []) if int(x) in maps.item]
        hist_pool = [maps.item[int(x)] for x in ex.get("H_pre", []) if int(x) in maps.item]
        # Match build_sesrec_trainer_bundle exactly: validation/test rows use
        # search pool, history pool, then all compact items, with neutral only
        # as final fallback.  Do not add Y_post_rec here or the checkpoint
        # readout scores a different candidate contract than R158 training.
        negs = fixed_negs(target, [search_pool, hist_pool, all_items], all_items + [neutral_item], neg_k, rng)
        candidates[row_idx] = [target] + negs
    return candidates


def map_canonical_candidate_lists(
    selected: list[dict[str, Any]],
    maps: CompactMaps,
    candidate_raw_by_example: dict[str, list[int]],
    neg_k: int,
    row_indices: list[int] | None = None,
) -> list[list[int]]:
    if row_indices is None:
        row_indices = list(range(len(selected)))
    candidates: list[list[int]] = [[] for _ in selected]
    for row_idx in row_indices:
        ex = selected[row_idx]
        raw_candidates = candidate_raw_by_example[str(ex["example_id"])]
        target = int(ex["primary_target_item"])
        if len(raw_candidates) != neg_k + 1 or int(raw_candidates[0]) != target:
            raise ValueError(f"invalid canonical candidates for {ex['example_id']}")
        if len(set(int(item) for item in raw_candidates)) != len(raw_candidates):
            raise ValueError(f"duplicate canonical candidates for {ex['example_id']}")
        candidates[row_idx] = [maps.item[int(item)] for item in raw_candidates]
    return candidates


def rr_from_scores(scores: torch.Tensor) -> tuple[list[float], list[float]]:
    """Tie-aware midpoint rank and RR with target in column zero.

    Official SESRec test groups place the positive item first. A strict
    greater-than rank would turn all-equal scores into rank 1 and inflate
    zero-input control views. Exact ties therefore receive their midpoint
    rank, matching the UniSAR controlled-view evaluator.
    """
    return midpoint_rr_from_scores(scores)


def ndcg_at_k_from_scores(scores: torch.Tensor, k: int = 10) -> list[float]:
    """Tie-aware expected NDCG@k for one positive candidate in column zero."""
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite scores cannot be evaluated")
    values: list[float] = []
    for row in scores.detach().cpu():
        target_score = row[0]
        greater = int((row > target_score).sum().item())
        tied = int((row == target_score).sum().item())
        gains = [
            1.0 / math.log2(rank + 1)
            for rank in range(greater + 1, greater + tied + 1)
            if rank <= k
        ]
        values.append(sum(gains) / max(tied, 1))
    return values


def view_parts(
    ex: dict[str, Any],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    swap_ex: dict[str, Any] | None = None,
    finite_neutral_controls: bool = False,
) -> tuple[list[int], list[int], list[int], int, bool, int | None]:
    max_click = 5
    max_words = 6
    hist = compact_item_sequence(maps, ex.get("H_pre", []), 50, neutral_item)
    src_ex = swap_ex if swap_ex is not None else ex
    search_items = compact_item_sequence(maps, src_ex.get("E_search_all", []), max_click, neutral_item)
    q_tokens = compact_query_tokens(maps, src_ex.get("query_tokens", []), max_words)
    source_id = sesrec_source_id(str(src_ex.get("source", "USER_INPUT")))

    if view == "search_only":
        hist = [neutral_item] if finite_neutral_controls else []
    if view in {"history_only", "search_masked"}:
        if finite_neutral_controls:
            neutral_session = (max(maps.session.values()) if maps.session else 0) + 1
            return hist, [neutral_item], [neutral_query], 1, True, neutral_session
        return hist, [], [], 0, False, None
    return hist, search_items, q_tokens, source_id, True, None


def build_sesrec_batch(
    selected: list[dict[str, Any]],
    row_indices: list[int],
    candidates: list[list[int]],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    swap_lookup: dict[int, int | None] | None = None,
    finite_neutral_controls: bool = False,
) -> tuple[list[Any], int, int]:
    max_rec = 150
    max_src = 25
    max_click = 5
    max_words = 6
    users_id: list[int] = []
    rec_lists: list[list[int]] = []
    src_query: list[list[int]] = []
    src_source: list[list[int]] = []
    src_click_items: list[list[list[int]]] = []
    src_words: list[list[list[int]]] = []
    flat_candidates: list[int] = []

    for idx in row_indices:
        ex = selected[idx]
        swap_ex = None
        if swap_lookup is not None:
            swap_idx = swap_lookup.get(idx)
            if swap_idx is not None:
                swap_ex = selected[int(swap_idx)]
        hist, clicks, words, source_id, has_search, q_id_override = view_parts(
            ex,
            maps,
            neutral_item,
            neutral_query,
            view,
            swap_ex,
            finite_neutral_controls,
        )
        # Preserve official SESRec sampler semantics: padding is id 0, and
        # masks are derived by `rec_his[0] == 0` and `src_his[0] == 0`.
        rec_seq = (hist[-max_rec:] + [0] * max_rec)[:max_rec]
        q_id = (
            q_id_override
            if q_id_override is not None
            else maps.session[swap_ex["example_id"] if swap_ex is not None else ex["example_id"]]
            if has_search
            else 0
        )

        for cand in candidates[idx]:
            users_id.append(maps.user[row_user_key(ex)])
            rec_lists.append(rec_seq)
            src_query.append(([q_id] if has_search else []) + [0] * (max_src - (1 if has_search else 0)))
            src_source.append(([source_id] if has_search else []) + [0] * (max_src - (1 if has_search else 0)))
            padded_clicks = (clicks + [0] * max_click)[:max_click]
            if has_search:
                src_click_items.append([padded_clicks] + [[0] * max_click for _ in range(max_src - 1)])
            else:
                src_click_items.append([[0] * max_click for _ in range(max_src)])
            padded_words = (words + [0] * max_words)[:max_words]
            if has_search:
                src_words.append([padded_words] + [[0] * max_words for _ in range(max_src - 1)])
            else:
                src_words.append([[0] * max_words for _ in range(max_src)])
            flat_candidates.append(cand)

    n = len(flat_candidates)
    rec_flat = [item for row in rec_lists for item in row]
    click_flat = [item for row in src_click_items for sess in row for item in sess]
    users = [
        torch.tensor(users_id, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
        torch.ones(n, dtype=torch.long),
    ]
    rec_tensors = sesrec_item_tensors(rec_flat, (n, max_rec))
    click_tensors = sesrec_item_tensors(click_flat, (n, max_src, max_click))
    src_his = [
        torch.tensor(src_query, dtype=torch.long),
        torch.tensor(src_source, dtype=torch.long),
        click_tensors,
        torch.tensor(src_words, dtype=torch.long),
    ]
    item_tensors = sesrec_item_tensors(flat_candidates, (n,))
    return [users, rec_tensors, src_his, item_tensors], len(row_indices), len(candidates[row_indices[0]])


def build_sesrec_grouped_batch(
    selected: list[dict[str, Any]],
    row_indices: list[int],
    candidates: list[list[int]],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    swap_lookup: dict[int, int | None] | None = None,
    finite_neutral_controls: bool = False,
) -> tuple[list[Any], list[torch.Tensor], int, int]:
    max_rec = 150
    max_src = 25
    max_click = 5
    max_words = 6
    users_id: list[int] = []
    rec_lists: list[list[int]] = []
    src_query: list[list[int]] = []
    src_source: list[list[int]] = []
    src_click_items: list[list[list[int]]] = []
    src_words: list[list[list[int]]] = []
    row_items: list[int] = []
    flat_candidates: list[int] = []

    for idx in row_indices:
        ex = selected[idx]
        swap_ex = None
        if swap_lookup is not None:
            swap_idx = swap_lookup.get(idx)
            if swap_idx is not None:
                swap_ex = selected[int(swap_idx)]
        hist, clicks, words, source_id, has_search, q_id_override = view_parts(
            ex,
            maps,
            neutral_item,
            neutral_query,
            view,
            swap_ex,
            finite_neutral_controls,
        )
        rec_seq = (hist[-max_rec:] + [0] * max_rec)[:max_rec]
        q_id = (
            q_id_override
            if q_id_override is not None
            else maps.session[swap_ex["example_id"] if swap_ex is not None else ex["example_id"]]
            if has_search
            else 0
        )

        users_id.append(maps.user[row_user_key(ex)])
        rec_lists.append(rec_seq)
        src_query.append(([q_id] if has_search else []) + [0] * (max_src - (1 if has_search else 0)))
        src_source.append(([source_id] if has_search else []) + [0] * (max_src - (1 if has_search else 0)))
        padded_clicks = (clicks + [0] * max_click)[:max_click]
        if has_search:
            src_click_items.append([padded_clicks] + [[0] * max_click for _ in range(max_src - 1)])
        else:
            src_click_items.append([[0] * max_click for _ in range(max_src)])
        padded_words = (words + [0] * max_words)[:max_words]
        if has_search:
            src_words.append([padded_words] + [[0] * max_words for _ in range(max_src - 1)])
        else:
            src_words.append([[0] * max_words for _ in range(max_src)])
        row_items.append(candidates[idx][0])
        flat_candidates.extend(candidates[idx])

    n_rows = len(row_indices)
    cand_count = len(candidates[row_indices[0]]) if row_indices else 0
    rec_flat = [item for row in rec_lists for item in row]
    click_flat = [item for row in src_click_items for sess in row for item in sess]
    users = [
        torch.tensor(users_id, dtype=torch.long),
        torch.ones(n_rows, dtype=torch.long),
        torch.ones(n_rows, dtype=torch.long),
        torch.ones(n_rows, dtype=torch.long),
    ]
    rec_tensors = sesrec_item_tensors(rec_flat, (n_rows, max_rec))
    click_tensors = sesrec_item_tensors(click_flat, (n_rows, max_src, max_click))
    src_his = [
        torch.tensor(src_query, dtype=torch.long),
        torch.tensor(src_source, dtype=torch.long),
        click_tensors,
        torch.tensor(src_words, dtype=torch.long),
    ]
    row_item_tensors = sesrec_item_tensors(row_items, (n_rows,))
    flat_item_tensors = sesrec_item_tensors(flat_candidates, (n_rows * cand_count,))
    return [users, rec_tensors, src_his, row_item_tensors], flat_item_tensors, n_rows, cand_count


def to_device_nested(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, list):
        return [to_device_nested(x, device) for x in obj]
    return obj


def score_view_expanded(
    model: torch.nn.Module,
    selected: list[dict[str, Any]],
    row_indices: list[int],
    candidates: list[list[int]],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    device: torch.device,
    swap_lookup: dict[int, int | None] | None = None,
    finite_neutral_controls: bool = False,
) -> torch.Tensor:
    input_data, bsz, cand_count = build_sesrec_batch(
        selected,
        row_indices,
        candidates,
        maps,
        neutral_item,
        neutral_query,
        view,
        swap_lookup,
        finite_neutral_controls,
    )
    with torch.no_grad():
        flat_scores = model.predict(to_device_nested(input_data, device))
    return flat_scores.reshape(bsz, cand_count)


def score_view_grouped(
    model: torch.nn.Module,
    selected: list[dict[str, Any]],
    row_indices: list[int],
    candidates: list[list[int]],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    device: torch.device,
    swap_lookup: dict[int, int | None] | None = None,
    finite_neutral_controls: bool = False,
) -> torch.Tensor:
    input_data, flat_item_tensors, bsz, cand_count = build_sesrec_grouped_batch(
        selected,
        row_indices,
        candidates,
        maps,
        neutral_item,
        neutral_query,
        view,
        swap_lookup,
        finite_neutral_controls,
    )
    with torch.no_grad():
        input_data = to_device_nested(input_data, device)
        flat_item_tensors = to_device_nested(flat_item_tensors, device)
        _, rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask, user_emb = (
            model.parse_input_test(input_data)
        )
        rec_his_emb, src_his_emb = model.his2feats(
            rec_his_emb, rec_his_mask, query_his_emb, click_item_his_emb, src_his_mask
        )
        mask_rec_pos, mask_rec_neg, mask_src_pos, mask_src_neg = model.interest_contrast(
            rec_his_emb, src_his_emb, rec_his_mask, src_his_mask, return_loss=False
        )
        user_feats = [
            rec_his_emb,
            rec_his_mask,
            src_his_emb,
            src_his_mask,
            user_emb,
            mask_rec_pos,
            mask_rec_neg,
            mask_src_pos,
            mask_src_neg,
        ]
        repeated_user_feats = [torch.repeat_interleave(user_f, cand_count, dim=0) for user_f in user_feats]
        item_emb = model.query_item_feat.get_item_emb(flat_item_tensors)
        flat_scores = model.inter_pred(repeated_user_feats, item_emb)
    return flat_scores.reshape(bsz, cand_count)


def score_view(
    model: torch.nn.Module,
    selected: list[dict[str, Any]],
    row_indices: list[int],
    candidates: list[list[int]],
    maps: CompactMaps,
    neutral_item: int,
    neutral_query: int,
    view: str,
    device: torch.device,
    swap_lookup: dict[int, int | None] | None = None,
    candidate_expanded: bool = False,
    finite_neutral_controls: bool = False,
) -> torch.Tensor:
    scorer = score_view_expanded if candidate_expanded else score_view_grouped
    return scorer(
        model,
        selected,
        row_indices,
        candidates,
        maps,
        neutral_item,
        neutral_query,
        view,
        device,
        swap_lookup,
        finite_neutral_controls,
    )


def aggregate(rows: list[dict[str, Any]], swap_levels: list[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "rows": len(rows),
        "full_sar_rr": mean([float(r["full_sar_rr"]) for r in rows]),
        "history_only_rr": mean([float(r["history_only_rr"]) for r in rows]),
        "search_only_rr": mean([float(r["search_only_rr"]) for r in rows]),
        "search_masked_rr": mean([float(r["search_masked_rr"]) for r in rows]),
        "full_sar_ndcg10": mean([float(r["full_sar_ndcg10"]) for r in rows]),
        "search_masked_ndcg10": mean([float(r["search_masked_ndcg10"]) for r in rows]),
        "full_minus_history_rr": mean([float(r["full_sar_rr"]) - float(r["history_only_rr"]) for r in rows]),
        "full_minus_masked_rr": mean([float(r["full_sar_rr"]) - float(r["search_masked_rr"]) for r in rows]),
    }
    for level in swap_levels:
        matched = [row for row in rows if int(row[f"{level}_matched"])]
        wins = sum(int(row[f"{level}_true_win"]) for row in matched)
        ties = sum(int(row[f"{level}_tie"]) for row in matched)
        metrics[f"{level}_matched"] = len(matched)
        metrics[f"{level}_true_win_rate"] = frac(wins, len(matched))
        metrics[f"{level}_tie_rate"] = frac(ties, len(matched))
        metrics[f"{level}_half_tie_win_rate"] = frac(wins + 0.5 * ties, len(matched))
        metrics[f"{level}_true_minus_swap_rr"] = mean(
            [float(row[f"{level}_true_minus_swap"]) for row in matched]
        )
    return metrics


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    reports = Path(args.reports_dir).resolve()
    reports.mkdir(parents=True, exist_ok=True)
    official_repo = Path(args.official_root).resolve() / "SESREC-SIGIR-2023"
    if not official_repo.exists():
        raise FileNotFoundError(f"SESRec official repo not found: {official_repo}")
    examples_path = Path(args.examples_path).resolve()
    item_metadata_path = Path(args.item_metadata).resolve()
    candidate_ledger_path = Path(args.candidate_ledger).resolve() if args.candidate_ledger else None
    swap_ledger_path = Path(args.swap_ledger).resolve() if args.swap_ledger else None
    selected = read_jsonl(examples_path, limit=args.row_limit)
    if len(selected) < args.row_limit:
        raise ValueError(f"expected {args.row_limit} rows, loaded {len(selected)}")
    # R161_SESREC_OFFICIAL_FIVE_VIEW hard-fail gate: the readout must score
    # target plus at least one negative and later must produce L1/L2 matches.
    if args.neg_k < 1:
        raise ValueError(f"{CONTRACT_ID} requires at least one negative candidate")

    candidate_raw_by_example = None
    if candidate_ledger_path is not None:
        from common_candidate_contract import load_candidate_ledger

        candidate_raw_by_example = load_candidate_ledger(
            candidate_ledger_path,
            {str(row["example_id"]) for row in selected},
        )
    maps = build_maps(selected, candidate_raw_by_example)
    eval_indices = split_indices(selected, args.split)[: args.max_eval_rows]
    candidate_indices = eval_indices if args.eval_candidates_only else None
    if candidate_raw_by_example is None:
        candidates = build_candidate_lists(
            selected,
            maps,
            args.neg_k,
            args.seed,
            row_indices=candidate_indices,
        )
    else:
        candidates = map_canonical_candidate_lists(
            selected,
            maps,
            candidate_raw_by_example,
            args.neg_k,
            row_indices=candidate_indices,
        )
    neutral_item = (max(maps.item.values()) if maps.item else 1) + 1
    neutral_query = (max(maps.query_token.values()) if maps.query_token else 1) + 1
    validate_r172_contract(args, examples_path, selected, candidates, eval_indices)
    emit_progress(
        "prepared",
        candidate_expanded=bool(args.candidate_expanded),
        contract_check_only=bool(args.contract_check_only),
        eval_rows=len(eval_indices),
        neg_k=args.neg_k,
        row_limit=args.row_limit,
        split=args.split,
    )
    if args.contract_check_only:
        emit_progress("contract_check_done", candidate_count=len(candidates[eval_indices[0]]) if eval_indices else 0)
        return
    item_cat = load_item_categories(item_metadata_path)
    if swap_ledger_path is not None:
        if not args.extended_swap_ladder:
            raise ValueError("--swap-ledger requires --extended-swap-ladder")
        from common_swap_contract import load_swap_ledger

        swap_lookups = load_swap_ledger(swap_ledger_path, selected, eval_indices)
    elif args.extended_swap_ladder:
        swap_lookups, _ = build_extended_swap_lookups(
            selected,
            eval_indices,
            item_cat,
            args.seed,
            args.swap_tries,
        )
    else:
        profiles = build_swap_profiles(selected, item_cat)
        swap_indices = build_swap_indices(profiles)
        rng = random.Random(args.seed)
        l1_lookup: dict[int, int | None] = {}
        l2_lookup: dict[int, int | None] = {}
        for idx in eval_indices:
            l1_lookup[idx] = pick_swap(idx, "L1", profiles, swap_indices, rng, args.swap_tries)
            l2_lookup[idx] = pick_swap(idx, "L2", profiles, swap_indices, rng, args.swap_tries)
        swap_lookups = {"l1": l1_lookup, "l2": l2_lookup}
    swap_levels = list(swap_lookups)

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    reset_names = ["config", "config.const", "models", "models.SESRec", "models.Inputs", "models.module"]
    old_modules = reset_imports(reset_names)
    try:
        os.chdir(official_repo)
        sys.path.insert(0, str(official_repo))
        const = importlib.import_module("config.const")
        inputs_mod = importlib.import_module("models.Inputs")
        sesrec_mod = importlib.import_module("models.SESRec")
        const.init_dataset_setting_commercial()
        inputs_mod.init_data_attribute_ls("commercial")
        with (official_repo / "config/SESRec_commercial.yaml").open("r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        device = torch.device(args.device)
        model = sesrec_mod.SESRec(config).to(device)
        mode = "interface_smoke"
        checkpoint = str(args.checkpoint).strip()
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state)
            mode = "checkpoint_readout"
        model.eval()
        emit_progress("model_ready", device=str(device), mode=mode)

        details: list[dict[str, Any]] = []
        rank_rr_checks: list[bool] = []
        total_batches = math.ceil(len(eval_indices) / args.batch_size) if args.batch_size else 0
        for batch_no, start in enumerate(range(0, len(eval_indices), args.batch_size), start=1):
            batch_idx = eval_indices[start : start + args.batch_size]
            if (
                args.progress_every > 0
                and (batch_no == 1 or batch_no == total_batches or batch_no % args.progress_every == 0)
            ):
                emit_progress(
                    "batch_start",
                    batch=batch_no,
                    batch_rows=len(batch_idx),
                    total_batches=total_batches,
                )
            scores = {
                "full_sar": score_view(
                    model,
                    selected,
                    batch_idx,
                    candidates,
                    maps,
                    neutral_item,
                    neutral_query,
                    "full_sar",
                    device,
                    candidate_expanded=args.candidate_expanded,
                    finite_neutral_controls=args.finite_neutral_controls,
                ),
                "history_only": score_view(
                    model,
                    selected,
                    batch_idx,
                    candidates,
                    maps,
                    neutral_item,
                    neutral_query,
                    "history_only",
                    device,
                    candidate_expanded=args.candidate_expanded,
                    finite_neutral_controls=args.finite_neutral_controls,
                ),
                "search_only": score_view(
                    model,
                    selected,
                    batch_idx,
                    candidates,
                    maps,
                    neutral_item,
                    neutral_query,
                    "search_only",
                    device,
                    candidate_expanded=args.candidate_expanded,
                    finite_neutral_controls=args.finite_neutral_controls,
                ),
                "search_masked": score_view(
                    model,
                    selected,
                    batch_idx,
                    candidates,
                    maps,
                    neutral_item,
                    neutral_query,
                    "search_masked",
                    device,
                    candidate_expanded=args.candidate_expanded,
                    finite_neutral_controls=args.finite_neutral_controls,
                ),
            }
            for level, lookup in swap_lookups.items():
                scores[level] = score_view(
                    model,
                    selected,
                    batch_idx,
                    candidates,
                    maps,
                    neutral_item,
                    neutral_query,
                    "counterfactual_search",
                    device,
                    lookup,
                    args.candidate_expanded,
                    args.finite_neutral_controls,
                )
            nonfinite = {
                view: int((~torch.isfinite(score_tensor)).sum().item())
                for view, score_tensor in scores.items()
            }
            if any(nonfinite.values()):
                raise RuntimeError(f"non-finite SESRec scores: {nonfinite}")
            rr: dict[str, list[float]] = {}
            rank: dict[str, list[int]] = {}
            target_score: dict[str, list[float]] = {}
            ndcg10: dict[str, list[float]] = {}
            for view, score_tensor in scores.items():
                rr[view], rank[view] = rr_from_scores(score_tensor)
                rank_rr_checks.append(rank_rr_identity_pass(rr[view], rank[view]))
                ndcg10[view] = ndcg_at_k_from_scores(score_tensor, 10)
                target_score[view] = [
                    float(value) for value in score_tensor[:, 0].detach().cpu().tolist()
                ]
            for pos, idx in enumerate(batch_idx):
                row = {
                    "row_index": idx,
                    "example_id": selected[idx]["example_id"],
                    "split": selected[idx].get("split", ""),
                    "user_id": int(selected[idx].get("user_id", -1)),
                    "target_item": int(selected[idx].get("primary_target_item", -1)),
                    "source_session_id": idx + 1,
                    "candidate_count": len(candidates[idx]),
                    "source": selected[idx].get("source", ""),
                    "search_pos_count": len(selected[idx].get("E_search_pos", [])),
                    "search_action_count": len(selected[idx].get("E_search_all", [])),
                    "pre_len": len(selected[idx].get("H_pre", [])),
                    "full_sar_rr": rr["full_sar"][pos],
                    "history_only_rr": rr["history_only"][pos],
                    "search_only_rr": rr["search_only"][pos],
                    "search_masked_rr": rr["search_masked"][pos],
                    "full_sar_rank": rank["full_sar"][pos],
                    "history_only_rank": rank["history_only"][pos],
                    "search_only_rank": rank["search_only"][pos],
                    "search_masked_rank": rank["search_masked"][pos],
                    "full_sar_target_score": target_score["full_sar"][pos],
                    "search_masked_target_score": target_score["search_masked"][pos],
                    "full_sar_ndcg10": ndcg10["full_sar"][pos],
                    "search_masked_ndcg10": ndcg10["search_masked"][pos],
                }
                for level, lookup in swap_lookups.items():
                    swap_index = lookup[idx]
                    matched = swap_index is not None
                    score_delta = (
                        target_score["full_sar"][pos] - target_score[level][pos]
                        if matched
                        else 0.0
                    )
                    row[f"{level}_matched"] = int(matched)
                    row[f"{level}_swap_row_index"] = int(swap_index) if matched else -1
                    row[f"{level}_swap_source_session_id"] = (
                        int(swap_index) + 1 if matched else 0
                    )
                    row[f"{level}_swap_rr"] = rr[level][pos] if matched else 0.0
                    row[f"{level}_true_minus_swap"] = (
                        rr["full_sar"][pos] - rr[level][pos] if matched else 0.0
                    )
                    row[f"{level}_score_delta"] = score_delta
                    if args.score_level_half_tie:
                        row[f"{level}_true_win"] = int(matched and score_delta > 0.0)
                        row[f"{level}_tie"] = int(matched and score_delta == 0.0)
                    else:
                        row[f"{level}_true_win"] = int(
                            matched and rr["full_sar"][pos] > rr[level][pos]
                        )
                        row[f"{level}_tie"] = int(
                            matched and rr["full_sar"][pos] == rr[level][pos]
                        )
                details.append(row)
            if (
                args.progress_every > 0
                and (batch_no == 1 or batch_no == total_batches or batch_no % args.progress_every == 0)
            ):
                emit_progress("batch_done", batch=batch_no, rows_done=len(details), total_batches=total_batches)

        scores_csv = reports / f"{args.report_prefix}_five_view_scores.csv"
        json_path = reports / f"{args.report_prefix}_five_view_readout.json"
        md_path = reports / f"{args.report_prefix}_five_view_readout.md"
        write_csv(scores_csv, details)
        metrics = aggregate(details, swap_levels)
        if metrics["rows"] <= 0:
            raise RuntimeError(f"{CONTRACT_ID} produced no evaluated rows")
        if not rank_rr_checks or not all(rank_rr_checks):
            raise RuntimeError(f"{CONTRACT_ID} rank/RR consistency check failed")
        missing_levels = [level for level in swap_levels if metrics[f"{level}_matched"] <= 0]
        if missing_levels:
            raise RuntimeError(
                f"{CONTRACT_ID} requires non-empty matched sets; missing={missing_levels}"
            )
        payload = {
            "schema": "sarcheck.r161.sesrec_official_five_view.v0.2",
            "contract_id": CONTRACT_ID,
            "surface_id": args.surface_id or None,
            "status": "done",
            "mode": mode,
            "official_repo": str(official_repo),
            "official_commit": get_git_commit(official_repo),
            "checkpoint": checkpoint or None,
            "split": args.split,
            "row_limit": args.row_limit,
            "max_eval_rows": args.max_eval_rows,
            "neg_k": args.neg_k,
            "candidate_contract": (
                "canonical_raw_item_ledger" if candidate_raw_by_example is not None else "model_local"
            ),
            "candidate_ledger": str(candidate_ledger_path) if candidate_ledger_path else None,
            "candidate_ledger_sha256": (
                file_sha256(candidate_ledger_path) if candidate_ledger_path else None
            ),
            "swap_levels": swap_levels,
            "swap_ledger": str(swap_ledger_path) if swap_ledger_path else None,
            "swap_ledger_sha256": file_sha256(swap_ledger_path) if swap_ledger_path else None,
            "specificity_basis": "target_score" if args.score_level_half_tie else "reciprocal_rank",
            "scoring_mode": "candidate_expanded" if args.candidate_expanded else "grouped_user_features",
            "candidate_tensors_frozen": True,
            "finite_all_scores": True,
            "rr_tie_policy": "midpoint rank among exact candidate-score ties",
            "rank_rr_identity_pass": bool(rank_rr_checks) and all(rank_rr_checks),
            "finite_neutral_controls": bool(args.finite_neutral_controls),
            "same_native_scorer": (
                "SESRec.predict"
                if args.candidate_expanded
                else "SESRec.parse_input_test->his2feats->interest_contrast->inter_pred"
            ),
            "metrics": metrics,
            "outputs": {
                "scores_csv": str(scores_csv),
                "json": str(json_path),
                "report_md": str(md_path),
            },
            "limits": [
                "This is a bounded official SESRec checkpoint readout on the SARCheck-converted split-complete bundle.",
                "Counterfactual-search views replace only focal search content while keeping user history, target, and candidates fixed.",
                "If mode is interface_smoke, metrics validate only the forward/readout interface and must not be interpreted as trained-model evidence.",
            ],
        }
        write_json(json_path, payload)
        md = [
            f"# {args.report_prefix} SESRec Official Five-view Readout",
            "",
            f"- Status: done",
            f"- Mode: `{mode}`",
            f"- Rows: {metrics['rows']}",
            f"- Full-SAR RR: {metrics['full_sar_rr']:.4f}",
            f"- History-only RR: {metrics['history_only_rr']:.4f}",
            f"- Search-only RR: {metrics['search_only_rr']:.4f}",
            f"- Search-masked RR: {metrics['search_masked_rr']:.4f}",
            f"- Full-SAR NDCG@10: {metrics['full_sar_ndcg10']:.4f}",
            f"- RR tie policy: `{payload['rr_tie_policy']}`",
            f"- Rank/RR identity: `{payload['rank_rr_identity_pass']}`",
            f"- Full minus History-only RR: {metrics['full_minus_history_rr']:.4f}",
            f"- Full minus Search-masked RR: {metrics['full_minus_masked_rr']:.4f}",
        ]
        for level in swap_levels:
            md.append(
                f"- {level} true-search win / tie / half-tie win: "
                f"{metrics[f'{level}_true_win_rate']:.4f} / "
                f"{metrics[f'{level}_tie_rate']:.4f} / "
                f"{metrics[f'{level}_half_tie_win_rate']:.4f} "
                f"({metrics[f'{level}_matched']} matched)"
            )
        md.extend(
            [
                "",
                "Interpretation: use this report as official SESRec five-view readout evidence only when mode is `checkpoint_readout`.",
            ]
        )
        write_md(md_path, "\n".join(md))
    finally:
        restore_imports(old_modules, reset_names)
        os.chdir(old_cwd)
        sys.path[:] = old_path


if __name__ == "__main__":
    main()
