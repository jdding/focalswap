#!/usr/bin/env python3
"""R208 UniSAR FocalSwap runner on the official processed KuaiSAR contract.

This runner is separate from the older compact-bundle adapter.  It consumes the
official UniSAR `data/KuaiSAR` processed files, so any checkpoint trained by the
official code scores the same id space.  The script can run as a CPU interface
smoke without a checkpoint, or as a checkpoint-backed five-view readout when a
trained `best.pt` is supplied.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import pickle
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
REPORTS = ROOT / "reports"
CONTRACT_ID = "R208_UNISAR_OFFICIAL_PROCESSED_CHECKPOINT_FOCALSWAP"
if not CONTRACT_ID.startswith("R208_UNISAR_OFFICIAL_PROCESSED"):
    raise RuntimeError(f"Invalid experiment contract: {CONTRACT_ID}")
EXPECTED_LEVELS = [
    "l1_nuisance",
    "l2_same_category",
    "category_broken_source",
    "same_category_source",
]

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from official_adapter import (  # noqa: E402
    get_git_commit,
    tensor_to_device,
    write_csv,
    write_json,
    write_md,
)
from focal_source_views import (  # noqa: E402
    changed_slot_counts,
    focal_source_ids,
    replace_matched_focal_source,
    replace_source_view,
    source_slot_counts,
)
from controlled_view_ranking import (  # noqa: E402
    midpoint_rr_from_scores,
    rank_rr_identity_pass,
)


def rr_from_scores(scores: torch.Tensor) -> tuple[list[float], list[float]]:
    return midpoint_rr_from_scores(scores)


def manifest_path(path: Path) -> str:
    """Keep repository outputs relative and external run outputs absolute."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--processed-data-root", required=True)
    p.add_argument(
        "--work-dir",
        default="",
        help="Run-specific scratch directory; defaults to a process-specific path.",
    )
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--report-prefix", default="r208_unisar_official_processed")
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--row-limit", type=int, default=512, help="0 means all rows in the split")
    p.add_argument("--max-eval-rows", type=int, default=64, help="0 means all selected rows")
    p.add_argument("--expected-split-rows", type=int, default=0, help="0 disables processed split row-count validation")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-batch-size", type=int, default=32)
    p.add_argument("--swap-tries", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260620)
    p.add_argument(
        "--reference-source-mode",
        choices=["true", "matched_random"],
        default="true",
        help=(
            "Source used as the reference side in swap comparisons. "
            "`true` is the promoted FocalSwap readout. `matched_random` "
            "uses an independently matched source as a reverse/source-ablation "
            "control, removing the true focal-source identity from the "
            "reference side while keeping the candidate set frozen."
        ),
    )
    p.add_argument(
        "--reference-seed",
        type=int,
        default=0,
        help="Seed for matched_random reference-source lookups; 0 uses seed+1000003.",
    )
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else math.nan


def frac(num: float, den: int) -> float:
    return float(num / den) if den else math.nan


def bucket_len(n: int) -> str:
    n = int(n)
    if n <= 0:
        return "0"
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    return "8+"


def json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def split_file(split: str) -> str:
    return "rec_val.pkl" if split == "validation" else "rec_test.pkl"


def prepare_work_data(processed_root: Path, work_dir: Path) -> dict[str, Any]:
    """Create a lightweight official data directory with one appended mask row."""
    src_root = processed_root
    if (src_root / "KuaiSAR").exists():
        src_root = src_root / "KuaiSAR"
    if not (src_root / "dataset").exists() or not (src_root / "vocab").exists():
        raise FileNotFoundError(f"processed UniSAR KuaiSAR data not found under {processed_root}")

    data_root = work_dir / "data" / "KuaiSAR"
    dataset_out = data_root / "dataset"
    vocab_out = data_root / "vocab"
    if data_root.exists():
        shutil.rmtree(data_root)
    dataset_out.mkdir(parents=True, exist_ok=True)
    vocab_out.mkdir(parents=True, exist_ok=True)

    for src in (src_root / "dataset").iterdir():
        if src.is_file():
            os.symlink(src, dataset_out / src.name)
    for name in ["user_vocab.pkl", "query_vocab.pkl", "user_vocab_np.pkl", "item_vocab_np.pkl"]:
        os.symlink(src_root / "vocab" / name, vocab_out / name)

    src_map = load_pickle(src_root / "vocab" / "src_session_vocab_np.pkl")
    neutral_src_session_id = int(src_map["keyword"].shape[0])
    extended = {}
    for key, arr in src_map.items():
        zeros_shape = (1, *arr.shape[1:])
        extended[key] = np.concatenate([arr, np.zeros(zeros_shape, dtype=arr.dtype)], axis=0)
    dump_pickle(vocab_out / "src_session_vocab_np.pkl", extended)

    item_map = load_pickle(src_root / "vocab" / "item_vocab_np.pkl")
    item_rows = min(int(np.asarray(values).shape[0]) for values in item_map.values())
    neutral_item_id = 0
    for candidate in range(item_rows - 1, 0, -1):
        if all(int(np.count_nonzero(np.asarray(values)[candidate])) == 0 for values in item_map.values()):
            neutral_item_id = candidate
            break
    if neutral_item_id <= 0:
        raise RuntimeError(
            "official item vocabulary has no non-padding all-zero metadata row for the search-only control"
        )
    return {
        "source_root": str(src_root),
        "work_data_root": str(data_root),
        "neutral_src_session_id": neutral_src_session_id,
        "neutral_item_id": int(neutral_item_id),
        "neutral_item_policy": "highest existing non-padding row with all-zero item metadata",
        "item_vocab_rows_before_mask": int(item_rows),
        "src_session_rows_before_mask": neutral_src_session_id,
    }


def source_session_for_row(rec_row: pd.Series, user_vocab: dict[int, dict[str, Any]]) -> int:
    uid = int(rec_row["user_id"])
    count = int(rec_row["src_session_his"])
    sessions = [int(x) for x in user_vocab[uid]["src_session_his"][:count] if int(x) != 0]
    return sessions[-1] if sessions else 0


def rec_history_for_row(rec_row: pd.Series, user_vocab: dict[int, dict[str, Any]]) -> list[int]:
    uid = int(rec_row["user_id"])
    count = int(rec_row["rec_his"])
    return [int(x) for x in user_vocab[uid]["rec_his"][:count] if int(x) != 0][-30:]


def nonzero_list(arr: np.ndarray | list[int]) -> list[int]:
    return [int(x) for x in list(arr) if int(x) != 0]


def build_rows(
    rec_df: pd.DataFrame,
    user_vocab: dict[int, dict[str, Any]],
    src_map: dict[str, np.ndarray],
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec_idx, rec_row in rec_df.iterrows():
        sid = source_session_for_row(rec_row, user_vocab)
        src_items = nonzero_list(src_map["pos_items"][sid]) if sid else []
        query_tokens = nonzero_list(src_map["keyword"][sid]) if sid else []
        target = int(rec_row["item_id"])
        rows.append(
            {
                "schema": "focalswap.r208.unisar_official_processed_row.v1",
                "example_id": f"unisar:{split}:{int(rec_idx)}",
                "official_row_index": int(rec_idx),
                "query_id": int(sid),
                "split": split,
                "user_id": int(rec_row["user_id"]),
                "source": f"src_session:{int(sid)}",
                "source_surface": "unisar_official_processed",
                "H_pre": rec_history_for_row(rec_row, user_vocab),
                "E_search_pos": src_items,
                "E_search_neg": [],
                "E_search_all": src_items,
                "query_tokens": query_tokens,
                "Y_post_rec": [target],
                "primary_target_item": target,
                "official_source_session_id": sid,
                "features": {
                    "pre_len": int(rec_row["rec_his"]),
                    "search_action_count": len(src_items),
                    "source_session_his": int(rec_row["src_session_his"]),
                    "candidate_count": 1 + len(rec_row["neg_items"]),
                },
            }
        )
    return rows


def select_eval_rows(rows: list[dict[str, Any]], row_limit: int, max_eval_rows: int) -> tuple[list[dict[str, Any]], list[int]]:
    selected = rows if row_limit <= 0 else rows[:row_limit]
    eval_indices = list(range(len(selected)))
    if max_eval_rows > 0:
        eval_indices = eval_indices[:max_eval_rows]
    return selected, eval_indices


def build_candidate_batch(
    rec_dataset: Any,
    row_indices: list[int],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feeds = []
    for idx in row_indices:
        official_idx = int(selected[idx]["official_row_index"])
        feeds.append(rec_dataset[official_idx])
    return feeds


def replace_source(batch: dict[str, Any], view: str, neutral_src_id: int, neutral_item_id: int) -> dict[str, Any]:
    return replace_source_view(batch, view, neutral_src_id, neutral_item_id)


def compare_frozen_tensors(base: dict[str, Any], other: dict[str, Any]) -> bool:
    required = ["item", "neg_items", "user", "all_his_ts", "all_his_type"]
    for key in required:
        if key not in base or key not in other:
            return False
        if not isinstance(base[key], torch.Tensor) or not isinstance(other[key], torch.Tensor):
            return False
        if not torch.equal(base[key], other[key]):
            return False
    return True


def replace_matched_source_only(batch: dict[str, Any], repl: torch.Tensor) -> dict[str, Any]:
    """Compatibility wrapper for latest-slot matched replacement."""
    return replace_matched_focal_source(batch, repl)


def replacement_ids_for_level(
    row_indices: list[int],
    selected: list[dict[str, Any]],
    lookup: dict[int, int | None],
) -> list[int]:
    repl_ids = []
    for idx in row_indices:
        swap_idx = lookup.get(idx)
        repl_ids.append(0 if swap_idx is None else int(selected[int(swap_idx)]["official_source_session_id"]))
    return repl_ids


def item_categories(items: list[int], item_cat: dict[int, int]) -> set[int]:
    return {int(item_cat[int(x)]) for x in items if int(x) in item_cat and int(item_cat[int(x)]) != 0}


def source_profile(row: dict[str, Any], item_cat: dict[int, int]) -> dict[str, Any]:
    """Pre/search-only matching profile for the R208 official-processed ladder.

    This profile intentionally does not inspect post-search targets, ranks, or
    candidate scores.  It uses only row split, user/query ids for exclusion,
    pre-history length, source-session content size, query length, and source
    item categories observed before the recommendation target is scored.
    """
    search_items = [int(x) for x in row.get("E_search_all", [])]
    return {
        "split": str(row.get("split")),
        "user_id": int(row.get("user_id", -1)),
        "query_id": int(row.get("query_id", 0)),
        "source_surface": str(row.get("source_surface", "unisar_official_processed")),
        "pre_bucket": bucket_len(len(row.get("H_pre", []))),
        "source_item_bucket": bucket_len(len(search_items)),
        "query_token_bucket": bucket_len(len(row.get("query_tokens", []))),
        "search_cats": item_categories(search_items, item_cat),
    }


def build_source_profile_swap_lookups(
    selected: list[dict[str, Any]],
    eval_indices: list[int],
    item_cat: dict[int, int],
    seed: int,
    tries: int,
) -> tuple[dict[str, dict[int, int | None]], dict[str, dict[str, Any]], dict[str, str]]:
    """Build R208 swaps without target-label or score-conditioned selection."""
    rng = random.Random(seed)
    profiles = [source_profile(row, item_cat) for row in selected]
    by_nuisance: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    by_source_profile: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    by_category: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    by_category_source_profile: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for idx, p in enumerate(profiles):
        source_key = (p["split"], p["source_surface"], p["source_item_bucket"], p["query_token_bucket"])
        by_source_profile[source_key].append(idx)
        by_nuisance[(p["split"], p["source_surface"], p["pre_bucket"], p["source_item_bucket"], p["query_token_bucket"])].append(idx)
        for cat in p["search_cats"]:
            by_category[(p["split"], cat)].append(idx)
            by_category_source_profile[(*source_key, cat)].append(idx)

    def frozen_pool(pool: list[int]) -> tuple[int, ...]:
        return tuple(sorted(set(pool)))

    by_nuisance_pool = {k: frozen_pool(v) for k, v in by_nuisance.items()}
    by_source_profile_pool = {k: frozen_pool(v) for k, v in by_source_profile.items()}
    by_category_pool = {k: frozen_pool(v) for k, v in by_category.items()}
    by_category_source_profile_pool = {k: frozen_pool(v) for k, v in by_category_source_profile.items()}
    cat_union_cache: dict[tuple[Any, ...], tuple[int, ...]] = {}
    cat_source_union_cache: dict[tuple[Any, ...], tuple[int, ...]] = {}

    level_rules = {
        "l1_nuisance": "same split/source-surface/pre-length/source-session-size/query-length buckets; different user and source session",
        "l2_same_category": "same split and at least one shared source-item category; different user and source session",
        "category_broken_source": "same split/source-surface/source-session-size/query-length buckets and no shared source-item category; different user and source session",
        "same_category_source": "same split/source-surface/source-session-size/query-length buckets and at least one shared source-item category; different user and source session",
    }

    def valid(i: int, j: int, require_overlap: bool | None) -> bool:
        if i == j:
            return False
        p, q = profiles[i], profiles[j]
        if p["split"] != q["split"]:
            return False
        if p["user_id"] == q["user_id"] or p["query_id"] == q["query_id"]:
            return False
        overlap = bool(p["search_cats"] & q["search_cats"])
        if require_overlap is True and not overlap:
            return False
        if require_overlap is False and overlap:
            return False
        return True

    def choose(i: int, pool: tuple[int, ...], require_overlap: bool | None) -> int | None:
        if not pool:
            return None
        for _ in range(max(1, min(tries, len(pool)))):
            j = pool[rng.randrange(len(pool))]
            if valid(i, j, require_overlap):
                return j
        for j in pool:
            if valid(i, j, require_overlap):
                return j
        return None

    def category_pool(split: str, cats: set[int]) -> tuple[int, ...]:
        key = (split, *tuple(sorted(cats)))
        if key not in cat_union_cache:
            pooled: list[int] = []
            for cat in cats:
                pooled.extend(by_category_pool.get((split, cat), ()))
            cat_union_cache[key] = frozen_pool(pooled)
        return cat_union_cache[key]

    def category_source_pool(source_key: tuple[Any, ...], cats: set[int]) -> tuple[int, ...]:
        key = (*source_key, *tuple(sorted(cats)))
        if key not in cat_source_union_cache:
            pooled: list[int] = []
            for cat in cats:
                pooled.extend(by_category_source_profile_pool.get((*source_key, cat), ()))
            cat_source_union_cache[key] = frozen_pool(pooled)
        return cat_source_union_cache[key]

    out: dict[str, dict[int, int | None]] = {level: {} for level in EXPECTED_LEVELS}
    total_eval = len(eval_indices)
    for pos, idx in enumerate(eval_indices, start=1):
        if pos == 1 or pos % 10000 == 0 or pos == total_eval:
            print(f"[R208] building swap lookups {pos}/{total_eval}", flush=True)
        p = profiles[idx]
        source_key = (p["split"], p["source_surface"], p["source_item_bucket"], p["query_token_bucket"])
        nuisance_key = (p["split"], p["source_surface"], p["pre_bucket"], p["source_item_bucket"], p["query_token_bucket"])
        out["l1_nuisance"][idx] = choose(idx, by_nuisance_pool.get(nuisance_key, ()), None)
        out["l2_same_category"][idx] = choose(idx, category_pool(p["split"], p["search_cats"]), True)
        out["category_broken_source"][idx] = choose(idx, by_source_profile_pool.get(source_key, ()), False)
        out["same_category_source"][idx] = choose(idx, category_source_pool(source_key, p["search_cats"]), True)

    summary: dict[str, dict[str, Any]] = {}
    for level, lookup in out.items():
        matched = sum(v is not None for v in lookup.values())
        summary[level] = {
            "attempted": len(eval_indices),
            "matched": matched,
            "coverage": frac(matched, len(eval_indices)),
            "match_rule": level_rules[level],
            "future_label_or_score_used": False,
        }
    return out, summary, level_rules


def disable_nested_transformer_fastpath(model: torch.nn.Module) -> None:
    """Allow all-masked source slices in official UniSAR history-only views."""
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoder):
            if hasattr(module, "enable_nested_tensor"):
                module.enable_nested_tensor = False
            if hasattr(module, "use_nested_tensor"):
                module.use_nested_tensor = False


def summarize_level(rows: list[dict[str, Any]], level: str) -> dict[str, Any]:
    sub = [r for r in rows if int(r[f"{level}_matched"])]
    wins = sum(1 for r in sub if float(r[f"{level}_score_delta"]) > 0)
    ties = sum(1 for r in sub if float(r[f"{level}_score_delta"]) == 0)
    losses = sum(1 for r in sub if float(r[f"{level}_score_delta"]) < 0)
    n = len(sub)
    return {
        "matched": n,
        "strict_win": frac(wins, n),
        "tie": frac(ties, n),
        "loss": frac(losses, n),
        "half_tie": frac(wins + 0.5 * ties, n),
        "mean_rr_delta": mean([float(r[f"{level}_rr_delta"]) for r in sub]),
        "mean_score_delta": mean([float(r[f"{level}_score_delta"]) for r in sub]),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    official_repo = Path(args.official_root) / "UniSAR"
    if not official_repo.exists():
        raise FileNotFoundError(f"UniSAR official repo not found: {official_repo}")

    processed_root = Path(args.processed_data_root)
    work_dir = (
        Path(args.work_dir)
        if str(args.work_dir).strip()
        else Path(f"/private/tmp/focalswap_r208_unisar_official_processed_{os.getpid()}")
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare_work_data(processed_root, work_dir)
    source_root = Path(prep["source_root"])
    rec_df = pd.read_pickle(source_root / "dataset" / split_file(args.split)).reset_index(drop=True)
    if args.expected_split_rows > 0 and int(rec_df.shape[0]) != int(args.expected_split_rows):
        raise RuntimeError(
            "R208 official processed split-row mismatch: "
            f"{args.split} got {int(rec_df.shape[0])}, expected {int(args.expected_split_rows)}. "
            "Refusing to score a stale or bounded processed surface."
        )
    user_vocab = load_pickle(source_root / "vocab" / "user_vocab.pkl")
    src_map = load_pickle(source_root / "vocab" / "src_session_vocab_np.pkl")
    item_map = load_pickle(source_root / "vocab" / "item_vocab_np.pkl")
    item_cat = {int(i): int(c) for i, c in enumerate(item_map["first_level_category_id"]) if int(c) != 0}
    rows = build_rows(rec_df, user_vocab, src_map, args.split)
    selected, eval_indices = select_eval_rows(rows, args.row_limit, args.max_eval_rows)
    print(
        f"[R208] rows built: processed_split={int(rec_df.shape[0])} "
        f"selected={len(selected)} eval={len(eval_indices)}",
        flush=True,
    )
    swap_lookups, swap_summary, level_rules = build_source_profile_swap_lookups(
        selected, eval_indices, item_cat, args.seed, args.swap_tries
    )
    print(
        "[R208] swap lookups built: "
        + ", ".join(f"{level}={swap_summary[level]['matched']}/{swap_summary[level]['attempted']}" for level in EXPECTED_LEVELS),
        flush=True,
    )
    reference_seed = int(args.reference_seed) if int(args.reference_seed) else int(args.seed) + 1000003
    reference_swap_lookups: dict[str, dict[int, int | None]] | None = None
    reference_swap_summary: dict[str, dict[str, Any]] | None = None
    if args.reference_source_mode == "matched_random":
        reference_swap_lookups, reference_swap_summary, _ = build_source_profile_swap_lookups(
            selected, eval_indices, item_cat, reference_seed, args.swap_tries
        )
        print(
            "[R208] reference-source lookups built: "
            + ", ".join(
                f"{level}={reference_swap_summary[level]['matched']}/{reference_swap_summary[level]['attempted']}"
                for level in EXPECTED_LEVELS
            ),
            flush=True,
        )
    checkpoint_load_info: dict[str, Any] = {
        "used_checkpoint": bool(str(args.checkpoint).strip()),
        "strict": None,
        "missing_keys": [],
        "unexpected_keys": [],
        "allowed_unexpected_keys_only": True,
    }

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(work_dir)
        sys.path.insert(0, str(official_repo))
        const = importlib.import_module("utils.const")
        utils = importlib.import_module("utils.utils")
        dataset_mod = importlib.import_module("utils.dataset")
        model_mod = importlib.import_module("models.UniSAR")
        const.init_setting_KuaiSAR()
        user_vocab_work = utils.load_pickle(const.user_vocab)
        rec_dataset = dataset_mod.RecDataSet("val" if args.split == "validation" else "test", user_vocab=user_vocab_work)
        device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
        model_args = SimpleNamespace(
            device=device,
            model_path=str(work_dir / "model"),
            dropout=0.0,
            batch_size=args.model_batch_size,
            num_layers=1,
            num_heads=2,
            q_i_cl_temp=0.5,
            q_i_cl_weight=0.0,
            his_cl_temp=0.1,
            his_cl_weight=0.0,
            pred_hid_units=[200, 80, 1],
        )
        model = model_mod.UniSAR(model_args)
        checkpoint = str(args.checkpoint).strip()
        mode = "interface_smoke_random_init"
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            load_result = model.load_state_dict(state_dict, strict=False)
            missing_keys = list(load_result.missing_keys)
            unexpected_keys = list(load_result.unexpected_keys)
            allowed_prefixes = (
                "feature_alignment.infoNCE_loss.",
                "rec_his_cl.",
                "src_his_cl.",
            )
            unknown_unexpected = [
                key for key in unexpected_keys if not key.startswith(allowed_prefixes)
            ]
            checkpoint_load_info = {
                "used_checkpoint": True,
                "strict": False,
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "allowed_unexpected_keys_only": not unknown_unexpected,
            }
            if missing_keys or unknown_unexpected:
                raise RuntimeError(
                    "checkpoint load mismatch: "
                    f"missing_keys={missing_keys}, unknown_unexpected_keys={unknown_unexpected}"
                )
            mode = "checkpoint_readout"
        disable_nested_transformer_fastpath(model)
        model.eval()

        detail_rows: list[dict[str, Any]] = []
        frozen_field_checks: list[bool] = []
        view_input_equal: dict[str, list[bool]] = defaultdict(list)
        source_slot_histogram: Counter[int] = Counter()
        focal_id_checks: list[bool] = []
        focal_mask_change_checks: list[bool] = []
        history_mask_change_checks: list[bool] = []
        swap_change_checks: list[bool] = []
        rank_rr_checks: list[bool] = []
        for start in range(0, len(eval_indices), args.batch_size):
            row_indices = eval_indices[start : start + args.batch_size]
            if start == 0 or start % max(args.batch_size * 100, 1) == 0 or start + len(row_indices) >= len(eval_indices):
                print(f"[R208] scoring rows {start + len(row_indices)}/{len(eval_indices)}", flush=True)
            feed = build_candidate_batch(rec_dataset, row_indices, selected)
            batch = tensor_to_device(rec_dataset.collate_batch(feed), device)
            slot_counts = source_slot_counts(batch)
            if bool((slot_counts <= 0).any().item()):
                raise RuntimeError("focal-only audit encountered a row without a source session")
            source_slot_histogram.update(int(value) for value in slot_counts.detach().cpu().tolist())
            actual_focal_ids = focal_source_ids(batch).detach().cpu().tolist()
            expected_focal_ids = [int(selected[idx]["official_source_session_id"]) for idx in row_indices]
            focal_id_checks.extend(
                int(actual) == int(expected)
                for actual, expected in zip(actual_focal_ids, expected_focal_ids)
            )
            view_batches = {
                "full_sar": replace_source(batch, "full_sar", int(prep["neutral_src_session_id"]), int(prep["neutral_item_id"])),
                "history_only": replace_source(batch, "history_only", int(prep["neutral_src_session_id"]), int(prep["neutral_item_id"])),
                "search_only": replace_source(batch, "search_only", int(prep["neutral_src_session_id"]), int(prep["neutral_item_id"])),
                "search_masked": replace_source(batch, "search_masked", int(prep["neutral_src_session_id"]), int(prep["neutral_item_id"])),
            }
            frozen_field_checks.extend(compare_frozen_tensors(batch, vb) for vb in view_batches.values())
            focal_mask_change_checks.extend(
                int(value) == 1
                for value in changed_slot_counts(batch, view_batches["search_masked"])
                .detach()
                .cpu()
                .tolist()
            )
            history_mask_change_checks.extend(
                int(changed) == int(expected)
                for changed, expected in zip(
                    changed_slot_counts(batch, view_batches["history_only"])
                    .detach()
                    .cpu()
                    .tolist(),
                    slot_counts.detach().cpu().tolist(),
                )
            )
            view_input_equal["history_vs_masked"].append(
                torch.equal(view_batches["history_only"]["all_his"], view_batches["search_masked"]["all_his"])
            )
            view_scores: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for view, vb in view_batches.items():
                    view_scores[view] = model.rec_predict(vb)
                    if not torch.isfinite(view_scores[view]).all():
                        raise RuntimeError(f"non-finite scores in view {view}")
                for level, lookup in swap_lookups.items():
                    repl_ids = replacement_ids_for_level(row_indices, selected, lookup)
                    repl = torch.tensor(repl_ids, dtype=torch.long, device=device)
                    swapped_batch = replace_matched_source_only(batch, repl)
                    frozen_field_checks.append(compare_frozen_tensors(batch, swapped_batch))
                    swap_change_checks.extend(
                        int(changed) == (1 if int(replacement) > 0 else 0)
                        for changed, replacement in zip(
                            changed_slot_counts(batch, swapped_batch).detach().cpu().tolist(),
                            repl_ids,
                        )
                    )
                    view_scores[level] = model.rec_predict(swapped_batch)
                    if not torch.isfinite(view_scores[level]).all():
                        raise RuntimeError(f"non-finite scores in swap level {level}")
                    if args.reference_source_mode == "matched_random":
                        if reference_swap_lookups is None:
                            raise RuntimeError("matched_random reference mode requires reference swap lookups")
                        ref_repl_ids = replacement_ids_for_level(row_indices, selected, reference_swap_lookups[level])
                        ref_repl = torch.tensor(ref_repl_ids, dtype=torch.long, device=device)
                        ref_key = f"{level}_reference"
                        ref_batch = replace_matched_source_only(batch, ref_repl)
                        frozen_field_checks.append(compare_frozen_tensors(batch, ref_batch))
                        swap_change_checks.extend(
                            int(changed) == (1 if int(replacement) > 0 else 0)
                            for changed, replacement in zip(
                                changed_slot_counts(batch, ref_batch).detach().cpu().tolist(),
                                ref_repl_ids,
                            )
                        )
                        view_scores[ref_key] = model.rec_predict(ref_batch)
                        if not torch.isfinite(view_scores[ref_key]).all():
                            raise RuntimeError(f"non-finite scores in reference level {level}")

            rr: dict[str, list[float]] = {}
            rank: dict[str, list[int]] = {}
            target_score: dict[str, list[float]] = {}
            for view, scores in view_scores.items():
                rr[view], rank[view] = rr_from_scores(scores)
                rank_rr_checks.append(rank_rr_identity_pass(rr[view], rank[view]))
                target_score[view] = [float(x) for x in scores[:, 0].detach().cpu().tolist()]
            for pos, idx in enumerate(row_indices):
                rec = selected[idx]
                row = {
                    "row_index": idx,
                    "official_row_index": rec["official_row_index"],
                    "example_id": rec["example_id"],
                    "split": rec["split"],
                    "user_id": rec["user_id"],
                    "target_item": rec["primary_target_item"],
                    "source_session_id": rec["official_source_session_id"],
                    "source_slot_count": int(slot_counts[pos].detach().cpu().item()),
                    "candidate_count": 1 + int(batch["neg_items"].shape[1]),
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
                }
                for level, lookup in swap_lookups.items():
                    primary_matched = lookup.get(idx) is not None
                    reference_lookup = reference_swap_lookups[level] if reference_swap_lookups is not None else None
                    reference_matched = (
                        reference_lookup.get(idx) is not None
                        if reference_lookup is not None
                        else True
                    )
                    matched = primary_matched and reference_matched
                    if args.reference_source_mode == "true":
                        ref_rr = rr["full_sar"][pos]
                        ref_score = target_score["full_sar"][pos]
                        ref_source_session_id = rec["official_source_session_id"]
                        ref_row_index: int | str = rec["official_row_index"]
                    elif args.reference_source_mode == "matched_random":
                        ref_key = f"{level}_reference"
                        ref_rr = rr[ref_key][pos]
                        ref_score = target_score[ref_key][pos]
                        if reference_lookup is None or reference_lookup.get(idx) is None:
                            ref_source_session_id = ""
                            ref_row_index = ""
                        else:
                            ref_row_index = int(reference_lookup[idx])
                            ref_source_session_id = int(selected[int(reference_lookup[idx])]["official_source_session_id"])
                    else:
                        raise ValueError(args.reference_source_mode)
                    row[f"{level}_matched"] = int(matched)
                    row[f"{level}_reference_row_index"] = "" if not matched else ref_row_index
                    row[f"{level}_reference_source_session_id"] = "" if not matched else ref_source_session_id
                    row[f"{level}_reference_rr"] = ref_rr if matched else 0.0
                    row[f"{level}_swap_row_index"] = "" if not primary_matched else int(lookup[idx])
                    row[f"{level}_swap_source_session_id"] = "" if not primary_matched else int(selected[int(lookup[idx])]["official_source_session_id"])
                    row[f"{level}_swap_rr"] = rr[level][pos] if matched else 0.0
                    row[f"{level}_rr_delta"] = ref_rr - rr[level][pos] if matched else 0.0
                    row[f"{level}_score_delta"] = ref_score - target_score[level][pos] if matched else 0.0
                detail_rows.append(row)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    metrics = {
        "rows": len(detail_rows),
        "full_sar_rr": mean([float(r["full_sar_rr"]) for r in detail_rows]),
        "history_only_rr": mean([float(r["history_only_rr"]) for r in detail_rows]),
        "search_only_rr": mean([float(r["search_only_rr"]) for r in detail_rows]),
        "search_masked_rr": mean([float(r["search_masked_rr"]) for r in detail_rows]),
        "full_minus_masked_rr": mean([float(r["full_sar_rr"]) - float(r["search_masked_rr"]) for r in detail_rows]),
        "full_minus_history_rr": mean([float(r["full_sar_rr"]) - float(r["history_only_rr"]) for r in detail_rows]),
    }
    for level in EXPECTED_LEVELS:
        metrics[level] = summarize_level(detail_rows, level)

    complete_ladder = all(int(metrics[level]["matched"]) > 0 for level in EXPECTED_LEVELS)
    focal_contract_pass = (
        bool(focal_id_checks)
        and all(focal_id_checks)
        and bool(focal_mask_change_checks)
        and all(focal_mask_change_checks)
        and bool(history_mask_change_checks)
        and all(history_mask_change_checks)
        and bool(swap_change_checks)
        and all(swap_change_checks)
        and bool(frozen_field_checks)
        and all(frozen_field_checks)
        and bool(rank_rr_checks)
        and all(rank_rr_checks)
    )
    status = (
        "PASS_OFFICIAL_PROCESSED_CHECKPOINT_READOUT"
        if args.checkpoint
        and detail_rows
        and complete_ladder
        and focal_contract_pass
        and args.reference_source_mode == "true"
        else "PASS_OFFICIAL_PROCESSED_SOURCE_ABLATION_CONTROL"
        if args.checkpoint and detail_rows and complete_ladder and focal_contract_pass
        else "PASS_OFFICIAL_PROCESSED_INTERFACE_SMOKE"
        if detail_rows and complete_ladder and focal_contract_pass
        else "BLOCK_FOCAL_ONLY_CONTRACT"
        if detail_rows and complete_ladder
        else "BLOCK_OFFICIAL_PROCESSED_INCOMPLETE"
    )
    payload = {
        "schema": "focalswap.r208.unisar_official_processed_readout.v2",
        "status": status,
        "mode": mode,
        "contract_id": CONTRACT_ID,
        "surface_id": "K_CHECKPOINT_UNISAR_R208_OFFICIAL_PROCESSED",
        "official_repo": str(official_repo),
        "official_commit": get_git_commit(official_repo),
        "processed_data_root": str(source_root),
        "work_data_root": prep["work_data_root"],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_load": checkpoint_load_info,
        "split": args.split,
        "selected_rows": len(selected),
        "eval_rows": len(detail_rows),
        "processed_split_rows": int(rec_df.shape[0]),
        "reference_source_mode": args.reference_source_mode,
        "reference_seed": reference_seed if args.reference_source_mode == "matched_random" else None,
        "candidate_tensors_frozen": bool(frozen_field_checks) and all(frozen_field_checks),
        "frozen_tensor_fields": ["item", "neg_items", "user", "all_his_ts", "all_his_type"],
        "rank_rr_identity_pass": bool(rank_rr_checks) and all(rank_rr_checks),
        "focal_only_contract": {
            "declared_focal_matches_latest_source_slot": bool(focal_id_checks) and all(focal_id_checks),
            "search_masked_changes_exactly_one_slot": bool(focal_mask_change_checks)
            and all(focal_mask_change_checks),
            "history_only_changes_all_source_slots": bool(history_mask_change_checks)
            and all(history_mask_change_checks),
            "matched_swap_changes_exactly_one_or_zero_slots": bool(swap_change_checks)
            and all(swap_change_checks),
            "source_slot_count_histogram": {
                str(key): int(value) for key, value in sorted(source_slot_histogram.items())
            },
            "rows_with_multiple_source_slots": int(
                sum(value for key, value in source_slot_histogram.items() if key > 1)
            ),
        },
        "history_only_equals_search_masked_input": bool(view_input_equal["history_vs_masked"]) and all(view_input_equal["history_vs_masked"]),
        "neutral_src_session_id": prep["neutral_src_session_id"],
        "neutral_item_id": prep["neutral_item_id"],
        "neutral_item_policy": prep["neutral_item_policy"],
        "swap_builder": "r208_official_processed_source_profile_no_target_label",
        "rr_tie_policy": "midpoint rank among exact candidate-score ties",
        "view_definitions": {
            "history_only": "all non-padding source sessions replaced by the neutral source id",
            "search_masked": "only the latest non-padding source session replaced; prior source history fixed",
            "search_only": "recommendation history and prior source sessions neutralized; latest focal source retained",
            "counterfactual_search": "only the latest non-padding source session replaced by a matched donor",
        },
        "swap_level_rules": level_rules,
        "swap_summary": swap_summary,
        "reference_swap_summary": reference_swap_summary,
        "metrics": metrics,
        "claim_ceiling": (
            "checkpoint-backed focal-only official-processed UniSAR readout only when every focal-only contract assertion passes, "
            "mode is checkpoint_readout, and reference_source_mode is true; "
            "matched_random reference-source mode is a reverse/source-ablation control, not a promoted focal-transfer estimate; "
            "random-init smoke validates the scoring and swap interface but is not model evidence"
        ),
    }
    prefix = args.report_prefix
    csv_path = reports_dir / f"{prefix}_scores.csv"
    json_path = reports_dir / f"{prefix}_readout.json"
    md_path = reports_dir / f"{prefix}_readout.md"
    ladder_path = reports_dir / f"{prefix}_swap_ladder.csv"
    write_csv(csv_path, json_clean(detail_rows))
    ladder_rows = []
    for level in EXPECTED_LEVELS:
        s = metrics[level]
        effective_matched = int(s["matched"])
        ladder_rows.append({
            "surface_id": payload["surface_id"],
            "mode": mode,
            "reference_source_mode": args.reference_source_mode,
            "level": level,
            "attempted": swap_summary[level]["attempted"],
            "primary_matched": swap_summary[level]["matched"],
            "reference_matched": (
                None
                if reference_swap_summary is None
                else reference_swap_summary[level]["matched"]
            ),
            "matched": effective_matched,
            "coverage": frac(effective_matched, int(swap_summary[level]["attempted"])),
            **s,
        })
    write_csv(ladder_path, json_clean(ladder_rows))
    payload["outputs"] = {
        "scores_csv": manifest_path(csv_path),
        "swap_ladder_csv": manifest_path(ladder_path),
        "json": manifest_path(json_path),
        "md": manifest_path(md_path),
    }
    write_json(json_path, json_clean(payload))
    lines = [
        "# R208 UniSAR Official-Processed FocalSwap Readout",
        "",
        f"Status: `{status}`",
        f"Mode: `{mode}`",
        f"Reference source mode: `{args.reference_source_mode}`",
        f"Contract: `{CONTRACT_ID}`",
        f"Split / rows: `{args.split}` / `{len(detail_rows)}`",
        f"Official commit: `{payload['official_commit']}`",
        f"Checkpoint strict load: `{payload['checkpoint_load']['strict']}`",
        f"Ignored checkpoint keys: `{len(payload['checkpoint_load']['unexpected_keys'])}`",
        f"RR tie policy: `{payload['rr_tie_policy']}`",
        f"Candidate tensors frozen: `{payload['candidate_tensors_frozen']}`",
        f"Frozen tensor fields: `{payload['frozen_tensor_fields']}`",
        f"Rank/RR identity: `{payload['rank_rr_identity_pass']}`",
        f"Declared focal matches latest source slot: `{payload['focal_only_contract']['declared_focal_matches_latest_source_slot']}`",
        f"Search-masked changes exactly one slot: `{payload['focal_only_contract']['search_masked_changes_exactly_one_slot']}`",
        f"Matched swaps change exactly one or zero slots: `{payload['focal_only_contract']['matched_swap_changes_exactly_one_or_zero_slots']}`",
        f"Source-slot histogram: `{payload['focal_only_contract']['source_slot_count_histogram']}`",
        f"History-only equals Search-masked input: `{payload['history_only_equals_search_masked_input']}`",
        f"History-only definition: {payload['view_definitions']['history_only']}.",
        "",
        "## Five Views",
        "",
        f"- Full RR: `{metrics['full_sar_rr']:.6f}`",
        f"- History-only RR: `{metrics['history_only_rr']:.6f}`",
        f"- Search-only RR: `{metrics['search_only_rr']:.6f}`",
        f"- Search-masked RR: `{metrics['search_masked_rr']:.6f}`",
        f"- Full - Search-masked RR: `{metrics['full_minus_masked_rr']:.6f}`",
        "",
        "## Swap Ladder",
        "",
        "| Level | Matched | Coverage | Strict win | Tie | Half-tie | Mean RR delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level in EXPECTED_LEVELS:
        s = metrics[level]
        lines.append(
            f"| `{level}` | {int(s['matched'])} | {s['matched'] / max(len(detail_rows), 1):.3f} | "
            f"{s['strict_win']:.3f} | {s['tie']:.3f} | {s['half_tie']:.3f} | {s['mean_rr_delta']:.6f} |"
        )
    lines.extend([
        "",
        "## Swap Rules",
        "",
    ])
    for level in EXPECTED_LEVELS:
        lines.append(f"- `{level}`: {level_rules[level]}; future label/score used: `False`.")
    lines.extend([
        "",
        "## Boundary",
        "",
        payload["claim_ceiling"],
    ])
    write_md(md_path, "\n".join(lines))
    print(json.dumps({"status": status, "mode": mode, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
