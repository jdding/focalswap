#!/usr/bin/env python3
"""R208 UniSAR local preflight before any GPU launch.

This script is deliberately a gate, not a paper-result generator.  It checks
whether the current local UniSAR path can support the R208 contract:

* native/performance-anchor extraction from existing official-quality reports;
* row/candidate/split contract validation;
* <=128-row CPU five-view smoke through the official UniSAR rec_predict path;
* L1/L2 plus category-broken/source and same-category/source swap availability;
* explicit claim decision for whether the GPU gate can be unlocked.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
REPORTS = ROOT / "reports"
REFINE = ROOT / "refine-logs"
EXAMPLES = ROOT / "data/processed/backbone_audit/examples.jsonl"
ITEM_METADATA = ROOT / "data/processed/item_metadata.parquet"
DEFAULT_OFFICIAL_CANDIDATES = [
    Path("/private/tmp/focalswap_r225_official_20260710"),
    Path("/private/tmp/focalswap_r172_official_repos_20260618"),
    Path("/private/tmp/sarcheck_r127_official_repos_20260611"),
]
EXPECTED_SWAP_LEVELS = [
    "l1_nuisance",
    "l2_same_category",
    "category_broken_source",
    "same_category_source",
]

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories  # noqa: E402
from behavior_oracle_audit import profile as r174_profile  # noqa: E402
from transition_backbone import (  # noqa: E402
    build_swap_indices,
    build_swap_profiles,
    pick_swap as pick_l1_l2_swap,
)
from official_adapter import (  # noqa: E402
    build_unisar_tiny_bundle,
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
    midpoint_rr_from_scores as rr_from_scores,
    rank_rr_identity_pass,
)
from swap_ladder_category_broken import (  # noqa: E402
    LEVELS as R174_LEVELS,
    build_indices as build_r174_indices,
    pick_swap as pick_r174_swap,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", default="")
    p.add_argument("--examples-path", default=str(EXAMPLES))
    p.add_argument("--item-metadata", default=str(ITEM_METADATA))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument(
        "--work-dir",
        default="",
        help="Run-specific scratch directory; defaults to a report-prefix/PID path.",
    )
    p.add_argument("--report-prefix", default="r208_unisar")
    p.add_argument("--row-limit", type=int, default=128)
    p.add_argument("--max-eval-rows", type=int, default=48)
    p.add_argument("--neg-k", type=int, default=31)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--model-batch-size", type=int, default=32)
    p.add_argument("--swap-tries", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260620)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--candidate-ledger",
        default="",
        help="Optional R225 canonical raw-item candidate ledger.",
    )
    p.add_argument(
        "--focal-contract-report",
        default=str(REPORTS / "r230_focal_slot_contract_preflight_20260711.json"),
        help="Deterministic multi-slot focal transform contract report.",
    )
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else math.nan


def frac(num: int, den: int) -> float:
    return float(num / den) if den else math.nan


def json_clean(value: Any) -> Any:
    """Convert non-standard JSON floats to null before writing reports."""
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, dict):
        return {k: json_clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_clean(v) for v in value]
    return value


def rel_to_root(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def resolve_official_root(user_value: str) -> tuple[Path | None, list[str]]:
    candidates = [Path(user_value)] if user_value else DEFAULT_OFFICIAL_CANDIDATES
    checked: list[str] = []
    for root in candidates:
        checked.append(str(root))
        if (root / "UniSAR").exists():
            return root, checked
    return None, checked


def balanced_sample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split", "train"))].append(row)
    required = ["train", "validation", "test"]
    missing = [split for split in required if not by_split.get(split)]
    if missing:
        raise ValueError(f"cannot build balanced sample; missing splits: {missing}")
    per = max(1, limit // len(required))
    selected: list[dict[str, Any]] = []
    for split in required:
        take = per if split != required[-1] else limit - len(selected)
        selected.extend(by_split[split][: max(1, take)])
    return selected[:limit]


def to_r174_profile_row(row: dict[str, Any]) -> dict[str, Any]:
    """Expose a backbone row through the qbridge-style fields R174 expects."""
    if "search_pos_items" in row and "search_neg_items" in row:
        return row
    target_items = list(row.get("Y_post_rec", []))
    features = row.get("features", {}) or {}
    confounders = features.get("raw_confounders") or {}
    return {
        **row,
        "pre_items": list(row.get("H_pre", [])),
        "search_pos_items": list(row.get("E_search_pos", [])),
        "search_neg_items": list(row.get("E_search_neg", [])),
        "y_rank_positive": row.get("primary_target_item") is not None,
        "y_rank_item": row.get("primary_target_item"),
        "y_bridge_items": target_items,
        "confounders": {
            **confounders,
            "post_positive_count": len(target_items),
            "pre_len": len(row.get("H_pre", [])),
            "search_pos_count": len(row.get("E_search_pos", [])),
            "search_neg_count": len(row.get("E_search_neg", [])),
            "source": row.get("source"),
        },
    }


def ladder_aware_sample(
    rows: list[dict[str, Any]],
    limit: int,
    item_cat: dict[int, int],
    seed: int,
    tries: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a small sample that can exercise all R208 swap levels.

    The local smoke should remain small, but a first-N sample can miss rare
    ladder matches.  This sampler finds test anchors with known L1/L2 and
    category-ladder swaps on the local pool, includes the corresponding swap
    rows, then fills the remaining budget with split-balanced rows.
    """
    if limit < 24:
        selected = balanced_sample(rows, limit)
        return selected, {
            "sampling_strategy": "balanced_fallback_limit_too_small",
            "ladder_anchor_count": 0,
        }

    selected_indices: list[int] = []
    seen: set[int] = set()

    def add(idx: int) -> None:
        if idx not in seen and len(selected_indices) < limit:
            selected_indices.append(idx)
            seen.add(idx)

    # Keep the tiny official bundle split-complete.
    for split in ["train", "validation"]:
        for idx, row in enumerate(rows):
            if str(row.get("split")) == split:
                add(idx)
                break

    rng = random.Random(seed)
    r114_profiles = build_swap_profiles(rows, item_cat)
    r114_indices = build_swap_indices(r114_profiles)
    r174_profiles = [r174_profile(to_r174_profile_row(row), item_cat, r174_args()) for row in rows]
    local_rows_with_search_cats = sum(bool(p.get("search_cats")) for p in r174_profiles)
    test_rows_with_search_cats = sum(
        str(p.get("split")) == "test" and bool(p.get("search_cats")) for p in r174_profiles
    )
    r174_indices = build_r174_indices(r174_profiles)
    level_by_name = {level["level"]: level for level in R174_LEVELS}
    r174_map = {
        "category_broken_source": level_by_name["source_broken"],
        "same_category_source": level_by_name["same_category_source"],
    }

    anchors = 0
    target_anchors = max(4, min(16, limit // 6))
    for idx, row in enumerate(rows):
        if str(row.get("split")) != "test":
            continue
        local_rng = random.Random(rng.randint(0, 10**9) + idx)
        picked: dict[str, int | None] = {
            "l1_nuisance": pick_l1_l2_swap(idx, "L1", r114_profiles, r114_indices, local_rng, tries),
            "l2_same_category": pick_l1_l2_swap(idx, "L2", r114_profiles, r114_indices, local_rng, tries),
            "category_broken_source": pick_r174_swap(idx, r174_map["category_broken_source"], r174_profiles, r174_indices, local_rng, tries),
            "same_category_source": pick_r174_swap(idx, r174_map["same_category_source"], r174_profiles, r174_indices, local_rng, tries),
        }
        if not all(v is not None for v in picked.values()):
            continue
        new_indices = [idx] + [int(v) for v in picked.values() if v is not None]
        if len(seen | set(new_indices)) > limit:
            break
        for new_idx in new_indices:
            add(new_idx)
        anchors += 1
        if anchors >= target_anchors:
            break

    # Fill any remaining rows without disturbing the ladder-aware prefix.
    split_order = ["train", "validation", "test"]
    for split in split_order:
        for idx, row in enumerate(rows):
            if len(selected_indices) >= limit:
                break
            if str(row.get("split")) == split:
                add(idx)
        if len(selected_indices) >= limit:
            break

    if not any(str(rows[idx].get("split")) == "test" for idx in selected_indices):
        # Defensive fallback; build_unisar_tiny_bundle is split-sensitive.
        return balanced_sample(rows, limit), {
            "sampling_strategy": "balanced_fallback_no_test_anchor",
            "ladder_anchor_count": 0,
            "local_rows_with_search_cats": local_rows_with_search_cats,
            "test_rows_with_search_cats": test_rows_with_search_cats,
            "category_ladder_input_status": "BLOCK_BACKBONE_SLICE_HAS_NO_SEARCH_CATEGORY_PROFILE"
            if local_rows_with_search_cats == 0
            else "BLOCK_NO_COMPLETE_LADDER_ANCHOR_IN_LOCAL_SLICE",
        }

    selected = [rows[idx] for idx in selected_indices[:limit]]
    return selected, {
        "sampling_strategy": "ladder_aware_local_smoke",
        "ladder_anchor_count": anchors,
        "target_ladder_anchors": target_anchors,
        "local_rows_with_search_cats": local_rows_with_search_cats,
        "test_rows_with_search_cats": test_rows_with_search_cats,
        "category_ladder_input_status": "PASS_LOCAL_LADDER_INPUT_AVAILABLE",
        "selected_original_indices": selected_indices[:limit],
    }


def extract_quality_anchor(reports_dir: Path) -> dict[str, Any]:
    """Extract existing bounded UniSAR quality metrics without upgrading them."""
    r208 = (
        reports_dir
        / "r208_unisar_official_original_remote_20260620"
        / "R208_UNISAR_OFFICIAL_ORIGINAL_REMOTE_CLOSURE.md"
    )
    r157 = reports_dir / "r157_unisar_official_quality.md"
    r132 = reports_dir / "r132_official_gpu_batch_summary.md"
    text157 = r157.read_text(encoding="utf-8") if r157.exists() else ""
    text132 = r132.read_text(encoding="utf-8") if r132.exists() else ""

    def find(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    if r208.exists():
        text208 = r208.read_text(encoding="utf-8")
        test_section = text208.split("Test:", 1)[1].split("##", 1)[0] if "Test:" in text208 else ""
        native_metrics = {
            "r208_test_rec_hr10": find(r"Recommendation: HR@10 ([0-9.]+), NDCG@10 [0-9.]+", test_section),
            "r208_test_rec_ndcg10": find(r"Recommendation: HR@10 [0-9.]+, NDCG@10 ([0-9.]+)", test_section),
            "r208_test_src_hr10": find(r"Search: HR@10 ([0-9.]+), NDCG@10 [0-9.]+", test_section),
            "r208_test_src_ndcg10": find(r"Search: HR@10 [0-9.]+, NDCG@10 ([0-9.]+)", test_section),
        }
        if all(value is not None for value in native_metrics.values()):
            return {
                "status": "PASS_R208_NATIVE_QUALITY_ANCHOR",
                "quality_class": "official_processed_checkpoint_quality",
                "metrics": native_metrics,
                "missing": [],
                "interpretation": (
                    "R208 records nontrivial native recommendation and search quality on the retained "
                    "official-processed checkpoint. Readout repairs do not alter this training-quality anchor."
                ),
                "sources": {"r208": str(r208.relative_to(ROOT))},
            }

    metrics = {
        "r157_dev_rec_ndcg10": find(r"Dev rec NDCG@10=([0-9.]+)", text157),
        "r157_dev_rec_hr10": find(r"Dev rec NDCG@10=[0-9.]+, HR@10=([0-9.]+)", text157),
        "r157_test_rec_ndcg10": find(r"Test rec NDCG@10=([0-9.]+)", text157),
        "r157_test_rec_hr10": find(r"Test rec NDCG@10=[0-9.]+, HR@10=([0-9.]+)", text157),
        "r132_test_rec_ndcg10_mainloss": find(r"Rec test:.*NDCG@10 ([0-9.]+)", text132),
        "r132_test_rec_hr10_mainloss": find(r"Rec test:.*HR@10 ([0-9.]+)", text132),
    }
    available = {k: v for k, v in metrics.items() if v is not None}
    return {
        "status": "BLOCK_CONVERTED_QUALITY_ONLY",
        "quality_class": "converted_quality_only",
        "metrics": available,
        "missing": [
            "paper_or_readme_reference_values",
            "relative_gap_to_reference",
            "native_official_processed_KuaiSAR_metric_extraction",
            "MRR_or_documented_closest_official_metric",
        ],
        "interpretation": (
            "Existing local metrics are bounded converted-bundle quality anchors. "
            "They are useful for preflight but do not satisfy R208 native/near-native promotion."
        ),
        "sources": {
            "r157": str(r157.relative_to(ROOT)),
            "r132": str(r132.relative_to(ROOT)),
        },
    }


def r174_args() -> argparse.Namespace:
    return argparse.Namespace(
        max_pre_items=12,
        max_search_items=12,
        max_bridge_targets=3,
        max_candidates=80,
    )


def build_swap_lookups(
    selected: list[dict[str, Any]],
    eval_indices: list[int],
    item_cat: dict[int, int],
    seed: int,
    tries: int,
) -> tuple[dict[str, dict[int, int | None]], dict[str, dict[str, Any]]]:
    rng = random.Random(seed)
    profiles = build_swap_profiles(selected, item_cat)
    indices = build_swap_indices(profiles)
    out: dict[str, dict[int, int | None]] = {level: {} for level in EXPECTED_SWAP_LEVELS}
    for idx in eval_indices:
        out["l1_nuisance"][idx] = pick_l1_l2_swap(idx, "L1", profiles, indices, rng, tries)
        out["l2_same_category"][idx] = pick_l1_l2_swap(idx, "L2", profiles, indices, rng, tries)

    args = r174_args()
    r174_profiles = [r174_profile(to_r174_profile_row(row), item_cat, args) for row in selected]
    r174_indices = build_r174_indices(r174_profiles)
    level_by_name = {level["level"]: level for level in R174_LEVELS}
    r174_map = {
        "category_broken_source": level_by_name["source_broken"],
        "same_category_source": level_by_name["same_category_source"],
    }
    for level_name, level in r174_map.items():
        for idx in eval_indices:
            out[level_name][idx] = pick_r174_swap(idx, level, r174_profiles, r174_indices, rng, tries)

    summary: dict[str, dict[str, Any]] = {}
    for level, lookup in out.items():
        matched = sum(v is not None for v in lookup.values())
        summary[level] = {
            "attempted": len(eval_indices),
            "matched": matched,
            "coverage": frac(matched, len(eval_indices)),
        }
    return out, summary


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
    return replace_matched_focal_source(batch, repl)


def run_cpu_smoke(
    args: argparse.Namespace,
    official_root: Path,
    selected: list[dict[str, Any]],
    paths: dict[str, Path],
    candidate_raw_by_example: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    work_dir = Path(args.work_dir)
    reports_dir = Path(args.reports_dir).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    official_repo = official_root / "UniSAR"
    score_rows_path = reports_dir / f"{args.report_prefix}_cpu_smoke_score_rows.csv"
    bundle = build_unisar_tiny_bundle(
        selected,
        work_dir,
        len(selected),
        args.neg_k,
        args.seed,
        score_rows_path=score_rows_path,
        candidate_raw_by_example=candidate_raw_by_example,
    )
    item_cat = load_item_categories(Path(args.item_metadata))
    eval_indices = [i for i, row in enumerate(selected) if str(row.get("split")) == "test"][: args.max_eval_rows]
    swap_lookups, swap_summary = build_swap_lookups(selected, eval_indices, item_cat, args.seed, args.swap_tries)

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
        user_vocab = utils.load_pickle(const.user_vocab)
        rec_dataset = dataset_mod.RecDataSet("test", user_vocab=user_vocab)
        device = torch.device(args.device)
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
        model.eval()

        detail_rows: list[dict[str, Any]] = []
        frozen_field_checks: list[bool] = []
        mask_equivalence_checks: list[bool] = []
        focal_id_checks: list[bool] = []
        focal_mask_change_checks: list[bool] = []
        history_mask_change_checks: list[bool] = []
        swap_change_checks: list[bool] = []
        source_slot_histogram: Counter[int] = Counter()
        rank_rr_checks: list[bool] = []
        for start in range(0, len(eval_indices), args.batch_size):
            row_indices = eval_indices[start : start + args.batch_size]
            local_dataset_idx = list(range(start, start + len(row_indices)))
            feed = [rec_dataset[i] for i in local_dataset_idx]
            batch = tensor_to_device(rec_dataset.collate_batch(feed), device)
            slot_counts = source_slot_counts(batch)
            source_slot_histogram.update(int(value) for value in slot_counts.detach().cpu().tolist())
            expected_focal_ids = [int(idx) + 1 for idx in row_indices]
            actual_focal_ids = focal_source_ids(batch).detach().cpu().tolist()
            focal_id_checks.extend(
                int(actual) == int(expected)
                for actual, expected in zip(actual_focal_ids, expected_focal_ids)
            )
            view_batches = {
                "full_sar": replace_source_view(
                    batch,
                    "full_sar",
                    int(bundle["neutral_src_session_id"]),
                    int(bundle["neutral_item_id"]),
                ),
                "history_only": replace_source_view(
                    batch,
                    "history_only",
                    int(bundle["neutral_src_session_id"]),
                    int(bundle["neutral_item_id"]),
                ),
                "search_only": replace_source_view(
                    batch,
                    "search_only",
                    int(bundle["neutral_src_session_id"]),
                    int(bundle["neutral_item_id"]),
                ),
                "search_masked": replace_source_view(
                    batch,
                    "search_masked",
                    int(bundle["neutral_src_session_id"]),
                    int(bundle["neutral_item_id"]),
                ),
            }
            frozen_field_checks.extend(compare_frozen_tensors(batch, vb) for vb in view_batches.values())
            mask_equivalence_checks.append(torch.equal(view_batches["history_only"]["all_his"], view_batches["search_masked"]["all_his"]))
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
            view_scores: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for view, vb in view_batches.items():
                    view_scores[view] = model.rec_predict(vb)
                for level, lookup in swap_lookups.items():
                    repl_ids = []
                    for idx in row_indices:
                        swap_idx = lookup.get(idx)
                        repl_ids.append(0 if swap_idx is None else int(swap_idx) + 1)
                    repl_tensor = torch.tensor(repl_ids, dtype=torch.long, device=device)
                    swapped_batch = replace_matched_source_only(batch, repl_tensor)
                    frozen_field_checks.append(compare_frozen_tensors(batch, swapped_batch))
                    swap_change_checks.extend(
                        int(changed) == (1 if int(replacement) > 0 else 0)
                        for changed, replacement in zip(
                            changed_slot_counts(batch, swapped_batch).detach().cpu().tolist(),
                            repl_ids,
                        )
                    )
                    view_scores[level] = model.rec_predict(swapped_batch)

            rr: dict[str, list[float]] = {}
            rank: dict[str, list[int]] = {}
            for view, scores in view_scores.items():
                rr[view], rank[view] = rr_from_scores(scores)
                rank_rr_checks.append(rank_rr_identity_pass(rr[view], rank[view]))
            for pos, idx in enumerate(row_indices):
                row = {
                    "row_index": idx,
                    "example_id": selected[idx].get("example_id"),
                    "split": selected[idx].get("split"),
                    "source": selected[idx].get("source"),
                    "full_sar_rr": rr["full_sar"][pos],
                    "history_only_rr": rr["history_only"][pos],
                    "search_only_rr": rr["search_only"][pos],
                    "search_masked_rr": rr["search_masked"][pos],
                    "full_sar_rank": rank["full_sar"][pos],
                    "history_only_rank": rank["history_only"][pos],
                    "search_only_rank": rank["search_only"][pos],
                    "search_masked_rank": rank["search_masked"][pos],
                }
                for level in swap_lookups:
                    matched = swap_lookups[level].get(idx) is not None
                    row[f"{level}_matched"] = int(matched)
                    row[f"{level}_swap_rr"] = rr[level][pos] if matched else 0.0
                    row[f"{level}_true_minus_swap"] = rr["full_sar"][pos] - rr[level][pos] if matched else 0.0
                    row[f"{level}_true_win"] = int(matched and rr["full_sar"][pos] > rr[level][pos])
                detail_rows.append(row)

        write_csv(paths["scores_csv"], detail_rows)
        metrics = {
            "rows": len(detail_rows),
            "full_sar_rr": mean([float(r["full_sar_rr"]) for r in detail_rows]),
            "history_only_rr": mean([float(r["history_only_rr"]) for r in detail_rows]),
            "search_only_rr": mean([float(r["search_only_rr"]) for r in detail_rows]),
            "search_masked_rr": mean([float(r["search_masked_rr"]) for r in detail_rows]),
            "full_minus_masked_rr": mean([float(r["full_sar_rr"]) - float(r["search_masked_rr"]) for r in detail_rows]),
        }
        for level in swap_lookups:
            matched_rows = [r for r in detail_rows if int(r[f"{level}_matched"])]
            metrics[f"{level}_matched"] = len(matched_rows)
            metrics[f"{level}_coverage"] = frac(len(matched_rows), len(detail_rows))
            metrics[f"{level}_strict_win"] = frac(sum(int(r[f"{level}_true_win"]) for r in matched_rows), len(matched_rows))
            metrics[f"{level}_mean_true_minus_swap"] = mean([float(r[f"{level}_true_minus_swap"]) for r in matched_rows])

        required_levels = set(EXPECTED_SWAP_LEVELS)
        complete_ladder = all(swap_summary[level]["matched"] > 0 for level in required_levels)
        l1_l2_complete = all(
            swap_summary[level]["matched"] > 0
            for level in ["l1_nuisance", "l2_same_category"]
        )
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
        if detail_rows and complete_ladder and focal_contract_pass:
            status = "PASS_CPU_INTERFACE_SMOKE_FULL_LADDER"
        elif detail_rows and l1_l2_complete and focal_contract_pass:
            status = "PASS_CPU_INTERFACE_SMOKE_L1L2_ONLY"
        elif detail_rows and complete_ladder:
            status = "BLOCK_FOCAL_ONLY_CONTRACT"
        else:
            status = "BLOCK_INCOMPLETE_LADDER"
        return {
            "status": status,
            "mode": "interface_smoke_random_init",
            "official_repo": str(official_repo),
            "official_commit": get_git_commit(official_repo),
            "bundle": bundle,
            "eval_rows": len(detail_rows),
            "metrics": metrics,
            "swap_summary": swap_summary,
            "local_cpu_candidate_tensors_frozen": bool(frozen_field_checks) and all(frozen_field_checks),
            "frozen_tensor_fields": ["item", "neg_items", "user", "all_his_ts", "all_his_type"],
            "rank_rr_identity_pass": bool(rank_rr_checks) and all(rank_rr_checks),
            "candidate_contract_scope": (
                "local CPU 100-candidate smoke: target plus 99 negatives; promoted full score export still pending"
                if args.neg_k == 99
                else f"local CPU smoke only: target plus {args.neg_k} negatives; not the promoted 100-candidate contract"
            ),
            "history_only_equals_search_masked_input": bool(mask_equivalence_checks) and all(mask_equivalence_checks),
            "mask_semantics": (
                "one-source-slot compact smoke: history-only and search-masked inputs are expected to match; "
                "the deterministic multi-slot contract test verifies their distinction"
            ),
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
            },
            "outputs": {
                "scores_csv": str(paths["scores_csv"].relative_to(ROOT)),
            },
        }
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def row_contract_report(
    all_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    args: argparse.Namespace,
    sample_info: dict[str, Any],
) -> dict[str, Any]:
    counts = Counter(str(r.get("split", "train")) for r in all_rows)
    selected_counts = Counter(str(r.get("split", "train")) for r in selected)
    row_universe_declared = 96994
    category_status = sample_info.get("category_ladder_input_status")
    full_universe = len(all_rows) == row_universe_declared
    split_complete = all(counts.get(split, 0) > 0 for split in ["train", "validation", "test"])
    status = (
        "PASS_FULL_BACKBONE_PRECHECK_UNIVERSE"
        if full_universe and split_complete and category_status == "PASS_LOCAL_LADDER_INPUT_AVAILABLE"
        else "BLOCK_LOCAL_SLICE_NOT_FULL_R208_UNIVERSE"
    )
    return {
        "status": status,
        "examples_path": rel_to_root(Path(args.examples_path)),
        "local_rows": len(all_rows),
        "declared_r208_row_universe": row_universe_declared,
        "promoted_official_processed_test_rows": 157192,
        "split_counts": dict(counts),
        "selected_smoke_rows": len(selected),
        "selected_split_counts": dict(selected_counts),
        "sampling_strategy": sample_info.get("sampling_strategy"),
        "ladder_anchor_count": sample_info.get("ladder_anchor_count"),
        "local_rows_with_search_cats": sample_info.get("local_rows_with_search_cats"),
        "test_rows_with_search_cats": sample_info.get("test_rows_with_search_cats"),
        "category_ladder_input_status": sample_info.get("category_ladder_input_status"),
        "candidate_contract": f"fixed target plus {args.neg_k} negatives in local smoke",
        "held_out_status": (
            "full 96,994-row backbone precheck universe is present; the promoted UniSAR run uses its separate 157,192-row official-processed test split"
            if status.startswith("PASS")
            else "local smoke uses test split only for scoring; promoted run still requires held-out FocalSwap rows from the full 96,994 universe"
        ),
    }


def write_anchor(paths: dict[str, Path], anchor: dict[str, Any]) -> None:
    write_json(paths["anchor_json"], anchor)
    lines = [
        "# R208 UniSAR Native Quality Anchor Preflight",
        "",
        f"Status: `{anchor['status']}`",
        "",
        "## Extracted Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in anchor["metrics"].items():
        lines.append(f"| `{key}` | {value:.4f} |")
    lines.extend([
        "",
        "## Missing for Promotion",
        "",
    ])
    lines.extend(f"- `{item}`" for item in anchor["missing"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        anchor["interpretation"],
    ])
    write_md(paths["anchor_md"], "\n".join(lines))


def write_row_contract(paths: dict[str, Path], report: dict[str, Any]) -> None:
    write_json(paths["row_contract_json"], report)
    lines = [
        "# R208 UniSAR Row Contract Validation",
        "",
        f"Status: `{report['status']}`",
        "",
        f"- Local rows: `{report['local_rows']}`",
        f"- Declared R208 universe: `{report['declared_r208_row_universe']}`",
        f"- Promoted official-processed test rows: `{report['promoted_official_processed_test_rows']}`",
        f"- Split counts: `{report['split_counts']}`",
        f"- Selected smoke rows: `{report['selected_smoke_rows']}`",
        f"- Selected split counts: `{report['selected_split_counts']}`",
        f"- Sampling strategy: `{report.get('sampling_strategy')}`",
        f"- Ladder anchors: `{report.get('ladder_anchor_count')}`",
        f"- Local rows with search categories: `{report.get('local_rows_with_search_cats')}`",
        f"- Test rows with search categories: `{report.get('test_rows_with_search_cats')}`",
        f"- Category ladder input status: `{report.get('category_ladder_input_status')}`",
        f"- Candidate contract: `{report['candidate_contract']}`",
        f"- Held-out status: {report['held_out_status']}",
    ]
    write_md(paths["row_contract_md"], "\n".join(lines))


def write_cpu_smoke(paths: dict[str, Path], smoke: dict[str, Any]) -> None:
    smoke = json_clean(smoke)
    write_json(paths["smoke_json"], smoke)
    ladder_rows = []
    for level, summary in smoke.get("swap_summary", {}).items():
        ladder_rows.append({
            "surface_id": "K_CHECKPOINT_UNISAR_R208_PERF_ANCHORED",
            "mode": smoke.get("mode", ""),
            "level": level,
            "attempted": summary.get("attempted", 0),
            "matched": summary.get("matched", 0),
            "coverage": summary.get("coverage", math.nan),
            "strict_win": smoke.get("metrics", {}).get(f"{level}_strict_win", math.nan),
            "mean_true_minus_swap": smoke.get("metrics", {}).get(f"{level}_mean_true_minus_swap", math.nan),
        })
    write_csv(paths["ladder_csv"], json_clean(ladder_rows))
    lines = [
        "# R208 UniSAR CPU Smoke",
        "",
        f"Status: `{smoke['status']}`",
        f"Mode: `{smoke.get('mode', '')}`",
        "",
        f"- Eval rows: `{smoke.get('eval_rows', 0)}`",
        f"- Official repo: `{smoke.get('official_repo', '')}`",
        f"- Official commit: `{smoke.get('official_commit', '')}`",
        f"- Local CPU candidate tensors frozen: `{smoke.get('local_cpu_candidate_tensors_frozen')}`",
        f"- Candidate contract scope: `{smoke.get('candidate_contract_scope')}`",
        f"- History-only equals Search-masked input: `{smoke.get('history_only_equals_search_masked_input')}`",
        f"- Mask semantics: `{smoke.get('mask_semantics')}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(smoke.get("metrics", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## Swap Summary",
        "",
        "```json",
        json.dumps(smoke.get("swap_summary", {}), indent=2, sort_keys=True),
        "```",
    ]
    write_md(paths["smoke_md"], "\n".join(lines))


def write_claim_decision(paths: dict[str, Path], payload: dict[str, Any]) -> None:
    write_json(paths["claim_json"], payload)
    lines = [
        "# R208 UniSAR Local Preflight Claim Decision",
        "",
        f"Status: `{payload['status']}`",
        "",
        "## Gate Summary",
        "",
    ]
    for key, value in payload["gate_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend([
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Next Required Work",
        "",
    ])
    lines.extend(f"- {item}" for item in payload["next_required_work"])
    write_md(paths["claim_md"], "\n".join(lines))


def main() -> None:
    args = parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.report_prefix
    work_dir = (
        Path(args.work_dir)
        if str(args.work_dir).strip()
        else Path(f"/private/tmp/focalswap_unisar_preflight_{prefix}_{os.getpid()}")
    )
    args.work_dir = str(work_dir)
    paths = {
        "anchor_md": reports_dir / f"{prefix}_native_quality_anchor_local.md",
        "anchor_json": reports_dir / f"{prefix}_native_quality_anchor_local.json",
        "row_contract_md": reports_dir / f"{prefix}_row_contract_validation_local.md",
        "row_contract_json": reports_dir / f"{prefix}_row_contract_validation_local.json",
        "smoke_md": reports_dir / f"{prefix}_cpu_smoke_local.md",
        "smoke_json": reports_dir / f"{prefix}_cpu_smoke_local.json",
        "scores_csv": reports_dir / f"{prefix}_cpu_smoke_scores_local.csv",
        "ladder_csv": reports_dir / f"{prefix}_swap_ladder_local.csv",
        "claim_md": reports_dir / f"{prefix}_claim_decision_local.md",
        "claim_json": reports_dir / f"{prefix}_claim_decision_local.json",
        "balanced_examples": work_dir / "balanced_examples.jsonl",
    }

    all_rows = read_jsonl(Path(args.examples_path))
    item_cat_for_sample = load_item_categories(Path(args.item_metadata))
    selected, sample_info = ladder_aware_sample(
        all_rows,
        args.row_limit,
        item_cat_for_sample,
        args.seed,
        args.swap_tries,
    )
    write_rows_jsonl(paths["balanced_examples"], selected)
    candidate_raw_by_example = None
    if args.candidate_ledger:
        from common_candidate_contract import load_candidate_ledger

        candidate_raw_by_example = load_candidate_ledger(
            Path(args.candidate_ledger),
            {str(row["example_id"]) for row in selected},
        )

    official_root, checked_roots = resolve_official_root(args.official_root)
    anchor = extract_quality_anchor(reports_dir)
    row_contract = row_contract_report(all_rows, selected, args, sample_info)
    write_anchor(paths, anchor)
    write_row_contract(paths, row_contract)

    if official_root is None:
        smoke: dict[str, Any] = {
            "status": "BLOCK_OFFICIAL_REPO_MISSING",
            "checked_roots": checked_roots,
            "eval_rows": 0,
            "metrics": {},
            "swap_summary": {},
        }
    else:
        smoke = run_cpu_smoke(
            args,
            official_root,
            selected,
            paths,
            candidate_raw_by_example,
        )
    write_cpu_smoke(paths, smoke)

    focal_contract_path = Path(args.focal_contract_report)
    focal_contract_payload = (
        json.loads(focal_contract_path.read_text(encoding="utf-8"))
        if focal_contract_path.exists()
        else {}
    )
    focal_assertions = focal_contract_payload.get("assertions", {})
    deterministic_multi_slot_contract = (
        focal_contract_payload.get("status") == "PASS_FOCAL_SLOT_CONTRACT"
        and focal_assertions.get("history_only_masks_all_sources") is True
        and focal_assertions.get("search_masked_changes_only_focal") is True
        and focal_assertions.get("matched_swap_changes_only_focal") is True
        and focal_assertions.get("malformed_replacement_shape_is_rejected") is True
    )

    gate_summary = {
        "native_quality_anchor": anchor["status"],
        "row_contract": row_contract["status"],
        "cpu_smoke": smoke["status"],
        "local_cpu_candidate_tensors_frozen": smoke.get("local_cpu_candidate_tensors_frozen", False),
        "candidate_contract_scope": smoke.get("candidate_contract_scope"),
        "complete_ladder_smoke": (
            set(smoke.get("swap_summary", {}).keys()) == set(EXPECTED_SWAP_LEVELS)
            and all(smoke["swap_summary"][level].get("matched", 0) > 0 for level in EXPECTED_SWAP_LEVELS)
        ),
        "category_ladder_input_status": row_contract.get("category_ladder_input_status"),
        "mask_semantics": smoke.get("mask_semantics", "not_checked"),
        "deterministic_multi_slot_contract": deterministic_multi_slot_contract,
        "focal_contract_report": str(focal_contract_path),
    }
    promoted_ready = (
        anchor["status"].startswith("PASS")
        and row_contract["status"].startswith("PASS")
        and smoke["status"] == "PASS_CPU_INTERFACE_SMOKE_FULL_LADDER"
        and smoke.get("local_cpu_candidate_tensors_frozen") is True
        and gate_summary["complete_ladder_smoke"] is True
        and deterministic_multi_slot_contract
    )
    next_required_work: list[str] = []
    if not anchor["status"].startswith("PASS"):
        next_required_work.append("reconcile UniSAR official paper/README reference metrics and relative gap")
    if not row_contract["status"].startswith("PASS"):
        next_required_work.append("construct or restore the 96,994-row backbone precheck universe")
    else:
        next_required_work.append("run the promoted 157,192-row official-processed five-view and swap-ladder score export")
    if not gate_summary["complete_ladder_smoke"]:
        next_required_work.append("restore full category-broken/source and same-category/source ladder coverage under the promoted row universe")
    if not deterministic_multi_slot_contract:
        next_required_work.append("pass the deterministic multi-slot focal transform contract")
    next_required_work.append("rerun independent code review after the remaining preflight blockers are fixed")

    remaining_summary = "; ".join(next_required_work[:-1])
    decision = {
        "schema": "focalswap.r208.unisar_local_preflight.v2",
        "date": "2026-07-11",
        "status": "READY_FOR_CODE_REVIEW" if promoted_ready else "BLOCK_GPU_LAUNCH",
        "surface_id": "K_CHECKPOINT_UNISAR_R208_PERF_ANCHORED",
        "gate_summary": gate_summary,
        "decision": (
            "Local preflight is sufficient to request independent code review."
            if promoted_ready
            else f"Do not launch GPU. Remaining blockers: {remaining_summary}."
        ),
        "next_required_work": next_required_work,
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in paths.items() if k != "balanced_examples"},
    }
    write_claim_decision(paths, decision)

    print(json.dumps({"status": decision["status"], "claim_decision": str(paths["claim_md"])}, indent=2))


if __name__ == "__main__":
    main()
