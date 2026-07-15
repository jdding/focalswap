#!/usr/bin/env python3
"""R242 category-controlled and same-user source interventions for UniSAR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from controlled_view_ranking import rank_rr_identity_pass  # noqa: E402
from focal_source_views import source_slot_masks  # noqa: E402
from official_adapter import tensor_to_device  # noqa: E402
from unisar_five_view_readout import (  # noqa: E402
    build_candidate_batch,
    build_rows,
    compare_frozen_tensors,
    load_pickle,
    prepare_work_data,
    rr_from_scores,
    select_eval_rows,
    split_file,
)
from unisar_source_scope_localization import (  # noqa: E402
    EXPECTED_OFFICIAL_UNISAR_COMMIT,
    build_valid_source_derangement,
    collect_observed_source_counts,
    load_model,
    nonzero_count,
    replace_scope,
    sha256_file,
    stable_slot_key,
)


CONTRACT_ID = "R242_UNISAR_CATEGORY_CONTROL_LADDER"
SCOPES = ("latest_1", "all")
VIEWS = ("category_latest_1", "category_all", "same_user_latest_1", "same_user_all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--official-root", default="")
    parser.add_argument("--processed-data-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--report-prefix", default="r242_unisar")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--expected-split-rows", type=int, default=157192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20230601)
    parser.add_argument("--intervention-seed", type=int, default=20260712)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def item_count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def frequency_bucket(value: int) -> str:
    if value <= 1:
        return "1"
    exponent = int(np.floor(np.log2(value)))
    return f"2^{exponent}..2^{exponent + 1}-1"


def source_category_sets(
    src_map: dict[str, np.ndarray], item_categories: np.ndarray
) -> list[frozenset[int]]:
    output: list[frozenset[int]] = []
    for row in np.asarray(src_map["pos_items"]):
        categories = {
            int(item_categories[int(item_id)])
            for item_id in row
            if int(item_id) > 0
            and int(item_id) < len(item_categories)
            and int(item_categories[int(item_id)]) > 0
        }
        output.append(frozenset(categories))
    return output


def build_category_donor_map(
    src_map: dict[str, np.ndarray],
    item_categories: np.ndarray,
    observed_source_counts: Counter[int],
    neutral_source_id: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build a one-to-one derangement on strict category/profile strata."""
    source_rows = int(np.asarray(src_map["keyword"]).shape[0])
    if neutral_source_id != source_rows:
        raise RuntimeError("neutral source ID must equal the original vocabulary size")
    valid_ids = sorted(int(value) for value in observed_source_counts)
    categories = source_category_sets(src_map, item_categories)
    profile_exact: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    profiles: dict[int, tuple[str, str, str]] = {}
    for source_id in valid_ids:
        if source_id <= 0 or source_id >= source_rows:
            raise RuntimeError("observed support contains an invalid source ID")
        item_bucket = item_count_bucket(nonzero_count(src_map["pos_items"][source_id]))
        query_bucket = item_count_bucket(nonzero_count(src_map["keyword"][source_id]))
        count_bucket = frequency_bucket(observed_source_counts[source_id])
        profiles[source_id] = (item_bucket, query_bucket, count_bucket)
        for category in categories[source_id]:
            profile_exact[(category, item_bucket, query_bucket, count_bucket)].append(source_id)

    mapping = torch.zeros(neutral_source_id + 1, dtype=torch.long)
    route_counts: Counter[str] = Counter()
    eligible_occurrences = 0
    assigned_groups: dict[tuple[int, str, str, str], list[int]] = defaultdict(list)
    for source_id in valid_ids:
        if not categories[source_id]:
            route_counts["no_category"] += 1
            continue
        item_bucket, query_bucket, count_bucket = profiles[source_id]
        candidate_categories = sorted(
            categories[source_id],
            key=lambda category: (
                -len(profile_exact[(category, item_bucket, query_bucket, count_bucket)]),
                category,
            ),
        )
        canonical = candidate_categories[0]
        assigned_groups[(canonical, item_bucket, query_bucket, count_bucket)].append(source_id)

    for group_position, (key, source_ids) in enumerate(sorted(assigned_groups.items())):
        values = sorted(set(source_ids))
        if len(values) < 2:
            route_counts["strict_profile_singleton"] += len(values)
            continue
        offset = 1 + stable_slot_key(len(values), group_position, seed) % (len(values) - 1)
        for position, source_id in enumerate(values):
            donor_id = values[(position + offset) % len(values)]
            mapping[source_id] = donor_id
            eligible_occurrences += int(observed_source_counts[source_id])
            route_counts["strict_category_profile_cycle"] += 1

    mapped_ids = [source_id for source_id in valid_ids if int(mapping[source_id]) > 0]
    if any(int(mapping[source_id]) == source_id for source_id in mapped_ids):
        raise RuntimeError("category donor map contains a fixed point")
    mapped_donors = [int(mapping[source_id]) for source_id in mapped_ids]
    if sorted(mapped_donors) != sorted(mapped_ids):
        raise RuntimeError("category donor map is not a bijection on eligible support")
    if any(not (categories[source_id] & categories[int(mapping[source_id])]) for source_id in mapped_ids):
        raise RuntimeError("category donor map contains an overlap failure")
    if any(profiles[source_id] != profiles[int(mapping[source_id])] for source_id in mapped_ids):
        raise RuntimeError("category donor map crosses a strict profile stratum")
    induced_counts: Counter[int] = Counter()
    for source_id in mapped_ids:
        induced_counts[int(mapping[source_id])] += int(observed_source_counts[source_id])
    total_occurrences = int(sum(observed_source_counts.values()))
    support = sorted(set(mapped_ids) | set(induced_counts))
    induced_tv = (
        0.5
        * sum(
            abs(int(observed_source_counts[source_id]) - int(induced_counts[source_id]))
            for source_id in support
        )
        / max(eligible_occurrences, 1)
    )
    category_contract = [
        [source_id, int(mapping[source_id]), sorted(categories[source_id] & categories[int(mapping[source_id])])]
        for source_id in mapped_ids
    ]
    return mapping, {
        "valid_source_ids": len(valid_ids),
        "eligible_source_ids": len(mapped_ids),
        "eligible_source_id_fraction": len(mapped_ids) / max(len(valid_ids), 1),
        "observed_source_slot_occurrences": total_occurrences,
        "eligible_source_slot_occurrences": eligible_occurrences,
        "eligible_occurrence_fraction": eligible_occurrences / max(total_occurrences, 1),
        "route_counts": dict(sorted(route_counts.items())),
        "bijection_on_eligible_support": True,
        "fixed_points": 0,
        "category_overlap_failures": 0,
        "profile_mismatch_failures": 0,
        "all_donors_observed_and_nonempty": True,
        "max_donor_reuse": 1 if mapped_ids else 0,
        "induced_occurrence_total_variation_on_eligible_support": float(induced_tv),
        "category_contract_sha256": sha256_json(category_contract),
        "donor_map_sha256": hashlib.sha256(mapping.numpy().tobytes()).hexdigest(),
        "selection_inputs": [
            "observed source-session ID",
            "source item-count bucket",
            "query token-count bucket",
            "test-slot occurrence-count bucket",
            "source first-level item categories",
            "fixed intervention seed",
        ],
        "forbidden_inputs": ["user", "target", "candidate", "rank", "label", "model score"],
        "future_label_or_score_used": False,
    }


def scope_masks(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    source_mask, _ = source_slot_masks(batch)
    latest = torch.zeros_like(source_mask, dtype=torch.bool)
    for row in range(source_mask.shape[0]):
        positions = torch.nonzero(source_mask[row], as_tuple=False).flatten()
        if len(positions):
            latest[row, positions[-1]] = True
    return {"latest_1": latest, "all": source_mask}


def category_control_view(
    batch: dict[str, Any], selected_mask: torch.Tensor, donor_map: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    donor_map = donor_map.to(batch["all_his"].device)
    original = batch["all_his"]
    donor_ids = donor_map[original.clamp(min=0, max=len(donor_map) - 1)]
    selected_count = selected_mask.sum(dim=1)
    available_count = ((donor_ids > 0) & selected_mask).sum(dim=1)
    eligible = (selected_count > 0) & (available_count == selected_count)
    applied = selected_mask & eligible.unsqueeze(1)
    output = dict(batch)
    output["all_his"] = original.clone()
    output["all_his"][applied] = donor_ids[applied]
    changed = ((output["all_his"] != original) & applied).sum(dim=1)
    if not torch.equal(changed, torch.where(eligible, selected_count, torch.zeros_like(selected_count))):
        raise RuntimeError("category control did not change every eligible selected slot")
    return output, eligible, changed


def same_user_temporal_views(
    batch: dict[str, Any], source_mask: torch.Tensor
) -> tuple[dict[str, dict[str, Any]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    original = batch["all_his"]
    latest_output = original.clone()
    all_output = original.clone()
    latest_eligible = torch.zeros(original.shape[0], dtype=torch.bool, device=original.device)
    all_eligible = torch.zeros_like(latest_eligible)
    latest_changed = torch.zeros(original.shape[0], dtype=torch.long, device=original.device)
    all_changed = torch.zeros_like(latest_changed)
    for row in range(original.shape[0]):
        positions = torch.nonzero(source_mask[row], as_tuple=False).flatten().tolist()
        values = [int(original[row, position].item()) for position in positions]
        if len(values) >= 2:
            for donor in reversed(values[:-1]):
                if donor != values[-1]:
                    latest_output[row, positions[-1]] = donor
                    latest_eligible[row] = True
                    latest_changed[row] = 1
                    break
            for shift in range(1, len(values)):
                donors = values[-shift:] + values[:-shift]
                if all(donor != source for donor, source in zip(donors, values)):
                    all_output[row, positions] = torch.tensor(
                        donors, dtype=original.dtype, device=original.device
                    )
                    all_eligible[row] = True
                    all_changed[row] = len(values)
                    break
    views = {
        "same_user_latest_1": {**batch, "all_his": latest_output},
        "same_user_all": {**batch, "all_his": all_output},
    }
    eligibility = {
        "same_user_latest_1": latest_eligible,
        "same_user_all": all_eligible,
    }
    changes = {
        "same_user_latest_1": latest_changed,
        "same_user_all": all_changed,
    }
    return views, eligibility, changes


def build_execution_lineage(
    source_root: Path, official_repo: Path, checkpoint_path: Path, split: str
) -> dict[str, Any]:
    processed_paths = [source_root / "dataset" / split_file(split)]
    processed_paths.extend(sorted((source_root / "vocab").glob("*.pkl")))
    official_paths = [
        official_repo / "models" / "UniSAR.py",
        official_repo / "utils" / "const.py",
        official_repo / "utils" / "dataset.py",
        official_repo / "utils" / "utils.py",
    ]
    local_paths = [
        Path(__file__).resolve(),
        ROOT / "tools" / "unisar_source_scope_localization.py",
        ROOT / "tools" / "unisar_five_view_readout.py",
        ROOT / "tools" / "focal_source_views.py",
        ROOT / "tools" / "controlled_view_ranking.py",
        ROOT / "tools" / "official_adapter.py",
    ]
    commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(official_repo), "status", "--porcelain", "--", *[
            str(path.relative_to(official_repo)) for path in official_paths
        ]],
        text=True,
    ).strip()
    if commit != EXPECTED_OFFICIAL_UNISAR_COMMIT or status:
        raise RuntimeError("official UniSAR scorer lineage is not clean and frozen")
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_git_commit": commit,
        "official_reviewed_files_clean": True,
        "processed_input_sha256": {str(path): sha256_file(path) for path in processed_paths},
        "official_code_sha256": {str(path): sha256_file(path) for path in official_paths},
        "executed_local_code_sha256": {str(path): sha256_file(path) for path in local_paths},
    }


def self_test() -> None:
    src_map = {
        "pos_items": np.asarray([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]),
        "keyword": np.asarray([[0], [1], [2], [3], [4], [5]]),
    }
    item_categories = np.asarray([0, 1, 1, 2, 2, 1])
    counts = Counter({1: 4, 2: 5, 3: 4, 4: 5, 5: 6})
    donor_map, summary = build_category_donor_map(src_map, item_categories, counts, 6, 99)
    if int(donor_map[1]) not in (2, 5) or int(donor_map[3]) != 4:
        raise AssertionError("category donor construction failed")
    if summary["category_overlap_failures"] != 0:
        raise AssertionError("category overlap summary failed")
    batch = {
        "all_his": torch.tensor([[11, 1, 12, 2, 0], [21, 3, 22, 4, 23]]),
        "all_his_type": torch.tensor([[1, 2, 1, 2, 0], [1, 2, 1, 2, 1]]),
        "all_his_ts": torch.zeros((2, 5), dtype=torch.long),
        "item": torch.tensor([1, 2]),
        "neg_items": torch.tensor([[3, 4], [5, 6]]),
        "user": torch.tensor([1, 2]),
    }
    masks = scope_masks(batch)
    category_view, eligible, changed = category_control_view(batch, masks["all"], donor_map)
    if not bool(eligible[0]) or int(changed[0]) != 2:
        raise AssertionError("category all-scope replacement failed")
    if not compare_frozen_tensors(batch, category_view):
        raise AssertionError("category view changed a frozen tensor")
    temporal, temporal_eligible, temporal_changed = same_user_temporal_views(batch, masks["all"])
    if not bool(temporal_eligible["same_user_latest_1"][0]):
        raise AssertionError("same-user latest control failed")
    if int(temporal_changed["same_user_all"][0]) != 2:
        raise AssertionError("same-user all control failed")
    if not compare_frozen_tensors(batch, temporal["same_user_all"]):
        raise AssertionError("same-user view changed a frozen tensor")
    print(json.dumps({"contract": CONTRACT_ID, "self_test": "PASS"}, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = [args.official_root, args.processed_data_root, args.checkpoint, args.work_dir]
    if any(not str(value).strip() for value in required):
        raise ValueError("official root, processed data, checkpoint, and work dir are required")
    random.seed(args.intervention_seed)
    np.random.seed(args.intervention_seed)
    torch.manual_seed(args.intervention_seed)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_csv = reports_dir / f"{args.report_prefix}_control_scores.csv"
    output_json = reports_dir / f"{args.report_prefix}_integrity.json"
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    official_repo = Path(args.official_root) / "UniSAR"
    checkpoint_path = Path(args.checkpoint)
    prep = prepare_work_data(Path(args.processed_data_root), work_dir)
    source_root = Path(prep["source_root"])
    rec_df = pd.read_pickle(source_root / "dataset" / split_file(args.split)).reset_index(drop=True)
    if args.expected_split_rows and len(rec_df) != args.expected_split_rows:
        raise RuntimeError("processed split row count changed")
    user_vocab = load_pickle(source_root / "vocab" / "user_vocab.pkl")
    src_map = load_pickle(source_root / "vocab" / "src_session_vocab_np.pkl")
    item_map = load_pickle(source_root / "vocab" / "item_vocab_np.pkl")
    item_categories = np.asarray(item_map["first_level_category_id"], dtype=np.int64)
    rows = build_rows(rec_df, user_vocab, src_map, args.split)
    selected, eval_indices = select_eval_rows(rows, args.row_limit, args.max_eval_rows)
    lineage = build_execution_lineage(source_root, official_repo, checkpoint_path, args.split)

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    model = None
    sum_rr: dict[str, float] = defaultdict(float)
    eligible_counts: Counter[str] = Counter()
    eligible_users: dict[str, set[int]] = defaultdict(set)
    all_users: set[int] = set()
    candidate_counts: set[int] = set()
    frozen_checks = 0
    rank_rr_checks = 0
    try:
        model, rec_dataset, device, checkpoint_info = load_model(
            official_repo, work_dir, checkpoint_path, args.split, args.device, args.model_batch_size
        )
        observed_counts = collect_observed_source_counts(
            rec_dataset, selected, eval_indices, max(args.batch_size, 1024)
        )
        category_map, category_summary = build_category_donor_map(
            src_map,
            item_categories,
            observed_counts,
            int(prep["neutral_src_session_id"]),
            args.intervention_seed,
        )
        category_map = category_map.to(device)
        fieldnames = [
            "official_row_index",
            "user_id",
            "source_slot_count",
            "candidate_count",
            "full_rr",
            *[f"{view}_eligible" for view in VIEWS],
            *[f"{view}_changed_slots" for view in VIEWS],
            *[f"{view}_rr" for view in VIEWS],
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for start in range(0, len(eval_indices), args.batch_size):
                row_indices = eval_indices[start : start + args.batch_size]
                if start == 0 or start + len(row_indices) >= len(eval_indices):
                    print(f"[R242] seed={args.seed} scoring {start + len(row_indices)}/{len(eval_indices)}", flush=True)
                feed = build_candidate_batch(rec_dataset, row_indices, selected)
                batch = tensor_to_device(rec_dataset.collate_batch(feed), device)
                candidate_count = 1 + int(batch["neg_items"].shape[1])
                candidate_counts.add(candidate_count)
                if candidate_count != 100:
                    raise RuntimeError("candidate count changed")
                masks = scope_masks(batch)
                category_latest, category_latest_ok, category_latest_changed = category_control_view(
                    batch, masks["latest_1"], category_map
                )
                category_all, category_all_ok, category_all_changed = category_control_view(
                    batch, masks["all"], category_map
                )
                temporal_views, temporal_ok, temporal_changed = same_user_temporal_views(
                    batch, masks["all"]
                )
                views = {
                    "category_latest_1": category_latest,
                    "category_all": category_all,
                    **temporal_views,
                }
                eligibility = {
                    "category_latest_1": category_latest_ok,
                    "category_all": category_all_ok,
                    **temporal_ok,
                }
                changes = {
                    "category_latest_1": category_latest_changed,
                    "category_all": category_all_changed,
                    **temporal_changed,
                }
                for name, view in views.items():
                    if not compare_frozen_tensors(batch, view):
                        raise RuntimeError(f"frozen tensor changed in {name}")
                    frozen_checks += 1
                    eligible_counts[name] += int(eligibility[name].sum().item())
                    for position in torch.nonzero(
                        eligibility[name], as_tuple=False
                    ).flatten().cpu().tolist():
                        eligible_users[name].add(int(selected[row_indices[position]]["user_id"]))
                all_users.update(int(selected[index]["user_id"]) for index in row_indices)
                with torch.no_grad():
                    scores = {"full": model.rec_predict(batch)}
                    scores.update({name: model.rec_predict(view) for name, view in views.items()})
                rr: dict[str, list[float]] = {}
                for name, score in scores.items():
                    if tuple(score.shape) != (len(row_indices), candidate_count) or not torch.isfinite(score).all():
                        raise RuntimeError(f"invalid score tensor for {name}")
                    rr[name], ranks = rr_from_scores(score)
                    if not rank_rr_identity_pass(rr[name], ranks):
                        raise RuntimeError(f"rank/RR identity failed for {name}")
                    rank_rr_checks += 1
                    sum_rr[name] += float(sum(rr[name]))
                source_counts = masks["all"].sum(dim=1)
                for position, selected_index in enumerate(row_indices):
                    record = selected[selected_index]
                    output: dict[str, Any] = {
                        "official_row_index": int(record["official_row_index"]),
                        "user_id": int(record["user_id"]),
                        "source_slot_count": int(source_counts[position].item()),
                        "candidate_count": candidate_count,
                        "full_rr": rr["full"][position],
                    }
                    for view in VIEWS:
                        output[f"{view}_eligible"] = int(eligibility[view][position].item())
                        output[f"{view}_changed_slots"] = int(changes[view][position].item())
                        output[f"{view}_rr"] = rr[view][position]
                    writer.writerow(output)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    row_count = len(eval_indices)
    payload = {
        "schema": "focalswap.r242.unisar_control_integrity.v1",
        "contract_id": CONTRACT_ID,
        "seed": args.seed,
        "intervention_seed": args.intervention_seed,
        "split": args.split,
        "processed_split_rows": len(rec_df),
        "scored_rows": row_count,
        "candidate_count": next(iter(candidate_counts)) if len(candidate_counts) == 1 else None,
        "checkpoint_load": checkpoint_info,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "execution_lineage": lineage,
        "category_donor_map": category_summary,
        "eligible_rows": dict(sorted(eligible_counts.items())),
        "eligible_fractions": {
            name: count / row_count for name, count in sorted(eligible_counts.items())
        },
        "user_clusters": len(all_users),
        "eligible_user_clusters": {
            name: len(users) for name, users in sorted(eligible_users.items())
        },
        "eligible_user_fractions": {
            name: len(users) / max(len(all_users), 1)
            for name, users in sorted(eligible_users.items())
        },
        "mean_rr": {name: total / row_count for name, total in sorted(sum_rr.items())},
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256_file(output_csv),
        "integrity": {
            "frozen_tensor_checks": frozen_checks,
            "rank_rr_checks": rank_rr_checks,
            "candidate_counts_seen": sorted(candidate_counts),
            "all_pass": True,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"contract": CONTRACT_ID, "seed": args.seed, "rows": row_count, "integrity": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()
