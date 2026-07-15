#!/usr/bin/env python3
"""R233 source-position and intervention-dose audit for official UniSAR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import random
import subprocess
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
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from focal_source_views import source_slot_counts, source_slot_masks  # noqa: E402
from official_adapter import tensor_to_device  # noqa: E402
from unisar_five_view_readout import (  # noqa: E402
    build_candidate_batch,
    build_rows,
    compare_frozen_tensors,
    disable_nested_transformer_fastpath,
    load_pickle,
    prepare_work_data,
    rr_from_scores,
    select_eval_rows,
    split_file,
)
from controlled_view_ranking import rank_rr_identity_pass  # noqa: E402


CONTRACT_ID = "R233_UNISAR_SOURCE_SCOPE_LOCALIZATION"
EXPECTED_OFFICIAL_UNISAR_COMMIT = "1a14a246066cd1aa6ee789c57ff2c7499f3d7379"
SCOPE_SPECS: dict[str, tuple[str, int]] = {
    "latest_1": ("lag", 1),
    "lag_2": ("lag", 2),
    "lag_3": ("lag", 3),
    "lag_4": ("lag", 4),
    "recent_2": ("recent", 2),
    "recent_4": ("recent", 4),
    "recent_8": ("recent", 8),
    "all": ("all", 0),
    "random_1": ("random", 1),
    "random_2": ("random", 2),
    "random_4": ("random", 4),
    "random_8": ("random", 8),
}
MODES = ("neutral", "valid")
VARIANTS = tuple(f"{mode}_{scope}" for mode in MODES for scope in SCOPE_SPECS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--official-root", default="")
    parser.add_argument("--processed-data-root", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--work-dir", default="")
    parser.add_argument("--reports-dir", default=str(ROOT / "reports"))
    parser.add_argument("--report-prefix", default="r233_unisar_scope")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--row-limit", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--expected-split-rows", type=int, default=157192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--model-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20230601)
    parser.add_argument("--intervention-seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bucket_len(value: int) -> str:
    value = int(value)
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    return "8+"


def occurrence_bucket(value: int) -> str:
    value = int(value)
    if value <= 1:
        return "1"
    exponent = int(np.floor(np.log2(value)))
    return f"2^{exponent}..2^{exponent + 1}-1"


def nonzero_count(values: Any) -> int:
    return int(np.count_nonzero(np.asarray(values)))


def stable_slot_key(row_index: int, position: int, seed: int) -> int:
    """SplitMix64-style deterministic key independent of Python hash state."""
    mask = (1 << 64) - 1
    value = (
        int(seed)
        + (int(row_index) + 1) * 0x9E3779B97F4A7C15
        + (int(position) + 1) * 0xBF58476D1CE4E5B9
    ) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return int(value ^ (value >> 31))


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_file_set(paths: list[Path]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        output[str(path)] = sha256_file(path)
    return output


def build_scope_masks(
    batch: dict[str, Any],
    official_row_indices: list[int],
    seed: int,
) -> dict[str, torch.Tensor]:
    source_mask, _ = source_slot_masks(batch)
    if source_mask.shape[0] != len(official_row_indices):
        raise RuntimeError("row-index count does not match batch size")
    source_device = source_mask.device
    source_mask = source_mask.detach().cpu()
    masks = {
        scope: torch.zeros_like(source_mask, dtype=torch.bool)
        for scope in SCOPE_SPECS
    }
    for row_pos, row_index in enumerate(official_row_indices):
        positions = torch.nonzero(source_mask[row_pos], as_tuple=False).flatten().tolist()
        if not positions:
            raise RuntimeError(f"row {row_index} has no source slots")
        random_order = sorted(
            positions,
            key=lambda position: stable_slot_key(row_index, int(position), seed),
        )
        for scope, (kind, amount) in SCOPE_SPECS.items():
            selected: list[int] = []
            if kind == "all":
                selected = positions
            elif len(positions) >= amount:
                if kind == "lag":
                    selected = [positions[-amount]]
                elif kind == "recent":
                    selected = positions[-amount:]
                elif kind == "random":
                    selected = random_order[:amount]
                else:
                    raise ValueError(kind)
            if selected:
                masks[scope][row_pos, selected] = True
    return {scope: mask.to(source_device) for scope, mask in masks.items()}


def build_valid_source_derangement(
    src_map: dict[str, np.ndarray],
    observed_source_counts: Counter[int],
    neutral_source_id: int,
    seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_rows = int(np.asarray(src_map["keyword"]).shape[0])
    if neutral_source_id != source_rows:
        raise RuntimeError(
            f"neutral source ID {neutral_source_id} must equal original source rows {source_rows}"
        )
    valid_ids = sorted(int(source_id) for source_id in observed_source_counts)
    if len(valid_ids) < 2:
        raise RuntimeError("valid source-session vocabulary is too small for a derangement")
    if any(source_id <= 0 or source_id >= source_rows for source_id in valid_ids):
        raise RuntimeError("observed source support contains padding, neutral, or out-of-range IDs")
    empty_observed_ids = [
        source_id
        for source_id in valid_ids
        if nonzero_count(src_map["pos_items"][source_id]) == 0
        and nonzero_count(src_map["keyword"][source_id]) == 0
    ]
    if empty_observed_ids:
        raise RuntimeError(f"observed source support contains empty IDs: {empty_observed_ids[:8]}")

    primary: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    item_query: dict[tuple[str, str], list[int]] = defaultdict(list)
    secondary: dict[str, list[int]] = defaultdict(list)
    for source_id in valid_ids:
        item_bucket = bucket_len(nonzero_count(src_map["pos_items"][source_id]))
        query_bucket = bucket_len(nonzero_count(src_map["keyword"][source_id]))
        frequency_bucket = occurrence_bucket(observed_source_counts[source_id])
        primary[(item_bucket, query_bucket, frequency_bucket)].append(source_id)
        item_query[(item_bucket, query_bucket)].append(source_id)
        secondary[item_bucket].append(source_id)

    mapping = np.arange(neutral_source_id + 1, dtype=np.int64)
    route_counts: Counter[str] = Counter()

    unresolved = set(valid_ids)

    def assign_cycle(pool: list[int], route: str, salt: int) -> None:
        values = sorted(set(int(value) for value in pool if int(value) in unresolved))
        if len(values) < 2:
            return
        offset = 1 + stable_slot_key(len(values), salt, seed) % (len(values) - 1)
        for position, source_id in enumerate(values):
            mapping[source_id] = values[(position + offset) % len(values)]
            route_counts[route] += 1
            unresolved.remove(source_id)

    for group_position, key in enumerate(sorted(primary)):
        assign_cycle(primary[key], "item_query_and_frequency_bucket", 1000 + group_position)
    for group_position, key in enumerate(sorted(item_query)):
        assign_cycle(item_query[key], "item_and_query_bucket_fallback", 1500 + group_position)
    for group_position, key in enumerate(sorted(secondary)):
        assign_cycle(secondary[key], "item_bucket_fallback", 2000 + group_position)
    assign_cycle(sorted(unresolved), "global_fallback", 3000)
    if len(unresolved) == 1:
        source_id = next(iter(unresolved))
        partner = next(value for value in valid_ids if value != source_id)
        previous_partner_donor = int(mapping[partner])
        mapping[source_id] = previous_partner_donor
        mapping[partner] = source_id
        unresolved.remove(source_id)
        route_counts["global_singleton_swap"] += 1
    if unresolved:
        raise RuntimeError(f"unresolved source IDs in donor map: {sorted(unresolved)[:8]}")
    mapping[0] = 0
    mapping[neutral_source_id] = neutral_source_id
    mapped_valid = [int(mapping[source_id]) for source_id in valid_ids]
    if sorted(mapped_valid) != valid_ids:
        raise RuntimeError("valid source donor map is not a bijection")
    fixed_points = int(sum(mapping[source_id] == source_id for source_id in valid_ids))
    invalid_donors = int(
        sum(mapping[source_id] in (0, neutral_source_id) for source_id in valid_ids)
    )
    induced_counts: Counter[int] = Counter()
    for source_id, count in observed_source_counts.items():
        induced_counts[int(mapping[int(source_id)])] += int(count)
    total_occurrences = int(sum(observed_source_counts.values()))
    total_variation = 0.5 * sum(
        abs(int(observed_source_counts[source_id]) - int(induced_counts[source_id]))
        for source_id in valid_ids
    ) / total_occurrences
    weighted_frequency_bucket_match = sum(
        int(observed_source_counts[source_id])
        for source_id in valid_ids
        if occurrence_bucket(observed_source_counts[source_id])
        == occurrence_bucket(observed_source_counts[int(mapping[source_id])])
    ) / total_occurrences
    observed_support = [
        [int(source_id), int(observed_source_counts[source_id])] for source_id in valid_ids
    ]
    return torch.from_numpy(mapping), {
        "valid_source_ids": len(valid_ids),
        "observed_source_slot_occurrences": total_occurrences,
        "unobserved_vocab_ids_excluded": int((source_rows - 1) - len(valid_ids)),
        "neutral_source_id": int(neutral_source_id),
        "route_counts": dict(sorted(route_counts.items())),
        "bijection": True,
        "fixed_points": fixed_points,
        "padding_or_neutral_donors": invalid_donors,
        "all_donors_observed_and_nonempty": True,
        "observed_support_sha256": sha256_json(observed_support),
        "donor_map_sha256": hashlib.sha256(mapping.tobytes()).hexdigest(),
        "induced_occurrence_total_variation": float(total_variation),
        "occurrence_bucket_match_weighted": float(weighted_frequency_bucket_match),
        "selection_inputs": [
            "observed source-session ID",
            "source item-count bucket",
            "query token-count bucket",
            "test-slot occurrence-count bucket",
            "fixed intervention seed",
        ],
        "forbidden_inputs": ["user", "target", "candidate", "rank", "label", "model score"],
        "future_label_or_score_used": False,
    }


def collect_observed_source_counts(
    rec_dataset: Any,
    selected: list[dict[str, Any]],
    eval_indices: list[int],
    batch_size: int,
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for start in range(0, len(eval_indices), batch_size):
        row_indices = eval_indices[start : start + batch_size]
        feed = build_candidate_batch(rec_dataset, row_indices, selected)
        batch = rec_dataset.collate_batch(feed)
        source_mask, _ = source_slot_masks(batch)
        source_ids = batch["all_his"][source_mask].detach().cpu().tolist()
        counts.update(int(source_id) for source_id in source_ids)
    return counts


def build_execution_lineage(
    source_root: Path,
    official_repo: Path,
    checkpoint_path: Path,
    split: str,
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
        ROOT / "tools" / "unisar_five_view_readout.py",
        ROOT / "tools" / "focal_source_views.py",
        ROOT / "tools" / "controlled_view_ranking.py",
        ROOT / "tools" / "official_adapter.py",
    ]
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    tracked_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(official_repo),
            "status",
            "--porcelain",
            "--",
            *[str(path.relative_to(official_repo)) for path in official_paths],
        ],
        text=True,
    ).strip()
    if official_commit != EXPECTED_OFFICIAL_UNISAR_COMMIT:
        raise RuntimeError(
            f"official UniSAR commit {official_commit} != {EXPECTED_OFFICIAL_UNISAR_COMMIT}"
        )
    if tracked_status:
        raise RuntimeError(f"reviewed official UniSAR scorer files are modified: {tracked_status}")
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "official_git_commit": official_commit,
        "official_reviewed_files_clean": True,
        "processed_input_sha256": hash_file_set(processed_paths),
        "official_code_sha256": hash_file_set(official_paths),
        "executed_local_code_sha256": hash_file_set(local_paths),
    }


def replace_scope(
    batch: dict[str, Any],
    selected_mask: torch.Tensor,
    mode: str,
    neutral_source_id: int,
    valid_donor_map: torch.Tensor,
) -> dict[str, Any]:
    output = dict(batch)
    output["all_his"] = batch["all_his"].clone()
    if mode == "neutral":
        output["all_his"][selected_mask] = int(neutral_source_id)
    elif mode == "valid":
        donor_map = valid_donor_map.to(batch["all_his"].device)
        original = batch["all_his"][selected_mask]
        output["all_his"][selected_mask] = donor_map[original]
    else:
        raise ValueError(mode)
    return output


def expected_scope_changes(scope: str, slot_counts: torch.Tensor) -> torch.Tensor:
    kind, amount = SCOPE_SPECS[scope]
    if kind == "all":
        return slot_counts
    if kind == "lag":
        return (slot_counts >= amount).to(dtype=torch.long)
    return torch.where(
        slot_counts >= amount,
        torch.full_like(slot_counts, amount),
        torch.zeros_like(slot_counts),
    )


def self_test() -> None:
    batch = {
        "all_his": torch.tensor(
            [
                [11, 1, 12, 2, 13, 3, 0, 0, 0, 0],
                [21, 4, 22, 0, 0, 0, 0, 0, 0, 0],
                [5, 6, 7, 8, 9, 10, 11, 12, 31, 32],
            ],
            dtype=torch.long,
        ),
        "all_his_type": torch.tensor(
            [
                [1, 2, 1, 2, 1, 2, 0, 0, 0, 0],
                [1, 2, 1, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 1, 1],
            ],
            dtype=torch.long,
        ),
        "item": torch.tensor([1, 2, 3]),
        "neg_items": torch.tensor([[4, 5], [6, 7], [8, 9]]),
        "user": torch.tensor([1, 2, 3]),
        "all_his_ts": torch.zeros((3, 10), dtype=torch.long),
    }
    row_indices = [10, 11, 12]
    masks_a = build_scope_masks(batch, row_indices, 99)
    masks_b = build_scope_masks(batch, row_indices, 99)
    if not all(torch.equal(masks_a[key], masks_b[key]) for key in masks_a):
        raise AssertionError("random scope masks are not deterministic")
    slot_counts = source_slot_counts(batch)
    if slot_counts.tolist() != [3, 1, 8]:
        raise AssertionError(f"unexpected source counts: {slot_counts.tolist()}")
    if masks_a["latest_1"].sum(dim=1).tolist() != [1, 1, 1]:
        raise AssertionError("latest mask contract failed")
    if masks_a["lag_4"].sum(dim=1).tolist() != [0, 0, 1]:
        raise AssertionError("lag eligibility contract failed")
    if masks_a["recent_8"].sum(dim=1).tolist() != [0, 0, 8]:
        raise AssertionError("recent-dose eligibility contract failed")

    donor_map = torch.arange(0, 101, dtype=torch.long)
    donor_map[1:100] = torch.roll(donor_map[1:100], shifts=-1)
    valid = replace_scope(batch, masks_a["latest_1"], "valid", 100, donor_map)
    neutral = replace_scope(batch, masks_a["all"], "neutral", 100, donor_map)
    if not compare_frozen_tensors(batch, valid) or not compare_frozen_tensors(batch, neutral):
        raise AssertionError("frozen tensors changed in self-test")
    valid_changes = ((valid["all_his"] != batch["all_his"]) & masks_a["latest_1"]).sum(dim=1)
    if valid_changes.tolist() != [1, 1, 1]:
        raise AssertionError("valid replacement did not change every selected slot")
    neutral_changes = ((neutral["all_his"] != batch["all_his"]) & masks_a["all"]).sum(dim=1)
    if neutral_changes.tolist() != slot_counts.tolist():
        raise AssertionError("neutral all replacement contract failed")
    fake_src_map = {
        "pos_items": np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [3, 4, 0],
                [5, 6, 0],
                [7, 8, 9],
            ],
            dtype=np.int64,
        ),
        "keyword": np.asarray(
            [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 4],
                [5, 6],
                [7, 0],
            ],
            dtype=np.int64,
        ),
    }
    fake_counts = Counter({1: 8, 2: 7, 3: 4, 4: 4, 5: 1})
    fake_map, fake_summary = build_valid_source_derangement(fake_src_map, fake_counts, 6, 99)
    if sorted(fake_map[1:6].tolist()) != [1, 2, 3, 4, 5]:
        raise AssertionError("valid-source map is not a bijection")
    if any(fake_map[source_id].item() == source_id for source_id in range(1, 6)):
        raise AssertionError("valid-source map contains a fixed point")
    if not fake_summary["bijection"]:
        raise AssertionError("valid-source map summary did not record bijection")
    print(json.dumps({"contract": CONTRACT_ID, "self_test": "PASS"}, sort_keys=True))


def load_model(
    official_repo: Path,
    work_dir: Path,
    checkpoint_path: Path,
    split: str,
    device_name: str,
    model_batch_size: int,
) -> tuple[Any, Any, torch.device, dict[str, Any]]:
    os.chdir(work_dir)
    sys.path.insert(0, str(official_repo))
    const = importlib.import_module("utils.const")
    utils = importlib.import_module("utils.utils")
    dataset_mod = importlib.import_module("utils.dataset")
    model_mod = importlib.import_module("models.UniSAR")
    const.init_setting_KuaiSAR()
    user_vocab_work = utils.load_pickle(const.user_vocab)
    rec_dataset = dataset_mod.RecDataSet(
        "val" if split == "validation" else "test",
        user_vocab=user_vocab_work,
    )
    requested = torch.device(device_name)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_args = SimpleNamespace(
        device=requested,
        model_path=str(work_dir / "model"),
        dropout=0.0,
        batch_size=model_batch_size,
        num_layers=1,
        num_heads=2,
        q_i_cl_temp=0.5,
        q_i_cl_weight=0.0,
        his_cl_temp=0.1,
        his_cl_weight=0.0,
        pred_hid_units=[200, 80, 1],
    )
    model = model_mod.UniSAR(model_args)
    state_dict = torch.load(checkpoint_path, map_location=requested)
    load_result = model.load_state_dict(state_dict, strict=False)
    allowed_prefixes = (
        "feature_alignment.infoNCE_loss.",
        "rec_his_cl.",
        "src_his_cl.",
    )
    unknown_unexpected = [
        key for key in load_result.unexpected_keys if not key.startswith(allowed_prefixes)
    ]
    if load_result.missing_keys or unknown_unexpected:
        raise RuntimeError(
            "checkpoint load mismatch: "
            f"missing={list(load_result.missing_keys)}, unexpected={unknown_unexpected}"
        )
    disable_nested_transformer_fastpath(model)
    model.eval()
    return model, rec_dataset, requested, {
        "checkpoint": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "allowed_unexpected_keys_only": not unknown_unexpected,
    }


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    required = {
        "official_root": args.official_root,
        "processed_data_root": args.processed_data_root,
        "checkpoint": args.checkpoint,
        "work_dir": args.work_dir,
    }
    missing = [name for name, value in required.items() if not str(value).strip()]
    if missing:
        raise ValueError(f"missing required arguments: {missing}")

    random.seed(args.intervention_seed)
    np.random.seed(args.intervention_seed)
    torch.manual_seed(args.intervention_seed)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_csv = reports_dir / f"{args.report_prefix}_scope_scores.csv"
    output_json = reports_dir / f"{args.report_prefix}_integrity.json"
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    official_repo = Path(args.official_root) / "UniSAR"
    checkpoint_path = Path(args.checkpoint)
    if not official_repo.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"missing official repo or checkpoint: {official_repo}, {checkpoint_path}"
        )

    prep = prepare_work_data(Path(args.processed_data_root), work_dir)
    source_root = Path(prep["source_root"])
    rec_df = pd.read_pickle(
        source_root / "dataset" / split_file(args.split)
    ).reset_index(drop=True)
    if args.expected_split_rows > 0 and int(rec_df.shape[0]) != args.expected_split_rows:
        raise RuntimeError(
            f"split row mismatch: got {int(rec_df.shape[0])}, expected {args.expected_split_rows}"
        )
    user_vocab = load_pickle(source_root / "vocab" / "user_vocab.pkl")
    src_map = load_pickle(source_root / "vocab" / "src_session_vocab_np.pkl")
    rows = build_rows(rec_df, user_vocab, src_map, args.split)
    selected, eval_indices = select_eval_rows(rows, args.row_limit, args.max_eval_rows)
    execution_lineage = build_execution_lineage(
        source_root,
        official_repo,
        checkpoint_path,
        args.split,
    )

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    model = None
    rec_dataset = None
    device = None
    checkpoint_info: dict[str, Any] = {}
    source_histogram: Counter[int] = Counter()
    sum_rr: dict[str, float] = defaultdict(float)
    frozen_checks = 0
    rank_rr_checks = 0
    valid_id_checks = 0
    selected_change_checks = 0
    donor_summary: dict[str, Any] = {}
    donor_map: torch.Tensor | None = None
    observed_lookup: torch.Tensor | None = None
    candidate_counts_seen: set[int] = set()
    try:
        model, rec_dataset, device, checkpoint_info = load_model(
            official_repo,
            work_dir,
            checkpoint_path,
            args.split,
            args.device,
            args.model_batch_size,
        )
        observed_source_counts = collect_observed_source_counts(
            rec_dataset,
            selected,
            eval_indices,
            max(args.batch_size, 1024),
        )
        donor_map, donor_summary = build_valid_source_derangement(
            src_map,
            observed_source_counts,
            int(prep["neutral_src_session_id"]),
            args.intervention_seed,
        )
        if (
            donor_summary["fixed_points"]
            or donor_summary["padding_or_neutral_donors"]
            or donor_summary["all_donors_observed_and_nonempty"] is not True
        ):
            raise RuntimeError(f"invalid donor-map summary: {donor_summary}")
        donor_map = donor_map.to(device)
        observed_lookup = torch.zeros(
            int(prep["neutral_src_session_id"]) + 1,
            dtype=torch.bool,
            device=device,
        )
        observed_lookup[
            torch.tensor(sorted(observed_source_counts), dtype=torch.long, device=device)
        ] = True
        fieldnames = [
            "official_row_index",
            "user_id",
            "source_slot_count",
            "candidate_count",
            "full_rr",
            *[f"{variant}_rr" for variant in VARIANTS],
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for start in range(0, len(eval_indices), args.batch_size):
                row_indices = eval_indices[start : start + args.batch_size]
                if (
                    start == 0
                    or start % max(args.batch_size * 25, 1) == 0
                    or start + len(row_indices) >= len(eval_indices)
                ):
                    print(
                        f"[R233] seed={args.seed} scoring "
                        f"{start + len(row_indices)}/{len(eval_indices)}",
                        flush=True,
                    )
                feed = build_candidate_batch(rec_dataset, row_indices, selected)
                batch = tensor_to_device(rec_dataset.collate_batch(feed), device)
                candidate_count = 1 + int(batch["neg_items"].shape[1])
                candidate_counts_seen.add(candidate_count)
                if candidate_count != 100:
                    raise RuntimeError(f"candidate count {candidate_count} != 100")
                slot_counts = source_slot_counts(batch)
                if bool((slot_counts <= 0).any().item()):
                    raise RuntimeError("encountered row without a source slot")
                source_histogram.update(int(value) for value in slot_counts.cpu().tolist())
                official_indices = [
                    int(selected[index]["official_row_index"]) for index in row_indices
                ]
                masks = build_scope_masks(batch, official_indices, args.intervention_seed)
                view_batches: dict[str, dict[str, Any]] = {}
                for mode in MODES:
                    for scope, selected_mask in masks.items():
                        variant = f"{mode}_{scope}"
                        expected = expected_scope_changes(scope, slot_counts)
                        selected_counts = selected_mask.sum(dim=1).to(dtype=torch.long)
                        if not torch.equal(selected_counts, expected):
                            raise RuntimeError(
                                f"scope count mismatch for {variant}: "
                                f"got={selected_counts.cpu().tolist()[:8]} "
                                f"expected={expected.cpu().tolist()[:8]}"
                            )
                        view = replace_scope(
                            batch,
                            selected_mask,
                            mode,
                            int(prep["neutral_src_session_id"]),
                            donor_map,
                        )
                        if not compare_frozen_tensors(batch, view):
                            raise RuntimeError(f"frozen tensor changed in {variant}")
                        frozen_checks += 1
                        changed = ((view["all_his"] != batch["all_his"]) & selected_mask).sum(
                            dim=1
                        )
                        if not torch.equal(changed.to(dtype=torch.long), expected):
                            raise RuntimeError(f"selected slot failed to change in {variant}")
                        selected_change_checks += 1
                        if mode == "valid" and bool(selected_mask.any().item()):
                            if observed_lookup is None:
                                raise RuntimeError("observed source lookup is unavailable")
                            replacements = view["all_his"][selected_mask]
                            if bool(
                                (
                                    (replacements <= 0)
                                    | (replacements >= int(prep["neutral_src_session_id"]))
                                ).any().item()
                            ):
                                raise RuntimeError(f"invalid source ID emitted in {variant}")
                            if not bool(observed_lookup[replacements].all().item()):
                                raise RuntimeError(f"unobserved source ID emitted in {variant}")
                            valid_id_checks += 1
                        view_batches[variant] = view

                scores: dict[str, torch.Tensor] = {}
                with torch.no_grad():
                    scores["full"] = model.rec_predict(batch)
                    for variant in VARIANTS:
                        scores[variant] = model.rec_predict(view_batches[variant])
                expected_score_shape = (len(row_indices), candidate_count)
                rr: dict[str, list[float]] = {}
                for name, score_tensor in scores.items():
                    if tuple(score_tensor.shape) != expected_score_shape:
                        raise RuntimeError(
                            f"score shape for {name} is {tuple(score_tensor.shape)}, "
                            f"expected {expected_score_shape}"
                        )
                    if not torch.isfinite(score_tensor).all():
                        raise RuntimeError(f"non-finite scores in {name}")
                    rr[name], ranks = rr_from_scores(score_tensor)
                    if not rank_rr_identity_pass(rr[name], ranks):
                        raise RuntimeError(f"rank/RR identity failed in {name}")
                    rank_rr_checks += 1
                    sum_rr[name] += float(sum(rr[name]))
                for position, selected_index in enumerate(row_indices):
                    record = selected[selected_index]
                    output = {
                        "official_row_index": int(record["official_row_index"]),
                        "user_id": int(record["user_id"]),
                        "source_slot_count": int(slot_counts[position].item()),
                        "candidate_count": candidate_count,
                        "full_rr": rr["full"][position],
                    }
                    for variant in VARIANTS:
                        output[f"{variant}_rr"] = rr[variant][position]
                    writer.writerow(output)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    row_count = len(eval_indices)
    payload = {
        "schema": "focalswap.r233.unisar_source_scope_integrity.v1",
        "contract_id": CONTRACT_ID,
        "seed": int(args.seed),
        "intervention_seed": int(args.intervention_seed),
        "split": args.split,
        "processed_split_rows": int(rec_df.shape[0]),
        "scored_rows": row_count,
        "candidate_count": next(iter(candidate_counts_seen)) if len(candidate_counts_seen) == 1 else None,
        "checkpoint_load": checkpoint_info,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "execution_lineage": execution_lineage,
        "output_csv": str(output_csv),
        "output_csv_sha256": sha256_file(output_csv),
        "output_csv_bytes": int(output_csv.stat().st_size),
        "source_slot_histogram": {
            str(key): int(value) for key, value in sorted(source_histogram.items())
        },
        "donor_map": donor_summary,
        "variants": list(VARIANTS),
        "mean_rr": {
            name: float(total / row_count) for name, total in sorted(sum_rr.items())
        },
        "integrity": {
            "frozen_tensor_checks": int(frozen_checks),
            "selected_change_checks": int(selected_change_checks),
            "valid_id_checks": int(valid_id_checks),
            "rank_rr_checks": int(rank_rr_checks),
            "candidate_counts_seen": sorted(candidate_counts_seen),
            "all_pass": True,
        },
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "contract": CONTRACT_ID,
                "seed": args.seed,
                "rows": row_count,
                "output": str(output_csv),
                "integrity": "PASS",
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
