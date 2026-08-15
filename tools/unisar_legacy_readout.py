#!/usr/bin/env python3
"""R136 official UniSAR five-view readout.

This script keeps the R132 official-code evidence and the SARCheck five-view
audit interface connected without claiming a full leaderboard reproduction.
It can run in two modes:

* interface-smoke: no checkpoint is supplied; the official UniSAR forward path
  is exercised on a SARCheck-converted bundle.
* checkpoint-readout: a trained official UniSAR checkpoint is supplied; the same
  views are scored and summarized.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
REPORTS = ROOT / "reports"
EXAMPLES = ROOT / "data/processed/backbone_audit/examples.jsonl"
DEFAULT_OFFICIAL = ROOT / "external" / "official_repos"
DEFAULT_WORK = ROOT / ".work" / "unisar_five_view"

if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories  # noqa: E402
from transition_backbone import (  # noqa: E402
    build_swap_indices,
    build_swap_profiles,
    pick_swap,
)
from official_adapter import (  # noqa: E402
    build_unisar_tiny_bundle,
    clone_batch,
    get_git_commit,
    tensor_to_device,
    write_csv,
    write_json,
    write_md,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", default=str(DEFAULT_OFFICIAL))
    p.add_argument("--work-dir", default=str(DEFAULT_WORK))
    p.add_argument("--examples-path", default=str(EXAMPLES))
    p.add_argument("--item-metadata", default=str(ROOT / "data/processed/item_metadata.parquet"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--row-limit", type=int, default=3000)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--neg-k", type=int, default=99)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--model-batch-size", type=int, default=128)
    p.add_argument("--load-aux-modules", action="store_true")
    p.add_argument("--max-eval-rows", type=int, default=2000)
    p.add_argument("--swap-tries", type=int, default=60)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--report-prefix", default="r136")
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


def rr_from_scores(scores: torch.Tensor) -> tuple[list[float], list[int]]:
    """Target is column zero; higher scores are better."""
    out_rr: list[float] = []
    out_rank: list[int] = []
    scores_cpu = scores.detach().cpu()
    for row in scores_cpu:
        target_score = row[0]
        rank = int((row > target_score).sum().item()) + 1
        out_rank.append(rank)
        out_rr.append(1.0 / rank)
    return out_rr, out_rank


def mean(vals: list[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else math.nan


def frac(num: int, den: int) -> float:
    return float(num / den) if den else math.nan


def split_indices(selected: list[dict[str, Any]], split: str) -> list[int]:
    idx = [i for i, ex in enumerate(selected) if str(ex.get("split", "train")) == split]
    return idx if idx else list(range(len(selected)))


def replace_nonzero_history(batch: dict[str, Any], view: str, repl: torch.Tensor | int | None = None) -> dict[str, Any]:
    out = clone_batch(batch)
    all_his = out["all_his"]
    all_his_type = out["all_his_type"]
    if view == "full_sar":
        return out
    if view in {"history_only", "search_masked", "counterfactual_search"}:
        mask = (all_his_type == 2) & (all_his != 0)
        if repl is None:
            out["all_his"] = all_his.masked_fill(mask, 0)
        elif isinstance(repl, int):
            out["all_his"] = all_his.masked_fill(mask, repl)
        else:
            replacement = repl.to(all_his.device).reshape(-1, 1).expand_as(all_his)
            out["all_his"] = torch.where(mask, replacement, all_his)
    elif view == "search_only":
        mask = (all_his_type == 1) & (all_his != 0)
        if repl is None:
            out["all_his"] = all_his.masked_fill(mask, 0)
        elif isinstance(repl, int):
            out["all_his"] = all_his.masked_fill(mask, repl)
        else:
            replacement = repl.to(all_his.device).reshape(-1, 1).expand_as(all_his)
            out["all_his"] = torch.where(mask, replacement, all_his)
    else:
        raise ValueError(view)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l1 = [r for r in rows if int(r["l1_matched"])]
    l2 = [r for r in rows if int(r["l2_matched"])]
    return {
        "rows": len(rows),
        "full_sar_rr": mean([float(r["full_sar_rr"]) for r in rows]),
        "history_only_rr": mean([float(r["history_only_rr"]) for r in rows]),
        "search_only_rr": mean([float(r["search_only_rr"]) for r in rows]),
        "search_masked_rr": mean([float(r["search_masked_rr"]) for r in rows]),
        "full_minus_history_rr": mean([float(r["full_sar_rr"]) - float(r["history_only_rr"]) for r in rows]),
        "full_minus_masked_rr": mean([float(r["full_sar_rr"]) - float(r["search_masked_rr"]) for r in rows]),
        "l1_matched": len(l1),
        "l1_true_win_rate": frac(sum(int(r["l1_true_win"]) for r in l1), len(l1)),
        "l1_true_minus_swap_rr": mean([float(r["l1_true_minus_swap"]) for r in l1]),
        "l2_matched": len(l2),
        "l2_true_win_rate": frac(sum(int(r["l2_true_win"]) for r in l2), len(l2)),
        "l2_true_minus_swap_rr": mean([float(r["l2_true_minus_swap"]) for r in l2]),
    }


def main() -> None:
    args = parse_args()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    official_repo = Path(args.official_root) / "UniSAR"
    if not official_repo.exists():
        raise FileNotFoundError(f"UniSAR official repo not found: {official_repo}")

    examples = read_jsonl(Path(args.examples_path), limit=args.row_limit)
    if not examples:
        raise ValueError(f"no examples loaded from {args.examples_path}")
    selected = examples[: args.row_limit]
    score_rows_path = reports / f"{args.report_prefix}_unisar_official_score_rows.csv"
    bundle = build_unisar_tiny_bundle(
        selected,
        Path(args.work_dir),
        args.row_limit,
        args.neg_k,
        args.seed,
        score_rows_path=score_rows_path,
    )

    item_cat = load_item_categories(Path(args.item_metadata))
    profiles = build_swap_profiles(selected, item_cat)
    swap_indices = build_swap_indices(profiles)
    eval_selected_idx = split_indices(selected, args.split)[: args.max_eval_rows]
    rng = random.Random(args.seed)
    swap_rows: dict[int, dict[str, int | None]] = {}
    for idx in eval_selected_idx:
        swap_rows[idx] = {
            "l1": pick_swap(idx, "L1", profiles, swap_indices, rng, args.swap_tries),
            "l2": pick_swap(idx, "L2", profiles, swap_indices, rng, args.swap_tries),
        }

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(Path(args.work_dir))
        sys.path.insert(0, str(official_repo))
        const = importlib.import_module("utils.const")
        utils = importlib.import_module("utils.utils")
        dataset_mod = importlib.import_module("utils.dataset")
        model_mod = importlib.import_module("models.UniSAR")

        const.init_setting_KuaiSAR()
        user_vocab = utils.load_pickle(const.user_vocab)
        rec_dataset = dataset_mod.RecDataSet(args.split if eval_selected_idx else "test", user_vocab=user_vocab)

        device = torch.device(args.device)
        model_args = SimpleNamespace(
            device=device,
            model_path=str(Path(args.work_dir) / "model"),
            dropout=0.0,
            batch_size=args.model_batch_size,
            num_layers=1,
            num_heads=2,
            q_i_cl_temp=0.5,
            q_i_cl_weight=0.001 if args.load_aux_modules else 0.0,
            his_cl_temp=0.1,
            his_cl_weight=0.1 if args.load_aux_modules else 0.0,
            pred_hid_units=[200, 80, 1],
        )
        model = model_mod.UniSAR(model_args)
        mode = "interface_smoke"
        checkpoint = str(args.checkpoint).strip()
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
            model.load_model(str(checkpoint_path))
            mode = "checkpoint_readout"
        model.eval()

        details: list[dict[str, Any]] = []
        for start in range(0, len(eval_selected_idx), args.batch_size):
            selected_batch_idx = eval_selected_idx[start : start + args.batch_size]
            local_dataset_idx = list(range(start, start + len(selected_batch_idx)))
            feed = [rec_dataset[i] for i in local_dataset_idx]
            batch = tensor_to_device(rec_dataset.collate_batch(feed), device)

            view_scores: dict[str, torch.Tensor] = {}
            with torch.no_grad():
                for view in ["full_sar", "history_only", "search_only", "search_masked"]:
                    repl = None
                    if view in {"history_only", "search_masked"}:
                        repl = int(bundle["neutral_src_session_id"])
                    elif view == "search_only":
                        repl = int(bundle["neutral_item_id"])
                    view_scores[view] = model.rec_predict(replace_nonzero_history(batch, view, repl))

                for level in ["l1", "l2"]:
                    repl_ids = []
                    for idx in selected_batch_idx:
                        swap_idx = swap_rows[idx][level]
                        repl_ids.append(0 if swap_idx is None else int(swap_idx) + 1)
                    repl_tensor = torch.tensor(repl_ids, dtype=torch.long, device=device)
                    view_scores[level] = model.rec_predict(
                        replace_nonzero_history(batch, "counterfactual_search", repl_tensor)
                    )

            rr: dict[str, list[float]] = {}
            rank: dict[str, list[int]] = {}
            for view, scores in view_scores.items():
                rr[view], rank[view] = rr_from_scores(scores)

            for pos, idx in enumerate(selected_batch_idx):
                l1_matched = swap_rows[idx]["l1"] is not None
                l2_matched = swap_rows[idx]["l2"] is not None
                row = {
                    "example_id": selected[idx]["example_id"],
                    "split": selected[idx].get("split", ""),
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
                    "l1_matched": int(l1_matched),
                    "l1_swap_rr": rr["l1"][pos] if l1_matched else 0.0,
                    "l1_true_minus_swap": rr["full_sar"][pos] - rr["l1"][pos] if l1_matched else 0.0,
                    "l1_true_win": int(l1_matched and rr["full_sar"][pos] > rr["l1"][pos]),
                    "l2_matched": int(l2_matched),
                    "l2_swap_rr": rr["l2"][pos] if l2_matched else 0.0,
                    "l2_true_minus_swap": rr["full_sar"][pos] - rr["l2"][pos] if l2_matched else 0.0,
                    "l2_true_win": int(l2_matched and rr["full_sar"][pos] > rr["l2"][pos]),
                }
                details.append(row)

        scores_csv = reports / f"{args.report_prefix}_unisar_official_five_view_scores.csv"
        json_path = reports / f"{args.report_prefix}_unisar_official_five_view_readout.json"
        md_path = reports / f"{args.report_prefix}_unisar_official_five_view_readout.md"
        write_csv(scores_csv, details)
        metrics = aggregate(details)
        payload = {
            "schema": "sarcheck.r136.unisar_official_five_view.v0.1",
            "status": "done",
            "mode": mode,
            "official_repo": str(official_repo),
            "official_commit": get_git_commit(official_repo),
            "checkpoint": checkpoint or None,
            "split": args.split,
            "bundle": bundle,
            "metrics": metrics,
            "outputs": {
                "scores_csv": str(scores_csv),
                "json": str(json_path),
                "report_md": str(md_path),
                "score_rows": str(score_rows_path),
            },
            "limits": [
                "The readout uses the SARCheck-converted UniSAR bundle, not the original paper's full benchmark preprocessing.",
                "Counterfactual-search views replace the focal search session with matched L1/L2 search sessions while keeping the target and candidate set fixed.",
                "If mode is interface_smoke, metrics only validate the official forward/readout interface and must not be interpreted as trained-model evidence.",
            ],
        }
        write_json(json_path, payload)
        md = [
            f"# {args.report_prefix} UniSAR Official Five-view Readout",
            "",
            f"- Status: done",
            f"- Mode: `{mode}`",
            f"- Rows: {metrics['rows']}",
            f"- Full-SAR RR: {metrics['full_sar_rr']:.4f}",
            f"- History-only RR: {metrics['history_only_rr']:.4f}",
            f"- Search-only RR: {metrics['search_only_rr']:.4f}",
            f"- Search-masked RR: {metrics['search_masked_rr']:.4f}",
            f"- Full minus History-only RR: {metrics['full_minus_history_rr']:.4f}",
            f"- Full minus Search-masked RR: {metrics['full_minus_masked_rr']:.4f}",
            f"- L1 true-search win: {metrics['l1_true_win_rate']:.4f} ({metrics['l1_matched']} matched)",
            f"- L2 true-search win: {metrics['l2_true_win_rate']:.4f} ({metrics['l2_matched']} matched)",
            "",
            "Interpretation: use this report as official-code five-view readout evidence only when mode is `checkpoint_readout`.",
        ]
        write_md(md_path, "\n".join(md))
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


if __name__ == "__main__":
    main()
