#!/usr/bin/env python3
"""Checkpoint-backed UniSAR readout for the R225 common surface."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from official_adapter import (  # noqa: E402
    clone_batch,
    get_git_commit,
    tensor_to_device,
    write_csv,
    write_json,
    write_md,
)
from unisar_legacy_readout import replace_nonzero_history, rr_from_scores  # noqa: E402
from unisar_preflight import (  # noqa: E402
    compare_frozen_tensors,
    replace_matched_source_only,
)
from common_swap_contract import load_swap_ledger  # noqa: E402


SWAP_LEVELS = (
    "l1_nuisance",
    "l2_same_category",
    "category_broken_source",
    "same_category_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/eligible_examples.jsonl",
    )
    parser.add_argument(
        "--item-metadata",
        type=Path,
        default=ROOT / "data/processed/item_metadata.parquet",
    )
    parser.add_argument(
        "--candidate-ledger",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/candidates.jsonl.gz",
    )
    parser.add_argument("--bundle-report", type=Path, required=True)
    parser.add_argument("--swap-ledger", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--report-prefix", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-eval-rows", type=int, default=3_249)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else math.nan


def frac(num: float, den: int) -> float:
    return float(num / den) if den else math.nan


def ndcg_at_k_from_scores(scores: torch.Tensor, k: int = 10) -> list[float]:
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


def level_summary(rows: list[dict[str, Any]], level: str) -> dict[str, Any]:
    matched = [row for row in rows if int(row[f"{level}_matched"])]
    wins = sum(float(row[f"{level}_score_delta"]) > 0.0 for row in matched)
    ties = sum(float(row[f"{level}_score_delta"]) == 0.0 for row in matched)
    losses = sum(float(row[f"{level}_score_delta"]) < 0.0 for row in matched)
    return {
        "matched": len(matched),
        "coverage": frac(len(matched), len(rows)),
        "win": frac(wins, len(matched)),
        "tie": frac(ties, len(matched)),
        "loss": frac(losses, len(matched)),
        "half_tie": frac(wins + 0.5 * ties, len(matched)),
        "mean_score_delta": mean([float(row[f"{level}_score_delta"]) for row in matched]),
        "mean_rr_delta": mean([float(row[f"{level}_rr_delta"]) for row in matched]),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint) if str(args.checkpoint).strip() else None
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    bundle_report = json.loads(args.bundle_report.read_text(encoding="utf-8"))
    if bundle_report.get("status") != "PASS_FULL_COMMON_SURFACE_BUNDLES":
        raise ValueError(f"bundle gate not passed: {bundle_report.get('status')}")
    if file_sha256(args.candidate_ledger) != bundle_report["candidate_ledger_sha256"]:
        raise ValueError("candidate ledger hash differs from full bundle report")

    rows = read_jsonl(args.examples_path)
    if len(rows) != 77_661:
        raise ValueError(f"expected 77661 rows, got {len(rows)}")
    all_test_indices = [index for index, row in enumerate(rows) if str(row["split"]) == "test"]
    eval_indices = all_test_indices[: args.max_eval_rows]
    if len(eval_indices) != args.max_eval_rows:
        raise ValueError(f"expected {args.max_eval_rows} test rows, got {len(eval_indices)}")
    swap_lookups = load_swap_ledger(args.swap_ledger, rows, eval_indices)
    swap_summary = {}
    for level, lookup in swap_lookups.items():
        matched = sum(donor is not None for donor in lookup.values())
        swap_summary[level] = {
            "attempted": len(eval_indices),
            "matched": matched,
            "coverage": frac(matched, len(eval_indices)),
        }

    official_repo = args.official_root / "UniSAR"
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(official_repo)
        sys.path.insert(0, str(official_repo))
        const = importlib.import_module("utils.const")
        utils = importlib.import_module("utils.utils")
        dataset_mod = importlib.import_module("utils.dataset")
        model_mod = importlib.import_module("models.UniSAR")
        const.init_setting_KuaiSAR()
        user_vocab = utils.load_pickle(const.user_vocab)
        rec_dataset = dataset_mod.RecDataSet("test", user_vocab=user_vocab)
        if len(rec_dataset) != len(all_test_indices):
            raise ValueError(
                f"UniSAR test rows={len(rec_dataset)} expected={len(all_test_indices)}"
            )
        device = torch.device(args.device)
        model_args = SimpleNamespace(
            device=device,
            model_path=str(official_repo / "output/KuaiSAR/UniSAR/checkpoints/r225_readout"),
            dropout=0.0,
            batch_size=args.batch_size,
            num_layers=1,
            num_heads=2,
            q_i_cl_temp=0.5,
            q_i_cl_weight=0.001,
            his_cl_temp=0.1,
            his_cl_weight=0.1,
            pred_hid_units=[200, 80, 1],
        )
        model = model_mod.UniSAR(model_args)
        mode = "interface_smoke"
        if checkpoint_path is not None:
            model.load_model(str(checkpoint_path))
            mode = "checkpoint_readout"
        model.eval()

        details: list[dict[str, Any]] = []
        candidate_checks: list[bool] = []
        for start in range(0, len(eval_indices), args.batch_size):
            batch_indices = eval_indices[start : start + args.batch_size]
            local_indices = list(range(start, start + len(batch_indices)))
            feed = [rec_dataset[index] for index in local_indices]
            batch = tensor_to_device(rec_dataset.collate_batch(feed), device)
            view_batches = {
                "full_sar": replace_nonzero_history(batch, "full_sar"),
                "history_only": replace_nonzero_history(
                    batch, "history_only", int(bundle_report["bundles"]["unisar"]["neutral_src_session_id"])
                ),
                "search_only": replace_nonzero_history(
                    batch, "search_only", int(bundle_report["bundles"]["unisar"]["neutral_item_id"])
                ),
                "search_masked": replace_nonzero_history(
                    batch, "search_masked", int(bundle_report["bundles"]["unisar"]["neutral_src_session_id"])
                ),
            }
            candidate_checks.extend(
                compare_frozen_tensors(batch, view_batch)
                for view_batch in view_batches.values()
            )
            with torch.no_grad():
                scores = {view: model.rec_predict(view_batch) for view, view_batch in view_batches.items()}
                for level, lookup in swap_lookups.items():
                    replacement_ids = [
                        0 if lookup[index] is None else int(lookup[index]) + 1
                        for index in batch_indices
                    ]
                    replacement = torch.tensor(replacement_ids, dtype=torch.long, device=device)
                    swap_batch = replace_matched_source_only(batch, replacement)
                    candidate_checks.append(compare_frozen_tensors(batch, swap_batch))
                    scores[level] = model.rec_predict(swap_batch)
            nonfinite = {
                view: int((~torch.isfinite(tensor)).sum().item())
                for view, tensor in scores.items()
            }
            if any(nonfinite.values()):
                raise RuntimeError(f"non-finite UniSAR scores: {nonfinite}")

            rr: dict[str, list[float]] = {}
            rank: dict[str, list[int]] = {}
            target_score: dict[str, list[float]] = {}
            ndcg10: dict[str, list[float]] = {}
            for view, tensor in scores.items():
                rr[view], rank[view] = rr_from_scores(tensor)
                ndcg10[view] = ndcg_at_k_from_scores(tensor, 10)
                target_score[view] = [
                    float(value) for value in tensor[:, 0].detach().cpu().tolist()
                ]
            candidate_count = int(scores["full_sar"].shape[1])
            if candidate_count != 100:
                raise ValueError(f"R225 requires 100 candidates, got {candidate_count}")
            for position, row_index in enumerate(batch_indices):
                row = rows[row_index]
                detail = {
                    "row_index": row_index,
                    "example_id": str(row["example_id"]),
                    "split": str(row["split"]),
                    "user_id": int(row["user_id"]),
                    "target_item": int(row["primary_target_item"]),
                    "source_session_id": row_index + 1,
                    "candidate_count": candidate_count,
                    "full_sar_rr": rr["full_sar"][position],
                    "history_only_rr": rr["history_only"][position],
                    "search_only_rr": rr["search_only"][position],
                    "search_masked_rr": rr["search_masked"][position],
                    "full_sar_rank": rank["full_sar"][position],
                    "search_masked_rank": rank["search_masked"][position],
                    "full_sar_target_score": target_score["full_sar"][position],
                    "search_masked_target_score": target_score["search_masked"][position],
                    "full_sar_ndcg10": ndcg10["full_sar"][position],
                    "search_masked_ndcg10": ndcg10["search_masked"][position],
                }
                for level, lookup in swap_lookups.items():
                    swap_index = lookup[row_index]
                    matched = swap_index is not None
                    detail[f"{level}_matched"] = int(matched)
                    detail[f"{level}_swap_row_index"] = int(swap_index) if matched else -1
                    detail[f"{level}_swap_source_session_id"] = (
                        int(swap_index) + 1 if matched else 0
                    )
                    detail[f"{level}_swap_rr"] = rr[level][position] if matched else 0.0
                    detail[f"{level}_rr_delta"] = (
                        rr["full_sar"][position] - rr[level][position] if matched else 0.0
                    )
                    detail[f"{level}_score_delta"] = (
                        target_score["full_sar"][position] - target_score[level][position]
                        if matched
                        else 0.0
                    )
                details.append(detail)
            print(
                json.dumps(
                    {"progress": "readout", "rows_done": len(details), "rows_total": len(eval_indices)}
                ),
                flush=True,
            )
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.reports_dir / f"{args.report_prefix}_scores.csv"
    json_path = args.reports_dir / f"{args.report_prefix}_readout.json"
    md_path = args.reports_dir / f"{args.report_prefix}_readout.md"
    write_csv(scores_path, details)
    metrics = {
        "rows": len(details),
        "full_sar_rr": mean([float(row["full_sar_rr"]) for row in details]),
        "history_only_rr": mean([float(row["history_only_rr"]) for row in details]),
        "search_only_rr": mean([float(row["search_only_rr"]) for row in details]),
        "search_masked_rr": mean([float(row["search_masked_rr"]) for row in details]),
        "full_sar_ndcg10": mean([float(row["full_sar_ndcg10"]) for row in details]),
        "search_masked_ndcg10": mean(
            [float(row["search_masked_ndcg10"]) for row in details]
        ),
    }
    metrics["full_minus_masked_rr"] = metrics["full_sar_rr"] - metrics["search_masked_rr"]
    level_metrics = {level: level_summary(details, level) for level in SWAP_LEVELS}
    candidate_frozen = bool(candidate_checks) and all(candidate_checks)
    readout_pass = candidate_frozen and all(
        level_metrics[level]["matched"] > 0 for level in SWAP_LEVELS
    )
    if readout_pass:
        status = (
            "PASS_R225_UNISAR_CHECKPOINT_READOUT"
            if checkpoint_path is not None
            else "PASS_R225_UNISAR_INTERFACE_SMOKE"
        )
    else:
        status = "BLOCK_R225_UNISAR_READOUT"
    payload = {
        "schema": "focalswap.r225.unisar_common_surface_readout.v1",
        "status": status,
        "surface_id": "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED",
        "mode": mode,
        "official_commit": get_git_commit(args.official_root / "UniSAR"),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "candidate_contract": "canonical_raw_item_ledger",
        "candidate_ledger_sha256": file_sha256(args.candidate_ledger),
        "swap_ledger": str(args.swap_ledger),
        "swap_ledger_sha256": file_sha256(args.swap_ledger),
        "candidate_tensors_frozen": candidate_frozen,
        "finite_all_scores": True,
        "same_native_scorer": "UniSAR.rec_predict",
        "specificity_basis": "target_score",
        "swap_summary": swap_summary,
        "metrics": metrics,
        "level_metrics": level_metrics,
        "outputs": {"scores_csv": str(scores_path), "json": str(json_path), "md": str(md_path)},
        "claim_boundary": (
            "Controlled common-surface checkpoint evidence; not an official-dataset or leaderboard reproduction."
            if checkpoint_path is not None
            else "Full-cardinality random-init interface smoke only; not trained-model evidence."
        ),
    }
    write_json(json_path, payload)
    lines = [
        "# R225 UniSAR Common-Surface Readout",
        "",
        f"Status: `{status}`",
        "",
        f"- Rows: `{metrics['rows']:,}`",
        f"- Full RR: `{metrics['full_sar_rr']:.6f}`",
        f"- Search-masked RR: `{metrics['search_masked_rr']:.6f}`",
        f"- Full-minus-masked RR: `{metrics['full_minus_masked_rr']:+.6f}`",
        f"- Full NDCG@10: `{metrics['full_sar_ndcg10']:.6f}`",
        f"- Candidate tensors frozen: `{candidate_frozen}`",
        "",
        "## Swap Readouts",
        "",
    ]
    for level in SWAP_LEVELS:
        value = level_metrics[level]
        lines.append(
            f"- `{level}`: N={value['matched']:,}, half-tie={value['half_tie']:.6f}, "
            f"mean score delta={value['mean_score_delta']:+.6f}"
        )
    lines.extend(["", "## Claim Boundary", "", payload["claim_boundary"], ""])
    write_md(md_path, "\n".join(lines))
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
