#!/usr/bin/env python3
"""Run the frozen-row R243 semi-synthetic full-chain positive control."""

from __future__ import annotations

import json
import math
import time
import csv
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import qilin_trained_full_chain as base


def parse_args():
    parser = base.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, default=None)
    parser.add_argument("--query-embeddings", type=Path, default=None)
    parser.add_argument("--r239-reference-manifest", type=Path, default=None)
    parser.add_argument("--r239-reference-row-scores", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260713, help="Student seed only")
    parser.add_argument("--teacher-seed", type=int, default=243)
    parser.add_argument("--teacher-rank", type=int, default=8)
    parser.add_argument("--dose", type=float, choices=(0.0, 0.5, 1.0), required=True)
    parser.add_argument("--teacher-noise-scale", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--projection-rank", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--max-focal-items", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--donor-seed", type=int, default=20260712)
    parser.add_argument("--bootstrap-iters", type=int, default=10_000)
    parser.add_argument("--signflip-iters", type=int, default=10_000)
    parser.add_argument("--null-iters", type=int, default=1_000)
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--focal-horizon-seconds", type=int, default=base.HORIZON_SECONDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    return parser.parse_args()


def _mean_embedding(ids: Sequence[int], table: torch.Tensor) -> torch.Tensor:
    return (table[torch.tensor(ids, dtype=torch.long)].mean(0) if ids
            else torch.zeros(table.shape[1]))


def _zstandardize(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    return (values - values.mean()) / std.clamp_min(1e-8)


def plant_targets(episodes, item_embeddings, query_embeddings, args):
    """Materialize labels once; student training receives labels, never teacher maps."""
    dim = int(item_embeddings.shape[1])
    generator = torch.Generator().manual_seed(args.teacher_seed)
    history_left = torch.randn((dim, args.teacher_rank), generator=generator) / math.sqrt(dim)
    history_right = torch.randn((args.teacher_rank, dim), generator=generator) / math.sqrt(args.teacher_rank)
    focal_left = torch.randn((2 * dim, args.teacher_rank), generator=generator) / math.sqrt(2 * dim)
    focal_right = torch.randn((args.teacher_rank, dim), generator=generator) / math.sqrt(args.teacher_rank)
    item_table = torch.cat([torch.zeros((1, dim)), item_embeddings], dim=0)
    query_table = torch.cat([torch.zeros((1, dim)), query_embeddings], dim=0)
    audit = []
    for episode in episodes:
        observed = tuple(episode.positives)
        history_raw = _mean_embedding(episode.history[-args.max_history:], item_table)
        focal_items = _mean_embedding(episode.focal.clicked_items[-args.max_focal_items:], item_table)
        focal_raw = torch.cat([query_table[episode.focal.query_id], focal_items])
        candidates = item_table[torch.tensor(episode.slate, dtype=torch.long)]
        history_scores = _zstandardize(candidates @ (history_raw @ history_left @ history_right))
        focal_scores = _zstandardize(candidates @ (focal_raw @ focal_left @ focal_right))
        noise_gen = torch.Generator().manual_seed(
            int(base.stable_digest(args.teacher_seed, episode.episode_id)[:16], 16)
        )
        noise = torch.randn(len(episode.slate), generator=noise_gen) * args.teacher_noise_scale
        target_scores = history_scores + args.dose * focal_scores + noise
        target_index = int(torch.argmax(target_scores))
        episode.positives = (int(episode.slate[target_index]),)
        audit.append({
            "episode_id": episode.episode_id, "user_id": episode.user_id,
            "split": episode.split, "dose": args.dose,
            "candidate_item_ids_json": json.dumps(episode.slate, separators=(",", ":")),
            "observed_positive_item_ids_json": json.dumps(observed, separators=(",", ":")),
            "synthetic_positive_item_id": episode.positives[0],
            "synthetic_positive_was_observed": int(episode.positives[0] in observed),
            "history_teacher_scores_json": json.dumps(history_scores.tolist(), separators=(",", ":")),
            "focal_teacher_scores_json": json.dumps(focal_scores.tolist(), separators=(",", ":")),
            "fixed_noise_json": json.dumps(noise.tolist(), separators=(",", ":")),
            "target_scores_json": json.dumps(target_scores.tolist(), separators=(",", ":")),
        })
    teacher = {"history_left": history_left, "history_right": history_right,
               "focal_left": focal_left, "focal_right": focal_right}
    return audit, teacher


def _records_sha256(records: list[dict[str, Any]], keys: Sequence[str]) -> str:
    ordered = sorted(tuple(str(row[key]) for key in keys) for row in records)
    payload = "\n".join("\x1f".join(row) for row in ordered)
    return base.hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_r239_reference(episodes, donor_summary, item_path, query_path, args):
    """Bind R243 to the registered R239 validation/test row contract."""
    current_rows = []
    complete_rows = []
    for episode in episodes:
        complete_rows.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "candidate_hash": base.stable_digest(*episode.raw_slate),
            "history_hash": base.stable_digest(*episode.history),
            "focal_session_id": episode.focal.session_id,
            "focal_query_id": episode.focal.query_id,
            "focal_clicked_items_hash": base.stable_digest(*episode.focal.clicked_items),
            "matched_donor_d1_session_id": (
                episode.matched_donor_d1.session_id if episode.matched_donor_d1 else -1
            ),
            "matched_donor_d2_session_id": (
                episode.matched_donor_d2.session_id if episode.matched_donor_d2 else -1
            ),
        })
        if episode.split not in {"validation", "test"}:
            continue
        current_rows.append({
            "episode_id": episode.episode_id,
            "split": episode.split,
            "candidate_hash": base.stable_digest(*episode.raw_slate),
            "focal_session_id": episode.focal.session_id,
            "matched_donor_d1_session_id": episode.matched_donor_d1.session_id,
            "matched_donor_d2_session_id": episode.matched_donor_d2.session_id,
        })
    keys = (
        "episode_id", "split", "candidate_hash", "focal_session_id",
        "matched_donor_d1_session_id", "matched_donor_d2_session_id",
    )
    current_hash = _records_sha256(current_rows, keys)
    complete_keys = (
        "episode_id", "split", "candidate_hash", "history_hash",
        "focal_session_id", "focal_query_id", "focal_clicked_items_hash",
        "matched_donor_d1_session_id", "matched_donor_d2_session_id",
    )
    complete_hash = _records_sha256(complete_rows, complete_keys)
    result = {
        "current_eval_row_contract_sha256": current_hash,
        "current_complete_input_contract_sha256": complete_hash,
        "current_donor_map_sha256": donor_summary["donor_map_sha256"],
        "reference_checked": False,
        "pass": bool(args.smoke),
    }
    if args.smoke and not args.r239_reference_manifest and not args.r239_reference_row_scores:
        return result
    if not args.r239_reference_manifest or not args.r239_reference_row_scores:
        raise ValueError("Full R243 requires the R239 reference manifest and row scores")
    reference = json.loads(args.r239_reference_manifest.read_text(encoding="utf-8"))
    with args.r239_reference_row_scores.open(newline="", encoding="utf-8") as handle:
        reference_rows = list(csv.DictReader(handle))
    reference_hash = _records_sha256(reference_rows, keys)
    current_counts = {
        split: sum(episode.split == split for episode in episodes)
        for split in ("train", "validation", "test")
    }
    expected_counts = reference["selected_episode_counts"]
    snapshot_files = {
        "recommendation_train_snapshot": args.snapshot / "recommendation_train" / "train-00000-of-00001.parquet",
        "recommendation_test_snapshot": args.snapshot / "recommendation_test" / "train-00000-of-00001.parquet",
        "search_train_snapshot": args.snapshot / "search_train" / "train-00000-of-00001.parquet",
        "search_test_snapshot": args.snapshot / "search_test" / "train-00000-of-00001.parquet",
    }
    checks = {
        "eval_row_contract_hash_matches": current_hash == reference_hash,
        "donor_map_hash_matches": donor_summary["donor_map_sha256"] == reference["donors"]["donor_map_sha256"],
        "split_counts_match": current_counts == expected_counts,
        "item_embeddings_match": base.sha256_file(item_path) == reference["input_sha256"]["item_embeddings"],
        "query_embeddings_match": base.sha256_file(query_path) == reference["input_sha256"]["query_embeddings"],
        "raw_snapshot_hashes_match": all(
            base.sha256_file(path) == reference["input_sha256"][name]
            for name, path in snapshot_files.items()
        ),
        "donor_seed_matches": args.donor_seed == reference["donors"]["donor_seed"],
        "focal_horizon_matches": args.focal_horizon_seconds == reference["config"]["focal_horizon_seconds"],
    }
    result.update({
        "reference_checked": True,
        "reference_manifest": str(args.r239_reference_manifest),
        "reference_row_scores": str(args.r239_reference_row_scores),
        "reference_eval_row_contract_sha256": reference_hash,
        "raw_snapshot_sha256": {
            name: base.sha256_file(path) for name, path in snapshot_files.items()
        },
        "split_counts": current_counts,
        "checks": checks,
        "pass": all(checks.values()),
    })
    return result


def _oracle_gain(episodes) -> float:
    gains = []
    for episode in episodes:
        h = base.ranking_metrics(episode.scores["history_only"], episode.slate, episode.positives)["rr"]
        s = base.ranking_metrics(episode.scores["search_only"], episode.slate, episode.positives)["rr"]
        gains.append(max(h, s) - h)
    return base.mean(gains)


def g2_shuffled_source_null(model, episodes, args, device) -> dict[str, Any]:
    """Remap only legal focal sources and rescore each original row/slate in place."""
    observed = _oracle_gain(episodes)
    original_focals = [episode.focal for episode in episodes]
    legal = [
        [source for source in original_focals
         if source.user_id != episode.user_id and source.timestamp < episode.timestamp]
        for episode in episodes
    ]
    if any(not choices for choices in legal):
        raise ValueError("G2 has a row with no legal shuffled focal source")
    rng = np.random.default_rng(args.teacher_seed + 2430)
    null = np.empty(args.null_iters, dtype=np.float64)
    original_search = [episode.scores["search_only"].copy() for episode in episodes]
    history_rr = np.asarray([_branch_rr(episode, "history_only") for episode in episodes])
    chunk_size = 10
    for chunk_start in range(0, args.null_iters, chunk_size):
        draw_count = min(chunk_size, args.null_iters - chunk_start)
        remapped = []
        for _ in range(draw_count):
            for episode, choices in zip(episodes, legal):
                remapped.append(replace(
                    episode,
                    focal=choices[int(rng.integers(len(choices)))],
                    scores={},
                    gates={},
                ))
        base.score_loader(model, remapped, args, device, "search_only")
        for offset in range(draw_count):
            draw_rows = remapped[offset * len(episodes):(offset + 1) * len(episodes)]
            search_rr = np.asarray([_branch_rr(row, "search_only") for row in draw_rows])
            null[chunk_start + offset] = float(np.maximum(history_rr, search_rr).mean() - history_rr.mean())
    for episode, focal, scores in zip(episodes, original_focals, original_search):
        episode.focal = focal
        episode.scores["search_only"] = scores
    gain_rows = []
    for episode in episodes:
        history_rr = _branch_rr(episode, "history_only")
        search_rr = _branch_rr(episode, "search_only")
        gain_rows.append({
            "user_id": episode.user_id,
            "oracle_gain": max(history_rr, search_rr) - history_rr,
        })
    observed_ci = base.cluster_bootstrap(
        gain_rows,
        lambda rows: base.mean(float(row["oracle_gain"]) for row in rows),
        args.bootstrap_iters,
        args.seed + 2431,
    )
    return {
        "estimand": "mean(max(student_history_rr, student_search_rr)-student_history_rr)",
        "null_policy": "legal cross-user pre-request focal remap; original row/slate/target fixed",
        "observed_oracle_gain": observed, "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "observed_oracle_gain_cluster_bootstrap": observed_ci,
        "p_null_ge_observed": float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1)),
        "iters": int(len(null)),
    }


def _router_features(episode) -> np.ndarray:
    history = np.asarray(episode.scores["history_only"], dtype=np.float64)
    search = np.asarray(episode.scores["search_only"], dtype=np.float64)
    def stats(values):
        ordered = np.sort(values)
        margin = ordered[-1] - ordered[-2] if len(ordered) > 1 else 0.0
        return [values.mean(), values.std(), values.min(), values.max(), margin]
    history_items = set(episode.history[-20:])
    focal_items = set(episode.focal.clicked_items[-20:])
    union = history_items | focal_items
    similarity = len(history_items & focal_items) / len(union) if union else 0.0
    return np.asarray([
        len(episode.history), len(episode.focal.clicked_items),
        math.log1p(max(0.0, episode.timestamp - episode.focal.timestamp)), similarity,
        *stats(history), *stats(search),
    ], dtype=np.float64)


def _branch_rr(episode, branch: str) -> float:
    return base.ranking_metrics(episode.scores[branch], episode.slate, episode.positives)["rr"]


def _fit_logistic(x, y, l2: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(0); scale = x.std(0); scale[scale < 1e-8] = 1.0
    z = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    weights = np.zeros(z.shape[1])
    for _ in range(200):
        probability = 1.0 / (1.0 + np.exp(-np.clip(z @ weights, -30, 30)))
        gradient = z.T @ (probability - y) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= 0.1 * gradient
    return weights, mean, scale


def _route(episodes, model) -> list[dict[str, Any]]:
    weights, mean, scale = model
    output = []
    for episode in episodes:
        feature = _router_features(episode)
        probability = float(1.0 / (1.0 + np.exp(-np.clip(
            weights[0] + ((feature - mean) / scale) @ weights[1:], -30, 30))))
        branch = "search_only" if probability >= 0.5 else "history_only"
        h_rr, s_rr = _branch_rr(episode, "history_only"), _branch_rr(episode, "search_only")
        output.append({"episode_id": episode.episode_id, "user_id": episode.user_id,
                       "router_search_probability": probability, "router_branch": branch,
                       "router_rr": s_rr if branch == "search_only" else h_rr,
                       "history_only_rr": h_rr, "search_only_rr": s_rr})
    return output


def train_select_evaluate_router(train, validation, test, args):
    x_train = np.stack([_router_features(row) for row in train])
    y_train = np.asarray([_branch_rr(row, "search_only") > _branch_rr(row, "history_only") for row in train], dtype=float)
    candidates = []
    for l2 in (0.0, 1e-3, 1e-2, 1e-1, 1.0):
        model = _fit_logistic(x_train, y_train, l2)
        val_rows = _route(validation, model)
        candidates.append((base.mean(row["router_rr"] for row in val_rows), l2, model))
    val_rr, selected_l2, selected = max(candidates, key=lambda value: (value[0], -value[1]))
    selected_validation_rows = _route(validation, selected)
    validation_history_mean = base.mean(row["history_only_rr"] for row in selected_validation_rows)
    validation_search_mean = base.mean(row["search_only_rr"] for row in selected_validation_rows)
    best_fixed = (
        "history_only"
        if validation_history_mean >= validation_search_mean
        else "search_only"
    )
    test_rows = _route(test, selected)
    history_mean = base.mean(row["history_only_rr"] for row in test_rows)
    search_mean = base.mean(row["search_only_rr"] for row in test_rows)
    metric = lambda rows: base.mean(row["router_rr"] - row[f"{best_fixed}_rr"] for row in rows)
    comparison = base.cluster_bootstrap(test_rows, metric, args.bootstrap_iters, args.seed + 4404)
    return test_rows, {"selected_l2": selected_l2, "validation_router_rr": val_rr,
                       "best_fixed_test_branch": best_fixed,
                       "fixed_branch_selected_on_validation": best_fixed,
                       "best_fixed_test_rr": base.mean(row[f"{best_fixed}_rr"] for row in test_rows),
                       "router_test_rr": base.mean(row["router_rr"] for row in test_rows),
                       "router_minus_best_fixed_rr": comparison}, selected


def lower(metric: dict[str, Any]) -> float:
    return float(metric["ci_low"]) if metric.get("ci_low") is not None else -float("inf")


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2); args.bootstrap_iters = min(args.bootstrap_iters, 128)
        args.signflip_iters = min(args.signflip_iters, 128); args.null_iters = min(args.null_iters, 16)
        args.max_users = args.max_users or 32; args.max_episodes_per_split = args.max_episodes_per_split or 128
    elif args.dose == 1.0 and args.null_iters < 1000:
        raise ValueError("Lambda-one R243 requires at least 1000 legal shuffled-source null draws")
    elif args.dose != 1.0 and args.null_iters < 100:
        raise ValueError("Non-primary R243 doses require at least 100 shuffled-source null draws")
    base.set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = base.choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_path = args.item_embeddings or args.data_dir / "emb" / "bge-m3_item.pt"
    query_path = args.query_embeddings or args.data_dir / "emb" / "bge-m3_query.pt"
    started = time.time()
    raw_to_item, query_to_id, item_vocab = base.vocab_mappings(args.data_dir)
    item_embeddings, query_embeddings = base.load_tensor(item_path), base.load_tensor(query_path)
    sessions, sessions_by_user = base.load_focal_sessions(args.snapshot, raw_to_item, query_to_id)
    base.validate_embedding_contract(item_embeddings, query_embeddings, item_vocab, sessions)
    episodes, construction_counts = base.reconstruct_episodes(args.snapshot, raw_to_item, sessions_by_user, args.focal_horizon_seconds)
    episodes = base.bound_smoke_rows(episodes, args.max_users, args.max_episodes_per_split)
    donor_summary = base.freeze_donors(episodes, sessions, args.donor_seed)
    donor_rows = donor_summary.pop("donor_map")
    checks = base.integrity_checks(episodes, args.smoke, args.focal_horizon_seconds)
    failures = [row["name"] for row in checks if row["status"] != "pass"]
    if construction_counts.get("eligible_requests_with_unmapped_slate_item", 0): failures.append("embedding_coverage_for_original_logged_slates")
    if failures: raise RuntimeError("R243 integrity preflight failed: " + ", ".join(sorted(set(failures))))
    reference_check = verify_r239_reference(
        episodes, donor_summary, item_path, query_path, args
    )
    if not reference_check["pass"]:
        raise RuntimeError("R243 does not match the registered R239 surface")
    label_rows, teacher = plant_targets(episodes, item_embeddings, query_embeddings, args)
    train = [row for row in episodes if row.split == "train"]
    validation = [row for row in episodes if row.split == "validation"]
    test = [row for row in episodes if row.split == "test"]
    neutral = base.train_only_neutral_focal_raw(train, item_embeddings, query_embeddings, args.max_focal_items)
    model = base.R239Scorer(item_embeddings, query_embeddings, args.projection_rank, args.hidden_dim, args.dropout, neutral).to(device)
    training_history, best_state, best_epoch = base.train_model(model, train, validation, args, device)
    model.load_state_dict(best_state)
    for episode in episodes: episode.scores.clear(); episode.gates.clear()
    for view in base.VIEWS: base.score_loader(model, episodes, args, device, view)
    item_text_model, item_text_epoch = base.train_item_text_baseline(item_embeddings, train, validation, args, device)
    base.score_quality_baselines(item_text_model, train, validation + test, args, device)
    rows = base.episode_rows(validation + test, args.tie_tol)
    split_metrics = {split: base.summarize_split(rows, split, args) for split in ("validation", "test")}
    g2 = g2_shuffled_source_null(model, test, args, device)
    router_rows, g4, router_model = train_select_evaluate_router(train, validation, test, args)
    test_metrics = split_metrics["test"]
    neutral_dep = test_metrics["dependence_full_minus_masked_rr"]
    matched_dep = base.cluster_bootstrap(
        [row for row in rows if row["split"] == "test" and row["matched_donor_d1_rr"] is not None],
        lambda rs: base.mean(float(row["full_rr"]) - float(row["matched_donor_d1_rr"]) for row in rs),
        args.bootstrap_iters, args.seed + 3302)
    specificity = test_metrics["matched_true_vs_d1_half_tie"]
    adjusted = test_metrics["specificity_minus_placebo_half_tie"]
    gates = {
        "g1_fixed_row_and_label_provenance": (
            len(test) > 0 and not failures and reference_check["pass"]
        ),
        "g2_oracle_gain_ci_above_zero": lower(g2["observed_oracle_gain_cluster_bootstrap"]) > 0.0,
        "g2_oracle_above_legal_shuffled_p95": g2["observed_oracle_gain"] > g2["null_p95"],
        "g3_full_minus_neutral_mask_ci_above_zero": lower(neutral_dep) > 0.0,
        "g3_full_minus_matched_observed_source_ci_above_zero": lower(matched_dep) > 0.0,
        "g3_true_vs_d1_half_tie_ci_above_half": lower(specificity) > 0.5,
        "g3_placebo_adjusted_ci_above_zero": lower(adjusted) > 0.0,
        "g4_router_minus_best_fixed_ci_above_zero": lower(g4["router_minus_best_fixed_rr"]) > 0.0,
    }
    all_pass = all(gates.values())
    negative_control_pass = (
        gates["g1_fixed_row_and_label_provenance"]
        and not gates["g3_full_minus_neutral_mask_ci_above_zero"]
        and not gates["g3_full_minus_matched_observed_source_ci_above_zero"]
        and not gates["g3_true_vs_d1_half_tie_ci_above_half"]
        and not gates["g3_placebo_adjusted_ci_above_zero"]
    )
    if args.smoke:
        status = "SMOKE_COMPLETE_NON_CLAIM_BEARING"
    elif args.dose == 1.0:
        status = "RUN_COMPLETE_POSITIVE_CONTROL_PASS" if all_pass else "RUN_COMPLETE_GATE_MISS"
    elif args.dose == 0.0:
        status = "RUN_COMPLETE_NEGATIVE_CONTROL_PASS" if negative_control_pass else "RUN_COMPLETE_NEGATIVE_CONTROL_MISS"
    else:
        status = "RUN_COMPLETE_DOSE_CHECK"
    base.write_csv(args.output_dir / "r243_row_scores.csv", rows)
    base.write_csv(args.output_dir / "r243_label_audit.csv", label_rows)
    base.write_csv(args.output_dir / "r243_donor_map.csv", donor_rows)
    base.write_csv(args.output_dir / "r243_router_test_rows.csv", router_rows)
    payload = {"schema": "focalswap.r243.metrics.v2", "status": status, "claim_authorized": False,
               "dose": args.dose, "student_seed": args.seed, "teacher_seed": args.teacher_seed,
               "teacher_rank": args.teacher_rank, "teacher_noise_scale": args.teacher_noise_scale,
               "best_epoch": best_epoch, "item_text_best_epoch": item_text_epoch,
               "training_history": training_history, "split_metrics": split_metrics,
               "construction_counts": construction_counts,
               "integrity_checks": checks,
               "donor_summary": donor_summary,
               "r239_reference_check": reference_check,
               "g2_legal_shuffled_source_null": g2,
               "dependence_full_minus_matched_observed_source_rr": matched_dep,
               "g4_pre_outcome_router": g4, "promotion_gates": gates,
               "positive_control_all_gates_pass": all_pass,
               "negative_control_pass": negative_control_pass,
               "elapsed_seconds": time.time() - started}
    base.write_json(args.output_dir / "r243_metrics.json", payload)
    torch.save({"schema": "focalswap.r243.student_checkpoint.v2", "model_state": best_state,
                "router_weights": router_model[0], "router_feature_mean": router_model[1],
                "router_feature_scale": router_model[2],
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
               args.output_dir / "r243_checkpoint.pt")
    teacher_hashes = {name: base.hashlib.sha256(value.numpy().tobytes()).hexdigest() for name, value in teacher.items()}
    outputs = ("r243_row_scores.csv", "r243_label_audit.csv", "r243_donor_map.csv",
               "r243_router_test_rows.csv", "r243_metrics.json", "r243_checkpoint.pt")
    label_contract_hash = base.sha256_file(args.output_dir / "r243_label_audit.csv")
    manifest = {"schema": "focalswap.r243.manifest.v2", "status": status,
                "frozen_r239_contract": reference_check,
                "inputs": {"snapshot": str(args.snapshot), "item_embeddings": str(item_path), "query_embeddings": str(query_path)},
                "input_hashes": {"snapshot": base.sha256_directory(args.snapshot), "item_embeddings": base.sha256_file(item_path), "query_embeddings": base.sha256_file(query_path)},
                "teacher_weight_hashes_not_exposed_to_student": teacher_hashes,
                "planted_label_contract_sha256": label_contract_hash,
                "construction_counts": construction_counts,
                "integrity_checks": checks,
                "donor_summary": donor_summary,
                "outputs": {name: base.sha256_file(args.output_dir / name) for name in outputs}}
    base.write_json(args.output_dir / "r243_manifest.json", manifest)
    print(json.dumps({"status": status, "dose": args.dose, "seed": args.seed, "gates": gates}, sort_keys=True), flush=True)
    if args.smoke or args.dose == 0.5:
        return 0
    if args.dose == 1.0:
        return 0 if all_pass else 2
    return 0 if negative_control_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
