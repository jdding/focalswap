#!/usr/bin/env python3
"""R128-R130 official-code adapter gates for SARCheck.

The goal is deliberately narrow:

* R128 builds a tiny official-format KuaiSAR bundle and checks whether the
  official UniSAR code can instantiate its dataset/model and export rec scores.
* R129/R130 inspect official UnifiedSSR/SESRec schemas against the fixed
  SARCheck row contract and decide whether a same-task adapter is feasible.

This script does not train a model and does not use GPU.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import pickle
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    import joblib
except ImportError:  # pragma: no cover - checked in runtime reports
    joblib = None


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
EXAMPLES = ROOT / "data/processed/backbone_audit/examples.jsonl"
DEFAULT_OFFICIAL = Path("/private/tmp/sarcheck_r127_official_repos_20260611")
DEFAULT_WORK = Path("/private/tmp/sarcheck_r128_unisar_official_smoke")
DEFAULT_UNIFIEDSSR_WORK = Path("/private/tmp/sarcheck_r129_unifiedssr_official_smoke")
DEFAULT_SESREC_WORK = Path("/private/tmp/sarcheck_r130_sesrec_official_smoke")


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


@dataclass
class CompactMaps:
    user: dict[int, int]
    item: dict[int, int]
    query_token: dict[int, int]
    session: dict[str, int]


def compact_id(mapping: dict[Any, int], raw: Any) -> int:
    if raw not in mapping:
        mapping[raw] = len(mapping) + 1
    return mapping[raw]


def fixed_negs(
    positive: int,
    pools: list[list[int]],
    all_items: list[int],
    k: int,
    rng: random.Random,
) -> list[int]:
    negs: list[int] = []
    seen = {positive}
    for pool in pools:
        for item in pool:
            if item not in seen:
                seen.add(item)
                negs.append(item)
                if len(negs) >= k:
                    return negs
    # Avoid rebuilding and shuffling the full item universe for every row.
    # The bounded official-code runs can have thousands of rows and 99 eval
    # negatives; repeated full-list filtering makes bundle generation
    # CPU-bound on AutoDL before training starts.
    attempts = 0
    max_attempts = max(1000, 20 * k)
    while len(negs) < k and all_items and attempts < max_attempts:
        attempts += 1
        item = all_items[rng.randrange(len(all_items))]
        if item not in seen:
            seen.add(item)
            negs.append(item)
    if len(negs) < k:
        candidates = [i for i in all_items if i not in seen]
        rng.shuffle(candidates)
        negs.extend(candidates[: max(0, k - len(negs))])
    if len(negs) < k:
        filler = [i for i in all_items if i != positive] or [positive]
        while len(negs) < k:
            negs.append(filler[len(negs) % len(filler)])
    return negs[:k]


def reset_imports(names: list[str]) -> dict[str, Any]:
    old: dict[str, Any] = {}
    for name in names:
        if name in sys.modules:
            old[name] = sys.modules.pop(name)
    return old


def restore_imports(old: dict[str, Any], names: list[str]) -> None:
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(old)


def row_user_key(ex: dict[str, Any]) -> str:
    """Use row-scoped users so official user histories cannot bleed across splits."""
    return f"{ex.get('split', 'train')}::{ex.get('example_id')}::{ex.get('user_id')}"


def require_all_splits(split_rows: dict[str, list[Any]], context: str) -> None:
    missing = [split for split in ["train", "validation", "test"] if not split_rows.get(split)]
    if missing:
        raise ValueError(f"{context} missing required split rows: {missing}")


def reject_official_repo_workdir(work_dir: Path, repo: Path, context: str) -> None:
    try:
        work_resolved = work_dir.resolve()
        repo_resolved = repo.resolve()
    except FileNotFoundError:
        work_resolved = work_dir.absolute()
        repo_resolved = repo.absolute()
    if work_resolved == repo_resolved:
        raise ValueError(
            f"{context} work_dir must be a run-scoped scratch directory, not the official repo: {work_dir}"
        )


def compact_examples(
    examples: list[dict[str, Any]],
    row_limit: int,
    candidate_raw_by_example: dict[str, list[int]] | None = None,
) -> tuple[list[dict[str, Any]], CompactMaps]:
    selected = examples[:row_limit]
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
    return selected, maps


def compact_item_sequence(
    maps: CompactMaps,
    values: list[int],
    max_len: int,
    fallback: int,
) -> list[int]:
    seq = [maps.item[int(x)] for x in values if int(x) in maps.item][:max_len]
    return seq if seq else [fallback]


def compact_query_tokens(maps: CompactMaps, values: list[int], max_len: int) -> list[int]:
    seq = [maps.query_token[int(x)] for x in values if int(x) in maps.query_token][:max_len]
    return seq if seq else [1]


def ensure_min_sequence(seq: list[int], min_len: int, filler: int) -> list[int]:
    """Pad before the target so official next-item collators never see empty slices."""
    if not seq:
        seq = [filler]
    if len(seq) < min_len:
        seq = [filler] * (min_len - len(seq)) + seq
    return seq


def finite_score_row(view: str, status: str, scores: torch.Tensor | None, error: str = "") -> dict[str, Any]:
    if scores is None:
        return {
            "view": view,
            "status": status,
            "shape": [],
            "finite": False,
            "mean": math.nan,
            "std": math.nan,
            "error": error,
        }
    return {
        "view": view,
        "status": status,
        "shape": list(scores.shape),
        "finite": bool(torch.isfinite(scores).all().item()),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "error": error,
    }


def build_unisar_tiny_bundle(
    examples: list[dict[str, Any]],
    work_dir: Path,
    row_limit: int,
    neg_k: int,
    seed: int,
    score_rows_path: Path | None = None,
    candidate_raw_by_example: dict[str, list[int]] | None = None,
    enforce_official_limits: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    data_root = work_dir / "data/KuaiSAR"
    if data_root.exists():
        shutil.rmtree(data_root)
    (data_root / "dataset").mkdir(parents=True, exist_ok=True)
    (data_root / "vocab").mkdir(parents=True, exist_ok=True)

    selected, maps = compact_examples(examples, row_limit, candidate_raw_by_example)

    all_items = list(maps.item.values())
    max_user = max(maps.user.values()) if maps.user else 0
    max_item = max(maps.item.values()) if maps.item else 0
    max_token = max(maps.query_token.values()) if maps.query_token else 0
    max_session = max(maps.session.values()) if maps.session else 0
    neutral_item_id = max_item + 1
    official_limits = {
        "user_id_num": 22700,
        "item_id_num": 673415 + 1,
        "word_id_num": 394912,
    }
    if enforce_official_limits and max_user >= official_limits["user_id_num"]:
        raise ValueError(f"compact user id {max_user} exceeds UniSAR KuaiSAR user_id_num")
    if enforce_official_limits and neutral_item_id >= official_limits["item_id_num"]:
        raise ValueError(f"compact item id {neutral_item_id} exceeds UniSAR KuaiSAR item_id_num")
    if enforce_official_limits and max_token >= official_limits["word_id_num"]:
        raise ValueError(f"compact token id {max_token} exceeds UniSAR KuaiSAR word_id_num")

    user_vocab: dict[int, dict[str, list[int | float]]] = {}
    neutral_src_session_id = max_session + 1
    session_keyword = np.zeros((neutral_src_session_id + 1, 50), dtype=np.int64)
    session_pos = np.zeros((neutral_src_session_id + 1, 5), dtype=np.int64)
    query_vocab: list[list[int]] = [[] for _ in range(max_token + 1)]

    rec_rows: list[dict[str, Any]] = []
    src_rows: list[dict[str, Any]] = []
    rec_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    src_rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    score_rows: list[dict[str, Any]] = []

    for row_idx, ex in enumerate(selected):
        uid = maps.user[row_user_key(ex)]
        rec_his = [maps.item[int(x)] for x in ex.get("H_pre", [])][-30:]
        search_items = [maps.item[int(x)] for x in ex.get("E_search_all", [])][:5]
        target = maps.item[int(ex["primary_target_item"])]
        target_pool = [maps.item[int(x)] for x in ex.get("Y_post_rec", [])]
        search_pool = [maps.item[int(x)] for x in ex.get("E_search_all", [])]
        hist_pool = [maps.item[int(x)] for x in ex.get("H_pre", [])]
        if candidate_raw_by_example is None:
            neg_items = fixed_negs(target, [search_pool, target_pool, hist_pool], all_items, neg_k, rng)
        else:
            raw_candidates = candidate_raw_by_example[str(ex["example_id"])]
            raw_target = int(ex["primary_target_item"])
            if len(raw_candidates) != neg_k + 1 or int(raw_candidates[0]) != raw_target:
                raise ValueError(f"invalid canonical candidates for {ex['example_id']}")
            if len(set(int(item) for item in raw_candidates)) != len(raw_candidates):
                raise ValueError(f"duplicate canonical candidates for {ex['example_id']}")
            neg_items = [maps.item[int(item)] for item in raw_candidates[1:]]

        sid = maps.session[ex["example_id"]]
        q_tokens = [maps.query_token[int(x)] for x in ex.get("query_tokens", [])][:50]
        for token in q_tokens:
            query_vocab[token] = [token]
        session_keyword[sid, : len(q_tokens)] = q_tokens
        session_pos[sid, : len(search_items)] = search_items

        # Timestamps are synthetic but preserve the official sampler's ordering:
        # recommendation history precedes the focal search session.
        user_vocab[uid] = {
            "rec_his": rec_his,
            "rec_his_ts": list(range(1, len(rec_his) + 1)),
            "src_session_his": [sid],
            "src_session_his_ts": [len(rec_his) + 1],
        }

        rec_row = {
            "user_id": uid,
            "item_id": target,
            "neg_items": neg_items,
            "rec_his": len(rec_his),
            "src_session_his": 1,
        }
        src_row = {
            **rec_row,
            "keyword": q_tokens,
        }
        rec_rows.append(rec_row)
        src_rows.append(src_row)
        split = str(ex.get("split", "train"))
        if split not in rec_rows_by_split:
            split = "train"
        rec_rows_by_split[split].append(rec_row)
        src_rows_by_split[split].append(src_row)
        score_rows.append(
            {
                "row": row_idx,
                "example_id": ex["example_id"],
                "compact_user": uid,
                "compact_item": target,
                "neg_count": len(neg_items),
                "rec_his_len": len(rec_his),
                "src_session_his_len": 1,
            }
        )

    split_to_file = {
        "train": ("rec_train.pkl", "src_train.pkl"),
        "validation": ("rec_val.pkl", "src_val.pkl"),
        "test": ("rec_test.pkl", "src_test.pkl"),
    }
    require_all_splits(rec_rows_by_split, "UniSAR rec bundle")
    require_all_splits(src_rows_by_split, "UniSAR src bundle")
    for split, (rec_name, src_name) in split_to_file.items():
        rec_df = pd.DataFrame(rec_rows_by_split[split])
        src_df = pd.DataFrame(src_rows_by_split[split])
        rec_df.to_pickle(data_root / "dataset" / rec_name)
        src_df.to_pickle(data_root / "dataset" / src_name)
    user_np_len = max(max_user + 1, 2)
    item_np_len = max(neutral_item_id + 1, 2)
    user_map_vocab = {
        "onehot_feat1": np.zeros(user_np_len, dtype=np.int64),
        "onehot_feat2": np.zeros(user_np_len, dtype=np.int64),
        "search_active_level": np.ones(user_np_len, dtype=np.int64),
        "rec_active_level": np.ones(user_np_len, dtype=np.int64),
    }
    item_map_vocab = {
        "caption": np.zeros((item_np_len, 50), dtype=np.int64),
        "first_level_category_id": np.zeros(item_np_len, dtype=np.int64),
        "second_level_category_id": np.zeros(item_np_len, dtype=np.int64),
    }
    for raw_item, cid in maps.item.items():
        tok = (abs(int(raw_item)) % max(max_token, 1)) + 1
        tok = min(tok, max(max_token, 1))
        item_map_vocab["caption"][cid, 0] = tok
        item_map_vocab["first_level_category_id"][cid] = abs(int(raw_item)) % 37
        item_map_vocab["second_level_category_id"][cid] = abs(int(raw_item)) % 296

    dump_pickle(data_root / "vocab/user_vocab.pkl", user_vocab)
    dump_pickle(data_root / "vocab/query_vocab.pkl", query_vocab)
    dump_pickle(data_root / "vocab/user_vocab_np.pkl", user_map_vocab)
    dump_pickle(data_root / "vocab/item_vocab_np.pkl", item_map_vocab)
    dump_pickle(
        data_root / "vocab/src_session_vocab_np.pkl",
        {"keyword": session_keyword, "pos_items": session_pos},
    )
    if score_rows_path is None:
        score_rows_path = REPORTS / "r128_unisar_official_score_rows.csv"
    write_csv(score_rows_path, score_rows)

    return {
        "work_dir": str(work_dir),
        "data_root": str(data_root),
        "rows": len(rec_rows),
        "negatives_per_row": neg_k,
        "compact_users": max_user,
        "compact_items": max_item,
        "compact_tokens": max_token,
        "compact_sessions": max_session,
        "neutral_item_id": neutral_item_id,
        "neutral_src_session_id": neutral_src_session_id,
        "split_rows": {k: len(v) for k, v in rec_rows_by_split.items()},
        "score_rows": str(score_rows_path),
        "candidate_contract": (
            "canonical_raw_item_ledger" if candidate_raw_by_example is not None else "model_local"
        ),
        "official_limits_enforced": enforce_official_limits,
    }


def tensor_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.to(device)
        else:
            out[key] = val
    return out


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.clone()
        else:
            out[key] = val
    return out


def masked_view(
    batch: dict[str, Any],
    view: str,
    neutral_src_session_id: int | None = None,
    neutral_item_id: int | None = None,
) -> dict[str, Any]:
    out = clone_batch(batch)
    if view == "full_sar":
        return out
    all_his = out["all_his"]
    all_his_type = out["all_his_type"]
    if view in {"history_only", "search_masked"}:
        if neutral_src_session_id is None:
            out["all_his"] = all_his.masked_fill(all_his_type == 2, 0)
        else:
            out["all_his"] = all_his.masked_fill(all_his_type == 2, neutral_src_session_id)
    elif view == "search_only":
        if neutral_item_id is None:
            out["all_his"] = all_his.masked_fill(all_his_type == 1, 0)
        else:
            out["all_his"] = all_his.masked_fill(all_his_type == 1, neutral_item_id)
    else:
        raise ValueError(f"unknown view: {view}")
    return out


def run_unisar_official_smoke(
    official_root: Path,
    work_dir: Path,
    examples: list[dict[str, Any]],
    row_limit: int,
    neg_k: int,
    seed: int,
) -> dict[str, Any]:
    repo = official_root / "UniSAR"
    if not repo.exists():
        return {
            "run_id": "R128",
            "status": "blocked",
            "reason": f"UniSAR repo not found: {repo}",
        }
    reject_official_repo_workdir(work_dir, repo, "R128 UniSAR smoke")
    bundle = build_unisar_tiny_bundle(examples, work_dir, row_limit, neg_k, seed)

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    try:
        os.chdir(work_dir)
        sys.path.insert(0, str(repo))
        const = importlib.import_module("utils.const")
        utils = importlib.import_module("utils.utils")
        dataset_mod = importlib.import_module("utils.dataset")
        model_mod = importlib.import_module("models.UniSAR")

        const.init_setting_KuaiSAR()
        user_vocab = utils.load_pickle(const.user_vocab)
        rec_dataset = dataset_mod.RecDataSet("test", user_vocab=user_vocab)
        feed = [rec_dataset[i] for i in range(min(2, len(rec_dataset)))]
        batch = rec_dataset.collate_batch(feed)

        device = torch.device("cpu")
        args = SimpleNamespace(
            device=device,
            model_path=str(work_dir / "model"),
            dropout=0.0,
            batch_size=len(feed),
            num_layers=1,
            num_heads=2,
            q_i_cl_temp=0.5,
            q_i_cl_weight=0.0,
            his_cl_temp=0.1,
            his_cl_weight=0.0,
            pred_hid_units=[200, 80, 1],
        )
        model = model_mod.UniSAR(args)
        model.eval()
        batch = tensor_to_device(batch, device)

        required_views = {"full_sar", "history_only", "search_only", "search_masked"}
        view_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for view in ["full_sar", "history_only", "search_only", "search_masked"]:
                try:
                    view_batch = masked_view(
                        batch,
                        view,
                        int(bundle["neutral_src_session_id"]),
                        int(bundle["neutral_item_id"]),
                    )
                    scores = model.rec_predict(view_batch)
                    finite = bool(torch.isfinite(scores).all().item())
                    view_rows.append(
                        {
                            "view": view,
                            "required": view in required_views,
                            "status": "pass",
                            "shape": list(scores.shape),
                            "finite": finite,
                            "mean": float(scores.mean().item()),
                            "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
                            "error": "",
                        }
                    )
                except Exception as exc:
                    view_rows.append(
                        {
                            "view": view,
                            "required": view in required_views,
                            "status": "blocked",
                            "shape": [],
                            "finite": False,
                            "mean": math.nan,
                            "std": math.nan,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

        write_csv(REPORTS / "r128_unisar_official_view_scores.csv", view_rows)
        required_ok = all(
            r["status"] == "pass" and bool(r["finite"])
            for r in view_rows
            if r["required"]
        )
        status = "pass" if required_ok else "blocked"
        return {
            "run_id": "R128",
            "status": status,
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "bundle": bundle,
            "checks": {
                "official_import": True,
                "official_dataset_instantiated": True,
                "official_model_instantiated": True,
                "required_rec_predict_forward": required_ok,
                "views_exported": [r["view"] for r in view_rows if r["status"] == "pass"],
                "required_views": sorted(required_views),
                "all_required_scores_finite": required_ok,
            },
            "outputs": {
                "view_scores": str(REPORTS / "r128_unisar_official_view_scores.csv"),
                "score_rows": bundle["score_rows"],
            },
            "limits": [
                "CPU smoke only; no official training launched.",
                "Tiny official-format bundle is converted from R110 SARCheck windows, not the official processed KuaiSAR release.",
                "Contrastive losses are disabled for forward smoke; this validates code/data/readout contract, not model quality.",
                "Masked views use neutral non-informative item/session ids instead of all-padding histories because the official CPU Transformer path can fail when one history side is entirely padding.",
            ],
        }
    except Exception as exc:  # pragma: no cover - report path
        return {
            "run_id": "R128",
            "status": "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo) if repo.exists() else None,
            "bundle": bundle,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def build_unifiedssr_tiny_bundle(
    examples: list[dict[str, Any]],
    work_dir: Path,
    row_limit: int,
) -> dict[str, Any]:
    if joblib is None:
        raise RuntimeError("joblib is required because official UnifiedSSR uses joblib.load")
    data_root = work_dir / "datasets" / "JDsearch"
    if data_root.exists():
        shutil.rmtree(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    selected, maps = compact_examples(examples, row_limit)
    max_user = max(maps.user.values()) if maps.user else 1
    max_item = max(maps.item.values()) if maps.item else 1
    max_token = max(maps.query_token.values()) if maps.query_token else 1
    neutral_item = max_item + 1
    neutral_token = max_token + 1

    rec_records: list[dict[str, Any]] = []
    search_records: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    masked_records: list[dict[str, Any]] = []
    for ex in selected:
        uid = maps.user[row_user_key(ex)]
        target = maps.item[int(ex["primary_target_item"])]
        hist = compact_item_sequence(maps, ex.get("H_pre", []), 8, neutral_item)
        search_items = compact_item_sequence(maps, ex.get("E_search_all", []), 5, neutral_item)
        q_tokens = compact_query_tokens(maps, ex.get("query_tokens", []), 8)
        # UnifiedSSR's official finetune collator uses pid_list[:-3] and
        # pid_list[1:-2]. If any training sequence has length <= 3, the target
        # slice is empty and negative sampling returns ragged tensors. Pad
        # before the positive target so the official training objective remains
        # next-item prediction while satisfying this schema invariant.
        rec_seq = ensure_min_sequence((hist + [target])[-9:], 5, neutral_item)
        search_seq = ensure_min_sequence((search_items + [target])[-6:], 5, neutral_item)
        full_seq = ensure_min_sequence((hist[-5:] + search_items + [target])[-10:], 5, neutral_item)

        rec_records.append(
            {"uid": uid, "flag": "recommendation", "pid_list": rec_seq, "length": len(rec_seq)}
        )
        masked_records.append(
            {"uid": uid, "flag": "recommendation", "pid_list": rec_seq, "length": len(rec_seq)}
        )
        search_records.append(
            {
                "uid": uid,
                "flag": "search",
                "pid_list": search_seq,
                "qry_list": [q_tokens for _ in search_seq],
                "length": len(search_seq),
            }
        )
        full_records.append(
            {
                "uid": uid,
                "flag": "search",
                "pid_list": full_seq,
                "qry_list": [q_tokens for _ in full_seq],
                "length": len(full_seq),
            }
        )

    for phase in ["pretrain", "finetune"]:
        joblib.dump(rec_records, data_root / f"{phase}_recommendation.pkl")
        joblib.dump(search_records, data_root / f"{phase}_search.pkl")
    pd.DataFrame([[max_user, neutral_item, neutral_token]], columns=["user_num", "product_num", "term_num"]).to_csv(
        data_root / "meta.csv",
        sep="\t",
        index=False,
    )
    return {
        "work_dir": str(work_dir),
        "data_root": str(data_root),
        "rows": len(selected),
        "compact_users": max_user,
        "compact_items": max_item,
        "compact_tokens": max_token,
        "compact_sessions": max(maps.session.values()) if maps.session else 0,
        "neutral_item_id": neutral_item,
        "neutral_token_id": neutral_token,
        "records": {
            "history_only": rec_records,
            "search_only": search_records,
            "full_sar": full_records,
            "search_masked": masked_records,
        },
    }


def run_unifiedssr_official_smoke(
    official_root: Path,
    work_dir: Path,
    examples: list[dict[str, Any]],
    row_limit: int,
    seed: int,
) -> dict[str, Any]:
    repo = official_root / "UnifiedSSR"
    if not repo.exists():
        return {"run_id": "R129", "status": "blocked", "reason": f"UnifiedSSR repo not found: {repo}"}
    try:
        bundle = build_unifiedssr_tiny_bundle(examples, work_dir, row_limit)
    except Exception as exc:
        return {
            "run_id": "R129",
            "status": "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    old_cwd = Path.cwd()
    old_path = list(sys.path)
    reset_names = ["data", "model", "batch", "modules"]
    old_modules = reset_imports(reset_names)
    try:
        torch.manual_seed(seed)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        data_mod = importlib.import_module("data")
        model_mod = importlib.import_module("model")
        batch_mod = importlib.import_module("batch")

        _ = data_mod.UnifiedDataset("finetune", ["recommendation", "search"], data_root=bundle["data_root"])
        args = SimpleNamespace(
            user_vocab=int(bundle["compact_users"]) + 1,
            product_vocab=int(bundle["neutral_item_id"]) + 1,
            term_vocab=int(bundle["neutral_token_id"]) + 2,
            emb_size=16,
            hid_size=32,
            sub_seq_num=2,
            enc_num_layer=1,
            num_head=1,
            tasks=["recommendation", "search"],
            dropout=0.0,
            train_num_neg=2,
            test_num_neg=7,
            padding_value=0,
        )
        model = model_mod.get_model(args)
        model.eval()

        view_rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for view, records in bundle["records"].items():
                try:
                    task, batch = batch_mod.collate_test(records[: min(4, len(records))], args)
                    enc = model(task, batch)
                    if task == "recommendation":
                        scores = model.next_product_predict(enc, batch["pid_last_idx"], batch["pid_pred"])
                    else:
                        p_enc, q_enc = enc
                        scores = model.next_product_search(p_enc, q_enc, batch["pid_last_idx"], batch["pid_pred"])
                    view_rows.append(finite_score_row(view, "pass", scores))
                except Exception as exc:
                    view_rows.append(finite_score_row(view, "blocked", None, f"{type(exc).__name__}: {exc}"))

        write_csv(REPORTS / "r129_unifiedssr_official_view_scores.csv", view_rows)
        required_ok = all(row["status"] == "pass" and row["finite"] for row in view_rows)
        return {
            "run_id": "R129",
            "method": "UnifiedSSR",
            "status": "pass" if required_ok else "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "bundle": {k: v for k, v in bundle.items() if k != "records"},
            "checks": {
                "official_import": True,
                "official_dataset_instantiated": True,
                "official_model_instantiated": True,
                "official_forward_pass": required_ok,
                "views_exported": [r["view"] for r in view_rows if r["status"] == "pass"],
            },
            "outputs": {"view_scores": str(REPORTS / "r129_unifiedssr_official_view_scores.csv")},
            "limits": [
                "CPU smoke only; no official training launched.",
                "Tiny bundle is converted from SARCheck windows into official UnifiedSSR record fields.",
                "Full-SAR and search-only views use the official search branch; history-only and search-masked use the official recommendation branch.",
            ],
        }
    except Exception as exc:
        return {
            "run_id": "R129",
            "method": "UnifiedSSR",
            "status": "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "bundle": {k: v for k, v in bundle.items() if k != "records"},
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        restore_imports(old_modules, reset_names)
        os.chdir(old_cwd)
        sys.path[:] = old_path


def item_features(item_id: int) -> tuple[int, int, int]:
    if item_id == 0:
        return (0, 0, 0)
    return (item_id, (item_id % 37) + 1, (item_id % 36) + 1)


def pad_items(seq: list[int], max_len: int, neutral_item: int) -> list[int]:
    seq = seq[-max_len:]
    if not seq:
        seq = [neutral_item]
    if len(seq) < max_len:
        seq = seq + [neutral_item] * (max_len - len(seq))
    return seq[:max_len]


def sesrec_item_tensors(item_ids: list[int], shape: tuple[int, ...]) -> list[torch.Tensor]:
    attrs = [item_features(int(x)) for x in item_ids]
    out: list[torch.Tensor] = []
    for col in range(3):
        vals = [row[col] for row in attrs]
        out.append(torch.tensor(vals, dtype=torch.long).reshape(shape))
    return out


def build_sesrec_inputs(
    examples: list[dict[str, Any]],
    row_limit: int,
    view: str,
) -> tuple[list[Any], dict[str, Any]]:
    selected, maps = compact_examples(examples, row_limit)
    max_item = max(maps.item.values()) if maps.item else 1
    max_token = max(maps.query_token.values()) if maps.query_token else 1
    neutral_item = max_item + 1
    neutral_query = max_token + 1
    batch = selected[: min(4, len(selected))]
    max_rec = 150
    max_src = 25
    max_click = 5
    max_words = 6

    users = [
        torch.tensor([maps.user[row_user_key(ex)] for ex in batch], dtype=torch.long),
        torch.ones(len(batch), dtype=torch.long),
        torch.ones(len(batch), dtype=torch.long),
        torch.ones(len(batch), dtype=torch.long),
    ]
    rec_lists: list[list[int]] = []
    src_query: list[list[int]] = []
    src_source: list[list[int]] = []
    src_click_items: list[list[list[int]]] = []
    src_words: list[list[list[int]]] = []
    candidates: list[int] = []
    source_to_id = {"USER_INPUT": 1, "TRENDING": 2, "HISTORY": 3}

    for ex in batch:
        target = maps.item[int(ex["primary_target_item"])]
        candidates.append(target)
        hist = compact_item_sequence(maps, ex.get("H_pre", []), 24, neutral_item)
        search_items = compact_item_sequence(maps, ex.get("E_search_all", []), max_click, neutral_item)
        q_tokens = compact_query_tokens(maps, ex.get("query_tokens", []), max_words)
        if view == "search_only":
            rec_lists.append(pad_items([neutral_item], max_rec, neutral_item))
        else:
            rec_lists.append(pad_items(hist, max_rec, neutral_item))

        if view in {"history_only", "search_masked"}:
            q_id = neutral_query
            q_words = [neutral_query]
            clicks = [neutral_item]
            src_id = 1
        else:
            q_id = maps.session[ex["example_id"]]
            q_words = q_tokens
            clicks = search_items
            src_id = source_to_id.get(str(ex.get("source", "USER_INPUT")), 1)
        src_query.append([q_id] + [neutral_query] * (max_src - 1))
        src_source.append([src_id] + [1] * (max_src - 1))
        padded_clicks = (clicks + [neutral_item] * max_click)[:max_click]
        src_click_items.append([padded_clicks] + [[neutral_item] * max_click for _ in range(max_src - 1)])
        padded_words = (q_words + [neutral_query] * max_words)[:max_words]
        src_words.append([padded_words] + [[neutral_query] * max_words for _ in range(max_src - 1)])

    rec_flat = [item for row in rec_lists for item in row]
    rec_tensors = sesrec_item_tensors(rec_flat, (len(batch), max_rec))
    item_tensors = sesrec_item_tensors(candidates, (len(batch),))
    click_flat = [item for row in src_click_items for sess in row for item in sess]
    click_tensors = sesrec_item_tensors(click_flat, (len(batch), max_src, max_click))
    src_his = [
        torch.tensor(src_query, dtype=torch.long),
        torch.tensor(src_source, dtype=torch.long),
        click_tensors,
        torch.tensor(src_words, dtype=torch.long),
    ]
    return [users, rec_tensors, src_his, item_tensors], {
        "rows": len(selected),
        "scored_rows": len(batch),
        "compact_users": max(maps.user.values()) if maps.user else 1,
        "compact_items": max_item,
        "compact_tokens": max_token,
        "compact_sessions": max(maps.session.values()) if maps.session else 1,
        "neutral_item_id": neutral_item,
        "neutral_query_id": neutral_query,
    }


def run_sesrec_official_smoke(
    official_root: Path,
    work_dir: Path,
    examples: list[dict[str, Any]],
    row_limit: int,
    seed: int,
) -> dict[str, Any]:
    repo = official_root / "SESREC-SIGIR-2023"
    if not repo.exists():
        return {"run_id": "R130", "status": "blocked", "reason": f"SESRec repo not found: {repo}"}
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    reset_names = ["config", "config.const", "models", "models.SESRec", "models.Inputs", "models.module"]
    old_modules = reset_imports(reset_names)
    try:
        torch.manual_seed(seed)
        work_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        const = importlib.import_module("config.const")
        inputs_mod = importlib.import_module("models.Inputs")
        sesrec_mod = importlib.import_module("models.SESRec")
        const.init_dataset_setting_commercial()
        inputs_mod.init_data_attribute_ls("commercial")
        config = {
            "num_blocks": 1,
            "num_heads": 2,
            "dropout": 0.0,
            "infoNCE_temp": 0.1,
            "triplet_margin": 0.1,
            "pred_hid_units": [200, 80, 1],
        }
        model = sesrec_mod.SESRec(config)
        model.eval()
        view_rows: list[dict[str, Any]] = []
        bundle_meta: dict[str, Any] | None = None
        with torch.no_grad():
            for view in ["full_sar", "history_only", "search_only", "search_masked"]:
                try:
                    input_data, meta = build_sesrec_inputs(examples, row_limit, view)
                    bundle_meta = meta
                    scores = model.predict(input_data)
                    view_rows.append(finite_score_row(view, "pass", scores))
                except Exception as exc:
                    view_rows.append(finite_score_row(view, "blocked", None, f"{type(exc).__name__}: {exc}"))
        write_csv(REPORTS / "r130_sesrec_official_view_scores.csv", view_rows)
        required_ok = all(row["status"] == "pass" and row["finite"] for row in view_rows)
        return {
            "run_id": "R130",
            "method": "SESRec",
            "status": "pass" if required_ok else "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "bundle": {
                "work_dir": str(work_dir),
                **(bundle_meta or {}),
            },
            "checks": {
                "official_import": True,
                "official_model_instantiated": True,
                "official_predict_forward": required_ok,
                "views_exported": [r["view"] for r in view_rows if r["status"] == "pass"],
            },
            "outputs": {"view_scores": str(REPORTS / "r130_sesrec_official_view_scores.csv")},
            "limits": [
                "CPU smoke only; no official training launched.",
                "The smoke uses official SESRec model code and SARCheck-derived tensors rather than the private commercial processed bundle.",
                "Masked views use neutral non-informative ids to avoid all-padding Transformer paths.",
            ],
        }
    except Exception as exc:
        return {
            "run_id": "R130",
            "method": "SESRec",
            "status": "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        restore_imports(old_modules, reset_names)
        os.chdir(old_cwd)
        sys.path[:] = old_path


def sesrec_source_id(source: str) -> int:
    return {"USER_INPUT": 1, "TRENDING": 2, "HISTORY": 3}.get(source, 1)


def write_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def build_sesrec_trainer_bundle(
    repo: Path,
    examples: list[dict[str, Any]],
    row_limit: int,
    neg_k: int,
    seed: int,
    candidate_raw_by_example: dict[str, list[int]] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    data_root = repo / "data" / "commercial"
    if data_root.exists():
        shutil.rmtree(data_root)
    (data_root / "dataset").mkdir(parents=True, exist_ok=True)
    (data_root / "vocab").mkdir(parents=True, exist_ok=True)

    selected, maps = compact_examples(examples, row_limit, candidate_raw_by_example)
    all_items = sorted(maps.item.values())
    max_item = max(all_items) if all_items else 1
    neutral_item = max_item + 1
    neutral_query = (max(maps.query_token.values()) if maps.query_token else 1) + 1
    neutral_session = (max(maps.session.values()) if maps.session else 0) + 1

    user_vocab: dict[int, dict[str, Any]] = {}
    item_vocab: dict[int, dict[str, int]] = {}
    src_session_vocab: dict[str, dict[str, Any]] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    fallback_train_rows: list[dict[str, Any]] = []
    fallback_eval_rows: list[dict[str, Any]] = []

    def ensure_item(cid: int) -> None:
        if cid not in item_vocab:
            item_vocab[cid] = {
                "id": cid,
                "type1": (cid % 37) + 1,
                "cate": (cid % 36) + 1,
            }

    ensure_item(neutral_item)
    for cid in all_items:
        ensure_item(cid)

    for ex in selected:
        uid = maps.user[row_user_key(ex)]
        hist = compact_item_sequence(maps, ex.get("H_pre", []), 50, neutral_item)
        search_items = compact_item_sequence(maps, ex.get("E_search_all", []), 5, neutral_item)
        q_tokens = compact_query_tokens(maps, ex.get("query_tokens", []), 6)
        sid = str(maps.session[ex["example_id"]])
        target = maps.item[int(ex["primary_target_item"])]
        src_session_vocab[sid] = {
            "query_id": maps.session[ex["example_id"]],
            "search_source": sesrec_source_id(str(ex.get("source", "USER_INPUT"))),
            "click_items": search_items[:5],
            "relevant_item": search_items[:1] or [neutral_item],
            "keyword_seg": q_tokens[:6],
        }
        user_vocab[uid] = {
            "id": uid,
            "gender": 1,
            "age": 1,
            "src_level": 1,
            "rec_his": hist,
            "src_his": [sid],
        }

        split = str(ex.get("split", "train"))
        if split not in split_rows:
            split = "train"
        pos_row = {"u_id": uid, "i_id": target, "rec_his": len(hist), "src_his": 1, "label": 1.0}
        fallback_train_rows.append(pos_row)
        if candidate_raw_by_example is None:
            neg_items = fixed_negs(
                target,
                [
                    [maps.item[int(x)] for x in ex.get("E_search_all", []) if int(x) in maps.item],
                    [maps.item[int(x)] for x in ex.get("H_pre", []) if int(x) in maps.item],
                    all_items,
                ],
                all_items + [neutral_item],
                neg_k,
                rng,
            )
        else:
            raw_candidates = candidate_raw_by_example[str(ex["example_id"])]
            raw_target = int(ex["primary_target_item"])
            if len(raw_candidates) != neg_k + 1 or int(raw_candidates[0]) != raw_target:
                raise ValueError(f"invalid canonical candidates for {ex['example_id']}")
            if len(set(int(item) for item in raw_candidates)) != len(raw_candidates):
                raise ValueError(f"duplicate canonical candidates for {ex['example_id']}")
            neg_items = [maps.item[int(item)] for item in raw_candidates[1:]]
        candidates = [(target, 1.0)] + [(item, 0.0) for item in neg_items]
        eval_rows: list[dict[str, Any]] = []
        for item, label in candidates:
            ensure_item(item)
            eval_rows.append({"u_id": uid, "i_id": item, "rec_his": len(hist), "src_his": 1, "label": label})
        fallback_eval_rows.extend(eval_rows)
        if split == "train":
            split_rows[split].append(pos_row)
        else:
            split_rows[split].extend(eval_rows)

    require_all_splits(split_rows, "SESRec trainer bundle")

    file_map = {
        "train": "train_inter.tsv",
        "validation": "valid_inter.tsv",
        "test": "test_inter.tsv",
    }
    for split, filename in file_map.items():
        pd.DataFrame(split_rows[split]).to_csv(data_root / "dataset" / filename, sep="\t", index=False)

    write_pickle(data_root / "vocab" / "user_vocab.pickle", user_vocab)
    write_pickle(data_root / "vocab" / "item_vocab.pickle", item_vocab)
    write_pickle(data_root / "vocab" / "s_session_vocab.pickle", src_session_vocab)
    return {
        "data_root": str(data_root),
        "rows": len(selected),
        "split_rows": {k: len(v) for k, v in split_rows.items()},
        "compact_users": max(maps.user.values()) if maps.user else 0,
        "compact_items": max_item,
        "compact_tokens": max(maps.query_token.values()) if maps.query_token else 0,
        "compact_sessions": max(maps.session.values()) if maps.session else 0,
        "neutral_item_id": neutral_item,
        "neutral_query_id": neutral_query,
        "neutral_session_id": neutral_session,
        "negatives_per_eval_positive": neg_k,
        "candidate_contract": (
            "canonical_raw_item_ledger" if candidate_raw_by_example is not None else "model_local"
        ),
    }


def run_sesrec_trainer_wrapper_smoke(
    official_root: Path,
    examples: list[dict[str, Any]],
    row_limit: int,
    neg_k: int,
    seed: int,
) -> dict[str, Any]:
    repo = official_root / "SESREC-SIGIR-2023"
    if not repo.exists():
        return {"run_id": "R130T", "status": "blocked", "reason": f"SESRec repo not found: {repo}"}
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    reset_names = [
        "config",
        "config.const",
        "models",
        "models.SESRec",
        "models.Inputs",
        "models.module",
        "recommender",
        "utils",
        "utils.data",
        "utils.data_sampler",
        "utils.data_utils",
    ]
    old_modules = reset_imports(reset_names)
    try:
        bundle = build_sesrec_trainer_bundle(repo, examples, row_limit, neg_k, seed)
        torch.manual_seed(seed)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        const = importlib.import_module("config.const")
        inputs_mod = importlib.import_module("models.Inputs")
        data_mod = importlib.import_module("utils.data")
        recommender_mod = importlib.import_module("recommender")
        const.init_dataset_setting_commercial()
        inputs_mod.init_data_attribute_ls("commercial")
        flags = SimpleNamespace(
            dataset_name="commercial",
            model="SESRec",
            use_gpu=False,
            gpu_id=0,
            batch_size=8,
            num_negs=4,
            name="sarcheck_r130t",
            description="sesrec_trainer_wrapper_smoke",
            workspace=str(Path(bundle["data_root"]).parent / "workspace"),
            tb=False,
            train_tb=False,
            verbose=False,
            epochs=1,
        )
        dm = SimpleNamespace(
            dataset_name="commercial",
            batch_size=8,
            num_negs=4,
            load_path=const.load_path,
            user_vocab=const.user_vocab,
            item_vocab=const.item_vocab,
            src_session_vocab=const.src_session_vocab,
        )
        rec = recommender_mod.SESRec_Recommender(
            flags,
            flags.workspace,
            dm,
            {
                "git_id": "sarcheck_r130t",
                "infoNCE_neg_sample": 8,
                "infoNCE_weight": 0.1,
                "triplet_weight": 0.001,
            },
        )
        train_ds = data_mod.SESRec_Dataset(const.train_file, dm, True)
        valid_ds = data_mod.SESRec_Dataset(const.valid_file, dm, False)
        neg_ds = data_mod.SESRec_neg_sample_Dataset(const.valid_file, dm)
        train_batch = next(iter(DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=0)))
        neg_batch = next(iter(DataLoader(neg_ds, batch_size=8, shuffle=False, num_workers=0)))
        losses = rec.get_loss(list(train_batch), neg_batch)
        loss_values = [float(x.detach().cpu().item()) for x in losses]
        valid_batch = next(iter(DataLoader(valid_ds, batch_size=min(16, len(valid_ds)), shuffle=False, num_workers=0)))
        scores = rec.predict(valid_batch)
        finite = bool(torch.isfinite(scores).all().item())
        return {
            "run_id": "R130T",
            "method": "SESRec",
            "status": "pass" if finite and all(math.isfinite(x) for x in loss_values) else "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo),
            "bundle": bundle,
            "checks": {
                "official_dataset_instantiated": True,
                "official_recommender_instantiated": True,
                "official_train_loss_forward": all(math.isfinite(x) for x in loss_values),
                "official_predict_forward": finite,
                "train_dataset_rows": len(train_ds),
                "valid_dataset_rows": len(valid_ds),
            },
            "loss_values": {
                "bce": loss_values[0],
                "feature_align": loss_values[1],
                "interest_contrast": loss_values[2],
            },
            "predict_scores": {
                "shape": list(scores.shape),
                "finite": finite,
                "mean": float(scores.mean().item()),
                "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
            },
            "limits": [
                "CPU trainer-wrapper smoke only; no optimizer step or official epoch launched.",
                "The wrapper generates official `data/commercial` files from SARCheck R110 windows.",
                "This clears SESRec for bounded GPU wrapper execution, but a full official training result still must retrieve logs/checkpoints/readouts.",
            ],
        }
    except Exception as exc:
        return {
            "run_id": "R130T",
            "method": "SESRec",
            "status": "blocked",
            "official_repo": str(repo),
            "official_commit": get_git_commit(repo) if repo.exists() else None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    finally:
        restore_imports(old_modules, reset_names)
        os.chdir(old_cwd)
        sys.path[:] = old_path


def get_git_commit(repo: Path) -> str | None:
    head = repo / ".git/HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = repo / ".git" / text.split(" ", 1)[1]
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    return text


def file_contains(path: Path, needles: list[str]) -> dict[str, bool]:
    if not path.exists():
        return {n: False for n in needles}
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {n: n in text for n in needles}


def adapter_feasibility(
    run_id: str,
    method: str,
    repo: Path,
    schema_file: Path,
    required_schema_terms: list[str],
    examples: list[dict[str, Any]],
    decision: str,
    rationale: list[str],
    limits: list[str],
) -> dict[str, Any]:
    example_keys = set()
    for ex in examples[:32]:
        example_keys.update(ex.keys())
    required_examples = [
        "user_id",
        "H_pre",
        "E_search_all",
        "E_search_pos",
        "query_tokens",
        "primary_target_item",
        "Y_post_rec",
    ]
    schema_hits = file_contains(schema_file, required_schema_terms)
    example_hits = {k: k in example_keys for k in required_examples}
    status = "pass" if all(schema_hits.values()) and all(example_hits.values()) else "blocked"
    return {
        "run_id": run_id,
        "method": method,
        "status": status,
        "official_repo": str(repo),
        "official_commit": get_git_commit(repo),
        "schema_file": str(schema_file),
        "schema_terms": schema_hits,
        "sarcheck_example_fields": example_hits,
        "decision": decision if status == "pass" else "needs_fix",
        "rationale": rationale,
        "limits": limits,
    }


def write_r128_report(payload: dict[str, Any]) -> None:
    status = payload["status"]
    checks = payload.get("checks", {})
    bundle = payload.get("bundle", {})
    lines = [
        "# R128 UniSAR Official-Code Local Smoke",
        "",
        f"Status: {status}. No training was launched and no GPU was used.",
        "",
        "## What This Gate Checks",
        "",
        "R128 checks whether the official UniSAR repository can accept a tiny KuaiSAR-format bundle converted from the fixed SARCheck R110 windows, instantiate the official dataset/model path, and export recommendation scores for the SARCheck readout views.",
        "",
        "## Result",
        "",
    ]
    if status == "pass":
        lines.extend(
            [
                "- Official import: pass",
                "- Official `RecDataSet` construction: pass",
                "- Official `UniSAR` model construction: pass",
                "- Official `rec_predict` forward pass: pass",
                f"- Exported views: {', '.join(checks.get('views_exported', []))}",
                f"- All required-view scores finite: {checks.get('all_required_scores_finite')}",
                "",
                "## Tiny Bundle",
                "",
                f"- Rows: {bundle.get('rows')}",
                f"- Negatives per row: {bundle.get('negatives_per_row')}",
                f"- Compact users/items/tokens/sessions: {bundle.get('compact_users')} / {bundle.get('compact_items')} / {bundle.get('compact_tokens')} / {bundle.get('compact_sessions')}",
                f"- Neutral item id for masked views: {bundle.get('neutral_item_id')}",
                f"- Neutral source-session id for masked views: {bundle.get('neutral_src_session_id')}",
                f"- Split rows: {bundle.get('split_rows')}",
                f"- Data root: `{bundle.get('data_root')}`",
                "",
                "## Interpretation",
                "",
                "UniSAR is now an actionable official-code candidate for SARCheck hardening. The next step is a real official-code training/evaluation run on a larger converted KuaiSAR slice or the official processed KuaiSAR bundle, followed by the same five-view readout.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Blocker: `{payload.get('error_type', 'n/a')}`",
                f"- Detail: {payload.get('error') or payload.get('reason')}",
                "",
                "## Interpretation",
                "",
                "R128 did not clear the local official-code gate. Do not launch a UniSAR GPU run until this blocker is fixed.",
            ]
        )
    if payload.get("limits"):
        lines.extend(["", "## Limits", ""])
        lines.extend(f"- {x}" for x in payload["limits"])
    write_md(REPORTS / "r128_unisar_official_smoke.md", "\n".join(lines))


def write_feasibility_report(payload: dict[str, Any], filename: str) -> None:
    lines = [
        f"# {payload['run_id']} {payload['method']} Official-Adapter Feasibility",
        "",
        f"Status: {payload['status']}. No training was launched and no GPU was used.",
        "",
        "## Decision",
        "",
        payload["decision"],
        "",
        "## Schema Checks",
        "",
    ]
    for term, ok in payload["schema_terms"].items():
        lines.append(f"- `{term}`: {'yes' if ok else 'no'}")
    lines.extend(["", "## SARCheck Field Checks", ""])
    for term, ok in payload["sarcheck_example_fields"].items():
        lines.append(f"- `{term}`: {'yes' if ok else 'no'}")
    lines.extend(["", "## Rationale", ""])
    lines.extend(f"- {x}" for x in payload["rationale"])
    lines.extend(["", "## Limits", ""])
    lines.extend(f"- {x}" for x in payload["limits"])
    write_md(REPORTS / filename, "\n".join(lines))


def write_official_smoke_report(payload: dict[str, Any], filename: str) -> None:
    checks = payload.get("checks", {})
    bundle = payload.get("bundle", {})
    method = payload.get("method", payload.get("run_id"))
    status = payload["status"]
    lines = [
        f"# {payload['run_id']} {method} Official-Code Local Smoke",
        "",
        f"Status: {status}. No training was launched and no GPU was used.",
        "",
        "## What This Gate Checks",
        "",
        f"This gate checks whether the official {method} code path can accept SARCheck-derived tiny inputs and export finite scores for the audit readout views.",
        "",
        "## Result",
        "",
    ]
    if status == "pass":
        lines.extend(
            [
                "- Official import: pass",
                f"- Official dataset construction: {checks.get('official_dataset_instantiated', 'n/a')}",
                f"- Official model construction: {checks.get('official_model_instantiated', 'n/a')}",
                f"- Official forward/predict pass: {checks.get('official_forward_pass', checks.get('official_predict_forward', 'n/a'))}",
                f"- Exported views: {', '.join(checks.get('views_exported', []))}",
                "",
                "## Tiny Bundle",
                "",
                f"- Rows: {bundle.get('rows')}",
                f"- Scored rows: {bundle.get('scored_rows', bundle.get('rows'))}",
                f"- Compact users/items/tokens/sessions: {bundle.get('compact_users')} / {bundle.get('compact_items')} / {bundle.get('compact_tokens')} / {bundle.get('compact_sessions')}",
                f"- Work/data root: `{bundle.get('data_root', bundle.get('work_dir'))}`",
                "",
                "## Interpretation",
                "",
                f"{method} is now an actionable official-code candidate for the combined GPU batch. The next step is larger-slice training/evaluation plus the same five-view transfer-validity readout.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Blocker: `{payload.get('error_type', 'n/a')}`",
                f"- Detail: {payload.get('error') or payload.get('reason')}",
                "",
                "## Interpretation",
                "",
                f"{method} did not clear the local official-code gate. Do not include it in the GPU batch until this blocker is fixed.",
            ]
        )
    if payload.get("limits"):
        lines.extend(["", "## Limits", ""])
        lines.extend(f"- {x}" for x in payload["limits"])
    write_md(REPORTS / filename, "\n".join(lines))


def write_sesrec_trainer_wrapper_report(payload: dict[str, Any]) -> None:
    status = payload["status"]
    bundle = payload.get("bundle", {})
    checks = payload.get("checks", {})
    lines = [
        "# R130T SESRec Official Trainer-Wrapper Smoke",
        "",
        f"Status: {status}. No GPU was used.",
        "",
        "## What This Gate Checks",
        "",
        "R130T checks whether SARCheck can generate the official SESRec `data/commercial` directory, instantiate the official SESRec datasets/recommender, run one official train-loss forward pass, and run one official prediction batch.",
        "",
        "## Result",
        "",
    ]
    if status == "pass":
        losses = payload.get("loss_values", {})
        pred = payload.get("predict_scores", {})
        lines.extend(
            [
                "- Official dataset construction: pass",
                "- Official recommender construction: pass",
                "- Official train-loss forward: pass",
                "- Official predict forward: pass",
                f"- Train / validation rows: {checks.get('train_dataset_rows')} / {checks.get('valid_dataset_rows')}",
                f"- Losses: BCE={losses.get('bce'):.6f}, feature-align={losses.get('feature_align'):.6f}, interest-contrast={losses.get('interest_contrast'):.6f}",
                f"- Predict shape / finite: {pred.get('shape')} / {pred.get('finite')}",
                "",
                "## Generated Official Data",
                "",
                f"- Data root: `{bundle.get('data_root')}`",
                f"- Source windows: {bundle.get('rows')}",
                f"- Split rows: {bundle.get('split_rows')}",
                f"- Evaluation negatives per positive: {bundle.get('negatives_per_eval_positive')}",
                "",
                "## Interpretation",
                "",
                "SESRec no longer has to remain smoke-only by default. It has a prepared official trainer wrapper and can be included in the bounded official SAR batch after a remote local recheck.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Blocker: `{payload.get('error_type', 'n/a')}`",
                f"- Detail: {payload.get('error') or payload.get('reason')}",
                "",
                "## Interpretation",
                "",
                "SESRec should remain official-code smoke-only until this trainer-wrapper blocker is fixed.",
            ]
        )
    if payload.get("limits"):
        lines.extend(["", "## Limits", ""])
        lines.extend(f"- {x}" for x in payload["limits"])
    write_md(REPORTS / "r130t_sesrec_trainer_wrapper_smoke.md", "\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--unifiedssr-work-dir", type=Path, default=DEFAULT_UNIFIEDSSR_WORK)
    parser.add_argument("--sesrec-work-dir", type=Path, default=DEFAULT_SESREC_WORK)
    parser.add_argument("--examples", type=Path, default=EXAMPLES)
    parser.add_argument("--row-limit", type=int, default=64)
    parser.add_argument("--neg-k", type=int, default=31)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    examples = read_jsonl(args.examples, limit=max(args.row_limit, 128))

    r128 = run_unisar_official_smoke(
        args.official_root,
        args.work_dir,
        examples,
        args.row_limit,
        args.neg_k,
        args.seed,
    )
    write_json(REPORTS / "r128_unisar_official_smoke.json", r128)
    write_r128_report(r128)

    r129 = run_unifiedssr_official_smoke(
        args.official_root,
        args.unifiedssr_work_dir,
        examples=examples,
        row_limit=args.row_limit,
        seed=args.seed,
    )
    write_json(REPORTS / "r129_unifiedssr_official_smoke.json", r129)
    write_official_smoke_report(r129, "r129_unifiedssr_official_smoke.md")

    r130 = run_sesrec_official_smoke(
        args.official_root,
        args.sesrec_work_dir,
        examples=examples,
        row_limit=args.row_limit,
        seed=args.seed,
    )
    write_json(REPORTS / "r130_sesrec_official_smoke.json", r130)
    write_official_smoke_report(r130, "r130_sesrec_official_smoke.md")

    r130t = run_sesrec_trainer_wrapper_smoke(
        args.official_root,
        examples=examples,
        row_limit=args.row_limit,
        neg_k=99,
        seed=args.seed,
    )
    write_json(REPORTS / "r130t_sesrec_trainer_wrapper_smoke.json", r130t)
    write_sesrec_trainer_wrapper_report(r130t)

    summary = {
        "runs": {
            "R128": r128["status"],
            "R129": r129["status"],
            "R130": r130["status"],
            "R130T": r130t["status"],
        },
        "gpu_needed_now": False,
        "next_gpu_candidate": "Combined official-code SAR batch for UniSAR, UnifiedSSR, and SESRec after local smoke/code review.",
    }
    write_json(REPORTS / "official_adapter_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(v == "pass" for v in summary["runs"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
