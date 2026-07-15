#!/usr/bin/env python3
"""R114P-R115P transition-aware SAR paper-method approximation.

This script is intentionally local-first. It trains the first Tier-A
paper-method-family approximation selected in R113P: a UniSAR/USER-style
behavior-transition backbone with typed search/recommendation events and an
explicit S->R transition branch.

It is not a faithful UniSAR or USER reproduction. Its role is to test whether a
mechanism-aligned local backbone can produce both recommendation ranking signal
and focal-search transfer validity under masked/L1/L2 controls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
RUNS = ROOT / "runs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--examples-path", default=str(DATA / "backbone_audit" / "examples.jsonl"))
    p.add_argument("--item-metadata", default=str(DATA / "item_metadata.parquet"))
    p.add_argument("--output-dir", default=str(RUNS / "transition_backbone"))
    p.add_argument("--reports-dir", default=str(REPORTS))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-negatives", type=int, default=96)
    p.add_argument("--eval-negatives", type=int, default=100)
    p.add_argument("--max-pre", type=int, default=24)
    p.add_argument("--max-search", type=int, default=24)
    p.add_argument("--max-query", type=int, default=12)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=20260610)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--swap-tries", type=int, default=60)
    p.add_argument("--transition-loss-weight", type=float, default=0.4)
    p.add_argument("--passive-loss-weight", type=float, default=0.2)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def frac(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def uniq(xs: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in xs:
        y = int(x)
        if y in seen:
            continue
        seen.add(y)
        out.append(y)
    return out


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [r for r in rows if str(r["split"]) == split and r["Y_post_rec"] and r["primary_target_item"] is not None]


def build_item_vocab(rows: list[dict[str, Any]]) -> dict[int, int]:
    items: set[int] = set()
    for row in rows:
        items.update(int(x) for x in row["H_pre"])
        items.update(int(x) for x in row["E_search_all"])
        items.update(int(x) for x in row["Y_post_rec"])
    return {item: idx + 1 for idx, item in enumerate(sorted(items))}


def build_query_vocab(rows: list[dict[str, Any]]) -> dict[int, int]:
    toks: set[int] = set()
    for row in rows:
        toks.update(int(x) for x in row["query_tokens"])
    return {tok: idx + 1 for idx, tok in enumerate(sorted(toks))}


def build_source_vocab(rows: list[dict[str, Any]]) -> dict[str, int]:
    vals = sorted({str(r["source"]) for r in rows})
    return {v: i for i, v in enumerate(vals)}


class SARDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        item_vocab: dict[int, int],
        query_vocab: dict[int, int],
        source_vocab: dict[str, int],
        args: argparse.Namespace,
    ) -> None:
        self.rows = rows
        self.item_vocab = item_vocab
        self.query_vocab = query_vocab
        self.source_vocab = source_vocab
        self.args = args

    def __len__(self) -> int:
        return len(self.rows)

    def pad_items(self, xs: list[int], max_len: int) -> tuple[list[int], int]:
        ids = [self.item_vocab[int(x)] for x in xs if int(x) in self.item_vocab][-max_len:]
        return [0] * (max_len - len(ids)) + ids, len(ids)

    def pad_query(self, xs: list[int]) -> tuple[list[int], int]:
        ids = [self.query_vocab[int(x)] for x in xs if int(x) in self.query_vocab][-self.args.max_query :]
        return [0] * (self.args.max_query - len(ids)) + ids, len(ids)

    def encode_row(
        self,
        row: dict[str, Any],
        search_override: list[int] | None = None,
        query_override: list[int] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        search_raw = row["E_search_all"] if search_override is None else search_override
        query_raw = row["query_tokens"] if query_override is None else query_override
        pre, pre_len = self.pad_items([int(x) for x in row["H_pre"]], self.args.max_pre)
        search, search_len = self.pad_items([int(x) for x in search_raw], self.args.max_search)
        query, query_len = self.pad_query([int(x) for x in query_raw])
        source = self.source_vocab[str(row["source"])]
        target = self.item_vocab[int(row["primary_target_item"])]
        return (
            torch.tensor(pre, dtype=torch.long),
            torch.tensor(pre_len, dtype=torch.long),
            torch.tensor(search, dtype=torch.long),
            torch.tensor(search_len, dtype=torch.long),
            torch.tensor(query, dtype=torch.long),
            torch.tensor(query_len, dtype=torch.long),
            torch.tensor(source, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return self.encode_row(self.rows[idx]) + (torch.tensor(idx, dtype=torch.long),)


class TransitionSARBackbone(nn.Module):
    def __init__(self, num_items: int, num_query_tokens: int, num_sources: int, dim: int, hidden: int, max_events: int) -> None:
        super().__init__()
        self.dim = dim
        self.item_emb = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.query_emb = nn.Embedding(num_query_tokens + 1, dim, padding_idx=0)
        self.source_emb = nn.Embedding(max(num_sources, 1), dim)
        self.type_emb = nn.Embedding(3, dim)
        self.pos_emb = nn.Embedding(max_events, dim)
        self.encoder = nn.GRU(dim, dim, batch_first=True)
        self.passive_tower = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.search_tower = nn.Sequential(nn.Linear(dim * 3, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.transition_tower = nn.Sequential(nn.Linear(dim * 4, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim))
        self.gate = nn.Sequential(nn.Linear(dim * 4, hidden), nn.ReLU(), nn.Linear(hidden, dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(dim * 4, hidden), nn.ReLU(), nn.LayerNorm(hidden), nn.Linear(hidden, dim))

    @staticmethod
    def mean_pool(emb: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        mask = ids.ne(0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (emb * mask).sum(dim=1) / denom

    @staticmethod
    def masked_mean(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        return (encoded * m).sum(dim=1) / denom

    def typed_item_events(self, ids: torch.Tensor, event_type: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, width = ids.shape
        pos = torch.arange(offset, offset + width, device=ids.device).unsqueeze(0).expand(bsz, width)
        typ = torch.full_like(ids, int(event_type))
        mask = ids.ne(0)
        emb = self.item_emb(ids) + self.type_emb(typ) + self.pos_emb(pos)
        return emb * mask.float().unsqueeze(-1), mask

    def typed_query_events(self, ids: torch.Tensor, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, width = ids.shape
        pos = torch.arange(offset, offset + width, device=ids.device).unsqueeze(0).expand(bsz, width)
        typ = torch.full_like(ids, 2)
        mask = ids.ne(0)
        emb = self.query_emb(ids) + self.type_emb(typ) + self.pos_emb(pos)
        return emb * mask.float().unsqueeze(-1), mask

    def encode_parts(self, pre: torch.Tensor, search: torch.Tensor, query: torch.Tensor, source: torch.Tensor) -> dict[str, torch.Tensor]:
        pre_events, pre_mask = self.typed_item_events(pre, 0, 0)
        search_events, search_mask = self.typed_item_events(search, 1, pre.shape[1])
        query_events, query_mask = self.typed_query_events(query, pre.shape[1] + search.shape[1])
        seq = torch.cat([pre_events, search_events, query_events], dim=1)
        mask = torch.cat([pre_mask, search_mask, query_mask], dim=1)
        encoded, _ = self.encoder(seq)

        pre_ctx = self.masked_mean(encoded[:, : pre.shape[1], :], pre_mask)
        search_ctx = self.masked_mean(encoded[:, pre.shape[1] : pre.shape[1] + search.shape[1], :], search_mask)
        query_ctx = self.masked_mean(encoded[:, pre.shape[1] + search.shape[1] :, :], query_mask)
        seq_ctx = self.masked_mean(encoded, mask)
        source_vec = self.source_emb(source)
        passive = self.passive_tower(pre_ctx)
        search_private = self.search_tower(torch.cat([search_ctx, query_ctx, source_vec], dim=1))
        transition = self.transition_tower(torch.cat([pre_ctx, search_ctx, query_ctx, source_vec], dim=1))
        gate = self.gate(torch.cat([passive, transition, seq_ctx, source_vec], dim=1))
        fused = self.fusion(torch.cat([passive, gate * transition, seq_ctx, source_vec], dim=1))
        return {"fused": fused, "passive": passive, "search": transition, "search_private": search_private, "gate_mean": gate.mean(dim=1)}

    def score(self, user_vec: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        cand = self.item_emb(candidates)
        return torch.einsum("bd,bkd->bk", user_vec, cand)


def sample_negatives(targets: torch.Tensor, num_items: int, k: int, device: torch.device) -> torch.Tensor:
    neg = torch.randint(1, num_items + 1, (targets.shape[0], k), device=device)
    clash = neg.eq(targets.unsqueeze(1))
    if clash.any():
        neg[clash] = torch.randint(1, num_items + 1, (int(clash.sum().item()),), device=device)
    return neg


def stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_candidates(
    row: dict[str, Any],
    item_vocab: dict[int, int],
    global_pop: list[int],
    args: argparse.Namespace,
    extras: list[int] | None = None,
) -> list[int]:
    target = item_vocab[int(row["primary_target_item"])]
    base = [target]
    for raw in row["H_pre"] + row["E_search_all"] + row["Y_post_rec"] + (extras or []):
        raw_i = int(raw)
        if raw_i in item_vocab:
            base.append(item_vocab[raw_i])
    for raw_i in global_pop[:50]:
        if raw_i in item_vocab:
            base.append(item_vocab[raw_i])
    base = uniq(base)
    seen = set(base)
    rng = random.Random((args.seed * 1000003) ^ stable_int(str(row["example_id"])))
    while len(base) < args.eval_negatives + 1 and len(seen) < len(item_vocab):
        cand = rng.randint(1, len(item_vocab))
        if cand in seen:
            continue
        seen.add(cand)
        base.append(cand)
    return base[: args.eval_negatives + 1]


def rank_metrics(scores: torch.Tensor, candidates: torch.Tensor, target: int) -> tuple[float, float, int]:
    scored = list(zip(candidates.tolist(), scores.tolist()))
    scored.sort(key=lambda x: (-x[1], x[0]))
    for idx, (item, _) in enumerate(scored, start=1):
        if int(item) == int(target):
            return 1.0 / idx, float(idx <= 10), idx
    return 0.0, 0.0, len(scored) + 1


def train_model(
    model: TransitionSARBackbone,
    train_ds: SARDataset,
    args: argparse.Namespace,
    device: torch.device,
    num_items: int,
) -> list[float]:
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(args.epochs):
        epoch_losses: list[float] = []
        for batch in loader:
            pre, _, search, _, query, _, source, target, _ = [x.to(device) for x in batch]
            parts = model.encode_parts(pre, search, query, source)
            neg = sample_negatives(target, num_items, args.num_negatives, device)
            candidates = torch.cat([target.unsqueeze(1), neg], dim=1)
            labels = torch.zeros(target.shape[0], dtype=torch.long, device=device)
            fused_logits = model.score(parts["fused"], candidates)
            passive_logits = model.score(parts["passive"], candidates)
            search_logits = model.score(parts["search"], candidates)
            loss = (
                loss_fn(fused_logits, labels)
                + args.passive_loss_weight * loss_fn(passive_logits, labels)
                + args.transition_loss_weight * loss_fn(search_logits, labels)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.item()))
        losses.append(avg(epoch_losses))
    return losses


def load_item_categories(path: Path) -> dict[int, int]:
    df = pd.read_parquet(path, columns=["item_id", "first_level_category_id"]).dropna()
    return dict(zip(df["item_id"].astype(int), df["first_level_category_id"].astype(int)))


def cats(items: list[int], item_cat: dict[int, int]) -> set[int]:
    return {int(item_cat[int(x)]) for x in items if int(x) in item_cat}


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


def target_bucket(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 3:
        return "2-3"
    return "4+"


def build_swap_profiles(rows: list[dict[str, Any]], item_cat: dict[int, int]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        search_pos = [int(x) for x in row["E_search_pos"]]
        search_neg = [int(x) for x in row["E_search_neg"]]
        search_all = [int(x) for x in row["E_search_all"]]
        scats = cats(search_all, item_cat)
        profiles.append(
            {
                "idx": i,
                "example_id": row["example_id"],
                "split": row["split"],
                "source": row["source"],
                "search_pos_bucket": qbucket(len(search_pos)),
                "search_neg_bucket": qbucket(len(search_neg)),
                "pre_bucket": pre_bucket(len(row["H_pre"])),
                "target_bucket": target_bucket(len(row["Y_post_rec"])),
                "search_cats": scats,
            }
        )
    return profiles


def build_swap_indices(profiles: list[dict[str, Any]]) -> dict[str, dict[tuple[Any, ...], list[int]]]:
    out: dict[str, dict[tuple[Any, ...], list[int]]] = {"L1": defaultdict(list), "L2": defaultdict(list)}
    for p in profiles:
        out["L1"][(p["split"], p["source"], p["search_pos_bucket"], p["search_neg_bucket"], p["pre_bucket"], p["target_bucket"])].append(p["idx"])
        for c in p["search_cats"]:
            out["L2"][(p["split"], c)].append(p["idx"])
    return out


def pick_swap(
    i: int,
    level: str,
    profiles: list[dict[str, Any]],
    indices: dict[str, dict[tuple[Any, ...], list[int]]],
    rng: random.Random,
    tries: int,
) -> int | None:
    p = profiles[i]
    if level == "L1":
        keys = [(p["split"], p["source"], p["search_pos_bucket"], p["search_neg_bucket"], p["pre_bucket"], p["target_bucket"])]
    elif level == "L2":
        keys = [(p["split"], c) for c in p["search_cats"]]
    else:
        raise ValueError(level)
    pool: list[int] = []
    for key in keys:
        pool.extend(indices[level].get(key, []))
    pool = [j for j in sorted(set(pool)) if j != i]
    if not pool:
        return None
    for _ in range(max(1, tries)):
        j = rng.choice(pool)
        if j != i:
            return j
    return None


@torch.no_grad()
def score_variant(
    model: TransitionSARBackbone,
    dataset: SARDataset,
    row: dict[str, Any],
    candidates: list[int],
    device: torch.device,
    name: str,
    search_override: list[int] | None = None,
    query_override: list[int] | None = None,
) -> tuple[float, float, int, float]:
    encoded = dataset.encode_row(row, search_override=search_override, query_override=query_override)
    pre, _, search, _, query, _, source, target = [x.unsqueeze(0).to(device) for x in encoded]
    parts = model.encode_parts(pre, search, query, source)
    candidates_t = torch.tensor(candidates, dtype=torch.long, device=device).unsqueeze(0)
    target_id = int(target.item())
    vec_name = "fused" if name.startswith("fused") else name
    scores = model.score(parts[vec_name], candidates_t)[0].detach().cpu()
    rr, hit10, rank = rank_metrics(scores, candidates_t[0].detach().cpu(), target_id)
    return rr, hit10, rank, float(parts["gate_mean"].detach().cpu().item())


def eval_rows(
    model: TransitionSARBackbone,
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    dataset: SARDataset,
    item_vocab: dict[int, int],
    global_pop: list[int],
    args: argparse.Namespace,
    device: torch.device,
    profiles: list[dict[str, Any]],
    swap_indices: dict[str, dict[tuple[Any, ...], list[int]]],
) -> list[dict[str, Any]]:
    rng = random.Random(args.seed)
    model.eval()
    detail: list[dict[str, Any]] = []
    profile_by_id = {p["example_id"]: p for p in profiles}
    row_by_idx = {i: r for i, r in enumerate(all_rows)}
    for local_idx, row in enumerate(rows):
        profile_idx = profile_by_id[row["example_id"]]["idx"]
        l1_idx = pick_swap(profile_idx, "L1", profiles, swap_indices, rng, args.swap_tries)
        l2_idx = pick_swap(profile_idx, "L2", profiles, swap_indices, rng, args.swap_tries)
        extras: list[int] = []
        if l1_idx is not None:
            extras.extend(row_by_idx[l1_idx]["E_search_all"])
        if l2_idx is not None:
            extras.extend(row_by_idx[l2_idx]["E_search_all"])
        cand = deterministic_candidates(row, item_vocab, global_pop, args, extras=extras)

        fused_rr, fused_hit10, fused_rank, gate_mean = score_variant(model, dataset, row, cand, device, "fused")
        passive_rr, passive_hit10, passive_rank, _ = score_variant(model, dataset, row, cand, device, "passive")
        search_rr, search_hit10, search_rank, _ = score_variant(model, dataset, row, cand, device, "search")
        masked_rr, masked_hit10, masked_rank, masked_gate = score_variant(
            model, dataset, row, cand, device, "fused_masked", search_override=[], query_override=[]
        )
        out = {
            "example_id": row["example_id"],
            "split": row["split"],
            "source": row["source"],
            "search_pos_count": len(row["E_search_pos"]),
            "search_action_count": len(row["E_search_all"]),
            "pre_len": len(row["H_pre"]),
            "cat_conflict": int(row["audit"].get("cat_conflict", 0) or 0),
            "audit_transferable": int(bool(row["audit"].get("transferable_by_outcome"))),
            "audit_nontransfer": int(bool(row["audit"].get("nontransfer_by_outcome"))),
            "fused_rr": fused_rr,
            "passive_rr": passive_rr,
            "search_rr": search_rr,
            "masked_rr": masked_rr,
            "fused_hit10": fused_hit10,
            "passive_hit10": passive_hit10,
            "search_hit10": search_hit10,
            "masked_hit10": masked_hit10,
            "fused_rank": fused_rank,
            "passive_rank": passive_rank,
            "search_rank": search_rank,
            "masked_rank": masked_rank,
            "gate_mean": gate_mean,
            "masked_gate_mean": masked_gate,
            "l1_matched": int(l1_idx is not None),
            "l2_matched": int(l2_idx is not None),
            "l1_swap_rr": 0.0,
            "l2_swap_rr": 0.0,
            "l1_true_minus_swap": 0.0,
            "l2_true_minus_swap": 0.0,
            "l1_true_win": 0,
            "l2_true_win": 0,
        }
        if l1_idx is not None:
            swap = row_by_idx[l1_idx]
            l1_rr, _, _, _ = score_variant(
                model, dataset, row, cand, device, "fused_l1", search_override=swap["E_search_all"], query_override=swap["query_tokens"]
            )
            out["l1_swap_rr"] = l1_rr
            out["l1_true_minus_swap"] = fused_rr - l1_rr
            out["l1_true_win"] = int(fused_rr > l1_rr)
        if l2_idx is not None:
            swap = row_by_idx[l2_idx]
            l2_rr, _, _, _ = score_variant(
                model, dataset, row, cand, device, "fused_l2", search_override=swap["E_search_all"], query_override=swap["query_tokens"]
            )
            out["l2_swap_rr"] = l2_rr
            out["l2_true_minus_swap"] = fused_rr - l2_rr
            out["l2_true_win"] = int(fused_rr > l2_rr)
        detail.append(out)
    return detail


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l1 = [r for r in rows if int(r["l1_matched"])]
    l2 = [r for r in rows if int(r["l2_matched"])]
    return {
        "rows": len(rows),
        "fused_mrr": avg([float(r["fused_rr"]) for r in rows]),
        "passive_mrr": avg([float(r["passive_rr"]) for r in rows]),
        "search_mrr": avg([float(r["search_rr"]) for r in rows]),
        "masked_mrr": avg([float(r["masked_rr"]) for r in rows]),
        "fused_hit10": avg([float(r["fused_hit10"]) for r in rows]),
        "passive_hit10": avg([float(r["passive_hit10"]) for r in rows]),
        "search_hit10": avg([float(r["search_hit10"]) for r in rows]),
        "masked_hit10": avg([float(r["masked_hit10"]) for r in rows]),
        "fused_minus_passive_mrr": avg([float(r["fused_rr"]) - float(r["passive_rr"]) for r in rows]),
        "fused_minus_search_mrr": avg([float(r["fused_rr"]) - float(r["search_rr"]) for r in rows]),
        "fused_minus_masked_mrr": avg([float(r["fused_rr"]) - float(r["masked_rr"]) for r in rows]),
        "gate_mean": avg([float(r["gate_mean"]) for r in rows]),
        "l1_matched": len(l1),
        "l1_true_win_rate": frac(sum(int(r["l1_true_win"]) for r in l1), len(l1)),
        "l1_true_minus_swap_mrr": avg([float(r["l1_true_minus_swap"]) for r in l1]),
        "l2_matched": len(l2),
        "l2_true_win_rate": frac(sum(int(r["l2_true_win"]) for r in l2), len(l2)),
        "l2_true_minus_swap_mrr": avg([float(r["l2_true_minus_swap"]) for r in l2]),
    }


def bucket_value(row: dict[str, Any], bucket: str) -> str:
    if bucket == "all":
        return "all"
    if bucket == "split":
        return str(row["split"])
    if bucket == "source":
        return str(row["source"])
    if bucket == "search_pos":
        return qbucket(int(row["search_pos_count"]))
    if bucket == "search_actions":
        return qbucket(int(row["search_action_count"]))
    if bucket == "pre_len":
        return pre_bucket(int(row["pre_len"]))
    if bucket == "cat_conflict":
        return "conflict" if int(row["cat_conflict"]) else "overlap_or_missing"
    if bucket == "passive_strength":
        return "passive_strong" if float(row["passive_rr"]) >= 0.05 else "passive_weak"
    if bucket == "audit_transfer":
        if int(row["audit_transferable"]):
            return "transferable"
        if int(row["audit_nontransfer"]):
            return "nontransfer"
        return "neutral"
    raise ValueError(bucket)


def bucketed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bucket in ["all", "split", "source", "search_pos", "search_actions", "pre_len", "cat_conflict", "passive_strength", "audit_transfer"]:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[bucket_value(row, bucket)].append(row)
        for value, group in sorted(groups.items()):
            if len(group) < 20 and bucket not in {"all", "split"}:
                continue
            agg = aggregate(group)
            out.append({"bucket": bucket, "value": value, **agg})
    return out


def metrics_table_line(name: str, m: dict[str, Any]) -> str:
    return (
        f"| {name} | {m['fused_mrr']:.4f} | {m['passive_mrr']:.4f} | {m['search_mrr']:.4f} | "
        f"{m['masked_mrr']:.4f} | {m['fused_hit10']:.4f} |"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    reports_dir = Path(args.reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = (Path.cwd() / reports_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_jsonl(Path(args.examples_path))
    train_rows = split_rows(all_rows, "train")
    val_rows = split_rows(all_rows, "validation")
    test_rows = split_rows(all_rows, "test")
    model_rows = train_rows + val_rows + test_rows
    item_vocab = build_item_vocab(model_rows)
    query_vocab = build_query_vocab(model_rows)
    source_vocab = build_source_vocab(model_rows)
    global_pop = [item for item, _ in Counter(t for row in train_rows for t in row["Y_post_rec"]).most_common()]
    train_ds = SARDataset(train_rows, item_vocab, query_vocab, source_vocab, args)
    val_ds = SARDataset(val_rows, item_vocab, query_vocab, source_vocab, args)
    test_ds = SARDataset(test_rows, item_vocab, query_vocab, source_vocab, args)
    all_ds = SARDataset(all_rows, item_vocab, query_vocab, source_vocab, args)

    model = TransitionSARBackbone(
        len(item_vocab),
        len(query_vocab),
        len(source_vocab),
        args.embedding_dim,
        args.hidden_dim,
        args.max_pre + args.max_search + args.max_query,
    ).to(device)
    start = time.time()
    losses = train_model(model, train_ds, args, device, len(item_vocab))
    train_seconds = time.time() - start

    item_cat = load_item_categories(Path(args.item_metadata))
    profiles = build_swap_profiles(all_rows, item_cat)
    swap_indices = build_swap_indices(profiles)
    val_detail = eval_rows(model, val_rows, all_rows, val_ds, item_vocab, global_pop, args, device, profiles, swap_indices)
    test_detail = eval_rows(model, test_rows, all_rows, test_ds, item_vocab, global_pop, args, device, profiles, swap_indices)
    detail_rows = val_detail + test_detail
    val_metrics = aggregate(val_detail)
    test_metrics = aggregate(test_detail)
    all_metrics = aggregate(detail_rows)

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "item_vocab": item_vocab,
            "query_vocab": query_vocab,
            "source_vocab": source_vocab,
            "model_type": "transition_aware_typed_gru",
            "args": vars(args),
        },
        checkpoint_path,
    )

    r114p = {
        "schema": "searchcpt.r114p.transition_backbone.v1",
        "device": str(device),
        "model_type": "transition_aware_typed_gru",
        "method_family": "UniSAR/USER-style transition approximation",
        "epochs": args.epochs,
        "train_seconds": train_seconds,
        "train_loss": losses[-1] if losses else 0.0,
        "losses": losses,
        "train_rows": len(train_rows),
        "validation": val_metrics,
        "test": test_metrics,
        "num_items": len(item_vocab),
        "num_query_tokens": len(query_vocab),
        "num_sources": len(source_vocab),
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    r115p = {
        "schema": "searchcpt.r115p.transfer_validity_audit.v1",
        "validation": val_metrics,
        "test": test_metrics,
        "eval_all": all_metrics,
        "status": "pass_signal" if test_metrics["fused_mrr"] >= max(test_metrics["passive_mrr"], test_metrics["search_mrr"]) else "no_fused_ranking_gain",
    }
    bucket_rows = bucketed(detail_rows)
    r115p_buckets = {
        "schema": "searchcpt.r115p.bucketed_validity.v1",
        "num_bucket_rows": len(bucket_rows),
        "top_positive_true_minus_l1": sorted(
            [r for r in bucket_rows if r["l1_matched"] >= 20],
            key=lambda r: (float(r["l1_true_minus_swap_mrr"]), int(r["rows"])),
            reverse=True,
        )[:10],
        "top_negative_true_minus_l1": sorted(
            [r for r in bucket_rows if r["l1_matched"] >= 20],
            key=lambda r: (float(r["l1_true_minus_swap_mrr"]), -int(r["rows"])),
        )[:10],
    }

    write_json(out_dir / "metrics.json", r114p)
    write_json(out_dir / "r115p_transfer_validity_audit.json", r115p)
    write_json(out_dir / "r115p_bucketed_validity_analysis.json", r115p_buckets)
    write_csv(out_dir / "r115p_transfer_validity_scores.csv", detail_rows)
    write_csv(out_dir / "r115p_bucketed_validity_analysis.csv", bucket_rows)
    write_json(reports_dir / "transition_backbone.json", r114p)
    write_json(reports_dir / "r115p_transfer_validity_audit.json", r115p)
    write_json(reports_dir / "r115p_bucketed_validity_analysis.json", r115p_buckets)
    write_csv(reports_dir / "r115p_transfer_validity_scores.csv", detail_rows)
    write_csv(reports_dir / "r115p_bucketed_validity_analysis.csv", bucket_rows)

    r114p_report = [
        "# R114P Transition-Aware SAR Backbone",
        "",
        "R114P is the first Tier-A paper-method-family approximation. It uses typed recommendation/search/query events, a GRU sequence encoder, an explicit S->R transition branch, and the same R110 split/candidate protocol.",
        "",
        "- Method family: UniSAR/USER-style transition approximation",
        "- Faithfulness: local approximation, not a faithful UniSAR or USER reproduction",
        f"- Device: {device}",
        f"- Epochs: {args.epochs}",
        f"- Train seconds: {train_seconds:.2f}",
        f"- Train loss: {r114p['train_loss']:.4f}",
        f"- Rows train/validation/test: {len(train_rows)} / {len(val_rows)} / {len(test_rows)}",
        "",
        "| Split | Fused MRR | Passive MRR | Search MRR | Masked MRR | Fused Hit@10 |",
        "|---|---:|---:|---:|---:|---:|",
        metrics_table_line("validation", val_metrics),
        metrics_table_line("test", test_metrics),
        "",
        "## Interpretation",
        "",
        "R114P passes the export gate if it writes fused/passive/search/masked/L1/L2 scores. It passes the ranking sanity gate only if fused ranking is competitive with passive and search branches.",
    ]
    (reports_dir / "transition_backbone.md").write_text("\n".join(r114p_report) + "\n", encoding="utf-8")

    r115p_report = [
        "# R115P Transfer-Validity Audit",
        "",
        "R115P audits the R114P model under true focal search, masked search, L1 nuisance swaps, and L2 same-category swaps.",
        "",
        "| Split | True-Masked MRR | L1 true-swap MRR | L1 win | L2 true-swap MRR | L2 win |",
        "|---|---:|---:|---:|---:|---:|",
        f"| validation | {val_metrics['fused_minus_masked_mrr']:.4f} | {val_metrics['l1_true_minus_swap_mrr']:.4f} | {val_metrics['l1_true_win_rate']:.4f} | {val_metrics['l2_true_minus_swap_mrr']:.4f} | {val_metrics['l2_true_win_rate']:.4f} |",
        f"| test | {test_metrics['fused_minus_masked_mrr']:.4f} | {test_metrics['l1_true_minus_swap_mrr']:.4f} | {test_metrics['l1_true_win_rate']:.4f} | {test_metrics['l2_true_minus_swap_mrr']:.4f} | {test_metrics['l2_true_win_rate']:.4f} |",
        "",
        "## Interpretation",
        "",
        "Transfer validity requires true search to improve ranking against masked and swapped search. Positive aggregate ranking alone is insufficient.",
    ]
    (reports_dir / "r115p_transfer_validity_audit.md").write_text("\n".join(r115p_report) + "\n", encoding="utf-8")

    r115p_bucket_report = [
        "# R115P Bucketed Transfer-Validity Analysis",
        "",
        "R115P searches for buckets where the R114P model has valid focal-search sensitivity rather than nuisance replay.",
        "",
        "| Bucket | Value | Rows | Fused MRR | True-Masked | L1 true-swap | L1 win | L2 true-swap | L2 win |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows[:40]:
        r115p_bucket_report.append(
            f"| {row['bucket']} | {row['value']} | {int(row['rows'])} | {float(row['fused_mrr']):.4f} | "
            f"{float(row['fused_minus_masked_mrr']):.4f} | {float(row['l1_true_minus_swap_mrr']):.4f} | "
            f"{float(row['l1_true_win_rate']):.4f} | {float(row['l2_true_minus_swap_mrr']):.4f} | {float(row['l2_true_win_rate']):.4f} |"
        )
    r115p_bucket_report.extend(
        [
            "",
            "## Interpretation",
            "",
            "Buckets are useful only if they combine non-trivial row counts with positive true-minus-swap and a meaningful true-win rate. Tiny or tie-dominated buckets are not paper-grade evidence.",
        ]
    )
    (reports_dir / "r115p_bucketed_validity_analysis.md").write_text("\n".join(r115p_bucket_report) + "\n", encoding="utf-8")
    print(json.dumps({"R114P": r114p, "R115P": r115p, "R115P_BUCKETS": r115p_buckets}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
