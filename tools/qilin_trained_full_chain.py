#!/usr/bin/env python3
"""Train and score the R239 Qilin natural-data full-chain calibration.

The split unit is an eligible recommendation request, never an individual
clicked item.  Every controlled view shares the request's original logged
slate.  BGE item/query tensors are fixed; only compact low-rank projections
and a pre-outcome gate are trained, using a multi-positive listwise ranking
loss on logged clicks.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


HORIZON_SECONDS = 24 * 60 * 60
SPLITS = ("train", "validation", "test")
ROOT = Path(__file__).resolve().parents[1]
VIEWS = (
    "full",
    "history_only",
    "search_only",
    "focal_masked",
    "matched_donor_d1",
    "matched_donor_d2",
    "invalid_source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--item-embeddings", type=Path, default=None)
    parser.add_argument("--query-embeddings", type=Path, default=None)
    parser.add_argument("--bge-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260712)
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
    parser.add_argument("--tie-tol", type=float, default=1e-12)
    parser.add_argument("--focal-horizon-seconds", type=int, default=HORIZON_SECONDS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--max-episodes-per-split", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError(f"Expected a rank-2 tensor at {path}, got {type(value)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"Embedding tensor contains non-finite values: {path}")
    return value.float().contiguous()


def read_split(snapshot: Path, split: str) -> pd.DataFrame:
    path = snapshot / split / "train-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def stable_digest(*parts: Any) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str | None:
    if not path.exists() or not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(sha256_file(file_path).encode("ascii"))
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class FocalSession:
    session_id: int
    user_id: int
    timestamp: float
    query_id: int
    clicked_items: tuple[int, ...]


@dataclass
class Episode:
    episode_id: str
    user_id: int
    raw_user_id: int
    request_id: int
    timestamp: float
    slate: tuple[int, ...]
    raw_slate: tuple[int, ...]
    positives: tuple[int, ...]
    history: tuple[int, ...]
    focal: FocalSession
    split: str = ""
    matched_donor_d1: FocalSession | None = None
    matched_donor_d2: FocalSession | None = None
    invalid_source: FocalSession | None = None
    scores: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    gates: dict[str, float] = field(default_factory=dict, repr=False)


def vocab_mappings(
    data_dir: Path,
) -> tuple[dict[int, int], dict[str, int], dict[int, Any]]:
    item_vocab = load_pickle(data_dir / "vocab" / "item_vocab.pkl")
    query_vocab = load_pickle(data_dir / "vocab" / "query_vocab.pkl")
    raw_to_item: dict[int, int] = {}
    for encoded, payload in item_vocab.items():
        encoded_id = int(encoded)
        if encoded_id == 0:
            continue
        raw_to_item[int(payload["item"])] = encoded_id
    query_to_id = {
        str(payload["query"]): int(encoded)
        for encoded, payload in enumerate(query_vocab)
        if int(encoded) != 0
    }
    return raw_to_item, query_to_id, item_vocab


def load_focal_sessions(
    snapshot: Path,
    raw_to_item: dict[int, int],
    query_to_id: dict[str, int],
) -> tuple[list[FocalSession], dict[int, list[FocalSession]]]:
    sessions: list[FocalSession] = []
    seen: dict[tuple[int, int], FocalSession] = {}
    session_owner: dict[int, int] = {}
    for split in ("search_train", "search_test"):
        frame = read_split(snapshot, split)
        for row in frame.itertuples(index=False):
            details = list(row.search_result_details_with_idx)
            if not details:
                continue
            query = str(row.query)
            if query not in query_to_id:
                raise ValueError(f"Query absent from R239 vocabulary: {query!r}")
            clicked = tuple(
                raw_to_item[int(detail["note_idx"])]
                for detail in details
                if int(detail.get("click", 0)) != 0
            )
            session = FocalSession(
                session_id=int(row.search_idx),
                user_id=int(row.user_idx),
                timestamp=max(float(detail["search_timestamp"]) for detail in details),
                query_id=query_to_id[query],
                clicked_items=clicked,
            )
            if session.session_id in session_owner and session_owner[session.session_id] != session.user_id:
                raise ValueError(
                    f"Search session id {session.session_id} belongs to multiple users"
                )
            session_owner[session.session_id] = session.user_id
            key = (session.user_id, session.session_id)
            if key in seen and seen[key] != session:
                raise ValueError(f"Conflicting duplicate search session: {key}")
            seen[key] = session
    sessions.extend(seen.values())
    sessions.sort(key=lambda session: (session.timestamp, session.user_id, session.session_id))
    by_user: dict[int, list[FocalSession]] = defaultdict(list)
    for session in sessions:
        by_user[session.user_id].append(session)
    return sessions, dict(by_user)


def latest_prior_session(
    sessions: Sequence[FocalSession], timestamp: float, horizon: int
) -> FocalSession | None:
    times = [session.timestamp for session in sessions]
    index = bisect.bisect_left(times, timestamp) - 1
    if index < 0:
        return None
    focal = sessions[index]
    age = timestamp - focal.timestamp
    return focal if 0.0 < age <= horizon else None


def reconstruct_episodes(
    snapshot: Path,
    raw_to_item: dict[int, int],
    sessions_by_user: dict[int, list[FocalSession]],
    horizon: int,
) -> tuple[list[Episode], dict[str, int]]:
    requests: dict[tuple[int, int], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for split in ("recommendation_train", "recommendation_test"):
        frame = read_split(snapshot, split)
        for row in frame.itertuples(index=False):
            raw_user = int(row.user_idx)
            details = list(row.rec_result_details_with_idx)
            counts["requests_seen"] += 1
            if not details:
                counts["requests_empty_slate"] += 1
                continue
            raw_slate_occurrences = [int(detail["note_idx"]) for detail in details]
            raw_slate = tuple(dict.fromkeys(raw_slate_occurrences))
            counts["duplicate_slate_occurrences_removed"] += (
                len(raw_slate_occurrences) - len(raw_slate)
            )
            clicked_raw = tuple(
                dict.fromkeys(
                int(detail["note_idx"]) for detail in details if int(detail.get("click", 0)) != 0
                )
            )
            if not clicked_raw:
                counts["requests_without_click"] += 1
                continue
            timestamps = [float(detail["request_timestamp"]) for detail in details]
            missing = tuple(item for item in raw_slate if item not in raw_to_item)
            missing_clicked = tuple(item for item in clicked_raw if item not in raw_to_item)
            if missing_clicked:
                raise ValueError(
                    f"Clicked items absent from R239 item vocabulary: user={raw_user} "
                    f"request={row.request_idx} items={missing_clicked[:5]}"
                )
            key = (raw_user, int(row.request_idx))
            payload = {
                "timestamp": max(timestamps),
                "raw_slate": raw_slate,
                "missing_slate_items": missing,
                "slate": tuple(raw_to_item[item] for item in raw_slate if item in raw_to_item),
                "positives": tuple(raw_to_item[item] for item in clicked_raw),
            }
            if key in requests and requests[key] != payload:
                raise ValueError(f"Conflicting duplicate request record: {key}")
            requests[key] = payload

    by_user_requests: dict[int, list[tuple[tuple[int, int], dict[str, Any]]]] = defaultdict(list)
    for key, payload in requests.items():
        by_user_requests[key[0]].append((key, payload))

    eligible: list[Episode] = []
    for raw_user, rows in by_user_requests.items():
        user_id = raw_user
        sessions = sessions_by_user.get(user_id, [])
        history: list[int] = []
        ordered = sorted(rows, key=lambda pair: (pair[1]["timestamp"], pair[0][1]))
        by_timestamp: dict[float, list[tuple[tuple[int, int], dict[str, Any]]]] = defaultdict(list)
        for pair in ordered:
            by_timestamp[float(pair[1]["timestamp"])].append(pair)
        for timestamp in sorted(by_timestamp):
            same_time = by_timestamp[timestamp]
            for (owner, request_id), payload in same_time:
                focal = latest_prior_session(sessions, timestamp, horizon)
                if focal is None:
                    counts["clicked_requests_without_24h_focal"] += 1
                elif payload["missing_slate_items"]:
                    counts["eligible_requests_with_unmapped_slate_item"] += 1
                else:
                    eligible.append(
                        Episode(
                            episode_id=stable_digest(owner, request_id, timestamp),
                            user_id=user_id,
                            raw_user_id=raw_user,
                            request_id=request_id,
                            timestamp=timestamp,
                            slate=payload["slate"],
                            raw_slate=payload["raw_slate"],
                            positives=payload["positives"],
                            history=tuple(history),
                            focal=focal,
                        )
                    )
                    counts["eligible_clicked_requests_before_user_filter"] += 1
            for _key, payload in same_time:
                history.extend(payload["positives"])

    grouped: dict[int, list[Episode]] = defaultdict(list)
    for episode in eligible:
        grouped[episode.user_id].append(episode)
    final: list[Episode] = []
    for user_id, rows in grouped.items():
        rows.sort(key=lambda episode: (episode.timestamp, episode.request_id, episode.episode_id))
        timestamps = sorted({episode.timestamp for episode in rows})
        if len(timestamps) < 3:
            counts["eligible_requests_removed_user_lt3"] += len(rows)
            continue
        validation_timestamp, test_timestamp = timestamps[-2], timestamps[-1]
        for episode in rows:
            episode.split = (
                "test"
                if episode.timestamp == test_timestamp
                else "validation"
                if episode.timestamp == validation_timestamp
                else "train"
            )
            final.append(episode)
    final.sort(key=lambda episode: (episode.user_id, episode.timestamp, episode.request_id))
    counts["eligible_users"] = len({episode.user_id for episode in final})
    for split in SPLITS:
        counts[f"{split}_episodes"] = sum(episode.split == split for episode in final)
        counts[f"{split}_target_rows"] = sum(
            len(episode.positives) for episode in final if episode.split == split
        )
    return final, dict(counts)


def bound_smoke_rows(episodes: list[Episode], max_users: int, max_per_split: int) -> list[Episode]:
    selected = episodes
    if max_users > 0:
        users = sorted({episode.user_id for episode in episodes}, key=lambda uid: stable_digest(uid))[:max_users]
        selected = [episode for episode in selected if episode.user_id in set(users)]
    if max_per_split > 0:
        kept: list[Episode] = []
        for split in SPLITS:
            rows = [episode for episode in selected if episode.split == split]
            rows.sort(key=lambda episode: stable_digest(episode.episode_id))
            kept.extend(rows[:max_per_split])
        selected = kept
    return sorted(selected, key=lambda episode: (episode.split, episode.user_id, episode.timestamp))


def clicked_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def donor_candidates(
    episode: Episode,
    sessions: Sequence[FocalSession],
    times: Sequence[float],
    caliper_seconds: int,
) -> list[FocalSession]:
    lower = episode.focal.timestamp - caliper_seconds
    upper = min(episode.focal.timestamp + caliper_seconds, math.nextafter(episode.timestamp, -math.inf))
    left = bisect.bisect_left(times, lower)
    right = bisect.bisect_right(times, upper)
    bucket = clicked_count_bucket(len(episode.focal.clicked_items))
    return [
        donor
        for donor in sessions[left:right]
        if donor.user_id != episode.user_id
        and donor.query_id != episode.focal.query_id
        and donor.session_id != episode.focal.session_id
        and donor.timestamp < episode.timestamp
        and clicked_count_bucket(len(donor.clicked_items)) == bucket
    ]


def donor_reuse_summary(episodes: list[Episode], attribute: str) -> dict[str, Any]:
    donors = [getattr(episode, attribute) for episode in episodes]
    donors = [donor for donor in donors if donor is not None]
    counts = Counter(donor.session_id for donor in donors)
    multiplicities = list(counts.values())
    return {
        "assignments": len(donors),
        "unique_sessions": len(counts),
        "max_reuse": max(multiplicities, default=0),
        "p95_reuse": float(np.percentile(multiplicities, 95)) if multiplicities else 0.0,
    }


def freeze_donors(
    episodes: list[Episode],
    all_sessions: list[FocalSession],
    donor_seed: int,
) -> dict[str, Any]:
    by_bucket: dict[str, list[FocalSession]] = defaultdict(list)
    for session in all_sessions:
        by_bucket[clicked_count_bucket(len(session.clicked_items))].append(session)
    bucket_times = {
        bucket: [session.timestamp for session in sessions]
        for bucket, sessions in by_bucket.items()
    }
    calipers = (("5m", 5 * 60), ("15m", 15 * 60), ("60m", 60 * 60))
    fallback_counts: Counter[str] = Counter()
    donor_map: list[dict[str, Any]] = []
    time_differences_d1: list[float] = []
    time_differences_d2: list[float] = []
    all_session_times = [session.timestamp for session in all_sessions]
    eval_episodes = [episode for episode in episodes if episode.split in {"validation", "test"}]
    for episode in eval_episodes:
        bucket = clicked_count_bucket(len(episode.focal.clicked_items))
        sessions = by_bucket.get(bucket, [])
        times = bucket_times.get(bucket, [])
        d1_pool: list[FocalSession] = []
        d2_pool: list[FocalSession] = []
        d1_level = "unmatched"
        d2_level = "unmatched"
        for candidate_level, caliper in calipers:
            pool = donor_candidates(episode, sessions, times, caliper)
            if pool and not d1_pool:
                d1_pool = pool
                d1_level = candidate_level
            if len(pool) >= 2:
                d2_pool = pool
                d2_level = candidate_level
                break
        fallback_counts[f"d1_{d1_level}"] += 1
        fallback_counts[f"d2_{d2_level}"] += 1
        order_key = lambda donor: (
                    abs(donor.timestamp - episode.focal.timestamp),
                    stable_digest(donor_seed, episode.episode_id, donor.session_id),
                )
        if d1_pool:
            episode.matched_donor_d1 = sorted(d1_pool, key=order_key)[0]
            time_differences_d1.append(
                episode.matched_donor_d1.timestamp - episode.focal.timestamp
            )
        if d2_pool and episode.matched_donor_d1 is not None:
            episode.matched_donor_d2 = next(
                donor
                for donor in sorted(d2_pool, key=order_key)
                if donor.session_id != episode.matched_donor_d1.session_id
            )
            time_differences_d2.append(
                episode.matched_donor_d2.timestamp - episode.focal.timestamp
            )

        excluded = {
            episode.focal.session_id,
            episode.matched_donor_d1.session_id if episode.matched_donor_d1 else -1,
            episode.matched_donor_d2.session_id if episode.matched_donor_d2 else -1,
        }
        eligible_end = bisect.bisect_left(all_session_times, episode.timestamp)
        if eligible_end:
            start = int(
                stable_digest(donor_seed, "invalid", episode.episode_id)[:16], 16
            ) % eligible_end
            for offset in range(eligible_end):
                donor = all_sessions[(start + offset) % eligible_end]
                if (
                    donor.user_id != episode.user_id
                    and donor.query_id != episode.focal.query_id
                    and donor.session_id not in excluded
                ):
                    episode.invalid_source = donor
                    break
        donor_map.append(
            {
                "episode_id": episode.episode_id,
                "split": episode.split,
                "focal_session_id": episode.focal.session_id,
                "focal_user_id": episode.focal.user_id,
                "focal_query_id": episode.focal.query_id,
                "focal_timestamp": episode.focal.timestamp,
                "focal_clicked_count": len(episode.focal.clicked_items),
                "clicked_count_bucket": bucket,
                "d1_caliper_level": d1_level,
                "d2_caliper_level": d2_level,
                "matched_donor_d1_session_id": (
                    episode.matched_donor_d1.session_id if episode.matched_donor_d1 else None
                ),
                "matched_donor_d1_user_id": (
                    episode.matched_donor_d1.user_id if episode.matched_donor_d1 else None
                ),
                "matched_donor_d1_query_id": (
                    episode.matched_donor_d1.query_id if episode.matched_donor_d1 else None
                ),
                "matched_donor_d1_timestamp": (
                    episode.matched_donor_d1.timestamp if episode.matched_donor_d1 else None
                ),
                "matched_donor_d1_time_difference_seconds": (
                    episode.matched_donor_d1.timestamp - episode.focal.timestamp
                    if episode.matched_donor_d1
                    else None
                ),
                "matched_donor_d1_clicked_count": (
                    len(episode.matched_donor_d1.clicked_items)
                    if episode.matched_donor_d1
                    else None
                ),
                "matched_donor_d2_session_id": (
                    episode.matched_donor_d2.session_id if episode.matched_donor_d2 else None
                ),
                "matched_donor_d2_user_id": (
                    episode.matched_donor_d2.user_id if episode.matched_donor_d2 else None
                ),
                "matched_donor_d2_query_id": (
                    episode.matched_donor_d2.query_id if episode.matched_donor_d2 else None
                ),
                "matched_donor_d2_timestamp": (
                    episode.matched_donor_d2.timestamp if episode.matched_donor_d2 else None
                ),
                "matched_donor_d2_time_difference_seconds": (
                    episode.matched_donor_d2.timestamp - episode.focal.timestamp
                    if episode.matched_donor_d2
                    else None
                ),
                "matched_donor_d2_clicked_count": (
                    len(episode.matched_donor_d2.clicked_items)
                    if episode.matched_donor_d2
                    else None
                ),
                "invalid_source_session_id": (
                    episode.invalid_source.session_id if episode.invalid_source else None
                ),
                "invalid_source_user_id": (
                    episode.invalid_source.user_id if episode.invalid_source else None
                ),
                "invalid_source_query_id": (
                    episode.invalid_source.query_id if episode.invalid_source else None
                ),
                "invalid_source_timestamp": (
                    episode.invalid_source.timestamp if episode.invalid_source else None
                ),
                "invalid_source_clicked_count": (
                    len(episode.invalid_source.clicked_items)
                    if episode.invalid_source
                    else None
                ),
            }
        )
    coverage_by_split = {}
    for split in ("validation", "test"):
        rows = [episode for episode in eval_episodes if episode.split == split]
        d1 = sum(episode.matched_donor_d1 is not None for episode in rows)
        d2 = sum(episode.matched_donor_d2 is not None for episode in rows)
        invalid = sum(episode.invalid_source is not None for episode in rows)
        coverage_by_split[split] = {
            "episodes": len(rows),
            "d1_matched": d1,
            "d1_coverage": d1 / len(rows) if rows else 0.0,
            "d2_matched": d2,
            "d2_placebo_coverage": d2 / len(rows) if rows else 0.0,
            "invalid_source_matched": invalid,
            "invalid_source_coverage": invalid / len(rows) if rows else 0.0,
        }
    map_payload = json.dumps(donor_map, sort_keys=True, separators=(",", ":"))
    return {
        "policy": "cross-user query-distinct clicked-count-bucket; 5m/15m/60m calipers",
        "donor_seed": donor_seed,
        "fallback_counts": dict(fallback_counts),
        "coverage_by_split": coverage_by_split,
        "matched_time_difference_seconds": {
            "d1_mean": mean(time_differences_d1),
            "d1_mean_absolute": mean(abs(value) for value in time_differences_d1),
            "d1_p95_absolute": (
                float(np.percentile(np.abs(time_differences_d1), 95))
                if time_differences_d1
                else None
            ),
            "d2_mean": mean(time_differences_d2),
            "d2_mean_absolute": mean(abs(value) for value in time_differences_d2),
        },
        "clicked_count_bucket_mismatches": 0,
        "reuse": {
            attribute: donor_reuse_summary(eval_episodes, attribute)
            for attribute in ("matched_donor_d1", "matched_donor_d2", "invalid_source")
        },
        "donor_map_sha256": hashlib.sha256(map_payload.encode("utf-8")).hexdigest(),
        "donor_map": donor_map,
    }


def integrity_checks(
    episodes: list[Episode], smoke: bool, focal_horizon_seconds: int
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "status": "pass" if passed else "fail", "detail": detail})

    request_splits: dict[tuple[int, int], set[str]] = defaultdict(set)
    bad_target = bad_time = bad_donor = 0
    for episode in episodes:
        request_splits[(episode.user_id, episode.request_id)].add(episode.split)
        bad_target += int(not set(episode.positives).issubset(set(episode.slate)))
        bad_time += int(
            not (0.0 < episode.timestamp - episode.focal.timestamp <= focal_horizon_seconds)
        )
        if episode.split in {"validation", "test"}:
            matched = (episode.matched_donor_d1, episode.matched_donor_d2)
            for donor in matched:
                if donor is None:
                    continue
                bad_donor += int(
                    donor.user_id == episode.user_id
                    or donor.query_id == episode.focal.query_id
                    or donor.session_id == episode.focal.session_id
                    or donor.timestamp >= episode.timestamp
                    or clicked_count_bucket(len(donor.clicked_items))
                    != clicked_count_bucket(len(episode.focal.clicked_items))
                    or abs(donor.timestamp - episode.focal.timestamp) > 60 * 60
                )
            invalid = episode.invalid_source
            if invalid is not None:
                bad_donor += int(
                    invalid.user_id == episode.user_id
                    or invalid.query_id == episode.focal.query_id
                    or invalid.session_id
                    in {
                        episode.focal.session_id,
                        episode.matched_donor_d1.session_id
                        if episode.matched_donor_d1
                        else -1,
                        episode.matched_donor_d2.session_id
                        if episode.matched_donor_d2
                        else -1,
                    }
                    or invalid.timestamp >= episode.timestamp
                )
            if episode.matched_donor_d1 and episode.matched_donor_d2:
                bad_donor += int(
                    episode.matched_donor_d1.session_id
                    == episode.matched_donor_d2.session_id
                )
    split_leaks = sum(len(values) != 1 for values in request_splits.values())
    test = [episode for episode in episodes if episode.split == "test"]
    add("request_level_split_exclusivity", split_leaks == 0, f"bad_requests={split_leaks}")
    add("targets_in_original_logged_slate", bad_target == 0, f"bad_episodes={bad_target}")
    add("strict_24h_focal_temporality", bad_time == 0, f"bad_episodes={bad_time}")
    matched_test = sum(episode.matched_donor_d1 is not None for episode in test)
    add("donor_identity_and_strata", bad_donor == 0, f"bad_assignments={bad_donor}")
    d2_test = sum(episode.matched_donor_d2 is not None for episode in test)
    minimum_coverage = 0.90
    add(
        "matched_donor_test_coverage",
        smoke
        or (
            len(test) > 0
            and matched_test / len(test) >= minimum_coverage
            and d2_test / len(test) >= minimum_coverage
        ),
        f"d1={matched_test}/{len(test)} d2={d2_test}/{len(test)} minimum={minimum_coverage}",
    )
    add(
        "heldout_surface_minimum",
        smoke or (len(test) >= 1000 and len({episode.user_id for episode in test}) >= 500),
        f"smoke={smoke} test_episodes={len(test)} test_users={len({e.user_id for e in test})}",
    )
    add("all_splits_nonempty", all(any(e.split == split for e in episodes) for split in SPLITS), "request episodes")
    return checks


class EpisodeDataset(Dataset[Episode]):
    def __init__(self, episodes: Sequence[Episode]):
        self.episodes = list(episodes)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> Episode:
        return self.episodes[index]


def pad_sequences(sequences: Sequence[Sequence[int]], minimum: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(minimum, max((len(sequence) for sequence in sequences), default=0))
    values = torch.zeros((len(sequences), width), dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        if sequence:
            values[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)
            mask[row, : len(sequence)] = True
    return values, mask


def focal_for_view(episode: Episode, view: str) -> FocalSession | None:
    if view in {"full", "search_only"}:
        return episode.focal
    if view == "matched_donor_d1":
        return episode.matched_donor_d1
    if view == "matched_donor_d2":
        return episode.matched_donor_d2
    if view == "invalid_source":
        return episode.invalid_source
    if view in {"history_only", "focal_masked"}:
        return None
    raise ValueError(view)


def collate_episodes(
    episodes: Sequence[Episode],
    max_history: int,
    max_focal_items: int,
    view: str = "full",
) -> dict[str, Any]:
    histories = [episode.history[-max_history:] for episode in episodes]
    if view == "search_only":
        histories = [tuple() for _ in episodes]
    history, history_mask = pad_sequences(histories)
    candidates, candidate_mask = pad_sequences([episode.slate for episode in episodes])
    positive_mask = torch.zeros_like(candidate_mask)
    for row, episode in enumerate(episodes):
        positives = set(episode.positives)
        for col, candidate in enumerate(episode.slate):
            positive_mask[row, col] = candidate in positives

    focals = [focal_for_view(episode, view) for episode in episodes]
    source_mode = torch.tensor(
        [0 if view == "history_only" else 2 if view == "focal_masked" else 1]
        * len(episodes),
        dtype=torch.long,
    )
    query_ids = torch.tensor(
        [0 if focal is None else focal.query_id for focal in focals], dtype=torch.long
    )
    focal_items, focal_item_mask = pad_sequences(
        [
            tuple() if focal is None else focal.clicked_items[-max_focal_items:]
            for focal in focals
        ]
    )
    return {
        "episodes": list(episodes),
        "history": history,
        "history_mask": history_mask,
        "candidates": candidates,
        "candidate_mask": candidate_mask,
        "positive_mask": positive_mask,
        "query_ids": query_ids,
        "focal_items": focal_items,
        "focal_item_mask": focal_item_mask,
        "source_mode": source_mode,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


class LowRankProjection(nn.Module):
    def __init__(self, input_dim: int, rank: int, output_dim: int):
        super().__init__()
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, output_dim, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(values))


class R239Scorer(nn.Module):
    def __init__(
        self,
        item_embeddings: torch.Tensor,
        query_embeddings: torch.Tensor,
        rank: int,
        hidden_dim: int,
        dropout: float,
        neutral_focal_raw: torch.Tensor,
    ):
        super().__init__()
        if item_embeddings.shape[1] != query_embeddings.shape[1]:
            raise ValueError("Item and query BGE dimensions differ")
        input_dim = int(item_embeddings.shape[1])
        item_table = torch.cat([torch.zeros((1, input_dim)), item_embeddings], dim=0)
        query_table = torch.cat([torch.zeros((1, input_dim)), query_embeddings], dim=0)
        self.register_buffer("item_embeddings", item_table, persistent=False)
        self.register_buffer("query_embeddings", query_table, persistent=False)
        if neutral_focal_raw.shape != (2 * input_dim,):
            raise ValueError(
                f"neutral_focal_raw shape={tuple(neutral_focal_raw.shape)} "
                f"expected={(2 * input_dim,)}"
            )
        self.register_buffer(
            "neutral_focal_raw", neutral_focal_raw.float().clone(), persistent=True
        )
        self.history_projection = LowRankProjection(input_dim, rank, hidden_dim)
        self.focal_projection = LowRankProjection(2 * input_dim, rank, hidden_dim)
        self.item_projection = LowRankProjection(input_dim, rank, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.gate = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(hidden_dim)

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(values.dtype)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        return (values * weights).sum(dim=1) / denominator

    def forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        history_raw = self.masked_mean(
            self.item_embeddings[batch["history"]], batch["history_mask"]
        )
        query_raw = self.query_embeddings[batch["query_ids"]]
        focal_item_raw = self.masked_mean(
            self.item_embeddings[batch["focal_items"]], batch["focal_item_mask"]
        )
        focal_raw = torch.cat([query_raw, focal_item_raw], dim=-1)
        masked = batch["source_mode"].eq(2).unsqueeze(-1)
        focal_raw = torch.where(
            masked, self.neutral_focal_raw.unsqueeze(0).expand_as(focal_raw), focal_raw
        )

        history = self.dropout(self.history_projection(history_raw))
        focal = self.dropout(self.focal_projection(focal_raw))
        source_active = batch["source_mode"].ne(0).unsqueeze(-1).to(focal.dtype)
        focal = focal * source_active
        gate = torch.sigmoid(self.gate(torch.cat([history, focal], dim=-1)))
        fused = self.output_norm(history + gate * focal)
        candidates = self.dropout(
            self.item_projection(self.item_embeddings[batch["candidates"]])
        )
        logits = torch.einsum("bd,bkd->bk", fused, candidates) / self.scale
        logits = logits.masked_fill(~batch["candidate_mask"], float("-inf"))
        return logits, gate.squeeze(-1)


def listwise_multi_positive_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not positive_mask.any(dim=1).all():
        raise ValueError("Every training episode must contain at least one clicked slate item")
    log_prob = torch.log_softmax(logits, dim=1)
    targets = positive_mask.to(log_prob.dtype)
    targets = targets / targets.sum(dim=1, keepdim=True)
    return -(targets * torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))).sum(dim=1).mean()


@torch.no_grad()
def train_only_neutral_focal_raw(
    train_episodes: list[Episode],
    item_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    max_focal_items: int,
) -> torch.Tensor:
    item_table = torch.cat(
        [torch.zeros((1, item_embeddings.shape[1])), item_embeddings], dim=0
    )
    query_table = torch.cat(
        [torch.zeros((1, query_embeddings.shape[1])), query_embeddings], dim=0
    )
    rows: list[torch.Tensor] = []
    for episode in train_episodes:
        clicked = episode.focal.clicked_items[-max_focal_items:]
        item_mean = (
            item_table[torch.tensor(clicked, dtype=torch.long)].mean(dim=0)
            if clicked
            else torch.zeros(item_embeddings.shape[1])
        )
        rows.append(torch.cat([query_table[episode.focal.query_id], item_mean]))
    if not rows:
        raise ValueError("Cannot estimate neutral focal representation without train episodes")
    return torch.stack(rows).mean(dim=0)


@torch.no_grad()
def score_loader(
    model: R239Scorer,
    episodes: list[Episode],
    args: argparse.Namespace,
    device: torch.device,
    view: str,
) -> None:
    loader = DataLoader(
        EpisodeDataset(episodes),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate_episodes(
            rows, args.max_history, args.max_focal_items, view
        ),
    )
    model.eval()
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits, gates = model(batch)
        for row, episode in enumerate(raw_batch["episodes"]):
            width = len(episode.slate)
            scores = logits[row, :width].detach().cpu().float().numpy()
            if not np.isfinite(scores).all():
                raise ValueError(f"Non-finite {view} scores for {episode.episode_id}")
            episode.scores[view] = scores
            episode.gates[view] = float(gates[row].detach().cpu())


def ranking_metrics(
    scores: np.ndarray, slate: Sequence[int], positives: Sequence[int]
) -> dict[str, float]:
    order = sorted(range(len(slate)), key=lambda index: (-float(scores[index]), int(slate[index])))
    ranks = {int(slate[index]): rank for rank, index in enumerate(order, start=1)}
    positive_ranks = sorted(ranks[int(item)] for item in positives)
    rr = 1.0 / positive_ranks[0]
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank in positive_ranks if rank <= 10
    )
    ideal_count = min(len(positive_ranks), 10)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "rr": float(rr),
        "ndcg10": float(dcg / ideal_dcg) if ideal_dcg else 0.0,
        "hr10": float(positive_ranks[0] <= 10),
        "best_positive_rank": float(positive_ranks[0]),
    }


def episode_rows(episodes: Iterable[Episode], tie_tol: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        candidate_hash = stable_digest(*episode.raw_slate)
        record: dict[str, Any] = {
            "episode_id": episode.episode_id,
            "user_id": episode.user_id,
            "raw_user_id": episode.raw_user_id,
            "request_id": episode.request_id,
            "split": episode.split,
            "request_timestamp": episode.timestamp,
            "slate_size": len(episode.slate),
            "positive_count": len(episode.positives),
            "candidate_hash": candidate_hash,
            "candidate_item_ids_json": json.dumps(episode.slate, separators=(",", ":")),
            "candidate_raw_item_ids_json": json.dumps(episode.raw_slate, separators=(",", ":")),
            "positive_item_ids_json": json.dumps(episode.positives, separators=(",", ":")),
            "focal_session_id": episode.focal.session_id,
            "focal_age_seconds": episode.timestamp - episode.focal.timestamp,
            "matched_donor_d1_session_id": (
                episode.matched_donor_d1.session_id if episode.matched_donor_d1 else None
            ),
            "matched_donor_d2_session_id": (
                episode.matched_donor_d2.session_id if episode.matched_donor_d2 else None
            ),
            "invalid_source_session_id": (
                episode.invalid_source.session_id if episode.invalid_source else None
            ),
        }
        for view in (*VIEWS, "popularity", "item_text_only"):
            if view not in episode.scores:
                for metric in ("rr", "ndcg10", "hr10", "best_positive_rank"):
                    record[f"{view}_{metric}"] = None
                record[f"{view}_scores_json"] = None
                if view in VIEWS:
                    record[f"{view}_gate"] = None
                continue
            values = ranking_metrics(episode.scores[view], episode.slate, episode.positives)
            for metric, value in values.items():
                record[f"{view}_{metric}"] = value
            record[f"{view}_scores_json"] = json.dumps(
                [float(value) for value in episode.scores[view]], separators=(",", ":")
            )
            if view in VIEWS:
                record[f"{view}_gate"] = episode.gates[view]

        comparisons = (
            ("true_minus_d1_rr", "true_vs_d1_outcome", "full", "matched_donor_d1"),
            ("d1_minus_d2_rr", "d1_vs_d2_outcome", "matched_donor_d1", "matched_donor_d2"),
            ("true_minus_invalid_rr", "true_vs_invalid_outcome", "full", "invalid_source"),
        )
        for delta_key, outcome_key, left, right in comparisons:
            if record[f"{left}_rr"] is None or record[f"{right}_rr"] is None:
                record[delta_key] = None
                record[outcome_key] = "missing"
                continue
            delta = float(record[f"{left}_rr"]) - float(record[f"{right}_rr"])
            record[delta_key] = delta
            record[outcome_key] = (
                "win" if delta > tie_tol else "loss" if delta < -tie_tol else "tie"
            )
        rows.append(record)
    return rows


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    metric: Callable[[list[dict[str, Any]]], float],
    iters: int,
    seed: int,
) -> dict[str, Any]:
    by_user: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_user[int(row["user_id"])].append(row)
    users = sorted(by_user)
    point = metric(rows)
    if len(users) < 2 or iters <= 0:
        return {"point": point, "ci_low": None, "ci_high": None, "user_clusters": len(users)}
    rng = np.random.default_rng(seed)
    values = np.empty(iters, dtype=np.float64)
    for index in range(iters):
        sampled = rng.choice(users, size=len(users), replace=True)
        sample_rows = [row for user in sampled for row in by_user[int(user)]]
        values[index] = metric(sample_rows)
    return {
        "point": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "user_clusters": len(users),
    }


def half_tie(rows: list[dict[str, Any]], outcome_key: str) -> float:
    wins = sum(row[outcome_key] == "win" for row in rows)
    ties = sum(row[outcome_key] == "tie" for row in rows)
    return float((wins + 0.5 * ties) / len(rows)) if rows else float("nan")


def signflip_test(
    rows: list[dict[str, Any]], delta_key: str, iters: int, seed: int
) -> dict[str, Any]:
    by_user: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        by_user[int(row["user_id"])].append(float(row[delta_key]))
    values = np.asarray([np.mean(by_user[user]) for user in sorted(by_user)], dtype=np.float64)
    observed = float(values.mean()) if len(values) else float("nan")
    if len(values) < 2 or iters <= 0:
        return {"observed_mean_user_delta": observed, "p_one_sided": None, "user_clusters": len(values)}
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(iters):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(values))
        exceed += int(float(np.mean(values * signs)) >= observed)
    return {
        "observed_mean_user_delta": observed,
        "p_one_sided": float((exceed + 1) / (iters + 1)),
        "user_clusters": len(values),
    }


def random_ndcg10(rows: list[dict[str, Any]]) -> float:
    expectations = []
    for row in rows:
        size = int(row["slate_size"])
        positives = int(row["positive_count"])
        cutoff = min(10, size)
        expected_dcg = (positives / size) * sum(
            1.0 / math.log2(rank + 1) for rank in range(1, cutoff + 1)
        )
        ideal_dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(positives, cutoff) + 1)
        )
        expectations.append(expected_dcg / ideal_dcg if ideal_dcg else 0.0)
    return mean(expectations)


def summarize_split(
    rows: list[dict[str, Any]],
    split: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subset = [row for row in rows if row["split"] == split]
    metrics: dict[str, Any] = {
        "target_rows": len(subset),
        "episodes": len({row["episode_id"] for row in subset}),
        "users": len({row["user_id"] for row in subset}),
        "full_rr": mean(float(row["full_rr"]) for row in subset),
        "full_ndcg10": mean(float(row["full_ndcg10"]) for row in subset),
        "full_hr10": mean(float(row["full_hr10"]) for row in subset),
        "random_ranking_expected_ndcg10": random_ndcg10(subset),
        "view_means": {
            view: {
                "rr": mean(float(row[f"{view}_rr"]) for row in subset),
                "ndcg10": mean(float(row[f"{view}_ndcg10"]) for row in subset),
                "hr10": mean(float(row[f"{view}_hr10"]) for row in subset),
            }
            for view in VIEWS
        },
        "baseline_means": {
            "popularity_ndcg10": mean(
                float(row["popularity_ndcg10"]) for row in subset
            ),
            "item_text_only_ndcg10": mean(
                float(row["item_text_only_ndcg10"]) for row in subset
            ),
            "history_only_ndcg10": mean(
                float(row["history_only_ndcg10"]) for row in subset
            ),
        },
    }
    definitions = {
        "dependence_full_minus_masked_rr": lambda rs: mean(
            float(row["full_rr"]) - float(row["focal_masked_rr"]) for row in rs
        ),
        "recovery_full_minus_history_rr": lambda rs: mean(
            float(row["full_rr"]) - float(row["history_only_rr"]) for row in rs
        ),
        "matched_true_vs_d1_half_tie": lambda rs: half_tie(
            [row for row in rs if row["true_vs_d1_outcome"] != "missing"],
            "true_vs_d1_outcome",
        ),
        "destroyed_reference_d1_vs_d2_half_tie": lambda rs: half_tie(
            [row for row in rs if row["d1_vs_d2_outcome"] != "missing"],
            "d1_vs_d2_outcome",
        ),
        "true_vs_invalid_half_tie": lambda rs: half_tie(
            [row for row in rs if row["true_vs_invalid_outcome"] != "missing"],
            "true_vs_invalid_outcome",
        ),
        "specificity_minus_placebo_half_tie": lambda rs: (
            half_tie(
                [row for row in rs if row["d1_vs_d2_outcome"] != "missing"],
                "true_vs_d1_outcome",
            )
            - half_tie(
                [row for row in rs if row["d1_vs_d2_outcome"] != "missing"],
                "d1_vs_d2_outcome",
            )
        ),
    }
    for offset, (name, metric) in enumerate(definitions.items()):
        metrics[name] = cluster_bootstrap(subset, metric, args.bootstrap_iters, args.seed + 1000 + offset)
    metrics["matched_signflip"] = signflip_test(
        [row for row in subset if row["true_vs_d1_outcome"] != "missing"],
        "true_minus_d1_rr",
        args.signflip_iters,
        args.seed + 2001,
    )
    metrics["outcome_counts"] = {
        name: dict(Counter(row[key] for row in subset))
        for name, key in (
            ("true_vs_d1", "true_vs_d1_outcome"),
            ("d1_vs_d2", "d1_vs_d2_outcome"),
            ("true_vs_invalid", "true_vs_invalid_outcome"),
        )
    }
    return metrics


def train_model(
    model: R239Scorer,
    train_episodes: list[Episode],
    validation_episodes: list[Episode],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], int]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        EpisodeDataset(train_episodes),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate_episodes(
            rows, args.max_history, args.max_focal_items, "full"
        ),
    )
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_ndcg = -float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _gate = model(batch)
            loss = listwise_multi_positive_loss(logits, batch["positive_mask"])
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            size = len(raw_batch["episodes"])
            total_loss += float(loss.detach().cpu()) * size
            total_rows += size

        for episode in validation_episodes:
            episode.scores.clear()
            episode.gates.clear()
        score_loader(model, validation_episodes, args, device, "full")
        val_ndcgs = [
            ranking_metrics(episode.scores["full"], episode.slate, episode.positives)[
                "ndcg10"
            ]
            for episode in validation_episodes
        ]
        val_ndcg = mean(val_ndcgs)
        epoch_record = {
            "epoch": float(epoch),
            "train_listwise_cross_entropy": total_loss / max(1, total_rows),
            "validation_full_ndcg10": val_ndcg,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        if math.isfinite(val_ndcg) and val_ndcg > best_ndcg:
            best_ndcg = val_ndcg
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                break
    if best_state is None:
        raise ValueError("Validation NDCG@10 never became finite")
    return history, best_state, best_epoch


class ItemTextOnlyScorer(nn.Module):
    """Zero-context train-only baseline over frozen candidate text embeddings."""

    def __init__(self, item_embeddings: torch.Tensor):
        super().__init__()
        dim = int(item_embeddings.shape[1])
        self.register_buffer(
            "item_embeddings",
            torch.cat([torch.zeros((1, dim)), item_embeddings], dim=0),
            persistent=False,
        )
        self.head = nn.Linear(dim, 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        logits = self.head(self.item_embeddings[batch["candidates"]]).squeeze(-1)
        return logits.masked_fill(~batch["candidate_mask"], float("-inf"))


def train_item_text_baseline(
    item_embeddings: torch.Tensor,
    train_episodes: list[Episode],
    validation_episodes: list[Episode],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ItemTextOnlyScorer, int]:
    model = ItemTextOnlyScorer(item_embeddings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    generator = torch.Generator().manual_seed(args.seed + 991)
    loader = DataLoader(
        EpisodeDataset(train_episodes),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate_episodes(
            rows, args.max_history, args.max_focal_items, "full"
        ),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_ndcg = -float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = listwise_multi_positive_loss(logits, batch["positive_mask"])
            loss.backward()
            optimizer.step()
        model.eval()
        ndcgs: list[float] = []
        validation_loader = DataLoader(
            EpisodeDataset(validation_episodes),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda rows: collate_episodes(
                rows, args.max_history, args.max_focal_items, "full"
            ),
        )
        with torch.no_grad():
            for raw_batch in validation_loader:
                batch = move_batch(raw_batch, device)
                logits = model(batch)
                for row, episode in enumerate(raw_batch["episodes"]):
                    scores = logits[row, : len(episode.slate)].detach().cpu().numpy()
                    ndcgs.append(
                        ranking_metrics(scores, episode.slate, episode.positives)["ndcg10"]
                    )
        validation_ndcg = mean(ndcgs)
        if math.isfinite(validation_ndcg) and validation_ndcg > best_ndcg:
            best_ndcg = validation_ndcg
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.early_stopping_patience:
                break
    if best_state is None:
        raise ValueError("Item-text-only baseline validation NDCG never became finite")
    model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def score_quality_baselines(
    item_text_model: ItemTextOnlyScorer,
    train_episodes: list[Episode],
    eval_episodes: list[Episode],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    popularity = Counter(
        item for episode in train_episodes for item in episode.positives
    )
    item_text_model.eval()
    loader = DataLoader(
        EpisodeDataset(eval_episodes),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda rows: collate_episodes(
            rows, args.max_history, args.max_focal_items, "full"
        ),
    )
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        logits = item_text_model(batch)
        for row, episode in enumerate(raw_batch["episodes"]):
            width = len(episode.slate)
            episode.scores["item_text_only"] = (
                logits[row, :width].detach().cpu().float().numpy()
            )
            episode.scores["popularity"] = np.asarray(
                [float(popularity[item]) for item in episode.slate], dtype=np.float32
            )


def validate_embedding_contract(
    item_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    item_vocab: dict[int, Any],
    sessions: list[FocalSession],
) -> None:
    expected_items = len(item_vocab) - 1
    if item_embeddings.shape[0] != expected_items:
        raise ValueError(
            f"Item embedding rows={item_embeddings.shape[0]} but item vocab requires {expected_items}"
        )
    max_query = max((session.query_id for session in sessions), default=0)
    if query_embeddings.shape[0] < max_query:
        raise ValueError(
            f"Query embedding rows={query_embeddings.shape[0]} but max query id is {max_query}"
        )


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.bootstrap_iters = min(args.bootstrap_iters, 128)
        args.signflip_iters = min(args.signflip_iters, 128)
        args.max_users = args.max_users or 32
        args.max_episodes_per_split = args.max_episodes_per_split or 128
    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_path = args.item_embeddings or args.data_dir / "emb" / "bge-m3_item.pt"
    query_path = args.query_embeddings or args.data_dir / "emb" / "bge-m3_query.pt"

    start = time.time()
    raw_to_item, query_to_id, item_vocab = vocab_mappings(args.data_dir)
    item_embeddings = load_tensor(item_path)
    query_embeddings = load_tensor(query_path)
    all_sessions, sessions_by_user = load_focal_sessions(
        args.snapshot, raw_to_item, query_to_id
    )
    validate_embedding_contract(item_embeddings, query_embeddings, item_vocab, all_sessions)
    episodes, construction_counts = reconstruct_episodes(
        args.snapshot,
        raw_to_item,
        sessions_by_user,
        args.focal_horizon_seconds,
    )
    episodes = bound_smoke_rows(episodes, args.max_users, args.max_episodes_per_split)
    donor_summary = freeze_donors(episodes, all_sessions, args.donor_seed)
    donor_map_rows = donor_summary.pop("donor_map")
    checks = integrity_checks(episodes, args.smoke, args.focal_horizon_seconds)
    failed = [check["name"] for check in checks if check["status"] != "pass"]
    if construction_counts.get("eligible_requests_with_unmapped_slate_item", 0):
        failed.append("r239_embedding_coverage_for_original_logged_slates")
    if failed:
        raise RuntimeError("R239 integrity preflight failed: " + ", ".join(sorted(set(failed))))

    train_episodes = [episode for episode in episodes if episode.split == "train"]
    validation_episodes = [episode for episode in episodes if episode.split == "validation"]
    test_episodes = [episode for episode in episodes if episode.split == "test"]
    neutral_focal_raw = train_only_neutral_focal_raw(
        train_episodes, item_embeddings, query_embeddings, args.max_focal_items
    )
    model = R239Scorer(
        item_embeddings,
        query_embeddings,
        args.projection_rank,
        args.hidden_dim,
        args.dropout,
        neutral_focal_raw,
    ).to(device)
    training_history, best_state, best_epoch = train_model(
        model, train_episodes, validation_episodes, args, device
    )
    model.load_state_dict(best_state)

    eval_episodes = validation_episodes + test_episodes
    for episode in eval_episodes:
        episode.scores.clear()
        episode.gates.clear()
    for view in VIEWS:
        score_loader(model, eval_episodes, args, device, view)
    if all(
        np.allclose(
            episode.scores["history_only"],
            episode.scores["focal_masked"],
            rtol=0.0,
            atol=1e-12,
        )
        for episode in eval_episodes
    ):
        raise RuntimeError(
            "Focal-masked scores are identical to History-only on every eval row"
        )
    item_text_model, item_text_best_epoch = train_item_text_baseline(
        item_embeddings, train_episodes, validation_episodes, args, device
    )
    score_quality_baselines(
        item_text_model, train_episodes, eval_episodes, args, device
    )
    target_rows = episode_rows(eval_episodes, args.tie_tol)

    split_metrics = {
        split: summarize_split(target_rows, split, args)
        for split in ("validation", "test")
    }
    test_metric = split_metrics["test"]
    quality_values = [float(row["full_ndcg10"]) for row in target_rows if row["split"] == "test"]
    full_score_values = [
        float(value)
        for episode in test_episodes
        for value in episode.scores.get("full", np.asarray([], dtype=float))
    ]
    failures: list[str] = []
    if not quality_values or not all(math.isfinite(value) for value in quality_values):
        failures.append("nonfinite_test_quality")
    if len(full_score_values) < 2 or float(np.std(full_score_values)) <= 1e-12:
        failures.append("constant_test_scores")
    if test_metric["full_ndcg10"] <= test_metric["random_ranking_expected_ndcg10"]:
        failures.append("full_ndcg10_not_above_random")
    if test_metric["full_ndcg10"] <= test_metric["baseline_means"]["popularity_ndcg10"]:
        failures.append("full_ndcg10_not_above_popularity")
    if test_metric["full_ndcg10"] <= test_metric["baseline_means"]["item_text_only_ndcg10"]:
        failures.append("full_ndcg10_not_above_item_text_only")

    scores_path = args.output_dir / "r239_row_scores.csv"
    metrics_path = args.output_dir / "r239_metrics.json"
    checkpoint_path = args.output_dir / "r239_checkpoint.pt"
    manifest_path = args.output_dir / "r239_manifest.json"
    donor_map_path = args.output_dir / "r239_donor_map.csv"
    write_csv(scores_path, target_rows)
    write_csv(donor_map_path, donor_map_rows)
    status = (
        "SMOKE_COMPLETE_NON_CLAIM_BEARING"
        if args.smoke
        else "RUN_COMPLETE"
        if not failures
        else "QUALITY_GATE_FAIL"
    )
    metrics_payload = {
        "schema": "focalswap.qilin_trained_full_chain.metrics.v2",
        "status": status,
        "claim_authorized": False,
        "seed": args.seed,
        "smoke": args.smoke,
        "best_epoch_selected_on_validation_ndcg10": best_epoch,
        "item_text_only_best_epoch_selected_on_validation_ndcg10": item_text_best_epoch,
        "training_history": training_history,
        "split_metrics": split_metrics,
        "quality_failures": failures,
        "metric_definitions": {
            "target_unit": "one CSV row per eligible recommendation request with multi-positive labels; user-cluster inference",
            "request_unit": "eligible distinct recommendation request",
            "recovery": "paired Full minus History-only request RR",
            "destroyed_reference_placebo": "matched donor d1 versus distinct matched donor d2",
        },
    }
    write_json(metrics_path, metrics_payload)
    torch.save(
        {
            "schema": "focalswap.qilin_trained_full_chain.checkpoint.v2",
            "model_state": best_state,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "training_history": training_history,
        },
        checkpoint_path,
    )

    input_files = {
        "item_vocab": args.data_dir / "vocab" / "item_vocab.pkl",
        "query_vocab": args.data_dir / "vocab" / "query_vocab.pkl",
        "inventory_summary": args.data_dir / "inventory_summary.json",
        "item_embeddings": item_path,
        "query_embeddings": query_path,
        "recommendation_train_snapshot": args.snapshot / "recommendation_train" / "train-00000-of-00001.parquet",
        "recommendation_test_snapshot": args.snapshot / "recommendation_test" / "train-00000-of-00001.parquet",
        "search_train_snapshot": args.snapshot / "search_train" / "train-00000-of-00001.parquet",
        "search_test_snapshot": args.snapshot / "search_test" / "train-00000-of-00001.parquet",
        "canonical_preflight": ROOT / "reports" / "qilin_trained_full_chain_20260712" / "r239_full_data_preflight.json",
        "experiment_plan": ROOT / "refine-logs" / "EXPERIMENT_PLAN_R239_QILIN_TRAINED_FULL_CHAIN_CALIBRATION_20260712.md",
        "plan_to_run_gate": ROOT / "refine-logs" / "PLAN_TO_RUN_GATE_R239_QILIN_TRAINED_FULL_CHAIN_CALIBRATION_20260712.json",
        "preflight_code": ROOT / "tools" / "r239_qilin_full_chain_preflight.py",
        "artifact_builder_code": ROOT / "tools" / "r239_qilin_prepare_artifacts.py",
        "training_code": Path(__file__).resolve(),
        "wrapper_code": ROOT / "tools" / "run_qilin_trained_full_chain_autodl.sh",
    }
    embedding_summary = args.data_dir / "embedding_summary.json"
    if embedding_summary.exists():
        input_files["embedding_summary"] = embedding_summary
    manifest = {
        "schema": "focalswap.qilin_trained_full_chain.manifest.v2",
        "status": metrics_payload["status"],
        "claim_authorized": False,
        "contract": {
            "split_unit": "eligible distinct recommendation request; per-user leave-last-two",
            "candidate_surface": "original logged recommendation slate",
            "training_objective": "multi-positive listwise cross-entropy only",
            "model_selection": "validation NDCG@10 only; test scored after checkpoint selection",
            "bge_frozen": True,
            "views": list(VIEWS),
            "no_test_driven_tuning": True,
            "neutral_focal": "mean raw focal representation from train episodes only",
        },
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "construction_counts": construction_counts,
        "selected_episode_counts": dict(Counter(episode.split for episode in episodes)),
        "selected_target_counts": dict(
            Counter(episode.split for episode in episodes for _ in episode.positives)
        ),
        "donors": donor_summary,
        "neutral_focal_sha256": hashlib.sha256(
            neutral_focal_raw.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "integrity_checks": checks,
        "input_sha256": {name: sha256_file(path) for name, path in input_files.items()},
        "bge_model_directory_sha256": (
            sha256_directory(args.bge_model_dir) if args.bge_model_dir else None
        ),
        "output_sha256": {
            "row_scores_csv": sha256_file(scores_path),
            "metrics_json": sha256_file(metrics_path),
            "checkpoint": sha256_file(checkpoint_path),
            "donor_map_csv": sha256_file(donor_map_path),
        },
        "outputs": {
            "row_scores_csv": str(scores_path),
            "metrics_json": str(metrics_path),
            "checkpoint": str(checkpoint_path),
            "donor_map_csv": str(donor_map_path),
            "manifest": str(manifest_path),
        },
        "elapsed_seconds": time.time() - start,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": metrics_payload["status"],
                "output_dir": str(args.output_dir),
                "best_epoch": best_epoch,
                "quality_failures": failures,
            },
            indent=2,
        )
    )
    return 0 if not failures or args.smoke else 2


if __name__ == "__main__":
    raise SystemExit(main())
