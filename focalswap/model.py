"""Collect fixed-row model scores and construct observed-search donor maps."""

from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from .audit import validate_identity, view_rank


def match_donors(
    rows: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    seed: int = 2026,
    different_user: bool = True,
) -> dict[str, str | None]:
    """Use one seeded cyclic assignment within exact metadata match buckets.

    Match fields come from row['match']. Singleton buckets, identical search
    inputs and disallowed same-user pairs remain ineligible. Each donor is used
    at most once. The assignment uses only matching metadata, users and search.
    """
    if not fields or len(set(fields)) != len(fields):
        raise ValueError(
            "Provide distinct match fields derived from pre-outcome inputs."
        )
    buckets = defaultdict(list)
    mapping = {}
    for row in rows:
        validate_identity(row)
        if row["row_id"] in mapping:
            raise ValueError("row_id must be unique in the donor population.")
        mapping[row["row_id"]] = None
        search = row.get("search")
        if not isinstance(search, list):
            raise TypeError("search must be a list of observed search episodes.")
        metadata = row.get("match", {})
        if not isinstance(metadata, dict) or any(
            field not in metadata or metadata[field] is None for field in fields
        ):
            raise ValueError("Each row needs non-null values for every match field.")
        key = json.dumps(
            [metadata[field] for field in fields], sort_keys=True, allow_nan=False
        )
        if search:
            buckets[key].append(row)
    rng = random.Random(seed)
    for key in sorted(buckets):
        bucket = sorted(buckets[key], key=lambda row: row["row_id"])
        if len(bucket) < 2:
            continue
        rng.shuffle(bucket)
        for index, row in enumerate(bucket):
            donor = bucket[(index + 1) % len(bucket)]
            if row["search"] == donor["search"]:
                continue
            if different_user and row["user_id"] == donor["user_id"]:
                continue
            mapping[row["row_id"]] = donor["row_id"]
    return mapping


def collect_scores(
    rows: list[dict[str, Any]],
    scorer: Callable[[dict[str, Any]], list[float]],
    interventions: dict[str, dict[str, Callable[[dict[str, Any]], list[Any] | None]]],
) -> list[dict[str, Any]]:
    """Score rows under one callable while changing only their search inputs.

    Each intervention returns a search list or None when ineligible. The scorer
    receives a fresh row copy. Inputs must be JSON-compatible and the scorer
    must be deterministic and read-only.
    """
    if not interventions or any(not controls for controls in interventions.values()):
        raise ValueError("Provide at least one scope with at least one control.")
    for scope, controls in interventions.items():
        if not isinstance(scope, str) or not scope:
            raise ValueError("Scope names must be nonempty strings.")
        if any(
            not isinstance(name, str) or not name or name == "full" for name in controls
        ):
            raise ValueError("Control names must be nonempty and different from full.")
    output = []
    seen = set()

    def score(row):
        payload = copy.deepcopy(row)
        before = json.dumps(payload, sort_keys=True, allow_nan=False)
        values = scorer(payload)
        if json.dumps(payload, sort_keys=True, allow_nan=False) != before:
            raise ValueError(
                "The scorer mutated its row. Keep the model inputs read-only."
            )
        view_rank(values, row)
        return copy.deepcopy(values)

    for row in rows:
        validate_identity(row)
        if row["row_id"] in seen:
            raise ValueError("row_id must be unique in the model input.")
        seen.add(row["row_id"])
        if not isinstance(row.get("search"), list) or "history" not in row:
            raise ValueError("Each model row needs history and a search list.")
        full = score(row)
        for scope, controls in interventions.items():
            views = {"full": full}
            for name, intervention in controls.items():
                replacement = intervention(copy.deepcopy(row))
                views[name] = None
                if replacement is not None:
                    if not isinstance(replacement, list):
                        raise ValueError(
                            "An intervention must return a search list or None."
                        )
                    altered = copy.deepcopy(row)
                    altered["search"] = copy.deepcopy(replacement)
                    views[name] = score(altered)
            output.append(
                {
                    "row_id": row["row_id"],
                    "user_id": row["user_id"],
                    "target_id": row["target_id"],
                    "candidate_ids": copy.deepcopy(row["candidate_ids"]),
                    "scope": scope,
                    "views": copy.deepcopy(views),
                }
            )
    return output
