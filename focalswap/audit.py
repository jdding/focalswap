"""Analyze fixed-candidate scores or ranks with user-cluster intervals."""

from __future__ import annotations

import math
from collections import defaultdict
from numbers import Real
from typing import Any

import numpy as np


def target_rank(scores: list[float], target_index: int) -> float:
    """Return the midpoint rank among candidates tied with the target."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Scores must be a nonempty, finite one-dimensional array.")
    if (
        isinstance(target_index, bool)
        or not isinstance(target_index, int)
        or not 0 <= target_index < len(values)
    ):
        raise ValueError("Target index is outside the candidate array.")
    target = values[target_index]
    return float(1 + np.sum(values > target) + 0.5 * (np.sum(values == target) - 1))


def validate_identity(row: dict[str, Any]) -> None:
    if not isinstance(row, dict):
        raise TypeError("Each row must be a JSON object.")
    for field in ("row_id", "user_id", "target_id"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"{field} must be a nonempty string.")
    candidates = row.get("candidate_ids")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_ids must be a nonempty list of strings.")
    if not all(isinstance(candidate, str) and candidate for candidate in candidates):
        raise ValueError("Candidate identifiers must be nonempty strings.")
    if len(set(candidates)) != len(candidates):
        raise ValueError("A row contains duplicate candidate identifiers.")
    if row["target_id"] not in candidates:
        raise ValueError("The target must occur in candidate_ids.")


def view_rank(view: Any, row: dict[str, Any]) -> float:
    candidates = row["candidate_ids"]
    if isinstance(view, dict) and set(view) == {"rank"}:
        rank = view["rank"]
        if isinstance(rank, bool) or not isinstance(rank, Real):
            raise ValueError("A target rank must be numeric.")
        if not math.isfinite(rank) or not 1 <= rank <= len(candidates) or rank * 2 % 1:
            raise ValueError(
                "Ranks must be integer or half-integer and within the slate."
            )
        return float(rank)
    if not isinstance(view, list) or len(view) != len(candidates):
        raise ValueError("Each score list must align with candidate_ids.")
    if not all(
        isinstance(value, Real) and not isinstance(value, bool) for value in view
    ):
        raise ValueError("Scores must be numeric.")
    return target_rank(view, candidates.index(row["target_id"]))


def summarize(values, users, names, iterations, seed, confidence):
    """Resample whole users, retaining row weighting for unequal cluster sizes."""
    unique_users, codes = np.unique(users, return_inverse=True)
    counts = np.bincount(codes).astype(float)
    sums = np.column_stack(
        [
            np.bincount(codes, weights=values[:, column], minlength=len(unique_users))
            for column in range(values.shape[1])
        ]
    )
    estimates = values.mean(axis=0)
    intervals = None
    if len(unique_users) >= 2:
        rng = np.random.default_rng(seed)
        draws = np.empty((iterations, len(names)))
        probabilities = np.full(len(unique_users), 1 / len(unique_users))
        for start in range(0, iterations, 256):
            stop = min(start + 256, iterations)
            weights = rng.multinomial(
                len(unique_users), probabilities, size=stop - start
            )
            draws[start:stop] = (weights @ sums) / (weights @ counts)[:, None]
        tail = (1 - confidence) / 2
        intervals = np.quantile(draws, [tail, 1 - tail], axis=0)
    metrics = {}
    for index, name in enumerate(names):
        metrics[name] = {
            "estimate": float(estimates[index]),
            "ci_low": None if intervals is None else float(intervals[0, index]),
            "ci_high": None if intervals is None else float(intervals[1, index]),
        }
    return {
        "rows": len(users),
        "user_clusters": len(unique_users),
        "interval_status": "available"
        if intervals is not None
        else "requires_at_least_two_users",
        "metrics": metrics,
    }


def analyze(
    readouts: list[dict[str, Any]],
    *,
    bootstrap_iterations: int = 10000,
    seed: int = 2026,
    confidence: float = 0.95,
    family_size: int = 1,
    comparisons: tuple[tuple[str, str], ...] = (("C0", "C2"),),
) -> dict[str, Any]:
    """Summarize each control and compare contributions on common eligible rows.

    Views contain aligned score lists or {"rank": midpoint_target_rank}; null
    marks an ineligible control. family_size applies a Bonferroni adjustment to
    every reported interval. A single user yields point estimates only.
    """
    if (
        not isinstance(bootstrap_iterations, int)
        or isinstance(bootstrap_iterations, bool)
        or bootstrap_iterations < 2
    ):
        raise ValueError("bootstrap_iterations must be an integer of at least two.")
    if (
        not isinstance(family_size, int)
        or isinstance(family_size, bool)
        or family_size < 1
    ):
        raise ValueError("family_size must be a positive integer.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between zero and one.")
    if not readouts:
        raise ValueError("At least one readout is required.")
    if len(set(comparisons)) != len(comparisons):
        raise ValueError("Paired comparisons must be unique.")
    for left, right in comparisons:
        if left == right or "full" in (left, right):
            raise ValueError("A paired comparison needs two distinct control names.")
    level = 1 - (1 - confidence) / family_size
    if not level < 1:
        raise ValueError(
            "Adjusted confidence is too close to one for floating-point precision."
        )
    grouped = defaultdict(list)
    declared = defaultdict(set)
    identities = {}
    seen = set()
    for row in readouts:
        validate_identity(row)
        scope = row.get("scope")
        if not isinstance(scope, str) or not scope:
            raise ValueError("scope must be a nonempty string.")
        key = (scope, row["row_id"])
        if key in seen:
            raise ValueError(f"Duplicate scope/row pair: {key}.")
        seen.add(key)
        views = row.get("views")
        if not isinstance(views, dict) or views.get("full") is None or len(views) < 2:
            raise ValueError("Each row needs full and at least one named control view.")
        if not all(isinstance(name, str) and name for name in views):
            raise ValueError("View names must be nonempty strings.")
        ranks = {
            name: view_rank(view, row)
            for name, view in views.items()
            if view is not None
        }
        identity = (
            row["user_id"],
            row["target_id"],
            row["candidate_ids"],
            views["full"],
        )
        if row["row_id"] in identities and identities[row["row_id"]] != identity:
            raise ValueError(
                "The same row changed users, target, candidates, or full scores across scopes."
            )
        identities[row["row_id"]] = identity
        grouped[scope].append((row["row_id"], row["user_id"], ranks))
        declared[scope].update(name for name in views if name != "full")

    result = {}
    for scope, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row[0])
        summaries = {}
        for control in sorted(declared[scope]):
            eligible = [(user, ranks) for _, user, ranks in rows if control in ranks]
            summary = {
                "rows": 0,
                "coverage": 0.0,
                "user_clusters": 0,
                "metrics": {},
                "interval_status": "no_eligible_rows",
            }
            if eligible:
                users = [user for user, _ in eligible]
                full = np.asarray([ranks["full"] for _, ranks in eligible])
                altered = np.asarray([ranks[control] for _, ranks in eligible])
                delta = 1 / full - 1 / altered
                values = np.column_stack(
                    (
                        1 / full,
                        1 / altered,
                        delta,
                        full != altered,
                        (delta > 0) + 0.5 * (delta == 0),
                        np.abs(full - altered),
                    )
                )
                summary.update(
                    summarize(
                        values,
                        users,
                        [
                            "full_rr",
                            "control_rr",
                            "signed_contribution",
                            "rank_response",
                            "replacement_preference",
                            "absolute_rank_shift",
                        ],
                        bootstrap_iterations,
                        seed,
                        level,
                    )
                )
                summary["coverage"] = len(users) / len(rows)
            summaries[control] = summary
        paired = []
        for left, right in comparisons:
            if left not in declared[scope] or right not in declared[scope]:
                continue
            eligible = [
                (user, ranks)
                for _, user, ranks in rows
                if left in ranks and right in ranks
            ]
            summary = {
                "left": left,
                "right": right,
                "rows": 0,
                "coverage": 0.0,
                "user_clusters": 0,
                "metrics": {},
                "interval_status": "no_eligible_rows",
            }
            if eligible:
                users = [user for user, _ in eligible]
                values = np.asarray(
                    [
                        [
                            1 / ranks["full"] - 1 / ranks[left],
                            1 / ranks["full"] - 1 / ranks[right],
                            1 / ranks[right] - 1 / ranks[left],
                        ]
                        for _, ranks in eligible
                    ]
                )
                summary.update(
                    summarize(
                        values,
                        users,
                        [
                            "left_contribution",
                            "right_contribution",
                            "contribution_difference",
                        ],
                        bootstrap_iterations,
                        seed,
                        level,
                    )
                )
                summary["coverage"] = len(users) / len(rows)
            paired.append(summary)
        result[scope] = {
            "total_rows": len(rows),
            "controls": summaries,
            "paired": paired,
        }
    return {
        "schema": "focalswap.audit.v1",
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
        "confidence": confidence,
        "family_size": family_size,
        "interval_confidence": level,
        "resampling_unit": "user",
        "mean_weighting": "row",
        "scopes": result,
    }
