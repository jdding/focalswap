#!/usr/bin/env python3
"""Shared tie-aware ranking for FocalSwap controlled-view readouts."""

from __future__ import annotations

import torch


def midpoint_rr_from_scores(scores: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return midpoint ranks and reciprocal ranks for target column zero."""
    if scores.ndim != 2 or scores.shape[1] <= 0:
        raise ValueError(f"expected nonempty [batch, candidates] scores, got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite scores cannot be ranked")
    out_rr: list[float] = []
    out_rank: list[float] = []
    for row in scores.detach().cpu():
        target_score = row[0]
        greater = int((row > target_score).sum().item())
        tied_other = int((row == target_score).sum().item()) - 1
        rank = 1.0 + greater + 0.5 * max(0, tied_other)
        out_rank.append(rank)
        out_rr.append(1.0 / rank)
    return out_rr, out_rank


def rank_rr_identity_pass(rr: list[float], rank: list[float], tolerance: float = 1e-12) -> bool:
    return bool(rr) and len(rr) == len(rank) and all(
        abs(float(rr_value) * float(rank_value) - 1.0) <= tolerance
        for rr_value, rank_value in zip(rr, rank)
    )
