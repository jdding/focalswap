#!/usr/bin/env python3
"""Pure tensor transforms for focal-only UniSAR source interventions."""

from __future__ import annotations

from typing import Any

import torch


def clone_tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def source_slot_masks(batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all non-padding source slots and the latest source slot per row."""
    all_his = batch["all_his"]
    all_his_type = batch["all_his_type"]
    if all_his.ndim != 2 or all_his_type.shape != all_his.shape:
        raise ValueError(
            "expected aligned [batch, history] all_his/all_his_type tensors, "
            f"got {tuple(all_his.shape)} and {tuple(all_his_type.shape)}"
        )
    source_mask = (all_his_type == 2) & (all_his != 0)
    positions = torch.arange(all_his.shape[1], device=all_his.device).reshape(1, -1)
    positions = positions.expand_as(all_his)
    latest_position = torch.where(
        source_mask,
        positions,
        torch.full_like(positions, -1),
    ).max(dim=1).values
    focal_mask = source_mask & (positions == latest_position.reshape(-1, 1))
    return source_mask, focal_mask


def source_slot_counts(batch: dict[str, Any]) -> torch.Tensor:
    source_mask, _ = source_slot_masks(batch)
    return source_mask.sum(dim=1)


def focal_source_ids(batch: dict[str, Any]) -> torch.Tensor:
    all_his = batch["all_his"]
    _, focal_mask = source_slot_masks(batch)
    return torch.where(focal_mask, all_his, torch.zeros_like(all_his)).sum(dim=1)


def replace_source_view(
    batch: dict[str, Any],
    view: str,
    neutral_src_id: int,
    neutral_item_id: int,
) -> dict[str, Any]:
    """Construct UniSAR views while keeping pre-focal source history fixed."""
    out = clone_tensor_batch(batch)
    all_his = out["all_his"]
    all_his_type = out["all_his_type"]
    source_mask, focal_mask = source_slot_masks(out)
    if view == "full_sar":
        return out
    if view == "history_only":
        out["all_his"] = all_his.masked_fill(source_mask, neutral_src_id)
    elif view == "search_masked":
        out["all_his"] = all_his.masked_fill(focal_mask, neutral_src_id)
    elif view == "search_only":
        rec_mask = (all_his_type == 1) & (all_his != 0)
        prior_source_mask = source_mask & ~focal_mask
        # Keep one computable recommendation-history structure. The official
        # runner selects an existing non-padding item row whose metadata are all
        # zero, so this intervention never expands or changes checkpoint weights.
        changed = all_his.masked_fill(rec_mask, neutral_item_id)
        out["all_his"] = changed.masked_fill(prior_source_mask, neutral_src_id)
    else:
        raise ValueError(view)
    return out


def replace_matched_focal_source(batch: dict[str, Any], replacement: torch.Tensor) -> dict[str, Any]:
    """Replace exactly the latest source slot when a matched donor exists."""
    out = clone_tensor_batch(batch)
    all_his = out["all_his"]
    if replacement.ndim != 1 or replacement.numel() != all_his.shape[0]:
        raise ValueError(
            "replacement must contain exactly one source id per batch row; "
            f"got shape {tuple(replacement.shape)} for batch {all_his.shape[0]}"
        )
    _, focal_mask = source_slot_masks(out)
    row_replacement = replacement.to(all_his.device).reshape(-1, 1).expand_as(all_his)
    matched_mask = row_replacement > 0
    out["all_his"] = torch.where(focal_mask & matched_mask, row_replacement, all_his)
    return out


def changed_slot_counts(before: dict[str, Any], after: dict[str, Any]) -> torch.Tensor:
    return (before["all_his"] != after["all_his"]).sum(dim=1)
