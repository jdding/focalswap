#!/usr/bin/env python3
"""Analyze GenSAR R244 row-level readouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


VIEWS = (
    "full",
    "no_source",
    "focal_query_mask",
    "focal_query_swap",
    "focal_session_swap",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=244)
    return parser.parse_args()


def ci(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def half_tie(full: float, cf: float) -> float:
    if full > cf:
        return 1.0
    if full < cf:
        return 0.0
    return 0.5


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    rows = payload["rows"]

    by_row: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        by_row[int(row["row_index"])][row["view"]] = row

    complete = {
        row_index: view_rows
        for row_index, view_rows in by_row.items()
        if all(view in view_rows for view in VIEWS)
    }
    if not complete:
        raise ValueError("no complete five-view rows")

    row_records = []
    for row_index, view_rows in sorted(complete.items()):
        full = float(view_rows["full"]["rr_at_beam"])
        record = {
            "row_index": row_index,
            "user_id": int(view_rows["full"]["user_id"]),
            "full_rr": full,
        }
        for view in VIEWS:
            record[f"{view}_rr"] = float(view_rows[view]["rr_at_beam"])
        for view in ("no_source", "focal_query_mask", "focal_query_swap", "focal_session_swap"):
            cf = float(view_rows[view]["rr_at_beam"])
            record[f"delta_full_minus_{view}"] = full - cf
            record[f"half_tie_full_vs_{view}"] = half_tie(full, cf)
        row_records.append(record)

    users: dict[int, list[dict[str, float]]] = defaultdict(list)
    for record in row_records:
        users[int(record["user_id"])].append(record)
    user_ids = sorted(users)
    rng = np.random.default_rng(args.seed)

    metrics = {}
    metric_names = []
    for view in VIEWS:
        metric_names.append(f"{view}_rr")
    for view in ("no_source", "focal_query_mask", "focal_query_swap", "focal_session_swap"):
        metric_names.append(f"delta_full_minus_{view}")
        metric_names.append(f"half_tie_full_vs_{view}")

    def aggregate(selected_users: list[int]) -> dict[str, float]:
        selected_records = [record for user in selected_users for record in users[user]]
        return {
            name: float(np.mean([float(record[name]) for record in selected_records]))
            for name in metric_names
        }

    point = aggregate(user_ids)
    boot_values: dict[str, list[float]] = {name: [] for name in metric_names}
    for _ in range(args.bootstrap_iters):
        sampled = list(rng.choice(user_ids, size=len(user_ids), replace=True))
        sampled_point = aggregate(sampled)
        for name, value in sampled_point.items():
            boot_values[name].append(value)

    for name in metric_names:
        values = np.array(boot_values[name], dtype=float)
        metrics[name] = {
            "point": point[name],
            "ci95": ci(values),
        }

    output = {
        "experiment_id": payload.get("experiment_id"),
        "source_file": str(args.input),
        "task": payload.get("task"),
        "claim_bearing": payload.get("claim_bearing"),
        "status": "PASS",
        "completed_rows": len(row_records),
        "user_clusters": len(user_ids),
        "views": list(VIEWS),
        "bootstrap": {
            "iters": args.bootstrap_iters,
            "seed": args.seed,
            "cluster": "user_id",
        },
        "metrics": metrics,
        "source_payload_status": payload.get("status"),
        "model": payload.get("model"),
        "model_provenance": payload.get("model_provenance"),
        "semantic_index": payload.get("semantic_index"),
        "collaborate_index": payload.get("collaborate_index"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
