#!/usr/bin/env python3
"""Compare all-source and latest-slot dependence on corrected R230A rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "r230a_anchor_repair_20260711"
DEFAULT_OUTPUT = ROOT / "reports" / "r232_dual_estimand_20260711"
EXPECTED_SEEDS = [20230601, 20230602, 20230603]
EXPECTED_ROWS = 157192
EXPECTED_USERS = 4852
EXPECTED_INPUT_HASH_FILES = {
    "rec_test.pkl",
    "item_vocab_np.pkl",
    "src_session_vocab_np.pkl",
    "user_vocab_np.pkl",
}
STRATA = ["1", "2-3", "4-7", "8-15", "16-29", "30"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=str(DEFAULT_INPUT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    p.add_argument("--bootstrap-seed", type=int, default=20260711)
    p.add_argument("--upstream-run-manifest", default="")
    p.add_argument("--upstream-unisar-manifest", default="")
    p.add_argument("--upstream-claim-decision", default="")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value)))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slot_stratum(n: int) -> str:
    if n == 1:
        return "1"
    if n <= 3:
        return "2-3"
    if n <= 7:
        return "4-7"
    if n <= 15:
        return "8-15"
    if n <= 29:
        return "16-29"
    if n == 30:
        return "30"
    raise ValueError(f"unexpected source_slot_count={n}")


def cluster_bootstrap(
    rows: list[dict[str, Any]],
    value_fn: Callable[[dict[str, Any]], float],
    iters: int,
    seed: int,
) -> dict[str, Any]:
    by_user: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_user[str(row["user_id"])].append(float(value_fn(row)))
    users = sorted(by_user)
    sums = np.asarray([sum(by_user[u]) for u in users], dtype=np.float64)
    counts = np.asarray([len(by_user[u]) for u in users], dtype=np.float64)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    boot = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, len(users), size=len(users))
        boot[i] = sums[idx].sum() / counts[idx].sum()
    low, high = np.percentile(boot, [2.5, 97.5])
    return {
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "rows": int(counts.sum()),
        "user_clusters": len(users),
        "bootstrap_iters": iters,
        "bootstrap_seed": seed,
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            full = float(row["full_sar_rr"])
            aggregate_masked = float(row["history_only_rr"])
            focal_masked = float(row["search_masked_rr"])
            slots = int(row["source_slot_count"])
            out.append(
                {
                    "row_index": int(row["row_index"]),
                    "official_row_index": int(row["official_row_index"]),
                    "example_id": row["example_id"],
                    "user_id": row["user_id"],
                    "source_slot_count": slots,
                    "stratum": slot_stratum(slots),
                    "full_rr": full,
                    "aggregate_masked_rr": aggregate_masked,
                    "focal_masked_rr": focal_masked,
                    "aggregate_delta": full - aggregate_masked,
                    "focal_delta": full - focal_masked,
                    "estimand_gap": focal_masked - aggregate_masked,
                }
            )
    return out


def validate_contract(
    readout: dict[str, Any],
    rows: list[dict[str, Any]],
    expected_checkpoint: str,
) -> list[str]:
    issues: list[str] = []
    if readout.get("status") != "PASS_OFFICIAL_PROCESSED_CHECKPOINT_READOUT":
        issues.append(f"bad status: {readout.get('status')}")
    expected_fields = {
        "schema": "focalswap.r208.unisar_official_processed_readout.v2",
        "surface_id": "K_CHECKPOINT_UNISAR_R208_OFFICIAL_PROCESSED",
        "contract_id": "R208_UNISAR_OFFICIAL_PROCESSED_CHECKPOINT_FOCALSWAP",
        "mode": "checkpoint_readout",
        "reference_source_mode": "true",
        "split": "test",
        "checkpoint": expected_checkpoint,
    }
    for key, expected in expected_fields.items():
        if readout.get(key) != expected:
            issues.append(f"{key}={readout.get(key)!r} expected {expected!r}")
    if int(readout.get("eval_rows", -1)) != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
        issues.append(f"row count mismatch: json={readout.get('eval_rows')} csv={len(rows)}")
    if readout.get("candidate_tensors_frozen") is not True:
        issues.append("candidate_tensors_frozen is not true")
    if readout.get("rank_rr_identity_pass") is not True:
        issues.append("rank_rr_identity_pass is not true")
    contract = readout.get("focal_only_contract", {})
    for key in [
        "declared_focal_matches_latest_source_slot",
        "search_masked_changes_exactly_one_slot",
        "history_only_changes_all_source_slots",
        "matched_swap_changes_exactly_one_or_zero_slots",
    ]:
        if contract.get(key) is not True:
            issues.append(f"focal_only_contract.{key} is not true")
    histogram = contract.get("source_slot_count_histogram", {})
    if not isinstance(histogram, dict) or sum(int(v) for v in histogram.values()) != EXPECTED_ROWS:
        issues.append("source-slot histogram does not cover the expected rows")
    if int(contract.get("rows_with_multiple_source_slots", 0)) <= 0:
        issues.append("no multi-source rows reported by the producer contract")
    users = len({str(row["user_id"]) for row in rows})
    if users != EXPECTED_USERS:
        issues.append(f"user clusters {users} != {EXPECTED_USERS}")
    identities = [
        (int(row["row_index"]), int(row["official_row_index"]), str(row["example_id"]))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        issues.append("row identities are not unique")
    if identities != sorted(identities, key=lambda x: x[0]):
        issues.append("row identities are not ordered by row_index")
    one_slot = [row for row in rows if int(row["source_slot_count"]) == 1]
    if not one_slot:
        issues.append("no one-slot rows for equivalence check")
    elif any(abs(float(row["estimand_gap"])) > 1e-15 for row in one_slot):
        issues.append("aggregate and focal masked RR differ on one-slot rows")
    if any(int(row["source_slot_count"]) < 1 for row in rows):
        issues.append("row with fewer than one source slot")
    means = {
        "full_sar_rr": float(np.mean([row["full_rr"] for row in rows])),
        "history_only_rr": float(np.mean([row["aggregate_masked_rr"] for row in rows])),
        "search_masked_rr": float(np.mean([row["focal_masked_rr"] for row in rows])),
    }
    for key, actual in means.items():
        expected = float(readout.get("metrics", {}).get(key, math.nan))
        if not math.isfinite(expected) or abs(actual - expected) > 1e-12:
            issues.append(f"CSV/JSON mean mismatch for {key}: {actual} vs {expected}")
    return issues


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def upstream_file_index(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    if payload.get("schema") != "focalswap.r230a.run_manifest.v1":
        raise RuntimeError(f"unexpected upstream run schema: {payload.get('schema')}")
    if payload.get("status") != "FULL_RUN_COMPLETE_PENDING_LOCAL_PULLBACK_AND_CLAIM_DECISION":
        raise RuntimeError(f"unexpected upstream run status: {payload.get('status')}")
    rows = payload.get("files", [])
    basenames = [Path(row["path"]).name for row in rows]
    if len(set(basenames)) != len(basenames):
        raise RuntimeError("duplicate basenames in upstream run manifest")
    return {name: row for name, row in zip(basenames, rows)}


def verify_upstream_file(path: Path, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = index.get(path.name)
    if expected is None:
        raise RuntimeError(f"{path.name} absent from upstream run manifest")
    actual = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    if int(expected["bytes"]) != actual["bytes"] or expected["sha256"] != actual["sha256"]:
        raise RuntimeError(f"upstream hash/size mismatch for {path.name}")
    return actual


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    producer_path = ROOT / "tools" / "unisar_five_view_readout.py"
    view_path = ROOT / "tools" / "focal_source_views.py"
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    upstream_run_manifest = Path(args.upstream_run_manifest).resolve() if args.upstream_run_manifest else input_dir / "r230a_run_manifest.json"
    upstream_unisar_manifest = Path(args.upstream_unisar_manifest).resolve() if args.upstream_unisar_manifest else input_dir / "r230a_unisar_manifest.jsonl"
    upstream_claim_decision = Path(args.upstream_claim_decision).resolve() if args.upstream_claim_decision else ROOT / "refine-logs" / "R230A_CLAIM_DECISION_20260711.json"
    if not upstream_run_manifest.is_file() or not upstream_unisar_manifest.is_file() or not upstream_claim_decision.is_file():
        raise FileNotFoundError("missing R230A upstream manifests")
    run_index = upstream_file_index(upstream_run_manifest)
    verified_unisar_manifest = verify_upstream_file(upstream_unisar_manifest, run_index)
    unisar_manifest_rows = load_jsonl(upstream_unisar_manifest)
    if len(unisar_manifest_rows) != len(EXPECTED_SEEDS):
        raise RuntimeError("unexpected R230A UniSAR manifest row count")
    claim_decision = read_json(upstream_claim_decision)
    if claim_decision.get("schema") != "focalswap.r230a.claim_decision.v1":
        raise RuntimeError("unexpected R230A claim-decision schema")
    if claim_decision.get("source_manifest_hash_verification", {}).get("status") != "PASS":
        raise RuntimeError("R230A source-manifest hash verification did not pass")
    if claim_decision.get("r230b", {}).get("decision") != "NOT_LAUNCHED_CONSERVATIVE_CLOSURE":
        raise RuntimeError("unexpected historical R230B decision")
    executed_provenance = claim_decision.get("executed_provenance", {})
    checkpoint_hashes = executed_provenance.get("unisar_checkpoint_sha256", {})
    input_hashes = executed_provenance.get("unisar_test_input_sha256", {})
    if sorted(int(seed) for seed in checkpoint_hashes) != EXPECTED_SEEDS:
        raise RuntimeError("R230A checkpoint hash set does not match expected seeds")
    if set(input_hashes) != EXPECTED_INPUT_HASH_FILES:
        raise RuntimeError("R230A test-input hash file set is incomplete or unexpected")
    evaluator_hash = executed_provenance.get("unisar_evaluator_sha256")
    if any(not valid_sha256(value) for value in [*checkpoint_hashes.values(), *input_hashes.values(), evaluator_hash]):
        raise RuntimeError("R230A checkpoint/input provenance hashes are incomplete")
    if evaluator_hash != sha256(producer_path):
        raise RuntimeError("R230A evaluator hash does not match the current UniSAR producer")
    manifest_by_seed = {int(row["model_seed"]): row for row in unisar_manifest_rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = [verified_unisar_manifest]
    baseline_identity: list[tuple[int, int, str, str, int]] | None = None

    for pos, model_seed in enumerate(EXPECTED_SEEDS):
        prefix = f"r230a_unisar_seed_{model_seed}_true"
        readout_path = input_dir / f"{prefix}_readout.json"
        scores_path = input_dir / f"{prefix}_scores.csv"
        if not readout_path.is_file() or not scores_path.is_file():
            raise FileNotFoundError(f"missing corrected R230A input for seed {model_seed}")
        manifest_row = manifest_by_seed.get(model_seed)
        if manifest_row is None:
            raise RuntimeError(f"seed {model_seed} absent from R230A UniSAR manifest")
        if manifest_row.get("schema") != "focalswap.r230a.unisar_manifest_row.v1" or manifest_row.get("contract_id") != "R230A_ANCHOR_ESTIMAND_REPAIR":
            raise RuntimeError(f"bad R230A manifest contract for seed {model_seed}")
        if Path(str(manifest_row["true_readout_json"])).name != readout_path.name or Path(str(manifest_row["true_scores_csv"])).name != scores_path.name:
            raise RuntimeError(f"R230A manifest path mismatch for seed {model_seed}")
        readout = read_json(readout_path)
        rows = load_rows(scores_path)
        issues = validate_contract(readout, rows, str(manifest_row["checkpoint"]))
        if issues:
            raise RuntimeError(f"seed {model_seed} contract failed: {issues}")
        identity = [
            (
                int(row["row_index"]),
                int(row["official_row_index"]),
                str(row["example_id"]),
                str(row["user_id"]),
                int(row["source_slot_count"]),
            )
            for row in rows
        ]
        if baseline_identity is None:
            baseline_identity = identity
        elif identity != baseline_identity:
            raise RuntimeError(f"cross-seed row identity mismatch for seed {model_seed}")
        inputs.extend([verify_upstream_file(readout_path, run_index), verify_upstream_file(scores_path, run_index)])
        metrics: dict[str, dict[str, Any]] = {}
        for offset, (name, fn) in enumerate(
            [
                ("aggregate_dependence", lambda r: float(r["aggregate_delta"])),
                ("focal_dependence", lambda r: float(r["focal_delta"])),
                ("estimand_gap", lambda r: float(r["estimand_gap"])),
            ]
        ):
            metrics[name] = cluster_bootstrap(
                rows,
                fn,
                args.bootstrap_iters,
                args.bootstrap_seed + pos * 1009 + offset,
            )
        summary_rows.append(
            {
                "model_seed": model_seed,
                "rows": len(rows),
                "user_clusters": metrics["aggregate_dependence"]["user_clusters"],
                "full_rr": float(readout["metrics"]["full_sar_rr"]),
                "aggregate_masked_rr": float(readout["metrics"]["history_only_rr"]),
                "focal_masked_rr": float(readout["metrics"]["search_masked_rr"]),
                "aggregate_delta": metrics["aggregate_dependence"]["point"],
                "aggregate_ci_low": metrics["aggregate_dependence"]["ci_low"],
                "aggregate_ci_high": metrics["aggregate_dependence"]["ci_high"],
                "focal_delta": metrics["focal_dependence"]["point"],
                "focal_ci_low": metrics["focal_dependence"]["ci_low"],
                "focal_ci_high": metrics["focal_dependence"]["ci_high"],
                "estimand_gap": metrics["estimand_gap"]["point"],
                "gap_ci_low": metrics["estimand_gap"]["ci_low"],
                "gap_ci_high": metrics["estimand_gap"]["ci_high"],
                "aggregate_positive": metrics["aggregate_dependence"]["ci_low"] > 0.0,
                "focal_unresolved": metrics["focal_dependence"]["ci_low"] <= 0.0 <= metrics["focal_dependence"]["ci_high"],
                "gap_positive": metrics["estimand_gap"]["ci_low"] > 0.0,
                "contract_pass": True,
            }
        )
        for stratum_pos, stratum in enumerate(STRATA):
            sub = [row for row in rows if row["stratum"] == stratum]
            if not sub:
                continue
            agg = cluster_bootstrap(sub, lambda r: float(r["aggregate_delta"]), args.bootstrap_iters, args.bootstrap_seed + pos * 1009 + 100 + stratum_pos)
            focal = cluster_bootstrap(sub, lambda r: float(r["focal_delta"]), args.bootstrap_iters, args.bootstrap_seed + pos * 1009 + 200 + stratum_pos)
            gap = cluster_bootstrap(sub, lambda r: float(r["estimand_gap"]), args.bootstrap_iters, args.bootstrap_seed + pos * 1009 + 300 + stratum_pos)
            strata_rows.append(
                {
                    "model_seed": model_seed,
                    "source_slots": stratum,
                    "rows": len(sub),
                    "user_clusters": gap["user_clusters"],
                    "aggregate_delta": agg["point"],
                    "aggregate_ci_low": agg["ci_low"],
                    "aggregate_ci_high": agg["ci_high"],
                    "focal_delta": focal["point"],
                    "focal_ci_low": focal["ci_low"],
                    "focal_ci_high": focal["ci_high"],
                    "estimand_gap": gap["point"],
                    "gap_ci_low": gap["ci_low"],
                    "gap_ci_high": gap["ci_high"],
                }
            )

    stable = all(
        bool(row["aggregate_positive"])
        and bool(row["focal_unresolved"])
        and bool(row["gap_positive"])
        for row in summary_rows
    )
    status = (
        "PASS_NUMERICAL_PROVENANCE_INTERPRETATION_SUPERSEDED_R233"
        if stable
        else "CLOSED_MIXED_NEUTRAL_MASK_RESPONSES"
    )
    payload = {
        "schema": "focalswap.r232.dual_estimand_summary.v1",
        "status": status,
        "contract_id": "R232_DUAL_ESTIMAND_SOURCE_DEPENDENCE",
        "surface_id": "K_CHECKPOINT_UNISAR_R208_OFFICIAL_PROCESSED",
        "estimands": {
            "aggregate_dependence": "legacy key: Full RR - all-source neutral-mask RR; intervention response only",
            "focal_dependence": "Full RR - latest-source-slot-masked RR",
            "estimand_gap": "all-source neutral-mask response - latest-slot neutral-mask response",
        },
        "promotion_rule": "historical numerical rule: all three seeds have all-source neutral-mask CI low > 0, latest-slot CI contains 0, and paired gap CI low > 0; R233 supersedes the accumulated-use interpretation",
        "bootstrap_unit": "user",
        "bootstrap_iters": args.bootstrap_iters,
        "model_seeds": EXPECTED_SEEDS,
        "rows": summary_rows,
        "upstream_execution_provenance": {
            "unisar_checkpoint_sha256": checkpoint_hashes,
            "unisar_test_input_sha256": input_hashes,
            "unisar_evaluator_sha256": evaluator_hash,
            "historical_r230b_decision": "NOT_LAUNCHED_CONSERVATIVE_CLOSURE",
            "r232_scope": "post-R230A derived dual-estimand audit, distinct from R230B surface-factor localization",
        },
        "descriptive_across_seeds": {
            name: {
                "mean": float(np.mean([float(row[name]) for row in summary_rows])),
                "min": min(float(row[name]) for row in summary_rows),
                "max": max(float(row[name]) for row in summary_rows),
            }
            for name in ["aggregate_delta", "focal_delta", "estimand_gap"]
        },
        "slot_strata_are_diagnostic_only": True,
        "claim_ceiling": (
            "Historical all-source neutral-mask response only. R233 shows that the positive all-source margin does not survive one observed-support sensitivity control; "
            "no accumulated-use, matched-specificity, or architecture claim."
            if stable
            else "Mixed neutral-mask intervention responses; no stable dependence, accumulated-use, matched-specificity, or architecture claim."
        ),
        "inputs": inputs,
    }
    summary_json = output_dir / "r232_dual_estimand_summary.json"
    summary_csv = output_dir / "r232_dual_estimand_summary.csv"
    strata_csv = output_dir / "r232_slot_strata.csv"
    report_md = output_dir / "r232_dual_estimand_report.md"
    write_csv(summary_csv, summary_rows)
    write_csv(strata_csv, strata_rows)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# R232 Dual-Estimand Neutral-Mask Scope Audit",
        "",
        f"Status: `{status}`",
        "",
        "| Seed | Full RR | All-source masked RR | Focal-masked RR | All-source response [95% CI] | Focal response [95% CI] | Paired gap [95% CI] |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_seed']} | {row['full_rr']:.6f} | {row['aggregate_masked_rr']:.6f} | {row['focal_masked_rr']:.6f} | "
            f"{row['aggregate_delta']:.6f} [{row['aggregate_ci_low']:.6f}, {row['aggregate_ci_high']:.6f}] | "
            f"{row['focal_delta']:.6f} [{row['focal_ci_low']:.6f}, {row['focal_ci_high']:.6f}] | "
            f"{row['estimand_gap']:.6f} [{row['gap_ci_low']:.6f}, {row['gap_ci_high']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Descriptive Across-Seed Summary",
            "",
            "These are descriptive mean and range values across the three retained checkpoints, not a seed-level significance test.",
            "",
            "| Estimand | Mean | Range |",
            "|---|---:|---:|",
        ]
    )
    for label, name in [
        ("All-source neutral-mask response", "aggregate_delta"),
        ("Focal dependence", "focal_delta"),
        ("Paired estimand gap", "estimand_gap"),
    ]:
        desc = payload["descriptive_across_seeds"][name]
        lines.append(
            f"| {label} | {desc['mean']:.6f} | [{desc['min']:.6f}, {desc['max']:.6f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["claim_ceiling"],
            "",
            "Slot-count strata are saved as diagnostics and were not used for claim promotion.",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    output_files = [summary_json, summary_csv, strata_csv, report_md]
    code_path = Path(__file__).resolve()
    plan_path = ROOT / "refine-logs" / "EXPERIMENT_PLAN_R232_DUAL_ESTIMAND_20260711.md"
    gate_path = ROOT / "refine-logs" / "PLAN_TO_RUN_GATE_R232_DUAL_ESTIMAND_20260711.json"
    manifest = {
        "schema": "focalswap.r232.run_manifest.v1",
        "status": "PASS_R232_OUTPUT_MANIFEST",
        "inputs": inputs,
        "provenance": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [
                code_path,
                producer_path,
                view_path,
                plan_path,
                gate_path,
                upstream_run_manifest,
                upstream_unisar_manifest,
                upstream_claim_decision,
            ]
        ],
        "parameters": {
            "bootstrap_iters": args.bootstrap_iters,
            "bootstrap_seed": args.bootstrap_seed,
            "expected_seeds": EXPECTED_SEEDS,
            "expected_rows": EXPECTED_ROWS,
            "expected_users": EXPECTED_USERS,
        },
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        ],
    }
    (output_dir / "r232_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": status, "output_dir": str(output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
