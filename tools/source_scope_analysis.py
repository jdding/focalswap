#!/usr/bin/env python3
"""User-cluster bootstrap and frozen claim decision for R233."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from unisar_source_scope_localization import (
    EXPECTED_OFFICIAL_UNISAR_COMMIT,
    MODES,
    SCOPE_SPECS,
    VARIANTS,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = (20230601, 20230602, 20230603)
EXPECTED_ROWS = 157192
EXPECTED_USERS = 4852


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--r230a-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260711)
    parser.add_argument("--bootstrap-chunk-size", type=int, default=128)
    parser.add_argument("--validate-only-seed", type=int, choices=EXPECTED_SEEDS)
    parser.add_argument(
        "--r230a-claim-decision",
        default=str(ROOT / "refine-logs" / "R230A_CLAIM_DECISION_20260711.json"),
    )
    parser.add_argument("--r230a-run-manifest", default="")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def basename_hashes(values: dict[str, str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for path, digest in values.items():
        name = Path(path).name
        if name in output and output[name] != digest:
            raise RuntimeError(f"ambiguous basename hash for {name}")
        output[name] = str(digest)
    return output


def validate_execution_lineage(
    seed: int,
    integrity: dict[str, Any],
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    if int(integrity.get("intervention_seed", -1)) != 20260711:
        raise RuntimeError(f"unexpected intervention seed for model seed {seed}")
    if tuple(integrity.get("variants", [])) != VARIANTS:
        raise RuntimeError(f"variant contract mismatch for model seed {seed}")
    lineage = integrity.get("execution_lineage", {})
    if lineage.get("official_git_commit") != EXPECTED_OFFICIAL_UNISAR_COMMIT:
        raise RuntimeError(f"official UniSAR commit mismatch for model seed {seed}")
    if lineage.get("official_reviewed_files_clean") is not True:
        raise RuntimeError(f"official UniSAR reviewed files are not clean for model seed {seed}")
    expected_checkpoint = str(expected_provenance["unisar_checkpoint_sha256"][str(seed)])
    actual_checkpoint = str(lineage.get("checkpoint_sha256", ""))
    if actual_checkpoint != expected_checkpoint or integrity.get("checkpoint_sha256") != expected_checkpoint:
        raise RuntimeError(f"checkpoint lineage mismatch for model seed {seed}")
    processed = basename_hashes(lineage.get("processed_input_sha256", {}))
    expected_inputs = expected_provenance["unisar_test_input_sha256"]
    for name, digest in expected_inputs.items():
        if processed.get(name) != digest:
            raise RuntimeError(f"processed input hash mismatch for {name}, seed {seed}")
    local_executed = basename_hashes(lineage.get("executed_local_code_sha256", {}))
    reviewed_paths = [
        ROOT / "tools" / "unisar_source_scope_localization.py",
        ROOT / "tools" / "unisar_five_view_readout.py",
        ROOT / "tools" / "focal_source_views.py",
        ROOT / "tools" / "controlled_view_ranking.py",
        ROOT / "tools" / "official_adapter.py",
    ]
    reviewed = {path.name: sha256_file(path) for path in reviewed_paths}
    if local_executed != reviewed:
        raise RuntimeError(
            f"executed/reviewed local code mismatch for seed {seed}: "
            f"executed={local_executed}, reviewed={reviewed}"
        )
    donor = integrity.get("donor_map", {})
    required_donor_checks = {
        "bijection": True,
        "all_donors_observed_and_nonempty": True,
        "future_label_or_score_used": False,
    }
    for key, expected in required_donor_checks.items():
        if donor.get(key) is not expected:
            raise RuntimeError(f"donor-map contract {key} failed for seed {seed}")
    if int(donor.get("fixed_points", -1)) != 0 or int(donor.get("padding_or_neutral_donors", -1)) != 0:
        raise RuntimeError(f"donor-map fixed/invalid IDs for seed {seed}")
    for key in ["observed_support_sha256", "donor_map_sha256"]:
        value = str(donor.get(key, ""))
        if len(value) != 64:
            raise RuntimeError(f"missing donor lineage {key} for seed {seed}")
    return {
        "checkpoint_sha256": actual_checkpoint,
        "processed_input_sha256": processed,
        "official_code_sha256": basename_hashes(lineage.get("official_code_sha256", {})),
        "official_git_commit": lineage["official_git_commit"],
        "executed_local_code_sha256": local_executed,
        "observed_support_sha256": donor["observed_support_sha256"],
        "donor_map_sha256": donor["donor_map_sha256"],
    }


def scope_eligible(scope: str, source_counts: np.ndarray) -> np.ndarray:
    kind, amount = SCOPE_SPECS[scope]
    if kind == "all":
        return source_counts >= 1
    return source_counts >= amount


def add_metric(
    names: list[str],
    sums: list[np.ndarray],
    counts: list[np.ndarray],
    metadata: list[dict[str, Any]],
    name: str,
    values: np.ndarray,
    eligible: np.ndarray,
    user_codes: np.ndarray,
    user_count: int,
    meta: dict[str, Any],
) -> None:
    if values.shape != eligible.shape or values.shape != user_codes.shape:
        raise RuntimeError(f"shape mismatch while adding metric {name}")
    if not np.isfinite(values[eligible]).all():
        raise RuntimeError(f"non-finite values in metric {name}")
    metric_sums = np.bincount(
        user_codes[eligible],
        weights=values[eligible],
        minlength=user_count,
    ).astype(np.float64)
    metric_counts = np.bincount(
        user_codes[eligible],
        minlength=user_count,
    ).astype(np.float64)
    if metric_counts.sum() <= 0:
        raise RuntimeError(f"metric {name} has no eligible rows")
    names.append(name)
    sums.append(metric_sums)
    counts.append(metric_counts)
    metadata.append(
        {
            **meta,
            "metric_name": name,
            "rows": int(metric_counts.sum()),
            "user_clusters": int(np.count_nonzero(metric_counts)),
        }
    )


def bootstrap_metrics(
    sums: np.ndarray,
    counts: np.ndarray,
    iterations: int,
    seed: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sums.shape != counts.shape or sums.ndim != 2:
        raise RuntimeError("bootstrap input matrices must be aligned [users, metrics]")
    user_count, metric_count = sums.shape
    points = sums.sum(axis=0) / counts.sum(axis=0)
    draws = np.empty((iterations, metric_count), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(user_count, 1.0 / user_count, dtype=np.float64)
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        weights = rng.multinomial(user_count, probabilities, size=stop - start)
        denominators = weights @ counts
        if bool((denominators <= 0).any()):
            raise RuntimeError("zero bootstrap denominator")
        draws[start:stop] = (weights @ sums) / denominators
    lows, highs = np.percentile(draws, [2.5, 97.5], axis=0)
    return points, lows, highs


def validate_r230a_reproduction(
    seed: int,
    frame: pd.DataFrame,
    r230a_dir: Path,
    r230a_manifest_index: dict[str, dict[str, Any]],
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    path = r230a_dir / f"r230a_unisar_seed_{seed}_true_scores.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest_row = r230a_manifest_index.get(path.name)
    if manifest_row is None:
        raise RuntimeError(f"{path.name} is absent from the R230A run manifest")
    actual_hash = sha256_file(path)
    if actual_hash != manifest_row.get("sha256") or path.stat().st_size != int(manifest_row.get("bytes", -1)):
        raise RuntimeError(f"R230A run-manifest hash/size mismatch for {path.name}")
    reference = pd.read_csv(
        path,
        usecols=[
            "official_row_index",
            "user_id",
            "full_sar_rr",
            "history_only_rr",
            "search_masked_rr",
        ],
    )
    if len(reference) != len(frame):
        raise RuntimeError(f"R230A/R233 row mismatch for seed {seed}")
    if not np.array_equal(
        reference["official_row_index"].to_numpy(),
        frame["official_row_index"].to_numpy(),
    ) or not np.array_equal(reference["user_id"].to_numpy(), frame["user_id"].to_numpy()):
        raise RuntimeError(f"R230A/R233 row identity mismatch for seed {seed}")
    comparisons = {
        "full": (frame["full_rr"].to_numpy(), reference["full_sar_rr"].to_numpy()),
        "neutral_latest_1": (
            frame["neutral_latest_1_rr"].to_numpy(),
            reference["search_masked_rr"].to_numpy(),
        ),
        "neutral_all": (
            frame["neutral_all_rr"].to_numpy(),
            reference["history_only_rr"].to_numpy(),
        ),
    }
    result: dict[str, Any] = {}
    for name, (actual, expected) in comparisons.items():
        absolute = np.abs(actual - expected)
        maximum = float(absolute.max())
        mismatch_count = int(np.count_nonzero(absolute > tolerance))
        result[name] = {
            "max_abs_difference": maximum,
            "mismatch_count_above_tolerance": mismatch_count,
            "tolerance": tolerance,
            "pass": mismatch_count == 0,
        }
        if mismatch_count:
            raise RuntimeError(
                f"R230A reproduction failed for seed={seed} view={name}: "
                f"max={maximum}, mismatches={mismatch_count}"
            )
    result["r230a_scores_sha256"] = actual_hash
    result["r230a_run_manifest_bound"] = True
    return result


def analyze_seed(
    seed: int,
    frame: pd.DataFrame,
    iterations: int,
    bootstrap_seed: int,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(frame) != EXPECTED_ROWS:
        raise RuntimeError(f"seed {seed} rows={len(frame)} expected={EXPECTED_ROWS}")
    if set(frame["candidate_count"].astype(int)) != {100}:
        raise RuntimeError(f"seed {seed} candidate count is not fixed at 100")
    user_ids, user_codes = np.unique(frame["user_id"].to_numpy(), return_inverse=True)
    if len(user_ids) != EXPECTED_USERS:
        raise RuntimeError(f"seed {seed} users={len(user_ids)} expected={EXPECTED_USERS}")
    source_counts = frame["source_slot_count"].to_numpy(dtype=np.int64)
    full = frame["full_rr"].to_numpy(dtype=np.float64)

    names: list[str] = []
    sums: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for mode in MODES:
        for scope in SCOPE_SPECS:
            variant = f"{mode}_{scope}"
            variant_rr = frame[f"{variant}_rr"].to_numpy(dtype=np.float64)
            eligible = scope_eligible(scope, source_counts)
            delta = full - variant_rr
            half_tie = np.where(full > variant_rr, 1.0, np.where(full == variant_rr, 0.5, 0.0))
            wins = int(np.count_nonzero((full > variant_rr) & eligible))
            ties = int(np.count_nonzero((full == variant_rr) & eligible))
            losses = int(np.count_nonzero((full < variant_rr) & eligible))
            eligible_rows = int(np.count_nonzero(eligible))
            comparison_stats = {
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "strict_win": float(wins / eligible_rows),
                "tie_rate": float(ties / eligible_rows),
                "loss_rate": float(losses / eligible_rows),
            }
            add_metric(
                names,
                sums,
                counts,
                metadata,
                f"delta_rr__{mode}__{scope}",
                delta,
                eligible,
                user_codes,
                len(user_ids),
                {
                    "family": "scope",
                    "readout": "delta_rr",
                    "mode": mode,
                    "scope": scope,
                    **comparison_stats,
                },
            )
            add_metric(
                names,
                sums,
                counts,
                metadata,
                f"half_tie__{mode}__{scope}",
                half_tie,
                eligible,
                user_codes,
                len(user_ids),
                {
                    "family": "scope",
                    "readout": "half_tie",
                    "mode": mode,
                    "scope": scope,
                    **comparison_stats,
                },
            )

    all_rows = source_counts >= 1
    common_eight = source_counts >= 8
    for scope in SCOPE_SPECS:
        eligible = scope_eligible(scope, source_counts)
        neutral = frame[f"neutral_{scope}_rr"].to_numpy(dtype=np.float64)
        valid = frame[f"valid_{scope}_rr"].to_numpy(dtype=np.float64)
        add_metric(
            names,
            sums,
            counts,
            metadata,
            f"contrast_rr__neutral_minus_valid__{scope}",
            valid - neutral,
            eligible,
            user_codes,
            len(user_ids),
            {"family": "contrast", "contrast": "neutral_minus_valid", "scope": scope},
        )

    for mode in MODES:
        latest = frame[f"{mode}_latest_1_rr"].to_numpy(dtype=np.float64)
        all_scope = frame[f"{mode}_all_rr"].to_numpy(dtype=np.float64)
        add_metric(
            names,
            sums,
            counts,
            metadata,
            f"contrast_rr__{mode}_all_minus_latest",
            latest - all_scope,
            all_rows,
            user_codes,
            len(user_ids),
            {"family": "contrast", "contrast": "all_minus_latest", "mode": mode},
        )
        recent_eight = frame[f"{mode}_recent_8_rr"].to_numpy(dtype=np.float64)
        add_metric(
            names,
            sums,
            counts,
            metadata,
            f"contrast_rr__{mode}_recent8_minus_latest",
            latest - recent_eight,
            common_eight,
            user_codes,
            len(user_ids),
            {"family": "contrast", "contrast": "recent8_minus_latest", "mode": mode},
        )
        for amount in (2, 4, 8):
            recent = frame[f"{mode}_recent_{amount}_rr"].to_numpy(dtype=np.float64)
            random = frame[f"{mode}_random_{amount}_rr"].to_numpy(dtype=np.float64)
            add_metric(
                names,
                sums,
                counts,
                metadata,
                f"contrast_rr__{mode}_recent_minus_random_{amount}",
                random - recent,
                source_counts >= amount,
                user_codes,
                len(user_ids),
                {
                    "family": "contrast",
                    "contrast": "recent_minus_random",
                    "mode": mode,
                    "amount": amount,
                },
            )
        dose_scopes = ["latest_1", "recent_2", "recent_4", "recent_8"]
        for lower, upper in zip(dose_scopes, dose_scopes[1:]):
            lower_rr = frame[f"{mode}_{lower}_rr"].to_numpy(dtype=np.float64)
            upper_rr = frame[f"{mode}_{upper}_rr"].to_numpy(dtype=np.float64)
            add_metric(
                names,
                sums,
                counts,
                metadata,
                f"contrast_rr__{mode}_dose_{upper}_minus_{lower}",
                lower_rr - upper_rr,
                common_eight,
                user_codes,
                len(user_ids),
                {
                    "family": "contrast",
                    "contrast": "dose_increment_common_ge8",
                    "mode": mode,
                    "lower": lower,
                    "upper": upper,
                },
            )

    sum_matrix = np.column_stack(sums)
    count_matrix = np.column_stack(counts)
    points, lows, highs = bootstrap_metrics(
        sum_matrix,
        count_matrix,
        iterations,
        bootstrap_seed,
        chunk_size,
    )
    scope_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for index, meta in enumerate(metadata):
        row = {
            "model_seed": seed,
            **meta,
            "point": float(points[index]),
            "ci_low": float(lows[index]),
            "ci_high": float(highs[index]),
            "bootstrap_iters": iterations,
            "bootstrap_seed": bootstrap_seed,
        }
        if meta["family"] == "scope":
            scope_rows.append(row)
        else:
            contrast_rows.append(row)
    return scope_rows, contrast_rows


def metric_index(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
    return {(int(row["model_seed"]), str(row["metric_name"])): row for row in rows}


def frozen_decision(
    scope_rows: list[dict[str, Any]],
    contrast_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    index = metric_index([*scope_rows, *contrast_rows])

    def rows_for(metric_name: str) -> list[dict[str, Any]]:
        return [index[(seed, metric_name)] for seed in EXPECTED_SEEDS]

    valid_all = rows_for("delta_rr__valid__all")
    valid_all_latest = rows_for("contrast_rr__valid_all_minus_latest")
    valid_recent8_latest = rows_for("contrast_rr__valid_recent8_minus_latest")
    neutral_all = rows_for("delta_rr__neutral__all")
    neutral_valid_all = rows_for("contrast_rr__neutral_minus_valid__all")
    valid_latest_half_tie = rows_for("half_tie__valid__latest_1")

    distributed = (
        all(row["ci_low"] > 0.0 for row in valid_all)
        and all(row["ci_low"] > 0.0 for row in valid_all_latest)
        and all(row["point"] > 0.0 for row in valid_recent8_latest)
        and sum(row["ci_low"] > 0.0 for row in valid_recent8_latest) >= 2
    )
    neutral_only = (
        all(row["ci_low"] > 0.0 for row in neutral_all)
        and sum(row["ci_low"] <= 0.0 <= row["ci_high"] for row in valid_all) >= 2
        and all(row["ci_low"] > 0.0 for row in neutral_valid_all)
    )
    latest_identity = all(row["ci_low"] > 0.5 for row in valid_latest_half_tie)
    if distributed:
        primary = "DISTRIBUTED_VALID_SOURCE_ACCUMULATION"
    elif neutral_only:
        primary = "NEUTRALIZATION_SENSITIVE_ONLY"
    else:
        primary = "MIXED_OR_INCONCLUSIVE"
    return {
        "primary_label": primary,
        "valid_latest_identity_sensitivity": latest_identity,
        "distributed_rule_pass": distributed,
        "neutralization_sensitive_only_rule_pass": neutral_only,
        "rule_inputs": {
            "valid_all": valid_all,
            "valid_all_minus_latest": valid_all_latest,
            "valid_recent8_minus_latest": valid_recent8_latest,
            "neutral_all": neutral_all,
            "neutral_minus_valid_all": neutral_valid_all,
            "valid_latest_half_tie": valid_latest_half_tie,
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    r230a_dir = Path(args.r230a_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    claim_decision_path = Path(args.r230a_claim_decision).resolve()
    claim_decision = json.loads(claim_decision_path.read_text(encoding="utf-8"))
    if claim_decision.get("schema") != "focalswap.r230a.claim_decision.v1":
        raise RuntimeError("unexpected R230A claim-decision schema")
    expected_provenance = claim_decision.get("executed_provenance", {})
    r230a_run_manifest_path = (
        Path(args.r230a_run_manifest).resolve()
        if args.r230a_run_manifest
        else r230a_dir / "r230a_run_manifest.json"
    )
    r230a_run_manifest = json.loads(r230a_run_manifest_path.read_text(encoding="utf-8"))
    if r230a_run_manifest.get("schema") != "focalswap.r230a.run_manifest.v1":
        raise RuntimeError("unexpected R230A run-manifest schema")
    r230a_manifest_index = {
        Path(str(row["path"])).name: row for row in r230a_run_manifest.get("files", [])
    }
    all_scope_rows: list[dict[str, Any]] = []
    all_contrast_rows: list[dict[str, Any]] = []
    input_manifest: list[dict[str, Any]] = []
    reproduction: dict[str, Any] = {}
    lineage_by_seed: dict[str, Any] = {}
    baseline_identity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    expected_columns = {
        "official_row_index",
        "user_id",
        "source_slot_count",
        "candidate_count",
        "full_rr",
        *[f"{variant}_rr" for variant in VARIANTS],
    }
    seeds = (args.validate_only_seed,) if args.validate_only_seed else EXPECTED_SEEDS
    for position, seed in enumerate(seeds):
        prefix = f"r233_unisar_seed_{seed}"
        score_path = input_dir / f"{prefix}_scope_scores.csv"
        integrity_path = input_dir / f"{prefix}_integrity.json"
        if not score_path.is_file() or not integrity_path.is_file():
            raise FileNotFoundError(f"missing R233 inputs for seed {seed}")
        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity.get("contract_id") != "R233_UNISAR_SOURCE_SCOPE_LOCALIZATION":
            raise RuntimeError(f"bad R233 contract for seed {seed}")
        if int(integrity.get("seed", -1)) != seed:
            raise RuntimeError(f"integrity seed mismatch for {seed}")
        if integrity.get("integrity", {}).get("all_pass") is not True:
            raise RuntimeError(f"producer integrity failed for seed {seed}")
        if integrity.get("output_csv_sha256") != sha256_file(score_path):
            raise RuntimeError(f"score hash mismatch for seed {seed}")
        lineage_by_seed[str(seed)] = validate_execution_lineage(
            seed,
            integrity,
            expected_provenance,
        )
        frame = pd.read_csv(score_path)
        if set(frame.columns) != expected_columns:
            missing = sorted(expected_columns - set(frame.columns))
            extra = sorted(set(frame.columns) - expected_columns)
            raise RuntimeError(f"column contract failed for seed {seed}: missing={missing}, extra={extra}")
        identity = (
            frame["official_row_index"].to_numpy(),
            frame["user_id"].to_numpy(),
            frame["source_slot_count"].to_numpy(),
        )
        if baseline_identity is None:
            baseline_identity = tuple(value.copy() for value in identity)
        elif not all(np.array_equal(left, right) for left, right in zip(baseline_identity, identity)):
            raise RuntimeError(f"cross-seed row identity mismatch for seed {seed}")
        reproduction[str(seed)] = validate_r230a_reproduction(
            seed,
            frame,
            r230a_dir,
            r230a_manifest_index,
        )
        if args.validate_only_seed:
            validation_path = output_dir / f"r233_seed_{seed}_precontinuation_gate.json"
            validation_payload = {
                "schema": "focalswap.r233.precontinuation_gate.v1",
                "status": "PASS",
                "seed": seed,
                "rows": len(frame),
                "lineage": lineage_by_seed[str(seed)],
                "r230a_rowwise_reproduction": reproduction[str(seed)],
                "score_csv": str(score_path),
                "score_csv_sha256": sha256_file(score_path),
                "integrity_json": str(integrity_path),
                "integrity_json_sha256": sha256_file(integrity_path),
            }
            write_json(validation_path, validation_payload)
            print(json.dumps({"status": "PASS", "seed": seed, "gate": str(validation_path)}, sort_keys=True))
            return
        scope_rows, contrast_rows = analyze_seed(
            seed,
            frame,
            args.bootstrap_iters,
            args.bootstrap_seed + position * 100003,
            args.bootstrap_chunk_size,
        )
        all_scope_rows.extend(scope_rows)
        all_contrast_rows.extend(contrast_rows)
        input_manifest.extend(
            [
                {
                    "path": str(score_path),
                    "bytes": int(score_path.stat().st_size),
                    "sha256": sha256_file(score_path),
                },
                {
                    "path": str(integrity_path),
                    "bytes": int(integrity_path.stat().st_size),
                    "sha256": sha256_file(integrity_path),
                },
            ]
        )

    reference_lineage = lineage_by_seed[str(EXPECTED_SEEDS[0])]
    for seed in EXPECTED_SEEDS[1:]:
        current = lineage_by_seed[str(seed)]
        for key in [
            "processed_input_sha256",
            "official_code_sha256",
            "executed_local_code_sha256",
            "observed_support_sha256",
            "donor_map_sha256",
        ]:
            if current[key] != reference_lineage[key]:
                raise RuntimeError(f"cross-seed lineage mismatch for {key}, seed {seed}")
    decision = frozen_decision(all_scope_rows, all_contrast_rows)
    scope_path = output_dir / "r233_source_scope_summary.csv"
    contrast_path = output_dir / "r233_mechanism_contrasts.csv"
    payload_path = output_dir / "r233_source_scope_summary.json"
    report_path = output_dir / "r233_source_scope_report.md"
    manifest_path = output_dir / "r233_run_manifest.json"
    pd.DataFrame(all_scope_rows).to_csv(scope_path, index=False)
    pd.DataFrame(all_contrast_rows).to_csv(contrast_path, index=False)
    payload = {
        "schema": "focalswap.r233.source_scope_summary.v1",
        "status": "COMPLETE",
        "model_seeds": list(EXPECTED_SEEDS),
        "rows_per_seed": EXPECTED_ROWS,
        "user_clusters_per_seed": EXPECTED_USERS,
        "bootstrap_iters": args.bootstrap_iters,
        "bootstrap_unit": "user",
        "r230a_rowwise_reproduction": reproduction,
        "execution_lineage": lineage_by_seed,
        "decision": decision,
        "scope_summary_csv": str(scope_path),
        "mechanism_contrasts_csv": str(contrast_path),
    }
    write_json(payload_path, payload)
    selected = metric_index([*all_scope_rows, *all_contrast_rows])
    report_lines = [
        "# R233 Source-Scope Localization",
        "",
        f"Primary frozen label: `{decision['primary_label']}`.",
        "",
        "| Seed | Neutral all delta [95% CI] | Valid all delta [95% CI] | Valid all-latest gap [95% CI] | Valid latest half-tie [95% CI] |",
        "|---:|---:|---:|---:|---:|",
    ]
    for seed in EXPECTED_SEEDS:
        neutral = selected[(seed, "delta_rr__neutral__all")]
        valid = selected[(seed, "delta_rr__valid__all")]
        gap = selected[(seed, "contrast_rr__valid_all_minus_latest")]
        latest = selected[(seed, "half_tie__valid__latest_1")]
        report_lines.append(
            f"| {seed} | {neutral['point']:.6f} [{neutral['ci_low']:.6f}, {neutral['ci_high']:.6f}] "
            f"| {valid['point']:.6f} [{valid['ci_low']:.6f}, {valid['ci_high']:.6f}] "
            f"| {gap['point']:.6f} [{gap['ci_low']:.6f}, {gap['ci_high']:.6f}] "
            f"| {latest['point']:.4f} [{latest['ci_low']:.4f}, {latest['ci_high']:.4f}] |"
        )
    report_lines.extend(
        [
            "",
            "The observed-support intervention is a source-length/query-length matched derangement, not a semantic or category-matched swap.",
            "No manuscript claim is authorized until the integrity audit and result-to-claim review are complete.",
        ]
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    output_files = [scope_path, contrast_path, payload_path, report_path]
    manifest = {
        "schema": "focalswap.r233.run_manifest.v1",
        "status": "COMPLETE_PENDING_INTEGRITY_AND_CLAIM_REVIEW",
        "inputs": input_manifest,
        "r230a_claim_decision": {
            "path": str(claim_decision_path),
            "sha256": sha256_file(claim_decision_path),
        },
        "r230a_run_manifest": {
            "path": str(r230a_run_manifest_path),
            "sha256": sha256_file(r230a_run_manifest_path),
        },
        "executed_producer_sha256": lineage_by_seed[str(EXPECTED_SEEDS[0])]["executed_local_code_sha256"]["unisar_source_scope_localization.py"],
        "reviewed_producer_sha256": sha256_file(ROOT / "tools" / "unisar_source_scope_localization.py"),
        "analysis_sha256": sha256_file(Path(__file__)),
        "outputs": [
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in output_files
        ],
        "decision": decision,
    }
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "COMPLETE", "decision": decision["primary_label"]}, sort_keys=True))


if __name__ == "__main__":
    main()
