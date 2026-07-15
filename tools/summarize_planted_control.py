#!/usr/bin/env python3
"""Summarize and gate a complete R243 dose-by-seed matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.run_root.glob(f"r243_full_dose*_seed*_{args.run_stamp}/r243_metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label_path = path.parent / "r243_label_audit.csv"
        label_hash = hashlib.sha256(label_path.read_bytes()).hexdigest()
        manifest = json.loads((path.parent / "r243_manifest.json").read_text(encoding="utf-8"))
        records.append({
            "path": str(path),
            "dose": float(payload["dose"]),
            "seed": int(payload["student_seed"]),
            "status": payload["status"],
            "all_gates": bool(payload["positive_control_all_gates_pass"]),
            "negative_control_pass": bool(payload["negative_control_pass"]),
            "adjusted_specificity": float(
                payload["split_metrics"]["test"]["specificity_minus_placebo_half_tie"]["point"]
            ),
            "surface_hash": payload["r239_reference_check"]["current_complete_input_contract_sha256"],
            "label_hash": label_hash,
            "label_manifest_hash_matches": (
                label_hash == manifest["planted_label_contract_sha256"]
            ),
        })
    expected = {(dose, seed) for dose in (0.0, 0.5, 1.0) for seed in (20260713, 20260714, 20260715)}
    observed = {(row["dose"], row["seed"]) for row in records}
    by_dose = {
        dose: [row for row in records if row["dose"] == dose]
        for dose in (0.0, 0.5, 1.0)
    }
    means = {
        str(dose): sum(row["adjusted_specificity"] for row in rows) / len(rows)
        if rows else None
        for dose, rows in by_dose.items()
    }
    gates = {
        "complete_nine_run_matrix": observed == expected and len(records) == 9,
        "surface_hash_identical_across_matrix": len({row["surface_hash"] for row in records}) == 1,
        "label_hash_identical_within_each_dose": all(
            len({row["label_hash"] for row in by_dose[dose]}) == 1
            for dose in (0.0, 0.5, 1.0)
        ),
        "label_artifact_hashes_match_manifests": all(
            row["label_manifest_hash_matches"] for row in records
        ),
        "lambda_one_all_seeds_pass": len(by_dose[1.0]) == 3 and all(row["all_gates"] for row in by_dose[1.0]),
        "lambda_zero_all_seeds_negative_control_pass": len(by_dose[0.0]) == 3 and all(row["negative_control_pass"] for row in by_dose[0.0]),
        "adjusted_specificity_monotonic_by_dose": (
            all(means[str(dose)] is not None for dose in (0.0, 0.5, 1.0))
            and means["0.0"] < means["0.5"] < means["1.0"]
        ),
    }
    status = "R243_MATRIX_PASS" if all(gates.values()) else "R243_MATRIX_MISS"
    output = {"schema": "focalswap.r243.matrix_summary.v1", "status": status,
              "records": records, "dose_mean_adjusted_specificity": means,
              "matrix_gates": gates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "matrix_gates": gates}, sort_keys=True))
    return 0 if status == "R243_MATRIX_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
