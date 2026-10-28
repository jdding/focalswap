#!/usr/bin/env python3
"""Complete checkpoint diagnostics from existing, hash-locked RR exports.

No model is loaded. Supplementary half-tie intervals are pointwise 95%
user-cluster percentile intervals; they are not prespecified primary endpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from control_ladder_analysis import cluster_bootstrap, load_seed, sha256_file


SEEDS = (20230601, 20230602, 20230603)
SESREC_SCORES_SHA256 = "7c7591fb379a1f6d6e5c4d549da043c2763fa85b4700d015f0954c473f3d6b0b"


def half_tie(full: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    if full.shape != replacement.shape or full.ndim != 1 or not len(full):
        raise ValueError("RR arrays must be nonempty, aligned vectors")
    for values in (full, replacement):
        if not np.all(np.isfinite(values) & (values > 0) & (values <= 1)):
            raise ValueError("invalid RR")
        ranks = np.rint(2.0 / values) / 2.0
        if not np.allclose(values, 1.0 / ranks, rtol=0, atol=1e-12):
            raise ValueError("RR does not follow the declared midpoint-rank grid")
    z = (full > replacement).astype(float) + 0.5 * (full == replacement)
    rank_z = (np.rint(2.0 / full) < np.rint(2.0 / replacement)).astype(float)
    rank_z += 0.5 * (np.rint(2.0 / full) == np.rint(2.0 / replacement))
    if not np.array_equal(z, rank_z):
        raise ValueError("RR comparison and midpoint-rank comparison disagree")
    return z


def self_test() -> None:
    full = np.array([1.0, 0.5, 0.25, 1 / 2.5])
    replacement = np.array([0.5, 0.5, 0.5, 1 / 2.5])
    z = half_tie(full, replacement)
    assert np.array_equal(z, [1.0, 0.5, 0.0, 0.5])
    assert np.array_equal(half_tie(replacement, full), 1 - z)
    assert np.array_equal(half_tie(full, full), np.full(4, 0.5))
    users = np.array([1, 1, 1, 2])
    point, low, high, count = cluster_bootstrap(np.full(4, 0.5), users, 200, 7)
    assert (point, low, high, count) == (0.5, 0.5, 0.5, 2)
    point, _, _, _ = cluster_bootstrap(np.array([1., 1., 1., 0.]), users, 200, 7)
    assert point == 0.75  # row-weighted estimate, not a mean of user means
    try:
        half_tie(full, np.array([0.5, 0.5, 0.0, 0.5]))
    except ValueError:
        pass
    else:
        raise AssertionError("zero RR should be rejected")
    print("checkpoint reporting self-test passed")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--r242-dir", type=Path)
    p.add_argument("--r233-dir", type=Path)
    p.add_argument("--reference-manifest", type=Path)
    p.add_argument("--c1-summary", type=Path)
    p.add_argument("--sesrec-scores", type=Path)
    p.add_argument("--sesrec-summary", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--bootstrap-iters", type=int, default=10_000)
    p.add_argument("--bootstrap-seed", type=int, default=20261028)
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    for field in ("r242_dir", "r233_dir", "reference_manifest", "c1_summary",
                  "sesrec_scores", "sesrec_summary", "output_dir"):
        if getattr(a, field) is None:
            p.error(f"--{field.replace('_', '-')} is required")
    if a.bootstrap_iters < 1000:
        p.error("at least 1000 bootstrap draws are required")
    reference = json.loads(a.reference_manifest.read_text())
    prior_c1 = pd.read_csv(a.c1_summary)
    records, c1_records, paired_records, identities = [], [], [], {}
    supports = {}
    for index, seed in enumerate(SEEDS):
        frame, integrity, _ = load_seed(a.r242_dir, a.r233_dir, seed, reference)
        raw = pd.read_csv(a.r242_dir / f"r242_unisar_seed_{seed}_control_scores.csv",
                          usecols=["candidate_count"])
        if set(raw.candidate_count) != {100}:
            raise ValueError("UniSAR candidate count changed")
        identities[str(seed)] = {
            "r242_scores_sha256": integrity["output_csv_sha256"],
            "r233_scores_sha256": sha256_file(a.r233_dir / f"r233_unisar_seed_{seed}_scope_scores.csv"),
            "checkpoint_sha256": integrity["checkpoint_sha256"],
        }
        for offset, (scope, expected_rows, expected_users) in enumerate(
                (("latest_1", 87044, 2843), ("all", 66139, 2170))):
            mask = frame[f"category_{scope}_eligible"].astype(bool)
            sub = frame.loc[mask]
            if len(sub) != expected_rows or sub.user_id.nunique() != expected_users:
                raise ValueError("eligible support changed")
            keys = sub[["official_row_index", "user_id"]].to_numpy()
            if scope in supports and not np.array_equal(supports[scope], keys):
                raise ValueError("model seeds use different eligible rows")
            supports[scope] = keys
            full = sub.full_rr.to_numpy(float)
            donor = sub[f"category_{scope}_rr"].to_numpy(float)
            z = half_tie(full, donor)
            boot_seed = a.bootstrap_seed + index * 10 + offset
            point, low, high, clusters = cluster_bootstrap(
                z, sub.user_id.to_numpy(), a.bootstrap_iters, boot_seed)
            records.append({
                "checkpoint": f"UniSAR S{index+1}", "model_seed": seed,
                "scope": scope, "control": "C2_category_profile",
                "rows": len(sub), "user_clusters": clusters,
                "wins": int((z == 1).sum()), "ties": int((z == .5).sum()),
                "losses": int((z == 0).sum()), "half_tie": point,
                "ci_low": low, "ci_high": high, "ci_level": .95,
                "bootstrap_seed": boot_seed, "signed_delta_rr": float((full - donor).mean()),
            })
            if scope == "all":
                c1 = sub.valid_all_rr.to_numpy(float)
                old = prior_c1[(prior_c1.seed == seed) & (prior_c1.scope == "all") &
                               (prior_c1.control == "C1_profile_prioritized_valid")]
                if len(old) != 1 or abs(float(old.iloc[0].delta_rr) - (full-c1).mean()) > 1e-12:
                    raise ValueError("existing C1 point estimate does not reproduce")
                old_row = old.iloc[0]
                c1_records.append({
                    "checkpoint": f"UniSAR S{index+1}", "model_seed": seed,
                    "rows": len(sub), "user_clusters": clusters,
                    "delta_rr": float(old_row.delta_rr), "ci_low": float(old_row.ci_low),
                    "ci_high": float(old_row.ci_high), "ci_level": float(old_row.ci_level),
                    "comparison_population": "C2_common_support",
                })
                # delta(C1)-delta(C2) = RR(C2)-RR(C1), paired on identical rows.
                delta, dlo, dhi, _ = cluster_bootstrap(
                    donor-c1, sub.user_id.to_numpy(), a.bootstrap_iters, boot_seed+100)
                paired_records.append({
                    "checkpoint": f"UniSAR S{index+1}", "model_seed": seed,
                    "rows": len(sub), "user_clusters": clusters,
                    "contrast": "signed_contribution_C1_minus_C2",
                    "delta_rr": delta, "ci_low": dlo, "ci_high": dhi,
                    "ci_level": .95, "bootstrap_seed": boot_seed+100,
                })
        print(f"UniSAR S{index+1}: hash, identity, support and C1 parity checks passed", flush=True)

    if sha256_file(a.sesrec_scores) != SESREC_SCORES_SHA256:
        raise ValueError("SESRec score hash changed")
    ses = pd.read_csv(a.sesrec_scores)
    if len(ses) != 16788 or set(ses.candidate_count) != {100}:
        raise ValueError("SESRec row/candidate count changed")
    sub = ses.loc[ses.l2_matched.astype(bool)]
    if len(sub) != 8631 or sub.user_id.nunique() != 2134:
        raise ValueError("SESRec L2 support changed")
    full, donor = sub.full_sar_rr.to_numpy(float), sub.l2_swap_rr.to_numpy(float)
    z = half_tie(full, donor)
    point, low, high, clusters = cluster_bootstrap(
        z, sub.user_id.to_numpy(), a.bootstrap_iters, a.bootstrap_seed+30)
    records.append({
        "checkpoint": "SESRec", "model_seed": "", "scope": "single_session",
        "control": "L2_category_overlap", "rows": len(sub), "user_clusters": clusters,
        "wins": int((z == 1).sum()), "ties": int((z == .5).sum()),
        "losses": int((z == 0).sum()), "half_tie": point,
        "ci_low": low, "ci_high": high, "ci_level": .95,
        "bootstrap_seed": a.bootstrap_seed+30, "signed_delta_rr": float((full-donor).mean()),
    })
    ses_summary = json.loads(a.sesrec_summary.read_text())
    c0 = sub.search_masked_rr.to_numpy(float)
    values = {"C0_neutral_mask": full-c0,
              "C2_same_category_observed_support": full-donor,
              "paired_C0_minus_C2": donor-c0}
    for old in ses_summary["endpoints"]:
        if abs(old["delta_rr"] - float(values[old["endpoint"]].mean())) > 1e-12:
            raise ValueError("SESRec paired-control point estimate does not reproduce")
    identities["SESRec"] = {"scores_sha256": SESREC_SCORES_SHA256}
    result = {
        "schema": "focalswap.checkpoint_reporting_completion.v1",
        "analysis_kind": "supplementary_reanalysis_of_existing_exports",
        "new_model_inference": False, "bootstrap_unit": "user_cluster",
        "estimator": "row_weighted_mean_with_whole_user_cluster_resampling",
        "bootstrap_iters": a.bootstrap_iters, "ci_method": "percentile",
        "interval_scope": "Supplementary half-tie and C1-C2 intervals are pointwise 95%; no multiplicity or prespecification claim.",
        "half_tie": records, "c1_all_source": c1_records,
        "c1_minus_c2_all_source": paired_records,
        "sesrec_paired_control": ses_summary["endpoints"],
        "sesrec_donor": "L2 shares at least one search-item category; it does not enforce UniSAR C2 profile matching.",
        "input_hashes": identities,
        "source_summary_hashes": {"c1": sha256_file(a.c1_summary), "sesrec": sha256_file(a.sesrec_summary)},
        "verification": {"input_hashes": True, "same_row_support": True,
                         "rr_rank_tie_parity": True, "prior_signed_estimates_reproduced": True,
                         "independent_semantic_review": "not_performed"},
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("checkpoint_half_tie", records), ("checkpoint_c1_all_source", c1_records),
                       ("checkpoint_c1_minus_c2", paired_records)):
        pd.DataFrame(rows).to_csv(a.output_dir / f"{name}.csv", index=False)
    (a.output_dir / "checkpoint_reporting_completion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(pd.DataFrame(records)[["checkpoint", "scope", "half_tie", "ci_low", "ci_high"]].to_string(index=False))
    print(pd.DataFrame(paired_records)[["checkpoint", "delta_rr", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
