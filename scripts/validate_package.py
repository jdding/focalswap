#!/usr/bin/env python3
"""Validate the anonymous FocalSwap reviewer artifact."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import re
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXTERNAL_MODULES = {
    "joblib",
    "matplotlib",
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "yaml",
}


def require(path: str) -> Path:
    candidate = ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(path)
    return candidate


def rows(path: str) -> list[dict[str, str]]:
    with require(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(value: str, expected: float, tol: float = 5e-4) -> None:
    if abs(float(value) - expected) > tol:
        raise AssertionError(f"{value} differs from {expected}")


def validate_manifest_if_present() -> None:
    manifest_path = ROOT / "PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest.get("files", []):
        path = require(row["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise AssertionError(f"manifest hash mismatch: {row['path']}")


def validate_python_tools() -> None:
    tools_dir = ROOT / "tools"
    tool_paths = sorted(tools_dir.glob("*.py"))
    if not tool_paths:
        raise AssertionError("no packaged Python tools")
    local_modules = {path.stem for path in tool_paths}
    stdlib_modules = set(getattr(sys, "stdlib_module_names", ()))
    stdlib_root = Path(sysconfig.get_paths()["stdlib"]).resolve()

    def is_stdlib(module: str) -> bool:
        if module in stdlib_modules:
            return True
        spec = importlib.util.find_spec(module)
        if spec is None:
            return False
        if spec.origin in {"built-in", "frozen"}:
            return True
        if not spec.origin:
            return False
        origin = Path(spec.origin).resolve()
        return stdlib_root == origin or (
            stdlib_root in origin.parents and "site-packages" not in origin.parts
        )
    for path in tool_paths:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        tree = ast.parse(source, filename=str(path))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.split(".")[0])
        unresolved = sorted(
            module
            for module in imported_roots
            if module not in local_modules
            and not is_stdlib(module)
            and module not in EXTERNAL_MODULES
        )
        if unresolved:
            raise AssertionError(
                f"unresolved local imports in {path.name}: {', '.join(unresolved)}"
            )


def main() -> None:
    root_git = ROOT / ".git"
    nested_git = [path for path in ROOT.rglob(".git") if path != root_git]
    if nested_git:
        raise AssertionError("nested Git metadata must not be packaged")
    if list(ROOT.rglob("._*")):
        raise AssertionError("AppleDouble files must not be packaged")

    validate_manifest_if_present()
    validate_python_tools()

    manifest = json.loads(require("paper/figures/table_figure_manifest.json").read_text())
    expected_manifest_ids = {
        "fig1",
        "fig2",
        "fig3",
        "fig4",
        "table1",
        "table2",
        "table3",
    }
    if {row["id"] for row in manifest} != expected_manifest_ids:
        raise AssertionError("unexpected figure/table manifest")
    for row in manifest:
        for field in ("script", "source_data"):
            if row.get(field):
                require(f"paper/{row[field]}")

    control_contract = json.loads(
        require("paper/figures/source_data/r242_control_contract.json").read_text()
    )
    terminology = control_contract["terminology"]
    if {
        key: value["canonical_name"]
        for key, value in terminology["controls"].items()
    } != {
        "C0_neutral": "C0 neutral-mask control",
        "C1_profile_prioritized_valid": "C1 profile-prioritized observed-support donor",
        "C2_category_valid": "C2 category/profile-matched observed-support donor",
        "C3_same_user_temporal": "C3 same-user temporal donor",
    }:
        raise AssertionError("checkpoint control terminology contract changed")
    if terminology["paired_comparison"]["canonical_name"] != "paired C0-C2 difference":
        raise AssertionError("paired control terminology changed")

    figure_controls = rows("paper/figures/source_data/fig4_control_sensitivity.csv")
    if {row["condition"] for row in figure_controls if row["panel"] == "control"} != {
        "C0 neutral-mask",
        "C1 profile-prioritized donor",
        "C2 category/profile-matched donor",
        "paired C0-C2 difference",
    }:
        raise AssertionError("Figure 4 control labels diverge from the terminology contract")

    anchors = rows("paper/figures/source_data/table4_checkpoint_anchor_readouts.csv")
    if [row["checkpoint"] for row in anchors] != ["SESRec", "UniSAR", "UniSAR", "UniSAR"]:
        raise AssertionError("unexpected corrected checkpoint rows")
    close(anchors[0]["full_rr"], 0.042319281)
    close(anchors[0]["latest_slot_delta_rr"], 0.000137847)
    for row in anchors:
        if not float(row["latest_slot_delta_ci_low"]) <= 0 <= float(row["latest_slot_delta_ci_high"]):
            raise AssertionError("registered latest-slot interval unexpectedly clears zero")
        if row["latest_slot_dependence_gate"] != "not_established":
            raise AssertionError("registered latest-slot gate must remain closed")

    scope = rows("paper/figures/source_data/table4_dependence_scope_readouts.csv")
    if [row["checkpoint"] for row in scope] != ["SESRec", "UniSAR", "UniSAR", "UniSAR"]:
        raise AssertionError("unexpected dependence-scope rows")
    for row in scope:
        if not float(row["latest_slot_ci_low"]) <= 0 <= float(row["latest_slot_ci_high"]):
            raise AssertionError("latest-slot interval unexpectedly clears zero")
        if row["latest_slot_dependence_gate"] != "not_established":
            raise AssertionError("latest-slot gate must remain closed")
    if scope[0]["all_source_control_status"] != "not_tested":
        raise AssertionError("SESRec all-source control must remain untested")
    for row in scope[1:]:
        if float(row["all_source_c0_ci_low"]) <= 0:
            raise AssertionError("UniSAR all-source C0 interval must clear zero")
        if float(row["c0_minus_c1_ci_low"]) <= 0:
            raise AssertionError("UniSAR paired C0-C1 interval must clear zero")
        if row["all_source_control_status"] != "neutralization_sensitive_only":
            raise AssertionError("UniSAR control-calibration status changed")
    if float(scope[1]["all_source_c1_ci_high"]) >= 0:
        raise AssertionError("seed 20230601 C1 interval must remain below zero")
    for row in scope[2:]:
        if not float(row["all_source_c1_ci_low"]) <= 0 <= float(row["all_source_c1_ci_high"]):
            raise AssertionError("seed 20230602/03 C1 interval must include zero")

    sesrec = json.loads(require("evidence/sesrec_corrected_bootstrap.json").read_text())
    if not sesrec["dependence"]["ci_low"] <= 0 <= sesrec["dependence"]["ci_high"]:
        raise AssertionError("SESRec dependence interval unexpectedly clears zero")
    if not sesrec["l1_half_tie"]["ci_low"] <= 0.5 <= sesrec["l1_half_tie"]["ci_high"]:
        raise AssertionError("SESRec L1 diagnostic excludes symmetry")
    if not sesrec["l2_half_tie"]["ci_low"] <= 0.5 <= sesrec["l2_half_tie"]["ci_high"]:
        raise AssertionError("SESRec L2 diagnostic excludes symmetry")

    multiseed = rows("evidence/unisar_corrected_multiseed_summary.csv")
    if len(multiseed) != 3:
        raise AssertionError("expected three UniSAR seeds")
    for row in multiseed:
        if not float(row["dep_ci_low"]) <= 0 <= float(row["dep_ci_high"]):
            raise AssertionError("UniSAR dependence interval unexpectedly clears zero")
        if not (
            float(row["category_broken_source_reverse_ci_low"])
            <= 0.5
            <= float(row["category_broken_source_reverse_ci_high"])
        ):
            raise AssertionError("reverse-source interval excludes symmetry")
        if float(row["category_broken_source_half_tie"]) >= 0.5:
            raise AssertionError("corrected true-reference direction changed")

    dual = rows("evidence/unisar_dual_estimand_summary.csv")
    if len(dual) != 3:
        raise AssertionError("expected three UniSAR dual-estimand rows")
    for row in dual:
        if float(row["aggregate_ci_low"]) <= 0:
            raise AssertionError("all-source C0 dependence interval does not clear zero")
        if not float(row["focal_ci_low"]) <= 0 <= float(row["focal_ci_high"]):
            raise AssertionError("latest-slot dependence interval unexpectedly clears zero")
        if float(row["gap_ci_low"]) <= 0:
            raise AssertionError("paired aggregate-minus-focal gap does not clear zero")

    source_scope = rows("evidence/unisar_source_scope_summary.csv")
    if {row["model_seed"] for row in source_scope} != {"20230601", "20230602", "20230603"}:
        raise AssertionError("unexpected R233 seed set")
    contrasts = rows("evidence/unisar_mechanism_contrasts.csv")
    all_scope = [
        row for row in contrasts
        if row["contrast"] == "neutral_minus_valid" and row["scope"] == "all"
    ]
    if len(all_scope) != 3 or any(float(row["ci_low"]) <= 0 for row in all_scope):
        raise AssertionError("R233 all-source paired C0-C1 difference changed")

    gensar = json.loads(require("evidence/gensar_persrc_calibration.json").read_text())
    if gensar["status"] != "PASS" or gensar["task"] != "persrc":
        raise AssertionError("GenSAR PerSrc calibration status changed")
    if gensar["checkpoint_selection"] != {
        "selected_before_focalswap_readout": True,
        "checkpoint_epoch": 3,
        "native_quality_gate": "passed",
        "aggregation": "single_deterministic_checkpoint",
    }:
        raise AssertionError("GenSAR checkpoint-selection provenance changed")
    if gensar["evaluation_population"] != {
        "split": "official_validation",
        "requested_rows": 10000,
        "current_query_in_history_exclusions": 1042,
        "no_history_exclusions": 0,
        "retained_rows": 8958,
    }:
        raise AssertionError("GenSAR validation filtering contract changed")
    if gensar["completed_rows"] != 8958 or gensar["user_clusters"] != 8958:
        raise AssertionError("GenSAR PerSrc eligible population changed")
    gensar_metrics = gensar["metrics"]
    close(str(gensar_metrics["delta_full_minus_focal_query_mask"]["point"]), 0.034823)
    close(str(gensar_metrics["delta_full_minus_focal_query_swap"]["point"]), 0.105596)
    close(str(gensar_metrics["half_tie_full_vs_focal_query_swap"]["point"]), 0.635298)
    close(str(gensar_metrics["delta_full_minus_focal_session_swap"]["point"]), 0.002054)
    close(str(gensar_metrics["half_tie_full_vs_focal_session_swap"]["point"]), 0.517749)
    if gensar_metrics["delta_full_minus_focal_query_swap"]["ci95"][0] <= 0:
        raise AssertionError("GenSAR current-query swap interval changed")

    planted = json.loads(
        require("evidence/r243_full_chain_planted_control.json").read_text()
    )
    if len(planted["records"]) != 9:
        raise AssertionError("R243 planted-control matrix must contain nine runs")
    dose0 = [row for row in planted["records"] if row["dose"] == 0.0]
    dose1 = [row for row in planted["records"] if row["dose"] == 1.0]
    if len(dose0) != 3 or any(
        row["status"] != "RUN_COMPLETE_NEGATIVE_CONTROL_PASS" for row in dose0
    ):
        raise AssertionError("R243 dose-zero negative-control contract changed")
    if len(dose1) != 3:
        raise AssertionError("R243 dose-one seed set changed")
    for row in dose1:
        gates = row["promotion_gates"]
        for name in [
            "g2_oracle_above_legal_shuffled_p95",
            "g3_full_minus_neutral_mask_ci_above_zero",
            "g3_placebo_adjusted_ci_above_zero",
            "g3_true_vs_d1_half_tie_ci_above_half",
        ]:
            if not gates[name]:
                raise AssertionError(f"R243 dose-one gate changed: {name}")
        if gates["g4_router_minus_best_fixed_ci_above_zero"]:
            raise AssertionError("R243 G4 recovery unexpectedly promotes")

    r242 = rows("evidence/unisar_r242_control_ladder_summary.csv")
    c2_latest = [
        row for row in r242
        if row["scope"] == "latest_1" and row["control"] == "C2_category_valid"
    ]
    c2_all = [
        row for row in r242
        if row["scope"] == "all" and row["control"] == "C2_category_valid"
    ]
    if len(c2_latest) != 3 or any(
        not float(row["ci_low"]) <= 0 <= float(row["ci_high"])
        for row in c2_latest
    ):
        raise AssertionError("R242 focal C2 endpoint changed")
    if len(c2_all) != 3 or any(float(row["delta_rr"]) > 0 for row in c2_all):
        raise AssertionError("R242 all-source C2 endpoint changed")
    reversal = rows("evidence/unisar_r242_control_reversal_contrasts.csv")
    reversal_all = [
        row for row in reversal
        if row["scope"] == "all" and row["contrast"] == "C0_minus_C2"
    ]
    if len(reversal_all) != 3 or any(float(row["ci_low"]) <= 0 for row in reversal_all):
        raise AssertionError("R242 all-source control reversal changed")

    eligibility = rows("evidence/r250_c2_eligibility_balance.csv")
    if {row["metric"] for row in eligibility} != {
        "source_slot_count", "user_test_rows"
    }:
        raise AssertionError("R250 C2 balance metrics changed")
    source_slots = next(
        row for row in eligibility if row["metric"] == "source_slot_count"
    )
    close(source_slots["eligible_mean"], 14.723975, tol=1e-6)
    close(source_slots["ineligible_mean"], 18.692981, tol=1e-6)

    c0_balance = rows("evidence/r250_c0_margin_by_eligibility.csv")
    if len(c0_balance) != 9:
        raise AssertionError("R250 C0 population-margin rows changed")
    all_source = [
        row for row in c0_balance if row["metric"] == "c0_all_source_delta_rr"
    ]
    if len(all_source) != 3 or any(
        float(row["eligible_ci_low"]) <= 0
        or float(row["ineligible_ci_low"]) <= 0
        for row in all_source
    ):
        raise AssertionError("C0 response must remain positive in both populations")

    slot_validity = json.loads(
        require("evidence/r250_latest_slot_validity.json").read_text()
    )
    if slot_validity["rows_without_source_slot"] != 0:
        raise AssertionError("source-slot validity changed")
    if slot_validity["c2_latest_eligible_rows"] != 87044:
        raise AssertionError("latest-slot eligible count changed")
    if slot_validity["c2_latest_bad_changed_slot_rows"] != 0:
        raise AssertionError("latest-slot intervention no longer changes exactly one slot")

    common = rows("evidence/common_surface_summary.csv")
    expected = {(method, str(seed)) for method in ("sesrec", "unisar") for seed in (20230601, 20230602, 20230603)}
    if {(row["method"], row["seed"]) for row in common} != expected:
        raise AssertionError("common-surface row set mismatch")
    if {row["alignment_sha256"] for row in common} != {
        "4abed4db55e237568d1bf8ff0e42d711311fc315765017120d6761edd9e50869"
    }:
        raise AssertionError("common-surface alignment mismatch")
    if any(int(row["rows"]) != 3249 or float(row["full_ndcg10"]) <= 0.045436 for row in common):
        raise AssertionError("common-surface quality gate failed")

    qilin_mult = json.loads(
        require("evidence/qilin_ten_bucket_multiplicity.json").read_text()
    )
    if qilin_mult["n_tests"] != 10:
        raise AssertionError("Qilin ten-bucket multiplicity family changed")
    if qilin_mult["bootstrap"]["n_boot"] != 1000:
        raise AssertionError("Qilin ten-bucket bootstrap count changed")
    if qilin_mult["p_value"]["n_perm"] != 10000:
        raise AssertionError("Qilin ten-bucket sign-flip count changed")
    tax4 = next(row for row in qilin_mult["rows"] if row["bucket"] == "tax_near/share>=4")
    close(str(tax4["half_tie"]), 0.5656836461)
    close(str(tax4["matched_minus_random_delta"]), 0.0844504021)

    kuaisar_swap = rows("evidence/kuaisar_swap_ladder_metrics.csv")
    source_broken = next(row for row in kuaisar_swap if row["level"] == "source_broken")
    same_category_source = next(
        row for row in kuaisar_swap if row["level"] == "same_category_source"
    )
    close(source_broken["half_tie"], 0.5005816772)
    close(same_category_source["half_tie"], 0.5096974399)
    if float(source_broken["tie"]) <= 0.75 or float(same_category_source["tie"]) <= 0.75:
        raise AssertionError("KuaiSAR tie-dominance evidence changed")

    gate_curve = rows("evidence/kuaisar_gate_learning_curve.csv")
    learned = [
        row for row in gate_curve
        if row["gate"] in {"gbdt_soft", "logistic_soft", "mlp_soft"}
    ]
    if {int(row["train_limit"]) for row in learned} != {500, 1000, 3000, 7000, 10000}:
        raise AssertionError("KuaiSAR G4 learning-curve grid changed")
    if max(float(row["score_rr"]) for row in learned) >= 0.2421895035:
        raise AssertionError("KuaiSAR learned gate unexpectedly beats History-only")

    positive_curve = rows("evidence/positive_control_dose_curve.csv")
    positive_rows = [row for row in positive_curve if float(row["strength"]) > 0]
    if [float(row["strength"]) for row in positive_rows] != [0.01, 0.02, 0.04, 0.08, 0.16]:
        raise AssertionError("positive-control dose grid changed")
    if any(float(row["best_gate_headroom_recovery"]) != 1.0 for row in positive_rows):
        raise AssertionError("positive-control recovery is no longer complete")

    qilin_g2_payload = json.loads(
        require("evidence/qilin_g2_null_robustness.json").read_text()
    )
    construction = qilin_g2_payload["null_construction"]
    if construction != {
        "seeds": 30,
        "draws_per_seed": 5000,
        "row_resampling": False,
        "fixed_row_operator": (
            "For each fixed evaluation row, sample one precomputed "
            "alternative-query RR column from same_user, time_near, "
            "tax_near, or random; keep the history RR and row set fixed."
        ),
    }:
        raise AssertionError("Qilin G2 null-construction contract changed")
    qilin_g2 = qilin_g2_payload["r167_g2_robustness"]
    close(str(qilin_g2["observed_gain"]), 0.1242896293)
    close(str(qilin_g2["null_p95"]), 0.1237893677)
    close(str(qilin_g2["null_p99"]), 0.1252752172)
    if len(qilin_g2["per_seed"]) != 30 or qilin_g2["seed_pass_rate"] != 1.0:
        raise AssertionError("Qilin G2 30-seed null summary changed")

    interface_rows = rows("evidence/runtime_boundary_dependence_bootstrap.csv")
    if not {
        "Q_CHECKPOINT_GSEREC_R192B",
        "Q_CHECKPOINT_LCRSER_R195",
    } == {row["surface_id"] for row in interface_rows}:
        raise AssertionError("controlled-interface dependence rows changed")

    unifiedssr = json.loads(
        require("evidence/unifiedssr_interface_boundary.json").read_text()
    )
    if unifiedssr["status"] != "CLOSED_PROTOCOL_MISMATCH":
        raise AssertionError("UnifiedSSR interface-boundary status changed")
    if unifiedssr["same_scorer_five_view_available"]:
        raise AssertionError("UnifiedSSR unexpectedly exposes one five-view scorer")

    candidate_contract = json.loads(
        require("evidence/common_candidate_contract.json").read_text()
    )
    if candidate_contract["candidate_count"] != 100:
        raise AssertionError("common candidate cardinality changed")
    if candidate_contract["known_positive_negative_violations"] != 0:
        raise AssertionError("common candidate exclusion contract failed")
    if candidate_contract["nontrain_negative_violations"] != 0:
        raise AssertionError("common candidate train-universe contract failed")

    hashes = json.loads(require("evidence/common_surface_artifact_hashes.json").read_text())
    if len(hashes) != 6:
        raise AssertionError("expected six artifact hash rows")
    sha = re.compile(r"^[0-9a-f]{64}$")
    for row in hashes:
        if not all(sha.fullmatch(value) for value in row["artifact_sha256"].values()):
            raise AssertionError("invalid artifact hash")

    for path in [
        "paper/sections/table_data_regimes.tex",
        "paper/sections/table_gate_ledger.tex",
        "paper/sections/table_tier_a.tex",
        "tools/unisar_five_view_readout.py",
        "tools/sesrec_five_view_readout.py",
        "tools/common_surface_summary.py",
        "tools/dual_estimand_audit.py",
        "tools/source_scope_analysis.py",
        "tools/unisar_source_scope_localization.py",
        "tools/control_ladder_analysis.py",
        "tools/control_reversal_analysis.py",
        "tools/unisar_semantic_control_ladder.py",
        "tools/gensar_persrc_readout_analysis.py",
        "tools/qilin_trained_full_chain.py",
        "tools/full_chain_planted_control.py",
        "tools/summarize_planted_control.py",
        "tools/review_evidence_closure.py",
        "tools/tv_random_baseline.py",
        "scripts/check_tool_imports.py",
    ]:
        require(path)

    print("FocalSwap reviewer artifact validation passed")


if __name__ == "__main__":
    main()
