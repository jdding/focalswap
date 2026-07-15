#!/usr/bin/env python3
"""R225 common-surface bundle and candidate-identity preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories  # noqa: E402
from official_adapter import (  # noqa: E402
    build_sesrec_trainer_bundle,
    build_unisar_tiny_bundle,
    compact_examples,
    get_git_commit,
)
from unisar_preflight import ladder_aware_sample  # noqa: E402
from common_candidate_contract import file_sha256, load_candidate_ledger  # noqa: E402


DEFAULT_OFFICIAL = Path("/private/tmp/focalswap_r225_official_20260710")
DEFAULT_WORK = Path("/private/tmp/focalswap_r225_preflight_20260710")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/eligible_examples.jsonl",
    )
    parser.add_argument(
        "--item-metadata",
        type=Path,
        default=ROOT / "data/processed/item_metadata.parquet",
    )
    parser.add_argument(
        "--candidate-ledger",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/candidates.jsonl.gz",
    )
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-prefix", default="common_surface_preflight_20260710")
    parser.add_argument("--row-limit", type=int, default=128)
    parser.add_argument("--neg-k", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20_260_710)
    parser.add_argument("--swap-tries", type=int, default=80)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def digest_candidate_rows(rows: list[list[int]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update((",".join(str(item) for item in row) + "\n").encode("utf-8"))
    return digest.hexdigest()


def emitted_unisar_candidates(data_root: Path, split: str) -> list[list[int]]:
    name = {"validation": "rec_val.pkl", "test": "rec_test.pkl"}[split]
    frame = pd.read_pickle(data_root / "dataset" / name)
    return [
        [int(row.item_id), *[int(item) for item in row.neg_items]]
        for row in frame.itertuples(index=False)
    ]


def emitted_sesrec_candidates(data_root: Path, split: str, candidate_count: int) -> list[list[int]]:
    name = {"validation": "valid_inter.tsv", "test": "test_inter.tsv"}[split]
    frame = pd.read_csv(data_root / "dataset" / name, sep="\t")
    if len(frame) % candidate_count:
        raise ValueError(f"SESRec {split} rows are not divisible by {candidate_count}")
    values = [int(item) for item in frame["i_id"].tolist()]
    return [values[start : start + candidate_count] for start in range(0, len(values), candidate_count)]


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# R225 Common-Surface Preflight",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Sample rows: `{payload['sample_rows']}`",
        f"- Sample split counts: `{payload['sample_split_counts']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- SESRec commit: `{payload['official_commits']['sesrec']}`",
        f"- UniSAR commit: `{payload['official_commits']['unisar']}`",
        "",
        "## Candidate Identity",
        "",
    ]
    for split, result in payload["candidate_identity"].items():
        lines.append(
            f"- `{split}`: match=`{result['match']}`, rows=`{result['rows']}`, "
            f"SHA-256=`{result['compact_candidate_sha256']}`"
        )
    lines.extend(
        [
            "",
            "## Full-Surface Cardinality Check",
            "",
            f"- Row-scoped users: `{payload['full_cardinalities']['users']:,}`",
            f"- Raw items: `{payload['full_cardinalities']['raw_items']:,}`",
            f"- Query tokens: `{payload['full_cardinalities']['query_tokens']:,}`",
            f"- Source sessions: `{payload['full_cardinalities']['source_sessions']:,}`",
            f"- UniSAR user embedding resize required: `{payload['full_cardinalities']['unisar_user_resize_required']}`",
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
            "## Claim Boundary",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.neg_k != 99:
        raise ValueError("R225 preflight requires 99 negatives")
    sesrec_repo = args.official_root / "SESREC-SIGIR-2023"
    unisar_repo = args.official_root / "UniSAR"
    if not sesrec_repo.exists() or not unisar_repo.exists():
        raise FileNotFoundError(f"official repos missing under {args.official_root}")

    all_rows = read_jsonl(args.examples_path)
    item_cat = load_item_categories(args.item_metadata)
    selected, sample_info = ladder_aware_sample(
        all_rows,
        args.row_limit,
        item_cat,
        args.seed,
        args.swap_tries,
    )
    args.work_root.mkdir(parents=True, exist_ok=True)
    sample_path = args.work_root / "smoke_examples.jsonl"
    write_jsonl(sample_path, selected)
    selected_ids = {str(row["example_id"]) for row in selected}
    candidate_raw = load_candidate_ledger(args.candidate_ledger, selected_ids)

    unisar_work = args.work_root / "unisar_work"
    if unisar_work.exists():
        shutil.rmtree(unisar_work)
    unisar_bundle = build_unisar_tiny_bundle(
        selected,
        unisar_work,
        len(selected),
        args.neg_k,
        args.seed,
        score_rows_path=args.work_root / "unisar_score_rows.csv",
        candidate_raw_by_example=candidate_raw,
    )
    sesrec_bundle = build_sesrec_trainer_bundle(
        sesrec_repo,
        selected,
        len(selected),
        args.neg_k,
        args.seed,
        candidate_raw_by_example=candidate_raw,
    )

    _, maps = compact_examples(selected, len(selected), candidate_raw)
    expected_by_split: dict[str, list[list[int]]] = {"validation": [], "test": []}
    for row in selected:
        split = str(row["split"])
        if split in expected_by_split:
            expected_by_split[split].append(
                [maps.item[int(item)] for item in candidate_raw[str(row["example_id"])]]
            )

    candidate_identity: dict[str, Any] = {}
    for split, expected in expected_by_split.items():
        unisar_rows = emitted_unisar_candidates(Path(unisar_bundle["data_root"]), split)
        sesrec_rows = emitted_sesrec_candidates(
            Path(sesrec_bundle["data_root"]), split, args.neg_k + 1
        )
        match = expected == unisar_rows == sesrec_rows
        candidate_identity[split] = {
            "match": match,
            "rows": len(expected),
            "compact_candidate_sha256": digest_candidate_rows(expected),
        }

    token_universe = {
        int(token) for row in all_rows for token in row.get("query_tokens", [])
    }
    contract_report = json.loads(
        (args.reports_dir / "common_candidate_contract_20260710.json").read_text(
            encoding="utf-8"
        )
    )
    current_ledger_sha = file_sha256(args.candidate_ledger)
    if current_ledger_sha != contract_report["ledger_sha256"]:
        raise ValueError("candidate ledger hash differs from the registered contract report")
    all_match = all(result["match"] for result in candidate_identity.values())
    split_counts = dict(Counter(str(row["split"]) for row in selected))
    status = "PASS_COMMON_SURFACE_BUNDLE_PREFLIGHT" if all_match else "BLOCK_CANDIDATE_MISMATCH"
    payload = {
        "schema": "focalswap.r225.common_surface_preflight.v1",
        "status": status,
        "surface_id": "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED",
        "sample_rows": len(selected),
        "sample_split_counts": split_counts,
        "sample_info": sample_info,
        "sample_examples": str(sample_path),
        "candidate_count": args.neg_k + 1,
        "candidate_ledger": str(args.candidate_ledger),
        "candidate_ledger_sha256": current_ledger_sha,
        "candidate_identity": candidate_identity,
        "official_commits": {
            "sesrec": get_git_commit(sesrec_repo),
            "unisar": get_git_commit(unisar_repo),
        },
        "bundle_contracts": {
            "sesrec": sesrec_bundle,
            "unisar": unisar_bundle,
        },
        "full_cardinalities": {
            "users": len(all_rows),
            "raw_items": int(contract_report["train_item_universe"]),
            "query_tokens": len(token_universe),
            "source_sessions": len(all_rows),
            "unisar_user_resize_required": len(all_rows) >= 22_700,
        },
        "decision": (
            "Both adapters emit identical validation/test candidates from the canonical "
            "raw-item ledger. Proceed to same-scorer CPU forward smoke and runner review."
            if all_match
            else "Do not proceed: emitted candidate identities differ across adapters."
        ),
        "claim_boundary": (
            "This preflight validates shared rows/vocabulary/candidates only. Random-init "
            "or bundle-level checks are not trained-model evidence."
        ),
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{args.report_prefix}.json"
    md_path = args.reports_dir / f"{args.report_prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
