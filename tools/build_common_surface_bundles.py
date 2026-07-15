#!/usr/bin/env python3
"""Build full SESRec and UniSAR bundles for the R225 controlled surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from official_adapter import (  # noqa: E402
    build_sesrec_trainer_bundle,
    build_unisar_tiny_bundle,
    get_git_commit,
)
from common_candidate_contract import load_candidate_ledger  # noqa: E402
from common_surface_preflight import (  # noqa: E402
    digest_candidate_rows,
    emitted_sesrec_candidates,
    emitted_unisar_candidates,
)


EXPECTED_COMMITS = {
    "sesrec": "de9c0ae10066704d2e97783ce2e6985a4dea508e",
    "unisar": "1a14a246066cd1aa6ee789c57ff2c7499f3d7379",
}
EXPECTED_ROWS = 77_661
EXPECTED_SPLITS = {"train": 68_057, "validation": 6_355, "test": 3_249}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/eligible_examples.jsonl",
    )
    parser.add_argument(
        "--candidate-ledger",
        type=Path,
        default=ROOT / "data/processed/r225_common_surface/candidates.jsonl.gz",
    )
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-prefix", default="r225_common_surface_bundles")
    parser.add_argument("--neg-k", type=int, default=99)
    parser.add_argument("--seed", type=int, default=20_260_710)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_assignments(path: Path, replacements: dict[str, int]) -> dict[str, Any]:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")
    for name, value in replacements.items():
        pattern = rf"({name}\s*=\s*)[^\n#]+"
        text, count = re.subn(pattern, rf"\g<1>{value}", text, count=1)
        if count != 1:
            raise RuntimeError(f"failed to patch {name} in {path}")
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "before_sha256": before, "after_sha256": sha256(path), "values": replacements}


def patch_unisar_kuaisar_assignments(path: Path, replacements: dict[str, int]) -> dict[str, Any]:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")
    start = text.index("def init_setting_KuaiSAR():")
    next_def = text.find("\ndef ", start + 1)
    end = len(text) if next_def < 0 else next_def
    prefix, block, suffix = text[:start], text[start:end], text[end:]
    for name, value in replacements.items():
        pattern = rf"({name}\s*=\s*)[^\n#]+"
        block, count = re.subn(pattern, rf"\g<1>{value}", block, count=1)
        if count != 1:
            raise RuntimeError(f"failed to patch KuaiSAR {name} in {path}")
    path.write_text(prefix + block + suffix, encoding="utf-8")
    return {"path": str(path), "before_sha256": before, "after_sha256": sha256(path), "values": replacements}


def patch_sesrec_seed_control(path: Path) -> dict[str, Any]:
    before = sha256(path)
    text = path.read_text(encoding="utf-8")
    old = "setup_seed(20210823)"
    new = "setup_seed(int(os.environ.get('SAR_SEED', '20210823')))"
    if new not in text:
        if old not in text:
            raise RuntimeError(f"cannot locate SESRec seed initialization in {path}")
        text = text.replace(old, new, 1)
        path.write_text(text, encoding="utf-8")
    return {
        "path": str(path),
        "before_sha256": before,
        "after_sha256": sha256(path),
        "control": "SAR_SEED environment variable; default remains 20210823",
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# R225 Full Common-Surface Bundles",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Rows: `{payload['rows']:,}`",
        f"- Splits: `{payload['split_counts']}`",
        f"- Candidate ledger SHA-256: `{payload['candidate_ledger_sha256']}`",
        f"- Validation candidate SHA-256: `{payload['candidate_identity']['validation']['sha256']}`",
        f"- Test candidate SHA-256: `{payload['candidate_identity']['test']['sha256']}`",
        "",
        "## Adapter Patches",
        "",
        "Only dataset-dependent embedding cardinalities and the exposed SESRec random seed are patched. "
        "No model layer, loss, forward path, or scorer is changed.",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.neg_k != 99:
        raise ValueError("R225 requires 99 negatives")
    sesrec_repo = args.official_root / "SESREC-SIGIR-2023"
    unisar_repo = args.official_root / "UniSAR"
    commits = {"sesrec": get_git_commit(sesrec_repo), "unisar": get_git_commit(unisar_repo)}
    if commits != EXPECTED_COMMITS:
        raise ValueError(f"official commit mismatch: {commits}")

    rows = read_jsonl(args.examples_path)
    split_counts = dict(Counter(str(row["split"]) for row in rows))
    if len(rows) != EXPECTED_ROWS or split_counts != EXPECTED_SPLITS:
        raise ValueError(f"R225 row/split mismatch: rows={len(rows)}, splits={split_counts}")
    candidate_raw = load_candidate_ledger(
        args.candidate_ledger,
        {str(row["example_id"]) for row in rows},
    )

    unisar_bundle = build_unisar_tiny_bundle(
        rows,
        unisar_repo,
        len(rows),
        args.neg_k,
        args.seed,
        score_rows_path=args.reports_dir / f"{args.report_prefix}_unisar_rows.csv",
        candidate_raw_by_example=candidate_raw,
        enforce_official_limits=False,
    )
    sesrec_bundle = build_sesrec_trainer_bundle(
        sesrec_repo,
        rows,
        len(rows),
        args.neg_k,
        args.seed,
        candidate_raw_by_example=candidate_raw,
    )

    patches = {
        "sesrec_const": patch_assignments(
            sesrec_repo / "config/const.py",
            {
                "item_id_num": int(sesrec_bundle["neutral_item_id"]) + 1,
                "user_id_num": int(sesrec_bundle["compact_users"]) + 1,
                "query_id_num": int(sesrec_bundle["neutral_session_id"]) + 1,
                "query_word_segs_num": int(sesrec_bundle["neutral_query_id"]) + 1,
            },
        ),
        "sesrec_seed": patch_sesrec_seed_control(sesrec_repo / "main.py"),
        "unisar_const": patch_unisar_kuaisar_assignments(
            unisar_repo / "utils/const.py",
            {
                "item_id_num": int(unisar_bundle["neutral_item_id"]) + 1,
                "user_id_num": int(unisar_bundle["compact_users"]) + 1,
                "word_id_num": int(unisar_bundle["compact_tokens"]) + 1,
            },
        ),
    }

    candidate_identity: dict[str, Any] = {}
    for split in ("validation", "test"):
        unisar_candidates = emitted_unisar_candidates(Path(unisar_bundle["data_root"]), split)
        sesrec_candidates = emitted_sesrec_candidates(
            Path(sesrec_bundle["data_root"]), split, args.neg_k + 1
        )
        match = unisar_candidates == sesrec_candidates
        candidate_identity[split] = {
            "match": match,
            "rows": len(unisar_candidates),
            "sha256": digest_candidate_rows(unisar_candidates),
        }
    all_match = all(result["match"] for result in candidate_identity.values())
    status = "PASS_FULL_COMMON_SURFACE_BUNDLES" if all_match else "BLOCK_CANDIDATE_MISMATCH"
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{args.report_prefix}.json"
    md_path = args.reports_dir / f"{args.report_prefix}.md"
    payload = {
        "schema": "focalswap.r225.full_common_surface_bundles.v1",
        "status": status,
        "surface_id": "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED",
        "rows": len(rows),
        "split_counts": split_counts,
        "candidate_ledger": str(args.candidate_ledger),
        "candidate_ledger_sha256": sha256(args.candidate_ledger),
        "official_commits": commits,
        "bundles": {"sesrec": sesrec_bundle, "unisar": unisar_bundle},
        "patches": patches,
        "candidate_identity": candidate_identity,
        "decision": (
            "Full bundles share exact validation/test candidates. Training may begin under the reviewed runner."
            if all_match
            else "Do not train: emitted candidates differ."
        ),
        "claim_boundary": (
            "R225 is controlled common-surface retraining. It is not an official-dataset or leaderboard reproduction."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"status": status, "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
