#!/usr/bin/env python3
"""Build the canonical R225 target-plus-99 candidate ledger.

The ledger is expressed in raw item ids before either model adapter performs
compact-id mapping. This prevents SESRec and UniSAR from silently auditing
different candidate sets on the proposed common surface.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAMPLES = ROOT / "data/processed/backbone_audit_full/examples.jsonl"
DEFAULT_LEDGER = ROOT / "data/processed/r225_common_surface/candidates.jsonl.gz"
DEFAULT_ELIGIBLE_EXAMPLES = ROOT / "data/processed/r225_common_surface/eligible_examples.jsonl"
DEFAULT_REPORTS = ROOT / "reports"
EXPECTED_SOURCE_ROWS = 96_994
EXPECTED_SOURCE_SPLITS = {"train": 68_057, "validation": 12_149, "test": 16_788}
EXPECTED_ELIGIBLE_SPLITS = {"train": 68_057, "validation": 6_355, "test": 3_249}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--eligible-examples-path", type=Path, default=DEFAULT_ELIGIBLE_EXAMPLES
    )
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--report-prefix", default="common_candidate_contract_20260710")
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_row_seed(seed: int, example_id: str) -> int:
    payload = f"{seed}|{example_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def unique_ints(values: Iterable[Any]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_candidates(
    row: dict[str, Any],
    all_items: list[int],
    allowed_items: set[int],
    neg_k: int,
    seed: int,
) -> list[int]:
    target = int(row["primary_target_item"])
    negatives: list[int] = []
    forbidden = {
        target,
        *[int(item) for item in row.get("H_pre", [])],
        *[int(item) for item in row.get("E_search_pos", [])],
        *[int(item) for item in row.get("Y_post_rec", [])],
    }
    seen = set(forbidden)
    for pool in (row.get("E_search_neg", []),):
        for item in unique_ints(pool):
            if item not in allowed_items:
                continue
            if item not in seen:
                seen.add(item)
                negatives.append(item)
                if len(negatives) == neg_k:
                    return [target, *negatives]

    row_seed = stable_row_seed(seed, str(row["example_id"]))
    rng = random.Random(row_seed)
    attempts = 0
    max_attempts = max(20 * neg_k, 2_000)
    while len(negatives) < neg_k and attempts < max_attempts:
        attempts += 1
        item = all_items[rng.randrange(len(all_items))]
        if item not in seen:
            seen.add(item)
            negatives.append(item)

    if len(negatives) < neg_k:
        start = row_seed % len(all_items)
        for offset in range(len(all_items)):
            item = all_items[(start + offset) % len(all_items)]
            if item not in seen:
                seen.add(item)
                negatives.append(item)
                if len(negatives) == neg_k:
                    break
    if len(negatives) != neg_k:
        raise RuntimeError(
            f"could not build {neg_k} unique negatives for example {row['example_id']}"
        )
    return [target, *negatives]


def load_candidate_ledger(
    path: Path,
    example_ids: set[str] | None = None,
) -> dict[str, list[int]]:
    candidates: dict[str, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = str(row["example_id"])
            if example_ids is None or example_id in example_ids:
                if example_id in candidates:
                    raise ValueError(f"duplicate candidate row for {example_id}")
                candidates[example_id] = [int(item) for item in row["candidates"]]
    if example_ids is not None:
        missing = sorted(example_ids - set(candidates))
        if missing:
            raise ValueError(f"candidate ledger missing {len(missing)} examples; first={missing[0]}")
    return candidates


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# R225 Common Candidate Contract",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Surface: `{payload['surface_id']}`",
        f"- Source rows: `{payload['source_rows']:,}`",
        f"- Eligible rows: `{payload['eligible_rows']:,}`",
        f"- Split counts: `{payload['split_counts']}`",
        f"- Split eligibility: `{payload['split_eligibility']}`",
        f"- Candidate count: `{payload['candidate_count']}`",
        f"- Training item universe: `{payload['train_item_universe']:,}`",
        f"- Source examples SHA-256: `{payload['source_examples_sha256']}`",
        f"- Eligible examples SHA-256: `{payload['eligible_examples_sha256']}`",
        f"- Ledger SHA-256: `{payload['ledger_sha256']}`",
        f"- Canonical content SHA-256: `{payload['content_sha256']}`",
        f"- Known-positive negatives: `{payload['known_positive_negative_violations']}`",
        f"- Negatives outside train universe: `{payload['nontrain_negative_violations']}`",
        "",
        "## Split Content Digests",
        "",
    ]
    for split, digest in payload["split_content_sha256"].items():
        lines.append(f"- `{split}`: `{digest}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Every R225 model bundle and score export must consume the eligible examples and this raw-item ledger. "
            "Model-local negative sampling is not valid for the common-surface comparison.",
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
        raise ValueError("R225 requires exactly 99 negatives")
    source_rows = read_jsonl(args.examples_path)
    source_split_counts = Counter(str(row.get("split", "")) for row in source_rows)
    if len(source_rows) != EXPECTED_SOURCE_ROWS or dict(source_split_counts) != EXPECTED_SOURCE_SPLITS:
        raise ValueError(
            "R225 source row/split mismatch: "
            f"rows={len(source_rows)}, splits={dict(source_split_counts)}"
        )

    train_item_universe: set[int] = set()
    for row in source_rows:
        if str(row["split"]) != "train":
            continue
        train_item_universe.update(int(item) for item in row.get("H_pre", []))
        train_item_universe.update(int(item) for item in row.get("E_search_all", []))
        train_item_universe.update(int(item) for item in row.get("Y_post_rec", []))
    all_items = sorted(train_item_universe)
    if len(all_items) <= args.neg_k:
        raise ValueError(f"item universe too small: {len(all_items)}")

    rows: list[dict[str, Any]] = []
    for source_row_index, row in enumerate(source_rows):
        split = str(row["split"])
        target_seen = int(row["primary_target_item"]) in train_item_universe
        if split == "train" or target_seen:
            eligible = dict(row)
            eligible["r225_source_row_index"] = source_row_index
            rows.append(eligible)
    split_counts = Counter(str(row["split"]) for row in rows)
    if dict(split_counts) != EXPECTED_ELIGIBLE_SPLITS:
        raise ValueError(f"R225 eligible split mismatch: {dict(split_counts)}")
    split_eligibility = {
        split: {
            "source": int(source_split_counts[split]),
            "eligible": int(split_counts[split]),
            "dropped": int(source_split_counts[split] - split_counts[split]),
            "coverage": float(split_counts[split] / source_split_counts[split]),
        }
        for split in EXPECTED_ELIGIBLE_SPLITS
    }
    args.eligible_examples_path.parent.mkdir(parents=True, exist_ok=True)
    with args.eligible_examples_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    args.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    content_digest = hashlib.sha256()
    split_digests = {split: hashlib.sha256() for split in EXPECTED_ELIGIBLE_SPLITS}
    known_positive_negative_violations = 0
    nontrain_negative_violations = 0
    with args.ledger_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            for row_index, row in enumerate(rows):
                candidates = build_candidates(
                    row,
                    all_items,
                    train_item_universe,
                    args.neg_k,
                    args.seed,
                )
                if candidates[0] != int(row["primary_target_item"]):
                    raise RuntimeError(f"target is not first for row {row_index}")
                if len(candidates) != args.neg_k + 1 or len(set(candidates)) != len(candidates):
                    raise RuntimeError(f"candidate uniqueness failed for row {row_index}")
                known_positive = {
                    *[int(item) for item in row.get("H_pre", [])],
                    *[int(item) for item in row.get("E_search_pos", [])],
                    *[int(item) for item in row.get("Y_post_rec", [])],
                }
                known_positive_negative_violations += len(
                    set(candidates[1:]) & known_positive
                )
                nontrain_negative_violations += sum(
                    int(item not in train_item_universe) for item in candidates[1:]
                )
                ledger_row = {
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "example_id": str(row["example_id"]),
                    "row_index": row_index,
                    "source_row_index": int(row["r225_source_row_index"]),
                    "split": str(row["split"]),
                    "target_item": int(row["primary_target_item"]),
                    "user_id": int(row["user_id"]),
                }
                line = (json.dumps(ledger_row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
                gzip_handle.write(line)
                content_digest.update(line)
                split_digests[ledger_row["split"]].update(line)

    verify = load_candidate_ledger(args.ledger_path)
    if len(verify) != len(rows):
        raise RuntimeError(f"ledger verification rows={len(verify)} expected={len(rows)}")
    if known_positive_negative_violations or nontrain_negative_violations:
        raise RuntimeError(
            "candidate contamination detected: "
            f"known_positive={known_positive_negative_violations}, "
            f"nontrain={nontrain_negative_violations}"
        )

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{args.report_prefix}.json"
    md_path = args.reports_dir / f"{args.report_prefix}.md"
    payload = {
        "schema": "focalswap.r225.common_candidate_contract.v1",
        "status": "PASS_COMMON_CANDIDATE_CONTRACT",
        "surface_id": "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED",
        "source_rows": len(source_rows),
        "eligible_rows": len(rows),
        "split_counts": dict(split_counts),
        "split_eligibility": split_eligibility,
        "candidate_count": args.neg_k + 1,
        "train_item_universe": len(all_items),
        "seed": args.seed,
        "source_examples_path": str(args.examples_path),
        "source_examples_sha256": file_sha256(args.examples_path),
        "eligible_examples_path": str(args.eligible_examples_path),
        "eligible_examples_sha256": file_sha256(args.eligible_examples_path),
        "ledger_path": str(args.ledger_path),
        "ledger_sha256": file_sha256(args.ledger_path),
        "content_sha256": content_digest.hexdigest(),
        "split_content_sha256": {
            split: split_digests[split].hexdigest() for split in EXPECTED_ELIGIBLE_SPLITS
        },
        "known_positive_negative_violations": known_positive_negative_violations,
        "nontrain_negative_violations": nontrain_negative_violations,
        "claim_boundary": (
            "Training retains every source-train row; validation/test use the fixed "
            "train-seen-target eligibility rule. Negatives come only from the training "
            "item universe and exclude row history, search positives, and post-rec positives. "
            "This is a shared data/candidate contract only, not a model result, "
            "an official-dataset reproduction, or evidence of architecture differences."
        ),
        "outputs": {"json": str(json_path), "md": str(md_path)},
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(md_path, payload)
    print(json.dumps({"status": payload["status"], "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
