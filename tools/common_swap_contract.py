#!/usr/bin/env python3
"""Build and load the frozen four-level R225 swap-donor ledger."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from carryover_oracle import load_item_categories  # noqa: E402
from unisar_preflight import build_swap_lookups  # noqa: E402
from common_candidate_contract import file_sha256  # noqa: E402


LEVELS = (
    "l1_nuisance",
    "l2_same_category",
    "category_broken_source",
    "same_category_source",
)
DEFAULT_EXAMPLES = ROOT / "data/processed/r225_common_surface/eligible_examples.jsonl"
DEFAULT_LEDGER = ROOT / "data/processed/r225_common_surface/swap_donors.jsonl.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-path", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument(
        "--item-metadata",
        type=Path,
        default=ROOT / "data/processed/item_metadata.parquet",
    )
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--report-prefix", default="common_swap_contract_20260710")
    parser.add_argument("--seed", type=int, default=20_260_710)
    parser.add_argument("--swap-tries", type=int, default=80)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_swap_ledger(
    path: Path,
    rows: list[dict[str, Any]],
    eval_indices: list[int],
) -> dict[str, dict[int, int | None]]:
    expected = set(eval_indices)
    lookups: dict[str, dict[int, int | None]] = {level: {} for level in LEVELS}
    seen: set[int] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            row_index = int(item["row_index"])
            if row_index not in expected:
                continue
            if row_index in seen:
                raise ValueError(f"duplicate swap ledger row {row_index}")
            if str(rows[row_index]["example_id"]) != str(item["example_id"]):
                raise ValueError(f"swap ledger example mismatch at row {row_index}")
            seen.add(row_index)
            for level in LEVELS:
                donor = int(item[f"{level}_swap_row_index"])
                lookups[level][row_index] = donor if donor >= 0 else None
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"swap ledger missing {len(missing)} eval rows; first={missing[0]}")
    return lookups


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.examples_path)
    split_counts = {
        split: sum(str(row["split"]) == split for row in rows)
        for split in ("train", "validation", "test")
    }
    expected_counts = {"train": 68_057, "validation": 6_355, "test": 3_249}
    if split_counts != expected_counts:
        raise ValueError(f"eligible split mismatch: {split_counts}")
    eval_indices = [index for index, row in enumerate(rows) if str(row["split"]) == "test"]
    item_cat = load_item_categories(args.item_metadata)
    lookups, summary = build_swap_lookups(
        rows,
        eval_indices,
        item_cat,
        args.seed,
        args.swap_tries,
    )

    args.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    content_digest = hashlib.sha256()
    with args.ledger_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            for row_index in eval_indices:
                item: dict[str, Any] = {
                    "example_id": str(rows[row_index]["example_id"]),
                    "row_index": row_index,
                    "source_row_index": int(rows[row_index]["r225_source_row_index"]),
                }
                for level in LEVELS:
                    donor = lookups[level][row_index]
                    item[f"{level}_swap_row_index"] = int(donor) if donor is not None else -1
                    item[f"{level}_swap_example_id"] = (
                        str(rows[int(donor)]["example_id"]) if donor is not None else ""
                    )
                line = (json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n").encode(
                    "utf-8"
                )
                gzip_handle.write(line)
                content_digest.update(line)

    verified = load_swap_ledger(args.ledger_path, rows, eval_indices)
    if any(verified[level] != lookups[level] for level in LEVELS):
        raise RuntimeError("swap ledger round-trip mismatch")
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.reports_dir / f"{args.report_prefix}.json"
    md_path = args.reports_dir / f"{args.report_prefix}.md"
    payload = {
        "schema": "focalswap.r225.common_swap_contract.v1",
        "status": "PASS_COMMON_SWAP_CONTRACT",
        "surface_id": "K_COMMON_QBRIDGE_R225_SEEN_TARGET_CONTROLLED",
        "rows": len(rows),
        "test_rows": len(eval_indices),
        "seed": args.seed,
        "swap_tries": args.swap_tries,
        "examples_path": str(args.examples_path),
        "examples_sha256": file_sha256(args.examples_path),
        "ledger_path": str(args.ledger_path),
        "ledger_sha256": file_sha256(args.ledger_path),
        "content_sha256": content_digest.hexdigest(),
        "levels": list(LEVELS),
        "coverage": summary,
        "claim_boundary": (
            "This ledger freezes counterfactual donors for cross-model control; it is not a model result."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# R225 Common Swap Contract",
        "",
        f"Status: `{payload['status']}`",
        "",
        f"- Test rows: `{payload['test_rows']:,}`",
        f"- Seed / tries: `{args.seed}` / `{args.swap_tries}`",
        f"- Ledger SHA-256: `{payload['ledger_sha256']}`",
        "",
        "## Coverage",
        "",
    ]
    for level in LEVELS:
        value = summary[level]
        lines.append(
            f"- `{level}`: {value['matched']:,}/{value['attempted']:,} ({value['coverage']:.3f})"
        )
    lines.extend(["", "## Claim Boundary", "", payload["claim_boundary"], ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
