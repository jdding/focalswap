"""Analyze JSONL scores or target ranks exported by a recommender."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .audit import analyze


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, help="JSONL readouts with fixed candidates and named views"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Aggregate JSON report"
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--family-size",
        type=int,
        default=1,
        help="Number of tests in the Bonferroni family",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        action="append",
        metavar=("LEFT", "RIGHT"),
        help="Paired contribution comparison; repeat for multiple pairs (default: C0 C2)",
    )
    args = parser.parse_args()
    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output paths must differ.")
        content = args.input.read_bytes()
        readouts = [
            json.loads(line)
            for line in content.decode("utf-8").splitlines()
            if line.strip()
        ]
        result = analyze(
            readouts,
            bootstrap_iterations=args.bootstrap_iterations,
            seed=args.seed,
            confidence=args.confidence,
            family_size=args.family_size,
            comparisons=tuple(tuple(pair) for pair in args.compare)
            if args.compare
            else (("C0", "C2"),),
        )
        result["input_sha256"] = hashlib.sha256(content).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=args.output.parent,
                delete=False,
                suffix=".json",
            ) as handle:
                temporary = Path(handle.name)
                json.dump(result, handle, indent=2, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, args.output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"focalswap: {error}\n")
    print(f"Saved {len(result['scopes'])} audit scope(s) to {args.output}")


if __name__ == "__main__":
    main()
