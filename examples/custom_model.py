"""Connect a small deterministic recommender to the FocalSwap Python API."""

import argparse
import json
from pathlib import Path

from focalswap import analyze, collect_scores, match_donors

ITEM_TERMS = {
    "camera": {"camera", "photo"},
    "lens": {"lens", "photo"},
    "novel": {"novel", "book"},
    "poetry": {"poetry", "book"},
}


def predict(row):
    """Return one score per candidate, in the supplied candidate order."""
    history = set(row["history"])
    latest = set(row["search"][-1]["query"].split()) if row["search"] else set()
    earlier = set()
    for episode in row["search"][:-1]:
        earlier.update(episode["query"].split())
    return [
        float(
            len(ITEM_TERMS[item] & history)
            + 2 * len(ITEM_TERMS[item] & latest)
            + 0.5 * len(ITEM_TERMS[item] & earlier)
        )
        for item in row["candidate_ids"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path(__file__).with_name("custom_dataset.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/custom_model"))
    args = parser.parse_args()
    rows = [
        json.loads(line) for line in args.data.read_text().splitlines() if line.strip()
    ]
    donors = match_donors(rows, ("category", "length_bucket"), seed=2026)
    by_id = {row["row_id"]: row for row in rows}

    def mask_latest(row):
        return row["search"][:-1] + [{"query": ""}] if row["search"] else None

    def mask_all(row):
        return [{"query": ""} for _ in row["search"]] if row["search"] else None

    def replace_latest(row):
        donor_id = donors[row["row_id"]]
        if (
            donor_id is None
            or not row["search"]
            or row["search"][-1] == by_id[donor_id]["search"][-1]
        ):
            return None
        return row["search"][:-1] + [by_id[donor_id]["search"][-1]]

    def replace_all(row):
        donor_id = donors[row["row_id"]]
        return None if donor_id is None else by_id[donor_id]["search"]

    readouts = collect_scores(
        rows,
        predict,
        {
            "latest_slot": {"C0": mask_latest, "C2": replace_latest},
            "all_source": {"C0": mask_all, "C2": replace_all},
        },
    )
    report = analyze(readouts, seed=2026, bootstrap_iterations=1000, family_size=2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "readouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in readouts)
    )
    (args.output_dir / "donors.json").write_text(json.dumps(donors, indent=2) + "\n")
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    for scope, result in report["scopes"].items():
        for control, estimate in result["controls"].items():
            metrics = estimate["metrics"]
            print(
                f"{scope} {control}: {estimate['rows']}/{result['total_rows']} rows, "
                f"RR difference {metrics['signed_contribution']['estimate']:+.4f}, "
                f"rank response {metrics['rank_response']['estimate']:.1%}"
            )
    print(f"Saved scores, donor mapping and audit report to {args.output_dir}")


if __name__ == "__main__":
    main()
