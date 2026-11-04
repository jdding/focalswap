"""Audit three synthetic rank exports without loading a model."""

import json

from focalswap import analyze

rows = [
    {
        "row_id": str(index),
        "user_id": str(index),
        "target_id": "a",
        "candidate_ids": ["a", "b", "c"],
        "scope": "all_source",
        "views": {"full": {"rank": full}, "C0": {"rank": mask}, "C2": {"rank": donor}},
    }
    for index, (full, mask, donor) in enumerate([(1, 3, 2), (2, 3, 1), (3, 2, 3)])
]
print(json.dumps(analyze(rows, bootstrap_iterations=1000), indent=2))
