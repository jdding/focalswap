import ast
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from focalswap import analyze, collect_scores, match_donors, target_rank

ROOT = Path(__file__).resolve().parents[1]


def row(index=0, full=1, mask=2, donor=3, user=None):
    return {
        "row_id": str(index),
        "user_id": str(index) if user is None else user,
        "target_id": "b",
        "candidate_ids": ["a", "b", "c"],
        "scope": "latest_slot",
        "views": {
            "full": {"rank": full},
            "C0": {"rank": mask},
            "C2": None if donor is None else {"rank": donor},
        },
    }


class AuditTests(unittest.TestCase):
    def test_midpoint_rank_and_target_position(self):
        self.assertEqual(target_rank([2, 1, 1], 1), 2.5)
        self.assertEqual(target_rank([1, 1, 1], 2), 2)
        with self.assertRaises(ValueError):
            target_rank([1, 2], True)
        exported = row()
        exported["views"] = {"full": [2, 1, 1], "C0": [1, 1, 1]}
        metrics = analyze([exported], bootstrap_iterations=20)["scopes"]["latest_slot"][
            "controls"
        ]["C0"]["metrics"]
        self.assertAlmostEqual(metrics["full_rr"]["estimate"], 0.4)
        self.assertIsNone(metrics["full_rr"]["ci_low"])

    def test_rank_response_and_zero_mean_are_separate(self):
        result = analyze([row(0, 1, 2, 2), row(1, 2, 1, 1)], bootstrap_iterations=20)
        metrics = result["scopes"]["latest_slot"]["controls"]["C2"]["metrics"]
        self.assertEqual(metrics["signed_contribution"]["estimate"], 0)
        self.assertEqual(metrics["rank_response"]["estimate"], 1)
        self.assertEqual(metrics["replacement_preference"]["estimate"], 0.5)

    def test_common_support_and_coverage(self):
        rows = [row(0, 1, 2, 3), row(1, 2, 3, None), row(2, 3, 1, 2)]
        result = analyze(rows, bootstrap_iterations=100, family_size=2)
        self.assertAlmostEqual(result["interval_confidence"], 0.975)
        scope = result["scopes"]["latest_slot"]
        self.assertEqual(scope["controls"]["C0"]["rows"], 3)
        self.assertEqual(scope["controls"]["C2"]["rows"], 2)
        paired = scope["paired"][0]
        self.assertEqual(paired["rows"], 2)
        self.assertAlmostEqual(paired["coverage"], 2 / 3)
        self.assertAlmostEqual(
            paired["metrics"]["contribution_difference"]["estimate"],
            (-1 / 6 - 1 / 2) / 2,
        )

    def test_no_eligible_donors(self):
        result = analyze([row(donor=None)], bootstrap_iterations=20)["scopes"][
            "latest_slot"
        ]
        self.assertEqual(
            result["controls"]["C2"]["interval_status"], "no_eligible_rows"
        )
        self.assertEqual(result["paired"][0]["rows"], 0)

    def test_replay_order_independence_and_ties(self):
        rows = [row(0, 1, 1, 1, "a"), row(1, 2, 3, 1, "a"), row(2, 3, 1, 2, "b")]
        self.assertEqual(
            analyze(rows, bootstrap_iterations=80),
            analyze(rows[::-1], bootstrap_iterations=80),
        )
        self.assertEqual(
            analyze([row(full=2, mask=2, donor=2)], bootstrap_iterations=20)["scopes"][
                "latest_slot"
            ]["controls"]["C2"]["metrics"]["replacement_preference"]["estimate"],
            0.5,
        )

    def test_reject_invalid_inputs(self):
        invalid = []
        for value in (0, 4, 1.2, float("nan"), True):
            invalid.append(row(full=value))
        item = row()
        item["views"]["C0"] = [float("inf"), 1, 2]
        invalid.append(item)
        item = row()
        item["candidate_ids"] = ["a", "b", "b"]
        invalid.append(item)
        item = row()
        item["target_id"] = "missing"
        invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                analyze([item], bootstrap_iterations=20)
        with self.assertRaises(ValueError):
            analyze([row(), row()], bootstrap_iterations=20)
        with self.assertRaises(ValueError):
            analyze([], bootstrap_iterations=20)
        changed = row()
        changed["scope"] = "all_source"
        changed["candidate_ids"].reverse()
        with self.assertRaises(ValueError):
            analyze([row(), changed], bootstrap_iterations=20)

    def test_cluster_bootstrap_matches_paper_implementation(self):
        source = ast.parse((ROOT / "tools/control_ladder_analysis.py").read_text())
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name == "cluster_bootstrap"
        )
        namespace = {"np": np}
        exec(  # noqa: S102 -- trusted function extracted from this repository for parity.
            compile(
                ast.Module(body=[function], type_ignores=[]), "paper_bootstrap", "exec"
            ),
            namespace,
        )
        cluster_bootstrap = namespace["cluster_bootstrap"]
        rows = [
            row(0, 1, 2, 3, "a"),
            row(1, 2, 3, 1, "a"),
            row(2, 3, 1, 2, "b"),
            row(3, 1, 3, 2, "c"),
        ]
        values = np.array(
            [
                1 / r["views"]["full"]["rank"] - 1 / r["views"]["C0"]["rank"]
                for r in rows
            ]
        )
        expected = cluster_bootstrap(
            values, np.array([r["user_id"] for r in rows]), 300, 42
        )
        actual = analyze(rows, bootstrap_iterations=300, seed=42)["scopes"][
            "latest_slot"
        ]["controls"]["C0"]["metrics"]["signed_contribution"]
        np.testing.assert_allclose(
            [actual["estimate"], actual["ci_low"], actual["ci_high"]],
            expected[:3],
            rtol=0,
            atol=1e-14,
        )

    def test_custom_model_preserves_inputs_and_matches_exports(self):
        spec = importlib.util.spec_from_file_location(
            "custom_model", ROOT / "examples/custom_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rows = [
            json.loads(line)
            for line in (ROOT / "examples/custom_dataset.jsonl")
            .read_text()
            .splitlines()
        ]
        before = copy.deepcopy(rows)
        readouts = collect_scores(
            rows,
            module.predict,
            {"latest_slot": {"C0": lambda r: r["search"][:-1] + [{"query": ""}]}},
        )
        self.assertEqual(rows, before)
        report = analyze(readouts, bootstrap_iterations=20)
        ranked = copy.deepcopy(readouts)
        for item in ranked:
            target = item["candidate_ids"].index(item["target_id"])
            item["views"] = {
                key: {"rank": target_rank(values, target)}
                for key, values in item["views"].items()
            }
        self.assertEqual(analyze(ranked, bootstrap_iterations=20), report)

        def bad_scorer(payload):
            payload["candidate_ids"].reverse()
            return [1, 2, 3, 4]

        with self.assertRaises(ValueError):
            collect_scores(
                rows, bad_scorer, {"latest_slot": {"C0": lambda r: r["search"]}}
            )

    def test_donors_use_match_fields_and_do_not_reuse(self):
        rows = [
            json.loads(line)
            for line in (ROOT / "examples/custom_dataset.jsonl")
            .read_text()
            .splitlines()
        ]
        original = copy.deepcopy(rows)
        mapping = match_donors(rows, ("category", "length_bucket"))
        self.assertEqual(
            mapping, match_donors(rows[::-1], ("category", "length_bucket"))
        )
        donors = [value for value in mapping.values() if value is not None]
        self.assertTrue(donors)
        self.assertEqual(len(donors), len(set(donors)))
        by_id = {r["row_id"]: r for r in rows}
        for source, donor in mapping.items():
            if donor is not None:
                self.assertNotEqual(source, donor)
                self.assertNotEqual(by_id[source]["user_id"], by_id[donor]["user_id"])
                self.assertEqual(by_id[source]["match"], by_id[donor]["match"])
        self.assertEqual(rows, original)
        self.assertEqual(
            match_donors(rows[:1], ("category",)), {rows[0]["row_id"]: None}
        )

    def test_cli_report_and_invalid_input_leave_output_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "scores.jsonl"
            output = Path(directory) / "audit.json"
            source.write_text(json.dumps(row()) + "\n")
            command = [
                sys.executable,
                "-m",
                "focalswap",
                str(source),
                "--output",
                str(output),
                "--bootstrap-iterations",
                "20",
            ]
            subprocess.run(command, check=True, capture_output=True)
            report = json.loads(output.read_text())
            self.assertEqual(report["schema"], "focalswap.audit.v1")
            self.assertEqual(len(report["input_sha256"]), 64)
            original = output.read_bytes()
            source.write_text("{invalid}\n")
            result = subprocess.run(command, check=False, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
