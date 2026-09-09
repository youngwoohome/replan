from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RuntimeLockTests(unittest.TestCase):
    def test_canonical_experiment_versions_are_locked(self) -> None:
        lock = json.loads(
            (ROOT / "artifact" / "runtime-lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            lock["vllm"]["commit"],
            "0b3ba88f165976e77ca5e6a7a3f5bba4562b80af",
        )
        self.assertEqual(lock["vllm"]["tag"], "v0.22.0")
        self.assertEqual(
            lock["vllm"]["repository"],
            "https://github.com/vllm-project/vllm.git",
        )
        self.assertEqual(lock["vllm"]["engine"], "V1")
        self.assertEqual(lock["python_packages"]["torch"], "2.11.0+cu130")
        self.assertEqual(lock["python_packages"]["transformers"], "5.14.1")
        self.assertEqual(lock["execution"]["max_model_len"], 8192)
        self.assertEqual(lock["execution"]["serving_max_num_seqs"], 256)
        self.assertEqual(lock["execution"]["gpu_memory_utilization"], 0.78)
        self.assertEqual(lock["execution"]["branch_bound"], 256)
        self.assertEqual(lock["method"]["confidence_rule"], "beta")
        self.assertEqual(lock["method"]["confidence_threshold"], 0.95)
        self.assertEqual(lock["method"]["minimum_evidence"], 4)
        self.assertEqual(lock["method"]["per_parent_lead"], "launched - evidence")
        self.assertEqual(
            lock["method"]["admission_quantum"],
            "ceil(sqrt(branch_bound))",
        )
        self.assertEqual(
            lock["primary_metrics"],
            [
                "bct_s",
                "mean_parent_e2e_s",
                "p95_parent_e2e_s",
                "deadline_attainment_fraction",
                "deadline_qualified_goodput_s",
            ],
        )

    def test_readme_uses_the_locked_stock_vllm_commit(self) -> None:
        lock = json.loads(
            (ROOT / "artifact" / "runtime-lock.json").read_text(
                encoding="utf-8"
            )
        )
        commit = lock["vllm"]["commit"]
        readme = (ROOT / "README.md").read_text()
        self.assertIn(commit, readme)
        self.assertFalse(any(ROOT.rglob("*.patch")))

    def test_runtime_metric_names_match_the_locked_paper_contract(self) -> None:
        lock = json.loads(
            (ROOT / "artifact" / "runtime-lock.json").read_text(
                encoding="utf-8"
            )
        )
        metric_sources = "\n".join(
            (ROOT / "replan" / name).read_text()
            for name in ("metrics.py", "reporting.py")
        )
        for metric in lock["primary_metrics"] + lock["diagnostic_metrics"]:
            self.assertIn(f'"{metric}"', metric_sources)

    def test_reported_runner_contains_only_thesis_experiment_sections(self) -> None:
        lock = json.loads(
            (ROOT / "artifact" / "runtime-lock.json").read_text(
                encoding="utf-8"
            )
        )
        runner = (
            ROOT / "experiments" / "reproduce" / "run_reported.sh"
        ).read_text()
        sections = re.findall(r"^  ([a-z-]+)\)$", runner, flags=re.MULTILINE)
        self.assertEqual(
            sections,
            lock["reported_experiments"]["fixed_cohort_sections"],
        )


if __name__ == "__main__":
    unittest.main()
