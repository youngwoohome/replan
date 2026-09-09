import json
import sys

import pytest

from experiments.analysis.summarize_open_loop import main


def _result(*, goodput: float, attainment: float) -> dict:
    metrics = {
        "accuracy": 0.5,
        "maximum_observed_outstanding_branches": 128,
        "parents": 2,
        "bct_s": 10.0,
        "mean_parent_e2e_s": 5.0,
        "p95_parent_e2e_s": 8.0,
        "output_token_throughput_s": 100.0,
        "completed_by_deadline": {"600": int(2 * attainment)},
        "deadline_attainment_fraction": {"600": attainment},
        "deadline_qualified_goodput_s": {"600": goodput},
    }
    return {
        "config": {
            "strict_outstanding_cap": True,
            "max_outstanding_branches": 256,
            "arrival_rate_parents_s": 0.4,
        },
        "metrics": metrics,
        "parents": [
            {"parent_id": "p0", "selected_answer": "1", "correct": True},
            {"parent_id": "p1", "selected_answer": "2", "correct": False},
        ],
    }


def test_open_loop_summary_uses_deadline_qualified_goodput(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    baseline = root / "mixed_p1each_n128_poisson_load1_adaptive_seed1.json"
    treatment = root / "mixed_p1each_n128_poisson_load1_tail_refill_seed1.json"
    baseline.write_text(json.dumps(_result(goodput=0.2, attainment=0.5)))
    treatment.write_text(json.dumps(_result(goodput=0.3, attainment=0.75)))
    output = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys, "argv", ["summarize-open-loop", "--root", str(root), "--output", str(output)]
    )

    main()

    condition = json.loads(output.read_text())["conditions"][0]
    deadline = condition["deadline_qualified_goodput_s"]["600"]
    assert deadline["adaptive_mean"] == 0.2
    assert deadline["tail_refill_mean"] == 0.3
    assert deadline["mean_relative_change_pct"] == pytest.approx(50.0)
