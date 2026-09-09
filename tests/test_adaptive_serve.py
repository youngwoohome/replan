import json
import sys

import pytest

from replan.cli import parse_args
from replan.config import resolve_base_lead
from replan.metrics import deadline_metrics
from replan.replay import load_quality_traces


def test_load_quality_trace_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    row = {"parent_id": "p0", "candidate_index": 0}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_quality_traces([f"math={path}"])


def test_deadline_metrics_match_paper_goodput_definition() -> None:
    metrics = deadline_metrics(
        [1.0, 5.0], deadlines=(2, 10), offered_rate=0.5
    )

    assert metrics["completed_by_deadline"] == {"2": 1, "10": 2}
    assert metrics["deadline_attainment_fraction"] == {"2": 0.5, "10": 1.0}
    assert metrics["deadline_qualified_goodput_s"] == {"2": 0.25, "10": 0.5}


def test_batch_deadline_goodput_is_undefined_without_offered_rate() -> None:
    metrics = deadline_metrics(
        [1.0, 5.0], deadlines=(2, 10), offered_rate=None
    )

    assert metrics["deadline_attainment_fraction"] == {"2": 0.5, "10": 1.0}
    assert metrics["deadline_qualified_goodput_s"] is None


def test_batch_derives_base_lead_from_submitted_parent_count() -> None:
    assert resolve_base_lead(
        arrival_mode="batch",
        branch_bound=256,
        parent_count=36,
        n_max=128,
        reference_concurrency=0,
    ) == 7


def test_open_loop_requires_the_paper_base_lead_explicitly() -> None:
    with pytest.raises(ValueError, match="reference-concurrency"):
        resolve_base_lead(
            arrival_mode="poisson",
            branch_bound=256,
            parent_count=36,
            n_max=128,
            reference_concurrency=0,
        )
    assert resolve_base_lead(
        arrival_mode="poisson",
        branch_bound=256,
        parent_count=36,
        n_max=128,
        reference_concurrency=36,
    ) == 7


def test_tail_refill_is_available(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "adaptive-bon-serve",
            "--input-root",
            str(tmp_path),
            "--dataset",
            "math",
            "--parents-per-dataset",
            "1",
            "--policy",
            "tail_refill",
            "--output",
            str(tmp_path / "result.json"),
            "--model",
            "model",
        ],
    )

    assert parse_args().policy == "tail_refill"


def test_naive_refill_ablation_is_available(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replan-serve",
            "--input-root",
            str(tmp_path),
            "--dataset",
            "math",
            "--parents-per-dataset",
            "1",
            "--policy",
            "naive_refill",
            "--output",
            str(tmp_path / "result.json"),
            "--model",
            "model",
        ],
    )

    assert parse_args().policy == "naive_refill"
