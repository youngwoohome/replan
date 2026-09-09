from pathlib import Path

from experiments.analysis.persistent_tail_diagnostics import (
    is_paper_persistent_pair,
    is_persistent_stop_depth,
    paired_result_paths,
)


def test_parent_stopping_at_midpoint_is_not_persistent() -> None:
    assert not is_persistent_stop_depth(64, 128)
    assert is_persistent_stop_depth(65, 128)
    assert is_persistent_stop_depth(128, 128)


def test_diagnostics_find_current_nested_result_layout(tmp_path: Path) -> None:
    condition = tmp_path / "naive_amc_n128"
    condition.mkdir()
    adaptive = condition / "adaptive.r1.json"
    tail = condition / "tail_refill.r1.json"
    adaptive.touch()
    tail.touch()

    assert paired_result_paths(tmp_path) == [(adaptive, tail, 1)]


def test_paper_diagnostics_select_only_individual_frozen_n128_pairs() -> None:
    config = {
        "n_max": 128,
        "max_outstanding_branches": 256,
        "arrival_mode": "batch",
        "quality_replay": True,
        "datasets": ["fasttts_amc2023"],
        "parents_per_dataset": 40,
    }
    adaptive = {"config": {**config, "policy": "adaptive"}}
    tail = {"config": {**config, "policy": "tail_refill"}}

    assert is_paper_persistent_pair(adaptive, tail)
    adaptive["config"]["quality_replay"] = False
    assert not is_paper_persistent_pair(adaptive, tail)
