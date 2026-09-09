from itertools import pairwise

import pytest

from replan.controller import plan_refill
from replan.policy import (
    StreamingParent,
    beta_leader_probability,
)


def test_streaming_parent_stops_without_waiting_for_launched_siblings() -> None:
    parent = StreamingParent(n_max=128)
    assert parent.confidence_rule == "beta"
    parent.commit_launch(64)
    for candidate_index in range(4):
        parent.record(candidate_index, "7")

    decision = parent.decide()
    assert decision.stop
    assert decision.selected_answer == "7"
    assert parent.completed == 4
    assert parent.launched == 64


def test_streaming_parent_can_refill_one_slot_asynchronously() -> None:
    parent = StreamingParent(n_max=8)
    parent.commit_launch(4)
    for candidate_index, answer in enumerate(("1", "2", "3", "4")):
        parent.record(candidate_index, answer)

    decision = parent.decide()
    assert not decision.stop
    assert decision.next_branches == 1
    parent.commit_launch()
    assert parent.launched == 5


def test_streaming_parent_waits_for_minimum_expansion_evidence() -> None:
    parent = StreamingParent(n_max=128)
    parent.commit_launch(64)
    for candidate_index, answer in enumerate(("1", "2", "3")):
        parent.record(candidate_index, answer)

    decision = parent.decide()
    assert not decision.stop
    assert decision.next_branches == 0


def test_streaming_parent_ignores_out_of_order_short_completions() -> None:
    parent = StreamingParent(n_max=128)
    parent.commit_launch(64)
    for candidate_index in (20, 21, 22, 23):
        parent.record(candidate_index, "7")

    decision = parent.decide()
    assert not decision.stop
    assert decision.next_branches == 0


def test_completion_order_uses_out_of_order_completions() -> None:
    parent = StreamingParent(
        n_max=128,
        confidence_rule="beta",
        evidence_order="completion",
    )
    parent.commit_launch(64)
    for candidate_index in (20, 21, 22, 23):
        parent.record(candidate_index, "7")

    decision = parent.decide()
    assert decision.stop
    assert decision.selected_answer == "7"
    assert parent.evidence == 4


def test_streaming_parent_rejects_unknown_evidence_order() -> None:
    with pytest.raises(ValueError, match="evidence order"):
        StreamingParent(n_max=8, evidence_order="runtime")


def test_none_answer_is_completed_evidence_not_a_missing_candidate() -> None:
    parent = StreamingParent(n_max=8)
    parent.commit_launch(4)
    for candidate_index, answer in enumerate((None, "1", "2", "3")):
        parent.record(candidate_index, answer)

    decision = parent.decide()

    assert not decision.stop
    assert decision.next_branches == 1


def test_adaptive_consistency_beta_probability() -> None:
    assert beta_leader_probability(["1"] * 9 + ["2"]) == pytest.approx(2036 / 2048)
    assert beta_leader_probability(["1"] * 4) == pytest.approx(31 / 32)
    assert beta_leader_probability(["1"] * 4 + ["2"] * 4) == pytest.approx(0.5)


def test_beta_parent_stops_at_first_confident_prefix() -> None:
    parent = StreamingParent(n_max=16, confidence_rule="beta")
    parent.commit_launch(8)
    for candidate_index, answer in enumerate(("1", "1", "1", "1", "2", "2", "2", "2")):
        parent.record(candidate_index, answer)

    decision = parent.decide()
    assert decision.stop
    assert decision.selected_answer == "1"
    assert decision.consensus_lower_bound == pytest.approx(31 / 32)


def test_streaming_parent_bounds_speculative_lead() -> None:
    parent = StreamingParent(n_max=128, max_speculative_lead=4)
    parent.commit_launch(4)
    for candidate_index, answer in enumerate(("1", "2", "3", "4")):
        parent.record(candidate_index, answer)

    for _ in range(4):
        decision = parent.decide()
        assert decision.next_branches == 1
        parent.commit_launch()
    assert parent.launched == 8
    assert parent.decide().next_branches == 0


def test_tail_refill_uses_only_free_width() -> None:
    plan = plan_refill(
        [16, 63, 64, 80],
        policy="tail_refill",
        n_max=128,
        base_lead=16,
        available_width=256,
    )

    assert plan.leads == (16, 16, 112, 112)
    assert plan.eligible_indices == (2, 3)
    assert plan.activation_evidence == 64


def test_tail_refill_waits_until_half_budget() -> None:
    assert plan_refill(
        [16, 63],
        policy="tail_refill",
        n_max=128,
        base_lead=16,
        available_width=256,
    ).leads == (16, 16)


def test_tail_refill_respects_available_width() -> None:
    assert plan_refill(
        [64, 64],
        policy="tail_refill",
        n_max=128,
        base_lead=16,
        available_width=96,
    ).leads == (48, 48)


def test_tail_refill_is_noop_for_n1() -> None:
    assert plan_refill(
        [1, 1],
        policy="tail_refill",
        n_max=1,
        base_lead=1,
        available_width=256,
    ).leads == (1, 1)


def test_tail_refill_preserves_strict_equality() -> None:
    plan = plan_refill(
        [64, 64, 64],
        policy="tail_refill",
        n_max=128,
        base_lead=16,
        available_width=100,
    )

    assert plan.leads == (33, 33, 33)
    assert sum(plan.leads) == 99
    assert plan.unused_slots == 1


def test_tail_refill_stops_at_n_max() -> None:
    plan = plan_refill(
        [64, 64],
        policy="tail_refill",
        n_max=64,
        base_lead=16,
        available_width=256,
    )

    assert plan.leads == (64, 64)


def test_tail_refill_rejects_width_below_base_allocation() -> None:
    with pytest.raises(ValueError, match="base allocation"):
        plan_refill(
            [64, 64],
            policy="tail_refill",
            n_max=128,
            base_lead=16,
            available_width=31,
        )


def test_naive_refill_allocates_to_every_unresolved_parent() -> None:
    plan = plan_refill(
        [16, 63, 64, 80],
        policy="naive_refill",
        n_max=128,
        base_lead=16,
        available_width=256,
    )

    assert plan.leads == (64, 64, 64, 64)
    assert plan.eligible_indices == (0, 1, 2, 3)
    assert plan.activation_evidence is None


def test_adaptive_only_does_not_reallocate_spare_slots() -> None:
    plan = plan_refill(
        [64, 80],
        policy="adaptive",
        n_max=128,
        base_lead=16,
        available_width=96,
    )

    assert plan.leads == (16, 16)
    assert plan.unused_slots == 64


def test_largest_evidence_refill_greedily_prioritizes_progress() -> None:
    plan = plan_refill(
        [16, 80, 64],
        policy="largest_evidence_refill",
        n_max=128,
        base_lead=16,
        available_width=128,
    )

    assert plan.leads == (16, 96, 16)
    assert plan.unused_slots == 0


def test_remaining_budget_refill_prioritizes_more_remaining_work() -> None:
    plan = plan_refill(
        [16, 112],
        policy="remaining_budget_refill",
        n_max=128,
        base_lead=16,
        available_width=64,
    )

    assert plan.leads[0] > plan.leads[1]
    assert sum(plan.leads) == 64


def test_tail_boundary_can_be_overridden_for_sensitivity_only() -> None:
    plan = plan_refill(
        [31, 32, 63],
        policy="tail_refill",
        n_max=128,
        base_lead=16,
        available_width=128,
        activation_evidence=32,
    )

    assert plan.eligible_indices == (1, 2)
    assert plan.activation_evidence == 32


def test_midpoint_does_not_depend_on_the_base_lead() -> None:
    plan = plan_refill(
        [64, 63],
        policy="tail_refill",
        n_max=128,
        base_lead=80,
        available_width=192,
    )

    assert plan.eligible_indices == (0,)
    assert plan.activation_evidence == 64


def test_exact_tail_refill_consumes_integer_remainder() -> None:
    plan = plan_refill(
        [64, 64, 64],
        policy="exact_tail_refill",
        n_max=128,
        base_lead=16,
        available_width=100,
    )

    assert plan.leads == (34, 33, 33)
    assert plan.unused_slots == 0
    assert max(plan.leads) - min(plan.leads) == 1


def test_exact_tail_refill_accepts_cumulative_fair_order() -> None:
    plan = plan_refill(
        [64, 64, 64],
        policy="exact_tail_refill",
        n_max=128,
        base_lead=16,
        available_width=100,
        remainder_order=[2, 0, 1],
    )

    assert plan.leads == (33, 33, 34)


def test_confidence_weighted_refill_uses_marginal_need() -> None:
    plan = plan_refill(
        [64, 64],
        policy="confidence_weighted_refill",
        n_max=128,
        base_lead=2,
        available_width=7,
        marginal_need_curves=[
            (0.9, 0.8, 0.7, 0.6),
            (0.9, 0.2, 0.1, 0.0),
        ],
    )

    assert plan.leads == (4, 3)
    assert sum(plan.leads) == 7


def test_confidence_weighted_refill_breaks_ties_by_evidence_depth() -> None:
    plan = plan_refill(
        [64, 80],
        policy="confidence_weighted_refill",
        n_max=128,
        base_lead=2,
        available_width=5,
        marginal_need_curves=[(0.9, 0.8, 0.7), (0.9, 0.8, 0.7)],
    )

    assert plan.leads == (2, 3)


def test_streaming_parent_exposes_causal_marginal_need_curve() -> None:
    parent = StreamingParent(n_max=16, confidence_rule="beta")
    parent.commit_launch(8)
    for candidate_index, answer in enumerate(("1", "2", "1", "2")):
        parent.record(candidate_index, answer)

    curve = parent.marginal_need_curve()

    assert len(curve) == 12
    assert all(left >= right for left, right in pairwise(curve))
