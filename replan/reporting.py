"""Build the paper-aligned JSON result after a serving run completes."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from replan.answers import numeric_answer_is_correct
from replan.arrival import percentile
from replan.metrics import deadline_metrics


@dataclass(frozen=True)
class RefillTelemetry:
    enabled: bool
    tail_policies: frozenset[str]
    base_lead: int
    steps: int
    refilled_parent_count: int
    maximum_tail_lead: int
    policy_evaluations: int
    policy_time_s: float
    remainder_allocation_events: int
    remainder_slots_assigned: int
    maximum_remainder: int


@dataclass(frozen=True)
class RunTelemetry:
    maximum_outstanding: int
    arrival_queue_peak: int
    admission_events: int
    refill: RefillTelemetry


def build_result(
    *,
    args: Any,
    policy: str,
    live_parents: list[Any],
    bct: float,
    telemetry: RunTelemetry,
    quality_replay: bool,
) -> dict[str, Any]:
    parent_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for live in live_parents:
        assert live.completion_s is not None
        correct = numeric_answer_is_correct(
            live.selected_answer, live.spec.reference_answer
        )
        parent_rows.append(
            {
                "parent_id": live.spec.parent_id,
                "dataset": live.spec.dataset,
                "completion_s": live.completion_s,
                "arrival_s": live.arrival_offset_s,
                "admission_s": live.admission_s,
                "queue_delay_s": (
                    float(live.admission_s) - live.arrival_offset_s
                    if live.admission_s is not None
                    else 0.0
                ),
                "e2e_s": live.completion_s - live.arrival_offset_s,
                "executed_branches": len(live.completed),
                "launched_branches": live.controller.launched,
                "aborted_branches": live.aborted_branches,
                "output_tokens": sum(
                    int(row["output_tokens"]) for row in live.completed.values()
                ),
                "selected_answer": live.selected_answer,
                "reference_answer": live.spec.reference_answer,
                "correct": correct,
            }
        )
        branch_rows.extend(
            {"parent_id": live.spec.parent_id, **row}
            for _, row in sorted(live.completed.items())
        )

    e2e_values = [float(row["e2e_s"]) for row in parent_rows]
    deadline_values = deadline_metrics(
        e2e_values,
        deadlines=(30, 60, 120, 180, 240, 360, 600),
        offered_rate=(
            args.arrival_rate if args.arrival_mode != "batch" else None
        ),
    )
    refill = telemetry.refill
    return {
        "format": "replan_online_v2",
        "status": "completed",
        "config": {
            "policy": policy,
            "serving_policy": "stock",
            "confidence_rule": args.confidence_rule,
            "evidence_order": args.evidence_order,
            "datasets": args.dataset,
            "parents_per_dataset": args.parents_per_dataset,
            "n_max": args.n_max,
            "initial_wave": refill.base_lead,
            "initial_wave_source": (
                "submitted_parent_count"
                if args.arrival_mode == "batch"
                else "reference_concurrency"
            ),
            "reference_concurrency": (
                len(live_parents)
                if args.arrival_mode == "batch"
                else args.reference_concurrency
            ),
            "max_outstanding_branches": args.max_outstanding_branches,
            "refill_quantum": math.ceil(
                math.sqrt(args.max_outstanding_branches)
            ),
            "quality_replay": quality_replay,
            "enable_prefix_caching": args.enable_prefix_caching,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "refill_policy": policy if refill.enabled else "none",
            "arrival_mode": args.arrival_mode,
            "arrival_rate_parents_s": args.arrival_rate,
            "arrival_seed": args.arrival_seed,
            "burst_size": args.burst_size,
            "strict_outstanding_cap": True,
        },
        "metrics": {
            "bct_s": bct,
            "parents": len(parent_rows),
            "accuracy": sum(row["correct"] for row in parent_rows)
            / len(parent_rows),
            "executed_branches": len(branch_rows),
            "launched_branches": sum(
                int(row["launched_branches"]) for row in parent_rows
            ),
            "aborted_branches": sum(
                int(row["aborted_branches"]) for row in parent_rows
            ),
            "output_tokens": sum(
                int(row["output_tokens"]) for row in branch_rows
            ),
            "mean_parent_e2e_s": sum(e2e_values) / len(parent_rows),
            "p95_parent_e2e_s": percentile(e2e_values, 0.95),
            "output_token_throughput_s": sum(
                int(row["output_tokens"]) for row in branch_rows
            )
            / bct,
            "maximum_observed_outstanding_branches": (
                telemetry.maximum_outstanding
            ),
            "arrival_queue_peak": telemetry.arrival_queue_peak,
            "admission_events": telemetry.admission_events,
            **deadline_values,
        },
        "parents": parent_rows,
        "branches": branch_rows,
        "refill": {
            "enabled": refill.enabled,
            "policy": policy if refill.enabled else "none",
            "activation_evidence": (
                args.tail_activation_evidence or math.ceil(args.n_max / 2)
                if policy in refill.tail_policies
                else None
            ),
            "base_lead": refill.base_lead,
            "refill_steps": refill.steps,
            "refilled_parents": refill.refilled_parent_count,
            "maximum_tail_lead": refill.maximum_tail_lead,
            "policy_evaluations": refill.policy_evaluations,
            "policy_time_s": refill.policy_time_s,
            "remainder_allocation_events": refill.remainder_allocation_events,
            "remainder_slots_assigned": refill.remainder_slots_assigned,
            "maximum_remainder": refill.maximum_remainder,
        },
    }


def write_result(result: dict[str, Any], output: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


__all__ = ["RefillTelemetry", "RunTelemetry", "build_result", "write_result"]
