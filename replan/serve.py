"""Run progressive quality-aware BoN on one vLLM GPU."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from replan.answers import extract_answer
from replan.arrival import arrival_offsets
from replan.cli import parse_args
from replan.config import resolve_base_lead
from replan.controller import plan_refill
from replan.policy import StreamingParent
from replan.reporting import (
    RefillTelemetry,
    RunTelemetry,
    build_result,
    write_result,
)
from replan.replay import (
    dataset_root,
    load_metadata,
    load_quality_traces,
    replay_sampling_params,
    sampling_params,
    token_hash,
)
from replan.workload import QualityParent, load_quality_workload


@dataclass
class LiveParent:
    spec: QualityParent
    order: int
    controller: StreamingParent
    inflight: set[int] = field(default_factory=set)
    completed: dict[int, dict[str, Any]] = field(default_factory=dict)
    completion_s: float | None = None
    selected_answer: str | None = None
    aborted_branches: int = 0
    arrival_offset_s: float = 0.0
    released: bool = False
    admission_s: float | None = None


def _adaptive_ready(live: LiveParent) -> int | None:
    if live.completion_s is not None:
        return None
    decision = live.controller.decide()
    if decision.stop or decision.next_branches <= 0:
        return None
    return decision.next_branches


def main() -> None:
    args = parse_args()
    policy = args.policy
    refill_enabled = policy in {
        "naive_refill",
        "tail_refill",
        "largest_evidence_refill",
        "remaining_budget_refill",
        "confidence_weighted_refill",
        "exact_tail_refill",
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if (
        args.n_max <= 0
        or args.max_outstanding_branches <= 0
        or args.reference_concurrency < 0
    ):
        raise ValueError(
            "N and branch bound must be positive; reference concurrency "
            "must be non-negative"
        )
    if args.tail_activation_evidence < 0 or args.tail_activation_evidence > args.n_max:
        raise ValueError("tail activation evidence must be 0 or in [1, n_max]")
    tail_policies = {
        "tail_refill",
        "confidence_weighted_refill",
        "exact_tail_refill",
    }
    if args.tail_activation_evidence and policy not in tail_policies:
        raise ValueError("tail activation evidence applies only to tail policies")
    metadata_by_dataset = {
        dataset: load_metadata(args.input_root, dataset, args.n_max)
        for dataset in args.dataset
    }
    parent_specs: list[QualityParent] = []
    for dataset in args.dataset:
        selected_root = dataset_root(args.input_root, dataset)
        parents = load_quality_workload(
            selected_root / f"n{args.n_max}" / "traces",
            selected_root / "prompts",
            max_parents=args.parents_per_dataset,
        )
        parent_specs.extend(parents)
    offsets = arrival_offsets(
        len(parent_specs),
        mode=args.arrival_mode,
        rate=args.arrival_rate,
        seed=args.arrival_seed,
        burst_size=args.burst_size,
    )
    quality_rows = load_quality_traces(args.quality_trace)
    if quality_rows:
        for parent in parent_specs:
            for candidate in parent.candidates[: args.n_max]:
                key = (parent.dataset, parent.parent_id, candidate.index)
                if key not in quality_rows:
                    raise ValueError(f"quality trace is missing {key}")
    initial_wave = resolve_base_lead(
        arrival_mode=args.arrival_mode,
        branch_bound=args.max_outstanding_branches,
        parent_count=len(parent_specs),
        n_max=args.n_max,
        reference_concurrency=args.reference_concurrency,
    )
    if (
        args.arrival_mode == "batch"
        and initial_wave * len(parent_specs) > args.max_outstanding_branches
    ):
        raise ValueError("initial waves exceed outstanding branch capacity")
    refill_quantum = math.ceil(math.sqrt(args.max_outstanding_branches))

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLMEngine, SamplingParams
    from vllm.engine.arg_utils import EngineArgs

    engine = LLMEngine.from_engine_args(
        EngineArgs(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_chunked_prefill=True,
            enforce_eager=args.enforce_eager,
            disable_log_stats=False,
            # Some evaluated text-only models expose multimodal model classes.
            # Do not reserve image/video profiling capacity that we never use.
            limit_mm_per_prompt={"image": 0, "video": 0},
        ),
        enable_multiprocessing=False,
    )
    live_parents = [
        LiveParent(
            spec=parent,
            order=order,
            controller=StreamingParent(
                args.n_max,
                confidence_rule=args.confidence_rule,
                max_speculative_lead=initial_wave,
                evidence_order=args.evidence_order,
            ),
            arrival_offset_s=offsets[order],
        )
        for order, parent in enumerate(parent_specs)
    ]
    by_id = {parent.spec.parent_id: parent for parent in live_parents}
    admission_cursor = 0
    tail_refill_steps = 0
    maximum_tail_lead = initial_wave
    refilled_parent_ids: set[str] = set()
    refill_policy_evaluations = 0
    refill_policy_time_s = 0.0
    cached_refill_signature: tuple[Any, ...] | None = None
    cached_refill_leads: tuple[int, ...] = ()
    request_lookup: dict[str, tuple[str, int]] = {}
    outstanding: set[str] = set()
    maximum_outstanding = 0
    arrival_queue_peak = 0
    admission_events = 0
    fair_deficits: dict[str, float] = {}
    fair_signature: tuple[Any, ...] | None = None
    fair_remainder_order: tuple[str, ...] = ()
    remainder_allocation_events = 0
    remainder_slots_assigned = 0
    maximum_remainder = 0

    def exact_remainder_order(
        active_parents: list[LiveParent], available_width: int
    ) -> list[int]:
        """Rotate indivisible slots using cumulative allocation deficits."""

        nonlocal fair_signature, fair_remainder_order
        nonlocal remainder_allocation_events, remainder_slots_assigned
        nonlocal maximum_remainder
        threshold = args.tail_activation_evidence or math.ceil(args.n_max / 2)
        eligible = [
            live
            for live in active_parents
            if live.controller.evidence >= threshold
        ]
        if not eligible:
            fair_signature = (available_width, ())
            fair_remainder_order = ()
            return []
        signature = (
            available_width,
            tuple(live.spec.parent_id for live in eligible),
        )
        if signature != fair_signature:
            reserved = initial_wave * len(active_parents)
            spare = max(0, available_width - reserved)
            common, remainder = divmod(spare, len(eligible))
            if common >= args.n_max - initial_wave:
                remainder = 0
            share = remainder / len(eligible)
            for live in eligible:
                fair_deficits[live.spec.parent_id] = (
                    fair_deficits.get(live.spec.parent_id, 0.0) + share
                )
            ranked = sorted(
                eligible,
                key=lambda live: (
                    -fair_deficits[live.spec.parent_id],
                    live.order,
                ),
            )
            for live in ranked[:remainder]:
                fair_deficits[live.spec.parent_id] -= 1.0
            fair_remainder_order = tuple(
                live.spec.parent_id for live in ranked
            )
            fair_signature = signature
            if remainder:
                remainder_allocation_events += 1
                remainder_slots_assigned += remainder
                maximum_remainder = max(maximum_remainder, remainder)
        active_index = {
            live.spec.parent_id: index for index, live in enumerate(active_parents)
        }
        return [active_index[parent_id] for parent_id in fair_remainder_order]

    def add_branch(live: LiveParent, branch_index: int) -> None:
        nonlocal maximum_outstanding
        if len(outstanding) >= args.max_outstanding_branches:
            raise RuntimeError("strict outstanding-branch cap would be exceeded")
        candidate = live.spec.candidates[branch_index]
        request_id = f"quality-{live.spec.parent_id}-b{branch_index}"
        quality_row = quality_rows.get(
            (live.spec.dataset, live.spec.parent_id, branch_index)
        )
        params = (
            replay_sampling_params(
                metadata_by_dataset[live.spec.dataset],
                candidate.seed,
                int(quality_row["actual_tokens"]),
                SamplingParams,
            )
            if quality_row is not None
            else sampling_params(
                metadata_by_dataset[live.spec.dataset],
                candidate.seed,
                SamplingParams,
            )
        )
        engine.add_request(request_id, live.spec.prompt_input, params)
        request_lookup[request_id] = (live.spec.parent_id, branch_index)
        outstanding.add(request_id)
        live.inflight.add(branch_index)
        maximum_outstanding = max(maximum_outstanding, len(outstanding))

    def fill_adaptive(*, force: bool = False) -> None:
        nonlocal admission_cursor, maximum_tail_lead, tail_refill_steps
        nonlocal refill_policy_evaluations, refill_policy_time_s
        nonlocal cached_refill_signature, cached_refill_leads
        active_parents = [
            live
            for live in live_parents
            if live.admission_s is not None and live.completion_s is None
        ]
        # A released parent's base admission has priority over speculative
        # tail refill. Already-running tail branches drain naturally, while
        # admitted parents retain only their base lead until the queue clears.
        has_waiting_parent = args.arrival_mode != "batch" and any(
            live.released and live.admission_s is None for live in live_parents
        )
        if has_waiting_parent:
            maximum_active_parents = (
                args.max_outstanding_branches // min(initial_wave, args.n_max)
            )
            if len(active_parents) < maximum_active_parents:
                # Leave the partial free capacity untouched until enough slots
                # exist to admit the oldest waiting parent's complete base wave.
                return
            for live in active_parents:
                live.controller.set_max_speculative_lead(initial_wave)
        if active_parents and refill_enabled and not has_waiting_parent:
            available_width = args.max_outstanding_branches
            evidence = [
                live.controller.evidence for live in active_parents
            ]
            remainder_order = (
                exact_remainder_order(active_parents, available_width)
                if policy == "exact_tail_refill"
                else None
            )
            signature = (
                available_width,
                tuple(remainder_order or ()),
                tuple(
                    (live.spec.parent_id, live.controller.evidence)
                    for live in active_parents
                ),
            )
            if signature == cached_refill_signature:
                leads = cached_refill_leads
            else:
                policy_start = time.perf_counter()
                refill_plan = plan_refill(
                    evidence,
                    policy=policy,
                    n_max=args.n_max,
                    base_lead=initial_wave,
                    available_width=available_width,
                    activation_evidence=(
                        args.tail_activation_evidence or None
                    ),
                    marginal_need_curves=(
                        [
                            live.controller.marginal_need_curve()
                            for live in active_parents
                        ]
                        if policy == "confidence_weighted_refill"
                        else None
                    ),
                    remainder_order=remainder_order,
                )
                leads = refill_plan.leads
                refill_policy_time_s += time.perf_counter() - policy_start
                refill_policy_evaluations += 1
                cached_refill_signature = signature
                cached_refill_leads = leads
            for live, lead in zip(active_parents, leads):
                live.controller.set_max_speculative_lead(lead)
                if lead > initial_wave:
                    refilled_parent_ids.add(live.spec.parent_id)
                    maximum_tail_lead = max(maximum_tail_lead, lead)
            if any(lead > initial_wave for lead in leads):
                tail_refill_steps += 1
        free_slots = args.max_outstanding_branches - len(outstanding)
        if outstanding and not force and free_slots < refill_quantum:
            return
        while len(outstanding) < args.max_outstanding_branches:
            ready: list[tuple[LiveParent, int]] = []
            for live in active_parents:
                count = _adaptive_ready(live)
                if count is None:
                    continue
                if count <= args.max_outstanding_branches - len(outstanding):
                    ready.append((live, count))
            if not ready:
                return
            live, count = min(
                ready,
                key=lambda item: (
                    (item[0].order - admission_cursor) % len(live_parents),
                    item[0].order,
                ),
            )
            admission_cursor = (live.order + 1) % len(live_parents)
            start = live.controller.launched
            live.controller.commit_launch(count)
            for branch_index in range(start, start + count):
                add_branch(live, branch_index)

    start_time = time.perf_counter()

    def admit_released(elapsed: float) -> bool:
        """Release due parents and admit their base waves in FIFO order."""

        nonlocal arrival_queue_peak, admission_events
        for live in live_parents:
            if not live.released and live.arrival_offset_s <= elapsed:
                live.released = True

        waiting = sorted(
            (
                live
                for live in live_parents
                if live.released and live.admission_s is None
            ),
            key=lambda live: (live.arrival_offset_s, live.order),
        )
        arrival_queue_peak = max(arrival_queue_peak, len(waiting))
        admitted = False
        seed_count = min(initial_wave, args.n_max)
        maximum_active_parents = args.max_outstanding_branches // seed_count
        active_count = sum(
            live.admission_s is not None and live.completion_s is None
            for live in live_parents
        )
        while (
            waiting
            and active_count < maximum_active_parents
            and len(outstanding) + seed_count <= args.max_outstanding_branches
        ):
            live = waiting.pop(0)
            live.admission_s = elapsed
            live.controller.commit_launch(seed_count)
            for branch_index in range(seed_count):
                add_branch(live, branch_index)
            active_count += 1
            admission_events += 1
            admitted = True
        return admitted

    if args.arrival_mode == "batch":
        for live in live_parents:
            live.released = True
            live.admission_s = 0.0
        seed_count = min(initial_wave, args.n_max)
        for live in live_parents:
            live.controller.commit_launch(seed_count)
        for branch_index in range(seed_count):
            for live in live_parents:
                add_branch(live, branch_index)
        fill_adaptive()
    else:
        admit_released(0.0)
        fill_adaptive(force=True)

    while any(live.completion_s is None for live in live_parents):
        elapsed = time.perf_counter() - start_time
        admit_released(elapsed)
        if not engine.has_unfinished_requests():
            for live in live_parents:
                if (
                    live.admission_s is None
                    or live.completion_s is not None
                    or live.inflight
                ):
                    continue
                decision = live.controller.decide()
                if decision.stop:
                    live.selected_answer = decision.selected_answer
                    live.completion_s = elapsed
            if not any(live.completion_s is None for live in live_parents):
                break
            fill_adaptive(force=True)
            if not engine.has_unfinished_requests():
                pending_offsets = [
                    live.arrival_offset_s
                    for live in live_parents
                    if not live.released
                ]
                waiting = [
                    live
                    for live in live_parents
                    if live.released and live.admission_s is None
                ]
                if pending_offsets or waiting:
                    if pending_offsets and not waiting:
                        wait_s = max(0.0, min(pending_offsets) - elapsed)
                        time.sleep(min(wait_s, 0.01))
                    continue
                stalled = [
                    {
                        "parent_id": live.spec.parent_id,
                        "launched": live.controller.launched,
                        "completed": live.controller.completed,
                        "inflight": len(live.inflight),
                        "decision": live.controller.decide(),
                    }
                    for live in live_parents
                    if (
                        live.admission_s is not None
                        and live.completion_s is None
                    )
                ]
                raise RuntimeError(
                    f"unfinished parents have no runnable branches: {stalled}"
                )
        outputs = engine.step()
        elapsed = time.perf_counter() - start_time
        touched: set[str] = set()
        for request_output in outputs:
            if not request_output.finished:
                continue
            parent_id, branch_index = request_lookup[request_output.request_id]
            live = by_id[parent_id]
            sample = request_output.outputs[0]
            quality_row = quality_rows.get(
                (live.spec.dataset, live.spec.parent_id, branch_index)
            )
            if quality_row is not None:
                expected_tokens = int(quality_row["actual_tokens"])
                if len(sample.token_ids) != expected_tokens:
                    raise RuntimeError(
                        f"replay length mismatch for {parent_id}-b{branch_index}"
                    )
                answer = quality_row.get("extracted_answer")
            else:
                answer = extract_answer(sample.text)
            row = {
                "candidate_index": branch_index,
                "seed": live.spec.candidates[branch_index].seed,
                "output_tokens": len(sample.token_ids),
                "answer": answer,
                "token_ids_sha256": token_hash(sample.token_ids),
                "completion_s": elapsed,
            }
            live.completed[branch_index] = row
            live.inflight.remove(branch_index)
            outstanding.remove(request_output.request_id)
            touched.add(parent_id)
            live.controller.record(branch_index, answer)

        for parent_id in touched:
            live = by_id[parent_id]
            decision = live.controller.decide()
            if decision.stop:
                live.selected_answer = decision.selected_answer
                live.completion_s = elapsed
                aborted_ids = [
                    f"quality-{live.spec.parent_id}-b{branch_index}"
                    for branch_index in live.inflight
                ]
                if aborted_ids:
                    engine.abort_request(aborted_ids)
                    for request_id in aborted_ids:
                        outstanding.remove(request_id)
                    live.aborted_branches += len(aborted_ids)
                    live.inflight.clear()
        admit_released(elapsed)
        fill_adaptive()

    bct = time.perf_counter() - start_time
    telemetry = RunTelemetry(
        maximum_outstanding=maximum_outstanding,
        arrival_queue_peak=arrival_queue_peak,
        admission_events=admission_events,
        refill=RefillTelemetry(
            enabled=refill_enabled,
            tail_policies=frozenset(tail_policies),
            base_lead=initial_wave,
            steps=tail_refill_steps,
            refilled_parent_count=len(refilled_parent_ids),
            maximum_tail_lead=maximum_tail_lead,
            policy_evaluations=refill_policy_evaluations,
            policy_time_s=refill_policy_time_s,
            remainder_allocation_events=remainder_allocation_events,
            remainder_slots_assigned=remainder_slots_assigned,
            maximum_remainder=maximum_remainder,
        ),
    )
    result = build_result(
        args=args,
        policy=policy,
        live_parents=live_parents,
        bct=bct,
        telemetry=telemetry,
        quality_replay=bool(quality_rows),
    )
    write_result(result, args.output)


if __name__ == "__main__":
    main()
