"""Parent-level branch-admission policies used by RePLAN.

This module is deliberately independent of vLLM.  It converts online parent
evidence into per-parent speculative leads; the serving adapter is responsible
only for admitting the resulting branches to stock vLLM.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Literal

RefillPolicy = Literal[
    "adaptive",
    "naive_refill",
    "tail_refill",
    "largest_evidence_refill",
    "remaining_budget_refill",
    "confidence_weighted_refill",
    "exact_tail_refill",
]


@dataclass(frozen=True)
class RefillPlan:
    """Desired speculative lead for every unresolved parent."""

    leads: tuple[int, ...]
    eligible_indices: tuple[int, ...]
    activation_evidence: int | None
    unused_slots: int


def plan_refill(
    evidence: list[int],
    *,
    policy: RefillPolicy,
    n_max: int,
    base_lead: int,
    available_width: int,
    activation_evidence: int | None = None,
    marginal_need_curves: list[tuple[float, ...]] | None = None,
    remainder_order: list[int] | None = None,
) -> RefillPlan:
    """Allocate spare branch slots for a method or its reviewer baselines.

    ``adaptive`` retains only the equal base allocation. ``naive_refill``
    splits spare capacity across every unresolved parent. ``tail_refill`` is
    RePLAN: only parents unresolved after the analytical midpoint (or an
    explicitly configured sensitivity boundary) receive an equal refill.
    """

    if not evidence:
        return RefillPlan((), (), None, available_width)
    if n_max <= 0 or base_lead <= 0 or available_width <= 0:
        raise ValueError("width and sampling limits must be positive")

    base = min(base_lead, n_max)
    leads = [base for _ in evidence]
    reserved = sum(leads)
    if available_width < reserved:
        raise ValueError("available width cannot cover the base allocation")

    if policy == "adaptive":
        return RefillPlan(
            tuple(leads),
            (),
            None,
            available_width - reserved,
        )
    if policy == "naive_refill":
        threshold = None
        eligible = list(range(len(evidence)))
    elif policy == "largest_evidence_refill":
        threshold = None
        eligible = sorted(
            range(len(evidence)), key=lambda index: (-evidence[index], index)
        )
        spare = available_width - reserved
        for index in eligible:
            extra = min(spare, n_max - leads[index])
            leads[index] += extra
            spare -= extra
            if spare == 0:
                break
        return RefillPlan(
            tuple(leads), tuple(eligible), threshold, spare
        )
    elif policy == "remaining_budget_refill":
        threshold = None
        eligible = list(range(len(evidence)))
        spare = available_width - reserved
        # One-slot weighted fair allocation is deterministic and exact at the
        # small branch widths used by the reviewer baseline.
        assigned = [0 for _ in evidence]
        weights = [max(0, n_max - count) for count in evidence]
        while spare > 0:
            candidates = [
                index
                for index in eligible
                if leads[index] < n_max and weights[index] > 0
            ]
            if not candidates:
                break
            index = min(
                candidates,
                key=lambda item: (
                    assigned[item] / weights[item],
                    item,
                ),
            )
            leads[index] += 1
            assigned[index] += 1
            spare -= 1
        return RefillPlan(
            tuple(leads), tuple(eligible), threshold, spare
        )
    elif policy in {
        "tail_refill",
        "confidence_weighted_refill",
        "exact_tail_refill",
    }:
        threshold = (
            math.ceil(n_max / 2)
            if activation_evidence is None
            else activation_evidence
        )
        if threshold <= 0 or threshold > n_max:
            raise ValueError("activation evidence must be in [1, n_max]")
        eligible = [
            index for index, count in enumerate(evidence) if count >= threshold
        ]
    else:
        raise ValueError(f"unknown refill policy: {policy}")

    if eligible:
        if policy == "confidence_weighted_refill":
            if marginal_need_curves is None or len(marginal_need_curves) != len(
                evidence
            ):
                raise ValueError(
                    "confidence-weighted refill requires one curve per parent"
                )
            spare = available_width - reserved
            heap: list[tuple[float, int, int, int]] = []
            for index in eligible:
                offset = base + 1
                curve = marginal_need_curves[index]
                if offset <= len(curve):
                    heapq.heappush(
                        heap,
                        (-curve[offset - 1], -evidence[index], index, offset),
                    )
            while spare > 0 and heap:
                _, _, index, offset = heapq.heappop(heap)
                leads[index] = offset
                spare -= 1
                next_offset = offset + 1
                curve = marginal_need_curves[index]
                if next_offset <= min(n_max, len(curve)):
                    heapq.heappush(
                        heap,
                        (
                            -curve[next_offset - 1],
                            -evidence[index],
                            index,
                            next_offset,
                        ),
                    )
        else:
            extra_per_parent = (available_width - reserved) // len(eligible)
            for index in eligible:
                leads[index] = min(n_max, leads[index] + extra_per_parent)
            if policy == "exact_tail_refill":
                spare = available_width - sum(leads)
                order = eligible if remainder_order is None else remainder_order
                if sorted(order) != sorted(eligible):
                    raise ValueError(
                        "exact refill requires a permutation of eligible parents"
                    )
                for index in order:
                    if spare == 0:
                        break
                    if leads[index] < n_max:
                        leads[index] += 1
                        spare -= 1

    return RefillPlan(
        tuple(leads),
        tuple(eligible),
        threshold,
        available_width - sum(leads),
    )


__all__ = ["RefillPlan", "RefillPolicy", "plan_refill"]
