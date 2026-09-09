"""Derived serving configuration fixed by the paper contract."""

from __future__ import annotations


def resolve_base_lead(
    *,
    arrival_mode: str,
    branch_bound: int,
    parent_count: int,
    n_max: int,
    reference_concurrency: int,
) -> int:
    """Resolve the fixed base lead without observing future arrivals."""

    if arrival_mode == "batch":
        base_concurrency = parent_count
    elif reference_concurrency > 0:
        base_concurrency = reference_concurrency
    else:
        raise ValueError(
            "--reference-concurrency is required for open-loop serving"
        )
    return max(1, min(n_max, branch_bound // max(1, base_concurrency)))


__all__ = ["resolve_base_lead"]
