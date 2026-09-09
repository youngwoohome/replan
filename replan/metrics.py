"""Paper metric definitions shared by serving and analysis."""

from __future__ import annotations

from typing import Any


def deadline_metrics(
    e2e_values: list[float],
    *,
    deadlines: tuple[int, ...],
    offered_rate: float | None,
) -> dict[str, Any]:
    """Compute deadline attainment and G(d) = lambda F(d)."""

    if not e2e_values:
        raise ValueError("deadline metrics require at least one parent")
    if offered_rate is not None and offered_rate <= 0.0:
        raise ValueError("offered rate must be positive")
    completed = {
        str(deadline): sum(value <= deadline for value in e2e_values)
        for deadline in deadlines
    }
    attainment = {
        deadline: count / len(e2e_values)
        for deadline, count in completed.items()
    }
    goodput = (
        {
            deadline: offered_rate * fraction
            for deadline, fraction in attainment.items()
        }
        if offered_rate is not None
        else None
    )
    return {
        "completed_by_deadline": completed,
        "deadline_attainment_fraction": attainment,
        "deadline_qualified_goodput_s": goodput,
    }


__all__ = ["deadline_metrics"]
