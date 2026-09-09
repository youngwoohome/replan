"""Deterministic parent-arrival schedules for open-loop experiments."""

from __future__ import annotations

import random


def arrival_offsets(
    count: int,
    *,
    mode: str,
    rate: float | None,
    seed: int,
    burst_size: int = 4,
) -> list[float]:
    """Return release times without exposing future arrivals to the policy."""

    if count <= 0:
        return []
    if mode == "batch":
        return [0.0] * count
    if rate is None or rate <= 0:
        raise ValueError("a positive arrival rate is required")
    if mode not in {"poisson", "bursty"}:
        raise ValueError(f"unsupported arrival mode: {mode}")
    if burst_size <= 0:
        raise ValueError("burst size must be positive")

    generator = random.Random(seed)
    offsets: list[float] = []
    current = 0.0
    if mode == "poisson":
        for index in range(count):
            if index:
                current += generator.expovariate(rate)
            offsets.append(current)
        return offsets

    # A burst is one arrival event. Scaling the event rate by burst size
    # preserves the requested long-run parent arrival rate.
    burst_rate = rate / burst_size
    while len(offsets) < count:
        if offsets:
            current += generator.expovariate(burst_rate)
        offsets.extend([current] * min(burst_size, count - len(offsets)))
    return offsets


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


__all__ = ["arrival_offsets", "percentile"]
