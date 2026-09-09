"""Summarize paired strict-cap open-loop runs and validate invariants."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

NAME = re.compile(
    r"mixed_p(?P<parents>\d+)each_n(?P<n>\d+)_"
    r"(?P<mode>poisson|bursty)_load(?P<load>[^_]+)_"
    r"(?P<policy>adaptive|tail_refill)_seed(?P<seed>\d+)\.json$"
)
METRICS = (
    "bct_s",
    "mean_parent_e2e_s",
    "p95_parent_e2e_s",
    "output_token_throughput_s",
)


def confidence_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [values[0], values[0]]
    # Student-t 97.5th percentiles for the seed counts used by this artifact.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    radius = critical * stdev(values) / math.sqrt(len(values))
    return [mean(values) - radius, mean(values) + radius]


def deadline_metrics(result: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Read the deadline metrics emitted by the current serving runner."""

    metrics = result["metrics"]
    return (
        {
            key: float(value)
            for key, value in metrics["deadline_attainment_fraction"].items()
        },
        {
            key: float(value)
            for key, value in metrics["deadline_qualified_goodput_s"].items()
        },
    )


def selection_signature(result: dict) -> list[tuple[str, str | None, bool]]:
    """Return the quality decision that paired scheduling must preserve."""

    return sorted(
        (
            str(parent["parent_id"]),
            parent.get("selected_answer"),
            bool(parent["correct"]),
        )
        for parent in result["parents"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = defaultdict(dict)
    for path in args.root.rglob("*.json"):
        match = NAME.match(path.name)
        if not match:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        config, metrics = result["config"], result["metrics"]
        assert config["strict_outstanding_cap"] is True
        assert metrics["maximum_observed_outstanding_branches"] <= int(
            config["max_outstanding_branches"]
        )
        key = (
            int(match.group("parents")),
            int(match.group("n")),
            match.group("mode"),
            match.group("load"),
            int(match.group("seed")),
        )
        runs[key][match.group("policy")] = result

    grouped = defaultdict(list)
    for key, policies in runs.items():
        if set(policies) != {"adaptive", "tail_refill"}:
            continue
        parents, n_max, mode, load, seed = key
        adaptive, tail = policies["adaptive"], policies["tail_refill"]
        assert adaptive["metrics"]["accuracy"] == tail["metrics"]["accuracy"]
        assert selection_signature(adaptive) == selection_signature(tail)
        row = {"seed": seed}
        for metric in METRICS:
            baseline = float(adaptive["metrics"][metric])
            method = float(tail["metrics"][metric])
            row[f"{metric}_adaptive"] = baseline
            row[f"{metric}_tail_refill"] = method
            row[f"{metric}_relative_change_pct"] = 100.0 * (
                method / baseline - 1.0
            )
        adaptive_attainment, adaptive_goodput = deadline_metrics(adaptive)
        tail_attainment, tail_goodput = deadline_metrics(tail)
        for deadline, baseline in adaptive_goodput.items():
            method = float(tail_goodput[deadline])
            baseline = float(baseline)
            row[f"deadline_qualified_goodput_{deadline}_adaptive"] = baseline
            row[f"deadline_qualified_goodput_{deadline}_tail_refill"] = method
            row[f"deadline_qualified_goodput_{deadline}_relative_change_pct"] = (
                100.0 * (method / baseline - 1.0) if baseline > 0.0 else None
            )
            row[f"deadline_attainment_{deadline}_delta"] = (
                tail_attainment[deadline] - adaptive_attainment[deadline]
            )
        grouped[(parents, n_max, mode, load)].append(row)

    summary = []
    for (parents, n_max, mode, load), rows in sorted(grouped.items()):
        item = {
            "parents_per_dataset": parents,
            "total_parents": parents * 3,
            "n_max": n_max,
            "arrival_mode": mode,
            "load": load,
            "seeds": len(rows),
            "paired_runs": rows,
            "mean_relative_change_pct": {},
            "relative_change_95pct_ci": {},
        }
        for metric in METRICS:
            values = [row[f"{metric}_relative_change_pct"] for row in rows]
            item["mean_relative_change_pct"][metric] = mean(values)
            item["relative_change_95pct_ci"][metric] = confidence_interval(values)
        deadlines = sorted(
            key.removeprefix("deadline_qualified_goodput_").removesuffix(
                "_adaptive"
            )
            for key in rows[0]
            if key.startswith("deadline_qualified_goodput_")
            and key.endswith("_adaptive")
        )
        item["deadline_qualified_goodput_s"] = {}
        for deadline in deadlines:
            changes = [
                row[
                    f"deadline_qualified_goodput_{deadline}_relative_change_pct"
                ]
                for row in rows
            ]
            finite_changes = [value for value in changes if value is not None]
            item["deadline_qualified_goodput_s"][deadline] = {
                "adaptive_mean": mean(
                    row[f"deadline_qualified_goodput_{deadline}_adaptive"]
                    for row in rows
                ),
                "tail_refill_mean": mean(
                    row[f"deadline_qualified_goodput_{deadline}_tail_refill"]
                    for row in rows
                ),
                "mean_relative_change_pct": (
                    mean(finite_changes) if finite_changes else None
                ),
                "relative_change_95pct_ci": (
                    confidence_interval(finite_changes)
                    if finite_changes
                    else None
                ),
            }
        summary.append(item)

    if not summary:
        raise RuntimeError("no complete strict-cap pairs found")
    output = {"conditions": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
