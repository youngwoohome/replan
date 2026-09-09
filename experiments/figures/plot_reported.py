#!/usr/bin/env python3
"""Regenerate the two quantitative figures used in the results chapter."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUNTIME = re.compile(
    r"Avg generation throughput: (?P<tps>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)
POLICIES = {
    "adaptive": ("Adaptive-only", "#6f7d8a"),
    "remaining_budget_refill": ("Remaining budget", "#d95f02"),
    "largest_evidence_refill": ("Largest evidence", "#2a9d6f"),
    "tail_refill": ("RePLAN", "#2673b8"),
}
DEADLINES = (120, 240, 360, 600)


def policy_from_name(path: Path) -> str | None:
    return next((policy for policy in POLICIES if policy in path.stem), None)


def runtime_rows(path: Path) -> list[dict[str, float]]:
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        match = RUNTIME.search(line)
        if match:
            values = match.groupdict()
            rows.append(
                {
                    "time": 10.0 * len(rows),
                    "tps": float(values["tps"]),
                    "waiting": float(values["waiting"]),
                    "kv": float(values["kv"]),
                }
            )
    return rows


def result_completions(path: Path) -> list[float]:
    result = json.loads(path.read_text(encoding="utf-8"))
    return sorted(float(parent["completion_s"]) for parent in result["parents"])


def interpolate(rows: list[dict[str, float]], key: str, grid: np.ndarray) -> np.ndarray:
    times = np.array([row["time"] for row in rows])
    values = np.array([row[key] for row in rows])
    output = np.interp(grid, times, values)
    output[grid > times[-1]] = np.nan
    return output


def plot_runtime(root: Path, output: Path) -> None:
    runs: dict[str, list[tuple[list[dict[str, float]], list[float]]]] = defaultdict(list)
    for log in sorted(root.rglob("*.log")):
        policy = policy_from_name(log)
        rows = runtime_rows(log)
        result = log.with_suffix(".json")
        if policy and rows and result.exists():
            runs[policy].append((rows, result_completions(result)))
    missing = set(POLICIES) - set(runs)
    if missing:
        raise ValueError(f"missing runtime policies: {sorted(missing)}")

    xmax = max(max(completions) for items in runs.values() for _, completions in items)
    grid = np.arange(0.0, xmax + 10.0, 10.0)
    panels = (
        ("kv", "KV-cache utilisation (%)"),
        ("waiting", "Waiting requests"),
        ("tps", "Generation tokens/s"),
        ("completed", "Completed parents"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for axis, (key, ylabel) in zip(axes.flat, panels):
        for policy, (label, colour) in POLICIES.items():
            series = []
            for rows, completions in runs[policy]:
                if key == "completed":
                    series.append(
                        np.array([sum(value <= time for value in completions) for time in grid])
                    )
                else:
                    series.append(interpolate(rows, key, grid))
            values = np.vstack(series)
            valid = ~np.all(np.isnan(values), axis=0)
            selected_grid = grid[valid]
            selected_values = values[:, valid]
            axis.plot(
                selected_grid,
                np.nanmean(selected_values, axis=0),
                label=label,
                color=colour,
            )
            axis.fill_between(
                selected_grid,
                np.nanmin(selected_values, axis=0),
                np.nanmax(selected_values, axis=0),
                color=colour,
                alpha=0.12,
            )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Elapsed serving time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def arrival_label(result: dict) -> str:
    config = result["config"]
    mode = config["arrival_mode"]
    if mode == "bursty":
        return "Bursty 1.0x"
    rate = float(config["arrival_rate_parents_s"])
    reference = 36 / 1376.6
    return f"Poisson {rate / reference:.1f}x"


def plot_deadlines(root: Path, output: Path) -> None:
    grouped: dict[str, dict[str, list[list[float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in sorted(root.rglob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        policy = result["config"]["policy"]
        if policy not in {"adaptive", "tail_refill"}:
            continue
        counts = result["metrics"]["completed_by_deadline"]
        grouped[arrival_label(result)][policy].append(
            [100.0 * float(counts[str(deadline)]) / 36.0 for deadline in DEADLINES]
        )
    if len(grouped) != 4:
        raise ValueError("expected four open-loop arrival conditions")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, label in zip(axes.flat, sorted(grouped)):
        for policy, name, colour in (
            ("adaptive", "Adaptive-only", "#6f7d8a"),
            ("tail_refill", "RePLAN", "#2673b8"),
        ):
            values = np.array(grouped[label][policy])
            axis.plot(DEADLINES, values.mean(axis=0), marker="o", label=name, color=colour)
        axis.set_title(label)
        axis.grid(alpha=0.25)
    for axis in axes[:, 0]:
        axis.set_ylabel("Deadline completion (%)")
    for axis in axes[-1]:
        axis.set_xlabel("Arrival-relative deadline (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--open-loop-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    plot_runtime(args.runtime_root, args.output_root / "runtime_timeline.pdf")
    plot_deadlines(args.open_loop_root, args.output_root / "deadline_attainment.pdf")


if __name__ == "__main__":
    main()
