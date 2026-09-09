#!/usr/bin/env python3
"""Summarise periodic vLLM runtime metrics from completed serving logs."""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

METRIC_RE = re.compile(
    r"Avg generation throughput: (?P<throughput>[0-9.]+) tokens/s, "
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<kv>[0-9.]+)%"
)


def policy_name(path: Path) -> str:
    for name in (
        "adaptive",
        "remaining_budget_refill",
        "largest_evidence_refill",
        "tail_refill",
    ):
        if name in path.name:
            return name
    raise ValueError(f"cannot infer policy from {path}")


def summarise_log(path: Path) -> dict[str, float | int | str]:
    samples = []
    for line in path.read_text(errors="replace").splitlines():
        match = METRIC_RE.search(line)
        if match:
            samples.append({key: float(value) for key, value in match.groupdict().items()})
    if not samples:
        raise ValueError(f"no vLLM metric samples in {path}")
    return {
        "path": str(path),
        "samples": len(samples),
        "generation_tokens_per_second": statistics.mean(x["throughput"] for x in samples),
        "running_requests": statistics.mean(x["running"] for x in samples),
        "waiting_requests": statistics.mean(x["waiting"] for x in samples),
        "kv_cache_usage_percent": statistics.mean(x["kv"] for x in samples),
        "kv_saturation_fraction": statistics.mean(x["kv"] >= 99.0 for x in samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log paths or glob patterns")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted({Path(path) for pattern in args.logs for path in glob.glob(pattern)})
    runs = [summarise_log(path) for path in paths]
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for run in runs:
        grouped.setdefault(policy_name(Path(str(run["path"]))), []).append(run)

    metrics = (
        "generation_tokens_per_second",
        "running_requests",
        "waiting_requests",
        "kv_cache_usage_percent",
        "kv_saturation_fraction",
    )
    policies = {
        policy: {
            "runs": len(policy_runs),
            **{
                metric: statistics.mean(float(run[metric]) for run in policy_runs)
                for metric in metrics
            },
        }
        for policy, policy_runs in sorted(grouped.items())
    }
    result = {"sampling_interval_seconds": 10, "policies": policies, "run_summaries": runs}
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
