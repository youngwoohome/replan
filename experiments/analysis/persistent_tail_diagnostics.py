"""Test whether midpoint-unresolved parents lie on the batch critical path."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from replan.policy import beta_leader_probability

CURRENT_ADAPTIVE = re.compile(r"adaptive\.r(?P<rep>\d+)\.json$")
PAPER_PARENT_COUNTS = {
    "fasttts_amc2023": 40,
    "fasttts_aime2024": 30,
    "math500": 32,
}


def stop_depth(branches: list[dict], n_max: int) -> int:
    answers = {
        int(branch["candidate_index"]): branch.get("answer")
        for branch in branches
    }
    ordered: list[str | None] = []
    for index in range(n_max):
        if index not in answers:
            break
        ordered.append(answers[index])
        if len(ordered) >= 4 and beta_leader_probability(ordered) > 0.95:
            return len(ordered)
    return min(n_max, len(ordered))


def is_persistent_stop_depth(depth: int, n_max: int) -> bool:
    """A parent is persistent only if it remains unresolved at the midpoint."""

    return depth > math.ceil(n_max / 2)


def paired_result_paths(root: Path) -> list[tuple[Path, Path, int]]:
    """Find paired results in the current per-condition directory layout."""

    pairs: list[tuple[Path, Path, int]] = []
    for adaptive_path in sorted(root.rglob("*.json")):
        current = CURRENT_ADAPTIVE.match(adaptive_path.name)
        if current is None:
            continue
        repetition = int(current.group("rep"))
        tail_path = adaptive_path.with_name(f"tail_refill.r{repetition}.json")
        if tail_path.exists():
            pairs.append((adaptive_path, tail_path, repetition))
    return pairs


def is_paper_persistent_pair(adaptive: dict, tail: dict) -> bool:
    """Select the nine individual frozen runs described in the thesis."""

    left = adaptive["config"]
    right = tail["config"]
    datasets = left.get("datasets", [])
    if len(datasets) != 1:
        return False
    dataset = datasets[0]
    expected_parents = PAPER_PARENT_COUNTS.get(dataset)
    return (
        left.get("policy") == "adaptive"
        and right.get("policy") == "tail_refill"
        and left.get("n_max") == right.get("n_max") == 128
        and left.get("max_outstanding_branches")
        == right.get("max_outstanding_branches")
        == 256
        and left.get("arrival_mode") == right.get("arrival_mode") == "batch"
        and left.get("quality_replay") is right.get("quality_replay") is True
        and right.get("datasets") == datasets
        and expected_parents is not None
        and left.get("parents_per_dataset")
        == right.get("parents_per_dataset")
        == expected_parents
    )


def ranks(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for index in ordered[position:end]:
            result[index] = rank
        position = end
    return result


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs = []
    parent_records = []
    for adaptive_path, tail_path, repetition in paired_result_paths(args.root):
        adaptive, tail = load(adaptive_path), load(tail_path)
        pair_name = str(adaptive_path.parent.relative_to(args.root))
        if not is_paper_persistent_pair(adaptive, tail):
            continue
        n_max = int(adaptive["config"]["n_max"])
        adaptive_parents = {row["parent_id"]: row for row in adaptive["parents"]}
        tail_parents = {row["parent_id"]: row for row in tail["parents"]}
        branches = defaultdict(list)
        for row in adaptive["branches"]:
            branches[row["parent_id"]].append(row)

        rows = []
        for parent_id, adaptive_parent in adaptive_parents.items():
            depth = stop_depth(branches[parent_id], n_max)
            persistent = is_persistent_stop_depth(depth, n_max)
            tail_parent = tail_parents[parent_id]
            row = {
                "pair": pair_name,
                "repetition": repetition,
                "parent_id": parent_id,
                "stop_depth": depth,
                "persistent": persistent,
                "adaptive_e2e_s": float(adaptive_parent["completion_s"]),
                "tail_e2e_s": float(tail_parent["completion_s"]),
                "e2e_reduction_s": float(adaptive_parent["completion_s"])
                - float(tail_parent["completion_s"]),
            }
            rows.append(row)
            parent_records.append(row)

        slowest = max(rows, key=lambda row: row["adaptive_e2e_s"])
        top_count = max(1, math.ceil(len(rows) * 0.1))
        top_rows = sorted(
            rows, key=lambda row: row["adaptive_e2e_s"], reverse=True
        )[:top_count]
        pairs.append(
            {
                "pair": pair_name,
                "repetition": repetition,
                "parents": len(rows),
                "persistent_parents": sum(row["persistent"] for row in rows),
                "slowest_parent_is_persistent": slowest["persistent"],
                "top_10pct_persistent_fraction": mean(
                    row["persistent"] for row in top_rows
                ),
                "spearman_stop_depth_vs_adaptive_e2e": correlation(
                    ranks([float(row["stop_depth"]) for row in rows]),
                    ranks([row["adaptive_e2e_s"] for row in rows]),
                ),
            }
        )

    if not pairs:
        raise RuntimeError("no Adaptive/RePLAN pairs found")
    persistent_deltas = [
        row["e2e_reduction_s"] for row in parent_records if row["persistent"]
    ]
    other_deltas = [
        row["e2e_reduction_s"] for row in parent_records if not row["persistent"]
    ]
    result = {
        "pairs": pairs,
        "aggregate": {
            "pairs": len(pairs),
            "parents": len(parent_records),
            "slowest_parent_persistent_fraction": mean(
                row["slowest_parent_is_persistent"] for row in pairs
            ),
            "top_10pct_persistent_fraction": mean(
                row["top_10pct_persistent_fraction"] for row in pairs
            ),
            "mean_spearman_stop_depth_vs_adaptive_e2e": mean(
                row["spearman_stop_depth_vs_adaptive_e2e"] for row in pairs
            ),
            "persistent_parent_mean_e2e_reduction_s": (
                mean(persistent_deltas) if persistent_deltas else None
            ),
            "nonpersistent_parent_mean_e2e_reduction_s": (
                mean(other_deltas) if other_deltas else None
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
