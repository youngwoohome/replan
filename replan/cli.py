"""Command-line contract for the reported RePLAN serving runner."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run progressive quality-aware BoN on one vLLM GPU."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--parents-per-dataset", type=int, required=True)
    parser.add_argument("--n-max", type=int, default=128)
    parser.add_argument(
        "--policy",
        choices=(
            "adaptive",
            "naive_refill",
            "tail_refill",
            "largest_evidence_refill",
            "remaining_budget_refill",
            "confidence_weighted_refill",
            "exact_tail_refill",
        ),
        required=True,
        help=(
            "adaptive is stopping-only; naive_refill gives spare slots to "
            "every unresolved parent; tail_refill is RePLAN"
        ),
    )
    parser.add_argument(
        "--tail-activation-evidence",
        type=int,
        default=0,
        help=(
            "ablation only: 0 uses ceil(N/2); otherwise set the candidate "
            "count at which an unresolved parent becomes refill-eligible"
        ),
    )
    parser.add_argument(
        "--confidence-rule",
        choices=("beta",),
        default="beta",
        help="the paper's Beta-based Adaptive-Consistency stopping criterion",
    )
    parser.add_argument(
        "--evidence-order",
        choices=("sampling", "completion"),
        default="sampling",
        help=(
            "ablation only: sampling preserves Adaptive-Consistency order; "
            "completion consumes answers as branches finish"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--reference-concurrency",
        type=int,
        default=0,
        help=(
            "P_base in L0=floor(B/P_base); 0 uses the submitted parent count "
            "for a fixed batch and is invalid for open-loop serving"
        ),
    )
    parser.add_argument(
        "--quality-trace",
        action="append",
        default=[],
        metavar="DATASET=JSONL",
        help="reveal stored candidate answer only after its replay completes",
    )
    parser.add_argument("--max-outstanding-branches", type=int, default=256)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument(
        "--arrival-mode",
        choices=("batch", "poisson", "bursty"),
        default="batch",
        help="batch is the original offline experiment; other modes are open loop",
    )
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--arrival-seed", type=int, default=20260809)
    parser.add_argument("--burst-size", type=int, default=4)
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share repeated parent prompts (enabled by default for BoN)",
    )
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


__all__ = ["parse_args"]
