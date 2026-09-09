#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU SECTION" >&2
  echo "sections: live naive midpoint order baselines allocation mixed memory branch-limit zero-slack" >&2
  exit 2
fi

gpu=$1
section=$2
runner=$(cd "$(dirname "$0")" && pwd)/run_arm.sh

pair() {
  local repetition=$1
  shift
  if (( repetition % 2 == 1 )); then
    "$runner" "$gpu" adaptive "$repetition" "$@"
    "$runner" "$gpu" tail_refill "$repetition" "$@"
  else
    "$runner" "$gpu" tail_refill "$repetition" "$@"
    "$runner" "$gpu" adaptive "$repetition" "$@"
  fi
}

case "$section" in
  live)
    export REPLAY=0
    for dataset in amc aime; do
      parents=40; [[ "$dataset" == aime ]] && parents=30
      for n in 32 64 128; do
        for repetition in 1 2 3; do
          pair "$repetition" "$n" 256 0.78 0 "$dataset" "$parents" \
            "live_${dataset}_n${n}"
        done
      done
    done
    ;;
  naive)
    export REPLAY=1
    for dataset in amc aime; do
      parents=40; [[ "$dataset" == aime ]] && parents=30
      for repetition in 1 2 3; do
        if (( repetition == 1 )); then
          policies=(adaptive naive_refill tail_refill)
        elif (( repetition == 2 )); then
          policies=(naive_refill tail_refill adaptive)
        else
          policies=(tail_refill adaptive naive_refill)
        fi
        for policy in "${policies[@]}"; do
          "$runner" "$gpu" "$policy" "$repetition" 128 256 0.78 0 \
            "$dataset" "$parents" "naive_${dataset}_n128"
        done
      done
    done
    ;;
  midpoint)
    export REPLAY=1
    for boundary in 32 64 96; do
      for repetition in 1 2 3; do
        "$runner" "$gpu" tail_refill "$repetition" 128 256 0.78 \
          "$boundary" amc,aime,math 12 "midpoint_k${boundary}"
      done
    done
    ;;
  order)
    export REPLAY=1
    for evidence_order in sampling completion; do
      export EVIDENCE_ORDER="$evidence_order"
      for repetition in 1 2 3; do
        "$runner" "$gpu" tail_refill "$repetition" 128 256 0.78 0 \
          amc,aime,math 12 "order_${evidence_order}_n128"
      done
    done
    unset EVIDENCE_ORDER
    ;;
  baselines)
    export REPLAY=1
    for repetition in 1 2 3; do
      if (( repetition == 1 )); then
        policies=(adaptive largest_evidence_refill remaining_budget_refill tail_refill)
      elif (( repetition == 2 )); then
        policies=(tail_refill adaptive largest_evidence_refill remaining_budget_refill)
      else
        policies=(remaining_budget_refill tail_refill adaptive largest_evidence_refill)
      fi
      for policy in "${policies[@]}"; do
        "$runner" "$gpu" "$policy" "$repetition" 128 256 0.78 0 \
          amc,aime,math 12 "baselines_mixed_n128"
      done
    done
    ;;
  allocation)
    export REPLAY=1
    for dataset in amc aime math; do
      parents=40
      [[ "$dataset" == aime ]] && parents=30
      [[ "$dataset" == math ]] && parents=32
      for n in 32 64 128; do
        if (( n == 64 )); then
          policies=(confidence_weighted_refill tail_refill)
        else
          policies=(tail_refill confidence_weighted_refill)
        fi
        for policy in "${policies[@]}"; do
          "$runner" "$gpu" "$policy" 1 "$n" 256 0.78 0 \
            "$dataset" "$parents" "confidence_${dataset}_n${n}"
        done
      done
    done
    for repetition in 1 2; do
      if (( repetition == 1 )); then
        policies=(tail_refill exact_tail_refill)
      else
        policies=(exact_tail_refill tail_refill)
      fi
      for policy in "${policies[@]}"; do
        "$runner" "$gpu" "$policy" "$repetition" 128 256 0.78 0 \
          amc,aime,math 12 "exact_rounding_mixed_n128"
      done
    done
    ;;
  mixed)
    export REPLAY=1
    for n in 64 128; do
      for repetition in 1 2 3; do
        pair "$repetition" "$n" 256 0.78 0 amc,aime,math 12 "mixed_n${n}"
      done
    done
    ;;
  memory)
    export REPLAY=1
    for memory in 0.70 0.78 0.90; do
      for repetition in 1 2 3; do
        pair "$repetition" 128 256 "$memory" 0 amc,aime,math 12 \
          "memory_${memory/./p}"
      done
    done
    ;;
  branch-limit)
    export REPLAY=1
    for bound in 144 192 256; do
      for repetition in 1 2 3; do
        pair "$repetition" 128 "$bound" 0.78 0 amc,aime,math 12 \
          "branch_bound_${bound}"
      done
    done
    ;;
  zero-slack)
    export REPLAY=1
    for repetition in 1 2 3; do
      pair "$repetition" 128 256 0.78 0 math 32 "zero_slack_math"
      pair "$repetition" 128 144 0.78 0 amc,aime,math 12 "zero_slack_mixed"
    done
    ;;
  *) echo "unknown section: $section" >&2; exit 2 ;;
esac
