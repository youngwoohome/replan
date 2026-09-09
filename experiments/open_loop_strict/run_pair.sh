#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 GPU PARENTS_PER_DATASET MODE RATE LABEL SEED RESULT_ROOT" >&2
  exit 2
fi

gpu=$1
parents_per_dataset=$2
mode=$3
rate=$4
label=$5
seed=$6
result_root=$7
reference_concurrency=${REFERENCE_CONCURRENCY:-36}
n_max=${N_MAX:-128}

experiment_root=$(cd "$(dirname "$0")/../.." && pwd)
code_root=${CODE_ROOT:-$experiment_root}
vllm_root=${VLLM_ROOT:?set VLLM_ROOT to the pinned vLLM checkout}
input_root=${INPUT_ROOT:?set INPUT_ROOT to the workload matrix}
python_bin=${PYTHON:-python3}
model=${MODEL:?set MODEL to a model path or Hugging Face identifier}
quality_amc=${QUALITY_AMC:?set QUALITY_AMC to the AMC candidate JSONL}
quality_aime=${QUALITY_AIME:?set QUALITY_AIME to the AIME candidate JSONL}
quality_math=${QUALITY_MATH:?set QUALITY_MATH to the MATH500 candidate JSONL}
cache_root=${CACHE_ROOT:-$result_root/cache}
native_site=${NATIVE_SITE:-}
hf_home=${HF_HOME:-$cache_root/huggingface}

mkdir -p "$result_root" "$cache_root/gpu${gpu}/xdg" \
  "$cache_root/gpu${gpu}/tmp" "$cache_root/torch" "$hf_home"

wait_for_gpu() {
  while true; do
    used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')
    (( used < 1000 )) && return
    sleep 15
  done
}

validate_result() {
  "$python_bin" - "$1" <<'PY'
import json
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
assert result["status"] == "completed"
assert result["config"]["strict_outstanding_cap"] is True
assert result["metrics"]["maximum_observed_outstanding_branches"] <= 256
assert result["metrics"]["parents"] == len(result["parents"])
assert result["config"]["reference_concurrency"] > 0
assert result["config"]["initial_wave"] == (
    result["config"]["max_outstanding_branches"]
    // result["config"]["reference_concurrency"]
)
assert all(parent["queue_delay_s"] >= 0 for parent in result["parents"])
PY
}

run_arm() {
  policy=$1
  tag="mixed_p${parents_per_dataset}each_n${n_max}_${mode}_load${label}_${policy}_seed${seed}"
  output="$result_root/${tag}.json"
  log="$result_root/${tag}.log"
  if [[ -s "$output" ]]; then
    validate_result "$output"
    return
  fi
  wait_for_gpu

  export CUDA_VISIBLE_DEVICES="$gpu"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export PYTHONPATH="${native_site:+$native_site:}$code_root:$vllm_root"
  export HF_HOME="$hf_home"
  export TORCH_HOME="$cache_root/torch"
  export XDG_CACHE_HOME="$cache_root/gpu${gpu}/xdg"
  export TMPDIR="$cache_root/gpu${gpu}/tmp"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
  export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
  export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=1
  unset VLLM_SCHEDULER_POLICY

  echo "[$(date --iso-8601=seconds)] START $tag gpu=$gpu" | tee "$log"
  "$python_bin" -m replan.serve \
    --input-root "$input_root" \
    --dataset fasttts_amc2023 \
    --dataset fasttts_aime2024 \
    --dataset math500 \
    --parents-per-dataset "$parents_per_dataset" \
    --n-max "$n_max" \
    --policy "$policy" \
    --confidence-rule beta \
    --evidence-order sampling \
    --reference-concurrency "$reference_concurrency" \
    --quality-trace "fasttts_amc2023=$quality_amc" \
    --quality-trace "fasttts_aime2024=$quality_aime" \
    --quality-trace "math500=$quality_math" \
    --model "$model" \
    --dtype float16 \
    --max-outstanding-branches 256 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.78 \
    --arrival-mode "$mode" \
    --arrival-rate "$rate" \
    --arrival-seed "$seed" \
    --burst-size 4 \
    --enable-prefix-caching \
    --enforce-eager \
    --output "$output" 2>&1 | tee -a "$log"
  validate_result "$output"
  echo "[$(date --iso-8601=seconds)] DONE $tag" | tee -a "$log"
  sleep 10
}

if (( seed % 2 == 0 )); then
  run_arm tail_refill
  run_arm adaptive
else
  run_arm adaptive
  run_arm tail_refill
fi
