#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 GPU MODEL_LABEL MODEL THINKING RESULT_ROOT" >&2
  exit 2
fi

gpu=$1
model_label=$2
model=$3
thinking=$4
full_root=$5
experiment_root=${EXPERIMENT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
input_root=${INPUT_ROOT:?set INPUT_ROOT to the source workload matrix}
python_bin=${PYTHON:-python3}

wait_for_gpu() {
  while true; do
    local memory_used
    memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')
    (( memory_used < 1000 )) && return
    sleep 15
  done
}

collect_dataset() {
  local dataset=$1
  local parents=$2
  local max_tokens=$3
  local trace_name=$4
  local prompt_name=$5
  local dataset_root="$full_root/$model_label/$dataset"
  local source_trace="$input_root/$dataset/n128/traces/$trace_name"
  local source_prompts="$input_root/$dataset/prompts/$prompt_name"
  local log_dir="$full_root/$model_label/logs"
  local attempt=0
  mkdir -p "$log_dir"
  until grep -q '"status": "completed"' "$dataset_root/manifest.json" 2>/dev/null; do
    attempt=$((attempt + 1))
    if (( attempt > 3 )); then
      echo "trace collection failed three times for $dataset" >&2
      return 1
    fi
    wait_for_gpu
    if [[ -e "$dataset_root/manifest.json" ]]; then
      RESUME=1 "$experiment_root/run_collect.sh" \
        "$gpu" "$model_label" "$model" "$source_trace" "$source_prompts" \
        "$dataset_root" "$parents" 128 "$max_tokens" "$thinking" \
        > "$log_dir/${dataset}.collect.attempt${attempt}.log" 2>&1 || true
    else
      "$experiment_root/run_collect.sh" \
        "$gpu" "$model_label" "$model" "$source_trace" "$source_prompts" \
        "$dataset_root" "$parents" 128 "$max_tokens" "$thinking" \
        > "$log_dir/${dataset}.collect.attempt${attempt}.log" 2>&1 || true
    fi
  done
  if [[ ! -d "$dataset_root/n64" ]]; then
    "$python_bin" "$experiment_root/project_n.py" \
      --input-root "$dataset_root" --source-n 128 --target-n 64
  fi
}

run_arm() {
  local dataset=$1
  local parents=$2
  local n=$3
  local policy=$4
  local repetition=$5
  local dataset_root="$full_root/$model_label/$dataset"
  local result_dir="$dataset_root/serving/n${n}"
  local output="$result_dir/${policy}.r${repetition}.json"
  [[ -s "$output" ]] && return
  mkdir -p "$result_dir"
  wait_for_gpu
  "$experiment_root/run_serve.sh" \
    "$gpu" "$dataset_root" "$dataset" "$parents" "$n" "$policy" \
    "$output" "$model" > "$result_dir/${policy}.r${repetition}.log" 2>&1
}

run_matrix() {
  local dataset=$1
  local parents=$2
  local n repetition
  for n in 128 64; do
    for repetition in 1 2 3; do
      if (( repetition % 2 == 1 )); then
        run_arm "$dataset" "$parents" "$n" adaptive "$repetition"
        run_arm "$dataset" "$parents" "$n" tail_refill "$repetition"
      else
        run_arm "$dataset" "$parents" "$n" tail_refill "$repetition"
        run_arm "$dataset" "$parents" "$n" adaptive "$repetition"
      fi
    done
  done
}

collect_dataset fasttts_aime2024 30 2048 fasttts_aime2024_seed42_bon128.json fasttts_aime2024_all.jsonl
run_matrix fasttts_aime2024 30

date -Is > "$full_root/$model_label/CHAIN_COMPLETED"
