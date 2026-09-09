#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 GPU POLICY REP N B MEMORY BOUNDARY DATASETS PARENTS TAG" >&2
  exit 2
fi

gpu=$1
policy=$2
repetition=$3
n=$4
branch_bound=$5
memory=$6
boundary=$7
dataset_csv=$8
parents=$9
tag=${10}

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
vllm_root=${VLLM_ROOT:?set VLLM_ROOT to the pinned vLLM checkout}
input_root=${INPUT_ROOT:?set INPUT_ROOT to the workload matrix}
model=${MODEL:?set MODEL to the model path or identifier}
result_root=${RESULT_ROOT:?set RESULT_ROOT for experiment outputs}
python_bin=${PYTHON:-python3}
cache_root=${CACHE_ROOT:-$result_root/cache}
replay=${REPLAY:-1}
evidence_order=${EVIDENCE_ORDER:-sampling}
output="$result_root/$tag/${policy}.r${repetition}.json"
log="${output%.json}.log"

[[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 3; }
used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')
(( used < 1000 )) || { echo "GPU $gpu is busy: ${used} MiB" >&2; exit 75; }

IFS=, read -r -a datasets <<< "$dataset_csv"
dataset_args=()
quality_args=()
for short_name in "${datasets[@]}"; do
  case "$short_name" in
    amc) dataset=fasttts_amc2023; quality=${QUALITY_AMC:-} ;;
    aime) dataset=fasttts_aime2024; quality=${QUALITY_AIME:-} ;;
    math) dataset=math500; quality=${QUALITY_MATH:-} ;;
    *) echo "unknown dataset: $short_name" >&2; exit 2 ;;
  esac
  dataset_args+=(--dataset "$dataset")
  if [[ "$replay" == 1 ]]; then
    [[ -n "$quality" ]] || {
      echo "set the quality trace environment variable for $short_name" >&2
      exit 2
    }
    quality_args+=(--quality-trace "$dataset=$quality")
  fi
done

boundary_args=()
(( boundary == 0 )) || boundary_args=(--tail-activation-evidence "$boundary")

export CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$repo_root:$vllm_root"
export TORCH_HOME="$cache_root/torch"
export XDG_CACHE_HOME="$cache_root/gpu${gpu}/xdg"
export TMPDIR="$cache_root/gpu${gpu}/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=1
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$output")"

"$python_bin" -m replan.serve \
  --input-root "$input_root" "${dataset_args[@]}" \
  --parents-per-dataset "$parents" --n-max "$n" --policy "$policy" \
  --confidence-rule beta --evidence-order "$evidence_order" \
  "${boundary_args[@]}" "${quality_args[@]}" \
  --model "$model" --dtype float16 \
  --max-outstanding-branches "$branch_bound" \
  --max-model-len 8192 --max-num-seqs 256 \
  --gpu-memory-utilization "$memory" --enable-prefix-caching --enforce-eager \
  --output "$output" 2>&1 | tee "$log"
