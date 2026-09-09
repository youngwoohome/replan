#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 GPU INPUT_ROOT DATASET PARENTS N POLICY OUTPUT MODEL" >&2
  exit 2
fi

gpu=$1
input_root=$2
dataset=$3
parents=$4
n=$5
policy=$6
output=$7
model=$8
quality="$input_root/quality/candidates.jsonl"

memory_used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
  --format=csv,noheader,nounits | tr -d ' ')
if (( memory_used > 1000 )); then
  echo "GPU $gpu is busy: ${memory_used} MiB" >&2
  exit 75
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 3
fi

python_bin=${PYTHON:-python3}
experiment_root=${EXPERIMENT_ROOT:-$(cd "$(dirname "$0")" && pwd)}
code_root=${CODE_ROOT:-$(cd "$experiment_root/../.." && pwd)}
vllm_root=${VLLM_ROOT:?set VLLM_ROOT to the pinned vLLM checkout}
cache_root=${CACHE_ROOT:?set CACHE_ROOT to a writable experiment cache}

export CUDA_VISIBLE_DEVICES="$gpu"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$code_root:$vllm_root"
export TORCH_HOME="$cache_root/torch"
export XDG_CACHE_HOME="$cache_root/gpu${gpu}/xdg"
export TMPDIR="$cache_root/gpu${gpu}/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=1
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$output")"

echo "host=$(hostname) gpu=$gpu dataset=$dataset parents=$parents N=$n policy=$policy"
exec "$python_bin" -m replan.serve \
  --input-root "$input_root" \
  --dataset "$dataset" \
  --parents-per-dataset "$parents" \
  --n-max "$n" \
  --policy "$policy" \
  --confidence-rule beta \
  --quality-trace "$dataset=$quality" \
  --model "$model" \
  --dtype bfloat16 \
  --max-outstanding-branches 256 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.78 \
  --enable-prefix-caching \
  --enforce-eager \
  --output "$output"
