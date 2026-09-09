#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: $0 GPU MODEL_LABEL MODEL SOURCE_TRACE SOURCE_PROMPTS OUTPUT PARENTS N MAX_TOKENS THINKING" >&2
  exit 2
fi

gpu=$1
model_label=$2
model=$3
source_trace=$4
source_prompts=$5
output=$6
parents=$7
n=$8
max_tokens=$9
thinking=${10}

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
export PYTHONPATH="$experiment_root:$code_root:$vllm_root"
export TORCH_HOME="$cache_root/torch"
export XDG_CACHE_HOME="$cache_root/gpu${gpu}/xdg"
export TMPDIR="$cache_root/gpu${gpu}/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=1
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" "$(dirname "$output")"

thinking_args=()
case "$thinking" in
  true) thinking_args+=(--enable-thinking) ;;
  false) thinking_args+=(--no-enable-thinking) ;;
  none) ;;
  *) echo "THINKING must be true, false, or none" >&2; exit 2 ;;
esac
resume_args=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_args+=(--resume)
fi

echo "host=$(hostname) gpu=$gpu model=$model_label parents=$parents N=$n thinking=$thinking"
exec "$python_bin" "$experiment_root/collect_fresh.py" \
  --source-trace "$source_trace" \
  --source-prompts "$source_prompts" \
  --output-root "$output" \
  --model "$model" \
  --model-label "$model_label" \
  --parents "$parents" \
  --n "$n" \
  --max-tokens "$max_tokens" \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.78 \
  --enforce-eager \
  "${resume_args[@]}" \
  "${thinking_args[@]}"
