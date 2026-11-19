#!/usr/bin/env bash
# Serve a (V)LM with vLLM (OpenAI-compatible). Weights resolved ModelScope-first into storage/models.
# usage: bash scripts/serve_vlm.sh <model id|local dir> [port=8000] [served-name] [gpu=1]
#   bash scripts/serve_vlm.sh Qwen/Qwen2.5-VL-7B-Instruct 8000 qwen-vl 1
#   bash scripts/serve_vlm.sh storage/runs/query/qrewrite_sft/merged 8002 rewriter 1
# GPU plan (2×H20): GPU0 = retrieval encoders / training, GPU1 = vLLM. Two servers on GPU1: GPU_UTIL=0.45 each.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-Qwen/Qwen2.5-VL-7B-Instruct}
PORT=${2:-8000}
NAME=${3:-$(basename "$MODEL")}
GPU=${4:-1}
LOCAL=$(python -m mmsearch.utils.hub --model "$MODEL" | tail -1 | awk -F' -> ' '{print $2}')
echo "serving $LOCAL as '$NAME' on :$PORT (GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU exec vllm serve "$LOCAL" \
  --port "$PORT" --served-model-name "$NAME" \
  --max-model-len "${MAX_LEN:-32768}" \
  --limit-mm-per-prompt '{"image": 12}' \
  --gpu-memory-utilization "${GPU_UTIL:-0.85}" \
  --max-num-seqs "${MAX_SEQS:-64}" \
  --enable-prefix-caching
