#!/usr/bin/env bash
# Install only what the server image lacks (torch/transformers/vllm/flash-attn/modelscope are preinstalled — don't touch them).
set -euo pipefail
cd "$(dirname "$0")/.."
PIP_INDEX=${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}

# --no-deps for faiss to avoid any numpy re-resolution; peft's deps are all present already.
pip install -i "$PIP_INDEX" "peft>=0.17" matplotlib pytest pytrec-eval-terrier
pip install -i "$PIP_INDEX" --no-deps faiss-cpu || echo "[warn] faiss-cpu unavailable; FaissTokenIndex disabled (PLAID-lite needs no faiss)"
pip install -e . --no-deps

# HF is unreachable on this server: hub resolution goes ModelScope first; keep HF as mirror fallback.
grep -q MMS_HUB_ORDER ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<'EOF'
export MMS_HUB_ORDER=modelscope,hf
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
EOF
python scripts/check_env.py
