#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/hoon/joonha_llm/KoRaptor
ENV=/home/hoon/joonha_llm/.venv
CONFIG=${1:-$ROOT/configs/raptor_1b_nemotron.yaml}
if [[ $# -gt 0 ]]; then
  shift
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=7
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

exec "$ENV/bin/python" -m Pretrain.pretrain_nemotron \
  --config "$CONFIG" \
  --resume auto \
  "$@"
