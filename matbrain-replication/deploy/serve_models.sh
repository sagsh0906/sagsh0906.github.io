#!/usr/bin/env bash
# Serve Mat-T1 and Mat-R1 with vLLM on one workstation (e.g. 4 x RTX 4090 24 GB).
# Mat-T1 (14B, BF16 ~28 GB) -> GPUs 0,1 ; Mat-R1 (30B-A3B MoE, BF16 ~61 GB) -> GPUs 2,3 + FP8 KV cache,
# or use FP8/AWQ-quantised checkpoints to fit fewer cards.
set -euo pipefail
T1=${T1:-checkpoints/Mat-T1/hf}
R1=${R1:-checkpoints/Mat-R1/hf}

CUDA_VISIBLE_DEVICES=0,1 vllm serve "$T1" --served-model-name Mat-T1 --port 8001 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 &

CUDA_VISIBLE_DEVICES=2,3 vllm serve "$R1" --served-model-name Mat-R1 --port 8002 \
  --tensor-parallel-size 2 --max-model-len 32768 --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 &

wait
