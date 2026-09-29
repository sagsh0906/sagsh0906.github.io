#!/usr/bin/env bash
# Scaled-down Mat-R1 for a single consumer GPU (e.g. RTX 4090 24 GB):
# LoRA on Qwen3-4B with the same data mixture, schedule shape and 2 epochs.
# Not the paper's setting (full-parameter 30B-A3B), but reproduces the
# qualitative effects (loss curve, entropy drop, tool-free benchmark gains).
set -xeuo pipefail
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-4B}

swift sft \
    --model "${MODEL}" \
    --train_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --dataset "${TRAIN_DATA:-${REPO_DIR}/data/sft/mat_r1_train.jsonl}" \
    --val_dataset "${VAL_DATA:-${REPO_DIR}/data/sft/mat_r1_val.jsonl}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_length 8192 \
    --packing true \
    --attn_impl flash_attn \
    --gradient_checkpointing true \
    --eval_steps 500 \
    --save_steps 500 \
    --logging_steps 5 \
    --seed 42 \
    --output_dir "${OUT:-${REPO_DIR}/checkpoints/Mat-R1-small}" "$@"
