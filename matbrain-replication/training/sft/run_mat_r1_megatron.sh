#!/usr/bin/env bash
# Mat-R1: full-parameter SFT of Qwen3-30B-A3B with Megatron-SWIFT (ms-swift 3.10).
# Hardware in the paper: 1 node x 8 NVIDIA H800 (80 GB), BF16.
#
# Paper hyper-parameters (Methods, "Implementation details of Mat-R1"):
#   data = Mat-252K-SFT + open-r1/Mixture-of-Thoughts (science, 173k) + self-identity
#   TP = 2, EP = 8, MoE aux-loss coefficient 1e-3, Flash-Attention 2, full recompute
#   global batch 16, max length 16,384, sample packing
#   AdamW, linear warm-up 1e-6 -> 2e-5 over the first 5% of steps, cosine decay to 1e-6
#   2 epochs
set -xeuo pipefail

REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
TRAIN_DATA=${TRAIN_DATA:-${REPO_DIR}/data/sft/mat_r1_train.jsonl}
VAL_DATA=${VAL_DATA:-${REPO_DIR}/data/sft/mat_r1_val.jsonl}
OUT=${OUT:-${REPO_DIR}/checkpoints/Mat-R1}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=${NPROC_PER_NODE:-8} \
megatron sft \
    --model "${MODEL}" \
    --load_safetensors true \
    --save_safetensors true \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --torch_dtype bfloat16 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 8 \
    --sequence_parallel true \
    --moe_aux_loss_coeff 1e-3 \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --packing true \
    --max_length 16384 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --attention_backend flash \
    --cross_entropy_loss_fusion true \
    --optimizer adam \
    --lr 2e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.05 \
    --lr_decay_style cosine \
    --megatron_extra_kwargs '{"lr_warmup_init": 1e-6}' \
    --max_epochs 2 \
    --finetune true \
    --seed 42 \
    --eval_interval 500 \
    --save_interval 500 \
    --save "${OUT}" \
    --no_save_optim true \
    --no_save_rng true \
    --num_workers 8 \
    --dataset_num_proc 16 \
    --log_interval 5 \
    --wandb_project MatBrain \
    --wandb_exp_name Mat-R1-Qwen3-30B-A3B-SFT "$@"
