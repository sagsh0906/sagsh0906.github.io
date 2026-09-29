#!/usr/bin/env bash
# Scaled-down Mat-T1 run for a single workstation (e.g. 2 x RTX 4090 24 GB or
# 1 x A100 80 GB): Qwen3-1.7B, same reward / algorithm / turn budgets, smaller
# parallelism and rollout budget. Useful to validate the whole pipeline before
# renting 8 x H800 for the 14B run. Extra Hydra overrides are passed through.
set -xeuo pipefail
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-1.7B} \
NGPUS=${NGPUS:-2} \
CKPT_DIR=${CKPT_DIR:-${REPO_DIR}/checkpoints/Mat-T1-small} \
bash "${REPO_DIR}/training/rl/run_mat_t1_dapo.sh" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    data.max_prompt_length=8192 \
    trainer.experiment_name=Mat-T1-Qwen3-1.7B-small \
    "$@"
