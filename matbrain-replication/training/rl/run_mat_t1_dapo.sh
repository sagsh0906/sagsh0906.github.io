#!/usr/bin/env bash
# Mat-T1: multi-turn agentic RL of Qwen3-14B on Mat-20K-RL with verl v0.6.1.
# Hardware in the paper: 1 node x 8 NVIDIA H800 (80 GB).
#
# Paper hyper-parameters (Methods, "Implementation details of Mat-T1"):
#   DAPO decoupled clip [0.2, 0.28] | KL coefficient 0 | AdamW, constant lr 1e-6
#   global batch 16 | group size G = 8 | 1 epoch (20k prompts -> ~1,250 steps, cf. Fig. 2g,h)
#   vLLM rollout TP = 4 | FSDP + Ulysses SP = 4 + CPU offload
#   max prompt 16,384 | max response 3,072 | max 4 user + 4 assistant turns
#   reward = 0.1 R_turns + 0.3 R_think + 0.25 R_format + 0.35 R_syntax (matbrain/rl/rewards.py)
#
# Note on "dynamic sampling": in verl v0.6.1 the DAPO recipe trainer
# (recipe/dapo) only supports single-turn synchronous rollout, while
# multi-turn tool calling requires the async agent loop of main_ppo. We
# therefore run main_ppo with DAPO's decoupled clipping, token-level loss
# aggregation and no KL term. Groups whose 8 rewards are identical receive
# zero GRPO advantage and thus no gradient (what DAPO's group filter removes);
# with the continuous composite reward such groups are rare.
set -xeuo pipefail

REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd)
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-14B}
TRAIN_FILE=${TRAIN_FILE:-${REPO_DIR}/data/rl/mat20k_rl_train.parquet}
VAL_FILE=${VAL_FILE:-${REPO_DIR}/data/rl/mat20k_rl_val.parquet}
TOOL_CONFIG=${TOOL_CONFIG:-${REPO_DIR}/training/rl/tool_config/mat_mcp_local.yaml}
CKPT_DIR=${CKPT_DIR:-${REPO_DIR}/checkpoints/Mat-T1}
NGPUS=${NGPUS:-8}
# exact token counts for the think-length reward
export MATBRAIN_REWARD_TOKENIZER=${MATBRAIN_REWARD_TOKENIZER:-${MODEL_PATH}}

if [ ! -f "${TOOL_CONFIG}" ]; then
  python "${REPO_DIR}/training/rl/make_tool_config.py" --mode local --out "${TOOL_CONFIG}"
fi

max_prompt_length=16384
max_response_length=3072
max_tokens=$((max_prompt_length + max_response_length))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.return_raw_chat=True \
    data.train_batch_size=16 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_tokens} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${max_tokens} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=4 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REPO_DIR}/matbrain/rl/verl_reward.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=MatBrain \
    trainer.experiment_name=Mat-T1-Qwen3-14B-DAPO \
    trainer.n_gpus_per_node=${NGPUS} \
    trainer.nnodes=1 \
    trainer.val_before_train=True \
    trainer.test_freq=50 \
    trainer.save_freq=100 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto "$@"
