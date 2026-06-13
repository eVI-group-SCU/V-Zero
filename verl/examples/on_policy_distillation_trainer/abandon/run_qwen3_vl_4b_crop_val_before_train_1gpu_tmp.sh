#!/usr/bin/env bash
# Temporary two-GPU smoke script for checking VERL validation rewards via val_before_train.

set -euo pipefail

# ---- user-adjustable ----
MODEL_ROOT=${MODEL_ROOT:-/root/autodl-tmp/models}
STUDENT_MODEL=${STUDENT_MODEL:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}
TEACHER_MODEL=${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}

TRAIN_FILE=${TRAIN_FILE:-/root/autodl-tmp/dataset/ZwZ-RL-VQA/verl_format_teacher_crop_train37k/train.parquet}
VAL_FILE=${VAL_FILE:-/root/autodl-tmp/dataset/validation_450/vstar_hrbench_mme_realworld_val450.parquet}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:-1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-96}

train_batch_size=${TRAIN_BATCH_SIZE:-1}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-1}
max_prompt_length=${MAX_PROMPT_LENGTH:-20000}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS:-16}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
use_system_prompt=${USE_SYSTEM_PROMPT:-True}
print_first_train_sample=${PRINT_FIRST_TRAIN_SAMPLE:-True}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_temperature=${ROLLOUT_TEMPERATURE:-0.7}
rollout_top_p=${ROLLOUT_TOP_P:-0.8}
rollout_top_k=${ROLLOUT_TOP_K:-20}
rollout_presence_penalty=${ROLLOUT_PRESENCE_PENALTY:-1.5}
rollout_repetition_penalty=${ROLLOUT_REPETITION_PENALTY:-1.0}
rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.75}
rollout_max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-1}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.5}

teacher_crop_enabled=${TEACHER_CROP_ENABLED:-True}
teacher_crop_prompt_key=${TEACHER_CROP_PROMPT_KEY:-teacher_raw_prompt}
teacher_crop_image_key=${TEACHER_CROP_IMAGE_KEY:-teacher_images}
teacher_crop_extra_tokens=${TEACHER_CROP_EXTRA_TOKENS:-0}
teacher_max_model_len_override=${TEACHER_MAX_MODEL_LEN:-32768}
data_seed=${DATA_SEED:-42}

val_n=${VAL_N:-1}
# max new token
# 超参数
# llm as judge

distillation_enabled=${DISTILLATION_ENABLED:-False}
distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
distillation_topk=${DISTILLATION_TOPK:-64}
use_policy_gradient=${USE_POLICY_GRADIENT:-False}

project_name=${PROJECT_NAME:-verl_distill_zwz_vqa}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_4b_val_before_train_2gpu_tmp}
logger=${LOGGER:-'["console"]'}
rollout_data_dir=${ROLLOUT_DATA_DIR:-./logs/rollouts_val_before_train_2gpu_tmp}
validation_data_dir=${VALIDATION_DATA_DIR:-./logs/val_before_train_2gpu_tmp}
log_val_generations=${LOG_VAL_GENERATIONS:-0}
# ---- end user-adjustable ----

vqa_judge_env_file=${VQA_JUDGE_ENV_FILE:-env/.openai_api.env}
if [[ -f "${vqa_judge_env_file}" ]]; then
    set -a
    source "${vqa_judge_env_file}"
    set +a
fi

vqa_judge_api_key=${VQA_JUDGE_API_KEY:-${JUDGE_API_KEY:-${JUDGE_OPENAI_API_KEY:-${API_KEY:-}}}}
vqa_judge_base_url=${VQA_JUDGE_BASE_URL:-${JUDGE_BASE_OPENAI_URL:-${JUDGE_OPENAI_BASE_URL:-${JUDGE_BASE_URL:-${BASE_OPENAI_URL:-}}}}}
vqa_judge_model=${VQA_JUDGE_MODEL:-${JUDGE_MODEL:-${API_JUDGE_MODEL:-mimo-v2.5}}}

if [[ -n "${vqa_judge_api_key}" ]]; then
    export VQA_JUDGE_API_KEY="${vqa_judge_api_key}"
fi
if [[ -n "${vqa_judge_base_url}" ]]; then
    export VQA_JUDGE_BASE_URL="${vqa_judge_base_url}"
fi
export VQA_JUDGE_MODEL="${vqa_judge_model}"
export VQA_JUDGE_RETRIES="${VQA_JUDGE_RETRIES:-${JUDGE_RETRIES:-2}}"
export VQA_JUDGE_RETRY_SLEEP="${VQA_JUDGE_RETRY_SLEEP:-${JUDGE_RETRY_SLEEP:-5}}"

if [[ -z "${VQA_JUDGE_API_KEY:-}" || -z "${VQA_JUDGE_BASE_URL:-}" ]]; then
    echo "WARNING: VQA judge API env is not fully set; validation judge rewards will score 0 on judge calls." >&2
fi

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
if [[ -n "${teacher_max_model_len_override}" ]]; then
    teacher_max_num_tokens=${teacher_max_model_len_override}
else
    teacher_max_num_tokens=$(( max_prompt_length + teacher_crop_extra_tokens + max_response_length + 1 ))
fi
echo "student max_model_len=${max_num_tokens}; teacher max_model_len=${teacher_max_num_tokens}; teacher_crop_extra_tokens=${teacher_crop_extra_tokens}" >&2

DATA=(
    data=qwen3_vl_crop_template
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$VAL_FILE']"
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.train_batch_size=${train_batch_size}
    data.seed=${data_seed}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers}
    data.truncation='error'
    data.use_system_prompt=${use_system_prompt}
    data.print_first_train_sample=${print_first_train_sample}
    data.validation_shuffle=False
    data.image_key=images
    +data.teacher_crop.enabled=${teacher_crop_enabled}
    +data.teacher_crop.prompt_key=${teacher_crop_prompt_key}
    +data.teacher_crop.image_key=${teacher_crop_image_key}
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]'
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.top_k=${rollout_top_k}
    actor_rollout_ref.rollout.presence_penalty=${rollout_presence_penalty}
    actor_rollout_ref.rollout.repetition_penalty=${rollout_repetition_penalty}
    actor_rollout_ref.rollout.do_sample=${rollout_do_sample}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.max_num_seqs=${rollout_max_num_seqs}
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.val_kwargs.n=${val_n}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${rollout_do_sample}
    actor_rollout_ref.rollout.val_kwargs.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${rollout_top_k}
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=${rollout_presence_penalty}
    actor_rollout_ref.rollout.val_kwargs.repetition_penalty=${rollout_repetition_penalty}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=False
    trainer.logger="${logger}"
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=True
    trainer.val_only=True
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=1
    trainer.total_training_steps=1
    trainer.rollout_data_dir=${rollout_data_dir}
    trainer.validation_data_dir=${validation_data_dir}
    trainer.log_val_generations=${log_val_generations}
)

EXTRA=(
    distillation.enabled=${distillation_enabled}
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_num_tokens}
    distillation.teacher_crop.enabled=${teacher_crop_enabled}
    distillation.teacher_crop.prompt_key=${teacher_crop_prompt_key}
    distillation.teacher_crop.image_key=${teacher_crop_image_key}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
