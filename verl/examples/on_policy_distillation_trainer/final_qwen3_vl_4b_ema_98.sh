#!/usr/bin/env bash
# On-policy distillation | vision (Qwen3-VL teacher crop) | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail
export VERL_PRINT_STEP_PROGRESS=${VERL_PRINT_STEP_PROGRESS:-1}
export VERL_TEACHER_LOGPROB_STREAMING_MODE=${VERL_TEACHER_LOGPROB_STREAMING_MODE:-nonblocking}
export OMP_NUM_THREADS=${VERL_OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${VERL_MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${VERL_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${VERL_NUMEXPR_NUM_THREADS:-1}

# ---- user-adjustable ----
MODEL_ROOT=${MODEL_ROOT:-/root/autodl-tmp/models}
STUDENT_MODEL=${STUDENT_MODEL:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}
teacher_ema_enabled=${TEACHER_EMA_ENABLED:-True}
if [[ "${teacher_ema_enabled}" == "True" || "${teacher_ema_enabled}" == "true" || "${teacher_ema_enabled}" == "1" ]]; then
    TEACHER_MODEL=${TEACHER_MODEL:-$STUDENT_MODEL}
else
    TEACHER_MODEL=${TEACHER_MODEL:-${MODEL_ROOT}/Qwen3-VL-4B-Instruct}
fi

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_k=${ROLLOUT_TOP_K:--1}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}
rollout_presence_penalty=${ROLLOUT_PRESENCE_PENALTY:-0.0}
rollout_repetition_penalty=${ROLLOUT_REPETITION_PENALTY:-1.0}
rollout_do_sample=${ROLLOUT_DO_SAMPLE:-True}

train_batch_size=${TRAIN_BATCH_SIZE:-256}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-25000}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS:-False}
filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS:-4}
data_seed=${DATA_SEED:-67280421310721}
use_system_prompt=${USE_SYSTEM_PROMPT:-False}
print_first_train_sample=${PRINT_FIRST_TRAIN_SAMPLE:-True}
# teacher prompt uses <image> for data.image_key images and <crop> for teacher_crop_image_key images.
teacher_crop_enabled=${TEACHER_CROP_ENABLED:-True}
teacher_crop_prompt_key=${TEACHER_CROP_PROMPT_KEY:-teacher_raw_prompt}
teacher_crop_image_key=${TEACHER_CROP_IMAGE_KEY:-teacher_images}
teacher_crop_extra_tokens=${TEACHER_CROP_EXTRA_TOKENS:-0}
teacher_crop_async_prepare_enabled=${TEACHER_CROP_ASYNC_PREPARE_ENABLED:-True}
teacher_crop_async_prepare_concurrency=${TEACHER_CROP_ASYNC_PREPARE_CONCURRENCY:-256}
teacher_max_model_len_override=${TEACHER_MAX_MODEL_LEN:-32768}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}
teacher_ema_decay=${TEACHER_EMA_DECAY:-0.98}
teacher_ema_dtype=${TEACHER_EMA_DTYPE:-bf16}
teacher_ema_update_freq=${TEACHER_EMA_UPDATE_FREQ:-1}
teacher_ema_sync_freq=${TEACHER_EMA_SYNC_FREQ:-1}
teacher_ema_warmup_steps=${TEACHER_EMA_WARMUP_STEPS:-0}

checkpoint_engine_backend_default=naive
checkpoint_engine_nccl_rebuild_group_default=False
if [[ "${teacher_ema_enabled}" == "True" || "${teacher_ema_enabled}" == "true" || "${teacher_ema_enabled}" == "1" ]]; then
    checkpoint_engine_backend_default=nccl
    checkpoint_engine_nccl_rebuild_group_default=True
fi
checkpoint_engine_backend=${CHECKPOINT_ENGINE_BACKEND:-${checkpoint_engine_backend_default}}
checkpoint_engine_nccl_rebuild_group=${CHECKPOINT_ENGINE_NCCL_REBUILD_GROUP:-${checkpoint_engine_nccl_rebuild_group_default}}
update_weights_bucket_mb=${UPDATE_WEIGHTS_BUCKET_MB:-512}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.6}
rollout_max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
teacher_tp=${TEACHER_TP:-2}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.5}

total_epochs=${TOTAL_EPOCHS:-10}
save_freq=${SAVE_FREQ:-5}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-False}
rollout_data_dir=${ROLLOUT_DATA_DIR:-./final_logs_ema_98/rollouts}
validation_data_dir=${VALIDATION_DATA_DIR:-./final_logs_ema_98/val_generations}
log_val_generations=${LOG_VAL_GENERATIONS:-0}
val_n=${VAL_N:-1}

vqa_judge_env_file=${VQA_JUDGE_ENV_FILE:-env/.openai_api.env}

project_name=${PROJECT_NAME:-verl_distill_zwz_vqa}
experiment_name=${EXPERIMENT_NAME:-qwen3_vl_4b_from_qwen3_vl_8b_crop_vllm_fsdp}
# ---- end user-adjustable ----

_xtrace_enabled=0
case "$-" in
    *x*)
        _xtrace_enabled=1
        set +x
        ;;
esac
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
if [[ -z "${VQA_JUDGE_API_KEY:-}" || -z "${VQA_JUDGE_BASE_URL:-}" ]]; then
    echo "WARNING: VQA judge API env is not fully set; validation judge rewards will score 0 on judge calls." >&2
fi
if [[ "${_xtrace_enabled}" == "1" ]]; then
    set -x
fi

filtered_train_file=${FILTERED_TRAIN_FILE:-/root/autodl-tmp/dataset/ZwZ-RL-VQA/verl_format_teacher_crop_train37k_filtered_maxprompt25000_teachermax32768/train.parquet}
train_file=${TRAIN_FILE:-${filtered_train_file}}
val_file=${VAL_FILE:-/root/autodl-tmp/dataset/validation_450/vstar_hrbench_mme_realworld_val450.parquet}
val_batch_size=${VAL_BATCH_SIZE:-128}

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
if [[ -n "${teacher_max_model_len_override}" ]]; then
    teacher_max_num_tokens=${teacher_max_model_len_override}
else
    teacher_max_num_tokens=$(( max_prompt_length + teacher_crop_extra_tokens + max_response_length + 1 ))
fi
echo "student max_model_len=${max_num_tokens}; teacher max_model_len=${teacher_max_num_tokens}; teacher_crop_extra_tokens=${teacher_crop_extra_tokens}" >&2
########################### parameter arrays ###########################

DATA=(
    data=qwen3_vl_crop_template
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$train_file']"
    data.val_files="['$val_file']"
    data.val_batch_size=${val_batch_size}
    data.train_batch_size=${train_batch_size}
    data.seed=${data_seed}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=${filter_overlong_prompts}
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers}
    data.truncation='error'
    data.use_system_prompt=${use_system_prompt}
    data.print_first_train_sample=${print_first_train_sample}
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
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
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
    actor_rollout_ref.rollout.checkpoint_engine.backend=${checkpoint_engine_backend}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_mb}
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=${checkpoint_engine_nccl_rebuild_group}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${val_before_train}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.rollout_data_dir=${rollout_data_dir}
    trainer.validation_data_dir=${validation_data_dir}
    trainer.log_val_generations=${log_val_generations}
)

EXTRA=(
    distillation.enabled=True
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
    +distillation.teacher_crop.async_prepare.enabled=${teacher_crop_async_prepare_enabled}
    +distillation.teacher_crop.async_prepare.concurrency=${teacher_crop_async_prepare_concurrency}
    distillation.teacher_ema.enabled=${teacher_ema_enabled}
    distillation.teacher_ema.decay=${teacher_ema_decay}
    distillation.teacher_ema.device=cpu
    distillation.teacher_ema.dtype=${teacher_ema_dtype}
    distillation.teacher_ema.update_freq=${teacher_ema_update_freq}
    distillation.teacher_ema.sync_freq=${teacher_ema_sync_freq}
    distillation.teacher_ema.warmup_steps=${teacher_ema_warmup_steps}
    distillation.teacher_ema.init_from_student=True
    distillation.teacher_ema.require_same_model=True
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
python3 -u -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
