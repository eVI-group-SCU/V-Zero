#!/usr/bin/env bash
# V-Zero on-policy distillation launcher.
#
# This script launches a Qwen3.5-VL student with V-Zero evidence-gated
# on-policy distillation from a teacher model. It expects a verl checkout or
# installed package that contains the V-Zero/OPD distillation extensions.

set -euo pipefail
if [[ "${DEBUG:-0}" == "1" ]]; then
    set -x
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_VERL_REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)

if [[ -n "${VERL_REPO_ROOT:-}" ]]; then
    if [[ ! -d "${VERL_REPO_ROOT}/verl" ]]; then
        echo "VERL_REPO_ROOT does not look like a verl checkout: ${VERL_REPO_ROOT}" >&2
        exit 2
    fi
else
    VERL_REPO_ROOT="${DEFAULT_VERL_REPO_ROOT}"
fi

cd "${VERL_REPO_ROOT}"
export PYTHONPATH="${VERL_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi

export FLA_DISABLE_BACKEND_DISPATCH=${FLA_DISABLE_BACKEND_DISPATCH:-1}
export VERL_PRINT_STEP_PROGRESS=${VERL_PRINT_STEP_PROGRESS:-1}
export VERL_TEACHER_LOGPROB_STREAMING_MODE=${VERL_TEACHER_LOGPROB_STREAMING_MODE:-nonblocking}
export OMP_NUM_THREADS=${VERL_OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${VERL_MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${VERL_OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${VERL_NUMEXPR_NUM_THREADS:-1}

# ---- required paths ----
MODEL_ROOT=${MODEL_ROOT:-}
STUDENT_MODEL=${STUDENT_MODEL:-${MODEL_ROOT:+${MODEL_ROOT}/Qwen3.5-4B}}
TEACHER_MODEL=${TEACHER_MODEL:-${MODEL_ROOT:+${MODEL_ROOT}/Qwen3.5-27B}}
TRAIN_FILE=${TRAIN_FILE:-}
VAL_FILE=${VAL_FILE:-}

: "${STUDENT_MODEL:?Set STUDENT_MODEL or MODEL_ROOT.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL or MODEL_ROOT.}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the training parquet path.}"
: "${VAL_FILE:?Set VAL_FILE to the validation parquet path.}"

# ---- model and optimization ----
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4}

ACTOR_LR=${ACTOR_LR:-1e-6}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}

DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}

# ---- rollout generation ----
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-32768}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
ROLLOUT_PRESENCE_PENALTY=${ROLLOUT_PRESENCE_PENALTY:-0.0}
ROLLOUT_REPETITION_PENALTY=${ROLLOUT_REPETITION_PENALTY:-1.0}
ROLLOUT_DO_SAMPLE=${ROLLOUT_DO_SAMPLE:-True}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_N=${VAL_N:-1}

# ---- teacher inference ----
TEACHER_TP=${TEACHER_TP:-2}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.5}
TEACHER_MAX_MODEL_LEN=${TEACHER_MAX_MODEL_LEN:-32768}

# ---- data ----
DATA_SEED=${DATA_SEED:-67280421310721}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-25000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
FILTER_OVERLONG_PROMPTS=${FILTER_OVERLONG_PROMPTS:-False}
FILTER_OVERLONG_PROMPTS_WORKERS=${FILTER_OVERLONG_PROMPTS_WORKERS:-4}
USE_SYSTEM_PROMPT=${USE_SYSTEM_PROMPT:-False}
PRINT_FIRST_TRAIN_SAMPLE=${PRINT_FIRST_TRAIN_SAMPLE:-True}
APPLY_CHAT_TEMPLATE_ENABLE_THINKING=${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}

# teacher prompt uses <image> for DATA_IMAGE_KEY and <crop> for TEACHER_CROP_IMAGE_KEY.
DATA_IMAGE_KEY=${DATA_IMAGE_KEY:-images}
TEACHER_CROP_ENABLED=${TEACHER_CROP_ENABLED:-True}
TEACHER_CROP_PROMPT_KEY=${TEACHER_CROP_PROMPT_KEY:-teacher_raw_prompt}
TEACHER_CROP_IMAGE_KEY=${TEACHER_CROP_IMAGE_KEY:-teacher_images}
TEACHER_CROP_EXTRA_TOKENS=${TEACHER_CROP_EXTRA_TOKENS:-0}
TEACHER_CROP_ASYNC_PREPARE_ENABLED=${TEACHER_CROP_ASYNC_PREPARE_ENABLED:-True}
TEACHER_CROP_ASYNC_PREPARE_CONCURRENCY=${TEACHER_CROP_ASYNC_PREPARE_CONCURRENCY:-32}

# ---- V-Zero evidence weighting ----
REW_OPD_ENABLED=${REW_OPD_ENABLED:-True}
REW_OPD_NEGATIVE_MODE=${REW_OPD_NEGATIVE_MODE:-hard_negative_crop}
REW_OPD_NEGATIVE_IMAGE_KEY=${REW_OPD_NEGATIVE_IMAGE_KEY:-teacher_neg_images}
REW_OPD_NEGATIVE_PROMPT_KEY=${REW_OPD_NEGATIVE_PROMPT_KEY:-teacher_neg_raw_prompt}
REW_OPD_DOWNSAMPLE_FACTOR=${REW_OPD_DOWNSAMPLE_FACTOR:-2}
REW_OPD_GAMMA=${REW_OPD_GAMMA:-1.0}
REW_OPD_ALPHA=${REW_OPD_ALPHA:-0.5}
REW_OPD_W_MIN=${REW_OPD_W_MIN:-0.0}
REW_OPD_W_MAX=${REW_OPD_W_MAX:-2.0}
REW_OPD_EPS=${REW_OPD_EPS:-1e-6}

# ---- checkpoints and logging ----
CHECKPOINT_ENGINE_BACKEND=${CHECKPOINT_ENGINE_BACKEND:-naive}
CHECKPOINT_ENGINE_NCCL_REBUILD_GROUP=${CHECKPOINT_ENGINE_NCCL_REBUILD_GROUP:-False}
UPDATE_WEIGHTS_BUCKET_MB=${UPDATE_WEIGHTS_BUCKET_MB:-512}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-1000}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
RUN_TAG=${RUN_TAG:-vzero_qwen35_vl}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-./final_logs/${RUN_TAG}/rollouts}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-./final_logs/${RUN_TAG}/val_generations}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
PROJECT_NAME=${PROJECT_NAME:-vzero}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${RUN_TAG}}

max_num_tokens=$(( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1 ))
teacher_max_num_tokens=${TEACHER_MAX_MODEL_LEN}
echo "student max_model_len=${max_num_tokens}; teacher max_model_len=${teacher_max_num_tokens}" >&2
echo "teacher positive image key=${TEACHER_CROP_IMAGE_KEY}; negative image key=${REW_OPD_NEGATIVE_IMAGE_KEY}" >&2

DATA=(
    data=qwen3_vl_crop_template
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.seed=${DATA_SEED}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=${FILTER_OVERLONG_PROMPTS}
    data.filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS}
    data.truncation='error'
    data.use_system_prompt=${USE_SYSTEM_PROMPT}
    data.print_first_train_sample=${PRINT_FIRST_TRAIN_SAMPLE}
    +data.apply_chat_template_kwargs.enable_thinking=${APPLY_CHAT_TEMPLATE_ENABLE_THINKING}
    data.image_key=${DATA_IMAGE_KEY}
    +data.teacher_crop.enabled=${TEACHER_CROP_ENABLED}
    +data.teacher_crop.prompt_key=${TEACHER_CROP_PROMPT_KEY}
    +data.teacher_crop.image_key=${TEACHER_CROP_IMAGE_KEY}
)

if [[ "${REW_OPD_ENABLED}" != "False" && "${REW_OPD_ENABLED}" != "false" && "${REW_OPD_NEGATIVE_MODE}" == "hard_negative_crop" ]]; then
    DATA+=(
        +data.teacher_crop.negative_image_key=${REW_OPD_NEGATIVE_IMAGE_KEY}
        +data.teacher_crop.negative_prompt_key=${REW_OPD_NEGATIVE_PROMPT_KEY}
    )
fi

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.checkpoint.save_contents='["model"]'
    actor_rollout_ref.actor.checkpoint.load_contents='["model"]'
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}
    actor_rollout_ref.rollout.presence_penalty=${ROLLOUT_PRESENCE_PENALTY}
    actor_rollout_ref.rollout.repetition_penalty=${ROLLOUT_REPETITION_PENALTY}
    actor_rollout_ref.rollout.do_sample=${ROLLOUT_DO_SAMPLE}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${ROLLOUT_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${ROLLOUT_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.presence_penalty=${ROLLOUT_PRESENCE_PENALTY}
    actor_rollout_ref.rollout.val_kwargs.repetition_penalty=${ROLLOUT_REPETITION_PENALTY}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND}
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${UPDATE_WEIGHTS_BUCKET_MB}
    +actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group=${CHECKPOINT_ENGINE_NCCL_REBUILD_GROUP}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${teacher_max_num_tokens}
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.mm_processor_cache_gb=0
    distillation.teacher_crop.enabled=${TEACHER_CROP_ENABLED}
    distillation.teacher_crop.prompt_key=${TEACHER_CROP_PROMPT_KEY}
    distillation.teacher_crop.image_key=${TEACHER_CROP_IMAGE_KEY}
    +distillation.teacher_crop.async_prepare.enabled=${TEACHER_CROP_ASYNC_PREPARE_ENABLED}
    +distillation.teacher_crop.async_prepare.concurrency=${TEACHER_CROP_ASYNC_PREPARE_CONCURRENCY}
    distillation.evidence_weighting.enabled=${REW_OPD_ENABLED}
    distillation.evidence_weighting.negative_mode=${REW_OPD_NEGATIVE_MODE}
    distillation.evidence_weighting.negative_image_key=${REW_OPD_NEGATIVE_IMAGE_KEY}
    distillation.evidence_weighting.negative_prompt_key=${REW_OPD_NEGATIVE_PROMPT_KEY}
    distillation.evidence_weighting.downsample_factor=${REW_OPD_DOWNSAMPLE_FACTOR}
    distillation.evidence_weighting.gamma=${REW_OPD_GAMMA}
    distillation.evidence_weighting.alpha=${REW_OPD_ALPHA}
    distillation.evidence_weighting.w_min=${REW_OPD_W_MIN}
    distillation.evidence_weighting.w_max=${REW_OPD_W_MAX}
    distillation.evidence_weighting.eps=${REW_OPD_EPS}
    distillation.teacher_ema.enabled=False
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

python3 -u -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
