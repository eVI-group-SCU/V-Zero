#!/usr/bin/env bash
# VA-OPD paper defaults; infrastructure and sampled reverse-KL estimator from V-Zero.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export STUDENT_MODEL=${STUDENT_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
export TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen3-VL-8B-Instruct}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_N=${ROLLOUT_N:-4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
export RUN_TAG=${RUN_TAG:-va_opd_qwen3_vl}
export PROJECT_NAME=${PROJECT_NAME:-va_opd}
export TEACHER_CROP_PROMPT_KEY=raw_prompt
export TEACHER_CROP_IMAGE_KEY=images
export REW_OPD_ENABLED=True
export REW_OPD_NEGATIVE_MODE=pixelate_teacher_image
exec bash "${SCRIPT_DIR}/run_vzero_qwen35_vl_fsdp.sh" \
    data.teacher_crop.enabled=False \
    distillation.evidence_weighting.objective=va_opd \
    distillation.evidence_weighting.temperature="${VA_TEMPERATURE:-1.0}" \
    distillation.evidence_weighting.high_va_fraction="${VA_HIGH_FRACTION:-0.2}" \
    distillation.evidence_weighting.high_va_weight="${VA_HIGH_WEIGHT:-0.5}" \
    distillation.evidence_weighting.pixelation_ratio="${VA_PIXELATION_RATIO:-0.1}" \
    distillation.evidence_weighting.rollout_reweighting="${VA_ROLLOUT_REWEIGHTING:-True}" \
    distillation.evidence_weighting.token_grouping="${VA_TOKEN_GROUPING:-True}" \
    distillation.distillation_loss.loss_max_clamp=null \
    distillation.distillation_loss.log_prob_min_clamp=null \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
    "$@"
