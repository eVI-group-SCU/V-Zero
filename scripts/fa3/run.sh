#!/usr/bin/env bash
# Shared FA3 setup; invoked by the public training entry.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
VERL_REPO_ROOT=${VERL_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}
FA3_PROFILE=${FA3_PROFILE:-four_b}
case "${FA3_PROFILE}" in
    four_b) FA3_NODES=2; FA3_ARM=baseline ;;
    four_b_boundary) FA3_NODES=2; FA3_ARM=boundary ;;
    nine_b) FA3_NODES=4; FA3_ARM=baseline ;;
    *) echo "Unknown FA3_PROFILE: ${FA3_PROFILE}" >&2; exit 2 ;;
esac
FA3_MODE=${1:---launch}
case "${FA3_MODE}" in
    --prepare) FA3_MODE=prepare ;;
    --launch) FA3_MODE=launch ;;
    *) echo "FA3 accepts --prepare or --launch; set configuration through environment variables." >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo "Unexpected arguments; Set FA3 configuration through environment variables." >&2; exit 2
fi
export PROJECT_ROOT="${VERL_REPO_ROOT}"
: "${RUN_TAG:?Set RUN_TAG to a unique run name.}"
: "${VZERO_FA3_SHARED_MOUNT:?Set VZERO_FA3_SHARED_MOUNT to the actual shared mount.}"
: "${VZERO_FA3_ENV_ROOT:?Set VZERO_FA3_ENV_ROOT to the shared runtime directory containing .venv.}"
: "${VZERO_FA3_PYTHON_BASE:?Set VZERO_FA3_PYTHON_BASE to the runtime Python base prefix.}"
: "${VZERO_FA3_WANDB_ENV_FILE:?Set VZERO_FA3_WANDB_ENV_FILE to a private file exporting W&B settings.}"
export RUN_TAG VZERO_FA3_SHARED_MOUNT VZERO_FA3_ENV_ROOT VZERO_FA3_PYTHON_BASE VZERO_FA3_WANDB_ENV_FILE
export VZERO_FA3_TRAIN_PYTHON="${VZERO_FA3_TRAIN_PYTHON:-${VZERO_FA3_ENV_ROOT}/.venv/bin/python}"
export VZERO_FA3_TRAIN_OUTPUT_DIR="${VZERO_FA3_TRAIN_OUTPUT_DIR:-${PROJECT_ROOT}/results/${RUN_TAG}}"
export VZERO_FA3_TRAIN_HARNESS="${SCRIPT_DIR}/fa3/launch.py"
export VZERO_FA3_BOUNDARY_MODULE="${SCRIPT_DIR}/fa3/packed_boundary.py"
export VZERO_PACKED_BOUNDARY_ARM="${FA3_ARM}"
export VZERO_FA3_BACKEND="${VZERO_FA3_BACKEND:-kernels-community/flash-attn3@7cb368cf8278b583132eb72cbf312d54586df2e2}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-${VZERO_FA3_BACKEND}}"
if [[ "${ATTN_IMPLEMENTATION}" != "${VZERO_FA3_BACKEND}" ]]; then
    echo "ATTN_IMPLEMENTATION must match VZERO_FA3_BACKEND for audited FA3 training." >&2; exit 2
fi
export VZERO_FA3_FORWARD_TARGET="${VZERO_FA3_FORWARD_TARGET:-_flash_attn3_cuda_0f52d9d.flash_attn_interface:_flash_attn_forward}"
export VZERO_FA3_BACKWARD_TARGET="${VZERO_FA3_BACKWARD_TARGET:-_flash_attn3_cuda_0f52d9d.flash_attn_interface:_flash_attn_backward}"
export HF_HOME="${HF_HOME:-${VZERO_FA3_ENV_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export KERNELS_CACHE="${KERNELS_CACHE:-${VZERO_FA3_ENV_ROOT}/.cache/kernels}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FLA_DISABLE_BACKEND_DISPATCH=0 FLA_TILELANG=1
export VZERO_FA3_NNODES="${NNODES:-${FA3_NODES}}"
export VZERO_FA3_CPUS_PER_NODE="${CPUS_PER_NODE:-184}"
export NNODES="${VZERO_FA3_NNODES}" NGPUS_PER_NODE=4 TEACHER_WORLD_SIZE=4
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ACTOR_LR="${ACTOR_LR:-1e-6}" REW_OPD_ENABLED="${REW_OPD_ENABLED:-True}"
export VZERO_FA3_TRAIN_STEPS="${TOTAL_TRAINING_STEPS:-100}" VZERO_FA3_SAVE_FREQ="${SAVE_FREQ:-10}"
export VZERO_FA3_CHECKPOINT_MIN_FREE_GIB="${VZERO_FA3_CHECKPOINT_MIN_FREE_GIB:-48}"
export STUDENT_MODEL="${STUDENT_MODEL:-${MODEL_ROOT:+${MODEL_ROOT}/Qwen3.5-$([[ "${FA3_PROFILE}" == nine_b ]] && echo 9B || echo 4B)}}"
export TEACHER_MODEL="${TEACHER_MODEL:-${MODEL_ROOT:+${MODEL_ROOT}/Qwen3.5-27B}}"
: "${STUDENT_MODEL:?Set STUDENT_MODEL or MODEL_ROOT.}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL or MODEL_ROOT.}"
: "${TRAIN_FILE:?Set TRAIN_FILE.}"
: "${VAL_FILE:?Set VAL_FILE.}"
export TRAIN_FILE VAL_FILE
export LD_LIBRARY_PATH="${VZERO_FA3_PYTHON_BASE}/lib:${LD_LIBRARY_PATH:-}"
export PATH="$(dirname "${VZERO_FA3_TRAIN_PYTHON}"):${PATH}"
export PYTHONUNBUFFERED=1
if [[ "${FA3_MODE}" == prepare ]]; then
    # prepare() performs the full input/schema/image/model audit before writing prepared.json.
    export VZERO_FA3_INPUTS_VERIFIED=1
else
    : "${RANK:?Set the node rank (0-based).}"
    : "${MASTER_ADDR:?Set the head node address.}"
    export RANK MASTER_ADDR MASTER_PORT="${MASTER_PORT:-6379}"
    export WORLD_SIZE="${VZERO_FA3_NNODES}" NPROC_PER_NODE=8 VZERO_FA3_TRAIN_AUTHORIZED=1
    if [[ ! -f "${VZERO_FA3_TRAIN_OUTPUT_DIR}/prepared.json" ]]; then
        echo "Run this entry with --prepare before launching the nodes." >&2; exit 2
    fi
fi
exec "${VZERO_FA3_TRAIN_PYTHON}" -I -B "${VZERO_FA3_TRAIN_HARNESS}" "${FA3_MODE}"
