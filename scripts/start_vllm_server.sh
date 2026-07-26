#!/usr/bin/env bash
set -euo pipefail

HOST="127.0.0.1"
PORT="8000"
TENSOR_PARALLEL_SIZE="1"
DATA_PARALLEL_SIZE="1"
GPU_MEMORY_UTILIZATION="0.85"
MAX_MODEL_LEN="32768"
TRUST_REMOTE_CODE="1"
MODEL=""
SERVED_MODEL_NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --served-model-name)
      SERVED_MODEL_NAME="$2"
      shift 2
      ;;
    --served-model-name=*)
      SERVED_MODEL_NAME="${1#*=}"
      shift
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --host=*)
      HOST="${1#*=}"
      shift
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --port=*)
      PORT="${1#*=}"
      shift
      ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --tensor-parallel-size=*)
      TENSOR_PARALLEL_SIZE="${1#*=}"
      shift
      ;;
    --data-parallel-size)
      DATA_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --data-parallel-size=*)
      DATA_PARALLEL_SIZE="${1#*=}"
      shift
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --gpu-memory-utilization=*)
      GPU_MEMORY_UTILIZATION="${1#*=}"
      shift
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --max-model-len=*)
      MAX_MODEL_LEN="${1#*=}"
      shift
      ;;
    --trust-remote-code)
      TRUST_REMOTE_CODE="1"
      shift
      ;;
    --no-trust-remote-code)
      TRUST_REMOTE_CODE="0"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Usage: $0 --model MODEL [--served-model-name NAME]" >&2
  exit 2
fi

if [[ -z "${SERVED_MODEL_NAME}" ]]; then
  SERVED_MODEL_NAME="$(basename "${MODEL}")"
fi

ensure_positive_int_env() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" || ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
    export "${name}=1"
  fi
}

ensure_positive_int_env OMP_NUM_THREADS
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
fi
export FLA_DISABLE_BACKEND_DISPATCH="${FLA_DISABLE_BACKEND_DISPATCH:-1}"

CMD=(
  python -m vllm.entrypoints.openai.api_server
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  CMD+=(--trust-remote-code)
fi

exec "${CMD[@]}"
