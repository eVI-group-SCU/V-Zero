#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [[ -f ".seed.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".seed.env"
  set +a
fi

if [[ -n "${ARK_BASE_URL:-}" && -n "${ARK_MODEL:-}" ]]; then
  export JUDGE=seed
fi

if [[ -n "${REMOTE_HF_ENDPOINT:-}" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="${REMOTE_HF_ENDPOINT}"
fi

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

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

ARGS=("$@")
HAS_DATASET=0
for arg in "${ARGS[@]}"; do
  if [[ "${arg}" == "--dataset" || "${arg}" == --dataset=* ]]; then
    HAS_DATASET=1
    break
  fi
done

if [[ "${HAS_DATASET}" -eq 0 ]]; then
  ARGS=(--dataset default "${ARGS[@]}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
exec "${PYTHON_BIN}" -m eval.run "${ARGS[@]}"
