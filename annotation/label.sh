#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${env_file}"
    set +a
  fi
}

load_env_file ".project.env"
load_env_file ".seed.env"
load_env_file ".ssh_server.env"

if [[ -n "${REMOTE_CONDA_INIT:-}" ]]; then
  eval "${REMOTE_CONDA_INIT}"
fi

LABEL_CONDA_ENV="${LABEL_CONDA_ENV:-${SEED_CONDA_ENV:-${FILTER_CONDA_ENV:-verl}}}"
if [[ -n "${LABEL_CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available. Set REMOTE_CONDA_INIT in .project.env." >&2
    exit 1
  fi
  conda activate "${LABEL_CONDA_ENV}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m label.run "$@"
