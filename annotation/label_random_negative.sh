#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for env_file in .project.env .seed.env; do
  if [[ -f "$env_file" ]]; then
    set -a
    source "$env_file"
    set +a
  fi
done
exec "${PYTHON_BIN:-python}" -m annotation.label.random_negative "$@"
