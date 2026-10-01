#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="python3"
fi
exec "${PYTHON}" main.py \
  --model Qwen/Qwen3.5-0.8B \
  --prompt "Tell me a short story." \
  --benchmark true \
  --stream true