#!/usr/bin/env bash
# Fail fast (or install) before any HF Qwen3.5 logit / RoPE work on a GPU box.
set -euo pipefail

if ! python3 - <<'PY'
try:
    import causal_conv1d  # noqa: F401
    import fla  # noqa: F401
except ImportError:
    raise SystemExit(1)
PY
then
  echo "== installing qwen35 kernels (causal-conv1d, flash-linear-attention) =="
  pip install -q "causal-conv1d>=1.4.0" "flash-linear-attention>=0.2.0"
  python3 -c "import causal_conv1d, fla; print('qwen35 kernels ok')"
fi
