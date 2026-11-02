#!/usr/bin/env bash
# Prepare a DigitalOcean GPU Droplet (Ubuntu + NVIDIA drivers).
# Run ON the instance as root, from /root/unfuse (clone or rsync this repo first).
#
# DO images do not ship torch via apt. Creates /root/venv and installs CUDA torch
# from PyPI. HF_TOKEN must already be in the environment for Hub pulls — never
# write tokens to disk on a rented box.

set -euo pipefail

ROOT="${UNFUSE_ROOT:-/root/unfuse}"
VENV="${UNFUSE_VENV:-/root/venv}"

cd "$ROOT"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq software-properties-common git build-essential curl
if ! command -v python3.12 >/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
fi
PY=python3.12

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

if ! command -v nvidia-smi >/dev/null; then
  echo "FATAL: nvidia-smi not found — use a GPU Droplet image with NVIDIA drivers."
  exit 1
fi
nvidia-smi -L

if [ -n "${UNFUSE_TORCH_INDEX:-}" ]; then
  pip install -q torch torchvision --index-url "$UNFUSE_TORCH_INDEX"
else
  pip install -q "torch>=2.4" torchvision
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -q -e ".[notebook,gguf,dev,qwen35]"
pip install -q "huggingface_hub[cli]" datasets

python - <<'PY'
import importlib.util
for mod in ("causal_conv1d", "fla"):
    ok = importlib.util.find_spec(mod) is not None
    print(f"qwen35 kernel {mod}: {'OK' if ok else 'MISSING'}")
PY

python -m ipykernel install --user --name unfuse --display-name "unfuse (venv)"

python - <<'PY'
import torch, transformers
print("transformers", transformers.__version__, "at", transformers.__file__)
p = torch.cuda.get_device_properties(0)
print(f"gpu {p.name} sm_{p.major}{p.minor}")
print("  bf16 (needs sm_80):", p.major >= 8)
from transformers import Qwen3_5ForCausalLM  # noqa: F401
print("Qwen3_5ForCausalLM import OK")
PY

echo "SETUP_DONE venv=$VENV root=$ROOT"
