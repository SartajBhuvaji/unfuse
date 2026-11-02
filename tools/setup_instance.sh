#!/usr/bin/env bash
# Prepare a fresh Lambda Stack instance. Run ON the instance.
#
# Lambda Stack ships torch + CUDA via apt but NOT transformers. The venv uses
# --system-site-packages so it inherits the apt torch instead of pulling a second
# wheel that may not match the installed CUDA.

set -euo pipefail

python3 -m venv --system-site-packages ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# transformers 5.15.1 (2026-08-19) is the first release carrying gemma4_unified,
# so a plain install is enough now. Earlier versions need git main for Gemma 4.
pip install -q "transformers>=5.15.1"
pip install -q accelerate safetensors pillow huggingface_hub ipykernel jupyterlab
# Qwen3.5 hybrid checkpoints (RoPE / logit checks in transformers): avoid slow Mamba fallbacks.
pip install -q "causal-conv1d>=1.4.0" "flash-linear-attention>=0.2.0"

python -m ipykernel install --user --name unfuse --display-name "unfuse (venv)"

python - <<'PY'
import torch, transformers
print("transformers", transformers.__version__, "at", transformers.__file__)
p = torch.cuda.get_device_properties(0)
print(f"gpu {p.name} sm_{p.major}{p.minor}")
print("  bf16   (needs sm_80) :", p.major >= 8)
print("  fp8    (needs sm_89) :", p.major > 8 or (p.major == 8 and p.minor >= 9))
print("  nvfp4  (needs sm_100):", p.major >= 10)
from transformers import Glm4vForConditionalGeneration, Glm4ForCausalLM, Glm4Config
print("glm4v + glm4 classes import OK")
PY

echo "SETUP_DONE"
