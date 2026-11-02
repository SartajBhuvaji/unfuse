# GPU venv: Qwen3.5 hybrid kernels

Transformers warns (and runs much slower) when these are missing on RoPE / logit work:

- `causal-conv1d` → import `causal_conv1d`
- `flash-linear-attention` → import `fla`

On the droplet after torch is in the venv:

```bash
source /root/venv/bin/activate
pip install causal-conv1d flash-linear-attention
# or from repo root:
pip install -e ".[qwen35]"
python -c "import causal_conv1d, fla; print('ok')"
```

Fresh DO boxes: `tools/setup_instance_do.sh` now pulls the `qwen35` extra automatically.
