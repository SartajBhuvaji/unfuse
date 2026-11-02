# GPU venv: Qwen3.5 hybrid kernels

Transformers warns (and runs much slower) when these are missing on RoPE / logit work:

- `causal-conv1d` → import `causal_conv1d`
- `flash-linear-attention` → import `fla`

On the droplet after torch is in the venv:

```bash
source /root/venv/bin/activate
bash notebooks/qwen3.8-27b/followups/ensure_qwen35_kernels.sh
```

That script is invoked automatically by `run_rope_and_upload.sh`, `run_task1_rope.sh`, and `run_task1_rope_r3.sh`. Manual install:

```bash
pip install causal-conv1d flash-linear-attention
# or from repo root:
pip install -e ".[qwen35]"
python -c "import causal_conv1d, fla; print('ok')"
```

Fresh DO boxes: `tools/setup_instance_do.sh` now pulls the `qwen35` extra automatically.
