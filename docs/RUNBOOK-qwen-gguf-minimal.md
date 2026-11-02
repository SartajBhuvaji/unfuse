# Minimal run: Qwen3.8-27B GGUF + benchmarks

Same branch (`qwen-gguf-and-sweeps`), same rules as the full runbook (tokens in env
only, private Hub if you upload at all, no weights in git). This is the **small**
path; the full ladder and sweep matrix live in `RUNBOOK-qwen-gguf-and-sweeps.md`.

## Already on the droplet (2026-11-02)

- Text safetensors under `~/models/Qwen3.8-27B-text`
- BF16 GGUF `~/gguf/Qwen3.8-27B-text-BF16.gguf`
- llama.cpp built (see `notebooks/qwen3.8-27b/gguf/llama_cpp_rev.txt`)

Do **not** re-download or re-convert BF16 unless those files are missing.

## Phase A — validate GGUF (~2–4 h)

On the GPU box, with `HF_TOKEN` exported in the shell (never written to disk):

```bash
cd /root/unfuse && source /root/venv/bin/activate
bash tools/run_phase_a_minimal.sh
```

Produces under `notebooks/qwen3.8-27b/gguf/`:

- `Q4_K_M` quant (only), RoPE warning + metadata, smoke log, optional perplexity (BF16 + Q4)
- mmproj F16 + one mtmd image test (downloads parent ~56 GB once)

Hub upload is **off** unless `UPLOAD=1` (then only `Q4_K_M.gguf` + `mmproj-…-F16.gguf`).

## Phase B — benchmarks (~2–4 h)

After Phase A, same machine:

```bash
bash tools/run_phase_b_minimal.sh
```

Three runs under `notebooks/qwen3.8-27b/bench/sweeps/`:

1. Baseline 512/256, `CONC="1 16"`, unpinned scheduler
2. Long context `IN_LEN=4096`, `CTX=8192`, `NPROMPTS=32`
3. Concurrency `CONC=32` with `MAX_SEQS=32` and `MAX_RUNNING=32`

## Deliverables

Commit JSON/logs and `notebooks/qwen3.8-27b/RESULTS.md`. Pull artifacts from the
droplet with rsync before committing.
