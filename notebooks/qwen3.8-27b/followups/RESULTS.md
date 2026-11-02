# Qwen3.8-27B follow-ups (`qwen-gguf-followups`)

Machine: 1× H100 80GB, Ubuntu 22.04, llama.cpp `d02b4c0d66796e0a44380611d038123fefb9fe83`, vLLM **0.27.1**, SGLang **0.5.18** (SGLang not re-run here). Branch: `qwen-gguf-followups`. Artifacts under this directory only.

## Findings

**RoPE warning:** Converter still emits `Unknown RoPE type: default`; GGUF metadata carries `qwen35.rope.dimension_sections` / `rope.freq_base` (see prior `gguf/` logs). **Safetensors self-check** on seven prompts (six defaults + ~1086-word long prompt) is **exact** (max |Δlogit| = 0). **`tools/qwen_gguf_rope_check.py` failed** on greedy-64 (`apply_chat_template` BatchEncoding passed to `generate`). **r3** `rope_check.py` ( `--reasoning off`, qwen35 kernels installed) finished **2026-11-02T22:45Z** but **GGUF first-token parse is still wrong**: stderr lines like `Loading` are mistaken for the sampled token (**0/7 argmax match** vs transformers next-token). Full-vocab **max |Δlogit|** GGUF vs HF was **not** measured.

**Quants:** Q8_0 and Q5_K_M cut on disk; smokes **PASS**. Full **wikitext-2 test** perplexity @ ctx **512** (not the old 256 KiB slice): BF16 **6.6497**, Q8_0 **6.6514**, Q5_K_M **6.6746**, Q4_K_M **6.6959** — table in `perplexity_full_table.md`.

**max-num-seqs vs ~360 tok/s:** Prior baseline used **`--max-num-seqs 256`**. Sweeping **64 / 128 / 256 / 320** at 512/256 warm **c=16** shows **~360–402 tok/s** aggregate for both parent and split; **no** return to published **555–636 tok/s** band at any cap tested. **320 started** for both models (see `maxseq-320/`). KV budget from serve JSON rises slightly with cap; split KV **≥** parent at equal cap (vision encoder budget on parent).

**Long context (vLLM, CTX=40960):** **8k** in / 256 out: parent **97.0** vs split **95.0** tok/s at c=16 (KV **278674** vs **283794**). **32k** in: **26.3** vs **26.4** tok/s at c=16; **15.2** tok/s both at c=1. Split KV advantage stays **modest** (~1–9% more tokens), not a large win even at 32k.

**SGLang parent @ c=32:** Read-only — parent capped to **5** running requests by Mamba state cache despite CLI **32**; see `task5_sglang_analysis.md`.

---

## Numbers

### Task 2 — full wikitext-2 test PPL (ctx 512)

| Quant | PPL |
|-------|-----|
| BF16 | 6.6497 |
| Q8_0 | 6.6514 |
| Q5_K_M | 6.6746 |
| Q4_K_M | 6.6959 |

### Task 3 — baseline 512/256, warm c=16, vLLM only

| max-num-seqs | Parent tok/s | Split tok/s | KV tokens (parent / split) |
|--------------|-------------|-------------|----------------------------|
| 64 | 402.6 | 398.4 | 284717 / 289814 |
| 128 | 367.9 | 359.9 | 281804 / 286902 |
| 256 | 381.3 | 377.1 | 277435 / 301465 |
| 320 | 379.9 | 380.1 | 280348 / 300737 |

Prior minimal run @ max-seq **256**: ~**360** tok/s (both). Mamba cache block counts at failed 1024-seq startup: parent **334**, split **362** (`baseline-512-vllm-refusal.json`).

### Task 4 — long context, vLLM, CTX=40960, warm

| Input len | Conc | Parent tok/s | Split tok/s | KV (parent / split) |
|-----------|------|-------------|-------------|---------------------|
| 8192 | 16 | 97.0 | 95.0 | 278674 / 283794 |
| 32768 | 16 | 26.3 | 26.4 | 278674 / 302811 |
| 32768 | 1 | 15.2 | 15.2 | (same serve rows) |

---

## Not run / failed

| Item | Status |
|------|--------|
| `tools/qwen_gguf_rope_check.py` | **Failed** — `AttributeError` on greedy-64 (`task1_official_failure.txt`) |
| Full GGUF vs HF max-logit vector diff | **Not run** (no full-vocab GGUF logits in tooling) |
| RoPE r3 with fixed parser | **Ran** 2026-11-02 ~22:45Z with qwen35 kernels; parser still grabs `Loading` from llama-cli progress → **0/7 argmax match** (see `rope_check.json`) |
| vLLM @ max-seq 1024 | **Refused** (unchanged from minimal run) |
| SGLang sweeps in follow-ups | **Not run** (Task 5 read-only) |
| Hub upload / model card edits | **Not run** (per scope) |

---

## Commands (on GPU, from repo root)

```bash
# Task 1 (official — fails on greedy-64)
python tools/qwen_gguf_rope_check.py \
  --hf-dir ~/models/Qwen3.8-27B-text \
  --gguf ~/gguf/Qwen3.8-27B-text-BF16.gguf \
  --llama-bin ~/llama.cpp/build/bin/llama-cli \
  --out notebooks/qwen3.8-27b/followups/rope_check_official.json

# Task 1 follow-up
python notebooks/qwen3.8-27b/followups/rope_check.py

# Task 2 (after Q8 on disk)
bash notebooks/qwen3.8-27b/followups/run_task2_resume.sh

# Tasks 3–4
bash notebooks/qwen3.8-27b/followups/run_task3_vllm_maxseq.sh
bash notebooks/qwen3.8-27b/followups/run_task4_vllm_longctx.sh

# Optional faster transformers Mamba paths
pip install causal-conv1d flash-linear-attention   # or pip install -e ".[qwen35]"
```

Chain driver: `bash notebooks/qwen3.8-27b/followups/run_followups_chain.sh` (finished **2026-11-02T22:21:08Z**).
