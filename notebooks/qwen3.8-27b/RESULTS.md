# Qwen3.8-27B minimal run (`qwen-gguf-and-sweeps`)

Machine: 1× H100 80GB, Ubuntu 22.04, llama.cpp `d02b4c0d66796e0a44380611d038123fefb9fe83`, vLLM 0.27.1, SGLang 0.5.18. Branch commits on `qwen-gguf-and-sweeps`; no changes to `main`.

## Findings

**GGUF:** Text-only bf16 convert + **Q4_K_M** quant succeeded after redoing a truncated bf16 file. llamacpp smoke passed on Q4. Perplexity on a 256 KiB wikitext-2 excerpt (ctx 512): bf16 **6.7324**, Q4_K_M **6.7769**. F16 mmproj from `Qwen/Qwen3.8-27B` + Q4 text + cat image → answer **cat**. Private Hub: `unfuse-Qwen3.8-27B-text-GGUF` (Q4 only), `unfuse-Qwen3.8-27B-mmproj-GGUF`.

**vLLM startup:** Default `max_num_seqs=1024` fails on this hybrid (Mamba cache blocks 334 parent / 362 split). Recorded in `bench/sweeps/baseline-512-vllm-refusal.json`. All successful vLLM serves used **`--max-num-seqs 256`**.

**Baseline 512/256 (warm, c=16):** vLLM aggregate **~360 tok/s** for both parent and split (no 1.15× split slowdown at c=16 with capped seqs). SGLang **332 tok/s** split vs **135 tok/s** parent — parent `concurrency_limit=5`, split **16** (`serve-sglang-*.json`).

**Long context 4096/256 (CTX=8192, warm c=16):** vLLM **158 tok/s** split vs **156 tok/s** parent (split slightly ahead). KV budget from logs: parent **223,524** tokens, split **227,035**. SGLang still scheduler-limited on parent (5 vs 16).

**Concurrency c=32 (pinned on both engines):** vLLM **646** vs **643 tok/s** (parent vs split). SGLang **346** vs **136 tok/s** with limits pinned to 32 — parent still tops out near **135 tok/s** aggregate.

**Not run:** Full GGUF quant ladder, full wikitext-2 perplexity, RoPE logit-diff table, 8k/32k bench rows, Task 3 Mamba block analysis, public Hub.

## Phase A commands (on GPU)

```bash
bash tools/run_phase_a_minimal.sh   # after bf16 on disk; UPLOAD=0 default
bash tools/run_perplexity_qwen.sh # parser fix rerun
bash tools/run_mmproj_qwen.sh
```

## Phase B commands

```bash
VLLM_MAX_SEQS=256 bash tools/run_phase_b_minimal.sh
```

Artifacts under `notebooks/qwen3.8-27b/bench/sweeps/{baseline-512,longctx-in4096,conc-c32}/`.

## Failures (first attempts)

- vLLM at max_num_seqs 1024: refused (see refusal JSON).
- Overlapping Phase B processes: SGLang killed mid-start; resolved with `flock` on `phase_b.lock` and a single rerun.
