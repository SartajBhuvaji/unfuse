# Qwen3.8-27B follow-ups

1x H100 80GB, llama.cpp `d02b4c0d66796e0a44380611d038123fefb9fe83`, vLLM 0.27.1,
SGLang 0.5.18 (not re-run here). Results are in this directory.

## Findings

**RoPE is fine.** Perplexity of the bf16 GGUF (llama-perplexity) and of the safetensors
model (transformers, chunked the same way) on the same wikitext-2 text: 6.6497 vs 6.6516,
a 0.03% gap over 602 chunks. The top-1 token matches on all seven prompts, including a
1,259-token one. The converter's `Unknown RoPE type: default` warning did not change
anything measurable. Details and protocol in `ROPE.md`. The top-20 logprobs differ by
0.13 to 0.41 between llama-server and transformers; that is consistent with bf16 kernel
noise but the exact cause was not isolated.

**Quants hold up.** Full wikitext-2 test at context 512: BF16 6.6497, Q8_0 6.6514,
Q5_K_M 6.6746, Q4_K_M 6.6959. Perplexity rises in step with the quant, and Q4_K_M is
0.7% above bf16. Known-answer smoke tests passed for Q8_0 and Q5_K_M.
Table in `perplexity_full_table.md`.

**The max-num-seqs cap is not why throughput dropped.** At 512 in / 256 out, warm, c=16,
vLLM reads 360 to 402 tok/s for both models at every cap tried (64, 128, 256, 320). The
earlier published 555 to 636 tok/s did not come back, and the parent and split are tied
at every setting. The earlier 1.15x split slowdown did not reproduce. Whatever made that
run different (vLLM version, engine defaults, the box) was not found here.

**Long context shows no throughput gain from the split.** vLLM, CTX 40960, c=16: 8k in
gives 97.0 (parent) vs 95.0 (split) tok/s, 32k gives 26.3 vs 26.4. The split does hold
more KV cache (278,674 vs 283,794 tokens at 8k, 278,674 vs 302,811 at 32k) but that did
not turn into throughput at this concurrency.

**The SGLang parent number is a scheduler cap, not a result.** At c=32 SGLang limits the
parent to 5 running requests because of its Mamba state cache (28 slots, 5 per request),
whatever the CLI flag says; the split gets 16. The 135 vs 346 tok/s gap compares unlike
things. See `task5_sglang_analysis.md`.

## Numbers

### vLLM, 512/256, warm, c=16

| max-num-seqs | parent tok/s | split tok/s | KV tokens (parent / split) |
|---|---|---|---|
| 64 | 402.6 | 398.4 | 284,717 / 289,814 |
| 128 | 367.9 | 359.9 | 281,804 / 286,902 |
| 256 | 381.3 | 377.1 | 277,435 / 301,465 |
| 320 | 379.9 | 380.1 | 280,348 / 300,737 |

At the default of 1024 vLLM refuses to start: the Mamba cache has 334 blocks for the
parent and 362 for the split (`../bench/sweeps/baseline-512-vllm-refusal.json`).

### vLLM, long context, CTX 40960, warm

| input | concurrency | parent tok/s | split tok/s |
|---|---|---|---|
| 8192 | 16 | 97.0 | 95.0 |
| 32768 | 16 | 26.3 | 26.4 |
| 32768 | 1 | 15.2 | 15.2 |

## Not run

- SGLang sweeps (read-only look only)
- What sets the 334 / 362 Mamba block counts
- Why the earlier run read 555 to 636 tok/s
- A Q6_K quant

## Commands

```bash
bash notebooks/qwen3.8-27b/followups/run_task2_resume.sh        # Q8_0, Q5_K_M and perplexity
bash notebooks/qwen3.8-27b/followups/run_task3_vllm_maxseq.sh
bash notebooks/qwen3.8-27b/followups/run_task4_vllm_longctx.sh
bash notebooks/qwen3.8-27b/followups/run_rope_and_upload.sh     # RoPE compare; uploads to the private repo
```

The first attempts at the RoPE check parsed `llama-cli` output and failed; those scripts
and logs were removed once `rope_server_compare.py` replaced them.
