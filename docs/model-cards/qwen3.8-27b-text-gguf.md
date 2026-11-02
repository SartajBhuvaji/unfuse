---
license: mit
base_model: sartajbhuvaji/unfuse-Qwen3.8-27B-text
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - gguf
  - qwen
  - qwen3
  - llama.cpp
  - quantized
  - vision-removed
private: true
---

# Qwen3.8-27B-text GGUF (draft)

GGUF quantizations of [`sartajbhuvaji/unfuse-Qwen3.8-27B-text`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text), the text-only split of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). This repo is **private** while results are being finalized; the card will be updated after serving benchmarks land.

| File | Quant | Size (approx.) | Notes |
|---|---|---|---|
| `Qwen3.8-27B-text-Q4_K_M.gguf` | Q4_K_M | ~16 GB | Primary publish target for llama.cpp |

Conversion used llama.cpp at commit `d02b4c0d66796e0a44380611d038123fefb9fe83`, **bf16** base (`BASE=bf16`), architecture `qwen35` in GGUF (866 tensors incl. MTP block 64).

## Usage

```bash
llama-cli -m Qwen3.8-27B-text-Q4_K_M.gguf -p "Your prompt" -n 512 --temp 0 -st -ngl 99
```

For scripted runs, use `-st` / `--single-turn` so `llama-cli` does not wait on stdin.

Pair with the vision projector in [`sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF) for multimodal (`llama-mtmd-cli`).

## Verification so far

- **Smoke (llama.cpp):** known-answer checks passed on Q4_K_M (capital / arithmetic / translation prompts).
- **Perplexity** (wikitext-2 test excerpt, 256 KiB, ctx 512, H100): BF16 **6.7324**, Q4_K_M **6.7769** — comparative slice only, not full test split.
- **RoPE:** converter logged `Unknown RoPE type: default`; GGUF carries `qwen35.rope.*` / `mrope` metadata. No full logit-diff write-up in this draft card yet.

## Serving benchmarks

vLLM / SGLang parent-vs-split sweeps were **in progress** when this card was uploaded. Throughput, KV capacity, and concurrency rows will be added here when Phase B finishes.

## Provenance

Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Text-only safetensors and verification live on the parent unfuse repo. MIT.
