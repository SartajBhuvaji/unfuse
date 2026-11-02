---
license: apache-2.0
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
  - unfuse
---

# unfuse-Qwen3.8-27B-text-GGUF

GGUF quantization of [`sartajbhuvaji/unfuse-Qwen3.8-27B-text`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text),
which is [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) with its vision tower
removed. Part of the [unfuse collection](https://huggingface.co/collections/sartajbhuvaji/unfuse-6ae8b265e63951518e4e75cc).

| File | Quant | Size |
|---|---|---|
| `Qwen3.8-27B-text-Q8_0.gguf` | Q8_0 | 29.1 GB |
| `Qwen3.8-27B-text-Q5_K_M.gguf` | Q5_K_M | 19.5 GB |
| `Qwen3.8-27B-text-Q4_K_M.gguf` | Q4_K_M | 16.8 GB |

All three were cut from a bf16 GGUF (not f16) with llama.cpp commit
`d02b4c0d66796e0a44380611d038123fefb9fe83`. The architecture is `qwen35`, 866 tensors,
including the multi-token-prediction block as block 64.

## Usage

```bash
llama-cli -m Qwen3.8-27B-text-Q4_K_M.gguf -p "Your prompt" -n 512 --temp 0 -st -ngl 99
```

`-st` (`--single-turn`) stops `llama-cli` from waiting on stdin, which matters in scripts.

For images, pair it with the projector in
[`unfuse-Qwen3.8-27B-mmproj-GGUF`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF)
and use `llama-mtmd-cli`.

## What was checked

- **Perplexity.** Full wikitext-2 test split, context 512, llama-perplexity:

  | | PPL | vs bf16 |
  |---|---|---|
  | BF16 | 6.6497 | |
  | Q8_0 | 6.6514 | +0.03% |
  | Q5_K_M | 6.6746 | +0.37% |
  | Q4_K_M | 6.6959 | +0.69% |

- **RoPE.** The converter prints `Unknown RoPE type: default`. It does not matter: the
  safetensors model scores 6.6516 on the same text with the same chunking, against 6.6497
  for the bf16 GGUF, and the top-1 token matches on seven prompts including a 1,259-token
  one.
- **Known answers.** Prompts with checkable answers (capital of France, `84 * 3 / 2`, a
  German translation) passed on Q4_K_M, and on Q8_0 and Q5_K_M.
- **With the projector.** One image (a cat) with Q4_K_M and the mmproj; the answer was
  "cat".

## What was not checked

Serving throughput of this GGUF, a Q6_K, and vision quality with the projector. The
parent-versus-split benchmark runs are in the
[unfuse repo](https://github.com/SartajBhuvaji/unfuse) under `notebooks/qwen3.8-27b/`.

## Provenance

Weights come from `Qwen/Qwen3.8-27B` (Apache 2.0). The text-only split is byte-exact on
every tensor it keeps; see the unfuse repo for how that was verified.
