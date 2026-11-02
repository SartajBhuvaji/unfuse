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
| `Qwen3.8-27B-text-Q4_K_M.gguf` | Q4_K_M | 16.8 GB |

Only Q4_K_M so far. It was cut from a bf16 GGUF (not f16) with llama.cpp commit
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

- **Known answers.** Three prompts through llama.cpp, each checked for the right answer
  rather than a clean exit: capital of France, `84 * 3 / 2`, and a German translation.
  All passed.
- **Perplexity.** On the first 256 KiB of the wikitext-2 test split at context 512:
  bf16 6.7324, Q4_K_M 6.7769 (+0.7%). This is a small slice. Use it to compare the two
  rows, not as a benchmark.
- **With the projector.** One image (a cat) with this file and the mmproj; the answer
  was "cat".

## What was not checked

- **RoPE.** The converter prints `Unknown RoPE type: default`. This model uses
  interleaved multimodal RoPE, and nobody has yet compared this GGUF's logits against
  the safetensors model. The outputs above look fine, which is encouraging but is not
  that comparison.
- Other quants, a full-length perplexity run, and serving throughput are not covered
  here. The parent-versus-split benchmark runs are in the
  [unfuse repo](https://github.com/SartajBhuvaji/unfuse) under `notebooks/qwen3.8-27b/`.

## Provenance

Weights come from `Qwen/Qwen3.8-27B` (Apache 2.0). The text-only split is byte-exact on
every tensor it keeps; see the unfuse repo for how that was verified.
