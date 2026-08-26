---
license: mit
base_model: sartajbhuvaji/GLM-4.6-Flash-text
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - gguf
  - glm
  - glm4
  - llama.cpp
  - quantized
  - vision-removed
---

# GLM-4.6-Flash-text GGUF

GGUF quantizations of [`sartajbhuvaji/GLM-4.6-Flash-text`](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text), which is `zai-org/GLM-4.6V-Flash` with its 892M-parameter vision stack removed.

9,400,279,040 params, `glm4` architecture, 131072 context, text-only.

| File | Bits | Size | Notes |
|---|---|---|---|
| `GLM-4.6-Flash-text-Q4_K_M.gguf` | 4 | 6.17 GB | recommended, best size/quality tradeoff |
| `GLM-4.6-Flash-text-Q5_K_M.gguf` | 5 | 7.05 GB | high quality |
| `GLM-4.6-Flash-text-Q6_K.gguf` | 6 | 8.27 GB | very high quality |
| `GLM-4.6-Flash-text-Q8_0.gguf` | 8 | 10.00 GB | near-lossless |
| `GLM-4.6-Flash-text-F16.gguf` | 16 | 18.81 GB | lossless, requantize from this |

Sizes are GB (10⁹ bytes) as the Hub reports them. `ls -h` shows smaller GiB numbers for the same files.

## Usage

```bash
llama-cli -hf sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M \
  -p "Explain gradient descent" -n 400 -st
```

```bash
ollama run hf.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M
```

`-st` (`--single-turn`) matters for scripted use. Without it `llama-cli` drops into interactive mode and waits on stdin. The older `-no-cnv` flag has been removed from current llama.cpp.

Budget your token limit. This is a reasoning model: it emits a thinking block before answering, sometimes in Chinese regardless of prompt language. 160 tokens is not enough to get past the reasoning to an answer, so use 600 or more.

## What was removed

The parent model is `Glm4vForConditionalGeneration`: a 24-layer ViT feeding soft tokens into a GLM-4 decoder via `masked_scatter`. Text tokens never touch a vision weight, so deleting the branch leaves the text computation alone.

| | Original | Text-only |
|---|---|---|
| Parameters | 10,292,777,472 | 9,400,279,040 |
| Tensors | 704 | 523 |
| bf16 size | 20.59 GB | 18.80 GB |

892,498,432 params removed, 8.671% of the model.

The extraction was verified bit-exact against the original on text input: `max|d| = 0.000e+00` across six prompts and again at 1,207 tokens. Full detail and the architecture diagram are on the [parent model card](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text).

## Verification of these quants

Each file was checked to load and generate coherent text under llama.cpp (`architecture: glm4` recognised, correct param count and context length). No perplexity or benchmark comparison against bf16 was run. The bit-exactness result above applies to the bf16 weights, not to these lossy quantizations. If you need a measured quality delta, compute it yourself.

## Provenance

Quantized with llama.cpp at master from the bf16 weights in the parent repo. Derived from [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (MIT). This repo is MIT as well.
