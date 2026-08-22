---
license: mit
base_model: zai-org/GLM-4.6V-Flash
base_model_relation: finetune
pipeline_tag: text-generation
library_name: transformers
language:
  - en
  - zh
tags:
  - glm
  - glm4
  - text-generation
  - vision-removed
  - pruning
  - gguf
---

# GLM-4.6-Flash-text

`zai-org/GLM-4.6V-Flash` with the vision stack surgically removed. **No fine-tuning, no distillation, no retraining** — the 181 vision tensors were deleted and the remaining 523 were re-keyed onto the `Glm4ForCausalLM` architecture.

The text path is **bit-identical** to the original. Verified, not assumed — see [Verification](#verification).

|  | Original | This model |
|---|---|---|
| Architecture | `Glm4vForConditionalGeneration` | `Glm4ForCausalLM` |
| Parameters | 10,292,777,472 | **9,400,279,040** |
| Tensors | 704 | **523** |
| Size (bf16) | 20.59 GB | **18.80 GB** |
| Accepts images | yes | no |

**Removed: 892,498,432 params — 8.671% of the model, 1.785 GB.**

## Architecture

![Forward path and the cut](architecture.svg)

GLM-4.6V-Flash is a classic bolted-on vision tower: a 24-layer ViT feeding soft tokens into the decoder's embedding stream via `masked_scatter`. Because text tokens never touch a vision weight, deleting the branch removes exactly one edge from the graph and leaves the text computation untouched.

| Vision component | Params |
|---|---|
| `visual.blocks.0–23` (ViT, 1536d, 12 heads) | 679,550,976 |
| `visual.merger` (proj + gate/up/down + norm) | 185,081,856 |
| `visual.downsample` (Conv2d, spatial_merge 2) | 25,169,920 |
| `visual.patch_embed.proj` (Conv3d 14×14×2) | 1,807,872 |
| `visual.embeddings.position_embedding` (576 × 1536) | 884,736 |
| `visual.post_conv_layernorm`, `visual.post_layernorm` | 3,072 |
| **Total deleted** | **892,498,432** |

What remains is a standard GLM-4 decoder: 40 layers, 4096 hidden, 13696 intermediate, 32 attention heads with 2 KV heads (GQA 16:1), 151552 vocab, 131072 max positions, untied `lm_head`.

## Verification

The one real risk in this conversion was **RoPE**. GLM-4.6V-Flash uses multimodal RoPE (`mrope_section: [8, 12, 12]`, summing to 32 = `head_dim 128 × partial_rotary_factor 0.5 ÷ 2`), which splits the rotary budget across temporal, height and width axes. For text-only input all three axes carry the same position index, so mRoPE *should* collapse to standard RoPE — but that is an argument, not a measurement.

Logits were compared against the original model on identical text input:

```
max|d|=0.000e+00  argmax_match=True  'The capital of France is'
max|d|=0.000e+00  argmax_match=True  'def fibonacci(n):'
max|d|=0.000e+00  argmax_match=True  'Explain why the sky appears blue, in one sentence.'
max|d|=0.000e+00  argmax_match=True  '1, 1, 2, 3, 5, 8, 13,'
max|d|=0.000e+00  argmax_match=True  'Translate to German: The weather is cold today.'
max|d|=0.000e+00  argmax_match=True  'The three laws of thermodynamics state that'

EXACT MATCH -- mRoPE collapsed to RoPE cleanly. Extraction is lossless.
```

Short prompts do not exercise RoPE at depth, where a position-encoding bug would actually surface. So the same check was run at length:

```
sequence length: 1207
long-context max|d| = 0.000e+00
PASS
```

**Zero divergence, bit-exact, at 1,207 tokens.** On text, this model is the original model.

## Files

Everything lives in one repo. bf16 safetensors at the root, quantizations under `gguf/`.

| File | Format | Size | Notes |
|---|---|---|---|
| `model-0000{1..4}.safetensors` | bf16 | 18.80 GB | reference weights, bit-exact |
| `gguf/GLM-4.6-Flash-text-F16.gguf` | F16 | 18.81 GB | lossless GGUF; requantize from this |
| `gguf/GLM-4.6-Flash-text-Q8_0.gguf` | Q8_0 | 10.00 GB | near-lossless |
| `gguf/GLM-4.6-Flash-text-Q6_K.gguf` | Q6_K | 8.27 GB | very high quality |
| `gguf/GLM-4.6-Flash-text-Q5_K_M.gguf` | Q5_K_M | 7.05 GB | high quality |
| `gguf/GLM-4.6-Flash-text-Q4_K_M.gguf` | Q4_K_M | 6.17 GB | **recommended** — best size/quality tradeoff |

All five load and generate coherently under llama.cpp (`architecture: glm4`, 131072 context). Sizes are GB (10⁹ bytes) as the Hub reports them; `ls -h` will show smaller GiB numbers for the same files.

> **Prefer GGUF?** The same quantizations are published at the repo root of
> **[sartajbhuvaji/GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF)**,
> where the Hub's quantization picker renders and `llama-cli -hf` / `ollama run hf.co/…`
> resolve directly. The `gguf/` copies here are identical — use whichever is convenient.

There is **no NVFP4 build.** NVFP4 is NVIDIA's Blackwell format (E2M1, 16-element blocks, FP8 E4M3 block scales) and requires SM100+ hardware to quantize and serve — it is not a GGUF quant and cannot be produced on Ampere. If you want it, use [LLM Compressor](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w4a4_fp4/) on a B200 with the bf16 weights here as input.

## Usage

### transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "sartajbhuvaji/GLM-4.6-Flash-text", dtype=torch.bfloat16, device_map="auto"
)
tok = AutoTokenizer.from_pretrained("sartajbhuvaji/GLM-4.6-Flash-text")

msgs = [{"role": "user", "content": "What is a mixture-of-experts layer?"}]
enc = tok.apply_chat_template(
    msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
out = model.generate(**enc, max_new_tokens=600)
print(tok.decode(out[0][enc["input_ids"].shape[1] :], skip_special_tokens=True))
```

Requires a transformers version with `Glm4ForCausalLM` (v4.52+).

### llama.cpp

Use the [GGUF repo](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF), where
the files sit at the root and `-hf` resolves them:

```bash
llama-cli -hf sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M \
  -p "Explain gradient descent" -n 400 -st
```

`-st` (`--single-turn`) matters for scripted use. Without it `llama-cli` drops into
interactive mode and waits on stdin. The older `-no-cnv` flag has been removed.

To use the copies in this repo instead, download by explicit path — `-hf` only
resolves root-level GGUFs:

```bash
hf download sartajbhuvaji/GLM-4.6-Flash-text \
  gguf/GLM-4.6-Flash-text-Q4_K_M.gguf --local-dir .
llama-cli -m gguf/GLM-4.6-Flash-text-Q4_K_M.gguf -p "Explain gradient descent" -n 400 -st
```

### Ollama

```bash
ollama run hf.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF:Q4_K_M
```

## Reproducing this

```python
from safetensors.torch import load_file, save_file

# 1. drop every tensor under model.visual.  (181 tensors, 892,498,432 params)
# 2. rename  model.language_model.*  ->  model.*
# 3. keep lm_head.weight -- it is UNTIED (tie_word_embeddings: false)
sd = {
    k.replace("model.language_model.", "model.", 1): v
    for k, v in load_file(shard).items()
    if not k.startswith("model.visual.")
}
```

Then rebuild the config through `Glm4Config`, dropping `mrope_section` and `vision_config`, and set `architectures = ["Glm4ForCausalLM"]` — `Glm4Config` does not populate that field, and without it `AutoModelForCausalLM` has nothing to dispatch to.

Vocabulary needs no work: `vocab_size` stays 151552, and the image/video token ids (151363/151364) remain in it. They are simply never emitted.

## Limitations

- **No vision.** Passing images does nothing; the tokens have no embedder behind them. Use the [original model](https://huggingface.co/zai-org/GLM-4.6V-Flash) if you need multimodal.
- **Reasoning model.** It emits `<think>` blocks before answering, sometimes in Chinese regardless of prompt language. Budget `max_new_tokens` accordingly — 160 is not enough to get past the reasoning to an answer.
- **Inherits everything else** from GLM-4.6V-Flash, including its biases and knowledge cutoff. Text behaviour is bit-identical, so any evaluation of the original's text ability transfers exactly.
- **Quantization is not verified bit-exact.** The bit-exactness result above applies to the bf16 weights only. The GGUF quants are lossy by construction; each was checked to load and generate coherent text under llama.cpp, but no perplexity or benchmark comparison against bf16 was run. If you need a measured quality delta, compute it yourself.

## Provenance

Derived from [zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (MIT). This model is MIT as well.

Conversion and verification were run on a single A100-SXM4-40GB with transformers 5.16.0.dev0 and torch 2.7.0. GGUF builds used llama.cpp at master with the `GLM4` architecture.
