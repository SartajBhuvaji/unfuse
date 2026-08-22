---
license: apache-2.0
base_model: nvidia/Gemma-4-31B-IT-NVFP4
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
  - nvfp4
  - fp8
  - modelopt
  - quantized
  - text-only
  - gemma4
language:
  - en
---

# Gemma-4-31B-IT-NVFP4-FP8attn-text

**23.00 GB, down from 32.63 GB (−29.5%).** NVIDIA's NVFP4 MLPs copied verbatim, all 230
attention projections requantized to FP8, and the vision tower removed.

![architecture](architecture.svg)

![composition](composition.svg)

| | Size | What changed |
|---|---|---|
| [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | 32.63 GB | starting point |
| + vision tower removed | 31.48 GB | −3.5% |
| + FP8 attention weights | **23.00 GB** | **−29.5%** |

Built with [`unfuse`](https://github.com/SartajBhuvaji/unfuse). The full notebook, with
committed outputs for every number below, is at
[`notebooks/gemma-4-31b-nvfp4/`](https://github.com/SartajBhuvaji/unfuse/tree/main/notebooks/gemma-4-31b-nvfp4).

## Why this exists

NVFP4 is a 4-bit weight *and* activation format, but NVIDIA's build excludes 63 modules —
and **60 of them are the attention blocks**, listed one layer at a time. Only the MLPs were
quantized, which left unquantized BF16 attention as the largest single block in the file:

| Component | Params | In NVIDIA's build | Here |
|---|---|---|---|
| MLPs (60 layers) | 20.8B | NVFP4, group 16 | **unchanged** |
| Attention (60 layers) | 7.93B | BF16, 15.85 GB | **FP8, 7.93 GB** |
| Embeddings (tied) | 1.41B | BF16 | unchanged |
| Vision tower (SigLIP) | ~576M | BF16 | **removed** |

Leaving attention in BF16 is defensible — NVFP4 quantizes activations too, and attention
activations are the outlier-heavy ones. It is also a deviation from NVIDIA's own practice:
their DeepSeek-V4 and Qwen3-Omni builds ship FP4 experts alongside **FP8** attention.

## What was quantized, exactly

**Weights only.** FP8 has two independent halves with very different risk profiles. The
weight half is a pure function of the weights — no calibration set, nothing to get wrong
beyond the scale arithmetic. The activation half needs calibration and is precisely where
attention's outliers cause trouble.

Weights-only still takes the full memory win and still speeds up batch-1 decode, which is
memory-bandwidth-bound. It gives up a prefill win, since BF16 activations cannot feed FP8
tensor cores.

- **230 projections**, per-tensor scale, `float8_e4m3`. Not 240: Gemma 4 sets
  `attention_k_eq_v`, so 10 layers share K and V and have no separate `v_proj`.
- **`q_norm` and `k_norm` are deliberately untouched.** They are 1-D RMSNorm scales — no
  size to win and among the most precision-sensitive weights in the model.
- MLP weights are byte-identical to NVIDIA's, so any accuracy difference is attributable to
  the attention change alone.

## Verification

Attention was never quantized by NVIDIA, so after extraction it must be **bit-identical**
to the BF16 original. It is:

```
model.layers.{0,30,59}.self_attn.q_proj.weight    max|d| = 0.0000
model.layers.*.mlp.gate_proj.weight               rel    = 4-10%   (correct for e2m1)
```

The NVFP4 two-level scale reconstructs the original range exactly —
`weight_scale.max() × weight_scale_2 × 6 = 0.261719`, against an original absmax of
`0.261719`, where 6.0 is the largest e2m1 magnitude.

Scored through the chat template against BF16 `google/gemma-4-31B-it`:

| | worst top-1 | worst KL |
|---|---|---|
| NVIDIA NVFP4 | 0.7826 | 1.448e-01 |
| **this model** | 0.9130 | 1.716e-01 |

This adds **1.19×** NVIDIA's KL divergence. A strict "no worse than the shipped build" gate
is the wrong test for a change that *adds* quantization — it cannot pass by construction —
so the increment is what is reported. Six prompts is a small sample and the top-1 column is
not cleanly ordered between the two builds; treat both rows as indicative.

Do not read KL as benchmark accuracy. NVIDIA published GPQA 85.35% vs 85.80% for their
build, which *is* near-lossless. KL over a 262k vocabulary is far more sensitive than
multiple-choice accuracy.

### Generation quality

Against the BF16 reference on chat prompts: 17 × 24 = **408** with the same two methods, a
**token-identical** German translation, and equivalent correct answers elsewhere.

## Speculative decoding costs more than you would expect

`google/gemma-4-31B-it-assistant` (0.47B) was trained against the **BF16** target.
Quantizing the target moves its distribution away from what the drafter learned to predict.
Measured as tokens per target forward pass — the arithmetic ceiling on speedup:

| target | tok/fwd | vs BF16 |
|---|---|---|
| BF16 base | 2.220 | — |
| NVIDIA NVFP4 | 1.655 | −25.4% |
| **this model** | 1.247 | −43.8% |

Monotonic in the amount of quantization. Speculative decoding stays **exact** — output
quality is unchanged — but a published "up to 3×" for the BF16 pairing does not transfer to
a quantized target. NVIDIA's own build surrenders a quarter of the benefit before anyone
adds anything.

Measured via transformers' `assistant_model` path rather than the native MTP path, on four
prompts. Directionally clear, not a precise coefficient.

## Limitations

- **Text only.** The vision tower is gone. Use the base model for images.
- **NVFP4 executes only on Blackwell** (sm_100+). On older hardware the MLPs cannot run
  natively.
- **transformers cannot load this checkpoint** — `quant_method: modelopt` is unsupported,
  and rather than erroring it ignores the config, loads packed `uint8` as dense, and fails
  on a shape mismatch. Serve with vLLM, or expand to BF16 with
  [`tools/dequantize.py`](https://github.com/SartajBhuvaji/unfuse/blob/main/tools/dequantize.py).
- Accuracy was verified on BF16-expanded copies. That is exact for weight-only
  quantization, but it does not simulate NVFP4's activation quantization — both sides of
  the comparison are equally optimistic.
- The KV cache at the full 262k window is ~11 GB, roughly half these weights.

## Usage — read this before trying to serve it

**vLLM 0.27.1 cannot load Gemma 4 31B at all.** Not this checkpoint specifically — *any*
Gemma 4 31B. It reads `config.head_dim` globally, and Gemma 4's layers are heterogeneous,
so transformers raises:

```
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute
```

Reproduced identically against `google/gemma-4-31B-it` and
`nvidia/Gemma-4-31B-IT-NVFP4`, so it is an upstream architecture-support gap, independent
of quantization. When vLLM adds Gemma 4 support, `--quantization modelopt` is the expected
path:

```bash
# not yet functional as of vLLM 0.27.1
vllm serve sartajbhuvaji/Gemma-4-31B-IT-NVFP4-FP8attn-text --quantization modelopt
```

**transformers cannot load it either**, for a different reason — `quant_method: modelopt`
is unsupported, so it ignores the quantization config, loads the packed `uint8` tensors as
dense weights, and fails on a 2× shape mismatch.

### What does work today

Expand the weights back to BF16 and run that. For weight-only quantization this is exact,
not an approximation — an FP4 or FP8 kernel dequantizes before multiplying against BF16
activations anyway:

```bash
git clone https://github.com/SartajBhuvaji/unfuse && cd unfuse
hf download sartajbhuvaji/Gemma-4-31B-IT-NVFP4-FP8attn-text --local-dir ./ckpt
python tools/dequantize.py ./ckpt ./ckpt-bf16
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("./ckpt")
model = AutoModelForCausalLM.from_pretrained("./ckpt-bf16", dtype="bfloat16", device_map="cuda")
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(tok.decode(model.generate(**tok(text, return_tensors="pt").to("cuda"),
                                max_new_tokens=8)[0], skip_special_tokens=True))
```

This round trip is verified: downloaded fresh from this repo on a clean machine, expanded,
and asked five questions — `Paris`, `408` with correct working, a correct German
translation, a correct Python one-liner, and Rayleigh scattering.

Note that expanding gives up the size saving; it is how you *run* the model today, not how
you store it.

## Credit

Base weights © Google DeepMind. NVFP4 quantization by NVIDIA. This build only removes the
vision tower and requantizes attention.
