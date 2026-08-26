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

NVFP4 is a 4-bit weight and activation format, but NVIDIA's build excludes 63 modules.
and **60 of them are the attention blocks**, listed one layer at a time. Only the MLPs were
quantized, which left unquantized BF16 attention as the largest single block in the file:

| Component | Params | In NVIDIA's build | Here |
|---|---|---|---|
| MLPs (60 layers) | 20.8B | NVFP4, group 16 | **unchanged** |
| Attention (60 layers) | 7.93B | BF16, 15.85 GB | **FP8, 7.93 GB** |
| Embeddings (tied) | 1.41B | BF16 | unchanged |
| Vision tower (SigLIP) | ~576M | BF16 | **removed** |

Leaving attention in BF16 is defensible. NVFP4 quantizes activations too, and attention
activations are the outlier-heavy ones. It is also a deviation from NVIDIA's own practice:
their DeepSeek-V4 and Qwen3-Omni builds ship FP4 experts alongside **FP8** attention.

## What was quantized, exactly

**Weights only.** FP8 has two independent halves with very different risk profiles. The
weight half is a pure function of the weights, so there is no calibration set and nothing to get wrong
beyond the scale arithmetic. The activation half needs calibration and is precisely where
attention's outliers cause trouble.

Weights-only still takes the full memory win and still speeds up batch-1 decode, which is
memory-bandwidth-bound. It gives up a prefill win, since BF16 activations cannot feed FP8
tensor cores.

- **230 projections**, per-tensor scale, `float8_e4m3`. Not 240: Gemma 4 sets
  `attention_k_eq_v`, so 10 layers share K and V and have no separate `v_proj`.
- `q_norm` and `k_norm` are deliberately untouched. They are 1-D RMSNorm scales, so no
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

The NVFP4 two-level scale reconstructs the original range exactly.
`weight_scale.max() × weight_scale_2 × 6 = 0.261719`, against an original absmax of
`0.261719`, where 6.0 is the largest e2m1 magnitude.

Scored through the chat template against BF16 `google/gemma-4-31B-it`:

| | worst top-1 | worst KL |
|---|---|---|
| NVIDIA NVFP4 | 0.7826 | 1.448e-01 |
| **this model** | 0.9130 | 1.716e-01 |

This adds **1.19×** NVIDIA's KL divergence. A strict "no worse than the shipped build" gate
is the wrong test for a change that adds quantization, since it cannot pass by construction,
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
Measured as tokens per target forward pass, which is the arithmetic ceiling on speedup:

| target | tok/fwd | vs BF16 |
|---|---|---|
| BF16 base | 2.220 | — |
| NVIDIA NVFP4 | 1.655 | −25.4% |
| **this model** | 1.247 | −43.8% |

Monotonic in the amount of quantization. Speculative decoding stays exact, so output
quality is unchanged, but a published "up to 3×" for the BF16 pairing does not transfer to
a quantized target. NVIDIA's own build surrenders a quarter of the benefit before anyone
adds anything.

Measured via transformers' `assistant_model` path rather than the native MTP path, on four
prompts. Directionally clear, not a precise coefficient.

## Limitations

- **Text only.** The vision tower is gone. Use the base model for images.
- **NVFP4 does not require Blackwell under vLLM.** It selects `MarlinNvFp4LinearKernel`,
  and `ModelOptMixedPrecisionConfig.get_min_capability()` is 75. Its comment records
  validation on a T4 (SM75) and an A100 (SM80). What Blackwell buys is native FP4 tensor
  cores; without them Marlin repacks to W4A16 and the *stored* size saving is not a
  *runtime* memory saving. A 40 GB A100 ran out of memory loading this 23 GB checkpoint.
- **FP8 KV cache needs SM89+.** On older cards pass `--kv-cache-dtype bfloat16`, or vLLM
  refuses with `FP8 KV cache is not supported by the Triton attention backend`.
- transformers cannot load this checkpoint. `quant_method: modelopt` is unsupported,
  and rather than erroring it ignores the config, loads packed `uint8` as dense, and fails
  on a shape mismatch of exactly 2x the input dimension. Expand to BF16 with
  [`tools/dequantize.py`](https://github.com/SartajBhuvaji/unfuse/blob/main/tools/dequantize.py)
  and load that.
- Accuracy was verified on BF16-expanded copies. That is exact for weight-only
  quantization, but it does not simulate NVFP4's activation quantization. Both sides of
  the comparison are equally optimistic.
- The KV cache at the full 262k window is ~11 GB, roughly half these weights.

## Usage

**The quantized weights do not serve correctly on vLLM today.** Use the BF16 expansion
below, which is tested and gives correct output. Why the quantized form fails is documented
under *Serving the quantized weights* further down.

### What works today (tested)

Verified 2026-08-22 on an H100 PCIe 80 GB against the current vLLM release:

```bash
pip install vllm==0.27.1 'transformers==5.14.1'

git clone https://github.com/SartajBhuvaji/unfuse && cd unfuse
hf download sartajbhuvaji/Gemma-4-31B-IT-NVFP4-FP8attn-text --local-dir ./ckpt
python tools/dequantize.py ./ckpt ./ckpt-bf16

vllm serve ./ckpt-bf16
```

Engine up in 103 s, 36.5 tok/s on a batch of 5 (eager). Answers checked: `Paris`; `408`
with correct working; `Das Wetter ist heute kalt.`; Rayleigh scattering;
`my_string[::-1]`. A vLLM nightly serves the same expansion at 42 s and 48.8 tok/s.

Expansion gives up the size saving: 23 GB becomes 61 GB, so this wants an 80 GB card.
It is how you *run* the model today, not how you store it.

**The transformers pin is required, and has nothing to do with quantization.**
transformers 5.15.0 made `head_dim` a per-layer attribute and raises
`AmbiguousGlobalPerLayerAttributeError` on a global read; vLLM reads it globally. 5.14.1
is the last version that works. This affects *any* Gemma 4 31B on vLLM, Google's and
NVIDIA's included. Measured: 5.5.3, 5.8.0, 5.11.0, 5.13.1, 5.14.0 and 5.14.1 all return
`head_dim = 256`; 5.15.0 and 5.15.1 raise.

### With transformers directly

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("./ckpt")
model = AutoModelForCausalLM.from_pretrained("./ckpt-bf16", dtype="bfloat16", device_map="cuda")
msgs = [{"role": "user", "content": "What is the capital of France? Answer in one word."}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
print(
    tok.decode(
        model.generate(**tok(text, return_tensors="pt").to("cuda"), max_new_tokens=8)[0],
        skip_special_tokens=True,
    )
)
```

### Serving the quantized weights

Three problems, each found behind the last. The first two are defects in this
checkpoint's config and are fixable. The third is not, and is the reason the quantized
form cannot currently be served correctly.

1. **`quant_algo` must be `MIXED_PRECISION`.** vLLM routes to its mixed-precision path
   only on that exact string, in
   `ModelOptMixedPrecisionConfig.override_quantization_method`. This checkpoint says
   `NVFP4`, so vLLM applies NVFP4 to everything including the FP8 attention, and the
   fused QKV loader asserts: `param=(16384, 2688)` against `loaded=(4096, 5376)`, where
   `2688 = 5376/2` is the FP4 packing.

   **This is left as `NVFP4` on purpose.** Setting it to `MIXED_PRECISION` does not make
   the model work. It makes problem 3 happen instead, which means silent nonsense rather
   than a loud assertion. Between two broken states, the one that reports itself is the
   better one to ship.

2. **`quantized_layers` keys must be exact module prefixes.** vLLM's lookup,
   `_quantized_layer_prefix_candidates`, is exact dict membership with no glob expansion.
   The original keys were `model.layers.N.self_attn*`, which match nothing. Fixed:
   the config now names each projection, `model.layers.0.self_attn.q_proj` and so on, 230
   FP8 entries alongside 180 NVFP4 MLP entries. The mapping is now accurate metadata, and
   correct for whatever reads it next; it does not by itself make the model servable.

3. **vLLM's ModelOpt FP8 path is W8A8-only; this model is W8A16.**
   `ModelOptFp8LinearMethod.create_weights` registers `input_scale` as a required
   per-tensor parameter, and `process_weights_after_loading` reads
   `layer.input_scale.max()`. Attention here is FP8 weights with BF16 activations, so
   there is no `input_scale` to load. With (1) and (2) patched the model loads and
   generates, since FP8 and BF16 weights have identical shapes, but the scales sit at their
   initialised sentinels and the output is nonsense.

Serving the quantized form therefore needs either a calibrated W8A8 attention variant, or
FP8 weight-only support in vLLM's ModelOpt path. Either way it needs vLLM newer than
0.27.1: the released 0.27.1 has no `quantized_layers` handling at all, so mixed NVFP4+FP8
cannot be expressed to it.

For reference, `nvidia/Gemma-4-31B-IT-NVFP4` **does** serve on stock vLLM 0.27.1 with the
transformers pin. Its attention sits in `exclude_modules`, which 0.27.1 does understand.

## Credit

Base weights © Google DeepMind. NVFP4 quantization by NVIDIA. This build only removes the
vision tower and requantizes attention.
