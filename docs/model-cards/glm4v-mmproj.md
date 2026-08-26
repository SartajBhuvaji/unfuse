---
license: mit
base_model: zai-org/GLM-4.6V-Flash
base_model_relation: quantized
language:
  - en
  - zh
tags:
  - gguf
  - mmproj
  - clip
  - glm
  - glm4v
  - llama.cpp
  - multimodal
---

# GLM-4.6V-Flash vision tower (mmproj GGUF)

The vision half of `zai-org/GLM-4.6V-Flash`, packaged as a llama.cpp multimodal
projector. This is the counterpart to
[sartajbhuvaji/GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF),
which is the same model with vision removed.

892,498,432 params, 24-layer ViT + merger, `clip.projector_type: glm4v`.

| File | Format | Size | Notes |
|---|---|---|---|
| `mmproj-GLM-4.6V-Flash-F32.gguf` | F32 | 3.57 GB | reference precision |
| `mmproj-GLM-4.6V-Flash-BF16.gguf` | BF16 | 1.84 GB | matches source dtype |
| `mmproj-GLM-4.6V-Flash-F16.gguf` | F16 | 1.79 GB | recommended |
| `mmproj-GLM-4.6V-Flash-Q8_0.gguf` | Q8_0 | 1.03 GB | smallest available |

## Why there is no Q4_K_M

Vision projectors are not quantized the way language models are. `llama-quantize`
and the K-quant ladder (`Q4_K_M`, `Q5_K_M`, `Q6_K`, …) apply to the **text** GGUF;
mmproj files are produced directly by `convert_hf_to_gguf.py --mmproj`, whose only
output types are `f32`, `bf16`, `f16`, and `q8_0`.

This matches what other publishers ship. unsloth's Gemma 3 and Gemma 4 mmproj files
are BF16/F16/F32 only, and ggml-org's Qwen2.5-VL mmproj is Q8_0/F16. Vision encoders
are more sensitive to low-bit quantization than decoders and the tooling reflects
that. Q8_0 at 1.03 GB is as small as this gets.

## Usage

An mmproj file does nothing on its own. It has to be paired with a text GGUF.

```bash
llama-mtmd-cli \
  -m GLM-4.6-Flash-text-Q4_K_M.gguf \
  --mmproj mmproj-GLM-4.6V-Flash-F16.gguf \
  --image photo.jpg \
  -p "Describe this image." -n 400 -st
```

Get the text half from
[sartajbhuvaji/GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF):

```bash
hf download sartajbhuvaji/GLM-4.6-Flash-text-GGUF \
  GLM-4.6-Flash-text-Q4_K_M.gguf --local-dir .
hf download sartajbhuvaji/GLM-4.6V-Flash-mmproj-GGUF \
  mmproj-GLM-4.6V-Flash-F16.gguf --local-dir .
```

`-st` (`--single-turn`) matters for scripted use. Without it llama.cpp drops into
interactive mode and waits on stdin.

### Pairing

The text repo is the vision-removed model (`Glm4ForCausalLM`, arch `glm4`).
llama.cpp has no separate `GLM4V` model architecture, only `CHATGLM`, `GLM4` and
`GLM4_MOE`, with `glm4v` existing as a projector type. So GLM-4V's text tower and
this text-only extraction both convert to arch `glm4`, and the metadata lines up.

This pairing was verified from GGUF metadata, not by running end-to-end multimodal
inference. The build machine had 7 GiB of RAM, and Q4_K_M (6.17 GB) plus a projector
does not fit. If you hit a mismatch, pair against a text GGUF converted directly from
`zai-org/GLM-4.6V-Flash` instead, and please open a discussion so this note can be
corrected.

## Contents

Derived from the 181 tensors under `model.visual.*` in the source checkpoint,
exported as 182 GGUF tensors (`v.*` for the ViT, `mm.*` for the merger).

| Component | Params |
|---|---|
| 24 × ViT block (1536d, 12 heads) | 679,550,976 |
| merger (proj + gate/up/down + norm) | 185,081,856 |
| downsample (Conv2d, spatial_merge 2) | 25,169,920 |
| patch_embed (Conv3d 14×14×2) | 1,807,872 |
| position embeddings (576 × 1536) | 884,736 |
| layernorms | 3,072 |
| **Total** | **892,498,432** |

Config: `image_size 336`, `patch_size 14`, `projection_dim 4096`, `block_count 24`,
`head_count 12`, `feed_forward_length 13696`, SiLU activation.

## Provenance

Exported with `convert_hf_to_gguf.py --mmproj` from llama.cpp at master. Derived from
[zai-org/GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) (MIT); this
repo is MIT.

Related: [GLM-4.6-Flash-text](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text)
(bf16 text-only) · [GLM-4.6-Flash-text-GGUF](https://huggingface.co/sartajbhuvaji/GLM-4.6-Flash-text-GGUF)
(text quants).
