---
license: mit
base_model: Qwen/Qwen3.8-27B
base_model_relation: adapter
pipeline_tag: image-text-to-text
tags:
  - gguf
  - mmproj
  - qwen
  - llama.cpp
  - multimodal
private: true
---

# Qwen3.8-27B mmproj GGUF (draft)

F16 vision projector for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), exported with `convert_hf_to_gguf.py --mmproj`. Use with a **text** GGUF from [`sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF), not the full multimodal weights repo.

| File | Size (approx.) | Notes |
|---|---|---|
| `mmproj-Qwen3.8-27B-F16.gguf` | ~885 MB | 334 tensors, f16 |

## Usage

```bash
llama-mtmd-cli \
  -m Qwen3.8-27B-text-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-F16.gguf \
  --image photo.jpg \
  -p "What is in this image?" -n 64 --temp 0 -ngl 99
```

(`llama-mtmd-cli` does not support `-st`; use `--temp 0` for greedy decoding.)

## Verification so far

Recombined Q4 text GGUF + this mmproj on a sample cat image; greedy answer **cat**. Full benchmark and quality notes will be added when serving sweeps complete.

## Provenance

llama.cpp `d02b4c0d66796e0a44380611d038123fefb9fe83`. Private staging repo; card will be updated after Phase B results.
