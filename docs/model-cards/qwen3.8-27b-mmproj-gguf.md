---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - gguf
  - mmproj
  - qwen
  - llama.cpp
  - multimodal
  - unfuse
---

# unfuse-Qwen3.8-27B-mmproj-GGUF

The vision projector for [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) as a
GGUF, exported with llama.cpp's `convert_hf_to_gguf.py --mmproj`. It is the encoder half
that [`unfuse-Qwen3.8-27B-text`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text)
removed, so the two recombine under llama.cpp. Part of the
[unfuse collection](https://huggingface.co/collections/sartajbhuvaji/unfuse-6ae8b265e63951518e4e75cc).

| File | Precision | Size |
|---|---|---|
| `mmproj-Qwen3.8-27B-F16.gguf` | F16 | 928 MB (334 tensors) |

Pair it with a text GGUF such as
[`unfuse-Qwen3.8-27B-text-GGUF`](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF).

## Usage

```bash
llama-mtmd-cli \
  -m Qwen3.8-27B-text-Q4_K_M.gguf \
  --mmproj mmproj-Qwen3.8-27B-F16.gguf \
  --image photo.jpg \
  -p "What is in this image?" -n 64 --temp 0 -ngl 99
```

`llama-mtmd-cli` has no `-st` flag; `--temp 0` gives greedy decoding.

## What was checked

One image, a cat, with the Q4_K_M text GGUF and this file. The greedy answer was "cat".
That shows the pieces load and talk to each other. It says nothing about vision quality,
which has not been measured.

## Provenance

Converted from `Qwen/Qwen3.8-27B` (Apache 2.0) with llama.cpp commit
`d02b4c0d66796e0a44380611d038123fefb9fe83`.
