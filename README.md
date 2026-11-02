# unfuse

Split a multimodal model into its parts, losslessly.

A model that reads images is two things glued together: an encoder that turns pixels into
vectors, and an ordinary language model that consumes them. `unfuse` takes them apart:
keep the language model, keep the encoder, or keep both separately. It is a tensor filter
and a rename. No retraining, no distillation.

The split is easy. The work is proving the result is unchanged, so every extraction here
is checked against the original before it ships. The technical report,
[`docs/latex/main.pdf`](docs/latex/main.pdf), covers the three case studies end to end.

## Models

| Model | Encoder | Removable | Text-only | Encoder-only | Notes |
|---|---|---|---|---|---|
| [GLM-4.6V-Flash](https://huggingface.co/zai-org/GLM-4.6V-Flash) | 24-layer ViT + merger | 892,498,432 (8.671%) | done | done | bit-exact on text |
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27-layer ViT | 460,730,096 (1.658%) | done | mmproj GGUF | hybrid linear attention; MTP head kept |
| [Gemma 4 31B NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | 27-layer SigLIP | ~576M (1.9%) | done | none | quantization saves more than the split |
| [Gemma 4 12B](https://huggingface.co/google/gemma-4-12B) | none (52.4M input embedder) | 52,379,904 (0.438%) | scripted | n/a | encoder-free, not worth splitting |

Check `model_type` before applying any recipe. Only Gemma 4 12B is encoder-free.

## Published

Everything is in the [unfuse collection](https://huggingface.co/collections/sartajbhuvaji/unfuse-6ae8b265e63951518e4e75cc).

- [unfuse-GLM-4.6-Flash-text](https://huggingface.co/sartajbhuvaji/unfuse-GLM-4.6-Flash-text) and its [GGUF quants](https://huggingface.co/sartajbhuvaji/unfuse-GLM-4.6-Flash-text-GGUF)
- [unfuse-GLM-4.6V-Flash-mmproj-GGUF](https://huggingface.co/sartajbhuvaji/unfuse-GLM-4.6V-Flash-mmproj-GGUF), the vision half
- [unfuse-Qwen3.8-27B-text](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text), its [GGUF quants](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-text-GGUF) and [mmproj](https://huggingface.co/sartajbhuvaji/unfuse-Qwen3.8-27B-mmproj-GGUF)
- [unfuse-Gemma-4-31B-IT-NVFP4-FP8attn-text](https://huggingface.co/sartajbhuvaji/unfuse-Gemma-4-31B-IT-NVFP4-FP8attn-text), 23.0 GB

The text and mmproj GGUFs recombine into the original multimodal model under llama.cpp.

## Quick start

```bash
pip install -e ".[notebook,gguf,dev]"

unfuse inspect ~/models/some-model --prefix model.visual.   # where does the encoder live?
unfuse verify  ORIGINAL SPLIT                               # bit-exact check for a split
unfuse agreement ORIGINAL QUANTIZED --baseline SHIPPED      # for quantized builds
```

Extraction either reproduces the original bits or it is broken, so `verify` is exact.
Quantization rounds every weight, so `agreement` scores it against the build already
shipping instead of a fixed threshold.

## Layout

```
src/unfuse/   inspect a checkpoint, verify two match, score a quantized one
notebooks/    one directory per model: the surgery, benchmarks, results
tools/        GPU provisioning, GGUF build, serving benchmarks, Hub upload
tests/        unit tests for the core (no GPU, no network)
docs/         report, findings, guides, model cards
```

The surgery lives in the notebooks, not the library, because every model needs different
treatment. See [`docs/ADDING-A-MODEL.md`](docs/ADDING-A-MODEL.md). The notebooks are
committed with their outputs, which are the evidence for the claims. Don't strip them.

## More

- [`docs/METHOD.md`](docs/METHOD.md): why the two checks differ, and the headline result
- [`docs/RUNNING.md`](docs/RUNNING.md): how to run the notebooks and tools, and what is next
- [`docs/GOTCHAS.md`](docs/GOTCHAS.md): the traps that cost real time
- [`docs/FINDINGS.md`](docs/FINDINGS.md): what was tried, what failed, what it taught
- [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md): the background the repo assumes
