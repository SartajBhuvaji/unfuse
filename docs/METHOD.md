# Method

## Splitting is not always the lever

`nvidia/Gemma-4-31B-IT-NVFP4` is in the table because it makes the point. Its vision tower
is real and removable, but at ~1.15 GB it is the *smallest* of three obvious wins. Reading
the checkpoint's own quantization metadata shows why:

```bash
unfuse inspect ~/models/gemma-4-31b-nvfp4 --quant
```

NVFP4 is a 4-bit weight and activation format, but 63 modules are excluded — and 60 of
them are the attention blocks, listed one layer at a time. Only the MLPs were quantized,
which leaves **unquantized attention as the largest single block in a 32.67 GB file**.
Removing the encoder saves 1.15 GB; taking attention to FP8 saves 7.9 GB.

So that model gets treated as a quantization problem with a split attached, rather than
the reverse. `notebooks/gemma-4-31b-nvfp4/` does all three — vision removal, FP8
attention, and pairing the unpaired MTP drafter — and measures each separately.

Adding a model means answering three questions: where does the encoder live in the tensor
names, does a standalone class exist for the half you want, and what does the config need
to become. The two notebooks in `notebooks/glm-4.6v-flash/` walk through both directions
end to end.

## Two bars, deliberately not shared

Extraction does no arithmetic on the weights, so it either reproduces the original bits or
it is broken. `unfuse verify` holds that line and a test pins it there.

Quantization is the opposite case: it rounds every weight it touches, so a non-zero
difference is not a bug, it is the operation. `unfuse agreement` scores that — but not
against a threshold, because no one can justify the number and any threshold that exists
will eventually be tuned until the checkpoint passes.

The bar is comparative instead. Whoever shipped the quantized build published benchmark
scores with it, so they have already decided how much damage is acceptable. Measure their
checkpoint against the original, measure yours the same way, require yours to be no worse:

```bash
unfuse agreement google/gemma-4-31B-it ~/models/mine --baseline nvidia/Gemma-4-31B-IT-NVFP4
```

The two modules share no code, and merging them would break both: the exact bar would
reject every correct quantization, and the tolerant bar would let a broken extraction ship.

## The headline result

The GLM-4.6V-Flash extraction is **bit-exact on text**. Logits compared against the
original model:

```
max|d| = 0.000e+00  across six prompts, argmax_match=True on all
max|d| = 0.000e+00  at 1,207 tokens
```

That mattered because the text config uses multimodal RoPE (`mrope_section: [8,12,12]`).
For text-only input all three axes carry the same position, so mRoPE *should* collapse
to standard RoPE — but that was an argument from the config, not a measurement. Now it
is measured, at depth.
