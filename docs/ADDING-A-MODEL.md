# Adding a model

Every model needs its own treatment. That is not a gap in the tooling, it is the
shape of the problem — the two models here needed almost nothing in common. So this
is a checklist, not an API.

Work through it in order. The first two steps are cheap and often tell you to stop.

## 1. Where does the encoder live in the tensor names?

```bash
unfuse inspect ~/models/the-model --prefix model.visual.
```

You are looking for a prefix that cleanly separates the modality from the rest. Try
`model.visual.`, `visual.`, `vision_tower.`, `model.vision_tower.`. If no prefix
isolates it, the model may not be separable at all.

The output also tells you what share of the weights the encoder is. **This is where
most models should stop.** Gemma 4 12B's vision path is 0.438% — quantizing is
roughly 160 times the saving, so splitting it for size is not worth doing.

## 2. Does it fit in one shard?

The same command reports this. If every tensor you want sits in one shard, you can
download that file alone. On GLM-4.6V-Flash that turned a 20.6 GB download into
4.7 GB, which is the difference between a laptop job and a rented GPU.

Do not hard-code the shard number. Assert the assumption instead, so a re-upload
that resplits the model fails loudly rather than producing a wrong checkpoint.

## 3. Is there a class for the half you want?

Look in the model's `modeling_*.py` in transformers, at `__all__`.

This is where the two directions diverge sharply, and it decides how much work you
are in for:

| | encoder | decoder |
|---|---|---|
| GLM-4V | `Glm4vVisionModel` exists | **no `Glm4vForCausalLM`** — retarget onto `Glm4ForCausalLM` |
| Gemma 4 | n/a | `Gemma4UnifiedForCausalLM` exists |

If a class exists, the surgery is: keep the matching tensors, strip the prefix so
the names match what that class expects, write a config, done.

If one does not, you have to find a different architecture whose layer structure
matches and retarget the config at it. For GLM-4V the text tower was a GLM-4
decoder, which is why `Glm4ForCausalLM` worked. Check the layer names match before
assuming.

## 4. What does the config have to become?

Three recurring traps:

**Fields that only made sense with the other modality.** GLM-4V's text config had
`mrope_section: [8, 12, 12]` — multimodal position encoding split across time,
height and width. With no image tokens those axes are meaningless, so it gets
dropped. Whether dropping it is safe is a real question, not a formality; see
step 5.

**`architectures` is never populated for you.** Neither `Glm4Config` nor
`Glm4vVisionConfig` sets it. Without it `AutoModelForCausalLM` has no class to
dispatch to and loading fails confusingly. Set it by hand, every time.

**Tied vs untied embeddings.** If `tie_word_embeddings` is false there is a separate
`lm_head` tensor, and nothing will warn you if you drop it. On GLM-4V that is 620
million parameters and the model's entire ability to produce words.

Build the config through the config class rather than editing JSON, so a renamed
field in a future transformers release surfaces as an error instead of silence.

## 5. Prove it

Two checks, and they catch different failures.

**Structural** — the tensors you kept should be bit-identical to the source, since
extraction copies rather than computes. `torch.equal`, not `allclose`.

**Behavioural** — the new checkpoint should produce identical logits to the
original.

```bash
unfuse verify zai-org/GLM-4.6V-Flash ~/models/GLM-4.6-Flash-text
unfuse verify zai-org/GLM-4.6V-Flash ~/models/GLM-4.6-Flash-text --long
```

The bar is **exactly zero**, not "close". No arithmetic was done on the weights, so
if the model computes the same function the operations are identical in the same
order and must produce identical bits. A tiny non-zero difference is the worst
outcome: it reads as harmless rounding, ships, and misbehaves later.

Run the long check too. Six short prompts do not exercise position encoding, which
is exactly what step 4 modified. On GLM-4V the short prompts passed *and* so did
1,207 tokens — but the second one is what earned the claim.

## 6. Write it down

Add a row to the table in `README.md` and the measured numbers to `FINDINGS.md`.
Whether a model was *not* worth splitting is as useful to record as one that was.
