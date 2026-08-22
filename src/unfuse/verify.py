"""Check that two checkpoints compute the same thing.

The split itself is a dictionary filter and rarely goes wrong in an interesting
way. What goes wrong is the config: a dropped field, a renamed key, a position
encoding that quietly behaves differently. Those produce a model that loads
happily and is subtly not the model you started with.

So the bar here is exact equality, not closeness. Extraction does no arithmetic on
the weights, so if the resulting model computes the same function the floating
point operations are identical in the same order and must produce identical bits.
A tiny non-zero difference is the worst outcome -- it looks like harmless rounding,
ships, and misbehaves later in a case nobody tested.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Explain why the sky appears blue, in one sentence.",
    "1, 1, 2, 3, 5, 8, 13,",
    "Translate to German: The weather is cold today.",
    "The three laws of thermodynamics state that",
]


@dataclass
class Comparison:
    results: list[tuple[str, float, bool]] = field(default_factory=list)

    @property
    def worst(self) -> float:
        return max((d for _, d, _ in self.results), default=0.0)

    @property
    def exact(self) -> bool:
        return self.results != [] and self.worst == 0.0

    def report(self) -> str:
        lines = [f"  max|d|={d:<12.3e} argmax_match={m}  {p[:44]!r}" for p, d, m in self.results]
        lines.append("")
        if self.exact:
            lines.append("EXACT MATCH -- the extraction is lossless on these inputs.")
        else:
            lines.append(f"DIVERGENCE {self.worst:.3e} -- do not ship this checkpoint.")
        return "\n".join(lines)


def _logits(model_id, tokenizer, prompts, device, dtype, loader):
    import torch

    model = loader(model_id, dtype=dtype).to(device).eval()
    out = {}
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").to(device)
            # float() so the later subtraction is not itself lossy; cpu() because
            # the second model is about to need this GPU memory.
            out[prompt] = model(**ids).logits.float().cpu()
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def compare_logits(
    reference: str,
    candidate: str,
    prompts: list[str] | None = None,
    device: str = "cuda",
    dtype=None,
    reference_loader=None,
    candidate_loader=None,
) -> Comparison:
    """Run the same prompts through both models and compare raw logits.

    Models are loaded one at a time. Two large checkpoints will not co-reside, and
    the whole point is to compare models big enough that this matters.

    Logits rather than generated text: logits are the model's score for every token
    in the vocabulary, so they catch differences that decoded words would hide.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = prompts or DEFAULT_PROMPTS
    dtype = dtype or torch.bfloat16
    load = AutoModelForCausalLM.from_pretrained

    def default(model_id, dtype):
        return load(model_id, dtype=dtype, low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=True)

    ref = _logits(reference, tokenizer, prompts, device, dtype, reference_loader or default)
    cand = _logits(candidate, tokenizer, prompts, device, dtype, candidate_loader or default)

    result = Comparison()
    for prompt in prompts:
        delta = (ref[prompt] - cand[prompt]).abs().max().item()
        same = ref[prompt].argmax(-1).eq(cand[prompt].argmax(-1)).all().item()
        result.results.append((prompt, delta, bool(same)))
    return result


def long_prompt(repeats: int = 120) -> str:
    """A prompt long enough to exercise position encoding.

    Short prompts do not test what a position-encoding change actually affects. A
    bug there can leave the first few tokens correct and drift later, so a handful
    of six-token prompts will pass while the model is broken at length.
    """
    filler = "The following is a technical description repeated for length. "
    return filler * repeats + "In summary, the key point is"
