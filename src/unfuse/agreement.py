"""Check that a *quantized* checkpoint still computes nearly the same thing.

This is deliberately not `verify.py`, and the two must not be merged.

Extraction does no arithmetic on the weights, so it either reproduces the original
bits or it is broken -- `verify.compare_logits` holds that line and a test pins it
there. Quantization is the opposite case: it rounds every weight it touches, so a
non-zero difference is not a bug, it is the entire operation. Applying the exact
bar here would reject a correct result, and relaxing the exact bar over there would
let a broken extraction ship. Hence two modules.

That leaves the harder question of what "nearly" means, because `max|d|` alone
cannot answer it. A logit delta of 0.3 is meaningless without knowing what delta
the model already tolerates, and nobody has an intuition for that number.

So the bar here is comparative, not absolute. NVIDIA already shipped a quantized
build of this model and published its benchmark scores, which means they have
already decided how much damage is acceptable. Measure their checkpoint against the
BF16 original, measure ours the same way, and require ours to be no worse. That
turns an unanswerable question ("is KL of 0.004 fine?") into an answerable one
("is ours closer to the original than the one NVIDIA shipped?").

`Agreement.is_no_worse_than` is that check.

It is the right check only when the change is not supposed to cost anything --
removing an encoder, or replacing a format with a strictly better one. A change
that *adds* quantization on top of what the baseline already did cannot pass it,
by construction, and pretending otherwise would be a rigged test. For those,
`Agreement.marginal_cost` reports the increment instead: how much damage the new
step adds relative to the damage the shipped build already accepted. Adding 8% to
an error someone already shipped is a very different claim from adding 300%, and
neither is visible in an absolute KL.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Reused so a quantized run is scored on the same inputs as an extraction run.
from unfuse.verify import DEFAULT_PROMPTS, long_prompt

__all__ = [
    "CHAT_PROMPTS",
    "DEFAULT_PROMPTS",
    "Agreement",
    "PromptScore",
    "as_chat",
    "compare_distributions",
    "long_prompt",
]

# Instruction-tuned models must be scored through their chat template.
#
# This is not a style preference. Feeding a raw completion prompt to an IT model
# drops it into a degenerate regime -- gemma-4-31B-it continues "The capital of
# France is" with "France is France is France is", while the same model answers
# "Paris" correctly through its template. Scoring in the degenerate regime measures
# noise: divergences come out an order of magnitude too large, and small
# perturbations move them in arbitrary directions, so a quantized model can appear
# *closer* to the reference than a less-quantized one.
#
# The raw prompts in `verify.DEFAULT_PROMPTS` stay as they are, because an
# extraction has to be bit-exact on any input, degenerate or not. Only the
# distributional comparison needs the model to be in a regime worth measuring.
CHAT_PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain in one sentence why the sky appears blue.",
    "Continue the sequence and explain the rule: 1, 1, 2, 3, 5, 8, 13",
    "Translate to German: The weather is cold today.",
    "State the three laws of thermodynamics.",
]


def as_chat(tokenizer, prompts: list[str]) -> list[str]:
    """Wrap each prompt in the model's chat template."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]


@dataclass(frozen=True)
class PromptScore:
    """How closely two models agreed on one prompt.

    `top1` is the headline. It is the fraction of token positions where both models
    would have made the same greedy choice, which is the thing a user actually
    experiences -- a model whose logits moved but whose argmax did not has not
    changed its behaviour under greedy decoding at all.
    """

    prompt: str
    top1: float  # fraction of positions with the same argmax
    top5: float  # mean overlap of the two top-5 sets, 0..1
    kl: float  # mean KL(reference || candidate) in nats
    max_abs: float  # largest single logit difference


@dataclass
class Agreement:
    """A quantized checkpoint scored against its unquantized reference."""

    scores: list[PromptScore] = field(default_factory=list)
    label: str = "candidate"

    @property
    def top1(self) -> float:
        """Worst per-prompt top-1 agreement. Worst, not mean: an average hides a
        single prompt the quantization broke, and one broken capability is what we
        are trying to detect."""
        return min((s.top1 for s in self.scores), default=0.0)

    @property
    def kl(self) -> float:
        return max((s.kl for s in self.scores), default=float("inf"))

    @property
    def max_abs(self) -> float:
        return max((s.max_abs for s in self.scores), default=float("inf"))

    def is_no_worse_than(self, baseline: Agreement) -> bool:
        """The shipping criterion: at least as close to the reference as `baseline`.

        Both must have been scored against the same reference on the same prompts,
        or this compares nothing. An empty run fails rather than passing vacuously,
        for the same reason `Comparison.exact` does.
        """
        if not self.scores or not baseline.scores:
            return False
        return self.top1 >= baseline.top1 and self.kl <= baseline.kl

    def marginal_cost(self, baseline: Agreement) -> dict[str, float]:
        """Damage added on top of the baseline's, for changes that add quantization.

        `kl_ratio` of 1.0 means this checkpoint diverges from the reference no more
        than the baseline does; 1.1 means it adds 10% to an error the baseline's
        publisher already judged acceptable. `top1_drop` is in absolute fraction of
        token positions, which is the more intuitive of the two.
        """
        return {
            "kl_ratio": self.kl / baseline.kl if baseline.kl else float("inf"),
            "top1_drop": baseline.top1 - self.top1,
        }

    def report(self, baseline: Agreement | None = None) -> str:
        lines = [f"{self.label} vs reference", ""]
        header = f"  {'top1':>7} {'top5':>7} {'KL':>10} {'max|d|':>10}  prompt"
        lines.append(header)
        for s in self.scores:
            lines.append(
                f"  {s.top1:>7.4f} {s.top5:>7.4f} {s.kl:>10.3e} {s.max_abs:>10.3e}  {s.prompt[:36]!r}"
            )
        lines.append("")
        lines.append(f"  worst top1 {self.top1:.4f}   worst KL {self.kl:.3e}")

        if baseline is not None:
            lines += [
                "",
                f"baseline ({baseline.label}): top1 {baseline.top1:.4f}  KL {baseline.kl:.3e}",
                "",
            ]
            if self.is_no_worse_than(baseline):
                lines.append("NO WORSE THAN BASELINE -- safe to publish on this evidence.")
            else:
                cost = self.marginal_cost(baseline)
                lines.append(
                    f"ADDS {cost['kl_ratio']:.2f}x the baseline's divergence, "
                    f"top1 -{cost['top1_drop']:.4f}."
                )
                lines.append("Expected when the change adds quantization the baseline did not")
                lines.append("have. Judge the increment against what it buys, not against zero.")
        return "\n".join(lines)


def _score(ref, cand, prompt: str) -> PromptScore:
    """Score one prompt's logits. Both tensors are [1, T, V] float32 on CPU."""
    import torch

    ref = ref[0].float()
    cand = cand[0].float()

    top1 = ref.argmax(-1).eq(cand.argmax(-1)).float().mean().item()

    k = min(5, ref.shape[-1])
    ref_top = ref.topk(k, dim=-1).indices
    cand_top = cand.topk(k, dim=-1).indices
    # Set overlap per position: how many of the reference's top-k survived.
    overlap = torch.stack(
        [torch.isin(ref_top[i], cand_top[i]).sum() for i in range(ref_top.shape[0])]
    )
    top5 = (overlap.float() / k).mean().item()

    # KL over the full vocabulary, in log space throughout. Softmaxing 262k logits
    # directly overflows in float32 for the larger activations.
    ref_log = torch.log_softmax(ref, dim=-1)
    cand_log = torch.log_softmax(cand, dim=-1)
    kl = (ref_log.exp() * (ref_log - cand_log)).sum(-1).mean().item()

    return PromptScore(prompt, top1, top5, kl, (ref - cand).abs().max().item())


def compare_distributions(
    reference: str,
    candidate: str,
    prompts: list[str] | None = None,
    device: str = "cuda",
    label: str | None = None,
    chat: bool = False,
    reference_loader=None,
    candidate_loader=None,
) -> Agreement:
    """Score `candidate` against `reference` on the same prompts.

    Models load one at a time, as in `verify` -- a 31B reference and a quantized
    candidate will not co-reside on one GPU, and those are exactly the sizes this
    is for.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unfuse.verify import _logits

    prompts = prompts or (CHAT_PROMPTS if chat else DEFAULT_PROMPTS)

    def default(model_id, dtype):
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(reference, trust_remote_code=True)
    if chat:
        prompts = as_chat(tokenizer, prompts)

    # dtype=None lets a quantized checkpoint keep the dtypes its config specifies.
    # Forcing bfloat16 here would silently dequantize the thing under test.
    ref = _logits(reference, tokenizer, prompts, device, None, reference_loader or default)
    cand = _logits(candidate, tokenizer, prompts, device, None, candidate_loader or default)

    return Agreement(
        scores=[_score(ref[p], cand[p], p) for p in prompts],
        label=label or candidate,
    )
