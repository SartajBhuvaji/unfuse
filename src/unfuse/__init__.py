"""Split multimodal models into their parts, losslessly.

Only the model-agnostic half lives here: reading a checkpoint's tensor inventory,
and checking that two checkpoints compute the same thing. Neither knows anything
about a particular architecture.

The surgery itself is deliberately not here. Every model needs different treatment
-- GLM-4V wanted a prefix rename, an untied lm_head kept, mRoPE dropped and a
retarget onto a different architecture class, while Gemma 4 needs none of that
because it ships a text-only class already. Hiding that behind one function would
produce a registry of special cases pretending to be an abstraction. The notebooks
show the surgery per model instead.

Quantization is scored separately, in `agreement`. Extraction must be bit-exact;
quantization cannot be, so the two need different bars and deliberately do not
share code. See that module's docstring for why.
"""

from unfuse.inspect import (
    Inventory,
    QuantConfig,
    TensorInfo,
    group_by_prefix,
    observed_state,
    quant_report,
    read_inventory,
    read_quant_config,
    split_by_prefix,
)

__all__ = [
    "Inventory",
    "QuantConfig",
    "TensorInfo",
    "group_by_prefix",
    "observed_state",
    "quant_report",
    "read_inventory",
    "read_quant_config",
    "split_by_prefix",
]
__version__ = "0.1.0"
