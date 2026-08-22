"""Read what is actually inside a checkpoint.

This is the first thing to run against any new model. It answers the question the
whole project turns on: which tensors belong to which part, and how big is each part.

Everything here reads safetensors *headers* only. No weights are loaded, so
inspecting a 20 GB checkpoint costs no memory and takes about a second.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BYTES_PER_DTYPE = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def params(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    @property
    def nbytes(self) -> int:
        return self.params * BYTES_PER_DTYPE.get(self.dtype, 2)


@dataclass
class Inventory:
    """Every tensor in a checkpoint, with no weights loaded."""

    tensors: list[TensorInfo]

    @property
    def params(self) -> int:
        return sum(t.params for t in self.tensors)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    def __len__(self) -> int:
        return len(self.tensors)


def read_inventory(path: str | Path) -> Inventory:
    """Read tensor names, shapes and dtypes from a checkpoint directory or file."""
    from safetensors import safe_open

    path = Path(path)
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .safetensors under {path}")

    tensors = []
    for file in files:
        with safe_open(file, framework="pt") as handle:
            # .keys() is required: safe_open is a Rust object and is not iterable,
            # so ruff's SIM118 suggestion (`for name in handle`) raises TypeError.
            for name in handle.keys():  # noqa: SIM118
                slice_ = handle.get_slice(name)
                tensors.append(TensorInfo(name, tuple(slice_.get_shape()), slice_.get_dtype()))
    return Inventory(sorted(tensors, key=lambda t: t.name))


def split_by_prefix(inv: Inventory, prefix: str) -> tuple[Inventory, Inventory]:
    """Partition into (matching, rest). This is the whole basis of a modality split."""
    matching = [t for t in inv.tensors if t.name.startswith(prefix)]
    rest = [t for t in inv.tensors if not t.name.startswith(prefix)]
    return Inventory(matching), Inventory(rest)


def group_by_prefix(inv: Inventory, depth: int = 2) -> dict[str, Inventory]:
    """Group tensors by the first `depth` dot-separated segments of their names.

    Names are paths through the model, so truncating them shows the model's shape
    rather than a list of several hundred individual tensors. Repeated layers are
    collapsed: `model.layers.0` and `model.layers.1` both become `model.layers.*`.
    """
    groups: dict[str, list[TensorInfo]] = defaultdict(list)
    for tensor in inv.tensors:
        parts = tensor.name.split(".")[:depth]
        # Collapse layer indices so a 40-layer stack reads as one row.
        parts = ["*" if p.isdigit() else p for p in parts]
        groups[".".join(parts)].append(tensor)
    return {k: Inventory(v) for k, v in sorted(groups.items())}


def find_index(path: str | Path) -> dict | None:
    """Return the safetensors index if the checkpoint is sharded.

    Useful before downloading: the index maps every tensor to its shard, so you can
    tell whether the part you want happens to live in one file.
    """
    index = Path(path) / "model.safetensors.index.json"
    return json.loads(index.read_text()) if index.exists() else None


def shards_for_prefix(index: dict, prefix: str) -> list[str]:
    """Which shard files hold tensors under `prefix`."""
    return sorted({shard for name, shard in index["weight_map"].items() if name.startswith(prefix)})


# --------------------------------------------------------------------------
# Quantization state
#
# A quantized checkpoint carries a second, parallel description of itself: which
# modules were quantized, to what, and which were skipped. That list is the most
# informative thing in the repo and it is easy to misread -- NVIDIA's NVFP4 build
# of Gemma 4 31B names 63 excluded modules, and reading them carefully is what
# reveals that every attention layer is still BF16 and only the MLPs were touched.
#
# Two formats exist. `quantization_config` inside config.json is the current one
# and carries per-layer algorithms; hf_quant_config.json is the legacy file and
# carries one global algorithm plus an exclusion list. Checkpoints in the wild use
# either, so read both.
# --------------------------------------------------------------------------

QUANT_BITS = {"NVFP4": 4, "MXFP4": 4, "FP8": 8, "INT8": 8, "INT4": 4, "W4A16": 4}


@dataclass(frozen=True)
class QuantConfig:
    algo: str | None
    kv_cache_algo: str | None
    group_size: int | None
    exclude_modules: tuple[str, ...]
    producer: str | None

    def state_for(self, module: str) -> str:
        """What happened to `module`: its algorithm, or "BF16" if it was skipped.

        Patterns are globs and `*` crosses dots, so `...layers.0.self_attn*`
        covers every projection inside that attention block.
        """
        from fnmatch import fnmatch

        if any(fnmatch(module, pattern) for pattern in self.exclude_modules):
            return "BF16"
        return self.algo or "BF16"


def read_quant_config(path: str | Path) -> QuantConfig | None:
    """Read a checkpoint's quantization metadata, whichever format it uses."""
    path = Path(path)

    config = path / "config.json"
    if config.exists():
        block = json.loads(config.read_text()).get("quantization_config")
        if block:
            return _quant_config(block)

    legacy = path / "hf_quant_config.json"
    if legacy.exists():
        raw = json.loads(legacy.read_text())
        block = raw.get("quantization", raw)
        producer = (raw.get("producer") or {}).get("name")
        return _quant_config(block, producer)

    return None


def _scheme_name(scheme: dict | None) -> str | None:
    """Turn a compressed-tensors `{num_bits, type}` block into a format name."""
    if not scheme:
        return None
    bits, kind = scheme.get("num_bits"), scheme.get("type")
    if kind == "float":
        return {4: "NVFP4", 8: "FP8"}.get(bits)
    if kind == "int":
        return {4: "INT4", 8: "INT8"}.get(bits)
    return None


def _quant_config(block: dict, producer: str | None = None) -> QuantConfig:
    """Parse either schema, including checkpoints that carry both at once.

    Gemma-4-31B-IT-NVFP4 is the awkward case that motivated this: its
    `quantization_config` mixes modelopt keys (`quant_algo`, `quant_method`) with
    compressed-tensors keys (`config_groups`, `ignore`, `kv_cache_scheme`) in one
    block. Reading only the modelopt half silently loses the group size and the KV
    cache format, both of which live in the compressed-tensors half.
    """
    # config_groups is compressed-tensors' per-group description. Real checkpoints
    # have used a single group so far, so take the first rather than modelling a
    # generality nothing has needed yet.
    groups = block.get("config_groups") or {}
    first = next(iter(groups.values()), {}) if isinstance(groups, dict) else {}
    weights = first.get("weights") or {}

    producer_block = producer or block.get("producer")
    if isinstance(producer_block, dict):
        producer_block = producer_block.get("name")

    algo = block.get("quant_algo") or _scheme_name(weights)
    if algo is None and block.get("quant_method") != "compressed-tensors":
        algo = block.get("quant_method")

    return QuantConfig(
        algo=algo,
        kv_cache_algo=(
            block.get("kv_cache_quant_algo") or _scheme_name(block.get("kv_cache_scheme"))
        ),
        group_size=block.get("group_size") or weights.get("group_size"),
        exclude_modules=tuple(block.get("exclude_modules") or block.get("ignore") or ()),
        producer=producer_block,
    )


# Suffixes that describe a quantized weight rather than being one. They must not
# be mistaken for the module's own name when matching exclusion patterns.
_SCALE_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".weight_scale_inv",
    ".act_scale",
    ".k_scale",
    ".v_scale",
)


def module_of(tensor_name: str) -> str:
    """The module a tensor belongs to: `...q_proj.weight` -> `...q_proj`."""
    for suffix in _SCALE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name.rsplit(".", 1)[0] if "." in tensor_name else tensor_name


# Storage dtypes that cannot be holding quantized weights, whatever the config says.
_UNQUANTIZED_DTYPES = {"BF16", "F16", "F32", "F64"}

_STORED_AS = {"F8_E4M3": "FP8", "F8_E5M2": "FP8"}


def observed_state(tensor: TensorInfo, qc: QuantConfig) -> str:
    """What a tensor's dtype says happened to it, which beats what the config claims.

    The two disagree more often than they should. A tied `lm_head` is the common
    case: the exclusion list names `lm_head`, but the tensor is stored under
    `embed_tokens`, so matching on the name alone reports it as quantized while
    the header plainly says BF16. Trusting the config there would overstate how
    much of the checkpoint was actually touched.

    Scale tensors are metadata about a quantized weight rather than weights, so
    they take their module's state instead of voting on their own.
    """
    if tensor.dtype in _STORED_AS:
        return _STORED_AS[tensor.dtype]

    module = module_of(tensor.name)
    claimed = qc.state_for(module)
    is_scale = tensor.name != module and not tensor.name.endswith(".weight")
    if tensor.dtype in _UNQUANTIZED_DTYPES and not is_scale:
        return "BF16"
    return claimed


def quant_report(inv: Inventory, qc: QuantConfig, depth: int = 2) -> dict[str, dict]:
    """Group tensors and report each group's parameters, bytes and quant state.

    Bytes and states both come from the safetensors headers, so this measures what
    is on disk rather than what the config claims -- see `observed_state`.
    """
    groups: dict[str, dict] = {}
    for tensor in inv.tensors:
        parts = ["*" if p.isdigit() else p for p in tensor.name.split(".")[:depth]]
        key = ".".join(parts)
        entry = groups.setdefault(key, {"tensors": 0, "params": 0, "nbytes": 0, "states": set()})
        entry["tensors"] += 1
        entry["params"] += tensor.params
        entry["nbytes"] += tensor.nbytes
        entry["states"].add(observed_state(tensor, qc))

    for entry in groups.values():
        entry["state"] = "+".join(sorted(entry.pop("states")))
    return dict(sorted(groups.items()))
