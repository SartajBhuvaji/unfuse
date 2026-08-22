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
