"""Expand a quantized checkpoint back to BF16, so it can be run and scored anywhere.

Why this is not a cheat
-----------------------
Weight-only quantization is *exactly* simulatable. An FP8 or NVFP4 kernel does not
multiply in 4 bits: it dequantizes the stored weight to a wide type and multiplies
against the activations. So a checkpoint whose weights have been round-tripped
through the format computes bit-identical results to the real one, and measuring
accuracy on the expanded copy measures the real thing.

What it does *not* simulate is activation quantization. NVFP4 is W4A4, and the A4
half is skipped here -- so an expanded NVFP4 checkpoint is very slightly optimistic
about NVIDIA's build. Both sides of any comparison are expanded the same way, so
the comparison stays fair for the weight change under test; it just is not a claim
about end-to-end deployed accuracy. Say so when reporting it.

This exists because NVFP4 executes only on Blackwell, and because transformers has
no loader for `quant_method: modelopt` at all -- it ignores the quantization config,
loads the packed uint8 tensors as dense weights, and fails on a shape mismatch of
exactly 2x the input dimension. That 2x is the tell: two 4-bit values per byte.

Formats
-------
NVFP4  weight [out, in/2] uint8, two e2m1 values per byte, low nibble first
       weight_scale   [out, in/16] float8_e4m3, one per group of 16
       weight_scale_2 scalar float32, one per tensor
       w = e2m1(nibble) * weight_scale * weight_scale_2

FP8    weight [out, in] float8_e4m3 with a per-tensor or per-channel weight_scale.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# e2m1: 1 sign, 2 exponent, 1 mantissa bit. Eight magnitudes, sign in bit 3.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def e2m1_table():
    import torch

    return torch.tensor(E2M1 + [-v for v in E2M1], dtype=torch.float32)


def unpack_nvfp4(packed, scale, scale_2, group_size: int = 16):
    """packed [out, in/2] uint8 -> [out, in] float32."""
    import torch

    lut = e2m1_table().to(packed.device)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # Low nibble is the earlier element, so interleave low,high along the row.
    nibbles = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1).long()
    values = lut[nibbles]

    scale = scale.to(torch.float32)
    if scale_2 is not None:
        scale = scale * scale_2.to(torch.float32)
    # One scale per group of `group_size` along the input dimension.
    expanded = scale.repeat_interleave(group_size, dim=-1)
    if expanded.shape[-1] != values.shape[-1]:
        raise ValueError(f"scale covers {expanded.shape[-1]} cols, weight has {values.shape[-1]}")
    return values * expanded


def unpack_fp8(weight, scale):
    import torch

    out = weight.to(torch.float32)
    if scale is None:
        return out
    scale = scale.to(torch.float32)
    return out * (scale.unsqueeze(-1) if scale.ndim == 1 else scale)


def convert(src: Path, dst: Path, group_size: int) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    stats = {"nvfp4": 0, "fp8": 0, "copied": 0}
    weight_map, total = {}, 0

    for shard in sorted(src.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            names = set(f.keys())
            meta = f.metadata() or {}
            out = {}
            for name in sorted(names):
                if name.endswith((".weight_scale", ".weight_scale_2", ".input_scale")):
                    continue  # consumed by the weight they belong to
                tensor = f.get_tensor(name)

                scale_name, scale2_name = f"{name}_scale", f"{name}_scale_2"
                scale = f.get_tensor(scale_name) if scale_name in names else None
                scale_2 = f.get_tensor(scale2_name) if scale2_name in names else None

                if tensor.dtype == torch.uint8 and scale is not None:
                    tensor = unpack_nvfp4(tensor, scale, scale_2, group_size).to(torch.bfloat16)
                    stats["nvfp4"] += 1
                elif tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    tensor = unpack_fp8(tensor, scale).to(torch.bfloat16)
                    stats["fp8"] += 1
                else:
                    stats["copied"] += 1

                out[name] = tensor
                weight_map[name] = shard.name
                total += tensor.numel() * tensor.element_size()

        save_file(out, dst / shard.name, metadata={**meta, "format": "pt"})
        print(f"  {shard.name}: nvfp4={stats['nvfp4']} fp8={stats['fp8']}", file=sys.stderr)
        del out

    _write_config(src, dst, weight_map, total)
    return stats


def _write_config(src: Path, dst: Path, weight_map: dict, total: int) -> None:
    """Strip the quantization metadata: the result is a plain BF16 checkpoint, and
    leaving the config claiming NVFP4 would make loaders look for scales that are
    no longer there."""
    config = json.loads((src / "config.json").read_text())
    config.pop("quantization_config", None)
    config["dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    index = src / "model.safetensors.index.json"
    if index.exists():
        data = json.loads(index.read_text())
        data["weight_map"] = weight_map
        data["metadata"] = {"total_size": total}
        (dst / "model.safetensors.index.json").write_text(json.dumps(data, indent=2))

    for path in src.iterdir():
        skip = path.name in ("hf_quant_config.json", "config.json")
        if path.is_file() and path.suffix != ".safetensors" and not skip:
            shutil.copy2(path, dst / path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--group-size", type=int, default=16)
    args = parser.parse_args(argv)

    stats = convert(args.src, args.dst, args.group_size)
    print(f"\nexpanded {stats['nvfp4']} NVFP4 and {stats['fp8']} FP8 tensors")
    print(f"copied {stats['copied']} unquantized tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
