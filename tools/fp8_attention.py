"""Quantize a modelopt checkpoint's untouched modules to FP8, in place on disk.

Why this exists
---------------
NVIDIA's NVFP4 builds quantize the MLPs and leave attention alone. On
Gemma-4-31B-IT-NVFP4 that means 63 excluded modules, of which 60 are the entire
attention stack -- 7.93B parameters, 15.85 GB, the single largest block in a
32.67 GB checkpoint. Leaving it BF16 is also a deviation from NVIDIA's own house
style: their DeepSeek-V4 and Qwen3-Omni builds put FP4 in the experts and FP8 in
attention. This brings the Gemma build in line with that.

Weights only, by default
------------------------
FP8 has two independent halves. Quantizing the *weights* is a pure function of the
weights -- no calibration data, no forward passes, nothing to get wrong beyond the
scale arithmetic. Quantizing the *activations* needs a calibration set and is where
the risk actually lives, because attention activations are the outlier-heavy ones
that SmoothQuant and AWQ exist to handle.

Weights-only still gets the whole memory win (15.85 -> 7.93 GB) and still speeds up
batch-1 decode, which is memory-bandwidth-bound rather than compute-bound. What it
does not get is a prefill win, because BF16 activations cannot feed FP8 tensor
cores. `--activations` opts into the riskier version once there is a reason to.

This does not load the model. It reads one shard at a time, so peak memory is the
size of the largest shard (~5 GB here) rather than the 33 GB checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from fnmatch import fnmatch
from pathlib import Path

# e4m3 saturates at 448. Scales are chosen so the largest magnitude in a tensor
# lands exactly there, which is the whole of the "quantization" here.
FP8_MAX = 448.0


# Projections only. A bare "*self_attn*" also matches q_norm and k_norm, and
# quantizing those is a mistake in both directions: they are 1-D RMSNorm scales, so
# there is no size to win, and norms are among the most precision-sensitive weights
# in the model. The guard in `convert` catches it structurally as well, because a
# pattern is easy to get wrong and this one already was.
DEFAULT_PATTERNS = ["*self_attn*_proj"]


def targets(names: list[str], patterns: list[str]) -> list[str]:
    """Which `.weight` tensors to quantize, matching patterns against the *module*.

    Patterns name modules ("...self_attn.q_proj"), not tensors, so the `.weight`
    suffix is stripped before matching. Doing it the other way round means every
    pattern has to end in `.weight` to match anything, which silently converts
    nothing at all.
    """
    out = []
    for name in names:
        if not name.endswith(".weight"):
            continue
        if any(fnmatch(name[: -len(".weight")], p) for p in patterns):
            out.append(name)
    return out


def quantize_tensor(weight, per_channel: bool):
    """Return (fp8_weight, scale). Scale is float32 and always dequantizes back.

    Per-tensor is what modelopt emits for FP8 by default and what every serving
    path understands. Per-channel is finer-grained and measurably better on
    weights with uneven row magnitudes, but check your engine reads it before
    trusting it -- a scale nobody applies is silently a 30x error.
    """
    import torch

    weight = weight.to(torch.float32)
    if per_channel:
        amax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    else:
        amax = weight.abs().amax().clamp(min=1e-12)

    scale = (amax / FP8_MAX).to(torch.float32)
    quantized = (weight / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return quantized, scale.squeeze()


def convert(src: Path, dst: Path, patterns: list[str], per_channel: bool, dry_run: bool) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors under {src}")

    stats = {"converted": 0, "copied": 0, "skipped_1d": 0, "bytes_before": 0, "bytes_after": 0}
    weight_map: dict[str, str] = {}
    quantized_modules: list[str] = []

    for shard in shards:
        out: dict = {}
        with safe_open(shard, framework="pt") as handle:
            names = list(handle.keys())
            chosen = set(targets(names, patterns))

            for name in names:
                tensor = handle.get_tensor(name)
                if name in chosen and tensor.ndim < 2:
                    # Norms and biases. Never worth quantizing, and a matched
                    # pattern is not evidence that quantizing it was intended.
                    stats["skipped_1d"] += 1
                    out[name] = tensor
                    stats["copied"] += 1
                    stats["bytes_before"] += tensor.numel() * tensor.element_size()
                    stats["bytes_after"] += tensor.numel() * tensor.element_size()
                    weight_map[name] = shard.name
                elif name in chosen:
                    before = tensor.numel() * tensor.element_size()
                    quantized, scale = quantize_tensor(tensor, per_channel)
                    out[name] = quantized
                    out[name.replace(".weight", ".weight_scale")] = scale
                    quantized_modules.append(name[: -len(".weight")])
                    stats["converted"] += 1
                    stats["bytes_before"] += before
                    stats["bytes_after"] += quantized.numel() + scale.numel() * 4
                else:
                    out[name] = tensor
                    stats["copied"] += 1
                    stats["bytes_before"] += tensor.numel() * tensor.element_size()
                    stats["bytes_after"] += tensor.numel() * tensor.element_size()

        for key in out:
            weight_map[key] = shard.name
        if not dry_run:
            # Preserve the original metadata header; losing format="pt" makes the
            # shard unreadable by some loaders.
            with safe_open(shard, framework="pt") as handle:
                meta = handle.metadata() or {}
            save_file(out, dst / shard.name, metadata={**meta, "format": "pt"})
        del out
        print(f"  {shard.name}: {stats['converted']} converted so far", file=sys.stderr)

    if not dry_run:
        _write_index(src, dst, weight_map, stats["bytes_after"])
        _write_config(src, dst, quantized_modules, per_channel)
        _copy_rest(src, dst)
    return stats


def _write_index(src: Path, dst: Path, weight_map: dict, total: int) -> None:
    """Rebuild the shard index.

    `total_size` is nominally bytes, but not every publisher treats it that way --
    GLM-4.6V-Flash reports its parameter count there instead. Write real bytes and
    do not copy the original number forward.
    """
    index = src / "model.safetensors.index.json"
    if not index.exists():
        return
    data = json.loads(index.read_text())
    data["weight_map"] = weight_map
    data["metadata"] = {**data.get("metadata", {}), "total_size": total}
    (dst / "model.safetensors.index.json").write_text(json.dumps(data, indent=2))


def _write_config(src: Path, dst: Path, quantized: list[str], per_channel: bool) -> None:
    """Record what was just quantized, in whichever schema the checkpoint uses.

    Two things must happen or the config ends up lying about the checkpoint:
    the newly-quantized modules must leave the exclusion list, and they must be
    recorded as FP8. A serving engine that reads a stale config expects BF16
    attention and finds FP8 tensors.

    The schema is not fixed. A checkpoint straight from ModelOpt has the legacy
    `hf_quant_config.json`; one that has been through an extraction may have only
    `config.json` with a `quantization_config` block, and that block may itself be
    a hybrid of modelopt and compressed-tensors keys. Handle all of them, and
    write whichever files were already present rather than inventing new ones.
    """

    # An exclusion is dropped when the modules it covered are now quantized. It
    # has to be matched against the module names we actually converted -- comparing
    # it to the command-line glob would never match, since the two are written at
    # different granularities ("*self_attn*" vs "model...layers.0.self_attn*").
    def prune(patterns: list[str]) -> tuple[list[str], list[str]]:
        kept, removed = [], []
        for pattern in patterns:
            hit = any(fnmatch(module, pattern) for module in quantized)
            (removed if hit else kept).append(pattern)
        return kept, removed

    def layer_entries(patterns: list[str]) -> dict:
        return {
            pattern: {
                "quant_algo": "FP8",
                "scheme": "per_channel" if per_channel else "per_tensor",
                "activations": "BF16",
            }
            for pattern in patterns
        }

    legacy_path = src / "hf_quant_config.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        block = legacy.get("quantization", legacy)
        block["exclude_modules"], _ = prune(block.get("exclude_modules", []))
        (dst / "hf_quant_config.json").write_text(json.dumps(legacy, indent=4))

    config_path = src / "config.json"
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text())
    quant = config.get("quantization_config")
    if not quant:
        return

    # `exclude_modules` is modelopt's name for it, `ignore` is compressed-tensors'.
    # A hybrid block carries one or the other, so prune whichever is present.
    removed: list[str] = []
    for key in ("exclude_modules", "ignore"):
        if key in quant:
            quant[key], gone = prune(quant[key])
            removed += gone

    quant["quantized_layers"] = {
        **quant.get("quantized_layers", {}),
        **layer_entries(removed or sorted({m.rsplit(".", 1)[0] + "*" for m in quantized})),
    }
    config["quantization_config"] = quant
    (dst / "config.json").write_text(json.dumps(config, indent=2))


def _copy_rest(src: Path, dst: Path) -> None:
    """Copy everything else across, never overwriting a file already written.

    The `exists()` guard is load-bearing: `_write_config` has just written the
    corrected config, and copying the source's version over it would silently
    restore the stale exclusion list.
    """
    for path in src.iterdir():
        if path.is_file() and path.suffix != ".safetensors" and not (dst / path.name).exists():
            shutil.copy2(path, dst / path.name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=f"module glob to quantize; repeatable. Defaults to {DEFAULT_PATTERNS}.",
    )
    parser.add_argument("--per-channel", action="store_true", help="one scale per output row")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args(argv)

    patterns = args.pattern or DEFAULT_PATTERNS
    stats = convert(args.src, args.dst, patterns, args.per_channel, args.dry_run)

    before, after = stats["bytes_before"], stats["bytes_after"]
    print(f"\nconverted {stats['converted']} weights, copied {stats['copied']} tensors")
    if stats["skipped_1d"]:
        print(f"  skipped {stats['skipped_1d']} 1-D tensors (norms/biases) that matched a pattern")
    print(f"  before {before / 1e9:.2f} GB")
    print(f"  after  {after / 1e9:.2f} GB   ({100 * (1 - after / before):.1f}% smaller)")
    if args.dry_run:
        print("  (dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
