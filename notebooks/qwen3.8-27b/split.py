"""Remove the vision tower from Qwen/Qwen3.8-27B.

Mirrors notebooks/glm-4.6v-flash/remove_vision.ipynb, with three differences that
matter and are the reason this is a script and not a copy of that notebook.

1. The tower is small. 460,730,096 params of 27,781,427,952 -- 1.658%, 0.92 GB of
   55.56 GB. GLM-4.6V-Flash was 8.671%. This split is a packaging win (loads as a
   plain causal LM, no vision deps, no image preprocessing) and NOT a size win.
   Read docs/FINDINGS.md before assuming otherwise.

2. `deepstack_visual_indexes` in vision_config looks like it should complicate the
   split -- it names the tower layers whose features are injected into the LLM. It
   does not. Zero tensors carry a deepstack name; the indices are resolved inside
   the visual tree. Verified by reading every shard header: the only trees are
   model.visual.*, model.language_model.*, lm_head, and mtp.*.

3. There is a fused MTP speculative-decoding head (`mtp.*`, 15 tensors,
   ~424,694,784 params). transformers has no Qwen3_5 MTP class, so
   Qwen3_5ForCausalLM will report these as unexpected keys. Dropping them silently
   removes a capability the community demonstrably wants, so the default is to KEEP
   and warn. Pass --drop-mtp for a strictly clean load.

The model is a HYBRID: layer_types is 48 linear_attention + 16 full_attention
(full_attention_interval 4). Do not assume vanilla-transformer behaviour for KV
cache sizing or GGUF conversion.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

VISION_PREFIX = "model.visual."
TEXT_PREFIX = "model.language_model."
MTP_PREFIX = "mtp."

# Measured by reading the shard headers of Qwen/Qwen3.8-27B over HTTP range
# requests. Asserted after the split so a silent upstream re-upload cannot pass.
EXPECTED = {
    "total_tensors": 1199,
    "vision_tensors": 333,
    "mtp_tensors": 15,
    "vision_params": 460_730_096,
    "total_params": 27_781_427_952,
}


def numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def classify(name: str) -> str:
    if name.startswith(VISION_PREFIX):
        return "vision"
    if name.startswith(MTP_PREFIX):
        return "mtp"
    if name.startswith(TEXT_PREFIX) or name == "lm_head.weight":
        return "text"
    return "other"


def rename(name: str) -> str:
    if name.startswith(TEXT_PREFIX):
        return "model." + name[len(TEXT_PREFIX) :]
    return name


def build_config(src: Path, drop_mtp: bool) -> dict:
    raw = json.loads((src / "config.json").read_text())
    text = dict(raw["text_config"])

    # Qwen3_5TextConfig is the registered config for model_type qwen3_5_text, but the
    # architecture that consumes these weights standalone is Qwen3_5ForCausalLM. The
    # config class will not set `architectures` for us -- the same trap as Glm4Config
    # in the GLM split, where forgetting this produced a checkpoint that loaded as the
    # wrong class.
    text["architectures"] = ["Qwen3_5ForCausalLM"]
    text.setdefault("model_type", "qwen3_5_text")

    # Carried over from the multimodal parent, meaningless without a tower.
    for k in ("image_token_id", "video_token_id", "vision_start_token_id",
              "vision_end_token_id", "language_model_only"):
        text.pop(k, None)

    if drop_mtp:
        text.pop("mtp_num_hidden_layers", None)
        text.pop("mtp_use_dedicated_embeddings", None)

    text["transformers_version"] = raw.get("transformers_version")
    return text


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="local Qwen3.8-27B checkpoint")
    p.add_argument("dst", type=Path, help="output directory")
    p.add_argument("--drop-mtp", action="store_true",
                   help="also remove the fused MTP speculative head")
    p.add_argument("--dry-run", action="store_true",
                   help="report the plan from the index only; touch no weights")
    p.add_argument("--shard-bytes", type=int, default=2_000_000_000,
                   help="output shard size budget; also the peak RAM ceiling "
                        "(default 2e9, lower it on a memory-starved box)")
    args = p.parse_args()

    index_path = args.src / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    counts = Counter(classify(n) for n in weight_map)
    print(f"source tensors: {len(weight_map)}")
    for k in ("text", "vision", "mtp", "other"):
        print(f"  {k:<7} {counts[k]:>5}")
    if counts["other"]:
        raise SystemExit(
            f"refusing to run: {counts['other']} tensors match no known tree. "
            "The checkpoint layout changed; re-read the headers before splitting."
        )
    if counts["vision"] != EXPECTED["vision_tensors"]:
        raise SystemExit(
            f"vision tensor count is {counts['vision']}, expected "
            f"{EXPECTED['vision_tensors']}. Upstream checkpoint changed."
        )

    drop = {n for n in weight_map if classify(n) == "vision"}
    if args.drop_mtp:
        drop |= {n for n in weight_map if classify(n) == "mtp"}
    keep = [n for n in weight_map if n not in drop]
    print(f"\ndropping {len(drop)}, keeping {len(keep)}")

    if args.dry_run:
        print("dry run: no weights written")
        return 0

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    args.dst.mkdir(parents=True, exist_ok=True)
    shards = sorted({weight_map[n] for n in keep})
    new_map: dict[str, str] = {}
    total_params = 0
    total_bytes = 0

    # Memory-bounded writer. The obvious implementation mirrors the input sharding --
    # read shard N, write shard N -- but that holds a whole shard in RAM, and Qwen's are
    # ~3 GB. On a machine with less free RAM than that it OOMs partway through and leaves
    # a half-written checkpoint that still looks plausible on disk. So we buffer by BYTES
    # and flush whenever the budget is hit, regardless of where input shard boundaries
    # fall. Output sharding therefore does NOT match input sharding. That is fine: the
    # index is what maps names to files.
    budget = args.shard_bytes
    buf: dict = {}
    buf_bytes = 0
    out_idx = 0

    def flush():
        nonlocal buf, buf_bytes, out_idx
        if not buf:
            return
        out_idx += 1
        name = f"model-{out_idx:05d}.safetensors"
        save_file(buf, args.dst / name, metadata={"format": "pt"})
        for k in buf:
            new_map[k] = name
        print(f"  {name}: {len(buf)} tensors, {buf_bytes / 1e9:.2f} GB")
        buf = {}
        buf_bytes = 0

    for shard in shards:
        with safe_open(args.src / shard, framework="pt") as f:
            for name in f.keys():
                if name in drop or name not in weight_map:
                    continue
                t_ = f.get_tensor(name)
                nbytes = t_.numel() * t_.element_size()
                if buf_bytes + nbytes > budget and buf:
                    flush()
                buf[rename(name)] = t_
                buf_bytes += nbytes
                total_params += t_.numel()
                total_bytes += nbytes
    flush()

    # Rename to the conventional -of- form now that the shard count is known.
    for k in range(1, out_idx + 1):
        src_name = f"model-{k:05d}.safetensors"
        dst_name = f"model-{k:05d}-of-{out_idx:05d}.safetensors"
        (args.dst / src_name).rename(args.dst / dst_name)
        for tname, fname in new_map.items():
            if fname == src_name:
                new_map[tname] = dst_name

    json.dump(
        {"metadata": {"total_size": total_bytes}, "weight_map": new_map},
        (args.dst / "model.safetensors.index.json").open("w"),
        indent=2,
    )

    cfg = build_config(args.src, args.drop_mtp)
    (args.dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for f in args.src.iterdir():
        if f.name.startswith(("tokenizer", "vocab", "merges", "chat_template")) or \
           f.name in ("generation_config.json", "special_tokens_map.json"):
            shutil.copy2(f, args.dst / f.name)

    expected_keep = EXPECTED["total_params"] - EXPECTED["vision_params"]
    if args.drop_mtp:
        expected_keep -= 424_694_784
    print(f"\nkept params: {total_params:,}  expected: {expected_keep:,}")
    if total_params != expected_keep:
        raise SystemExit("PARAM COUNT MISMATCH -- do not publish this checkpoint")
    print(f"size: {total_bytes / 1e9:.2f} GB")
    print("\nnext: verify against the parent before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
