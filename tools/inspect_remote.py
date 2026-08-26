"""Answer "is this model worth splitting?" without downloading it.

`unfuse inspect` needs a local checkpoint. For a 55 GB model that means pulling 55 GB
before you can find out the vision tower is 1.7% and not worth removing. This reads the
same numbers off the Hub in about a minute and roughly 200 KB of traffic.

How it works
------------
A safetensors file begins with a u64 little-endian header length, then that many bytes of
JSON naming every tensor with its dtype and shape. Two HTTP range requests per shard get
the whole map. No weights are transferred.

Usage
-----
    python tools/inspect_remote.py Qwen/Qwen3.8-27B
    python tools/inspect_remote.py Qwen/Qwen3.8-27B --prefix model.visual.
    python tools/inspect_remote.py org/model --json          # machine readable

Exit code is 0 always -- this is a reconnaissance tool, not a gate.
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
import urllib.error
import urllib.request

DTYPE_BYTES = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "F4": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}

# Prefixes that usually mean "not the language model". Order matters: the first match wins.
KNOWN_GROUPS = [
    ("vision", ("model.visual.", "visual.", "vision_tower.", "model.vision_tower.",
                "vision_model.", "model.vision_model.")),
    ("audio", ("audio_tower.", "model.audio_tower.")),
    ("projector", ("multi_modal_projector.", "model.multi_modal_projector.", "mm_projector.")),
    ("mtp/draft", ("mtp.", "model.mtp.", "draft.")),
]


def fetch(url: str, rng: tuple[int, int] | None = None, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "unfuse-inspect-remote"})
    if rng:
        req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
    return urllib.request.urlopen(req, timeout=timeout).read()


def numel(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def classify(name: str) -> str:
    for label, prefixes in KNOWN_GROUPS:
        if name.startswith(prefixes):
            return label
    return "language model"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="Hub id, e.g. Qwen/Qwen3.8-27B")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--prefix", help="also report this exact prefix as its own group")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args(argv)

    base = f"https://huggingface.co/{args.repo}/resolve/{args.revision}"

    try:
        cfg = json.loads(fetch(f"{base}/config.json"))
    except urllib.error.HTTPError as e:
        print(f"could not read config.json: {e}", file=sys.stderr)
        return 0

    try:
        index = json.loads(fetch(f"{base}/model.safetensors.index.json"))
        shards = sorted(set(index["weight_map"].values()))
    except urllib.error.HTTPError:
        shards = ["model.safetensors"]  # single-file checkpoint

    tensors: dict[str, dict] = {}
    for shard in shards:
        url = f"{base}/{shard}"
        try:
            n = struct.unpack("<Q", fetch(url, (0, 7)))[0]
            hdr = json.loads(fetch(url, (8, 8 + n - 1)))
        except urllib.error.HTTPError as e:
            print(f"  {shard}: unreadable ({e})", file=sys.stderr)
            continue
        hdr.pop("__metadata__", None)
        tensors.update(hdr)
        if not args.json:
            print(f"  read {shard} header ({n/1024:.0f} KiB, {len(hdr)} tensors)",
                  file=sys.stderr)

    if not tensors:
        print("no tensor headers could be read", file=sys.stderr)
        return 0

    params = collections.Counter()
    nbytes = collections.Counter()
    counts = collections.Counter()
    for name, meta in tensors.items():
        g = args.prefix if (args.prefix and name.startswith(args.prefix)) else classify(name)
        n = numel(meta["shape"])
        params[g] += n
        nbytes[g] += n * DTYPE_BYTES.get(meta["dtype"], 2)
        counts[g] += 1

    total = sum(params.values())
    total_b = sum(nbytes.values())
    # "Removable" for a text-only split means the encoder stack, NOT the MTP/draft head.
    # A fused drafter is part of the language model's serving story -- deleting it is a
    # separate decision with its own tradeoff, so it is reported but never counted here.
    encoder = ("vision", "audio", "projector")
    removable = sum(v for k, v in params.items() if k in encoder)
    removable_b = sum(v for k, v in nbytes.items() if k in encoder)
    draft = sum(v for k, v in params.items() if k == "mtp/draft")

    if args.json:
        print(json.dumps({
            "repo": args.repo,
            "model_type": cfg.get("model_type"),
            "architectures": cfg.get("architectures"),
            "total_params": total,
            "total_bytes": total_b,
            "removable_params": removable,
            "removable_share": round(100 * removable / total, 4) if total else 0,
            "groups": {k: {"tensors": counts[k], "params": params[k], "bytes": nbytes[k],
                           "share_pct": round(100 * params[k] / total, 4)}
                       for k in params},
        }, indent=2))
        return 0

    print(f"\n{args.repo}  ({cfg.get('model_type')}, {cfg.get('architectures')})")
    print(f"{len(tensors)} tensors across {len(shards)} shard(s)\n")
    print(f"{'group':<20}{'tensors':>9}{'params':>18}{'share':>9}{'GB':>9}")
    for g, n in params.most_common():
        print(f"{g:<20}{counts[g]:>9}{n:>18,}{100*n/total:>8.3f}%{nbytes[g]/1e9:>9.2f}")
    print(f"{'TOTAL':<20}{len(tensors):>9}{total:>18,}{100.0:>8.3f}%{total_b/1e9:>9.2f}")

    if removable:
        pct = 100 * removable / total
        print(f"\nencoder stack (removable for a text-only split): "
              f"{removable:,} params = {pct:.3f}%, {removable_b/1e9:.2f} GB")
        if pct < 3:
            print("VERDICT: small. Expect packaging and cache headroom, not a size win.")
        elif pct < 8:
            print("VERDICT: moderate. Worth it if you also want a plain causal-LM config.")
        else:
            print("VERDICT: worth splitting on size alone.")
    else:
        print("\nno encoder tensors found — nothing to split")
    if draft:
        print(f"note: a fused draft/MTP head of {draft:,} params "
              f"({100*draft/total:.3f}%) is present; keeping it is the default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
