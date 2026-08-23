#!/usr/bin/env python3
"""Benchmark any OpenAI-compatible endpoint. One harness for every engine.

Each engine ships its own benchmarker (`vllm bench serve`, `sglang.bench_serving`)
and they do not agree: different prompt sampling, different warmup, different
definition of "duration". Numbers from two different tools cannot be put in the
same table. This hits /v1/chat/completions over plain HTTP, so vLLM and SGLang
are measured by identical code and a gap between them is a real gap.

TTFT needs streaming. A non-streamed request only tells you when the whole
response landed, so prefill and decode collapse into one number and TTFT is
unmeasurable. Hence stream=true and SSE parsing.

`ignore_eos` pins every request to exactly --output-len tokens. Without it a
reasoning model decides its own lengths, throughput becomes a function of how
chatty it felt, and runs stop being comparable.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request


def stream_once(url, model, prompt, out_len, timeout=600):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": out_len,
        "temperature": 0,
        "stream": True,
        "ignore_eos": True,  # vLLM + SGLang both honour this; harmless elsewhere
    }
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft, n, last = None, 0, t0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            # Reasoning models stream thinking into a separate key before any
            # content appears. Counting only `content` would report a TTFT of
            # several seconds for a model that started emitting immediately.
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                last = time.perf_counter()
    return {"ttft": ttft, "tokens": n, "total": last - t0}


def run(url, model, prompt, out_len, n_req, concurrency):
    results, lock = [], threading.Lock()
    pending = list(range(n_req))
    idx = threading.Lock()

    def worker():
        while True:
            with idx:
                if not pending:
                    return
                pending.pop()
            try:
                r = stream_once(url, model, prompt, out_len)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
            with lock:
                results.append(r)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--label", default="")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--num-prompts", type=int, default=64)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16])
    p.add_argument("--json-out", default="")
    a = p.parse_args()

    base = f"http://{a.host}:{a.port}"
    model = json.load(urllib.request.urlopen(f"{base}/v1/models"))["data"][0]["id"]
    url = f"{base}/v1/chat/completions"
    # ~4 chars/token is close enough for GLM/Llama-family BPE at this length.
    prompt = (
        "Summarise the following text. "
        + "the quick brown fox jumps over the lazy dog. " * (a.input_len // 10)
    )[: a.input_len * 4]

    stream_once(url, model, "warmup", 8)  # first request pays graph capture, not the benchmark

    out = {"label": a.label, "model": model, "runs": []}
    for c in a.concurrency:
        n = max(c, a.num_prompts if c > 1 else 8)
        res, dur = run(url, model, prompt, a.output_len, n, c)
        errs = [r for r in res if "error" in r]
        good = [r for r in res if "error" not in r and r["ttft"] is not None]
        if not good:
            print(f"  c={c}: ALL FAILED  {errs[0]['error'] if errs else ''}", file=sys.stderr)
            continue
        ttfts = sorted(r["ttft"] * 1000 for r in good)
        toks = sum(r["tokens"] for r in good)
        # TPOT excludes the first token: (total - ttft) / (tokens - 1).
        tpots = [(r["total"] - r["ttft"]) * 1000 / max(r["tokens"] - 1, 1) for r in good]
        row = {
            "concurrency": c,
            "requests": len(good),
            "failed": len(errs),
            "duration_s": round(dur, 2),
            "output_tok_s": round(toks / dur, 1),
            "per_stream_tok_s": round(toks / dur / c, 1),
            "ttft_ms_mean": round(statistics.mean(ttfts), 1),
            "ttft_ms_p50": round(statistics.median(ttfts), 1),
            "ttft_ms_p99": round(ttfts[int(len(ttfts) * 0.99) - 1], 1),
            "tpot_ms_mean": round(statistics.mean(tpots), 2),
        }
        out["runs"].append(row)
        print(
            f"  c={c:<3} {row['output_tok_s']:>8.1f} out tok/s | "
            f"TTFT p50 {row['ttft_ms_p50']:>7.1f} ms | TPOT {row['tpot_ms_mean']:>6.2f} ms"
            + (f" | {len(errs)} FAILED" if errs else "")
        )

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
