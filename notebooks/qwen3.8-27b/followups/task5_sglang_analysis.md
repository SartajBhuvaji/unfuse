# Task 5 — SGLang parent throughput (read-only; no new runs)

## What the bench asked for

At `conc-c32`, both engines were pinned to `--max-running-requests 32`, yet SGLang
parent warm aggregate was **135.6 tok/s** vs split **346 tok/s** (`bench-sglang-*-warm.json`).

## What the serve JSON says vs the log

`serve-sglang-original.json` under `conc-c32/` reports `concurrency_limit: 32` because
`serve_stats.py` records the CLI flag. The **startup log overrides it**:

```
max_running_requests is capped to 5 by the mamba state cache
(max_mamba_cache_size=28, 5 state slots per request)
```

Split at the same sweep:

```
max_running_requests is capped to 16 by the mamba state cache
(max_mamba_cache_size=48, 3 state slots per request)
```

So at client concurrency 32, the **parent never runs more than 5 requests**; the split
caps at **16**. Aggregate tok/s is not comparable at equal client concurrency.

## Other parent costs (baseline-512, unpinned)

Parent log also shows **multimodal processor worker threads** and lower
`max_total_num_tokens` (**79998** vs split **129884**), reflecting vision-side memory
reservation on `Qwen/Qwen3.8-27B` even for text-only prompts.

## Warm JSON symptom

Parent `bench-sglang-original-warm.json` at c=32: **per_stream_tok_s 4.2**, **ttft_ms_mean ~35s** —
consistent with five long prefills queued behind a small running cap, not a 32-wide scheduler.

## Conclusion

The ~135 tok/s parent number is **scheduler-limited by Mamba state cache slots (5)**, not
the pinned CLI limit of 32. Fixing comparability requires raising mamba cache (`--mamba-full-memory-ratio` / `--max-mamba-cache-size`) or pinning client concurrency to **5** on both models.
