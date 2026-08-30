"""Tests for the startup-log scraper.

The numbers this pulls out end up in a model card as the headline result, and
they come from log lines that engines reword between releases. So the fixtures
below are verbatim lines from the runs the cards report, and a format change
that breaks a regex should fail here rather than silently ship a table of nulls.
"""

import importlib.util
import json
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "serve_stats", pathlib.Path(__file__).resolve().parents[1] / "tools" / "serve_stats.py"
)
serve_stats = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve_stats)


VLLM_LOG = """\
INFO 08-30 04:12:01 [api_server.py:1877] vLLM API server version 0.27.1
INFO 08-30 04:12:09 [core.py:99] Initializing a V1 LLM engine (v0.27.1) with config: \
model='zai-org/GLM-4.6V-Flash', max_model_len=32768, max_num_seqs=1024, dtype=torch.bfloat16
INFO 08-30 04:13:40 [gpu_model_runner.py:2653] Encoder cache will be initialized with a \
budget of 12800 tokens, and profiled with 1 image items of the maximum feature size.
INFO 08-30 04:13:52 [gpu_worker.py:298] Model loading took 19.1834 GiB and 61.204518 seconds
INFO 08-30 04:14:20 [gpu_worker.py:391] Available KV cache memory: 14.62 GiB
INFO 08-30 04:14:21 [kv_cache_utils.py:1229] GPU KV cache size: 383,424 tokens
INFO 08-30 04:14:21 [kv_cache_utils.py:1233] Maximum concurrency for 32,768 tokens per \
request: 11.70x
"""

SGLANG_LOG = """\
[2026-08-30 05:01:33] server_args=ServerArgs(model_path='sartajbhuvaji/GLM-4.6-Flash-text', \
context_len=32768, chunked_prefill_size=8192, max_running_requests=None)
[2026-08-30 05:02:10 TP0] Load weight end. type=Glm4ForCausalLM, dtype=torch.bfloat16, \
avail mem=20.15 GB, mem usage=17.51 GB.
[2026-08-30 05:02:12 TP0] KV Cache is allocated. dtype: torch.bfloat16, #tokens: 479280, \
K size: 4.57 GB, V size: 4.57 GB
[2026-08-30 05:02:12 TP0] max_total_num_tokens=479280, chunked_prefill_size=8192, \
max_prefill_tokens=16384, max_running_requests=2049, context_len=32768
"""


class TestEngineDetection:
    def test_detects_vllm(self):
        assert serve_stats.detect(VLLM_LOG) == "vllm"

    def test_detects_sglang(self):
        assert serve_stats.detect(SGLANG_LOG) == "sglang"

    def test_unknown_log_is_empty_not_a_guess(self):
        assert serve_stats.detect("Traceback (most recent call last):") == ""


VLLM_STATS, VLLM_MISSING = serve_stats.scrape(VLLM_LOG, serve_stats.VLLM)
SGLANG_STATS, _ = serve_stats.scrape(SGLANG_LOG, serve_stats.SGLANG)


class TestVllm:
    @pytest.fixture
    def stats(self):
        return VLLM_STATS

    def test_weights(self, stats):
        assert stats["weights_gib"] == 19.1834

    def test_kv_cache(self, stats):
        assert stats["kv_cache_gib"] == 14.62
        assert stats["kv_cache_tokens"] == 383_424

    def test_max_concurrency_is_read_not_derived(self, stats):
        assert stats["max_concurrency"] == 11.70
        assert stats["max_model_len"] == 32768

    def test_scheduler_limit(self, stats):
        assert stats["concurrency_limit"] == 1024

    def test_multimodal_encoder_budget_is_its_own_field(self, stats):
        # A vision profiling buffer is a cost of the code path, not of the
        # weights, so it must not be folded into the weight delta.
        assert stats["mm_encoder_budget_tokens"] == 12800

    def test_debug_level_default_is_read_as_the_limit(self):
        # With no --max-num-seqs the cap appears only at DEBUG, after clamping.
        debug = VLLM_LOG.replace("max_num_seqs=1024", "") + (
            "DEBUG 08-30 04:12:09 [arg_utils.py:2795] Defaulting max_num_seqs to 1024 "
            "for ENGINE_CONTEXT.OPENAI_API_SERVER usage context.\n"
        )
        s, _ = serve_stats.scrape(debug, serve_stats.VLLM)
        assert s["concurrency_limit"] == 1024

    def test_absent_limit_is_reported_missing_not_assumed(self):
        no_cap = VLLM_LOG.replace("max_num_seqs=1024", "")
        s, missing = serve_stats.scrape(no_cap, serve_stats.VLLM)
        assert "concurrency_limit" not in s
        assert "concurrency_limit" in missing

    def test_text_only_log_has_no_encoder_budget(self):
        text_only = VLLM_LOG.replace("Encoder cache", "Nothing here")
        s, missing = serve_stats.scrape(text_only, serve_stats.VLLM)
        assert "mm_encoder_budget_tokens" not in s
        assert "mm_encoder_budget_tokens" in missing


class TestSglang:
    @pytest.fixture
    def stats(self):
        return SGLANG_STATS

    def test_weights(self, stats):
        assert stats["weights_gib"] == 17.51

    def test_kv_tokens(self, stats):
        assert stats["kv_cache_tokens"] == 479_280

    def test_kv_size_is_k_plus_v(self, stats):
        assert stats["kv_k_gib"] + stats["kv_v_gib"] == pytest.approx(9.14)

    def test_scheduler_limit_takes_the_resolved_value_not_the_none_in_server_args(self, stats):
        # server_args prints max_running_requests=None before the scheduler picks
        # a number. Reading the first match would report None as the limit.
        assert stats["concurrency_limit"] == 2049


class TestMissingFieldsAreNamed:
    def test_a_log_that_matches_nothing_lists_every_field(self):
        s, missing = serve_stats.scrape("nothing useful here", serve_stats.VLLM)
        assert s == {}
        assert missing == [f for f, _, _ in serve_stats.VLLM]


_rp_spec = importlib.util.spec_from_file_location(
    "report_pair", pathlib.Path(__file__).resolve().parents[1] / "tools" / "report_pair.py"
)
report_pair = importlib.util.module_from_spec(_rp_spec)
_rp_spec.loader.exec_module(report_pair)


class TestKvArithmetic:
    """GLM-4.6V-Flash and its text-only split share an attention geometry:
    40 layers, 2 KV heads, head_dim 128, bf16 -> 40,960 bytes per token in both.
    So KV tokens must equal GiB * 2^30 / 40960, and a scraped pair that fails
    this is a misread log rather than a surprising result.
    """

    BYTES_PER_TOKEN = 40_960

    def test_published_kv_token_count_matches_its_cache_size(self):
        assert report_pair.gib_to_tokens(18.28, self.BYTES_PER_TOKEN) == pytest.approx(
            479_280, rel=0.002
        )

    def test_a_gib_figure_implies_its_token_count(self):
        assert report_pair.gib_to_tokens(14.62, self.BYTES_PER_TOKEN) == 383_255

    def test_delta_formatting_keeps_thousands_separators(self):
        assert report_pair.num(479280, 0) == "479,280"
        assert report_pair.num(14.62, 2) == "14.62"


class TestComparabilityVerdict:
    """Whether two benchmark rows may be compared is a three-way question, and
    collapsing it to == is what let a scheduler artifact reach a draft card.
    """

    def _dir(self, tmp_path, vllm_limits):
        for role, lim in vllm_limits.items():
            payload = {"engine": "vllm"}
            if lim is not None:
                payload["concurrency_limit"] = lim
            (tmp_path / f"serve-vllm-{role}.json").write_text(json.dumps(payload))
        return tmp_path

    def test_equal_limits_are_comparable(self, tmp_path):
        d = self._dir(tmp_path, {"original": 256, "derived": 256})
        assert "equal" in report_pair.limits(d, 16)

    def test_unequal_but_both_above_load_is_still_comparable(self, tmp_path):
        # SGLang gave GLM 3085 and 4096. Different, but nothing was throttled.
        d = self._dir(tmp_path, {"original": 3085, "derived": 4096})
        out = report_pair.limits(d, 16)
        assert "neither model is throttled" in out
        assert "STARVED" not in out

    def test_a_limit_at_or_below_the_load_is_starvation(self, tmp_path):
        # The Qwen case: SGLang gave the parent 5 against an offered 16.
        d = self._dir(tmp_path, {"original": 5, "derived": 16})
        assert "STARVED" in report_pair.limits(d, 16)

    def test_two_unknowns_are_unverified_not_a_match(self, tmp_path):
        d = self._dir(tmp_path, {"original": None, "derived": None})
        out = report_pair.limits(d, 16)
        assert "UNVERIFIED" in out
        assert "equal" not in out
