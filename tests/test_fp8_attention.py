"""Tests for the FP8 requantization tool.

The bug these exist for: `*self_attn*` also matches `q_norm` and `k_norm`, and an
early run quantized all 120 of them. Norms are 1-D RMSNorm scales -- no size to
win, and among the most precision-sensitive weights in the model.
"""

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from fp8_attention import (
    DEFAULT_PATTERNS,
    FP8_MAX,
    _copy_rest,
    _write_config,
    quantize_tensor,
    targets,
)

ATTENTION = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
]
NORMS = ["model.layers.0.self_attn.q_norm.weight", "model.layers.0.self_attn.k_norm.weight"]
OTHER = ["model.layers.0.mlp.up_proj.weight", "model.embed_tokens.weight"]


class TestTargeting:
    def test_projections_are_selected(self):
        assert set(targets(ATTENTION, DEFAULT_PATTERNS)) == set(ATTENTION)

    def test_norms_are_not_selected(self):
        assert targets(NORMS, DEFAULT_PATTERNS) == []

    def test_mlp_and_embeddings_are_untouched(self):
        assert targets(OTHER, DEFAULT_PATTERNS) == []

    def test_patterns_match_the_module_not_the_tensor(self):
        # Matching the full tensor name means every pattern needs a `.weight`
        # suffix to hit anything, and one that lacks it silently converts nothing.
        assert targets(ATTENTION, ["*self_attn.q_proj"]) == [ATTENTION[0]]

    def test_scale_tensors_are_never_targeted(self):
        assert targets(["model.layers.0.self_attn.q_proj.weight_scale"], DEFAULT_PATTERNS) == []


class TestQuantizeTensor:
    def test_round_trip_stays_close(self):
        torch = pytest.importorskip("torch")
        w = torch.randn(64, 128, dtype=torch.bfloat16) * 3
        q, s = quantize_tensor(w, per_channel=False)
        assert q.dtype == torch.float8_e4m3fn
        rel = (q.to(torch.float32) * s - w.float()).abs().max() / w.abs().max()
        assert rel < 0.10

    def test_per_channel_gives_one_scale_per_row(self):
        torch = pytest.importorskip("torch")
        _q, s = quantize_tensor(torch.randn(64, 128, dtype=torch.bfloat16), per_channel=True)
        assert s.shape == (64,)

    def test_largest_magnitude_saturates_the_format(self):
        # The scale is chosen so the biggest weight lands exactly at the top of
        # e4m3's range. If it landed short, the format's range is being wasted.
        torch = pytest.importorskip("torch")
        w = torch.randn(32, 32, dtype=torch.bfloat16)
        q, _s = quantize_tensor(w, per_channel=False)
        assert (q.to(torch.float32).abs().max() - FP8_MAX).abs() < 1e-3


class TestConfigRewrite:
    """The shipped config must describe the shipped weights.

    The bug: `_write_config` only looked for `hf_quant_config.json`. A checkpoint
    that has been through an extraction has only `config.json`, so it returned
    early, `_copy_rest` copied the stale config through, and the result claimed
    attention was still excluded from quantization while shipping FP8 tensors.
    """

    QUANTIZED: ClassVar[list[str]] = [
        "model.layers.0.self_attn.q_proj",
        "model.layers.1.self_attn.k_proj",
    ]

    def _config(self, tmp_path, block):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "config.json").write_text(json.dumps({"quantization_config": block}))
        _write_config(src, dst, self.QUANTIZED, per_channel=False)
        return json.loads((dst / "config.json").read_text())["quantization_config"]

    def test_compressed_tensors_ignore_is_pruned(self, tmp_path):
        out = self._config(
            tmp_path,
            {
                "quant_algo": "NVFP4",
                "ignore": ["lm_head", "model.layers.0.self_attn*", "model.layers.1.self_attn*"],
            },
        )
        assert out["ignore"] == ["lm_head"]

    def test_quantized_layers_records_what_moved(self, tmp_path):
        out = self._config(
            tmp_path,
            {
                "quant_algo": "NVFP4",
                "ignore": ["lm_head", "model.layers.0.self_attn*", "model.layers.1.self_attn*"],
            },
        )
        assert len(out["quantized_layers"]) == 2
        assert all(v["quant_algo"] == "FP8" for v in out["quantized_layers"].values())

    def test_modelopt_exclude_modules_is_pruned_too(self, tmp_path):
        out = self._config(
            tmp_path,
            {
                "quant_algo": "NVFP4",
                "exclude_modules": ["lm_head", "model.layers.0.self_attn*"],
            },
        )
        assert out["exclude_modules"] == ["lm_head"]

    def test_works_without_the_legacy_file(self, tmp_path):
        # The whole point: config.json alone must be enough.
        out = self._config(tmp_path, {"quant_algo": "NVFP4", "ignore": ["lm_head"]})
        assert "quantized_layers" in out

    def test_copy_rest_never_overwrites_the_written_config(self, tmp_path):
        src, dst = tmp_path / "src", tmp_path / "dst"
        src.mkdir()
        dst.mkdir()
        (src / "config.json").write_text(json.dumps({"stale": True}))
        (dst / "config.json").write_text(json.dumps({"corrected": True}))
        _copy_rest(src, dst)
        assert json.loads((dst / "config.json").read_text()) == {"corrected": True}
