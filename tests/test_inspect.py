"""Tests for the model-agnostic core.

These run without weights, without a GPU and without network, which is what makes
them worth having in CI. The parts that need a 20 GB checkpoint are exercised by
the notebooks instead, and their outputs are committed as the record.
"""

import json
from typing import ClassVar

import pytest

from unfuse.inspect import (
    Inventory,
    TensorInfo,
    _quant_config,
    group_by_prefix,
    module_of,
    observed_state,
    read_inventory,
    read_quant_config,
    shards_for_prefix,
    split_by_prefix,
)


def tensor(name, shape=(2, 2), dtype="BF16"):
    return TensorInfo(name, shape, dtype)


class TestTensorInfo:
    def test_params_is_product_of_shape(self):
        assert tensor("a", (1536, 1536)).params == 2_359_296

    def test_scalar_has_one_parameter(self):
        assert tensor("a", ()).params == 1

    def test_bytes_follow_dtype(self):
        assert tensor("a", (10,), "F32").nbytes == 40
        assert tensor("a", (10,), "BF16").nbytes == 20
        assert tensor("a", (10,), "I8").nbytes == 10

    def test_unknown_dtype_assumes_two_bytes(self):
        # Better to guess 16-bit than to crash on a format we have not met.
        assert tensor("a", (10,), "WHAT").nbytes == 20


class TestSplitByPrefix:
    def test_partitions_without_losing_tensors(self):
        inv = Inventory(
            [tensor("model.visual.a"), tensor("model.layers.0.b"), tensor("model.layers.1.c")]
        )
        vision, rest = split_by_prefix(inv, "model.visual.")
        assert len(vision) == 1
        assert len(rest) == 2
        assert vision.params + rest.params == inv.params

    def test_no_match_gives_empty_and_everything(self):
        inv = Inventory([tensor("model.layers.0.a")])
        matched, rest = split_by_prefix(inv, "model.visual.")
        assert len(matched) == 0
        assert len(rest) == 1

    def test_prefix_is_not_a_substring_match(self):
        # "model.visualiser" must not be caught by the "model.visual." prefix.
        inv = Inventory([tensor("model.visualiser.a")])
        matched, _ = split_by_prefix(inv, "model.visual.")
        assert len(matched) == 0


class TestGroupByPrefix:
    def test_collapses_layer_indices(self):
        inv = Inventory([tensor(f"model.layers.{i}.mlp") for i in range(40)])
        groups = group_by_prefix(inv, depth=3)
        assert list(groups) == ["model.layers.*"]
        assert len(groups["model.layers.*"]) == 40

    def test_depth_controls_granularity(self):
        inv = Inventory(
            [tensor("model.visual.blocks.0.attn"), tensor("model.language_model.layers.0.attn")]
        )
        assert len(group_by_prefix(inv, depth=1)) == 1
        assert len(group_by_prefix(inv, depth=2)) == 2

    def test_totals_are_preserved(self):
        inv = Inventory([tensor("a.b", (4, 4)), tensor("c.d", (2, 2))])
        assert sum(g.params for g in group_by_prefix(inv).values()) == inv.params


class TestShardsForPrefix:
    def test_finds_the_shards_holding_a_prefix(self):
        # The real GLM-4.6V-Flash shape: all vision tensors in one shard of four,
        # which is what turns a 20.6 GB download into 4.7 GB.
        index = {
            "weight_map": {
                "model.visual.a": "s4.safetensors",
                "model.visual.b": "s4.safetensors",
                "model.language_model.x": "s1.safetensors",
                "model.language_model.y": "s2.safetensors",
            }
        }
        assert shards_for_prefix(index, "model.visual.") == ["s4.safetensors"]

    def test_detects_a_prefix_spread_across_shards(self):
        index = {"weight_map": {"model.visual.a": "s1", "model.visual.b": "s2"}}
        assert shards_for_prefix(index, "model.visual.") == ["s1", "s2"]


class TestReadInventory:
    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_inventory(tmp_path)


class TestInventory:
    def test_empty_inventory_is_zero_not_an_error(self):
        inv = Inventory([])
        assert inv.params == 0 and inv.nbytes == 0 and len(inv) == 0


class TestQuantConfig:
    def test_exclusion_globs_cross_dots(self):
        # "...self_attn*" has to cover every projection inside the block, which
        # only works if * matches across dots. fnmatch does; a naive split would not.
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": ["*.self_attn*"]})
        assert qc.state_for("model.layers.7.self_attn.q_proj") == "BF16"
        assert qc.state_for("model.layers.7.mlp.up_proj") == "NVFP4"

    def test_unexcluded_modules_carry_the_global_algorithm(self):
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": []})
        assert qc.state_for("anything") == "NVFP4"

    def test_no_algorithm_means_bf16(self):
        assert _quant_config({}).state_for("anything") == "BF16"

    def test_reads_legacy_file(self, tmp_path):
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "producer": {"name": "modelopt"},
                    "quantization": {
                        "quant_algo": "NVFP4",
                        "kv_cache_quant_algo": "FP8",
                        "group_size": 16,
                        "exclude_modules": ["lm_head"],
                    },
                }
            )
        )
        qc = read_quant_config(tmp_path)
        assert (qc.algo, qc.kv_cache_algo, qc.group_size) == ("NVFP4", "FP8", 16)
        assert qc.producer == "modelopt"

    def test_config_json_wins_over_the_legacy_file(self, tmp_path):
        # The current format carries per-layer algorithms the legacy one cannot
        # express, so a checkpoint with both must be read from config.json.
        (tmp_path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_algo": "FP8"}})
        )
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "NVFP4"}})
        )
        assert read_quant_config(tmp_path).algo == "FP8"

    def test_unquantized_checkpoint_returns_none(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        assert read_quant_config(tmp_path) is None


class TestModuleOf:
    def test_strips_the_weight_suffix(self):
        assert module_of("model.layers.0.mlp.up_proj.weight") == "model.layers.0.mlp.up_proj"

    def test_strips_scale_suffixes_not_just_weight(self):
        # A scale tensor belongs to the module it scales. Getting this wrong makes
        # every scale look like an unmatched module and silently skews the report.
        for suffix in (".weight_scale", ".weight_scale_2", ".input_scale"):
            assert module_of(f"m.l.0.self_attn.q_proj{suffix}") == "m.l.0.self_attn.q_proj"


class TestObservedState:
    def test_dtype_overrides_a_config_claim(self, tmp_path):
        # The tied-lm_head case: the exclusion list names `lm_head`, but the
        # tensor ships as `embed_tokens`, so name matching alone calls it
        # quantized while the header says BF16. The header wins.
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": ["lm_head"]})
        tied = TensorInfo("model.embed_tokens.weight", (262144, 5376), "BF16")
        assert qc.state_for("model.embed_tokens") == "NVFP4"
        assert observed_state(tied, qc) == "BF16"

    def test_fp8_storage_is_reported_as_fp8(self):
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": []})
        t = TensorInfo("m.l.0.self_attn.q_proj.weight", (64, 64), "F8_E4M3")
        assert observed_state(t, qc) == "FP8"

    def test_packed_weights_keep_the_config_algorithm(self):
        # NVFP4 has no safetensors dtype -- it ships packed in U8, so the header
        # cannot identify it and the config is the only source.
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": []})
        t = TensorInfo("m.l.0.mlp.up_proj.weight", (128, 32), "U8")
        assert observed_state(t, qc) == "NVFP4"

    def test_scale_tensors_do_not_vote_as_unquantized(self):
        # A float32 scale sits beside a quantized weight. Counting it as BF16
        # would make every quantized group report "BF16+NVFP4".
        qc = _quant_config({"quant_algo": "NVFP4", "exclude_modules": []})
        scale = TensorInfo("m.l.0.mlp.up_proj.weight_scale", (128, 4), "F32")
        assert observed_state(scale, qc) == "NVFP4"


class TestHybridQuantSchema:
    """Gemma-4-31B-IT-NVFP4 carries modelopt and compressed-tensors keys at once."""

    HYBRID: ClassVar[dict] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "producer": {"name": "modelopt", "version": "0.37.0"},
        "kv_cache_scheme": {"dynamic": False, "num_bits": 8, "type": "float"},
        "config_groups": {
            "group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"],
            }
        },
        "ignore": ["lm_head", "model.language_model.layers.0.self_attn*"],
    }

    def test_group_size_comes_from_the_nested_group(self):
        # Not at the top level in this schema. Reading only the modelopt keys
        # returns None and the report silently loses it.
        assert _quant_config(self.HYBRID).group_size == 16

    def test_kv_cache_scheme_is_translated(self):
        # `kv_cache_scheme: {num_bits: 8, type: float}`, not `kv_cache_quant_algo`.
        assert _quant_config(self.HYBRID).kv_cache_algo == "FP8"

    def test_ignore_is_read_as_the_exclusion_list(self):
        assert len(_quant_config(self.HYBRID).exclude_modules) == 2
        assert _quant_config(self.HYBRID).state_for("lm_head") == "BF16"

    def test_producer_dict_is_flattened_to_a_name(self):
        assert _quant_config(self.HYBRID).producer == "modelopt"

    def test_pure_compressed_tensors_derives_the_algorithm(self):
        # No quant_algo key at all: the format has to come from the group's bits.
        pure = {
            "quant_method": "compressed-tensors",
            "config_groups": {"group_0": {"weights": {"num_bits": 8, "type": "float"}}},
            "ignore": [],
        }
        assert _quant_config(pure).algo == "FP8"
