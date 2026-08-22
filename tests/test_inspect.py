"""Tests for the model-agnostic core.

These run without weights, without a GPU and without network, which is what makes
them worth having in CI. The parts that need a 20 GB checkpoint are exercised by
the notebooks instead, and their outputs are committed as the record.
"""

import pytest

from unfuse.inspect import (
    Inventory,
    TensorInfo,
    group_by_prefix,
    read_inventory,
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
