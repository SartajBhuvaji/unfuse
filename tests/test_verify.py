"""Tests for the comparison logic.

The rule that matters is that only an exact match passes. These pin that down so
nobody later "fixes" it into a tolerance-based check.
"""

from unfuse.verify import DEFAULT_PROMPTS, Comparison, long_prompt


class TestComparison:
    def test_zero_difference_is_exact(self):
        assert Comparison([("a", 0.0, True), ("b", 0.0, True)]).exact

    def test_tiny_difference_still_fails(self):
        # The dangerous case: small enough to look like rounding, large enough to
        # mean the computation changed. It must not pass.
        assert not Comparison([("a", 1e-12, True)]).exact

    def test_empty_comparison_is_not_a_pass(self):
        # Running zero prompts proves nothing, so it must not report success.
        assert not Comparison([]).exact

    def test_worst_is_the_largest_delta(self):
        assert Comparison([("a", 1e-6, True), ("b", 3e-4, False)]).worst == 3e-4

    def test_report_names_the_outcome(self):
        assert "EXACT MATCH" in Comparison([("a", 0.0, True)]).report()
        assert "DIVERGENCE" in Comparison([("a", 0.5, False)]).report()


class TestPrompts:
    def test_defaults_cover_different_abilities(self):
        # Facts, code, reasoning, sequences, translation. A narrow set could miss
        # damage to one capability.
        assert len(DEFAULT_PROMPTS) >= 5

    def test_long_prompt_is_actually_long(self):
        assert len(long_prompt().split()) > 500

    def test_long_prompt_length_scales(self):
        assert len(long_prompt(200)) > len(long_prompt(50))
