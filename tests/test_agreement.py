"""Tests for the quantization bar.

The point of these is that `agreement` must stay a *comparative* check. An
absolute threshold on KL or logit delta would be a number nobody can justify, and
the moment one exists somebody will tune it until their checkpoint passes. The
only defensible statement is "no worse than the quantized build already shipping",
so that is what is pinned here.
"""

import pytest

from unfuse.agreement import Agreement, PromptScore
from unfuse.verify import Comparison


def score(top1=1.0, kl=0.0, top5=1.0, max_abs=0.0, prompt="p"):
    return PromptScore(prompt, top1, top5, kl, max_abs)


class TestShippingCriterion:
    def test_closer_than_baseline_passes(self):
        ours = Agreement([score(top1=0.99, kl=1e-3)])
        theirs = Agreement([score(top1=0.97, kl=4e-3)])
        assert ours.is_no_worse_than(theirs)

    def test_equal_to_baseline_passes(self):
        # "No worse" includes matching it. Reproducing NVIDIA's accuracy while
        # being smaller is a win, not a failure.
        same = [score(top1=0.97, kl=4e-3)]
        assert Agreement(list(same)).is_no_worse_than(Agreement(list(same)))

    def test_worse_top1_fails_even_with_better_kl(self):
        # Both must improve. A model can shave average divergence while flipping
        # more greedy choices, and greedy choices are what a user sees.
        ours = Agreement([score(top1=0.90, kl=1e-4)])
        theirs = Agreement([score(top1=0.97, kl=4e-3)])
        assert not ours.is_no_worse_than(theirs)

    def test_worse_kl_fails_even_with_better_top1(self):
        ours = Agreement([score(top1=0.99, kl=9e-3)])
        theirs = Agreement([score(top1=0.97, kl=4e-3)])
        assert not ours.is_no_worse_than(theirs)

    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [([], [score()]), ([score()], []), ([], [])],
    )
    def test_empty_run_never_passes(self, ours, theirs):
        # Scoring zero prompts proves nothing. Same rule as Comparison.exact.
        assert not Agreement(ours).is_no_worse_than(Agreement(theirs))


class TestAggregation:
    def test_top1_is_the_worst_prompt_not_the_mean(self):
        # One broken capability among five healthy ones is exactly what this is
        # meant to catch; a mean would bury it.
        a = Agreement([score(top1=1.0), score(top1=1.0), score(top1=0.4)])
        assert a.top1 == 0.4

    def test_kl_is_the_worst_prompt(self):
        assert Agreement([score(kl=1e-5), score(kl=2e-2)]).kl == 2e-2

    def test_empty_aggregates_are_failing_values(self):
        empty = Agreement([])
        assert empty.top1 == 0.0
        assert empty.kl == float("inf")


class TestReport:
    def test_report_names_the_outcome(self):
        ours = Agreement([score(top1=0.99, kl=1e-3)], label="ours")
        theirs = Agreement([score(top1=0.97, kl=4e-3)], label="nvidia")
        assert "NO WORSE THAN BASELINE" in ours.report(theirs)
        # Not a substring check for "WORSE THAN BASELINE": that also matches the
        # passing message. The failing branch reports the increment instead.
        assert "ADDS" in theirs.report(ours)
        assert "NO WORSE" not in theirs.report(ours)

    def test_report_without_baseline_states_no_verdict(self):
        text = Agreement([score()]).report()
        assert "BASELINE" not in text


class TestTheTwoBarsStaySeparate:
    def test_exactness_still_rejects_tiny_differences(self):
        # Guards the boundary: adding a tolerance-based check for quantization
        # must not have loosened the extraction check.
        assert not Comparison([("a", 1e-12, True)]).exact


class TestMarginalCost:
    """For a change that adds quantization, the strict gate is the wrong question."""

    def test_ratio_of_one_means_no_added_divergence(self):
        same = Agreement([score(kl=4e-3, top1=0.97)])
        assert same.marginal_cost(Agreement([score(kl=4e-3, top1=0.97)]))["kl_ratio"] == 1.0

    def test_ratio_reports_the_increment(self):
        ours = Agreement([score(kl=4.4e-3, top1=0.96)])
        theirs = Agreement([score(kl=4.0e-3, top1=0.97)])
        cost = ours.marginal_cost(theirs)
        assert cost["kl_ratio"] == pytest.approx(1.1)
        assert cost["top1_drop"] == pytest.approx(0.01)

    def test_report_shows_the_increment_rather_than_a_bare_failure(self):
        # Adding FP8 attention on top of NVFP4 MLPs cannot pass the strict gate.
        # The report has to say how much it costs, not just "worse".
        ours = Agreement([score(kl=4.4e-3, top1=0.96)], label="ours")
        theirs = Agreement([score(kl=4.0e-3, top1=0.97)], label="nvidia")
        text = ours.report(theirs)
        assert "ADDS" in text and "1.10x" in text
