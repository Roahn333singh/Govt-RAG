"""Paired eval stats used by --compare-to."""

from eval.stats import mcnemar, wilcoxon_signed_rank


def test_wilcoxon_and_mcnemar_smoke():
    diffs = [0.1, -0.2, 0.3, -0.05, 0.0, 0.4, -0.1, 0.2, -0.3, 0.15, 0.05]
    result = wilcoxon_signed_rank(diffs)
    assert result["n"] == 10
    assert result["p_value"] is not None
    assert 0.0 <= result["p_value"] <= 1.0

    hit = mcnemar(5, 1)
    assert hit["n_discordant"] == 6
    assert 0.0 < hit["p_value"] < 1.0
    assert mcnemar(0, 0)["p_value"] == 1.0
