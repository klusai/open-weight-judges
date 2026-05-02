"""Unit tests for agreement metrics with known inputs."""

import pytest
import numpy as np

from judges.agreement import krippendorff_alpha, pairwise_weighted_kappa, agreement_summary


class TestKrippendorffAlpha:
    def test_perfect_agreement(self):
        scores_by_judge = {
            "a": [{"item_id": str(i), "scores": {"dim": i}} for i in range(1, 6)],
            "b": [{"item_id": str(i), "scores": {"dim": i}} for i in range(1, 6)],
            "c": [{"item_id": str(i), "scores": {"dim": i}} for i in range(1, 6)],
        }
        alpha = krippendorff_alpha(scores_by_judge, "dim")
        assert alpha > 0.99

    def test_no_agreement(self):
        scores_by_judge = {
            "a": [{"item_id": "1", "scores": {"dim": 1}}, {"item_id": "2", "scores": {"dim": 5}}],
            "b": [{"item_id": "1", "scores": {"dim": 5}}, {"item_id": "2", "scores": {"dim": 1}}],
        }
        alpha = krippendorff_alpha(scores_by_judge, "dim")
        assert alpha < 0.1

    def test_empty_input(self):
        alpha = krippendorff_alpha({}, "dim")
        assert np.isnan(alpha)


class TestPairwiseKappa:
    def test_perfect_agreement(self):
        items_a = [{"item_id": str(i), "scores": {"dim": i}} for i in range(1, 11)]
        items_b = [{"item_id": str(i), "scores": {"dim": i}} for i in range(1, 11)]
        kappa = pairwise_weighted_kappa(items_a, items_b, "dim")
        assert kappa > 0.99

    def test_handles_nested_scores(self):
        items_a = [{"item_id": "1", "scores": {"dim": {"score": 3}}}]
        items_b = [{"item_id": "1", "scores": {"dim": {"score": 3}}}]
        kappa = pairwise_weighted_kappa(items_a, items_b, "dim")
        assert np.isnan(kappa)  # too few items for kappa


class TestAgreementSummary:
    def test_returns_dataframe_with_expected_columns(self):
        scores_by_judge = {
            "a": [{"item_id": str(i), "scores": {"g": i, "c": i + 1}} for i in range(1, 6)],
            "b": [{"item_id": str(i), "scores": {"g": i, "c": i + 1}} for i in range(1, 6)],
        }
        df = agreement_summary(scores_by_judge, ["g", "c"])
        assert "dimension" in df.columns
        assert "krippendorff_alpha" in df.columns
        assert "mean_weighted_kappa" in df.columns
        assert len(df) == 2
