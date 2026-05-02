"""Unit tests for aggregation with known inputs."""

import pytest
import numpy as np

from judges.aggregation import median_aggregate, win_rate


class TestMedianAggregate:
    def test_simple_median(self):
        scores_by_judge = {
            "judge_a": [
                {"item_id": "1", "scores": {"grammar": 8, "creativity": 6}},
                {"item_id": "2", "scores": {"grammar": 4, "creativity": 9}},
            ],
            "judge_b": [
                {"item_id": "1", "scores": {"grammar": 6, "creativity": 7}},
                {"item_id": "2", "scores": {"grammar": 5, "creativity": 8}},
            ],
            "judge_c": [
                {"item_id": "1", "scores": {"grammar": 7, "creativity": 5}},
                {"item_id": "2", "scores": {"grammar": 6, "creativity": 7}},
            ],
        }
        df = median_aggregate(scores_by_judge, ["grammar", "creativity"])

        item1 = df[df["item_id"] == "1"].iloc[0]
        assert item1["grammar_median"] == 7.0
        assert item1["creativity_median"] == 6.0

        item2 = df[df["item_id"] == "2"].iloc[0]
        assert item2["grammar_median"] == 5.0
        assert item2["creativity_median"] == 8.0

    def test_nested_scores(self):
        scores_by_judge = {
            "judge_a": [
                {"item_id": "1", "scores": {"accuracy": {"score": 4}, "fluency": {"score": 5}}},
            ],
            "judge_b": [
                {"item_id": "1", "scores": {"accuracy": {"score": 3}, "fluency": {"score": 4}}},
            ],
        }
        df = median_aggregate(scores_by_judge, ["accuracy", "fluency"])
        item = df.iloc[0]
        assert item["accuracy_median"] == 3.5
        assert item["fluency_median"] == 4.5

    def test_empty_input(self):
        df = median_aggregate({}, ["grammar"])
        assert len(df) == 0


class TestWinRate:
    def test_perfect_concordance(self):
        panel = {"a": 1.0, "b": 2.0, "c": 3.0}
        baseline = {"a": 10.0, "b": 20.0, "c": 30.0}
        assert win_rate(panel, baseline) == 1.0

    def test_perfect_discordance(self):
        panel = {"a": 3.0, "b": 2.0, "c": 1.0}
        baseline = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert win_rate(panel, baseline) == 0.0

    def test_partial(self):
        panel = {"a": 1.0, "b": 3.0, "c": 2.0}
        baseline = {"a": 1.0, "b": 2.0, "c": 3.0}
        rate = win_rate(panel, baseline)
        assert 0.0 < rate < 1.0
