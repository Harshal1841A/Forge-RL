"""
Layer 9 — Evaluation tests.
"""
import pytest
from evaluation.evaluator import run_evaluation, EvalMetrics

@pytest.fixture(scope="module")
def metrics():
    return run_evaluation(n_episodes=5, budget=3, seed=99)

class TestEvalMetrics:
    def test_returns_eval_metrics(self, metrics):
        assert isinstance(metrics, EvalMetrics)

    def test_n_episodes_positive(self, metrics):
        assert metrics.n_episodes > 0

    def test_mean_reward_in_range(self, metrics):
        assert -1.0 <= metrics.mean_reward <= 1.0

    def test_mean_ted_in_range(self, metrics):
        assert 0.0 <= metrics.mean_ted <= 1.0

    def test_mean_f1_in_range(self, metrics):
        assert 0.0 <= metrics.mean_f1 <= 1.0

    def test_chain_accuracy_in_range(self, metrics):
        assert 0.0 <= metrics.mean_chain_accuracy <= 1.0

    def test_over_budget_rate_in_range(self, metrics):
        assert 0.0 <= metrics.over_budget_rate <= 1.0

    def test_verdict_distribution_not_empty(self, metrics):
        assert isinstance(metrics.verdict_distribution, dict)

    def test_to_json_valid(self, metrics):
        import json
        data = json.loads(metrics.to_json())
        assert "mean_reward" in data

    def test_summary_table_renders(self, metrics):
        table = metrics.summary_table()
        assert "FORGE-MA" in table
        assert "Mean Reward" in table
