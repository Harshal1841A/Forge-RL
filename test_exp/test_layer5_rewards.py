"""
Layer 5 — Reward system tests.
Covers: BudgetPenalty components, HierarchicalReward weights/bounds/consensus routing.
"""
import pytest
from env.primitives import PrimitiveType
from rewards.budget_penalty import (
    compute_budget_penalty, STEP_COST, OVER_BUDGET_PENALTY,
    TOOL_EFFICIENCY_BONUS
)
from rewards.hierarchical_reward import (
    compute_reward, W_TED, W_F1, W_PLB,
    CONSENSUS_BONUS, EXPERT_APPROVE_BONUS
)

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
SHORT_CHAIN  = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.QUOTE_FABRICATE]
FULL_CHAIN   = list(PrimitiveType)[:4]   # K_MAX length
EMPTY_CHAIN  = []

REAL_CLAIM   = "Scientists confirm vaccines are safe and effective."
MISINFO_CLAIM = "Leaked docs show vaccines cause autism according to insider."


# ─────────────────────────────────────────────
# Weight constants
# ─────────────────────────────────────────────
class TestWeightConstants:
    def test_weights_sum_to_09(self):
        """TED+F1+PLB = 0.90; remaining 0.10 covered by bonuses."""
        total = W_TED + W_F1 + W_PLB
        assert abs(total - 0.90) < 1e-9, f"Weights sum to {total}, expected 0.90"

    def test_w_ted_dominant(self):
        assert W_TED == 0.40

    def test_w_f1_value(self):
        assert W_F1 == 0.30

    def test_w_plb_value(self):
        assert W_PLB == 0.20

    def test_consensus_unanimous_bonus(self):
        assert CONSENSUS_BONUS["unanimous"] == pytest.approx(0.10)

    def test_consensus_majority_bonus(self):
        assert CONSENSUS_BONUS["majority_3"] == pytest.approx(0.05)

    def test_consensus_split_penalty(self):
        assert CONSENSUS_BONUS["split_2_2"] == pytest.approx(-0.05)

    def test_consensus_all_diff_penalty(self):
        assert CONSENSUS_BONUS["all_different"] == pytest.approx(-0.05)

    def test_expert_approve_bonus(self):
        assert EXPERT_APPROVE_BONUS == pytest.approx(0.05)


# ─────────────────────────────────────────────
# BudgetPenalty
# ─────────────────────────────────────────────
class TestBudgetPenalty:
    def test_step_cost_accumulates(self):
        res = compute_budget_penalty(steps_taken=5, budget_limit=10, useful_tools_called=0)
        assert res.step_cost_total == pytest.approx(STEP_COST * 5)

    def test_no_over_budget_flag_within_limit(self):
        res = compute_budget_penalty(steps_taken=5, budget_limit=10, useful_tools_called=0)
        assert res.over_budget_hit is False
        assert res.over_budget_penalty == pytest.approx(0.0)

    def test_over_budget_flag_and_penalty(self):
        res = compute_budget_penalty(steps_taken=11, budget_limit=10, useful_tools_called=0)
        assert res.over_budget_hit is True
        assert res.over_budget_penalty == pytest.approx(OVER_BUDGET_PENALTY)

    def test_efficiency_bonus_granted(self):
        # 3 useful tools, only 4/10 steps used = 40% < 60% threshold
        res = compute_budget_penalty(steps_taken=4, budget_limit=10, useful_tools_called=3)
        assert res.efficiency_bonus == pytest.approx(TOOL_EFFICIENCY_BONUS)

    def test_efficiency_bonus_not_granted_high_usage(self):
        # 7/10 = 70% > 60% → no bonus
        res = compute_budget_penalty(steps_taken=7, budget_limit=10, useful_tools_called=5)
        assert res.efficiency_bonus == pytest.approx(0.0)

    def test_efficiency_bonus_not_granted_low_tools(self):
        # Only 1 useful tool, not ≥2
        res = compute_budget_penalty(steps_taken=3, budget_limit=10, useful_tools_called=1)
        assert res.efficiency_bonus == pytest.approx(0.0)

    def test_total_clipped_to_minus_one(self):
        # Extreme over-budget should not go below -1.0
        res = compute_budget_penalty(steps_taken=100, budget_limit=10, useful_tools_called=0)
        assert res.total >= -1.0

    def test_total_never_positive(self):
        # No scenario should produce a positive budget total
        res = compute_budget_penalty(steps_taken=1, budget_limit=10, useful_tools_called=10)
        assert res.total <= 0.0

    def test_zero_steps(self):
        res = compute_budget_penalty(steps_taken=0, budget_limit=10, useful_tools_called=0)
        assert res.step_cost_total == pytest.approx(0.0)
        assert res.over_budget_hit is False


# ─────────────────────────────────────────────
# HierarchicalReward
# ─────────────────────────────────────────────
class TestHierarchicalReward:

    def _base_kwargs(self, **overrides):
        """Default valid kwargs for compute_reward."""
        base = dict(
            predicted_chains=[SHORT_CHAIN, SHORT_CHAIN, SHORT_CHAIN, SHORT_CHAIN],
            true_chain=SHORT_CHAIN,
            claim_text_before=REAL_CLAIM,
            claim_text_after=REAL_CLAIM,
            consensus_level="unanimous",
            expert_decision="APPROVE",
            steps_taken=4,
            budget_limit=10,
            useful_tools_called=3,
        )
        base.update(overrides)
        return base

    def test_total_clipped_upper(self):
        res = compute_reward(**self._base_kwargs())
        assert res.total <= 1.0

    def test_total_clipped_lower(self):
        res = compute_reward(**self._base_kwargs(
            predicted_chains=[EMPTY_CHAIN] * 4,
            true_chain=FULL_CHAIN,
            consensus_level="all_different",
            expert_decision="REJECT",
            steps_taken=20,
            budget_limit=10,
        ))
        assert res.total >= -1.0

    def test_perfect_prediction_positive_reward(self):
        """Exact chain match → high TED + perfect F1 → should be positive."""
        res = compute_reward(**self._base_kwargs(
            predicted_chains=[SHORT_CHAIN] * 4,
            true_chain=SHORT_CHAIN,
            consensus_level="unanimous",
            expert_decision="APPROVE",
            steps_taken=2,
        ))
        assert res.total > 0.0, f"Expected positive reward, got {res.total}"

    def test_unanimous_beats_split(self):
        kwargs_u = self._base_kwargs(consensus_level="unanimous")
        kwargs_s = self._base_kwargs(consensus_level="split_2_2")
        r_u = compute_reward(**kwargs_u)
        r_s = compute_reward(**kwargs_s)
        assert r_u.total > r_s.total

    def test_approve_beats_reject(self):
        kwargs_a = self._base_kwargs(expert_decision="APPROVE")
        kwargs_r = self._base_kwargs(expert_decision="REJECT")
        r_a = compute_reward(**kwargs_a)
        r_r = compute_reward(**kwargs_r)
        assert r_a.total > r_r.total

    def test_ted_component_weight(self):
        """TED component must be W_TED * TED_best (≈W_TED for identical chains)."""
        res = compute_reward(**self._base_kwargs())
        # TED of identical chain vs itself should be close to 1.0 (after clipping)
        # ted_component should be close to W_TED
        assert 0.0 < res.ted_component <= W_TED + 0.01

    def test_f1_component_perfect(self):
        """Perfect prediction → F1=1.0 → f1_component = W_F1."""
        res = compute_reward(**self._base_kwargs(
            predicted_chains=[SHORT_CHAIN] * 4,
            true_chain=SHORT_CHAIN,
        ))
        assert res.f1_component == pytest.approx(W_F1, abs=1e-3)

    def test_consensus_bonus_routed_correctly(self):
        for level, expected_bonus in CONSENSUS_BONUS.items():
            res = compute_reward(**self._base_kwargs(consensus_level=level))
            assert res.consensus_bonus == pytest.approx(expected_bonus), \
                f"Consensus '{level}': expected {expected_bonus}, got {res.consensus_bonus}"

    def test_budget_over_limit_hurts(self):
        r_ok = compute_reward(**self._base_kwargs(steps_taken=4, budget_limit=10))
        r_over = compute_reward(**self._base_kwargs(steps_taken=15, budget_limit=10))
        assert r_ok.total > r_over.total

    def test_breakdown_str_no_crash(self):
        res = compute_reward(**self._base_kwargs())
        s = str(res)
        assert "R_total" in s

    def test_empty_predicted_chains(self):
        """Empty predictions must not crash — just produce poor reward."""
        res = compute_reward(**self._base_kwargs(
            predicted_chains=[[] for _ in range(4)],
            true_chain=SHORT_CHAIN,
        ))
        assert isinstance(res.total, float)
        assert -1.0 <= res.total <= 1.0
