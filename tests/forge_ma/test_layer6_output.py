"""
Layer 6 — Output tests.
Covers: EpisodeOutput immutability, JSON roundtrip, OversightReport rendering.
"""
import pytest
import json
from env.primitives import PrimitiveType
from rewards.hierarchical_reward import compute_reward
from env.episode_output import EpisodeOutput
from env.oversight_report import generate_oversight_report

SHORT_CHAIN = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.QUOTE_FABRICATE]
TRUE_CHAIN  = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.QUOTE_FABRICATE]

@pytest.fixture
def sample_reward():
    return compute_reward(
        predicted_chains=[SHORT_CHAIN]*4,
        true_chain=TRUE_CHAIN,
        claim_text_before="Scientists confirm vaccines are safe.",
        claim_text_after="Scientists confirm vaccines are safe.",
        consensus_level="unanimous",
        expert_decision="APPROVE",
        steps_taken=4, budget_limit=10, useful_tools_called=3,
    )

@pytest.fixture
def sample_episode(sample_reward):
    return EpisodeOutput.build(
        verdict="misinfo",
        predicted_chain=SHORT_CHAIN,
        true_chain=TRUE_CHAIN,
        reward=sample_reward,
        consensus_level="unanimous",
        expert_decision="APPROVE",
        steps_taken=4, budget_limit=10, useful_tools=3,
        agent_verdicts={"auditor": "misinfo", "historian": "misinfo",
                        "critic": "misinfo", "graph_specialist": "misinfo"},
        episode_id="test-0001",
    )


class TestEpisodeOutput:
    def test_build_returns_instance(self, sample_episode):
        assert isinstance(sample_episode, EpisodeOutput)

    def test_episode_id_set(self, sample_episode):
        assert sample_episode.episode_id == "test-0001"

    def test_chains_stored_as_tuples(self, sample_episode):
        assert isinstance(sample_episode.predicted_chain, tuple)
        assert isinstance(sample_episode.true_chain, tuple)

    def test_frozen_immutable(self, sample_episode):
        with pytest.raises((AttributeError, TypeError)):
            sample_episode.verdict = "real"  # type: ignore

    def test_reward_total_clipped(self, sample_episode):
        assert -1.0 <= sample_episode.reward_total <= 1.0

    def test_chain_accuracy_perfect(self, sample_episode):
        assert sample_episode.chain_accuracy == pytest.approx(1.0)

    def test_is_correct_true(self, sample_episode):
        assert sample_episode.is_correct is True

    def test_to_dict_has_all_keys(self, sample_episode):
        d = sample_episode.to_dict()
        required = {"episode_id","verdict","predicted_chain","true_chain",
                    "reward","consensus_level","expert_decision",
                    "steps_taken","budget_limit","over_budget","timestamp"}
        assert required.issubset(d.keys())

    def test_to_json_is_valid_json(self, sample_episode):
        j = sample_episode.to_json()
        parsed = json.loads(j)
        assert parsed["episode_id"] == "test-0001"

    def test_json_roundtrip(self, sample_episode):
        j = sample_episode.to_json()
        restored = EpisodeOutput.from_json(j)
        assert restored.episode_id == sample_episode.episode_id
        assert restored.verdict == sample_episode.verdict
        assert restored.predicted_chain == sample_episode.predicted_chain
        assert restored.reward_total == pytest.approx(sample_episode.reward_total)

    def test_from_dict_roundtrip(self, sample_episode):
        d = sample_episode.to_dict()
        restored = EpisodeOutput.from_dict(d)
        assert restored.consensus_level == sample_episode.consensus_level
        assert restored.over_budget == sample_episode.over_budget

    def test_agent_verdicts_sorted(self, sample_episode):
        names = [k for k, _ in sample_episode.agent_verdicts]
        assert names == sorted(names)

    def test_timestamp_is_iso(self, sample_episode):
        from datetime import datetime
        ts = sample_episode.timestamp
        # Should parse without error
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_auto_episode_id_generated(self, sample_reward):
        ep = EpisodeOutput.build(
            verdict="real", predicted_chain=[], true_chain=[],
            reward=sample_reward, consensus_level="unanimous",
            expert_decision="REJECT", steps_taken=1, budget_limit=10,
            useful_tools=0, agent_verdicts={},
        )
        assert ep.episode_id  # non-empty
        assert len(ep.episode_id) == 8  # uuid[:8]


class TestOversightReport:
    def test_returns_string(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert isinstance(report, str)

    def test_contains_episode_id(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert "test-0001" in report

    def test_contains_verdict(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert "MISINFO" in report

    def test_contains_chain(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert "TEMPORAL_SHIFT" in report

    def test_contains_reward_section(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert "Reward Breakdown" in report

    def test_contains_json_appendix(self, sample_episode):
        report = generate_oversight_report(sample_episode)
        assert "```json" in report

    def test_optional_claim_text(self, sample_episode):
        report = generate_oversight_report(
            sample_episode, claim_text="Test claim sentence."
        )
        assert "Test claim sentence." in report

    def test_over_budget_flag(self, sample_reward):
        ep = EpisodeOutput.build(
            verdict="misinfo", predicted_chain=SHORT_CHAIN, true_chain=TRUE_CHAIN,
            reward=sample_reward, consensus_level="split_2_2",
            expert_decision="REJECT", steps_taken=15, budget_limit=10,
            useful_tools=1, agent_verdicts={"auditor": "real"},
        )
        # Need a reward that hits over_budget — rebuild with steps > limit
        from rewards.hierarchical_reward import compute_reward as cr
        r2 = cr(
            predicted_chains=[SHORT_CHAIN]*4, true_chain=TRUE_CHAIN,
            claim_text_before="x", claim_text_after="x",
            consensus_level="split_2_2", expert_decision="REJECT",
            steps_taken=15, budget_limit=10, useful_tools_called=1,
        )
        ep2 = EpisodeOutput.build(
            verdict="misinfo", predicted_chain=SHORT_CHAIN, true_chain=TRUE_CHAIN,
            reward=r2, consensus_level="split_2_2",
            expert_decision="REJECT", steps_taken=15, budget_limit=10,
            useful_tools=1, agent_verdicts={"auditor": "real"},
        )
        report = generate_oversight_report(ep2)
        assert "OVER BUDGET" in report

    def test_generation_zero_renders(self, sample_episode):
        report = generate_oversight_report(sample_episode, generation=0)
        assert "Generation:** 0" in report
