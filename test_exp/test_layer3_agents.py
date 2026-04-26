"""
Behavioral tests for Layer 3 agents.
Medium Bug M3 fix: Replaced meaningless hasattr() checks with real behavioral tests.
  - NegotiatedSearch: verify generate_vectors() returns normalised 13-dim array
  - SocietyOfThought: verify investigate() returns SocietyResult with non-None fields
    and that the Auditor/Historian stubs produce structured (non-unknown) output
    for well-known misinformation claim patterns.
"""
import pytest
import numpy as np
import torch

from agents.expert_reviewer_agent import ExpertReviewerAgent
from blue_team.negotiated_search import NegotiatedSearch, TOOLS, PRIMITIVE_TO_TOOLS
from blue_team.society_of_thought import SocietyOfThought, _auditor_analyze, _historian_analyze
from blue_team.gin_predictor import GINPredictor
from env.primitives import PrimitiveType


# ── Expert Reviewer Agent ──────────────────────────────────────────────────────

def test_expert_reviewer_agent_approves_good_episode():
    agent = ExpertReviewerAgent(mode="dawid_skene")
    votes = agent.evaluate_profiles(
        verdict_correct=True,
        recall=0.80,
        confidence=0.90,
        hallucinations=0,
        budget_used=0.40,
        steps=3,
        tools_called=5,
        coverage=0.90,
        generation=0,
    )
    assert votes["legal"] is True
    assert votes["fast"] is True
    assert votes["lead"] is True
    assert votes["journalist"] is True
    result = agent.aggregate(votes)
    assert result == "APPROVE"


def test_expert_reviewer_agent_rejects_bad_episode():
    """Low recall + hallucinations should result in REJECT."""
    agent = ExpertReviewerAgent(mode="dawid_skene")
    votes = agent.evaluate_profiles(
        verdict_correct=False,
        recall=0.20,
        confidence=0.30,
        hallucinations=5,
        budget_used=0.90,
        steps=9,
        tools_called=0,
        coverage=0.10,
        generation=0,
    )
    result = agent.aggregate(votes)
    assert result == "REJECT"


# ── NegotiatedSearch — behavioral tests ───────────────────────────────────────

class _MockGIN:
    """Minimal GIN mock: returns uniform 0.125 presence probs."""
    def predict_chain(self, graph_data):
        return {
            "presence_probs": np.full(8, 0.125),
            "ordered_chain": [],
            "confidence": 0.0,
            "uncertainty": np.zeros(8),
        }


def test_negotiated_search_returns_13dim():
    """generate_vectors() must return a normalised 13-dim numpy array."""
    ns = NegotiatedSearch.__new__(NegotiatedSearch)
    # Inject mock historian/critic that return valid dicts
    from agents.llm_agent import LLMAgent
    mock_return = '{"tool_preferences": {}}'

    class _MockLLM:
        def query(self, prompt):
            return mock_return
        def parse_json(self, resp):
            return {"tool_preferences": {}}

    ns.historian = _MockLLM()
    ns.critic = _MockLLM()

    gin = _MockGIN()
    vec = ns.generate_vectors("Test claim about vaccines.", gin)

    assert isinstance(vec, np.ndarray), "Output must be numpy array"
    assert vec.shape == (13,), f"Expected shape (13,), got {vec.shape}"
    assert abs(vec.sum() - 1.0) < 1e-6, f"Vector must sum to 1.0, got {vec.sum()}"
    assert (vec >= 0).all(), "All values must be non-negative"


def test_primitive_to_tools_covers_all_primitives():
    """Every PrimitiveType must have an entry in PRIMITIVE_TO_TOOLS."""
    for prim in PrimitiveType:
        assert prim in PRIMITIVE_TO_TOOLS, f"{prim} missing from PRIMITIVE_TO_TOOLS"
        tools = PRIMITIVE_TO_TOOLS[prim]
        assert len(tools) >= 1, f"{prim} has empty tool list"
        for t in tools:
            assert t in TOOLS, f"Tool '{t}' in PRIMITIVE_TO_TOOLS[{prim}] not in TOOLS list"


def test_gin_probs_to_tool_vector_sums_to_one():
    """Semantic GIN→tool mapping must produce normalised 13-dim vector."""
    ns = NegotiatedSearch.__new__(NegotiatedSearch)
    gin_probs = np.array([0.9, 0.1, 0.5, 0.8, 0.2, 0.4, 0.6, 0.3])
    result = ns._gin_probs_to_tool_vector(gin_probs)
    assert result.shape == (13,)
    assert abs(result.sum() - 1.0) < 1e-6
    assert (result >= 0).all()


# ── Society of Thought — behavioral tests ─────────────────────────────────────

def test_auditor_analyze_returns_misinfo_for_fabrication_claim():
    """Auditor stub must return 'misinfo' for a claim with fabrication keywords."""
    result = _auditor_analyze(
        "Vaccines cause autism, leaked documents confirm.",
        "{}"
    )
    assert result["verdict"] in ("misinfo", "fabricated"), \
        f"Expected misinfo/fabricated, got {result['verdict']}"
    assert isinstance(result["predicted_chain"], list)
    assert result["confidence"] > 0.5
    assert "rationale" in result and len(result["rationale"]) > 0


def test_auditor_analyze_returns_structured_output_for_unknown():
    """Even for a neutral claim, auditor returns structured dict (not None)."""
    result = _auditor_analyze("The sky is blue.", "{}")
    assert "verdict" in result
    assert "predicted_chain" in result
    assert "confidence" in result
    assert "rationale" in result


def test_historian_analyze_detects_temporal_signal():
    """Historian stub must flag temporal mismatch in year-misdate claims."""
    result = _historian_analyze("Video shows 2015 protest mislabelled as 2024 riots.")
    assert result["verdict"] == "misinfo"
    assert PrimitiveType.TEMPORAL_SHIFT in result["predicted_chain"]
    assert result["confidence"] > 0.5


def test_historian_analyze_returns_structured_output():
    """Historian always returns a complete structured dict."""
    result = _historian_analyze("A random neutral claim.")
    assert all(k in result for k in ["verdict", "predicted_chain", "rationale", "confidence"])
    assert isinstance(result["predicted_chain"], list)


def test_society_investigate_returns_society_result():
    """investigate() must return a SocietyResult with all non-None fields."""
    gin = GINPredictor()
    sot = SocietyOfThought(
        auditor=None,
        historian=None,
        critic=None,
        graph_specialist=None,
        gin=gin,
    )

    result = sot.investigate(
        "Vaccines cause autism, leaked documents confirm.",
        true_chain=[PrimitiveType.QUOTE_FABRICATE, PrimitiveType.SOURCE_LAUNDER],
    )

    assert result is not None
    assert result.verdict is not None and len(result.verdict) > 0
    assert isinstance(result.predicted_chain, list)
    assert result.consensus_level in ("unanimous", "majority_3", "split_2_2", "all_different")
    assert result.consensus_bonus in (0.10, 0.05, -0.05)
    assert isinstance(result.agent_verdicts, dict)
    assert len(result.agent_verdicts) == 4
    # All 4 agents must be represented
    assert "auditor" in result.agent_verdicts
    assert "historian" in result.agent_verdicts
    assert "critic" in result.agent_verdicts
    assert "graph_specialist" in result.agent_verdicts
    # TED must be a non-negative float
    assert isinstance(result.ted_best, float)
    assert result.ted_best >= 0.0


def test_society_auditor_and_historian_never_always_unknown():
    """
    With a misinformation claim, Auditor and Historian must NOT both return 'unknown'.
    This verifies H1 is fixed — 2-agent always-unknown issue is resolved.
    """
    gin = GINPredictor()
    sot = SocietyOfThought(None, None, None, None, gin)

    result = sot.investigate("Vaccines cause autism, leaked documents confirm.")

    # At least one of Auditor/Historian must give a non-unknown verdict
    aud_verdict = result.agent_verdicts.get("auditor", "unknown")
    hist_verdict = result.agent_verdicts.get("historian", "unknown")
    assert not (aud_verdict == "unknown" and hist_verdict == "unknown"), \
        "Both Auditor and Historian returned 'unknown' — H1 fix not working"
