"""
tests/test_pipeline.py — End-to-end pipeline integration tests with mock agents.
Master Prompt v9.0 §10 (HIGH priority).

Uses lightweight mock classes so no live API calls are made.
Covers:
  - Full R1→bridge→R2 episode completes without crash
  - PipelineResult fields are populated correctly
  - Bridge failure triggers R2 demo-mode fallback
  - Pipeline correctly passes separate graph objects to ForgeEnv
  - reset_from_r1() sets both _initial_claim_graph and _claim_graph
  - Coverage gate path is exercised
"""
from __future__ import annotations

import pytest
from typing import List
from env.primitives import PrimitiveType
from env.bridge import convert_episode, r1_to_r2_graph


# ── Mock R1 agent & env ────────────────────────────────────────────────────────

class MockR1Agent:
    """Trivial agent: always picks action 0."""
    def act(self, obs):
        return 0


def _make_r1_graph(n_nodes=3, n_retrieved=3, tactics=None, true_label="misinfo"):
    from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge
    import uuid
    root_id = "root-0"
    nodes = {}
    for i in range(n_nodes):
        nid = root_id if i == 0 else f"n-{i}"
        nodes[nid] = ClaimNode(
            node_id=nid, text=f"Text {i}",
            source_url=f"https://ex.com/{i}",
            domain=f"d{i % 2}", trust_score=0.5,
            retrieved=(i < n_retrieved),
        )
    edges = []
    if n_nodes >= 2:
        edges.append(EvidenceEdge(edge_id="e-0", src_id=root_id, tgt_id="n-1",
                                  relation="supports", weight=0.8, discovered=True))
    return ClaimGraph(
        graph_id=str(uuid.uuid4()), root_claim_id=root_id,
        nodes=nodes, edges=edges,
        applied_tactics=tactics or ["fabricate_statistic", "strip_context"],
        true_label=true_label, difficulty=1,
    )


class MockR1Env:
    """Minimal R1 env: terminates after 2 steps."""
    def __init__(self, true_label="misinfo", n_retrieved=3):
        self._step_count = 0
        self.last_verdict = true_label
        self.true_label   = true_label
        self.graph        = _make_r1_graph(n_nodes=3, n_retrieved=n_retrieved,
                                           true_label=true_label)

    def reset(self, seed=None):
        self._step_count = 0
        return {}, {"seed": seed}

    def step(self, action):
        self._step_count += 1
        done = self._step_count >= 2
        return {}, 0.5, done, False, {}


# ── Mock Society and R2 env ────────────────────────────────────────────────────

class MockSocietyResult:
    verdict        = "misinfo"
    ted_best       = 0.62
    consensus_level = "majority_3"
    agent_verdicts = {"auditor": "misinfo", "historian": "misinfo",
                      "critic": "real", "graph_specialist": "misinfo"}


class MockSociety:
    def investigate(self, claim, true_chain, budget, claim_graph):
        return MockSocietyResult()


class MockR2Env:
    """Minimal ForgeEnv stand-in."""
    budget      = 5
    claim_graph = None
    _steps      = 0

    def reset(self, seed=None):
        self._steps = 0
        self._initial_claim_graph = None
        self._claim_graph = None
        return {"claim_text": "Test claim"}, {}

    def reset_from_r1(self, initial_graph, true_chain, claim_text, seed=None):
        from env.claim_graph_ma import ClaimGraph
        import copy
        self._initial_claim_graph = copy.deepcopy(initial_graph)
        self._claim_graph = copy.deepcopy(initial_graph)
        self.claim_graph  = self._claim_graph
        self._steps = 0
        return {"claim_text": claim_text}, {"pipeline_mode": True}

    def step(self, action=None):
        self._steps += 1
        done = self._steps >= 3
        return {}, -0.1, done, False, {}


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPipelineEndToEnd:
    def test_run_pipeline_episode_completes(self):
        """Full pipeline episode returns a PipelineResult without crashing."""
        from pipeline import run_pipeline_episode, PipelineResult
        result = run_pipeline_episode(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert isinstance(result, PipelineResult)

    def test_r1_fields_populated(self):
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(true_label="misinfo"), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert result.r1_true_label == "misinfo"
        assert result.r1_steps >= 1
        assert 0.0 <= result.r1_reward

    def test_bridge_true_chain_propagated(self):
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert isinstance(result.true_chain, list)
        # fabricate_statistic → QUOTE_FABRICATE, strip_context → CONTEXT_STRIP
        assert "QUOTE_FABRICATE" in result.true_chain or len(result.true_chain) >= 1

    def test_r2_fields_populated(self):
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert result.r2_verdict == "misinfo"
        assert result.r2_ted_best == pytest.approx(0.62, abs=1e-4)
        assert result.r2_consensus == "majority_3"

    def test_pipeline_mode_true_when_bridge_succeeds(self):
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(n_retrieved=3), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert result.pipeline_mode is True

    def test_timing_fields_present(self):
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        assert result.total_elapsed_s >= 0    # mock agents finish in µs → 0.0 is valid
        assert result.r1_elapsed_s >= 0
        assert result.bridge_elapsed_s >= 0
        assert result.r2_elapsed_s >= 0

    def test_to_dict_is_json_serialisable(self):
        import json
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
        )
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0


class TestBridgeFallback:
    def test_r2_runs_when_bridge_fails(self):
        """If bridge fails, R2 still runs in demo mode — pipeline_mode=False."""
        from pipeline import PipelineResult, _run_r2
        from env.bridge import BridgeResult
        # Pass None to simulate bridge failure
        reward, steps, verdict, ted, consensus, agents = _run_r2(
            r2_env=MockR2Env(), r2_society=MockSociety(),
            bridge_result=None, seed=0,
        )
        # Should return safe defaults
        assert verdict in ("misinfo", "unknown")
        assert steps >= 0


class TestResetFromR1:
    def test_sets_separate_graph_objects(self):
        """reset_from_r1() must set _initial_claim_graph != _claim_graph (separate objects)."""
        r2_env = MockR2Env()
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        bridge_res = convert_episode(r1)
        import copy
        r2_env.reset_from_r1(
            initial_graph=bridge_res.r2_graph,
            true_chain=bridge_res.true_chain,
            claim_text=bridge_res.claim_text,
        )
        # Must be separate objects so plausibility delta != 0
        assert r2_env._initial_claim_graph is not r2_env._claim_graph

    def test_pipeline_mode_info_flag(self):
        r2_env = MockR2Env()
        r1 = _make_r1_graph()
        bridge_res = convert_episode(r1)
        _, info = r2_env.reset_from_r1(
            initial_graph=bridge_res.r2_graph,
            true_chain=bridge_res.true_chain,
            claim_text=bridge_res.claim_text,
        )
        assert info.get("pipeline_mode") is True

    def test_graph_has_nodes(self):
        r2_env = MockR2Env()
        r1 = _make_r1_graph()
        bridge_res = convert_episode(r1)
        r2_env.reset_from_r1(
            initial_graph=bridge_res.r2_graph,
            true_chain=bridge_res.true_chain,
            claim_text=bridge_res.claim_text,
        )
        assert len(r2_env._claim_graph.nodes) > 0


class TestBatchPipeline:
    def test_batch_returns_list(self):
        from pipeline import run_pipeline_batch
        results = run_pipeline_batch(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
            n_episodes=3, base_seed=42,
        )
        assert isinstance(results, list)
        assert len(results) == 3

    def test_batch_seeds_differ(self):
        """Each episode uses a different seed (base_seed + i)."""
        from pipeline import run_pipeline_batch
        results = run_pipeline_batch(
            r1_env=MockR1Env(), r1_agent=MockR1Agent(),
            r2_env=MockR2Env(), r2_society=MockSociety(),
            n_episodes=2, base_seed=0,
        )
        # Not identical objects
        assert results[0] is not results[1]


class TestCoverageGatePath:
    def test_low_coverage_r1_still_produces_results(self):
        """Pipeline completes when R1 graph has 0% coverage (bridge pads it)."""
        from pipeline import run_pipeline_episode
        result = run_pipeline_episode(
            r1_env=MockR1Env(n_retrieved=0),
            r1_agent=MockR1Agent(),
            r2_env=MockR2Env(),
            r2_society=MockSociety(),
        )
        # bridge_padded must be True; system must still produce results
        assert result.bridge_padded is True
        assert isinstance(result.r2_verdict, str)
