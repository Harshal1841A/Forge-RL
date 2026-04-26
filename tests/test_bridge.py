"""
tests/test_bridge.py — Bridge layer unit tests.
Master Prompt v9.0 §10 (HIGH priority).

Covers:
  - TACTIC_TO_PRIMITIVE mapping completeness
  - r1_to_r2_graph() node/edge preservation
  - Coverage gate + padding
  - tactics_to_primitives() deduplication and K_MAX truncation
  - convert_episode() BridgeResult fields
  - Zero-node safety: stub graph returned, never crashes
"""
from __future__ import annotations

import pytest
from datetime import datetime

# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_r1_graph(
    n_nodes: int = 3,
    n_edges: int = 2,
    n_retrieved: int = 2,
    tactics: list | None = None,
    true_label: str = "misinfo",
    difficulty: int = 1,
):
    """Build a minimal R1 ClaimGraph for testing."""
    from env.claim_graph import ClaimGraph, ClaimNode, EvidenceEdge

    import uuid
    gid = str(uuid.uuid4())
    root_id = "root-0"

    nodes = {}
    for i in range(n_nodes):
        nid = root_id if i == 0 else f"n-{i}"
        nodes[nid] = ClaimNode(
            node_id=nid,
            text=f"Node text {i}",
            source_url=f"https://example.com/{i}",
            domain=f"domain-{i % 3}",
            trust_score=0.5 + i * 0.1,
            retrieved=(i < n_retrieved),
        )

    edges = []
    for i in range(n_edges):
        src = root_id if i == 0 else f"n-{i}"
        tgt = f"n-{i+1}" if f"n-{i+1}" in nodes else root_id
        edges.append(EvidenceEdge(
            edge_id=f"e-{i}",
            src_id=src,
            tgt_id=tgt,
            relation="supports",
            weight=0.8,
            discovered=(i % 2 == 0),   # alternating discovered
        ))

    return ClaimGraph(
        graph_id=gid,
        root_claim_id=root_id,
        nodes=nodes,
        edges=edges,
        applied_tactics=tactics or ["fabricate_statistic", "strip_context"],
        true_label=true_label,
        difficulty=difficulty,
    )


# ── TACTIC_TO_PRIMITIVE mapping ────────────────────────────────────────────────

class TestTacticMapping:
    def test_all_tactics_map_to_primitives(self):
        """Every known TacticType string has a mapping."""
        from env.bridge import TACTIC_TO_PRIMITIVE
        from env.claim_graph import TacticType
        import typing

        # Extract the Literal values from TacticType
        all_tactics = list(typing.get_args(TacticType))
        for tactic in all_tactics:
            assert tactic in TACTIC_TO_PRIMITIVE, (
                f"TacticType '{tactic}' has no entry in TACTIC_TO_PRIMITIVE"
            )

    def test_mapping_returns_primitive_type(self):
        from env.bridge import TACTIC_TO_PRIMITIVE
        from env.primitives import PrimitiveType
        for tac, prim in TACTIC_TO_PRIMITIVE.items():
            assert isinstance(prim, PrimitiveType), (
                f"Mapping for '{tac}' is {type(prim)}, expected PrimitiveType"
            )

    def test_canonical_values(self):
        from env.bridge import TACTIC_TO_PRIMITIVE
        from env.primitives import PrimitiveType
        assert TACTIC_TO_PRIMITIVE["fabricate_statistic"] == PrimitiveType.QUOTE_FABRICATE
        assert TACTIC_TO_PRIMITIVE["cherry_pick_study"]   == PrimitiveType.CITATION_FORGE
        assert TACTIC_TO_PRIMITIVE["backdate_article"]    == PrimitiveType.TEMPORAL_SHIFT
        assert TACTIC_TO_PRIMITIVE["amplify_via_bot_network"] == PrimitiveType.NETWORK_AMPLIFY
        assert TACTIC_TO_PRIMITIVE["parody_taken_literally"]  == PrimitiveType.SATIRE_REFRAME


# ── tactics_to_primitives ──────────────────────────────────────────────────────

class TestTacticsToPrimitives:
    def test_basic_conversion(self):
        from env.bridge import tactics_to_primitives
        from env.primitives import PrimitiveType
        result = tactics_to_primitives(["fabricate_statistic"])
        assert result == [PrimitiveType.QUOTE_FABRICATE]

    def test_deduplication_preserves_order(self):
        """Two tactics mapping to the same primitive → deduplicated."""
        from env.bridge import tactics_to_primitives
        from env.primitives import PrimitiveType
        # misattribute_quote → QUOTE_FABRICATE (same as fabricate_statistic)
        result = tactics_to_primitives(["fabricate_statistic", "misattribute_quote"])
        assert result.count(PrimitiveType.QUOTE_FABRICATE) == 1

    def test_k_max_truncation(self):
        """Output never exceeds K_MAX."""
        from env.bridge import tactics_to_primitives
        from env.primitives import K_MAX
        tactics = [
            "fabricate_statistic", "strip_context",
            "backdate_article", "amplify_via_bot_network",
            "cherry_pick_study",          # 5th — should be truncated
        ]
        result = tactics_to_primitives(tactics)
        assert len(result) <= K_MAX

    def test_unknown_tactic_fallback(self):
        """Unknown tactic → QUOTE_FABRICATE fallback, no exception."""
        from env.bridge import tactics_to_primitives, _FALLBACK_PRIMITIVE
        result = tactics_to_primitives(["totally_unknown_tactic"])
        assert result == [_FALLBACK_PRIMITIVE]

    def test_empty_tactics(self):
        from env.bridge import tactics_to_primitives
        assert tactics_to_primitives([]) == []


# ── r1_to_r2_graph ─────────────────────────────────────────────────────────────

class TestR1ToR2Graph:
    def test_preserves_node_count(self):
        from env.bridge import r1_to_r2_graph
        r1 = _make_r1_graph(n_nodes=4, n_retrieved=4)  # 100% coverage
        r2 = r1_to_r2_graph(r1)
        assert len(r2.nodes) == 4

    def test_preserves_root_id(self):
        from env.bridge import r1_to_r2_graph
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        r2 = r1_to_r2_graph(r1)
        assert r2.root_id == r1.root_claim_id

    def test_only_discovered_edges_transferred(self):
        from env.bridge import r1_to_r2_graph
        r1 = _make_r1_graph(n_nodes=4, n_retrieved=4, n_edges=4)
        discovered_count = sum(1 for e in r1.edges if e.discovered)
        r2 = r1_to_r2_graph(r1)
        assert len(r2.edges) == discovered_count

    def test_trust_score_preserved(self):
        from env.bridge import r1_to_r2_graph
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        r1_scores = {nid: n.trust_score for nid, n in r1.nodes.items()}
        r2 = r1_to_r2_graph(r1)
        r2_scores = {n.id: n.trust_score for n in r2.nodes}
        for nid, score in r1_scores.items():
            assert abs(r2_scores.get(nid, -1) - score) < 1e-6

    def test_no_injected_flags_on_r1_nodes(self):
        """R1 nodes are never Red-Team-injected."""
        from env.bridge import r1_to_r2_graph
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        r2 = r1_to_r2_graph(r1)
        assert all(not n.injected for n in r2.nodes)

    def test_never_crashes_on_zero_nodes(self):
        """Zero-node R1 graph → stub graph returned, no exception."""
        from env.bridge import r1_to_r2_graph
        from env.claim_graph import ClaimGraph, ClaimNode
        r1 = ClaimGraph(
            graph_id="empty", root_claim_id="root-0",
            nodes={"root-0": ClaimNode(node_id="root-0", text="x",
                                       source_url="", domain="d",
                                       trust_score=0.5, retrieved=True)},
            edges=[], applied_tactics=[], true_label="unknown",
        )
        r2 = r1_to_r2_graph(r1)
        assert len(r2.nodes) >= 1

    def test_coverage_gate_triggers_padding(self):
        """Low-coverage R1 graph still produces a non-zero R2 graph."""
        from env.bridge import r1_to_r2_graph, COVERAGE_GATE
        # Only root node, not retrieved → coverage = 0
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=0)
        assert r1.evidence_coverage < COVERAGE_GATE
        r2 = r1_to_r2_graph(r1)
        assert len(r2.nodes) > 0

    def test_r2_nodes_are_r2_claimnode_type(self):
        from env.bridge import r1_to_r2_graph
        from env.claim_graph_ma import ClaimNode as R2ClaimNode
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        r2 = r1_to_r2_graph(r1)
        for node in r2.nodes:
            assert isinstance(node, R2ClaimNode)


# ── convert_episode (BridgeResult) ────────────────────────────────────────────

class TestConvertEpisode:
    def test_returns_bridge_result(self):
        from env.bridge import convert_episode, BridgeResult
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        result = convert_episode(r1)
        assert isinstance(result, BridgeResult)

    def test_claim_text_matches_root(self):
        from env.bridge import convert_episode
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3)
        result = convert_episode(r1)
        assert result.claim_text == r1.root.text

    def test_true_chain_is_list_of_primitivetype(self):
        from env.bridge import convert_episode
        from env.primitives import PrimitiveType
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3,
                             tactics=["fabricate_statistic", "strip_context"])
        result = convert_episode(r1)
        assert all(isinstance(p, PrimitiveType) for p in result.true_chain)

    def test_difficulty_passed_through(self):
        from env.bridge import convert_episode
        r1 = _make_r1_graph(n_nodes=3, n_retrieved=3, difficulty=3)
        result = convert_episode(r1)
        assert result.difficulty == 3

    def test_padded_flag_set_correctly(self):
        from env.bridge import convert_episode, COVERAGE_GATE
        r1_low  = _make_r1_graph(n_nodes=5, n_retrieved=0)
        r1_high = _make_r1_graph(n_nodes=5, n_retrieved=5)
        assert convert_episode(r1_low).padded is True
        assert convert_episode(r1_high).padded is False

    def test_r2_graph_nodes_positive(self):
        from env.bridge import convert_episode
        r1 = _make_r1_graph()
        result = convert_episode(r1)
        assert len(result.r2_graph.nodes) > 0
