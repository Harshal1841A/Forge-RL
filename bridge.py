"""
env/bridge.py — R1 → R2 Bridge Layer.
Master Prompt v9.0 §2 — canonical and only point of contact between
R1 and R2 data models.

Responsibilities:
  1. Convert R1 ClaimGraph (dict-based, claim_graph.py) to
     R2 ClaimGraph (list-based, claim_graph_ma.py).
  2. Map R1 TacticType strings → R2 PrimitiveType enums (canonical mapping).
  3. Enforce the coverage gate: if R1 evidence_coverage < 0.30, pad the
     graph with synthetic "query_source" / "cross_reference" nodes so
     compute_plausibility(initial_graph) is never degenerate.

If ANY conversion step fails, the bridge logs a warning and returns a
single-node stub graph (the same as ForgeEnv._build_initial_graph).
This ensures R2 always has a valid starting graph even when R1 crashes.

Do NOT import from env/misinfo_env.py — that file is frozen (§11).
Do NOT call R1 reward functions — they remain exclusively in R1 episodes (§5).
"""
from __future__ import annotations

# ── Circular import guard ──────────────────────────────────────────────────────────
# This module is the canonical bridge between R1 and R2.
# We explicitly control import order to prevent circular dependencies:
# 1. stdlib + logging
# 2. local types (claim_graph, claim_graph_ma, primitives)
# 3. conversion functions (this file)
# NEVER import from misinfo_env.py, forge_env.py, or reward.py (frozen modules)

import copy
import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Lazy imports to avoid circular deps ────────────────────────────────────────
# R1 types (frozen — read-only)
from env.claim_graph import (
    ClaimGraph as R1ClaimGraph,
    ClaimNode as R1ClaimNode,
    EvidenceEdge as R1EvidenceEdge,
    TacticType,
)
# R2 types
from env.claim_graph_ma import (
    ClaimGraph as R2ClaimGraph,
    ClaimNode as R2ClaimNode,
    EvidenceEdge as R2EvidenceEdge,
)
from env.primitives import PrimitiveType, K_MAX

logger.debug("[bridge] Import guards passed - module initialized safely")


# ── Canonical TacticType → PrimitiveType mapping (Master Prompt v9.0 §2.2) ──
# When code and this document conflict, the document wins.
# This mapping is the ONLY authoritative source — never duplicate it elsewhere.
TACTIC_TO_PRIMITIVE: dict[str, PrimitiveType] = {
    "fabricate_statistic":       PrimitiveType.QUOTE_FABRICATE,
    "cherry_pick_study":         PrimitiveType.CITATION_FORGE,
    "misattribute_quote":        PrimitiveType.QUOTE_FABRICATE,
    "strip_context":             PrimitiveType.CONTEXT_STRIP,
    "backdate_article":          PrimitiveType.TEMPORAL_SHIFT,
    "translate_without_context": PrimitiveType.CONTEXT_STRIP,
    "amplify_via_bot_network":   PrimitiveType.NETWORK_AMPLIFY,
    "splice_image_caption":      PrimitiveType.CONTEXT_STRIP,
    "parody_taken_literally":    PrimitiveType.SATIRE_REFRAME,
}

# Fallback when a tactic has no mapping (§12 Q&A)
_FALLBACK_PRIMITIVE = PrimitiveType.QUOTE_FABRICATE

# Coverage gate: R1 graph must have at least this fraction retrieved (§2.4)
COVERAGE_GATE: float = 0.30


# ── Public API ─────────────────────────────────────────────────────────────────

def tactics_to_primitives(applied_tactics: List[str]) -> List[PrimitiveType]:
    """
    Convert a list of R1 TacticType strings to a deduplicated, order-preserving
    List[PrimitiveType] of length ≤ K_MAX.

    Unknown tactics log a warning and map to QUOTE_FABRICATE (§12).
    """
    seen: set[PrimitiveType] = set()
    result: List[PrimitiveType] = []
    for tactic in applied_tactics:
        prim = TACTIC_TO_PRIMITIVE.get(tactic)
        if prim is None:
            logger.warning(
                "[bridge] Unknown tactic %r — defaulting to %s",
                tactic, _FALLBACK_PRIMITIVE.value,
            )
            prim = _FALLBACK_PRIMITIVE
        if prim not in seen:
            seen.add(prim)
            result.append(prim)
        if len(result) >= K_MAX:
            break
    return result


def r1_to_r2_graph(r1_graph: R1ClaimGraph) -> R2ClaimGraph:
    """
    Convert a completed R1 ClaimGraph into the R2 MA format.

    Rules (Master Prompt v9.0 §2.3):
    - Only discovered=True edges are transferred (undiscovered = agent didn't
      find them → must not leak into R2).
    - If coverage < COVERAGE_GATE (0.30), synthetic padding nodes are added
      before conversion so compute_plausibility has a non-degenerate input.
    - On any exception, returns a single-node stub with the root claim text.
    """
    try:
        # ── Coverage gate check (§2.4) ────────────────────────────────────────
        coverage = r1_graph.evidence_coverage
        if coverage < COVERAGE_GATE:
            logger.warning(
                "[bridge] R1 coverage %.2f < gate %.2f — padding graph",
                coverage, COVERAGE_GATE,
            )
            r1_graph = _pad_graph(r1_graph)

        # ── Node conversion ───────────────────────────────────────────────────
        nodes: List[R2ClaimNode] = []
        for node_id, r1_node in r1_graph.nodes.items():
            r2_node = R2ClaimNode(
                id=node_id,
                text=r1_node.text,
                domain=r1_node.domain,
                trust_score=r1_node.trust_score,
                is_retrieved=r1_node.retrieved,
                injected=False,      # R1 nodes are never Red-Team-injected
                primitive=None,      # primitive attribution happens in R2
                fingerprints={},
            )
            nodes.append(r2_node)

        if not nodes:
            logger.warning("[bridge] R1 graph has zero nodes — returning stub")
            return _stub_graph(r1_graph.root.text if r1_graph.nodes else "unknown claim")

        # ── Edge conversion (discovered only) ─────────────────────────────────
        edges: List[R2EvidenceEdge] = []
        for r1_edge in r1_graph.edges:
            if r1_edge.discovered:
                r2_edge = R2EvidenceEdge(
                    source_id=r1_edge.src_id,
                    target_id=r1_edge.tgt_id,
                    relation=r1_edge.relation,
                    weight=r1_edge.weight,
                    injected=False,
                )
                edges.append(r2_edge)

        r2_graph = R2ClaimGraph(
            nodes=nodes,
            edges=edges,
            root_id=r1_graph.root_claim_id,
        )

        assert len(r2_graph.nodes) > 0, "Bridge produced zero-node R2 graph"
        logger.info(
            "[bridge] Converted R1→R2: %d nodes, %d edges (of %d total, %d discovered)",
            len(nodes), len(edges), len(r1_graph.edges),
            sum(1 for e in r1_graph.edges if e.discovered),
        )
        return r2_graph

    except Exception as exc:  # pragma: no cover — safety net
        logger.error("[bridge] Conversion failed (%s) — using stub graph", exc)
        try:
            claim_text = r1_graph.root.text
        except Exception:
            claim_text = "unknown claim"
        return _stub_graph(claim_text)


@dataclass
class BridgeResult:
    """Full output of one R1→R2 bridge pass."""
    r2_graph:    R2ClaimGraph
    true_chain:  List[PrimitiveType]
    claim_text:  str
    difficulty:  int
    r1_coverage: float   # for monitoring / kill criteria
    padded:      bool    # True if coverage gate was triggered


def convert_episode(r1_graph: R1ClaimGraph) -> BridgeResult:
    """
    One-stop conversion for a completed R1 episode.

    Returns a BridgeResult with everything ForgeEnv.reset_from_r1() needs.
    """
    coverage = r1_graph.evidence_coverage
    padded = coverage < COVERAGE_GATE

    r2_graph   = r1_to_r2_graph(r1_graph)
    true_chain = tactics_to_primitives(r1_graph.applied_tactics)
    claim_text = r1_graph.root.text
    difficulty = getattr(r1_graph, "difficulty", 1)

    return BridgeResult(
        r2_graph=r2_graph,
        true_chain=true_chain,
        claim_text=claim_text,
        difficulty=difficulty,
        r1_coverage=coverage,
        padded=padded,
    )


# ── Private helpers ────────────────────────────────────────────────────────────

def _pad_graph(r1_graph: R1ClaimGraph) -> R1ClaimGraph:
    """
    Add synthetic 'retrieved' nodes so evidence_coverage reaches the gate.
    Simulates two synthetic tool calls: query_source + cross_reference.
    Works on a shallow copy so the original R1 graph is not mutated.
    """
    import copy as _copy
    padded = _copy.copy(r1_graph)
    padded.nodes = dict(r1_graph.nodes)   # shallow copy of node dict

    # Mark all existing nodes as retrieved (simulates exhaustive tool calls)
    for nid in padded.nodes:
        node = _copy.copy(padded.nodes[nid])
        node.retrieved = True
        padded.nodes[nid] = node

    # If still below gate (e.g. 0 nodes), add a synthetic evidence node
    if padded.evidence_coverage < COVERAGE_GATE or not padded.nodes:
        from env.claim_graph import ClaimNode as R1CN
        synth_id = "synth-pad-0"
        padded.nodes[synth_id] = R1CN(
            node_id=synth_id,
            text="[Synthetic evidence node added by bridge padding]",
            source_url="https://bridge.forge.internal/pad",
            domain="bridge-internal",
            trust_score=0.5,
            retrieved=True,
        )

    return padded


def _stub_graph(claim_text: str) -> R2ClaimGraph:
    """Single-node fallback graph (identical to ForgeEnv._build_initial_graph)."""
    root_id = "root-0"
    root_node = R2ClaimNode(
        id=root_id,
        text=claim_text,
        domain="root",
        trust_score=0.5,
        is_retrieved=False,
        injected=False,
        primitive=None,
        fingerprints={},
    )
    return R2ClaimGraph(nodes=[root_node], edges=[], root_id=root_id)
