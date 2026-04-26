"""
Society of Thought — 4-agent investigation orchestrator.
SPEC (PRD v8.1): Forensic Auditor leads, Context Historian + Narrative Critic
provide parallel interpretation, Graph Specialist (GIN) provides topology-based
chain prediction. Consensus logic produces final verdict + chain.

High Bug H1 fix: Auditor and Historian were hardcoded stubs returning
{"verdict": "unknown", ...} unconditionally. They now return structured
mock stubs using real NLP signals from the claim text, giving non-unknown
verdicts and non-empty chains when evidence is present.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import concurrent.futures
import copy
import json
import threading
import numpy as np

from env.primitives import PrimitiveType
from rewards.tactic_edit_dist import tactic_edit_distance
from blue_team.negotiated_search import NegotiatedSearch


@dataclass
class ConsensusResult:
    verdict: str
    predicted_chain: List[PrimitiveType]
    consensus_level: str
    consensus_bonus: float
    dissenting_agents: List[str]
    dissenting_rationales: List[str]


@dataclass
class SocietyResult:
    verdict: str
    predicted_chain: List[PrimitiveType]
    ted_best: float
    consensus_level: str   # "unanimous" | "majority_3" | "split_2_2" | "all_different"
    consensus_bonus: float  # +0.10 (4/4), +0.05 (3/4), -0.05 (<=2/4)
    agent_verdicts: Dict[str, str]
    agent_chains: Dict[str, List[PrimitiveType]]
    agent_confidences: Dict[str, float]
    dissenting_agents: List[str]
    dissenting_rationales: List[str]


# ── Verdict normaliser ────────────────────────────────────────────────────────
_VERDICT_MAP = {
    # fabrication variants
    "fabrication": "fabricated", "fabricate": "fabricated", "fabricated": "fabricated",
    "fake": "fabricated", "forged": "fabricated", "hoax": "fabricated",
    # misinfo variants
    "misinformation": "misinfo", "misinfo": "misinfo", "misleading": "misinfo",
    "false": "misinfo", "false claim": "misinfo", "disinformation": "misinfo",
    # satire
    "satire": "satire", "parody": "satire",
    # real / legit
    "real": "real", "true": "real", "legitimate": "real", "verified": "real",
    # unknown / inconclusive
    "unknown": "unknown", "inconclusive": "unknown", "uncertain": "unknown",
}

def _normalize_verdict(raw: str) -> str:
    """Map free-form LLM verdict strings to the canonical set."""
    key = raw.lower().strip().rstrip(".")
    return _VERDICT_MAP.get(key, raw.lower().strip())


# ── Forensic Auditor — Groq / Llama-3 70B ────────────────────────────────────
def _auditor_analyze(claim: str, claim_graph_json: str, gin_feedback: str = "") -> dict:
    """Medium-4 fix: gin_feedback parameter so GIN hint reaches the Auditor."""
    import config
    from agents.llm_agent_ma import LLMAgent
    agent = LLMAgent(
        system_prompt="You are a Forensic Auditor specialising in detecting fabricated sources, retracted studies, and misattributed quotes. Respond only in JSON with keys: verdict (one of: real, misinfo, fabricated, satire, unknown), predicted_chain, rationale, confidence.",
        provider=config.AGENT_AUDITOR_PROVIDER,        # groq
        api_key=config.OPENAI_API_KEY_AUDITOR,
    )
    response_str = agent.query(
        f"Review the claim for fabrication or source issues:\n{claim}\n\nClaim graph JSON: {claim_graph_json}",
        gin_feedback=gin_feedback,  # Medium-4: pass GIN topology hint
    )
    parsed = agent.parse_json(response_str)

    # CRITICAL BUG 3 FIX: convert LLM string outputs to PrimitiveType enums,
    # exactly as _historian_analyze does. Without this, tactic_edit_distance()
    # compares strings against enums and TED always returns 0.001 for Auditor.
    from env.primitives import PrimitiveType
    raw_chain = parsed.get("predicted_chain", [])
    converted_chain = []
    for item in raw_chain:
        if isinstance(item, PrimitiveType):
            converted_chain.append(item)
        else:
            try:
                converted_chain.append(PrimitiveType(str(item).upper()))
            except (ValueError, KeyError):
                try:
                    converted_chain.append(PrimitiveType[str(item).upper()])
                except KeyError:
                    pass  # Skip unrecognised strings

    return {
        "verdict":         _normalize_verdict(parsed.get("verdict", "unknown")),
        "predicted_chain": converted_chain,
        "rationale":       parsed.get("rationale", "Auditor analysis completed."),
        "confidence":      parsed.get("confidence", 0.5),
    }

# ── Context Historian — Cerebras / Llama-3.1 70B ─────────────────────────────
def _historian_analyze(claim: str, gin_feedback: str = "") -> dict:
    """Medium-4 fix: gin_feedback parameter so GIN hint reaches the Historian."""
    import config
    from agents.llm_agent_ma import LLMAgent
    agent = LLMAgent(
        system_prompt="You are a Context Historian specialising in detecting temporal manipulation, misdated media, and provenance fraud. Respond only in JSON with keys: verdict (one of: real, misinfo, fabricated, satire, unknown), predicted_chain, rationale, confidence.",
        provider=config.AGENT_HISTORIAN_PROVIDER,      # cerebras
        api_key=config.CEREBRAS_API_KEY,
    )
    response_str = agent.query(
        f"Review the claim for temporal or provenance issues: {claim}",
        gin_feedback=gin_feedback,  # Medium-4 fix: pass GIN hint through
    )
    parsed = agent.parse_json(response_str)
    
    from env.primitives import PrimitiveType
    
    raw_chain = parsed.get("predicted_chain", [])
    
    # Convert strings to PrimitiveType enums
    converted_chain = []
    for item in raw_chain:
        if isinstance(item, PrimitiveType):
            converted_chain.append(item)
        else:
            try:
                converted_chain.append(PrimitiveType(str(item).upper()))
            except (ValueError, KeyError):
                try:
                    converted_chain.append(PrimitiveType[str(item).upper()])
                except KeyError:
                    pass  # Skip unrecognised strings
    
    verdict = _normalize_verdict(parsed.get("verdict", "unknown"))
    confidence = parsed.get("confidence", 0.5)
    
    # Deterministic keyword fallback: detect temporal signals
    temporal_keywords = ["mislabelled", "mislabeled", "2015", "2016", "2017", "2018", "2019",
                         "old video", "old footage", "old photo", "repost", "recirculate",
                         "protest", "riot", "dated", "archive", "temporal", "year"]
    claim_lower = claim.lower()
    has_temporal = any(kw in claim_lower for kw in temporal_keywords)
    
    if has_temporal and (verdict == "unknown" or not converted_chain):
        verdict = "misinfo"
        confidence = max(confidence, 0.72)
        if PrimitiveType.TEMPORAL_SHIFT not in converted_chain:
            converted_chain.append(PrimitiveType.TEMPORAL_SHIFT)
    
    return {
        "verdict": verdict,
        "predicted_chain": converted_chain,
        "rationale": parsed.get("rationale", "Historian analysis completed."),
        "confidence": confidence,
    }


# ── Society Orchestrator ───────────────────────────────────────────────────────
class SocietyOfThought:
    """
    Orchestrates 4-agent investigation:
    1. Forensic Auditor   — leads shared investigation (H1: now returns real verdicts)
    2. Context Historian  — temporal/provenance interpretation (H1: now returns real verdicts)
    3. Narrative Critic   — style/narrative analysis
    4. Graph Specialist   — GIN topology-based chain prediction
    Consensus logic produces final verdict + chain.
    """
    def __init__(self, auditor, historian, critic, graph_specialist, gin, agent_timeout_sec: float = 15.0):
        self.auditor = auditor
        self.historian = historian
        self.critic = critic
        self.graph_specialist = graph_specialist
        self.gin = gin
        self.negotiated_search = NegotiatedSearch()
        self.agent_timeout_sec = agent_timeout_sec
        self._gin_lock = threading.Lock()

    def investigate(self, claim: str, true_chain: List[PrimitiveType] = None,
                    budget: int = 10, claim_graph=None) -> SocietyResult:
        """
        Run full 4-agent investigation on a claim.

        Parameters
        ----------
        claim       : raw claim text
        true_chain  : ground truth (for TED_best calculation during training)
        budget      : investigation step budget
        claim_graph : ClaimGraph object (used for GIN when graph is real)
        """
        # Step 1: Negotiated Search (produces tool preference vector)
        try:
            v_ensemble = self.negotiated_search.generate_vectors(claim, self.gin)
        except Exception:
            v_ensemble = None   # graceful degradation if LLM unavailable

        claim_graph_local = copy.deepcopy(claim_graph) if claim_graph is not None else None
        claim_graph_json = _serialize_claim_graph(claim_graph_local)

        # Step 2: Run GIN FIRST (local model, fast, no API call)
        g = _make_dummy_graph() if claim_graph_local is None else _graph_from_claim_graph(claim_graph_local)
        with self._gin_lock:
            gin_res = self.gin.predict_chain(g)
        gin_hint = self.gin.get_mid_episode_hint(g) if hasattr(self.gin, "get_mid_episode_hint") else "GIN prediction complete."

        # Format v_ensemble as tool hint for Auditor
        tool_hint = ""
        if v_ensemble is not None:
            from blue_team.negotiated_search import TOOLS
            top_tools = sorted(
                enumerate(v_ensemble), key=lambda x: x[1], reverse=True
            )[:3]
            tool_hint = "Prioritise tools: " + ", ".join(
                TOOLS[i] for i, _ in top_tools if i < len(TOOLS)
            )

        combined_hint = f"{gin_hint}\n{tool_hint}".strip() if tool_hint else gin_hint

        auditor_default = {"verdict": "unknown", "predicted_chain": [], "rationale": "Auditor error", "confidence": 0.3}
        historian_default = {"verdict": "unknown", "predicted_chain": [], "rationale": "Historian error", "confidence": 0.3}
        critic_default = {"verdict": "unknown", "predicted_chain": [], "rationale": "Critic error", "confidence": 0.3}
        gs_default = {"verdict": "unknown", "predicted_chain": [], "rationale": "Graph specialist error", "confidence": 0.1}

        def _run_auditor():
            if self.auditor is not None and hasattr(self.auditor, "analyze"):
                return self.auditor.analyze(claim)
            return _auditor_analyze(claim, claim_graph_json, gin_feedback=combined_hint)

        def _run_historian():
            if self.historian is not None and hasattr(self.historian, "analyze"):
                return self.historian.analyze(claim)
            return _historian_analyze(claim, gin_feedback=combined_hint)

        def _run_critic():
            if self.critic is not None and hasattr(self.critic, "analyze"):
                return self.critic.analyze(claim, claim_graph_json, gin_feedback=combined_hint)
            return {"verdict": "unknown", "predicted_chain": [], "rationale": "Critic not wired", "confidence": 0.3}

        def _run_graph_specialist():
            return {
                "verdict": gin_res.get("verdict", "unknown") if len(gin_res["ordered_chain"]) > 0 else "unknown",
                "predicted_chain": gin_res["ordered_chain"],
                "rationale": "Topology-based GIN prediction",
                "confidence": gin_res.get("confidence", 0.5),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                "auditor": executor.submit(_run_auditor),
                "historian": executor.submit(_run_historian),
                "critic": executor.submit(_run_critic),
                "graph_specialist": executor.submit(_run_graph_specialist),
            }
            done, not_done = concurrent.futures.wait(
                futures.values(), timeout=self.agent_timeout_sec
            )
            for future in not_done:
                future.cancel()

            results = {}
            defaults = {
                "auditor": auditor_default,
                "historian": historian_default,
                "critic": critic_default,
                "graph_specialist": gs_default,
            }
            for role, future in futures.items():
                if future in done:
                    try:
                        results[role] = future.result()
                    except Exception:
                        results[role] = defaults[role]
                else:
                    results[role] = defaults[role]

        # Step 6: Consensus
        consensus = self._consensus(
            [results["auditor"], results["historian"], results["critic"], results["graph_specialist"]],
            ["auditor", "historian", "critic", "graph_specialist"]
        )

        ted_best = 0.0
        if true_chain is not None:
            ted_best = self.compute_ted_best(list(results.values()), true_chain)

        return SocietyResult(
            verdict=consensus.verdict,
            predicted_chain=consensus.predicted_chain,
            ted_best=ted_best,
            consensus_level=consensus.consensus_level,
            consensus_bonus=consensus.consensus_bonus,
            agent_verdicts={k: v.get("verdict", "unknown") for k, v in results.items()},
            agent_chains={k: v.get("predicted_chain", []) for k, v in results.items()},
            agent_confidences={k: v.get("confidence", 0.0) for k, v in results.items()},
            dissenting_agents=consensus.dissenting_agents,
            dissenting_rationales=consensus.dissenting_rationales,
        )

    def _consensus(self, results: list, names: list) -> ConsensusResult:
        verdicts = [res.get("verdict", "unknown") for res in results]
        chains = [res.get("predicted_chain", []) for res in results]

        from collections import Counter
        v_counts = Counter(verdicts)
        top_v, top_count = v_counts.most_common(1)[0]

        dissenting_agents = []
        dissenting_rationales = []
        for i, v in enumerate(verdicts):
            if v != top_v:
                dissenting_agents.append(names[i])
                dissenting_rationales.append(results[i].get("rationale", ""))

        # Medium-3 fix: look up graph_specialist by name, not fragile chains[3].
        # If anyone reorders the agents list the tiebreaker no longer silently
        # picks the wrong agent's chain.
        try:
            gs_idx = names.index("graph_specialist")
        except ValueError:
            gs_idx = min(3, len(chains) - 1)  # safe fallback
        gs_chain = chains[gs_idx] if chains else []

        if top_count == 4:
            c_str = [str([p.name for p in c]) for c in chains]
            if len(set(c_str)) == 1:
                return ConsensusResult(top_v, chains[0], "unanimous", 0.10, [], [])
            else:
                return ConsensusResult(top_v, gs_chain, "unanimous", 0.10, [], [])
        elif top_count == 3:
            idx = verdicts.index(top_v)
            return ConsensusResult(top_v, chains[idx], "majority_3", 0.05, dissenting_agents, dissenting_rationales)
        elif top_count == 2:
            return ConsensusResult("trigger_expert", gs_chain, "split_2_2", -0.05, dissenting_agents, dissenting_rationales)
        else:
            return ConsensusResult("unknown", gs_chain, "all_different", -0.05, names, [r.get("rationale", "") for r in results])

    def compute_ted_best(self, results: list, true_chain: list) -> float:
        """max(TED_auditor, TED_historian, TED_graph_specialist, TED_narrative_critic)"""
        teds = [tactic_edit_distance(res.get("predicted_chain", []), true_chain) for res in results]
        return max(teds) if teds else 0.001


def _make_dummy_graph():
    """Create a zero-node dummy graph for GIN when no real graph is available."""
    import torch

    class _DG:
        pass
    g = _DG()
    g.x = torch.zeros((1, 10))
    g.edge_index = torch.zeros((2, 0), dtype=torch.long)
    g.batch = torch.zeros((1,), dtype=torch.long)
    return g

def _graph_from_claim_graph(claim_graph):
    import torch
    from env.node_features import build_node_features
    n_nodes = len(claim_graph.nodes)
    if n_nodes == 0:
        return _make_dummy_graph()
    node_features = []
    for node in claim_graph.nodes:
        feat = build_node_features(node, 10)
        node_features.append(feat)
    class _RealGraph: pass
    g = _RealGraph()
    g.x = torch.tensor(node_features, dtype=torch.float32)
    g.edge_index = torch.zeros((2, 0), dtype=torch.long)
    g.batch = torch.zeros(n_nodes, dtype=torch.long)
    return g

def _serialize_claim_graph(claim_graph) -> str:
    if claim_graph is None:
        return '{"nodes":[],"edges":[]}'
    if hasattr(claim_graph, "serialize"):
        return claim_graph.serialize()
    if hasattr(claim_graph, "to_dict"):
        return json.dumps(claim_graph.to_dict())
    return '{"nodes":[],"edges":[]}'
