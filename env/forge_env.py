"""
FORGE-RL Gymnasium-compatible adversarial environment.
step() returns (obs, reward, terminated, truncated, info).
Blue GIN evaluates the claim graph at episode end.
"""
from __future__ import annotations
import copy
import random
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from env.primitives import PrimitiveType, K_MAX, FINGERPRINT_KEYS
from env.claim_graph_ma import ClaimGraph, ClaimNode, EvidenceEdge
from env.episode_output import EpisodeOutput
from rewards.hierarchical_reward import compute_reward
from rewards.red_step_reward import RedStepReward
from red_team.red_agent import RedAgent
from runtime import get_blue_gin

_DEMO_CLAIMS = [
    ("Vaccines cause autism, leaked documents confirm.", [PrimitiveType.QUOTE_FABRICATE, PrimitiveType.SOURCE_LAUNDER]),
    ("Video shows 2015 protest mislabelled as 2024 riots.", [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.CONTEXT_STRIP]),
    ("Politician quoted saying 'immigrants are criminals' — source: satirical site.", [PrimitiveType.SATIRE_REFRAME, PrimitiveType.QUOTE_FABRICATE]),
    ("Study claims 90% efficacy — journal retracted, still circulating.", [PrimitiveType.CITATION_FORGE, PrimitiveType.NETWORK_AMPLIFY]),
    ("Scientist replaced with lookalike in doctored photo.", [PrimitiveType.ENTITY_SUBSTITUTE]),
    ("NASA's Perseverance rover collected its 24th rock sample on Mars.", []),
    ("The European Central Bank held its benchmark interest rate steady at the September meeting.", []),
    ("Researchers at MIT published a peer-reviewed paper on lithium-sulfur battery cycle stability.", []),
    ("The IPCC released its latest synthesis report on global temperature anomalies.", []),
    ("WHO confirmed the elimination of trachoma as a public health problem in two more countries.", []),
]


@dataclass
class ForgeEnvConfig:
    budget: int = 10
    node_feat_dim: int = 10
    seed: Optional[int] = None


class ForgeEnv:
    """
    Gymnasium-style adversarial RL environment for FORGE-RL.

    Agents:
      - Red  : RedAgent (perturbs claim graph each step)
      - Blue : BlueGIN  (evaluates graph at episode end)

    Episode flow:
      reset() → [step() × budget] → terminated=True → episode_output
    """

    def __init__(self, config: Optional[ForgeEnvConfig] = None):
        self.config = config or ForgeEnvConfig()
        self.red_agent = RedAgent(mode="greedy")
        # Use the process-wide GIN singleton so training updates here are
        # visible to the deployed server (which used to hold a separate copy).
        self.gin = get_blue_gin()
        self.red_step_rewarder = RedStepReward(self.gin, alpha=1.0)

        # Episode state (reset each episode)
        self._claim_text: str = ""
        self._claim_text_initial: str = ""
        self._true_chain: List[PrimitiveType] = []
        self._steps: int = 0
        self._useful_tools: int = 0
        self._done: bool = False
        self._red_step_rewards: List[float] = []
        self._episode_output: Optional[EpisodeOutput] = None
        self._claim_graph: Optional[ClaimGraph] = None          # current (mutated) graph
        self._initial_claim_graph: Optional[ClaimGraph] = None  # Critical-1: snapshot at reset
        self._graph_lock = threading.RLock()

        if self.config.seed is not None:
            random.seed(self.config.seed)

    # ── Core Gymnasium API ──────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        if seed is not None:
            random.seed(seed)

        claim_text, true_chain = random.choice(_DEMO_CLAIMS)
        self._claim_text = claim_text
        self._claim_text_initial = claim_text
        self._true_chain = list(true_chain)
        self._steps = 0
        self._useful_tools = 0
        self._done = False
        self._red_step_rewards = []
        self._episode_output = None
        self.red_agent.reset()
        self.red_step_rewarder.reset()

        with self._graph_lock:
            self._claim_graph = self._build_initial_graph(claim_text)
            self._initial_claim_graph = self._build_initial_graph(claim_text)

        obs = self._build_obs()
        info = {
            "true_chain":       [p.value for p in true_chain],   # strings
            "true_chain_enums": list(true_chain),                # PrimitiveType enums
            "claim_text":       claim_text,
            "episode_id":       str(uuid.uuid4())[:8],
            "task":             "forge_ma_adversarial",
        }
        return obs, info

    def reset_from_r1(
        self,
        initial_graph: ClaimGraph,
        true_chain: List[PrimitiveType],
        claim_text: str,
        seed: Optional[int] = None,
    ) -> tuple:
        """
        Pipeline-mode reset (Master Prompt v9.0 §3.2).
        Starts from a pre-built R1→R2 converted graph instead of a demo claim.
        """
        if seed is not None:
            random.seed(seed)

        self._claim_text = claim_text
        self._claim_text_initial = claim_text
        self._true_chain = list(true_chain)
        self._steps = 0
        self._useful_tools = 0
        self._done = False
        self._red_step_rewards = []
        self._episode_output = None

        # Convert R1 graph to R2 graph if necessary
        from env.claim_graph_ma import ClaimGraph as R2ClaimGraph, ClaimNode as R2ClaimNode, EvidenceEdge as R2EvidenceEdge
        
        if not isinstance(initial_graph, R2ClaimGraph):
            r2_nodes = []
            r1_nodes = getattr(initial_graph, "nodes", {})
            if isinstance(r1_nodes, dict):
                r1_node_list = list(r1_nodes.values())
            else:
                r1_node_list = list(r1_nodes)
                
            for node in r1_node_list:
                node_id = getattr(node, "id", getattr(node, "node_id", str(id(node))))
                r2_node = R2ClaimNode(
                    id=node_id,
                    text=node.text if hasattr(node, "text") else str(node),
                    domain=getattr(node, "domain", "unknown"),
                    trust_score=getattr(node, "trust_score", 0.5),
                    is_retrieved=getattr(node, "is_retrieved", False),
                    injected=getattr(node, "injected", False),
                    primitive=None,
                    fingerprints={},
                )
                r2_nodes.append(r2_node)
                
            r2_edges = []
            for edge in getattr(initial_graph, "edges", []):
                r2_edge = R2EvidenceEdge(
                    source_id=getattr(edge, "source", getattr(edge, "source_id", "")),
                    target_id=getattr(edge, "target", getattr(edge, "target_id", "")),
                    relation=getattr(edge, "relation", "unknown"),
                    weight=getattr(edge, "weight", 0.5),
                    injected=getattr(edge, "injected", False)
                )
                r2_edges.append(r2_edge)
                
            initial_graph = R2ClaimGraph(
                nodes=r2_nodes,
                edges=r2_edges,
                root_id=getattr(initial_graph, "root_claim_id", getattr(initial_graph, "root_id", "root-0"))
            )

        # Both initial and current start as the R1-converted graph.
        # Deep copy ensures they are separate objects so plb_delta != 0.
        with self._graph_lock:
            self._initial_claim_graph = copy.deepcopy(initial_graph)
            self._claim_graph = copy.deepcopy(initial_graph)

        self.red_agent.reset()
        self.red_step_rewarder.reset()

        obs = self._build_obs()
        info = {
            "true_chain":       [p.value for p in true_chain],
            "true_chain_enums": list(true_chain),
            "claim_text":       claim_text,
            "episode_id":       str(uuid.uuid4())[:8],
            "task":             "forge_ma_adversarial",
            "pipeline_mode":    True,
        }
        return obs, info

    def step(self, action: Optional[Any] = None) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        One environment step. `action` is ignored — Red agent is autonomous.
        Returns (obs, reward, terminated, truncated, info).
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        self._steps += 1
        budget_remaining = self.config.budget - self._steps

        # Build graph tensor for HAE scoring
        x, edge_index = self._graph_to_tensors()
        red_action = self.red_agent.propose_action(x, edge_index, budget_remaining)

        if red_action is not None:
            # H2: Apply action to real ClaimGraph (not just text-append).
            self._apply_red_action_to_graph(red_action)
            # Medium-1 fix: do NOT append primitive names to _claim_text.
            # The bracketed tokens ([QUOTE_FABRICATE] etc.) were a debug
            # artefact that corrupted the plausibility scorer's linguistic
            # coherence signal. The ClaimGraph carries all structural info.
            self._useful_tools += 1
            
            # Compute dense step reward based on GIN probability shift
            x_after, edge_index_after = self._graph_to_tensors()
            
            # GINPredictor takes Data-like object, but RedStepReward takes duck-typed object.
            # We can mock it or just pass dict. RedStepReward expects `predict_chain(graph_data)`
            # and `graph_data` just needs to have x and edge_index or be accepted by GINPredictor.
            from torch_geometric.data import Data
            batch_data = Data(x=x_after, edge_index=edge_index_after)
            
            # Determine index of the primitive if we can
            prim_idx = None
            if red_action.primitive:
                from env.primitives import PrimitiveType
                all_prims = list(PrimitiveType)
                if red_action.primitive in all_prims:
                    prim_idx = all_prims.index(red_action.primitive)
                    
            r_step = self.red_step_rewarder.compute(batch_data, primitive_idx=prim_idx)
            self._red_step_rewards.append(r_step)

        terminated = (self._steps >= self.config.budget) or (budget_remaining <= 0)

        if terminated:
            reward, ep_output = self._evaluate_episode()
            self._done = True
            self._episode_output = ep_output
        else:
            reward = 0.0
            ep_output = None

        obs = self._build_obs()
        with self._graph_lock:
            graph_nodes = len(self._claim_graph.nodes) if self._claim_graph else 0
        info = {
            "steps": self._steps,
            "red_action": str(red_action) if red_action else "none",
            "episode_output": ep_output,
            "graph_nodes": graph_nodes,
            "red_step_rewards": self._red_step_rewards,
            "red_agent_history": self.red_agent.history,
            # Blue Team training: final graph state + ground-truth chain
            "blue_graph_x": x if terminated else None,
            "blue_graph_edge_index": edge_index if terminated else None,
            "true_chain": self._true_chain if terminated else None,
        }
        return obs, reward, terminated, False, info

    def close(self):
        pass

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _build_initial_graph(self, claim_text: str) -> ClaimGraph:
        """H2: Build a real ClaimGraph with root node from the claim text."""
        root_id = "root-0"
        root_node = ClaimNode(
            id=root_id,
            text=claim_text,
            domain="root",
            trust_score=0.5,
            is_retrieved=False,
            injected=False,
            primitive=None,
            fingerprints={},
        )
        return ClaimGraph(nodes=[root_node], edges=[], root_id=root_id)

    def _apply_red_action_to_graph(self, red_action) -> None:
        """
        H2: Apply a Red agent action to the real ClaimGraph.
        Inserts a new ClaimNode with the primitive's DISARM fingerprint set,
        and links it to the root with an 'adversarial' edge.
        """
        with self._graph_lock:
            prim = red_action.primitive
            fingerprint_key = FINGERPRINT_KEYS.get(prim, "unknown")
            node_id = f"adv-{self._steps}-{prim.value}"

            new_node = ClaimNode(
                id=node_id,
                text=f"[RED] {prim.name} via {red_action.tool_label}",
                domain="adversarial",
                # Adversarial nodes have low trust score to challenge Blue coherence
                trust_score=max(0.05, 0.3 - 0.05 * self._steps),
                is_retrieved=False,
                injected=True,
                primitive=prim,
                fingerprints={fingerprint_key: True},
            )

            # Edge: root → new adversarial node
            edge = EvidenceEdge(
                source_id=self._claim_graph.root_id,
                target_id=node_id,
                relation="adversarial",
                weight=0.4,
                injected=True,
            )

            self._claim_graph.nodes.append(new_node)
            self._claim_graph.edges.append(edge)

    def _graph_to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        H2: Convert ClaimGraph to (x, edge_index) tensors consumed by HAE/GIN.
        Node features: [trust_score, is_retrieved, injected, +fingerprint flags]
        """
        with self._graph_lock:
            if self._claim_graph is None or len(self._claim_graph.nodes) == 0:
                x = torch.zeros(1, self.config.node_feat_dim)
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                return x, edge_index

            nodes = self._claim_graph.nodes
            node_id_to_idx = {n.id: i for i, n in enumerate(nodes)}

            # Build feature matrix
            from env.node_features import build_node_features
            feats = []
            for node in nodes:
                feat = build_node_features(node, self.config.node_feat_dim)
                feats.append(feat)

            x = torch.tensor(feats, dtype=torch.float32)

            # Build edge index
            if len(self._claim_graph.edges) > 0:
                src_ids = [node_id_to_idx.get(e.source_id, 0) for e in self._claim_graph.edges]
                dst_ids = [node_id_to_idx.get(e.target_id, 0) for e in self._claim_graph.edges]
                edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

        return x, edge_index

    def _build_obs(self) -> "np.ndarray":
        """
        Returns flat numpy array compatible with Gymnasium spec.
        FIX 5C: was returning dict — Gymnasium requires np.ndarray.
        PPOAgent._flatten_obs() now passes through directly.
        """
        import numpy as np
        with self._graph_lock:
            budget_remaining = max(0, self.config.budget - self._steps)
            graph_nodes = len(self._claim_graph.nodes) if self._claim_graph else 0

        obs = np.zeros(3859, dtype=np.float32)
        obs[0] = float(budget_remaining) / max(self.config.budget, 1)
        obs[1] = float(self._steps) / max(self.config.budget, 1)
        obs[2] = min(float(graph_nodes) / 20.0, 1.0)

        # Red chain one-hot: 8 primitives × 4 slots = 32 dims at indices 3–34
        all_prims = list(PrimitiveType)
        n_prims = len(all_prims)  # 8
        chain = self.red_agent.current_chain
        for slot in range(4):
            for prim_idx, prim in enumerate(all_prims):
                feat_idx = 3 + slot * n_prims + prim_idx
                if feat_idx >= 3859:
                    break
                obs[feat_idx] = 1.0 if (
                    slot < len(chain) and chain[slot] == prim
                ) else 0.0

        return obs

    def _build_obs_dict(self) -> dict:
        """
        Returns dict observation for Society/LLM consumers that need claim_text.
        Use _build_obs() for Gymnasium-compatible numpy obs.
        """
        with self._graph_lock:
            return {
                "claim_text":       self._claim_text,
                "budget_remaining": max(0, self.config.budget - self._steps),
                "steps_taken":      self._steps,
                "red_chain":        [p.value for p in self.red_agent.current_chain],
                "graph_nodes":      len(self._claim_graph.nodes)
                                    if self._claim_graph else 0,
            }

    def _evaluate_episode(self) -> Tuple[float, "EpisodeOutput"]:
        """
        Run SocietyOfThought + real ExpertReviewerAgent and compute hierarchical reward.

        FIX 1: ExpertReviewerAgent is now called properly.
        FIX 2: Consensus detection uses a min-diversity guard to prevent
                unanimous=+0.10 becoming a free bonus after convergence.
        """
        # 1. Start Blue Team Society of Thought
        if not hasattr(self, 'society'):
            # Factory method handles all 4 agents + GIN
            from blue_team.society_of_thought import SocietyOfThought
            self.society = SocietyOfThought.create_default(gin=self.gin)

        with self._graph_lock:
            current_graph = self._claim_graph

        # FIX 5B: pre-declare _society_result so it survives the try/except
        _society_result = None
        society_verdict = "unknown"

        # 2. Run the investigation
        try:
            society_result = self.society.investigate(
                claim=self._claim_text,
                true_chain=self._true_chain,
                budget=self.config.budget,
                claim_graph=current_graph
            )
            _society_result = society_result
            society_verdict = society_result.agent_verdicts.get("gin", "unknown")
        except Exception as _soe:
            import logging as _log
            _log.getLogger("forge.env").warning(
                "SocietyOfThought.investigate failed: %s", _soe
            )
            # Build a minimal stub so the rest of _evaluate_episode can proceed
            class _StubResult:
                predicted_chain = []
                consensus_level = 0.0
                agent_confidences = {"gin": 0.5}
                agent_verdicts = {"gin": "unknown"}
                agent_chains = {}
            _society_result = _StubResult()
            society_result = _society_result

        predicted_chain = society_result.predicted_chain
        consensus_level = society_result.consensus_level
        gin_confidence = society_result.agent_confidences.get("gin", 0.5)
        gin_result = {"verdict": society_verdict}

        predicted_chains_ensemble = []
        for agent_name, chain in society_result.agent_chains.items():
            if chain:
                predicted_chains_ensemble.append(chain)
        if not predicted_chains_ensemble:
            predicted_chains_ensemble = [predicted_chain]

        # ── Expert decision — FIX: call real ExpertReviewerAgent ────────
        # Previously hardcoded to:
        #   "APPROVE" if (predicted_chain and gin_confidence >= 0.45)
        # This bypassed the entire Snorkel AI bonus implementation.
        try:
            from agents.expert_reviewer_agent import ExpertReviewerAgent
            if not hasattr(self, "_expert_reviewer"):
                self._expert_reviewer = ExpertReviewerAgent(mode="ising")

            # Compute recall for expert evaluation
            true_set  = set(
                p.value if hasattr(p, "value") else str(p)
                for p in self._true_chain
            )
            pred_set  = set(
                p.value if hasattr(p, "value") else str(p)
                for p in predicted_chain
            )
            recall = (
                len(pred_set & true_set) / len(true_set)
                if true_set else 1.0
            )
            hallucinations = len(pred_set - true_set)
            budget_used = self._steps / max(self.config.budget, 1)
            # Determine generation from GIN checkpoint name if possible
            generation = getattr(self, "_training_generation", 0)

            expert_decision = self._expert_reviewer.get_decision(
                verdict_correct=(
                    gin_result.get("verdict", "") in
                    {"fabricated", "misinfo", "satire", "out_of_context"}
                    if self._true_chain
                    else gin_result.get("verdict", "") in {"real", "verified"}
                ),
                recall=recall,
                confidence=gin_confidence,
                hallucinations=hallucinations,
                budget_used=budget_used,
                steps=self._steps,
                tools_called=self._useful_tools,
                coverage=min(1.0, len(pred_set) / max(len(true_set), 1)),
                generation=generation,
            )
        except Exception as _e:
            import logging
            logging.getLogger("forge.env").warning(
                "ExpertReviewerAgent failed, using confidence threshold: %s", _e
            )
            expert_decision = (
                "APPROVE"
                if (predicted_chain and gin_confidence >= 0.45)
                else "REJECT"
            )

        # ── Plausibility: use graph snapshots ──────────────────────────
        with self._graph_lock:
            graph_before = copy.deepcopy(self._initial_claim_graph)
            graph_after  = copy.deepcopy(self._claim_graph)

        r = compute_reward(
            predicted_chains=predicted_chains_ensemble,
            true_chain=self._true_chain,
            claim_text_before=self._claim_text_initial,
            claim_text_after=self._claim_text,
            consensus_level=consensus_level,
            expert_decision=expert_decision,
            steps_taken=self._steps,
            budget_limit=self.config.budget,
            useful_tools_called=self._useful_tools,
            claim_graph_before=graph_before,
            claim_graph_after=graph_after,
        )

        ep = EpisodeOutput.build(
            verdict=gin_result.get("verdict", "unknown"),
            predicted_chain=predicted_chain,
            true_chain=self._true_chain,
            reward=r,
            consensus_level=consensus_level,
            expert_decision=expert_decision,
            steps_taken=self._steps,
            budget_limit=self.config.budget,
            useful_tools=self._useful_tools,
            # FIX 5B: use real per-agent verdicts from _society_result
            agent_verdicts=(
                _society_result.agent_verdicts
                if _society_result is not None
                else {"gin": gin_result.get("verdict", "unknown")}
            ),
            red_step_rewards=self._red_step_rewards,
        )
        return r.total, ep

    @property
    def training_generation(self) -> int:
        return getattr(self, "_training_generation", 0)

    @training_generation.setter
    def training_generation(self, gen: int) -> None:
        self._training_generation = gen
        # Also propagate to expert reviewer if initialised
        if hasattr(self, "_expert_reviewer"):
            self._expert_reviewer.episode_count = gen * 50

    @property
    def episode_output(self) -> Optional[EpisodeOutput]:
        return self._episode_output

    @property
    def budget(self) -> int:
        return self.config.budget

    @property
    def claim_graph(self) -> Optional[ClaimGraph]:
        with self._graph_lock:
            return self._claim_graph

    @contextmanager
    def graph_lock(self):
        with self._graph_lock:
            yield
