"""
FORGE-MA Gymnasium-compatible Environment.
SPEC (Master Prompt §Layer7):
  - step() returns (obs, reward, terminated, truncated, info)
  - Episode budget: 10 steps default, configurable
  - Red agent perturbs claim graph each step
  - Blue team evaluates at episode end
  - reward computed via hierarchical_reward shaper
  - reset() seeds a new claim and clears all episode state
  - Observation space: flat dict with claim text + budget remaining

High Bug H2 fix: step() was text-appending primitive names instead of
building a ClaimGraph. Now maintains a real ClaimGraph, adding ClaimNode
with DISARM fingerprints per Red action. GIN receives the real graph
at evaluate time, not a structurally empty dummy.
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
from blue_team.gin_predictor import GINPredictor

# Sample claims for demo episodes (no LLM needed).
#
# Manipulated claims (true_chain non-empty) and real-news claims (true_chain empty)
# must both appear so the Blue GIN learns the "no manipulation" decision boundary.
# Without real-news positives, the supervised loss can never learn to predict
# all-zeros on a clean graph, and every claim collapses to "misinfo".
_DEMO_CLAIMS = [
    # ── Manipulated ──────────────────────────────────────────────────────────
    ("Vaccines cause autism, leaked documents confirm.", [PrimitiveType.QUOTE_FABRICATE, PrimitiveType.SOURCE_LAUNDER]),
    ("Video shows 2015 protest mislabelled as 2024 riots.", [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.CONTEXT_STRIP]),
    ("Politician quoted saying 'immigrants are criminals' — source: satirical site.", [PrimitiveType.SATIRE_REFRAME, PrimitiveType.QUOTE_FABRICATE]),
    ("Study claims 90% efficacy — journal retracted, still circulating.", [PrimitiveType.CITATION_FORGE, PrimitiveType.NETWORK_AMPLIFY]),
    ("Scientist replaced with lookalike in doctored photo.", [PrimitiveType.ENTITY_SUBSTITUTE]),
    # ── Real news (empty chain) — required so Blue can learn the negative class.
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
    Gymnasium-style adversarial RL environment for FORGE-MA.

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
        from runtime import get_blue_gin
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

        # H2: initialise real ClaimGraph with root node.
        # Critical-1 fix: keep a pristine copy so reward shaper gets a genuine
        # before/after pair for the plausibility delta.
        with self._graph_lock:
            self._claim_graph = self._build_initial_graph(claim_text)
            self._initial_claim_graph = self._build_initial_graph(claim_text)  # immutable snapshot

        obs = self._build_obs()
        info = {"true_chain": [p.value for p in true_chain]}
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
        Both _initial_claim_graph AND _claim_graph are set to the R1 graph
        (Red Team perturbs from this richer starting point).
        _initial_claim_graph is a deep copy so plausibility delta is non-zero.
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

        # Both initial and current start as the R1-converted graph.
        # Deep copy ensures they are separate objects so plb_delta != 0.
        with self._graph_lock:
            self._initial_claim_graph = copy.deepcopy(initial_graph)
            self._claim_graph = copy.deepcopy(initial_graph)

        self.red_agent.reset()
        self.red_step_rewarder.reset()

        obs = self._build_obs()
        info = {
            "true_chain": [p.value for p in true_chain],
            "pipeline_mode": True,
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

    def _build_obs(self) -> Dict:
        with self._graph_lock:
            return {
                "claim_text":        self._claim_text,
                "budget_remaining":  max(0, self.config.budget - self._steps),
                "steps_taken":       self._steps,
                "red_chain":         [p.value for p in self.red_agent.current_chain],
                "graph_nodes":       len(self._claim_graph.nodes) if self._claim_graph else 0,
            }

    def _evaluate_episode(self) -> Tuple[float, EpisodeOutput]:
        """Run GIN + lightweight consensus and compute hierarchical reward."""
        x, edge_index = self._graph_to_tensors()

        class _G:
            pass
        g = _G()
        g.x = x
        g.edge_index = edge_index
        g.batch = torch.zeros(x.size(0), dtype=torch.long)

        gin_result = self.gin.predict_chain(g)
        predicted_chain = gin_result["ordered_chain"]
        gin_confidence  = gin_result.get("confidence", 0.5)

        # Build a real MC-Dropout ensemble of N=4 stochastic forward passes.
        # This replaces the prior `[predicted_chain] * 4` (one prediction copied
        # four times — a fake "multi-agent" signal). Each ensemble member is an
        # independent dropout sample, so consensus/entropy reward components see
        # genuine inter-agent variation rather than zero variance.
        ensemble = self.gin.predict_chain_ensemble(g, n_agents=4)
        predicted_chains_ensemble = [m["ordered_chain"] for m in ensemble] or [predicted_chain]

        # Real consensus computed from the ensemble: count agents predicting
        # the modal chain. Replaces the prior fixed mapping from confidence.
        from collections import Counter
        chain_keys = [tuple(p.value if hasattr(p, "value") else str(p) for p in c) for c in predicted_chains_ensemble]
        if chain_keys:
            top_count = Counter(chain_keys).most_common(1)[0][1]
            n_total = len(chain_keys)
            if top_count == n_total:
                consensus_level = "unanimous"
            elif top_count >= 3:
                consensus_level = "majority_3"
            elif top_count == 2:
                consensus_level = "split_2_2"
            else:
                consensus_level = "all_different"
        else:
            consensus_level = "all_different"

        # Real expert: approve if predicted chain non-empty and confidence adequate
        expert_decision = "APPROVE" if (predicted_chain and gin_confidence >= 0.45) else "REJECT"

        with self._graph_lock:
            graph_before = copy.deepcopy(self._initial_claim_graph)
            graph_after = copy.deepcopy(self._claim_graph)

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
            agent_verdicts={"gin": gin_result.get("verdict", "unknown")},
            red_step_rewards=self._red_step_rewards,
        )
        return r.total, ep

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
