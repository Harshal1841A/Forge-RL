"""
FORGE — Main RL Environment (Gymnasium-compatible)
OpenEnv / Gymnasium interface for the MisInfo Forensics task.
"""

from __future__ import annotations
import copy
import logging
import random
import threading
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import types

import gymnasium as gym
import numpy as np
import torch

from env.claim_graph import ClaimGraph
from env.tasks import TASK_REGISTRY, BaseTask
from env.reward import (
    compute_potential,
    shaped_step_reward,
    verdict_reward,
    tool_call_reward,
    efficiency_penalty,
)
import config

logger = logging.getLogger(__name__)

# ─── Action indices ───────────────────────────────────────────────────────────
ACTIONS = [
    "query_source",         # 0
    "trace_origin",         # 1
    "cross_reference",      # 2
    "request_context",      # 3
    "entity_link",          # 4
    "temporal_audit",       # 5
    "network_cluster",      # 6
    "flag_manipulation",    # 7  — free action (no step cost)
    "submit_verdict_real",          # 8
    "submit_verdict_misinfo",       # 9
    "submit_verdict_satire",        # 10
    "submit_verdict_out_of_context",  # 11
    "submit_verdict_fabricated",    # 12
]
N_ACTIONS = len(ACTIONS)

VERDICT_ACTIONS = {
    8: "real",
    9: "misinfo",
    10: "satire",
    11: "out_of_context",
    12: "fabricated",
}


class MisInfoForensicsEnv(gym.Env):
    """
    Gymnasium-compatible environment for hierarchical misinformation investigation.

    Observation: flat numpy vector encoding [claim_embedding | tool_history |
                  budget_remaining | evidence_coverage | source_diversity |
                  contradiction_count | manipulation_flag | step_count]

    Action: Discrete(13) — tool calls + manipulation flag + 5 verdict options
    """

    metadata = {"render_modes": ["human", "json"]}

    def __init__(
        self,
        task_names: Optional[List[str]] = None,
        difficulty: int = 1,
        seed: int = 0,
        use_live_tools: bool = False,   # False = simulated tools (no API needed)
        render_mode: str = "json",
        curriculum_stage: int = 0,
        budget_multiplier: float = 1.0,
    ):
        super().__init__()
        self.task_names = task_names or list(TASK_REGISTRY.keys())
        self.difficulty = difficulty
        self.base_seed = seed
        self.use_live_tools = use_live_tools
        self.render_mode = render_mode
        self.curriculum_stage = curriculum_stage

        # Build task generators
        self.tasks: List[BaseTask] = [
            TASK_REGISTRY[name]() for name in self.task_names
        ]

        # Import tool registry (lazy to avoid API calls at init)
        if use_live_tools:
            from tools.tool_registry import ToolRegistry
            self.tool_registry = ToolRegistry()
        else:
            from tools.tool_registry import SimulatedToolRegistry
            self.tool_registry = SimulatedToolRegistry()

        # ── Spaces ────────────────────────────────────────────────────────────
        # Obs dim v2.0: (MAX_OBSERVATION_NODES * CLAIM_EMBED_DIM) + N_ACTIONS + 6
        # = 10 * 384 + 13 + 6 = 3859
        self.obs_dim = config.MAX_OBSERVATION_NODES * config.CLAIM_EMBED_DIM + N_ACTIONS + 6
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(N_ACTIONS)

        # ── Episode state ─────────────────────────────────────────────────────
        self.graph: Optional[ClaimGraph] = None
        self.current_task: Optional[BaseTask] = None
        self.steps: int = 0
        self.max_steps: int = config.MAX_EPISODE_STEPS
        self.manipulation_flagged: bool = False
        self.tool_call_counts: Dict[str, int] = {}
        self.episode_id: str = ""
        self.budget_multiplier: float = budget_multiplier
        self._prev_potential: float = 0.0
        self._tool_history: np.ndarray = np.zeros(N_ACTIONS, dtype=np.float32)
        self._done: bool = True
        self._graph_lock = threading.RLock()

        # Embedder (local, free)
        self._embedder = None   # lazy-loaded

    # ── Gymnasium Interface ───────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else random.randint(0, 2**31)

        # Sample task
        self.current_task = random.choice(self.tasks)
        with self._graph_lock:
            self.graph = self.current_task.generate(
                difficulty=self.difficulty, seed=ep_seed
            )

            # Dynamic step budget — scaled by curriculum budget multiplier
            self.max_steps = min(
                int(
                    (config.BASE_EPISODE_STEPS
                     + self.graph.num_tactics * config.STEP_COMPLEXITY_BONUS)
                    * self.budget_multiplier
                ),
                config.MAX_EPISODE_STEPS,
            )
            graph_id = self.graph.graph_id

        self.steps = 0
        self.manipulation_flagged = False
        self.manipulation_flag_count = 0  # NEW: limit free hits
        self.tool_call_counts = {}
        self.episode_id = str(uuid.uuid4())
        self._tool_history = np.zeros(N_ACTIONS, dtype=np.float32)
        self._done = False

        # Initialise prev potential for reward shaping
        with self._graph_lock:
            self._prev_potential = compute_potential(self.graph)

        obs = self._build_obs()
        info = {
            "episode_id": self.episode_id,
            "task_id": self.current_task.task_id,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
            "graph_id": graph_id,
        }
        logger.info("[START] episode=%s task=%s difficulty=%d",
                    self.episode_id, self.current_task.task_id, self.difficulty)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._done, "Call reset() before step()"
        assert self.graph is not None, "graph must be set — call reset() first"
        assert self.current_task is not None, "current_task must be set — call reset() first"
        action_name = ACTIONS[action]
        reward = config.REWARD_CLIP_MIN
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "step": self.steps,
            "action": action_name,
        }
        with self._graph_lock:
            prev_graph = copy.deepcopy(self.graph)

            # ── Free actions (no step cost) ───────────────────────────────────
            if action_name == "flag_manipulation":
                self.manipulation_flagged = True
                reward = config.REWARD_CLIP_MIN
                info["flagged"] = True
                logger.info("[STEP] %s step=%d action=flag_manipulation",
                            self.episode_id, self.steps)
                # Check truncation — must happen even for free actions
                truncated = self.steps >= self.max_steps
                if truncated:
                    self._done = True
                    logger.info("[END] %s truncated at step %d", self.episode_id, self.steps)
                obs = self._build_obs()
                return obs, reward, False, truncated, info

            # ── Verdict actions ───────────────────────────────────────────────
            if action in VERDICT_ACTIONS:
                predicted_label = VERDICT_ACTIONS[action]
                confidence = self._estimate_confidence()
                terminal_r = verdict_reward(
                    predicted_label=predicted_label,
                    true_label=self.graph.true_label,
                    predicted_confidence=confidence,
                    steps_used=self.steps,
                    max_steps=self.max_steps,
                    manipulation_flagged=self.manipulation_flagged,
                    true_manipulation=self.current_task.has_manipulation(self.graph),
                )
                terminal_r += efficiency_penalty(self.steps, self.graph.difficulty)

                # Policy invariance for terminal states: shaping = 0 - prev_potential
                terminal_r -= self._prev_potential

                # RL hardening: return raw reward for internal logic, but clip for the Gym interface
                info["raw_reward"] = terminal_r
                reward = float(np.clip(terminal_r, config.REWARD_CLIP_MIN, config.REWARD_CLIP_MAX))

                terminated = True
                self._done = True

                # Consult the shared Blue GIN so the trained model influences
                # the deployed verdict signal. Without this, server-side
                # episodes never benefited from any training that ran in
                # ForgeEnv. Surfaced in info[] for the API layer to expose.
                gin_verdict, gin_confidence, gin_chain = self._gin_verdict_for_graph()
                info.update({
                    "verdict": predicted_label,
                    "true_label": self.graph.true_label,
                    "confidence": confidence,
                    "correct": predicted_label == self.graph.true_label,
                    "total_reward": reward,
                    "gin_verdict": gin_verdict,
                    "gin_confidence": gin_confidence,
                    "gin_predicted_chain": gin_chain,
                })
                logger.info("[END] %s verdict=%s true=%s reward=%.3f",
                            self.episode_id, predicted_label, self.graph.true_label, reward)
                obs = self._build_obs()
                return obs, reward, terminated, truncated, info

            # ── Tool call actions ─────────────────────────────────────────────
            self.steps += 1
            is_dup = self.tool_call_counts.get(action_name, 0) > 0
            self.tool_call_counts[action_name] = self.tool_call_counts.get(action_name, 0) + 1
            self._tool_history[action] = min(self._tool_history[action] + 1, 5.0)

            # Execute tool
            tool_result = self.tool_registry.call(action_name, self.graph)
            info["tool_result"] = tool_result

            new_nodes = tool_result.get("new_nodes", 0)
            new_contradictions = tool_result.get("new_contradictions", 0)

            base_r = tool_call_reward(
                tool_name=action_name,
                new_nodes_discovered=new_nodes,
                new_contradictions=new_contradictions,
                is_duplicate_call=is_dup,
            )
            raw_r = shaped_step_reward(prev_graph, self.graph, base_r)
            self._prev_potential = compute_potential(self.graph)

            # Truncate if budget exceeded
            if self.steps >= self.max_steps:
                truncated = True
                self._done = True
                logger.info("[END] %s truncated at step %d", self.episode_id, self.steps)

            info["raw_reward"] = raw_r
            reward = float(np.clip(raw_r, config.REWARD_CLIP_MIN, config.REWARD_CLIP_MAX))

            logger.info("[STEP] %s step=%d action=%s reward=%.4f (raw=%.4f) nodes+=%d",
                        self.episode_id, self.steps, action_name, reward, raw_r, new_nodes)

            obs = self._build_obs()
            return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """Explicitly shut down tool registry to prevent resource leaks."""
        if hasattr(self, "tool_registry") and self.tool_registry:
            try:
                self.tool_registry.close()
                logger.debug("Environment and ToolRegistry closed successfully.")
            except Exception as e:
                logger.error("Error closing ToolRegistry: %s", e)

    def render(self) -> Any:

        with self._graph_lock:
            if self.graph is None:
                return None
            state = {
                "episode_id": self.episode_id,
                "step": self.steps,
                "max_steps": self.max_steps,
                "graph": self.graph.to_dict(),
                "manipulation_flagged": self.manipulation_flagged,
            }
        if self.render_mode == "human":
            import json
            print(json.dumps(state, indent=2, default=str))
        return state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_obs(self) -> np.ndarray:
        """
        v2.0 Multimodal Observation:
        Embeds up to MAX_OBSERVATION_NODES graph nodes sorted by discovery order.
        Nodes not yet in graph are zero-padded → fixed-size 2D matrix flattened to 1D.
        Shape: [MAX_OBSERVATION_NODES * CLAIM_EMBED_DIM | tool_history | scalars]
        = [3840 | 13 | 6] = 3859-dim
        """
        # Build per-node embedding matrix (sorted: root first, then discovered nodes)
        node_embeddings = []

        with self._graph_lock:
            if self.graph is not None:
                # Root node always comes first
                root_emb = self._embed(self.graph.root.text)
                node_embeddings.append(root_emb)

                # Add discovered (retrieved) non-root nodes, ordered by node_id for stability
                for node_id, node in sorted(self.graph.nodes.items()):
                    if node_id == self.graph.root_claim_id:
                        continue
                    if node.retrieved and len(node_embeddings) < config.MAX_OBSERVATION_NODES:
                        node_embeddings.append(self._embed(node.text))

        # Zero-pad to MAX_OBSERVATION_NODES
        pad_count = config.MAX_OBSERVATION_NODES - len(node_embeddings)
        for _ in range(pad_count):
            node_embeddings.append(np.zeros(config.CLAIM_EMBED_DIM, dtype=np.float32))

        node_matrix = np.concatenate(node_embeddings[:config.MAX_OBSERVATION_NODES])  # (3840,)

        budget_remaining = 1.0 - (self.steps / max(self.max_steps, 1))
        with self._graph_lock:
            scalars = np.array([
                self.graph.evidence_coverage if self.graph else 0.0,
                min(self.graph.source_diversity_entropy / 3.0, 1.0) if self.graph else 0.0,
                min(self.graph.contradiction_surface_area / 5.0, 1.0) if self.graph else 0.0,
                float(self.manipulation_flagged),
                budget_remaining,
                self.steps / config.MAX_EPISODE_STEPS,
            ], dtype=np.float32)

        obs = np.concatenate([node_matrix, self._tool_history, scalars])
        return obs.astype(np.float32)

    def _embed(self, text: str) -> np.ndarray:
        """Embed claim text using local sentence-transformers (free, offline)."""
        try:
            if self._embedder is None:
                # Reuse a pre-warmed instance if app.py loaded one at startup
                # (avoids a 30s blocking download on the first HF Spaces request)
                shared = getattr(MisInfoForensicsEnv, "_shared_embedder", None)
                if shared is not None:
                    self._embedder = shared
                else:
                    from sentence_transformers import SentenceTransformer
                    self._embedder = SentenceTransformer(config.HF_EMBEDDING_MODEL)
            emb = self._embedder.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        except Exception:
            return np.zeros(config.CLAIM_EMBED_DIM, dtype=np.float32)

    def _estimate_confidence(self) -> float:
        """Heuristic confidence based on evidence gathered."""
        with self._graph_lock:
            if self.graph is None:
                return 0.5
            cov = self.graph.evidence_coverage
            contra = min(self.graph.contradiction_surface_area / 3.0, 1.0)
            return min(0.5 + 0.3 * cov + 0.2 * contra, 0.99)

    def _gin_verdict_for_graph(self) -> Tuple[Optional[str], float, List[str]]:
        """Run the shared Blue GIN over the current claim graph and return
        (verdict, confidence, ordered_primitive_names).

        Returns ("unknown", 0.0, []) when the GIN, torch, or graph is unavailable.
        Errors are swallowed because verdict scoring must never crash a step.
        """
        try:
            from runtime import get_blue_gin

            with self._graph_lock:
                if self.graph is None:
                    return "unknown", 0.0, []

                # Build a minimal feature tensor from the graph nodes. We use
                # node trust_score + a few scalar coverage signals as a 10-dim
                # feature so the GIN sees real per-claim variation.
                nodes = list(self.graph.nodes.values()) if hasattr(self.graph, "nodes") else []

            if not nodes:
                return "unknown", 0.0, []

            feats = []
            for n in nodes[:10]:
                trust = float(getattr(n, "trust_score", 0.5) or 0.5)
                injected = 1.0 if n.metadata.get("injected", False) else 0.0
                retrieved = 1.0 if n.retrieved else 0.0
                feats.append([trust, injected, retrieved, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            x = torch.tensor(feats, dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)

            g = types.SimpleNamespace(
                x=x,
                edge_index=edge_index,
                batch=torch.zeros(x.size(0), dtype=torch.long),
            )

            gin = get_blue_gin()
            result = gin.predict_chain(g)
            chain_names = [p.name if hasattr(p, "name") else str(p) for p in result.get("ordered_chain", [])]
            return (
                result.get("verdict", "unknown"),
                float(result.get("confidence", 0.0)),
                chain_names,
            )
        except Exception:
            return "unknown", 0.0, []

    def get_episode_summary(self) -> dict:
        with self._graph_lock:
            if self.graph is None:
                return {}
            return {
                "episode_id": self.episode_id,
                "task_id": self.current_task.task_id if self.current_task else None,
                "difficulty": self.difficulty,
                "steps_used": self.steps,
                "max_steps": self.max_steps,
                "evidence_coverage": self.graph.evidence_coverage,
                "source_diversity": self.graph.source_diversity_entropy,
                "contradictions_found": self.graph.contradiction_surface_area,
                "manipulation_flagged": self.manipulation_flagged,
            }

    @contextmanager
    def graph_lock(self):
        with self._graph_lock:
            yield

    def has_graph(self) -> bool:
        with self._graph_lock:
            return self.graph is not None

    def get_graph_root_info(self) -> Tuple[str, str, float]:
        with self._graph_lock:
            if self.graph is None:
                return "", "unknown", 0.5
            root = self.graph.root
            return (
                getattr(root, "text", ""),
                getattr(root, "domain", "unknown"),
                getattr(root, "virality_score", 0.5),
            )

    def get_graph_metrics(self) -> Tuple[float, float, int]:
        with self._graph_lock:
            if self.graph is None:
                return 0.0, 0.0, 0
            return (
                float(getattr(self.graph, "evidence_coverage", 0.0)),
                float(getattr(self.graph, "source_diversity_entropy", 0.0)),
                int(getattr(self.graph, "contradiction_surface_area", 0)),
            )

    def get_graph_stats(self) -> Optional[Tuple[int, int, float, int]]:
        with self._graph_lock:
            if self.graph is None:
                return None
            retrieved = sum(1 for n in self.graph.nodes.values() if n.retrieved)
            total = len(self.graph.nodes)
            cov = float(getattr(self.graph, "evidence_coverage", 0.0))
            con = int(getattr(self.graph, "contradiction_surface_area", 0))
            return retrieved, total, cov, con

    def get_graph_true_label(self) -> str:
        with self._graph_lock:
            if self.graph is None:
                return "unknown"
            return getattr(self.graph, "true_label", "unknown")

    @staticmethod
    def parse_observation(obs: np.ndarray) -> dict:
        """
        Parse a flat observation vector back into a named dictionary.

        This is the ONLY sanctioned way to read observation fields.
        Consumers MUST NOT manually compute index offsets.

        Returns dict with keys:
          - tool_history: np.ndarray of per-action call counts
          - evidence_coverage: float 0-1
          - source_diversity: float 0-1 (normalised)
          - contradiction_norm: float 0-1 (normalised)
          - manipulation_flagged: bool
          - budget_remaining: float 0-1
          - step_ratio: float 0-1
        """
        embed_dim = config.MAX_OBSERVATION_NODES * config.CLAIM_EMBED_DIM
        expected = embed_dim + N_ACTIONS + 6
        if obs.shape[0] != expected:
            raise ValueError(f"Invalid observation length: {obs.shape[0]} (expected {expected})")
        tool_history = obs[embed_dim: embed_dim + N_ACTIONS]
        scalar_start = embed_dim + N_ACTIONS
        return {
            "tool_history": tool_history,
            "evidence_coverage": float(obs[scalar_start]),
            "source_diversity": float(obs[scalar_start + 1]),
            "contradiction_norm": float(obs[scalar_start + 2]),
            "manipulation_flagged": bool(obs[scalar_start + 3] > 0.5),
            "budget_remaining": float(obs[scalar_start + 4]),
            "step_ratio": float(obs[scalar_start + 5]),
        }
