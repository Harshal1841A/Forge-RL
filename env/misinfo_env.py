"""
FORGE — Main RL Environment (Gymnasium-compatible)
OpenEnv / Gymnasium interface for the MisInfo Forensics task.
"""

from __future__ import annotations
import copy
import logging
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

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
    "submit_verdict_out_of_context",# 11
    "submit_verdict_fabricated",    # 12
]
N_ACTIONS = len(ACTIONS)

VERDICT_ACTIONS = {
    8:  "real",
    9:  "misinfo",
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
        self._prev_potential: float = 0.0
        self._tool_history: np.ndarray = np.zeros(N_ACTIONS, dtype=np.float32)
        self._done: bool = True

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
        self.graph = self.current_task.generate(
            difficulty=self.difficulty, seed=ep_seed
        )

        # Dynamic step budget
        self.max_steps = min(
            config.BASE_EPISODE_STEPS
            + self.graph.num_tactics * config.STEP_COMPLEXITY_BONUS,
            config.MAX_EPISODE_STEPS,
        )

        self.steps = 0
        self.manipulation_flagged = False
        self.tool_call_counts = {}
        self.episode_id = str(uuid.uuid4())
        self._tool_history = np.zeros(N_ACTIONS, dtype=np.float32)
        self._done = False

        # Initialise prev potential for reward shaping
        self._prev_potential = compute_potential(self.graph)

        obs = self._build_obs()
        info = {
            "episode_id": self.episode_id,
            "task_id": self.current_task.task_id,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
            "graph_id": self.graph.graph_id,
        }
        logger.info("[START] episode=%s task=%s difficulty=%d",
                    self.episode_id, self.current_task.task_id, self.difficulty)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert not self._done, "Call reset() before step()"
        action_name = ACTIONS[action]
        prev_graph = copy.deepcopy(self.graph)

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "step": self.steps,
            "action": action_name,
        }

        # ── Free actions (no step cost) ───────────────────────────────────────
        if action_name == "flag_manipulation":
            self.manipulation_flagged = True
            reward = 0.0   # reward only at terminal
            info["flagged"] = True
            logger.info("[STEP] %s step=%d action=flag_manipulation",
                        self.episode_id, self.steps)
            obs = self._build_obs()
            return obs, reward, False, False, info

        # ── Verdict actions ───────────────────────────────────────────────────
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
            # Hackathon requirement: reward must be in 0.0-1.0 range
            reward = float(np.clip(terminal_r, 0.0, 1.0))
            terminated = True
            self._done = True
            info.update({
                "verdict": predicted_label,
                "true_label": self.graph.true_label,
                "confidence": confidence,
                "correct": predicted_label == self.graph.true_label,
                "total_reward": reward,
            })
            logger.info("[END] %s verdict=%s true=%s reward=%.3f",
                        self.episode_id, predicted_label, self.graph.true_label, reward)
            obs = self._build_obs()
            return obs, reward, terminated, truncated, info

        # ── Tool call actions ─────────────────────────────────────────────────
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
        reward = shaped_step_reward(prev_graph, self.graph, base_r)
        self._prev_potential = compute_potential(self.graph)

        # Truncate if budget exceeded
        if self.steps >= self.max_steps:
            truncated = True
            self._done = True
            # Penalty for not submitting verdict in time
            reward += -0.3
            logger.info("[END] %s truncated at step %d", self.episode_id, self.steps)

        # Hackathon requirement: reward must be in 0.0-1.0 range
        reward = float(np.clip(reward, 0.0, 1.0))

        logger.info("[STEP] %s step=%d action=%s reward=%.4f nodes+=%d",
                    self.episode_id, self.steps, action_name, reward, new_nodes)

        obs = self._build_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[dict]:
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
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(config.HF_EMBEDDING_MODEL)
            emb = self._embedder.encode(text, normalize_embeddings=True)
            return emb.astype(np.float32)
        except Exception:
            return np.zeros(config.CLAIM_EMBED_DIM, dtype=np.float32)

    def _estimate_confidence(self) -> float:
        """Heuristic confidence based on evidence gathered."""
        if self.graph is None:
            return 0.5
        cov = self.graph.evidence_coverage
        contra = min(self.graph.contradiction_surface_area / 3.0, 1.0)
        return min(0.5 + 0.3 * cov + 0.2 * contra, 0.99)

    def get_episode_summary(self) -> dict:
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
