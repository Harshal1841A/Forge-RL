"""
HeuristicAgent — rule-based investigator (~0.60 expected reward)

Decision logic:
  1. Always start with query_source (0)
  2. If source trust_score < 0.4 → trace_origin (1)
  3. Then cross_reference (2)
  4. If temporal features suggest mismatch → temporal_audit (5)
  5. If network signals suspicious → network_cluster (6)
  6. flag_manipulation (7) if enough evidence
  7. submit_verdict based on contradiction count
"""

from __future__ import annotations
import numpy as np
from env.misinfo_env import ACTIONS, N_ACTIONS


class HeuristicAgent:
    name = "heuristic"

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._flagged = False
        self._tool_used = set()

    def act(self, obs: np.ndarray, info: dict = None, **kwargs) -> int:
        """
        obs layout (positional):
          [0:384]   claim embedding (ignored — heuristic uses indices only)
          [384:397] tool call history counts (one per action)
          [397]     evidence_coverage
          [398]     source_diversity (normalised)
          [399]     contradiction_surface_area (normalised)
          [400]     manipulation_flagged
          [401]     budget_remaining
          [402]     step_ratio
        """
        from env.misinfo_env import ACTIONS
        import config
        embed_dim = config.MAX_OBSERVATION_NODES * config.CLAIM_EMBED_DIM
        hist = obs[embed_dim: embed_dim + N_ACTIONS]      # tool history
        coverage     = float(obs[embed_dim + N_ACTIONS])
        diversity    = float(obs[embed_dim + N_ACTIONS + 1])
        contra_norm  = float(obs[embed_dim + N_ACTIONS + 2])
        flagged      = bool(obs[embed_dim + N_ACTIONS + 3] > 0.5)
        budget       = float(obs[embed_dim + N_ACTIONS + 4])  # remaining fraction

        def used(action_name: str) -> bool:
            idx = ACTIONS.index(action_name)
            return hist[idx] > 0

        # ── Phase 1: Source investigation ─────────────────────────────────────
        if not used("query_source"):
            return ACTIONS.index("query_source")

        if not used("trace_origin"):
            return ACTIONS.index("trace_origin")

        # ── Phase 2: Cross-reference ──────────────────────────────────────────
        if not used("cross_reference"):
            return ACTIONS.index("cross_reference")

        # ── Phase 3: Specialised tools ────────────────────────────────────────
        if not used("temporal_audit"):
            return ACTIONS.index("temporal_audit")

        if not used("entity_link"):
            return ACTIONS.index("entity_link")

        if not used("network_cluster"):
            return ACTIONS.index("network_cluster")

        # ── Phase 4: Flag if evidence strong ──────────────────────────────────
        if not flagged and contra_norm > 0.2:
            return ACTIONS.index("flag_manipulation")

        # ── Phase 5: Verdict decision ─────────────────────────────────────────
        if budget < 0.4 or coverage > 0.6:
            if contra_norm > 0.3:
                return ACTIONS.index("submit_verdict_misinfo")
            elif contra_norm > 0.15:
                return ACTIONS.index("submit_verdict_out_of_context")
            else:
                return ACTIONS.index("submit_verdict_real")

        # Default: request more context
        if not used("request_context"):
            return ACTIONS.index("request_context")

        return ACTIONS.index("submit_verdict_misinfo")
