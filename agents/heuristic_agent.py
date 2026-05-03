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

NOTE: Observation parsing uses ForgeEnv.parse_observation()
      to avoid hardcoded index slicing. If the observation layout changes,
      only parse_observation() needs updating — not this agent.
"""

from __future__ import annotations
import numpy as np
from env.forge_env import ForgeEnv
N_ACTIONS = 13


class HeuristicAgent:
    name = "heuristic"

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._flagged = False
        self._tool_used = set()

    def act(self, obs, info: dict = None, **kwargs) -> int:
        """
        Select the next action based on structured observation fields.

        Supports both:
          - R1 (ForgeEnv) flat numpy array observations
          - R2 (ForgeEnv) dict observations

        When obs is a dict (ForgeEnv), synthesises the parsed fields from
        available keys rather than delegating to the R1 parse_observation API.
        """
        from env.forge_env import ACTIONS

        # ── Parse observation — dual-format support ───────────────────────────
        if isinstance(obs, dict):
            # ForgeEnv (R2) dict observation path
            n_actions = N_ACTIONS
            budget_raw = obs.get("budget_remaining", 10.0)
            budget_max = budget_raw + obs.get("steps_taken", 0.0)
            budget = budget_raw / max(budget_max, 1.0)  # normalise to [0,1]
            # No detailed tool history available from ForgeEnv obs — use step count proxy
            steps = obs.get("steps_taken", 0)
            hist = [0] * n_actions
            coverage = min(1.0, steps / 6.0)
            contra_norm = min(1.0, len(obs.get("red_chain", [])) * 0.2)
            flagged = len(obs.get("red_chain", [])) >= 2
        else:
            # R1 flat numpy array path
            parsed = ForgeEnv.parse_observation(obs)
            hist = parsed["tool_history"]
            coverage = parsed["evidence_coverage"]
            contra_norm = parsed["contradiction_norm"]
            flagged = parsed["manipulation_flagged"]
            budget = parsed["budget_remaining"]

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

        # ── Phase 5: Multi-class verdict based on graph signals ──────────────
        if budget < 0.4 or coverage > 0.6:
            if flagged and contra_norm > 0.35:
                return ACTIONS.index("submit_verdict_misinfo")
            elif flagged:
                return ACTIONS.index("submit_verdict_fabricated")
            elif contra_norm > 0.4:
                return ACTIONS.index("submit_verdict_misinfo")
            elif contra_norm > 0.2:
                return ACTIONS.index("submit_verdict_out_of_context")
            elif contra_norm > 0.08:
                return ACTIONS.index("submit_verdict_satire")
            else:
                return ACTIONS.index("submit_verdict_real")

        if not used("request_context"):
            return ACTIONS.index("request_context")

        # Final fallback
        if contra_norm > 0.25:
            return ACTIONS.index("submit_verdict_misinfo")
        elif contra_norm > 0.08:
            return ACTIONS.index("submit_verdict_out_of_context")
        else:
            return ACTIONS.index("submit_verdict_real")
