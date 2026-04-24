"""
Red Team Adversarial Agent.
SPEC (Master Prompt §Layer4):
  - Uses HAEModel to score candidate edit actions on the ClaimGraph
  - Proposes (primitive_sequence, disarm_ids, edit_text) per step
  - Respects K_MAX via ActionValidator
  - Two modes:
      greedy  : pick action with highest HAE score
      epsilon : epsilon-greedy exploration (for RL training)
  - Budget-aware: tracks remaining steps and avoids no-ops if budget low
"""
import random
from typing import List, Tuple, Optional
import torch

from env.primitives import PrimitiveType, K_MAX
from red_team.hae_model import HAEModel, NUM_TOOLS, NUM_PRIMITIVES
from red_team.action_validator import validate_action

# Map tool index → human label (must match NegotiatedSearch.TOOLS order)
TOOL_LABELS = [
    "trace_origin", "query_source", "temporal_audit", "quote_search",
    "cross_reference", "entity_link", "network_cluster", "context_expand",
    "image_reverse_search", "domain_whois", "sentiment_analysis",
    "semantic_similarity", "authorship_verification"
]

# All primitives in canonical enum order
ALL_PRIMITIVES = list(PrimitiveType)


class RedAction:
    """A single adversarial action proposed by the Red Agent."""
    __slots__ = ("primitive", "tool_label", "disarm_ids", "edit_text", "score")

    def __init__(self, primitive: PrimitiveType, tool_label: str,
                 disarm_ids: List[str], edit_text: str, score: float = 0.0):
        self.primitive = primitive
        self.tool_label = tool_label
        self.disarm_ids = disarm_ids
        self.edit_text = edit_text
        self.score = score

    def __repr__(self):
        return (f"RedAction(primitive={self.primitive.name}, "
                f"tool={self.tool_label}, score={self.score:.4f})")


class RedAgent:
    """
    Adversarial agent that perturbs ClaimGraph nodes/edges to
    maximise blue-team confusion while staying within K_MAX budget.

    Parameters
    ----------
    epsilon : float
        Exploration probability for epsilon-greedy mode (ignored in greedy).
    mode : str
        'greedy' | 'epsilon'
    """

    def __init__(self, epsilon: float = 0.15, mode: str = "greedy"):
        self.hae = HAEModel(node_feat_dim=10)
        self.epsilon = epsilon
        self.mode = mode
        self._current_chain: List[PrimitiveType] = []
        self.history: List[Tuple[torch.Tensor, torch.Tensor, int, int]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Call at the start of each episode."""
        self._current_chain = []
        self.history = []

    def propose_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        budget_remaining: int,
        batch: Optional[torch.Tensor] = None,
    ) -> Optional[RedAction]:
        """
        Propose the next adversarial action.
        Returns None if no valid action can be generated within constraints.

        Steps
        -----
        1. HAE forward pass → action_logits + primitive_logits
        2. Rank candidates by score
        3. Walk ranked list; accept first that passes validate_action
        4. Epsilon-greedy: with prob epsilon, pick a random valid action instead
        """
        if budget_remaining <= 0:
            return None

        # Already at K_MAX? Can't add another primitive.
        if len(self._current_chain) >= K_MAX:
            return None

        # --- Epsilon-greedy exploration ---
        self._last_x = x
        self._last_edge = edge_index
        if self.mode == "epsilon" and random.random() < self.epsilon:
            return self._random_valid_action()

        # --- HAE scoring ---
        self.hae.eval()
        with torch.no_grad():
            out = self.hae(x, edge_index, batch)
        a_probs = torch.softmax(out["action_logits"][0], dim=-1).cpu().tolist()
        p_probs = torch.softmax(out["primitive_logits"][0], dim=-1).cpu().tolist()

        # Build ranked candidate list: (score, tool_idx, prim_idx)
        candidates = []
        for ti in range(NUM_TOOLS):
            for pi in range(NUM_PRIMITIVES):
                score = (a_probs[ti] * p_probs[pi]) ** 0.5
                candidates.append((score, ti, pi))
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Accept first valid
        for score, ti, pi in candidates:
            prim = ALL_PRIMITIVES[pi]
            candidate_chain = self._current_chain + [prim]
            disarm_ids = self._disarm_for(prim)

            if validate_action(candidate_chain, disarm_ids):
                action = RedAction(
                    primitive=prim,
                    tool_label=TOOL_LABELS[ti],
                    disarm_ids=disarm_ids,
                    edit_text=self._edit_text_for(prim, TOOL_LABELS[ti]),
                    score=score,
                )
                self._current_chain = candidate_chain
                self.history.append((x.detach().cpu(), edge_index.detach().cpu(), ti, pi))
                return action

        return None  # No valid action found (shouldn't happen unless K_MAX=4 hit)

    def commit_action(self, action: RedAction):
        """
        Explicitly commit an action from an EXTERNAL caller that bypasses propose_action().
        DO NOT call this if you already called propose_action() for the same action.
        """
        if action.primitive not in self._current_chain and len(self._current_chain) < K_MAX:
            self._current_chain.append(action.primitive)

    @property
    def current_chain(self) -> List[PrimitiveType]:
        return list(self._current_chain)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_valid_action(self) -> Optional[RedAction]:
        """Purely random valid action (exploration fallback)."""
        primitives = list(PrimitiveType)
        random.shuffle(primitives)
        tools = list(range(NUM_TOOLS))
        random.shuffle(tools)

        for prim in primitives:
            candidate_chain = self._current_chain + [prim]
            disarm_ids = self._disarm_for(prim)
            ti = random.choice(tools)
            if validate_action(candidate_chain, disarm_ids):
                action = RedAction(
                    primitive=prim,
                    tool_label=TOOL_LABELS[ti],
                    disarm_ids=disarm_ids,
                    edit_text=self._edit_text_for(prim, TOOL_LABELS[ti]),
                    score=0.0,
                )
                self._current_chain = candidate_chain
                # If random action, we don't track x, edge_index because it's not on policy,
                # but to avoid breaking the trainer, we record it.
                if hasattr(self, '_last_x'):
                     pi = ALL_PRIMITIVES.index(prim)
                     self.history.append((self._last_x.detach().cpu(), self._last_edge.detach().cpu(), ti, pi))
                return action
        return None

    @staticmethod
    def _disarm_for(prim: PrimitiveType) -> List[str]:
        """Return representative T-prefix DISARM ids for a primitive."""
        _MAP = {
            PrimitiveType.SOURCE_LAUNDER:      ["T0013"],
            PrimitiveType.TEMPORAL_SHIFT:      ["T0046"],
            PrimitiveType.ENTITY_SUBSTITUTE:   ["T0075"],
            PrimitiveType.QUOTE_FABRICATE:     ["T0006"],
            PrimitiveType.CONTEXT_STRIP:       ["T0019"],
            PrimitiveType.CITATION_FORGE:      ["T0016"],
            PrimitiveType.NETWORK_AMPLIFY:     ["T0049"],
            PrimitiveType.SATIRE_REFRAME:      ["T0085"],
        }
        return _MAP.get(prim, ["T0001"])

    @staticmethod
    def _edit_text_for(prim: PrimitiveType, tool: str) -> str:
        """Generate a templated adversarial edit description."""
        return f"[RED] Apply {prim.name} via {tool}"
