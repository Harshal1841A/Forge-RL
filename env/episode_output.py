"""
EpisodeOutput — structured result of one full FORGE-RL episode.
SPEC (Master Prompt §Layer6):
  - Immutable dataclass (frozen=True)
  - JSON-serializable via to_dict() / from_dict()
  - Fields: episode_id, verdict, predicted_chain, true_chain,
    reward_breakdown, society_result summary, agent_chains,
    steps_taken, budget_limit, timestamp
  - to_json() produces deterministic, sorted-key output
"""
from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from env.primitives import PrimitiveType
from rewards.hierarchical_reward import RewardBreakdown


def _prim_list(chain: List[PrimitiveType]) -> List[str]:
    return [p.value for p in chain]


def _prim_from_list(lst: List[str]) -> List[PrimitiveType]:
    return [PrimitiveType(v) for v in lst]


@dataclass(frozen=True)
class EpisodeOutput:
    """
    Single-episode result container.
    All mutable containers (lists, dicts) stored as tuples for immutability.
    """
    episode_id:       str
    verdict:          str                          # blue-team final verdict
    predicted_chain:  tuple[str, ...]             # canonical primitive values
    true_chain:       tuple[str, ...]
    reward_total:     float
    ted_component:    float
    f1_component:     float
    plausibility_delta: float
    consensus_bonus:  float
    expert_bonus:     float
    budget_total:     float
    consensus_level:  str
    expert_decision:  str
    steps_taken:      int
    budget_limit:     int
    useful_tools:     int
    over_budget:      bool
    agent_verdicts:   tuple[tuple[str, str], ...]  # agent_name → verdict
    timestamp:        str                           # ISO-8601
    red_step_rewards: tuple[float, ...] = tuple()   # Dense step rewards for adversarial training

    # ── Factory methods ────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        *,
        verdict: str,
        predicted_chain: List[PrimitiveType],
        true_chain: List[PrimitiveType],
        reward: RewardBreakdown,
        consensus_level: str,
        expert_decision: str,
        steps_taken: int,
        budget_limit: int,
        useful_tools: int,
        agent_verdicts: Dict[str, str],
        episode_id: Optional[str] = None,
        red_step_rewards: List[float] = None,
    ) -> "EpisodeOutput":
        return cls(
            episode_id=episode_id or str(uuid.uuid4())[:8],
            verdict=verdict,
            predicted_chain=tuple(_prim_list(predicted_chain)),
            true_chain=tuple(_prim_list(true_chain)),
            reward_total=round(reward.total, 6),
            ted_component=round(reward.ted_component, 6),
            f1_component=round(reward.f1_component, 6),
            plausibility_delta=round(reward.plausibility_delta, 6),
            consensus_bonus=round(reward.consensus_bonus, 6),
            expert_bonus=round(reward.expert_bonus, 6),
            budget_total=round(reward.budget.total, 6),
            consensus_level=consensus_level,
            expert_decision=expert_decision.upper(),
            steps_taken=steps_taken,
            budget_limit=budget_limit,
            useful_tools=useful_tools,
            over_budget=reward.budget.over_budget_hit,
            agent_verdicts=tuple(sorted(agent_verdicts.items())),
            timestamp=datetime.now(timezone.utc).isoformat(),
            red_step_rewards=tuple(red_step_rewards or []),
        )

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id":        self.episode_id,
            "verdict":           self.verdict,
            "predicted_chain":   list(self.predicted_chain),
            "true_chain":        list(self.true_chain),
            "reward": {
                "total":             self.reward_total,
                "ted_component":     self.ted_component,
                "f1_component":      self.f1_component,
                "plausibility_delta": self.plausibility_delta,
                "consensus_bonus":   self.consensus_bonus,
                "expert_bonus":      self.expert_bonus,
                "budget_total":      self.budget_total,
            },
            "consensus_level":   self.consensus_level,
            "expert_decision":   self.expert_decision,
            "steps_taken":       self.steps_taken,
            "budget_limit":      self.budget_limit,
            "useful_tools":      self.useful_tools,
            "over_budget":       self.over_budget,
            "agent_verdicts":    dict(self.agent_verdicts),
            "timestamp":         self.timestamp,
            "red_step_rewards":  list(self.red_step_rewards),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodeOutput":
        reward_d = data["reward"]
        return cls(
            episode_id=data["episode_id"],
            verdict=data["verdict"],
            predicted_chain=tuple(data["predicted_chain"]),
            true_chain=tuple(data["true_chain"]),
            reward_total=reward_d["total"],
            ted_component=reward_d["ted_component"],
            f1_component=reward_d["f1_component"],
            plausibility_delta=reward_d["plausibility_delta"],
            consensus_bonus=reward_d["consensus_bonus"],
            expert_bonus=reward_d["expert_bonus"],
            budget_total=reward_d["budget_total"],
            consensus_level=data["consensus_level"],
            expert_decision=data["expert_decision"],
            steps_taken=data["steps_taken"],
            budget_limit=data["budget_limit"],
            useful_tools=data["useful_tools"],
            over_budget=data["over_budget"],
            agent_verdicts=tuple(sorted(data["agent_verdicts"].items())),
            timestamp=data["timestamp"],
            red_step_rewards=tuple(data.get("red_step_rewards", [])),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "EpisodeOutput":
        return cls.from_dict(json.loads(json_str))

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_correct(self) -> bool:
        """True if predicted_chain exactly matches true_chain."""
        return self.predicted_chain == self.true_chain

    @property
    def chain_accuracy(self) -> float:
        """Element-wise overlap (Jaccard) between predicted and true chain sets."""
        pred = set(self.predicted_chain)
        true = set(self.true_chain)
        union = pred | true
        if not union:
            return 1.0
        return len(pred & true) / len(union)
