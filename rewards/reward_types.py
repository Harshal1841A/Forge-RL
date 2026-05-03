"""
Shared reward dataclasses — no cross-package imports.

Kept in a standalone module so that both:
  rewards.hierarchical_reward  (which imports env.primitives)
  env.episode_output           (which is part of the env package)
can both import RewardBreakdown/BudgetBreakdown without creating a
circular dependency through the env ↔ rewards import chain.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class BudgetBreakdown:
    step_penalty:    float
    over_budget_hit: bool
    total:           float


@dataclass
class RewardBreakdown:
    ted_component:      float
    f1_component:       float
    plausibility_delta: float
    consensus_bonus:    float
    expert_bonus:       float
    budget:             BudgetBreakdown
    total:              float
