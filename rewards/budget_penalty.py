"""
Budget Penalty shaping for FORGE-RL.
SPEC (Master Prompt §Layer5 — updated v9.1):

FIX: efficiency_bonus now contributes positively to total.
Previous clip was max(-1.0, min(0.0, ...)) which zeroed out the
+0.05 efficiency bonus. Now clip is max(-1.0, min(0.10, ...))
to allow the efficiency bonus to produce positive signal.

Components:
  step_cost:          -0.01 per step
  over_budget_penalty:-0.50 if steps > budget
  efficiency_bonus:   +0.05 if useful_tools >= 2 AND ratio < 0.60
  total:              clipped to (-1.0, +0.10)
"""
from __future__ import annotations
from dataclasses import dataclass

STEP_COST: float                 = -0.01
OVER_BUDGET_PENALTY: float       = -0.50
TOOL_EFFICIENCY_THRESHOLD: float =  0.60
TOOL_EFFICIENCY_BONUS: float     =  0.05


@dataclass(frozen=True)
class BudgetPenaltyResult:
    step_cost_total:      float
    over_budget_hit:      bool
    over_budget_penalty:  float
    efficiency_bonus:     float
    total:                float   # clipped (-1.0, +0.10)


def compute_budget_penalty(
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,
) -> BudgetPenaltyResult:
    step_cost_total = STEP_COST * steps_taken

    over_budget_hit = steps_taken > budget_limit
    over_budget_val = OVER_BUDGET_PENALTY if over_budget_hit else 0.0

    budget_used_ratio = steps_taken / max(budget_limit, 1)
    efficiency_bonus = (
        TOOL_EFFICIENCY_BONUS
        if (useful_tools_called >= 2
            and budget_used_ratio < TOOL_EFFICIENCY_THRESHOLD)
        else 0.0
    )

    raw_total = step_cost_total + over_budget_val + efficiency_bonus

    # FIX: upper bound raised from 0.0 to +0.10 so efficiency bonus
    # can actually contribute a positive signal to the episode reward.
    clipped = max(-1.0, min(0.10, raw_total))

    return BudgetPenaltyResult(
        step_cost_total=round(step_cost_total, 6),
        over_budget_hit=over_budget_hit,
        over_budget_penalty=over_budget_val,
        efficiency_bonus=efficiency_bonus,
        total=round(clipped, 6),
    )
