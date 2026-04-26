"""
Budget Penalty shaping for FORGE-MA.
SPEC (Master Prompt §Layer5):
  - step_cost: small negative reward per step taken (-0.01 default)
  - over_budget_penalty: large penalty if final step count exceeds budget limit (-0.50)
  - tool_efficiency_bonus: small positive if useful tools called vs total budget used < 0.6
  - All outputs clipped to [-1.0, 0.0] (purely negative shaping)
"""
from __future__ import annotations
from dataclasses import dataclass

# Hard-coded shaping constants (non-negotiable per spec)
STEP_COST: float = -0.01
OVER_BUDGET_PENALTY: float = -0.50
TOOL_EFFICIENCY_THRESHOLD: float = 0.60
TOOL_EFFICIENCY_BONUS: float = 0.05   # positive offset within [-1, 0] space


@dataclass(frozen=True)
class BudgetPenaltyResult:
    step_cost_total: float    # accumulated (-0.01 * steps_taken)
    over_budget_hit: bool
    over_budget_penalty: float
    efficiency_bonus: float
    total: float              # clipped sum


def compute_budget_penalty(
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,
) -> BudgetPenaltyResult:
    """
    Parameters
    ----------
    steps_taken       : number of investigation steps consumed.
    budget_limit      : maximum allowed steps for the episode.
    useful_tools_called: tools that returned non-empty evidence.

    Returns
    -------
    BudgetPenaltyResult with all components and clipped total.
    """
    step_cost_total = STEP_COST * steps_taken

    over_budget_hit = steps_taken > budget_limit
    over_budget_val = OVER_BUDGET_PENALTY if over_budget_hit else 0.0

    # Efficiency: reward agents that finish early with good tool coverage
    budget_used_ratio = steps_taken / max(budget_limit, 1)
    efficiency_bonus = (
        TOOL_EFFICIENCY_BONUS
        if (useful_tools_called >= 2 and budget_used_ratio < TOOL_EFFICIENCY_THRESHOLD)
        else 0.0
    )

    raw_total = step_cost_total + over_budget_val + efficiency_bonus
    clipped = max(-1.0, min(0.0, raw_total))

    return BudgetPenaltyResult(
        step_cost_total=step_cost_total,
        over_budget_hit=over_budget_hit,
        over_budget_penalty=over_budget_val,
        efficiency_bonus=efficiency_bonus,
        total=clipped,
    )
