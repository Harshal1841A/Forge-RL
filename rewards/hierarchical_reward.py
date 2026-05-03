from __future__ import annotations
from typing import TYPE_CHECKING, List
from rewards.reward_types import BudgetBreakdown, RewardBreakdown
from rewards.tactic_edit_dist import tactic_edit_distance

if TYPE_CHECKING:
    from env.primitives import PrimitiveType

# ── R1/R2 expert constants (imported by ExpertReviewerAgent and forge_env) ──
EXPERT_APPROVE_BONUS: float =  0.15
EXPERT_REJECT_BONUS:  float = -0.10


# RewardBreakdown and BudgetBreakdown are imported from rewards.reward_types
# (kept here for backward-compat re-export so existing code still works)
__all__ = [
    "compute_reward", "RewardBreakdown", "BudgetBreakdown",
    "EXPERT_APPROVE_BONUS", "EXPERT_REJECT_BONUS",
    "_compute_plausibility_delta", "_chain_entropy_bonus",
]


def _compute_plausibility_delta(
    claim_text_before: str,
    claim_text_after: str,
    claim_graph_before,
    claim_graph_after,
    budget_limit: int = 10,
) -> float:
    """Exposed for validate.py tests — returns raw (un-gated) delta."""
    def _node_count(g):
        if g is None:
            return 1
        nodes = getattr(g, "nodes", [])
        return max(len(nodes) if isinstance(nodes, list) else len(list(nodes)), 1)

    n_before = _node_count(claim_graph_before)
    n_after  = _node_count(claim_graph_after)
    return float(n_after - n_before) / max(budget_limit, 1)


def _chain_entropy_bonus(chains: list, mean_ted: float) -> float:
    """
    Bonus for high-diversity chain predictions when TED is above threshold.
    Exposed for smoke-tests in run_validations.py.
    """
    if mean_ted < 0.25:
        return 0.0
    unique_primitives: set = set()
    for chain in chains:
        for p in chain:
            unique_primitives.add(p)
    diversity = len(unique_primitives) / max(1, sum(len(c) for c in chains))
    return round(min(0.05, diversity * 0.10), 4)


def compute_reward(
    *,
    predicted_chains: List[list],
    true_chain: List,
    claim_text_before: str,
    claim_text_after: str,
    consensus_level,
    expert_decision: str,
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,
    claim_graph_before=None,
    claim_graph_after=None,
) -> "RewardBreakdown":

    # ── TED component ───────────────────────────────────────────────────
    best_ted = 1.0
    for chain in predicted_chains:
        ted = tactic_edit_distance(chain, true_chain)
        best_ted = min(best_ted, ted)
    ted_component = 1.0 - best_ted   # 0 = wrong, 1 = perfect

    # ── F1 component ────────────────────────────────────────────────────
    # FIX R4: both-empty → 0.5 (neutral, matches tactic_pr.py), was 1.0
    best_f1 = 0.0
    true_set = set(p.value for p in true_chain)
    for chain in predicted_chains:
        pred_set = set(p.value for p in chain)
        if not true_set and not pred_set:
            f1 = 0.5   # correct but undemonstrated capability
        else:
            tp   = len(pred_set & true_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec  = tp / len(true_set) if true_set else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    f1_component = best_f1

    # ── Plausibility delta (FIX R3: gate on F1 to remove wrong-pred bonus)
    def _node_count(g):
        if g is None:
            return 1
        nodes = getattr(g, "nodes", [])
        return max(len(nodes) if isinstance(nodes, list) else len(list(nodes)), 1)

    n_before  = _node_count(claim_graph_before)
    n_after   = _node_count(claim_graph_after)
    raw_delta = float(n_after - n_before) / max(budget_limit, 1)
    # Only reward graph exploration when prediction shows nonzero F1.
    # Prevents ~+0.15 constant floor from masking completely wrong predictions.
    plausibility_delta = raw_delta * best_f1

    # ── Consensus bonus ─────────────────────────────────────────────────
    if isinstance(consensus_level, str):
        _map = {
            "unanimous":    0.10,
            "majority_3":   0.05,
            "split_2_2":   -0.05,
            "all_different":-0.05,
        }
        consensus_bonus = _map.get(consensus_level, 0.0)
    else:
        consensus_bonus = (
            0.10 if consensus_level >= 0.75 else
            0.05 if consensus_level >= 0.50 else
            -0.05
        )

    # ── Expert bonus ────────────────────────────────────────────────────
    expert_bonus = (
        EXPERT_APPROVE_BONUS
        if str(expert_decision).upper() == "APPROVE"
        else EXPERT_REJECT_BONUS
    )

    # ── Budget component (FIX R5: use compute_budget_penalty, not inline)
    from rewards.budget_penalty import compute_budget_penalty as _bp
    _bpr = _bp(steps_taken, budget_limit, useful_tools_called)
    budget = BudgetBreakdown(
        step_penalty=_bpr.step_cost_total,
        over_budget_hit=_bpr.over_budget_hit,
        total=_bpr.total,
    )

    # ── Composite total ──────────────────────────────────────────────────
    total = (
        0.40 * ted_component
        + 0.25 * f1_component
        + 0.15 * plausibility_delta
        + 0.10 * consensus_bonus
        + 0.05 * expert_bonus
        + 0.05 * budget.total
    )
    total = round(max(-1.0, min(1.0, total)), 6)

    return RewardBreakdown(
        ted_component=ted_component,
        f1_component=f1_component,
        plausibility_delta=plausibility_delta,
        consensus_bonus=consensus_bonus,
        expert_bonus=expert_bonus,
        budget=budget,
        total=total,
    )
