from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import math

from env.primitives import PrimitiveType
from rewards.tactic_edit_dist import tactic_edit_distance
import config

@dataclass
class BudgetBreakdown:
    step_penalty: float
    over_budget_hit: bool
    total: float

@dataclass
class RewardBreakdown:
    ted_component: float
    f1_component: float
    plausibility_delta: float
    consensus_bonus: float
    expert_bonus: float
    budget: BudgetBreakdown
    total: float

def compute_reward(
    *,
    predicted_chains: List[List[PrimitiveType]],
    true_chain: List[PrimitiveType],
    claim_text_before: str,
    claim_text_after: str,
    consensus_level: float,
    expert_decision: str,
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,
    claim_graph_before=None,
    claim_graph_after=None,
) -> RewardBreakdown:
    # ── TED component (best chain across ensemble) ──────────────────────
    best_ted = 1.0
    for chain in predicted_chains:
        ted = tactic_edit_distance(chain, true_chain)
        best_ted = min(best_ted, ted)
    ted_component = 1.0 - best_ted   # 0=wrong, 1=perfect

    # ── F1 component (set overlap) ──────────────────────────────────────
    best_f1 = 0.0
    true_set = set(p.value for p in true_chain)
    for chain in predicted_chains:
        pred_set = set(p.value for p in chain)
        if not true_set and not pred_set:
            f1 = 1.0
        else:
            tp = len(pred_set & true_set)
            prec = tp / len(pred_set) if pred_set else 0.0
            rec  = tp / len(true_set) if true_set else 0.0
            f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        best_f1 = max(best_f1, f1)
    f1_component = best_f1

    # ── Plausibility delta (graph complexity growth) ────────────────────
    def _node_count(g):
        if g is None: return 1
        nodes = getattr(g, "nodes", [])
        return max(len(nodes) if isinstance(nodes, list) else len(list(nodes)), 1)
    n_before = _node_count(claim_graph_before)
    n_after  = _node_count(claim_graph_after)
    plausibility_delta = float(n_after - n_before) / max(budget_limit, 1)

    # ── Consensus bonus ─────────────────────────────────────────────────
    if isinstance(consensus_level, str):
        _map = {"unanimous": 0.10, "majority_3": 0.05,
                "split_2_2": -0.05, "all_different": -0.05}
        consensus_bonus = _map.get(consensus_level, 0.0)
    else:
        # float consensus (0–1)
        consensus_bonus = 0.10 if consensus_level >= 0.75 else (
                          0.05 if consensus_level >= 0.50 else -0.05)

    # ── Expert bonus ────────────────────────────────────────────────────
    expert_bonus = 0.10 if str(expert_decision).upper() == "APPROVE" else -0.05

    # ── Budget component ────────────────────────────────────────────────
    step_ratio = steps_taken / max(budget_limit, 1)
    step_penalty = -0.05 * step_ratio
    over_budget  = steps_taken > budget_limit
    if over_budget:
        step_penalty -= 0.10
    budget = BudgetBreakdown(
        step_penalty=step_penalty,
        over_budget_hit=over_budget,
        total=step_penalty,
    )

    # ── Composite total (weighted) ───────────────────────────────────────
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
