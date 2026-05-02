"""
Hierarchical Reward Shaper for FORGE-RL v9.1
SPEC (PRD v9.0 Section 5 + OpenEnv compliance):

  R_total = w1*TED_mean + w2*TPR_F1 + w3*plausibility_delta
            + consensus_bonus + expert_bonus + budget_total
            + chain_entropy_bonus + chain_length_penalty

  Weights:
    w1 = 0.40  (TED — primary chain reconstruction signal)
    w2 = 0.30  (TPR F1 — tactic coverage)
    w3 = 0.20  (plausibility delta — structural coherence shift)

  Consensus bonus:
    unanimous (4/4) : +0.10
    majority  (3/4) : +0.05
    split/all-diff  : -0.05

  Expert bonus (matches ExpertReviewerAgent.bonus_reward()):
    APPROVE : +0.15
    REJECT  : -0.10
    other   :  0.00

  Anti-hacking:
    chain_entropy_bonus  : diversity reward, activates at TED > 0.25 (not 0.45)
    chain_length_penalty : over/under prediction regularisation

  Output clipped to (0.001, 0.999) — OpenEnv spec compliance.
  NEVER returns exactly 0.0 or 1.0.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

from env.primitives import PrimitiveType, K_MAX
from rewards.tactic_edit_dist import tactic_edit_distance
from rewards.tactic_pr import tactic_f1
from rewards.plausibility import (
    plausibility_score,
    compute_plausibility,
    semantic_drift_delta,
)
from rewards.budget_penalty import compute_budget_penalty, BudgetPenaltyResult

# ── fixed weights ─────────────────────────────────────────────────
W_TED: float = 0.40
W_F1:  float = 0.30
W_PLB: float = 0.20

# ── consensus bonus ───────────────────────────────────────────────
CONSENSUS_BONUS: dict[str, float] = {
    "unanimous":     +0.10,
    "majority_3":    +0.05,
    "split_2_2":     -0.05,
    "all_different": -0.05,
}

# ── expert bonus — MUST match ExpertReviewerAgent.bonus_reward() ──
# FIX: was +0.05 / +0.00 — corrected to +0.15 / -0.10
EXPERT_APPROVE_BONUS: float = +0.15
EXPERT_REJECT_BONUS:  float = -0.10

# ── anti-hacking ──────────────────────────────────────────────────
# FIX: entropy threshold lowered from 0.45 → 0.25
# Previous 0.45 meant bonus never fired in Gen 0 (TED ~ 0.10-0.25)
# which is exactly when reward hacking is most likely.
ENTROPY_BONUS_WEIGHT:  float = 0.05
ENTROPY_TED_THRESHOLD: float = 0.25   # was 0.45 — too high
LENGTH_PENALTY_WEIGHT: float = 0.03


@dataclass
class RewardBreakdown:
    ted_component:        float
    f1_component:         float
    plausibility_delta:   float
    consensus_bonus:      float
    expert_bonus:         float
    budget:               BudgetPenaltyResult
    chain_entropy_bonus:  float
    chain_length_penalty: float
    total:                float   # clipped (0.001, 0.999)

    def __str__(self) -> str:
        return (
            f"R_total={self.total:+.4f}  "
            f"[TED={self.ted_component:+.3f} "
            f"F1={self.f1_component:+.3f} "
            f"PLB={self.plausibility_delta:+.3f} "
            f"CON={self.consensus_bonus:+.3f} "
            f"EXP={self.expert_bonus:+.3f} "
            f"BUD={self.budget.total:+.3f} "
            f"ENT={self.chain_entropy_bonus:+.3f} "
            f"LEN={self.chain_length_penalty:+.3f}]"
        )


def _chain_entropy_bonus(
    predicted_chains: List[List[PrimitiveType]],
    mean_ted: float,
) -> float:
    """
    Anti-hacking: reward diversity of predictions across agent ensemble.
    FIX: threshold lowered from 0.45 to 0.25 so this fires in Gen 0.
    """
    if not predicted_chains:
        return 0.0
    # Quality gate — now 0.25 so it activates early in training
    if mean_ted < ENTROPY_TED_THRESHOLD:
        return 0.0

    prim_counts: dict[str, int] = {}
    total = 0
    for chain in predicted_chains:
        for p in chain:
            key = p.value if hasattr(p, "value") else str(p)
            prim_counts[key] = prim_counts.get(key, 0) + 1
            total += 1

    if total == 0:
        return 0.0

    entropy = 0.0
    for count in prim_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)

    max_entropy = math.log2(len(PrimitiveType))
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(ENTROPY_BONUS_WEIGHT * normalised)


def _chain_length_penalty(
    predicted_chains: List[List[PrimitiveType]],
    true_chain: List[PrimitiveType],
) -> float:
    """
    Anti-hacking: penalise trivially short or over-saturated chains.
    """
    if not predicted_chains:
        return 0.0

    true_len = len(true_chain)
    avg_pred_len = sum(len(c) for c in predicted_chains) / len(predicted_chains)
    length_diff = abs(avg_pred_len - true_len) / max(K_MAX, 1)
    base_penalty = float(-LENGTH_PENALTY_WEIGHT * min(1.0, length_diff))

    if avg_pred_len < 1.0 and true_len > 0:
        base_penalty -= 0.03
    return base_penalty


def _compute_plausibility_delta(
    claim_text_before: str,
    claim_text_after: str,
    claim_graph_before=None,
    claim_graph_after=None,
) -> float:
    """
    FIX: Previously always returned 0.0 because claim texts are identical
    (forge_env.py correctly does NOT modify claim_text when Red applies
    graph mutations — the text change was removed as a debug artefact).

    Correct approach:
    1. If texts differ → use sentence-transformer semantic drift
    2. If graphs provided → use graph-based plausibility delta (PRIMARY path)
    3. Fallback → text-only regex scorer

    The graph-based path is now PRIMARY because it reflects actual
    structural changes Red Team made to the ClaimGraph.
    """
    # Path 1: texts actually differ (rare — only in pipeline mode)
    if claim_text_before != claim_text_after:
        drift = semantic_drift_delta(claim_text_before, claim_text_after)
        if drift is not None:
            return max(0.0, min(1.0, drift))

    # Path 2: graph-based delta (PRIMARY for forge_env.py episodes)
    # This is non-zero because Red Team mutates the ClaimGraph even
    # when claim_text stays the same.
    if claim_graph_before is not None and claim_graph_after is not None:
        try:
            plb_before = compute_plausibility(claim_graph_before)
            plb_after  = compute_plausibility(claim_graph_after)
            delta = plb_before - plb_after
            return float(max(-1.0, min(1.0, delta)))
        except Exception:
            pass

    # Path 3: text-only fallback (last resort)
    plb_before = plausibility_score(claim_text_before)
    plb_after  = plausibility_score(claim_text_after)
    return float(max(-1.0, min(1.0, plb_before - plb_after)))


def compute_reward(
    *,
    predicted_chains: List[List[PrimitiveType]],
    true_chain: List[PrimitiveType],
    claim_text_before: str,
    claim_text_after: str,
    consensus_level: str,
    expert_decision: str,
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,
    claim_graph=None,          # deprecated — use before/after
    claim_graph_before=None,
    claim_graph_after=None,
) -> RewardBreakdown:
    """
    Compute full hierarchical reward. Output always in (0.001, 0.999).
    """
    # ── TED: mean over all agent predictions ────────────────────────
    # Mean rewards consistent quality across the ensemble.
    # Max() over 4 chains is easily gamed by one lucky sample.
    if not predicted_chains:
        predicted_chains = [[]]
    teds = [tactic_edit_distance(pc, true_chain) for pc in predicted_chains]
    ted_mean = sum(teds) / len(teds) if teds else 0.001
    ted_component = W_TED * ted_mean

    # ── TPR F1 ───────────────────────────────────────────────────────
    f1s = [tactic_f1(chain, true_chain) for chain in predicted_chains]
    f1_mean = sum(f1s) / len(f1s) if f1s else 0.001
    f1_component = W_F1 * f1_mean

    # ── Plausibility delta ───────────────────────────────────────────
    # FIX: now uses graph-based path as primary (text path was always 0)
    plb_delta = _compute_plausibility_delta(
        claim_text_before, claim_text_after,
        claim_graph_before, claim_graph_after,
    )
    plb_component = W_PLB * plb_delta

    # ── Consensus bonus ──────────────────────────────────────────────
    con_bonus = CONSENSUS_BONUS.get(consensus_level, -0.05)

    # ── Expert bonus — FIX: now uses correct +0.15 / -0.10 values ───
    exp_bonus = (
        EXPERT_APPROVE_BONUS if expert_decision.upper() == "APPROVE"
        else EXPERT_REJECT_BONUS if expert_decision.upper() == "REJECT"
        else 0.0
    )

    # ── Budget penalty ───────────────────────────────────────────────
    budget_result = compute_budget_penalty(
        steps_taken, budget_limit, useful_tools_called
    )

    # ── Anti-hacking ────────────────────────────────────────────────
    entropy_bonus  = _chain_entropy_bonus(predicted_chains, ted_mean)
    length_penalty = _chain_length_penalty(predicted_chains, true_chain)

    # ── Composite ────────────────────────────────────────────────────
    raw_total = (
        ted_component
        + f1_component
        + plb_component
        + con_bonus
        + exp_bonus
        + budget_result.total
        + entropy_bonus
        + length_penalty
    )

    # FIX: clip to (0.001, 0.999) — OpenEnv spec compliance
    # Previous [-1.0, 1.0] clip violated the declared reward_range.
    clipped = max(0.001, min(0.999, raw_total))

    return RewardBreakdown(
        ted_component=round(ted_component, 6),
        f1_component=round(f1_component, 6),
        plausibility_delta=round(plb_component, 6),
        consensus_bonus=round(con_bonus, 6),
        expert_bonus=round(exp_bonus, 6),
        budget=budget_result,
        chain_entropy_bonus=round(entropy_bonus, 6),
        chain_length_penalty=round(length_penalty, 6),
        total=round(clipped, 6),
    )
