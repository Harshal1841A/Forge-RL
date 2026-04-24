"""
Hierarchical Reward Shaper for FORGE-MA v9.0.
SPEC (Master Prompt §Layer5 + PRD v8.1 Section 5):

  R_total = w1*TED_best + w2*TPR_F1 + w3*plausibility_delta
            + consensus_bonus + expert_bonus + budget_total
            + chain_entropy_bonus + chain_length_penalty

  Weights (fixed, non-negotiable):
    w1 = 0.40  (TED — primary chain reconstruction signal)
    w2 = 0.30  (TPR F1 — tactic coverage)
    w3 = 0.20  (plausibility delta — claim-level coherence shift)

  Consensus bonus (from SocietyOfThought):
    unanimous (4/4) : +0.10
    majority  (3/4) : +0.05
    split/all-diff  : -0.05

  Expert bonus (from ExpertReviewerAgent):
    APPROVE : +0.05
    REJECT  :  0.00

  Anti-hacking components (PRD v8.1 §5):
    chain_entropy_bonus  : rewards distributional diversity across predicted chains
                           prevents Blue from always predicting the majority chain
    chain_length_penalty : penalises over-long or trivially short predicted chains

  Budget: see rewards/budget_penalty.py
  Output clipped to [-1.0, 1.0].
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional

from env.primitives import PrimitiveType, K_MAX
from rewards.tactic_edit_dist import tactic_edit_distance
from rewards.tactic_pr import tactic_f1
from rewards.plausibility import plausibility_score, compute_plausibility
from rewards.budget_penalty import compute_budget_penalty, BudgetPenaltyResult

# ── fixed weights ──────────────────────────────────────────────────────────────
W_TED: float = 0.40
W_F1:  float = 0.30
W_PLB: float = 0.20

# ── consensus bonus table (keys match SocietyResult.consensus_level) ──────────
CONSENSUS_BONUS: dict[str, float] = {
    "unanimous":    +0.10,
    "majority_3":   +0.05,
    "split_2_2":    -0.05,
    "all_different": -0.05,
}

# ── expert bonus ───────────────────────────────────────────────────────────────
EXPERT_APPROVE_BONUS: float = +0.05
EXPERT_REJECT_BONUS:  float = +0.00

# ── anti-hacking constants (PRD v8.1 §5) ──────────────────────────────────────
# chain_entropy_bonus: rewards Blue for predicting diverse primitive sets
# rather than always predicting the modal/majority chain.
ENTROPY_BONUS_WEIGHT: float = 0.05   # scales entropy contribution
# chain_length_penalty: penalises trivially empty or over-saturated chains
LENGTH_PENALTY_WEIGHT: float = 0.03  # scales length penalty


@dataclass
class RewardBreakdown:
    ted_component:        float   # W_TED * TED_best
    f1_component:         float   # W_F1  * tactic_F1
    plausibility_delta:   float   # W_PLB * Δplausibility
    consensus_bonus:      float
    expert_bonus:         float
    budget:               BudgetPenaltyResult
    chain_entropy_bonus:  float   # anti-hacking: diversity bonus
    chain_length_penalty: float   # anti-hacking: length regularisation
    total:                float   # clipped [-1, 1]

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


def _chain_entropy_bonus(predicted_chains: List[List[PrimitiveType]]) -> float:
    """
    Anti-hacking component: rewards Blue for predicting diverse sets of
    primitives across the 4 agent chains, rather than unanimous majority voting.

    Shannon entropy H over the union distribution of predicted primitives.
    H=0 → all 4 agents predict exactly same chain → bonus=0
    H>0 → diverse predictions → positive bonus (scaled by ENTROPY_BONUS_WEIGHT)

    This prevents the degenerate reward-hack where Blue always predicts
    the statistically dominant chain regardless of evidence.
    """
    if not predicted_chains:
        return 0.0

    # Count frequency of each primitive across all agent chains
    prim_counts: dict[str, int] = {}
    total = 0
    for chain in predicted_chains:
        for p in chain:
            key = p.value if hasattr(p, "value") else str(p)
            prim_counts[key] = prim_counts.get(key, 0) + 1
            total += 1

    if total == 0:
        return 0.0

    # Shannon entropy
    entropy = 0.0
    for count in prim_counts.values():
        prob = count / total
        if prob > 0:
            entropy -= prob * math.log2(prob)

    # Max entropy with K_MAX primitives (log2(8) ≈ 3.0)
    max_entropy = math.log2(len(PrimitiveType))
    normalised = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(ENTROPY_BONUS_WEIGHT * normalised)


def _chain_length_penalty(predicted_chains: List[List[PrimitiveType]],
                           true_chain: List[PrimitiveType]) -> float:
    """
    Anti-hacking component: penalises trivially short (empty) or
    over-saturated (length == K_MAX always) predicted chains.

    Penalty = -LENGTH_PENALTY_WEIGHT * |avg_pred_len - true_len| / K_MAX

    This penalises Blue for always predicting length=0 (reward-hacking by
    abstaining) or always predicting the full K_MAX chain (over-fitting to
    the prior chain length distribution).
    """
    if not predicted_chains:
        return 0.0

    true_len = len(true_chain)
    avg_pred_len = sum(len(c) for c in predicted_chains) / len(predicted_chains)
    length_diff = abs(avg_pred_len - true_len) / max(K_MAX, 1)
    return float(-LENGTH_PENALTY_WEIGHT * min(1.0, length_diff))


def compute_reward(
    *,
    # Chain signals
    predicted_chains: List[List[PrimitiveType]],   # all 4 agent chains
    true_chain: List[PrimitiveType],

    # Plausibility signals
    claim_text_before: str,
    claim_text_after: str,

    # Society signals
    consensus_level: str,      # "unanimous" | "majority_3" | "split_2_2" | "all_different"

    # Expert signal
    expert_decision: str,       # "APPROVE" | "REJECT"

    # Budget signals
    steps_taken: int,
    budget_limit: int,
    useful_tools_called: int,

    # Optional graphs for richer plausibility scoring (CRITICAL BUG 1 FIX).
    # Pass separate before/after graph objects so plb_delta != 0.
    # Legacy single-graph callers may pass claim_graph= for backward compat;
    # that path falls back to text-only scoring.
    claim_graph=None,           # deprecated: use claim_graph_before + claim_graph_after
    claim_graph_before=None,    # ClaimGraph at episode start (from reset)
    claim_graph_after=None,     # ClaimGraph at episode end   (after Red actions)
) -> RewardBreakdown:
    """
    Compute the full hierarchical reward for one episode end.

    Parameters are keyword-only to prevent positional mis-ordering.
    """
    # ── TED_best: max over all agent predictions ──────────────────────────────
    teds = [tactic_edit_distance(pc, true_chain) for pc in predicted_chains]
    ted_best = max(teds) if teds else 0.001
    ted_component = W_TED * ted_best

    # ── TPR F1 (union of all predicted chains vs true_chain) ──────────────────
    union_predicted = list({p for chain in predicted_chains for p in chain})
    f1 = tactic_f1(union_predicted, true_chain)
    f1_component = W_F1 * f1

    # ── Plausibility delta ────────────────────────────────────────────────────
    # CRITICAL BUG 1 FIX: Use separate before/after graphs so delta is non-zero.
    # Priority: (1) explicit before+after graphs, (2) text-only fallback.
    # The legacy `claim_graph` kwarg provides no delta signal and is ignored
    # in favour of text-based scoring to avoid the permanently-zero bug.
    if claim_graph_before is not None and claim_graph_after is not None:
        try:
            plb_before = compute_plausibility(claim_graph_before)
            plb_after  = compute_plausibility(claim_graph_after)
        except Exception:
            plb_before = plausibility_score(claim_text_before)
            plb_after  = plausibility_score(claim_text_after)
    else:
        # Text-only path (also covers legacy claim_graph= callers)
        plb_before = plausibility_score(claim_text_before)
        plb_after  = plausibility_score(claim_text_after)

    plb_delta = plb_before - plb_after   # positive if claim became less plausible
    plb_component = W_PLB * max(-1.0, min(1.0, plb_delta))

    # ── Consensus bonus ───────────────────────────────────────────────────────
    con_bonus = CONSENSUS_BONUS.get(consensus_level, -0.05)

    # ── Expert bonus ──────────────────────────────────────────────────────────
    exp_bonus = (
        EXPERT_APPROVE_BONUS if expert_decision.upper() == "APPROVE"
        else EXPERT_REJECT_BONUS
    )

    # ── Budget penalty ────────────────────────────────────────────────────────
    budget_result = compute_budget_penalty(steps_taken, budget_limit, useful_tools_called)

    # ── Anti-hacking components (PRD v8.1 §5) ─────────────────────────────────
    entropy_bonus    = _chain_entropy_bonus(predicted_chains)
    length_penalty   = _chain_length_penalty(predicted_chains, true_chain)

    # ── Composite ────────────────────────────────────────────────────────────
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
    clipped = max(-1.0, min(1.0, raw_total))

    return RewardBreakdown(
        ted_component=ted_component,
        f1_component=f1_component,
        plausibility_delta=plb_component,
        consensus_bonus=con_bonus,
        expert_bonus=exp_bonus,
        budget=budget_result,
        chain_entropy_bonus=entropy_bonus,
        chain_length_penalty=length_penalty,
        total=clipped,
    )
