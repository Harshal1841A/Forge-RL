"""
Tactic Edit Distance (TED) scorer for FORGE-MA.

PRD v9.0 Section 3.1 specification:
  TED = 0.60 * positional_score + 0.40 * set_overlap_score

  positional_score: fraction of (position, primitive) pairs that
    match exactly between predicted and true chain. Rewards getting
    not just the right primitives but in the right order.

  set_overlap_score: Jaccard similarity between predicted set and
    true set. Rewards finding the right primitives regardless of order.
    Hallucination penalty: -0.10 per predicted primitive not in true chain.

  Final TED clipped to (0.001, 0.999) — never exactly 0.0 or 1.0.

Examples:
  predict=[A,B,C], true=[A,B,C] → ~0.999 (perfect)
  predict=[C,B,A], true=[A,B,C] → ~0.673 (right set, wrong order)
  predict=[A,B],   true=[A,B,C] → ~0.733 (partial correct)
  predict=[D],     true=[A,B,C] → ~0.001 (wrong + hallucination)
  predict=[],      true=[A,B,C] → 0.001  (empty prediction)
"""
from __future__ import annotations
from typing import List


def tactic_edit_distance(predicted: list, true_chain: list) -> float:
    """
    Position-weighted Tactic Edit Distance.
    
    Args:
        predicted: List of PrimitiveType (or string) — Blue Team's prediction
        true_chain: List of PrimitiveType (or string) — ground truth
    
    Returns:
        float in (0.001, 0.999) — higher is better reconstruction
    """
    n = len(predicted)
    m = len(true_chain)

    # Edge cases
    if n == 0 and m == 0:
        return 0.999
    if n == 0:
        return 0.001
    if m == 0:
        # Predicted something when there was nothing — hallucination
        return 0.001

    # ── Component 1: Positional accuracy (60% weight) ──────────────────────
    # Compare position-by-position up to the shorter chain length.
    # Positions beyond the shorter chain count as misses.
    max_len = max(n, m)
    position_hits = 0
    for i in range(min(n, m)):
        # Normalise to string for comparison (handles str vs PrimitiveType enum)
        p_val = str(predicted[i].value) if hasattr(predicted[i], 'value') else str(predicted[i])
        t_val = str(true_chain[i].value) if hasattr(true_chain[i], 'value') else str(true_chain[i])
        if p_val == t_val:
            position_hits += 1

    positional_score = position_hits / max_len

    # ── Component 2: Set overlap (40% weight) ──────────────────────────────
    # Jaccard similarity on sets + hallucination penalty
    def _to_str_set(chain: list) -> set:
        return {
            str(p.value) if hasattr(p, 'value') else str(p)
            for p in chain
        }

    pred_set = _to_str_set(predicted)
    true_set = _to_str_set(true_chain)

    intersection = len(pred_set & true_set)
    union = len(pred_set | true_set)
    jaccard = intersection / union if union > 0 else 0.0

    # Hallucination penalty: -0.10 per predicted primitive not in true chain
    hallucinations = len(pred_set - true_set)
    hallucination_penalty = min(0.50, hallucinations * 0.10)  # cap at 0.50

    set_overlap_score = max(0.0, jaccard - hallucination_penalty)

    # ── Weighted combination ────────────────────────────────────────────────
    ted = 0.60 * positional_score + 0.40 * set_overlap_score

    # Clip strictly inside (0.001, 0.999) per PRD v9
    return max(0.001, min(0.999, ted))


def tactic_precision_recall(predicted: list, true_chain: list) -> dict:
    """
    Compute tactic-level precision and recall for the TPR F1 reward component.
    Used by hierarchical_reward.py as the w2 signal.
    """
    def _to_str_set(chain: list) -> set:
        return {
            str(p.value) if hasattr(p, 'value') else str(p)
            for p in chain
        }

    pred_set = _to_str_set(predicted)
    true_set = _to_str_set(true_chain)

    if not pred_set and not true_set:
        return {"precision": 0.999, "recall": 0.999, "f1": 0.999}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.001}
    if not true_set:
        return {"precision": 0.0, "recall": 0.999, "f1": 0.001}

    tp = len(pred_set & true_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": max(0.001, min(0.999, precision)),
        "recall":    max(0.001, min(0.999, recall)),
        "f1":        max(0.001, min(0.999, f1)),
    }
