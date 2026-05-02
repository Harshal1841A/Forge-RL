from __future__ import annotations
from env.primitives import PrimitiveType
from typing import List, Dict


def compute_tactic_pr(
    predicted_chain: List[PrimitiveType],
    true_chain: List[PrimitiveType],
) -> Dict[str, float]:
    """
    Precision, recall, F1 for set-based tactic detection.

    FIX: both-empty case now returns 0.5 (neutral) instead of 1.0.
    Rationale: an agent that predicts [] on a true_chain=[] task made
    no mistakes but also demonstrated no investigative capability.
    Rewarding it with 1.0 inflates the W2 component on real-news tasks
    and masks whether the agent is actually learning the negative class.
    """
    pred_set = set(predicted_chain)
    true_set = set(true_chain)

    # FIX: was returning 1.0 — changed to 0.5 (neutral, not perfect)
    if not pred_set and not true_set:
        return {"precision": 0.5, "recall": 0.5, "f1": 0.5}

    if not pred_set:
        return {"precision": 0.001, "recall": 0.001, "f1": 0.001}

    if not true_set:
        # Predicted something when nothing was applied — hallucination
        return {"precision": 0.001, "recall": 0.999, "f1": 0.001}

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": float(max(0.001, min(0.999, precision))),
        "recall":    float(max(0.001, min(0.999, recall))),
        "f1":        float(max(0.001, min(0.999, f1))),
    }


def tactic_f1(
    predicted_chain: List[PrimitiveType],
    true_chain: List[PrimitiveType],
) -> float:
    """Convenience wrapper — returns the F1 scalar directly."""
    return compute_tactic_pr(predicted_chain, true_chain)["f1"]
