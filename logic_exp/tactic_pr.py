from env.primitives import PrimitiveType
from typing import List, Dict

def compute_tactic_pr(predicted_chain: List[PrimitiveType], true_chain: List[PrimitiveType]) -> Dict[str, float]:
    """
    Computes precision, recall, and f1 score for the predicted sequence vs true sequence.
    Treated as a set for presence detection (order independent).
    """
    pred_set = set(predicted_chain)
    true_set = set(true_chain)
    
    if not pred_set and not true_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not true_set:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
        
    true_positives = len(pred_set.intersection(true_set))
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return {
        "precision": float(max(0.001, min(0.999, precision))),
        "recall": float(max(0.001, min(0.999, recall))),
        "f1": float(max(0.001, min(0.999, f1)))
    }


def tactic_f1(predicted_chain: List[PrimitiveType],
              true_chain: List[PrimitiveType]) -> float:
    """Convenience wrapper — returns the F1 scalar directly."""
    return compute_tactic_pr(predicted_chain, true_chain)["f1"]
