import torch
from typing import Optional

class RedStepReward:
    """Dense step reward for Red agent: reward proportional to GIN confusion delta."""
    def __init__(self, gin, alpha: float = 1.0):
        self.gin = gin       # GINPredictor instance (runtime/ singleton)
        self.alpha = alpha
        self._prev_max_prob: Optional[float] = None

    def reset(self):
        self._prev_max_prob = None

    def compute(self, graph_data, primitive_idx: Optional[int] = None) -> float:
        try:
            # GINPredictor manages eval/train internally; predict_chain is inference-safe.
            with torch.no_grad():
                result = self.gin.predict_chain(graph_data)
            # Use max probability as confusion proxy
            probs = result.get(
                "presence_probs",
                result.get("primitive_probs", [0.5]),
            )
            max_prob = float(max(probs)) if probs else 0.5
        except Exception:
            max_prob = 0.5
        if self._prev_max_prob is None:
            self._prev_max_prob = max_prob
            return 0.0
        delta = self._prev_max_prob - max_prob   # drop in GIN certainty = red win
        self._prev_max_prob = max_prob
        return float(self.alpha * delta)
