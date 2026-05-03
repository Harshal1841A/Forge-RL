import torch
from typing import Optional


class RedStepReward:
    def __init__(self, gin, alpha: float = 1.0):
        self.gin = gin
        self.alpha = alpha
        self._prev_max_prob: Optional[float] = None
        self._prev_node_count: int = 0
        self._actions_seen: set = set()

    def reset(self):
        self._prev_max_prob = None
        self._prev_node_count = 0
        self._actions_seen = set()

    def compute(self, graph_data, primitive_idx: Optional[int] = None) -> float:
        checkpoint_loaded = getattr(self.gin, "checkpoint_loaded", False)

        if checkpoint_loaded:
            try:
                with torch.no_grad():
                    result = self.gin.predict_chain(graph_data)
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
            delta = self._prev_max_prob - max_prob
            self._prev_max_prob = max_prob
            return float(self.alpha * delta)
        else:
            # Heuristic: graph growth + action diversity + base activity
            reward = 0.02
            current_nodes = (
                int(graph_data.x.size(0))
                if getattr(graph_data, "x", None) is not None
                else 0
            )
            node_delta = max(0, current_nodes - self._prev_node_count)
            reward += min(0.10, node_delta * 0.04)
            self._prev_node_count = current_nodes
            if primitive_idx is not None and primitive_idx not in self._actions_seen:
                self._actions_seen.add(primitive_idx)
                reward += 0.05
            return round(float(self.alpha * reward), 5)
