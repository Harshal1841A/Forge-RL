"""
Red Team Step Reward — HAE Per-Step Advantage Signal.
SPEC (PRD v8.1 Section 3.3):

  r_step_t = -ΔGIN_detection_prob

  Where ΔGIN_detection_prob = GIN_prob(t) - GIN_prob(t-1)
  for the detection probability of the Red primitive applied at step t.

  Intuition:
    - If after a Red action the GIN's detection probability INCREASES
      (the Blue team is now more confident), Red gets a NEGATIVE step reward.
    - If the action DECREASES detection probability (Blue is confused),
      Red gets a POSITIVE step reward.
    - This creates a dense per-step signal driving HAE optimisation,
      rather than relying solely on terminal episode reward.

  This is the "entire point" of hierarchical advantage estimation (HAE):
  the step-level advantage for Stage 2 PPO training.

Usage:
    stepper = RedStepReward(gin_predictor)
    stepper.reset()                         # call at episode start
    r_step = stepper.compute(graph_data)    # call after each Red action
"""
from __future__ import annotations
from typing import Optional
import numpy as np


class RedStepReward:
    """
    Per-step reward signal for the Red Team based on GIN detection probability.

    Parameters
    ----------
    gin : GINPredictor
        The Blue Team's GIN predictor, used to assess detectability.
    alpha : float
        Scaling factor for the step reward (default 1.0 matches PRD spec).
    """

    def __init__(self, gin, alpha: float = 1.0):
        self.gin = gin
        self.alpha = alpha
        self._prev_detection_prob: float = 0.5   # neutral prior at episode start

    def reset(self) -> None:
        """Call at the start of each episode to clear state."""
        self._prev_detection_prob = 0.5

    def compute(self, graph_data, primitive_idx: Optional[int] = None) -> float:
        """
        Compute r_step_t = -alpha * ΔGIN_detection_prob.

        Parameters
        ----------
        graph_data : PyG Data object (or duck-typed equivalent)
            Current claim graph state after Red action.
        primitive_idx : int, optional
            Index of the primitive just applied (0–7).
            If provided, targets that specific detection slot.
            Otherwise uses max detection probability across all primitives.

        Returns
        -------
        float : step reward (positive = Red made detection harder)
        """
        gin_result = self.gin.predict_chain(graph_data)
        presence_probs = gin_result["presence_probs"]  # shape (8,)

        if primitive_idx is not None and 0 <= primitive_idx < len(presence_probs):
            current_detection_prob = float(presence_probs[primitive_idx])
        else:
            # Use max detection probability as overall detectability measure
            current_detection_prob = float(np.max(presence_probs))

        delta = current_detection_prob - self._prev_detection_prob
        r_step = -self.alpha * delta

        self._prev_detection_prob = current_detection_prob
        return float(r_step)

    @property
    def prev_detection_prob(self) -> float:
        """Returns the GIN detection probability from the previous step."""
        return self._prev_detection_prob
