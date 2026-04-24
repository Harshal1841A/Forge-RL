"""
RandomAgent — uniform random baseline (~0.18 expected reward)
"""

from __future__ import annotations
import random
import numpy as np
from env.misinfo_env import N_ACTIONS


class RandomAgent:
    """Uniform random action selection — sets the performance floor."""

    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def act(self, obs: np.ndarray, **kwargs) -> int:
        return self.rng.randint(0, N_ACTIONS - 1)

    def reset(self) -> None:
        pass
