"""
BaseTask — abstract interface every task must implement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
from env.claim_graph import ClaimGraph


class BaseTask(ABC):
    task_id: str = "base"
    description: str = ""

    @abstractmethod
    def generate(self, difficulty: int = 1, seed: int = 0) -> ClaimGraph:
        """
        Generate a ClaimGraph at the given difficulty level.
        difficulty: 1 (single tactic, clear trail) → 4 (composited tactics, noisy)
        seed: reproducibility
        """
        ...

    @abstractmethod
    def oracle_steps(self, graph: ClaimGraph) -> int:
        """
        Return the minimum number of tool calls required to solve this task
        by a privileged oracle that knows the graph structure.
        """
        ...

    @abstractmethod
    def has_manipulation(self, graph: ClaimGraph) -> bool:
        """Whether this claim involves deliberate manipulation (vs innocent error)."""
        ...

    def grade(self, episode_trace: list[dict], graph: ClaimGraph) -> float:
        """
        Evaluate an agent's trace (list of tool calls and verdicts) and return a score.

        Grader contract:
        - Score requires both correct tool usage AND correct verdict
        - Score 0.001 if no verdict submitted
        - Score 0.001 if fewer than 2 investigation tools used
        - Score range (0.001, 0.999) — never exactly 0 or 1
        - Deterministic: same trace + graph = same score
        """
        return 0.001

    def metadata(self) -> Dict[str, Any]:
        return {"task_id": self.task_id, "description": self.description}
