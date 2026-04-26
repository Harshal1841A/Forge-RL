"""
Curriculum Manager — PAIRED-style automated difficulty progression.

Stages:
  0 → single tactic, clear evidence trail, large budget
  1 → 2 tactics, one unreliable source, standard budget
  2 → 3 tactics, bot amplification, reduced budget, noisy tools
  3 → 4 tactics, composited attacks, strict budget, full noise

Progression gate: agent must achieve CURRICULUM_GATE_REWARD mean reward
over the last 1000 episodes before advancing.
"""

from __future__ import annotations
import logging
from collections import deque
from typing import List

import config

logger = logging.getLogger(__name__)


class CurriculumManager:
    def __init__(self, stages: List[dict] = None):
        self.stages = stages or config.CURRICULUM_STAGES
        self.current_stage = 0
        self.reward_window: deque = deque(maxlen=1000)
        self.stage_history: List[dict] = []

    @property
    def stage_config(self) -> dict:
        return self.stages[self.current_stage]

    @property
    def difficulty(self) -> int:
        return self.current_stage + 1

    @property
    def noisy_tools(self) -> bool:
        return self.stage_config.get("noisy_tools", False)

    @property
    def budget_multiplier(self) -> float:
        return self.stage_config.get("budget_mult", 1.0)

    @property
    def at_final_stage(self) -> bool:
        return self.current_stage >= len(self.stages) - 1

    def record_episode_reward(self, reward: float) -> None:
        self.reward_window.append(reward)

    def check_progression(self) -> bool:
        """Returns True if stage was advanced."""
        if self.at_final_stage:
            return False
        if len(self.reward_window) < 200:
            return False   # need enough samples

        mean_reward = sum(self.reward_window) / len(self.reward_window)
        gate = config.CURRICULUM_GATE_REWARD

        if mean_reward >= gate:
            old_stage = self.current_stage
            self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)
            self.stage_history.append({
                "from_stage": old_stage,
                "to_stage": self.current_stage,
                "gate_reward": gate,
                "achieved_reward": round(mean_reward, 4),
            })
            self.reward_window.clear()
            logger.info(
                "✅ Curriculum advanced: Stage %d → %d (mean_reward=%.3f ≥ gate=%.3f)",
                old_stage, self.current_stage, mean_reward, gate,
            )
            return True
        return False

    def status(self) -> dict:
        mean_r = (
            sum(self.reward_window) / len(self.reward_window)
            if self.reward_window else 0.0
        )
        return {
            "current_stage": self.current_stage,
            "stage_name": self.stage_config.get("name", f"stage{self.current_stage}"),
            "difficulty": self.difficulty,
            "noisy_tools": self.noisy_tools,
            "budget_multiplier": self.budget_multiplier,
            "mean_reward_window": round(mean_r, 4),
            "gate_reward": config.CURRICULUM_GATE_REWARD,
            "samples_in_window": len(self.reward_window),
            "at_final_stage": self.at_final_stage,
        }
