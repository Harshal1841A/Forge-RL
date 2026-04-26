"""
Behavioral Replay Buffer for FORGE-MA v9.0.
SPEC (PRD v8.1 Section 4):

  Replaces the SkillBank concept.
  Stores complete episode trajectories for self-improvement training.

  Key properties:
  - Adaptive threshold: only stores episodes with reward > threshold
  - Threshold dynamically adjusts to keep buffer diversity high
  - Provides mini-batch sampling for off-policy fine-tuning
  - Max capacity configurable (default 1000 episodes)
  - Tracks generation number for curriculum learning

Demo story: "SkillBank / Replay Buffer grows from 0 to 50+ entries across
generations" — this file makes that story real.

Usage:
    buf = ReplayBuffer(capacity=1000, min_reward_threshold=0.35)
    buf.add(episode_output)          # auto-rejects below threshold
    batch = buf.sample(n=16)         # random mini-batch
    print(buf.size, buf.threshold)   # monitor growth
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional

from env.episode_output import EpisodeOutput


@dataclass
class ReplayEntry:
    """One stored episode in the replay buffer."""
    episode: EpisodeOutput
    generation: int         # PPO generation at which this was produced
    replay_count: int = 0   # how many times this entry has been sampled


class ReplayBuffer:
    """
    Adaptive replay buffer storing completed FORGE-MA episodes.

    Adaptive threshold logic (PRD v8.1 §4):
      threshold_t+1 = threshold_t + eta * (acceptance_rate - target_rate)

      Where:
        eta             = 0.01  (learning rate for threshold)
        target_rate     = 0.6   (target fraction of episodes to accept)
        acceptance_rate = rolling_accepted / rolling_total (last 100 episodes)

    This keeps the buffer filled with continuously improving episodes without
    manual threshold tuning, and prevents the buffer from only keeping
    top-1% episodes (which would starve diversity).
    """

    def __init__(
        self,
        capacity: int = 1000,
        min_reward_threshold: float = 0.35,   # PRD v9 §5.2: Gen 0 threshold = 0.35
        target_acceptance_rate: float = 0.60,
        threshold_lr: float = 0.01,
    ):
        self.capacity = capacity
        self._threshold = min_reward_threshold
        self.target_acceptance_rate = target_acceptance_rate
        self.threshold_lr = threshold_lr

        self._entries: List[ReplayEntry] = []
        self._generation: int = 0

        # Adaptive threshold tracking (rolling window)
        self._rolling_total: int = 0
        self._rolling_accepted: int = 0
        self._window: int = 100   # rolling window size

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Current number of stored episodes."""
        return len(self._entries)

    @property
    def threshold(self) -> float:
        """Current adaptive reward threshold."""
        return self._threshold

    @property
    def generation(self) -> int:
        return self._generation

    def set_generation(self, gen: int) -> None:
        """
        Update threshold per PRD v9 Section 5.2 curriculum schedule.
        Gen 0: 0.35 (permissive — early training, few good episodes)
        Gen 1: 0.50 (moderate)
        Gen 2+: 0.60 (strict — only high-quality episodes)
        """
        schedule = {0: 0.35, 1: 0.50}
        self._threshold = schedule.get(gen, 0.60)
        self._generation = gen

    def add(self, episode: EpisodeOutput) -> bool:
        """
        Attempt to add an episode to the buffer.

        Returns True if accepted (reward >= threshold), False if rejected.
        Automatically applies adaptive threshold update.
        """
        self._rolling_total += 1
        accepted = episode.reward_total >= self._threshold

        if accepted:
            self._rolling_accepted += 1
            entry = ReplayEntry(episode=episode, generation=self._generation)
            self._entries.append(entry)

            # Evict oldest if over capacity
            if len(self._entries) > self.capacity:
                self._entries.pop(0)

        # Adaptive threshold update every `_window` episodes
        if self._rolling_total % self._window == 0:
            self._update_threshold()

        return accepted

    def sample(self, n: int = 16) -> List[EpisodeOutput]:
        """
        Sample n episodes without replacement when the buffer is large enough.
        Falls back to sampling with replacement only when n > len(buffer),
        in which case duplicates are expected and intentional.
        Returns list of EpisodeOutput objects.
        Increments replay_count on sampled entries.

        LOW BUG 2 FIX: previously used random.choices (always with replacement)
        even after capping n = min(n, len(entries)), causing a buffer of 5 entries
        sampled with n=5 to potentially return the same episode 3 times.
        """
        if not self._entries:
            return []

        if n <= len(self._entries):
            # Without-replacement sampling — no duplicates in a single batch
            sampled = random.sample(self._entries, k=n)
        else:
            # Buffer smaller than requested batch; duplicates are unavoidable
            sampled = random.choices(self._entries, k=n)

        for entry in sampled:
            entry.replay_count += 1
        return [e.episode for e in sampled]

    def best_n(self, n: int = 5) -> List[EpisodeOutput]:
        """Return top-n episodes by reward_total (for curriculum warm-up)."""
        sorted_entries = sorted(self._entries,
                                key=lambda e: e.episode.reward_total,
                                reverse=True)
        return [e.episode for e in sorted_entries[:n]]

    def stats(self) -> dict:
        """Summary statistics for logging / UI display."""
        if not self._entries:
            return {
                "size": 0, "threshold": self._threshold,
                "mean_reward": 0.0, "max_reward": 0.0,
                "acceptance_rate": 0.0, "generation": self._generation
            }

        rewards = [e.episode.reward_total for e in self._entries]
        acc_rate = (self._rolling_accepted / self._rolling_total
                    if self._rolling_total > 0 else 0.0)
        return {
            "size":            len(self._entries),
            "capacity":        self.capacity,
            "threshold":       round(self._threshold, 4),
            "mean_reward":     round(sum(rewards) / len(rewards), 4),
            "max_reward":      round(max(rewards), 4),
            "min_reward":      round(min(rewards), 4),
            "acceptance_rate": round(acc_rate, 4),
            "generation":      self._generation,
        }

    def clear(self) -> None:
        """Empty the buffer (use with caution)."""
        self._entries.clear()
        self._rolling_total = 0
        self._rolling_accepted = 0

    # ── Private ────────────────────────────────────────────────────────────────

    def _update_threshold(self) -> None:
        """
        Adaptive threshold update:
          threshold += lr * (acceptance_rate - target_rate)

        If we're accepting TOO MANY episodes (acceptance_rate > target_rate),
        threshold rises, becoming more selective.
        If we're accepting TOO FEW, threshold falls, becoming more permissive.
        """
        if self._rolling_total == 0:
            return
        acceptance_rate = self._rolling_accepted / self._rolling_total
        delta = self.threshold_lr * (acceptance_rate - self.target_acceptance_rate)
        self._threshold = max(-1.0, min(1.0, self._threshold + delta))
        # Reset rolling counters after update
        self._rolling_total = 0
        self._rolling_accepted = 0

    def __repr__(self) -> str:
        return (f"ReplayBuffer(size={self.size}/{self.capacity}, "
                f"threshold={self._threshold:.3f}, gen={self._generation})")
