"""
FORGE-MA v9.0 — Evaluation Suite.
Layer 9: Benchmark metrics computation across N episodes.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import numpy as np

from env.forge_env import ForgeEnv, ForgeEnvConfig
from env.episode_output import EpisodeOutput
from rewards.tactic_edit_dist import tactic_edit_distance
from rewards.tactic_pr import compute_tactic_pr
from env.primitives import PrimitiveType


@dataclass
class EvalMetrics:
    """Aggregate evaluation metrics across N episodes."""
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_ted: float
    mean_f1: float
    mean_chain_accuracy: float
    over_budget_rate: float
    verdict_distribution: Dict[str, int]
    consensus_distribution: Dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def summary_table(self) -> str:
        lines = [
            "## 📊 FORGE-MA Evaluation Results",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Episodes | {self.n_episodes} |",
            f"| Mean Reward | `{self.mean_reward:+.4f} ± {self.std_reward:.4f}` |",
            f"| Mean TED | `{self.mean_ted:.4f}` |",
            f"| Mean F1 | `{self.mean_f1:.4f}` |",
            f"| Chain Accuracy | `{self.mean_chain_accuracy:.1%}` |",
            f"| Over-Budget Rate | `{self.over_budget_rate:.1%}` |",
            "",
            "### Verdict Distribution",
        ]
        for v, c in sorted(self.verdict_distribution.items()):
            pct = c / max(self.n_episodes, 1)
            lines.append(f"- **{v}**: {c} ({pct:.0%})")
        return "\n".join(lines)


def run_evaluation(
    n_episodes: int = 20,
    budget: int = 5,
    seed: int = 0,
) -> EvalMetrics:
    """
    Run N evaluation episodes with fixed seeds for reproducibility.
    Returns aggregate EvalMetrics.
    """
    cfg = ForgeEnvConfig(budget=budget, seed=seed)
    env = ForgeEnv(cfg)

    rewards: List[float] = []
    teds: List[float] = []
    f1s: List[float] = []
    accuracies: List[float] = []
    over_budget: List[bool] = []
    verdicts: Dict[str, int] = {}
    consensuses: Dict[str, int] = {}

    for ep_idx in range(n_episodes):
        obs, info = env.reset(seed=seed + ep_idx)
        for _ in range(budget):
            _, reward, terminated, truncated, step_info = env.step()
            if terminated or truncated:
                ep_out: Optional[EpisodeOutput] = step_info.get("episode_output")
                if ep_out:
                    rewards.append(ep_out.reward_total)

                    # TED: compare predicted to true
                    pred = [PrimitiveType(v) for v in ep_out.predicted_chain]
                    true = [PrimitiveType(v) for v in ep_out.true_chain]
                    ted = tactic_edit_distance(pred, true)
                    pr = compute_tactic_pr(pred, true)

                    teds.append(ted)
                    f1s.append(pr["f1"])
                    accuracies.append(ep_out.chain_accuracy)
                    over_budget.append(ep_out.over_budget)

                    v = ep_out.verdict
                    verdicts[v] = verdicts.get(v, 0) + 1
                    consensuses[ep_out.consensus_level] = \
                        consensuses.get(ep_out.consensus_level, 0) + 1
                break

    n = max(len(rewards), 1)
    return EvalMetrics(
        n_episodes=n,
        mean_reward=float(np.mean(rewards)) if rewards else 0.0,
        std_reward=float(np.std(rewards)) if rewards else 0.0,
        mean_ted=float(np.mean(teds)) if teds else 0.0,
        mean_f1=float(np.mean(f1s)) if f1s else 0.0,
        mean_chain_accuracy=float(np.mean(accuracies)) if accuracies else 0.0,
        over_budget_rate=float(np.mean(over_budget)) if over_budget else 0.0,
        verdict_distribution=verdicts,
        consensus_distribution=consensuses,
    )


if __name__ == "__main__":
    metrics = run_evaluation(n_episodes=10, budget=5)
    print(metrics.summary_table())
    print("\nJSON:")
    print(metrics.to_json())
