"""
Evaluation harness — runs any agent against the environment and computes
structured metrics: accuracy, macro-F1, efficiency, manipulation detection.
"""

from __future__ import annotations
import logging
from typing import Any, Dict

import numpy as np

from env.misinfo_env import MisInfoForensicsEnv

logger = logging.getLogger(__name__)

LABEL_LIST = ["real", "misinfo", "satire", "out_of_context", "fabricated"]


def evaluate_agent(
    agent,
    n_episodes: int = 200,
    difficulty: int = None,
    seed_start: int = 9999,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate any agent (must implement .act(obs) → int) on a fresh env.
    Returns a metrics dict suitable for leaderboard reporting.
    """
    env = MisInfoForensicsEnv()
    true_labels, pred_labels = [], []
    step_counts, efficiencies = [], []
    manip_tp, manip_fp, manip_fn = 0, 0, 0
    total_reward = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        if difficulty is not None:
            env.difficulty = difficulty

        ep_reward = 0.0
        done = False
        predicted = None

        if hasattr(agent, "reset"):
            agent.reset()

        while not done:
            action = agent.act(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if step_info.get("verdict"):
                predicted = step_info["verdict"]

        total_reward += ep_reward
        true_label    = env.graph.true_label if env.graph else "unknown"
        predicted     = predicted or "misinfo"   # default if timeout

        true_labels.append(true_label)
        pred_labels.append(predicted)
        step_counts.append(env.steps)

        oracle = env.current_task.oracle_steps(env.graph) if env.current_task else 5
        efficiencies.append(oracle / max(env.steps, 1))

        # Manipulation detection metrics
        true_manip = env.current_task.has_manipulation(env.graph) if env.current_task else True
        pred_manip = env.manipulation_flagged
        if true_manip and pred_manip:
            manip_tp += 1
        elif not true_manip and pred_manip:
            manip_fp += 1
        elif true_manip and not pred_manip:
            manip_fn += 1

        if verbose:
            correct = "✅" if predicted == true_label else "❌"
            logger.info("Ep %03d: %s pred=%s true=%s steps=%d",
                        ep, correct, predicted, true_label, env.steps)

    # ── Metrics computation ───────────────────────────────────────────────────
    accuracy = sum(p == t for p, t in zip(pred_labels, true_labels)) / n_episodes

    # Macro-F1
    f1_scores = []
    for label in LABEL_LIST:
        tp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t == label)
        fp = sum(1 for p, t in zip(pred_labels, true_labels) if p == label and t != label)
        fn = sum(1 for p, t in zip(pred_labels, true_labels) if p != label and t == label)
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    # Manipulation F1
    mp = manip_tp / (manip_tp + manip_fp + 1e-9)
    mr = manip_tp / (manip_tp + manip_fn + 1e-9)
    manip_f1 = 2 * mp * mr / (mp + mr + 1e-9)

    return {
        "n_episodes":         n_episodes,
        "accuracy":           round(accuracy, 4),
        "macro_f1":           round(macro_f1, 4),
        "mean_reward":        round(total_reward / n_episodes, 4),
        "mean_steps":         round(float(np.mean(step_counts)), 2),
        "mean_efficiency":    round(float(np.mean(efficiencies)), 4),
        "manipulation_f1":    round(manip_f1, 4),
        "manipulation_tp":    manip_tp,
        "manipulation_fp":    manip_fp,
        "manipulation_fn":    manip_fn,
        "label_distribution": {l: pred_labels.count(l) for l in LABEL_LIST},
    }
