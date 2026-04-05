"""
inference.py — OpenEnv Compatible Inference Script
Must use OpenAI client natively and output strictly formatted logs.
Runs under 20 mins, <8GB memory footprint.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Root is the directory containing this file (project root)
sys.path.insert(0, str(Path(__file__).parent))

import config

# Disable standard logging to prevent breaking stdout grader parsing
logging.getLogger("env").setLevel(logging.CRITICAL)
logging.getLogger("agents").setLevel(logging.CRITICAL)


def run_evaluation(n_episodes_per_task: int = 2, difficulty: int = 1):
    from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
    from agents.llm_agent import LLMAgent

    from env.tasks import TASK_REGISTRY

    # Required specification: must use LLMAgent configured to standard endpoints
    agent = LLMAgent()
    results = []
    
    tasks = list(TASK_REGISTRY.keys())

    for task_idx, task_name in enumerate(tasks):
        # We instantiate a specific task environment for reproducibility
        env = MisInfoForensicsEnv(task_names=[task_name], difficulty=difficulty)
        
        for ep in range(n_episodes_per_task):
            ep_absolute = (task_idx * n_episodes_per_task) + ep + 1
            obs, info = env.reset(seed=1000 + ep_absolute)
            if hasattr(agent, "reset"):
                agent.reset()

            ep_start = time.time()
            ep_reward = 0.0
            done = False
            verdict = None

            # [START] emit
            print(f"[START] {json.dumps({'episode': ep_absolute, 'task': info['task_id']})}", flush=True)

            step_info: dict = {}  # pre-initialise so first-iteration context build doesn't NameError
            while not done:
                context = {
                    "steps": env.steps,
                    "max_steps": env.max_steps,
                    "coverage": env.graph.evidence_coverage if env.graph else 0.0,
                    "contradictions": env.graph.contradiction_surface_area if env.graph else 0,
                    "last_tool_result": step_info.get("tool_result"),
                    "claim_text": env.graph.root.text if env.graph else ""
                }
                action = agent.act(obs, context=context)
                action_name = ACTIONS[action]
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward
                done = terminated or truncated

                # [STEP] emit
                print(f"[STEP] {json.dumps({'action': action_name, 'reward': round(float(reward), 4)})}", flush=True)

                if step_info.get("verdict"):
                    verdict = step_info["verdict"]

            elapsed = time.time() - ep_start
            true_label = env.graph.true_label if env.graph else "unknown"
            correct    = (verdict == true_label)
            
            # Emit actual unclamped reward allowing negative penalties to be fully scored
            final_reward = float(ep_reward)

            # [END] emit
            result = {
                "episode":    ep_absolute,
                "verdict":    verdict,
                "true_label": true_label,
                "correct":    correct,
                "reward":     round(final_reward, 4),
                "steps":      env.steps,
                "time_s":     round(elapsed, 2),
            }
            print(f"[END] {json.dumps(result)}", flush=True)
            results.append(result)

    accuracy = sum(r["correct"] for r in results) / len(results)
    mean_reward = sum(r["reward"] for r in results) / len(results)
    
    summary = {
        "total_episodes": len(results),
        "accuracy":    round(accuracy, 4),
        "mean_reward": round(mean_reward, 4),
    }
    # Final summary can be standard print
    print(f"\nFINAL_SUMMARY: {json.dumps(summary)}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORGE OpenEnv Eval Inference")
    parser.add_argument("--episodes",  type=int, default=5)
    parser.add_argument("--difficulty",type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    run_evaluation(args.episodes, args.difficulty)
