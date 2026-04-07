"""
inference.py — OpenEnv Compatible Inference Script
Must use OpenAI client natively and output strictly formatted logs.
Runs under 20 mins, <8GB memory footprint.
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from typing import List, Optional
from pathlib import Path

from openai import OpenAI
import numpy as np
import config

# Root is the directory containing this file (project root)
sys.path.insert(0, str(Path(__file__).parent))

# Disable standard logging to prevent breaking stdout grader parsing
import logging
logging.getLogger("env").setLevel(logging.CRITICAL)
logging.getLogger("agents").setLevel(logging.CRITICAL)

# ─── MANDATORY ORGANIZER VARIABLES ─────────────────────────────────────────────
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "forge")

# ─── ORGANIZER STDOUT FORMATTING ──────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def run_evaluation(n_episodes_per_task: int = 2, difficulty: int = 1):
    from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
    from agents.llm_agent import LLMAgent
    from env.tasks import TASK_REGISTRY

    # ─── Instantiate OpenAI Client locally per rules ──────────────────────────
    # Participants must use OpenAI Client for all LLM calls using above variables
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    agent = LLMAgent()
    agent._openai_client = client  # Proxy client into the agent

    tasks = list(TASK_REGISTRY.keys())
    
    for task_idx, task_name in enumerate(tasks):
        env = MisInfoForensicsEnv(task_names=[task_name], difficulty=difficulty)

        for ep in range(n_episodes_per_task):
            ep_absolute = (task_idx * n_episodes_per_task) + ep + 1
            obs, info = env.reset(seed=1000 + ep_absolute)
            if hasattr(agent, "reset"):
                agent.reset()

            ep_reward = 0.0
            done = False
            verdict = None
            step_rewards = []
            
            log_start(task=info['task_id'], env=BENCHMARK, model=MODEL_NAME)

            step_info: dict = {}
            error_val: Optional[str] = None

            while not done:
                context = {
                    "steps": env.steps,
                    "max_steps": env.max_steps,
                    "coverage": env.graph.evidence_coverage if env.graph else 0.0,
                    "contradictions": env.graph.contradiction_surface_area if env.graph else 0,
                    "last_tool_result": step_info.get("tool_result"),
                    "claim_text": env.graph.root.text if env.graph else ""
                }
                
                try:
                    action = agent.act(obs, context=context)
                    action_name = ACTIONS[action]
                    obs, reward, terminated, truncated, step_info = env.step(action)
                except Exception as e:
                    error_val = str(e)
                    reward = 0.0
                    action_name = "error"
                    terminated = True
                    truncated = False
                    step_info = {}

                ep_reward += reward
                done = terminated or truncated
                step_rewards.append(reward)

                log_step(step=env.steps, action=action_name, reward=reward, done=done, error=error_val)

                if step_info.get("verdict"):
                    verdict = step_info["verdict"]

            true_label = env.graph.true_label if env.graph else "unknown"
            correct = (verdict == true_label)

            terminal_reward = step_rewards[-1] if step_rewards else 0.0
            score = float(np.clip(terminal_reward, 0.0, 1.0))

            try:
                env.close()
            except Exception as e:
                pass

            log_end(success=correct, steps=env.steps, score=score, rewards=step_rewards)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FORGE OpenEnv Eval Inference")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--difficulty", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    run_evaluation(args.episodes, args.difficulty)
