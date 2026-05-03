"""
FORGE-RL inference entry point.
Required by OpenEnv Phase 2 agentic evaluation.
Runs one complete episode and prints structured JSON log.
"""
from __future__ import annotations
import json, sys, logging, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)

def run_episode(task_name: str = None, agent_id: str = "openenv_eval",
                max_steps: int = 10) -> dict:
    from env.forge_env import ForgeEnv, ACTIONS
    try:
        env = ForgeEnv(
            task_names=[task_name] if task_name else None,
        )
        obs, info = env.reset(seed=42)
        episode_id = info.get("episode_id", "ep_001")
        total_reward = 0.0
        steps = 0
        done = False
        log = []

        # Heuristic agent: investigate then submit
        TOOL_SEQUENCE = [0, 1, 5, 2, 3]  # query_source, cross_ref, entity_link, network, temporal
        while not done and steps < max_steps:
            if steps < len(TOOL_SEQUENCE):
                action = TOOL_SEQUENCE[steps]
            else:
                action = 11  # submit_verdict_misinfo
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            log.append({
                "step": steps + 1,
                "action": ACTIONS[action],
                "reward": round(float(reward), 5),
                "done": done,
            })
            steps += 1

        verdict = env.last_verdict if hasattr(env, "last_verdict") else "misinfo"
        true_label = env.graph.true_label if env.graph else "unknown"

        return {
            "episode_id": episode_id,
            "task": task_name or info.get("task_id", "unknown"),
            "agent_id": agent_id,
            "steps": steps,
            "total_reward": round(total_reward, 5),
            "verdict": verdict,
            "true_label": true_label,
            "correct": verdict == true_label,
            "log": log,
            "status": "completed",
        }
    except Exception as e:
        return {
            "episode_id": "error",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_episode(task_name=task)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("status") == "completed" else 1)
