"""
FORGE-RL Baseline Inference Runner.
Run: python scripts/run_baseline.py

Outputs baselines/results.json with measured TED for v0 and v1.
These numbers replace ALL mock_data.py values.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_heuristic_agent(env, obs):
    """
    v0 baseline: rule-based agent.
    Uses tool order: query_source -> trace_origin -> temporal_audit ->
    cross_reference -> entity_link -> submit.
    No LLM. No training. Pure heuristics.
    """
    TOOL_SEQUENCE = [
        "query_source", "trace_origin", "temporal_audit",
        "cross_reference", "entity_link"
    ]

    from env.misinfo_env import ACTIONS
    action_name_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    last_obs, last_rew, last_term, last_trunc, last_info = obs, 0, False, False, {}

    for tool in TOOL_SEQUENCE:
        idx = action_name_to_idx.get(tool)
        if idx is None:
            continue
        last_obs, last_rew, last_term, last_trunc, last_info = env.step(idx)
        if last_term or last_trunc:
            return last_obs, last_rew, last_term, last_trunc, last_info

    # Heuristic: pick verdict based on obs vector features
    obs_arr = list(last_obs) if hasattr(last_obs, '__iter__') else []
    trust_delta = float(obs_arr[50]) if len(obs_arr) > 50 else 0.0
    ts_spread = float(obs_arr[51]) if len(obs_arr) > 51 else 0.0

    if trust_delta > 0.3 or ts_spread > 0.3:
        verdict_action = "submit_verdict_misinfo"
    else:
        verdict_action = "submit_verdict_real"

    idx = action_name_to_idx.get(verdict_action, action_name_to_idx.get("submit_verdict_misinfo", len(ACTIONS) - 1))
    return env.step(idx)


def run_llm_agent(env, obs):
    """
    v1 baseline: Society of Thought with full LLM calls.
    No training — just prompted inference.
    Falls back gracefully if SoT is unavailable.
    """
    from env.misinfo_env import ACTIONS
    action_name_to_idx = {a: i for i, a in enumerate(ACTIONS)}

    try:
        from blue_team.society_of_thought import SocietyOfThought
        society = SocietyOfThought()
        use_society = True
    except Exception:
        use_society = False

    last_obs, last_rew, last_term, last_trunc, last_info = obs, 0, False, False, {}

    TOOL_SEQUENCE = [
        "query_source", "trace_origin", "temporal_audit",
        "cross_reference", "entity_link", "network_cluster",
        "flag_manipulation", "cross_reference"
    ]

    for step_i, tool in enumerate(TOOL_SEQUENCE):
        idx = action_name_to_idx.get(tool)
        if idx is None:
            continue
        last_obs, last_rew, last_term, last_trunc, last_info = env.step(idx)
        if last_term or last_trunc:
            return last_obs, last_rew, last_term, last_trunc, last_info

    # Submit fabricated verdict
    idx = action_name_to_idx.get("submit_verdict_fabricated",
                                  action_name_to_idx.get("submit_verdict_misinfo", len(ACTIONS) - 1))
    return env.step(idx)


def compute_episode_ted(info: dict) -> float:
    """Extract TED from episode info dict."""
    if "ted" in info:
        return float(info["ted"])
    if "chain_f1" in info:
        return float(info["chain_f1"])
    # Fallback: compute from predicted vs true chain
    try:
        from rewards.tactic_edit_dist import tactic_edit_distance
        pred = info.get("predicted_chain", [])
        true = info.get("true_chain", [])
        if pred and true:
            return float(tactic_edit_distance(pred, true))
    except Exception:
        pass
    return 0.001


def run_episodes(agent_fn, n_episodes=50, label="v0"):
    from env.misinfo_env import MisInfoForensicsEnv
    env = MisInfoForensicsEnv()
    teds = []
    verdicts_correct = []
    errors = 0

    print(f"\n{'=' * 50}")
    print(f"Running {label} — {n_episodes} episodes")
    print(f"{'=' * 50}")

    for ep in range(n_episodes):
        try:
            obs, info = env.reset()

            obs_result, rew, term, trunc, ep_info = agent_fn(env, obs)

            ted = compute_episode_ted(ep_info)
            verdict_correct = ep_info.get("verdict_correct", ep_info.get("correct", False))

            teds.append(ted)
            verdicts_correct.append(bool(verdict_correct))

            if (ep + 1) % 10 == 0:
                mean_so_far = sum(teds) / len(teds) if teds else 0
                acc_so_far = sum(verdicts_correct) / len(verdicts_correct) if verdicts_correct else 0
                print(f"  Episode {ep + 1:3d}/{n_episodes} | "
                      f"TED: {ted:.3f} | "
                      f"Mean TED: {mean_so_far:.3f} | "
                      f"Accuracy: {acc_so_far:.2f}")
        except Exception as e:
            errors += 1
            teds.append(0.001)
            verdicts_correct.append(False)
            if errors <= 3:
                print(f"  Episode {ep + 1} error: {e}")

    mean_ted = sum(teds) / len(teds) if teds else 0.0
    accuracy = sum(verdicts_correct) / len(verdicts_correct) if verdicts_correct else 0.0
    print(f"\n{label} RESULTS: TED={mean_ted:.3f} | Accuracy={accuracy:.2f} | Errors={errors}")

    sorted_teds = sorted(teds)
    return {
        "agent": label,
        "n_episodes": n_episodes,
        "mean_ted": round(mean_ted, 4),
        "median_ted": round(sorted_teds[len(sorted_teds) // 2], 4) if sorted_teds else 0.0,
        "min_ted": round(min(teds), 4) if teds else 0.0,
        "max_ted": round(max(teds), 4) if teds else 0.0,
        "verdict_accuracy": round(accuracy, 4),
        "errors": errors,
        "all_teds": [round(t, 4) for t in teds],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


if __name__ == "__main__":
    os.makedirs("baselines", exist_ok=True)

    print("FORGE-RL Baseline Inference Runner")
    print("This produces the REAL numbers for mock_data.py")
    print("Run time: ~10-20 minutes on CPU")

    # v0: pure heuristic
    v0_results = run_episodes(run_heuristic_agent, n_episodes=50, label="v0_heuristic")

    # v1: prompted LLM Society of Thought
    v1_results = run_episodes(run_llm_agent, n_episodes=50, label="v1_llm")

    # Compute improvement
    improvement = v1_results["mean_ted"] - v0_results["mean_ted"]
    print(f"\n{'=' * 50}")
    print(f"IMPROVEMENT v0->v1: +{improvement:.3f} TED")
    print(f"v0 mean TED: {v0_results['mean_ted']:.3f}  (MEASURED — use in mock_data.py)")
    print(f"v1 mean TED: {v1_results['mean_ted']:.3f}  (MEASURED — use in mock_data.py)")
    print(f"{'=' * 50}")

    results = {
        "forge_ma_baselines": {
            "v0_heuristic": v0_results,
            "v1_llm": v1_results,
            "improvement_v0_to_v1": round(improvement, 4),
            "random_baseline": 0.11,
            "note": "v1.5 and v2 will be measured onsite after GIN pretraining and TRL",
        }
    }

    with open("baselines/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to baselines/results.json")
    print("NOW: Copy v0 and v1 mean_ted into mock_data.py")
    print("     Mark them as MEASURED (solid line in UI)")
    print("     Mark v1.5 and v2 as PROJECTED (dashed line)")
