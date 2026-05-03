import os
import sys
import json
from pathlib import Path

# Add root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from env.forge_env import ForgeEnv, ACTIONS
from blue_team.society_of_thought import SocietyOfThought

def run_demo():
    print("--- Starting FORGE-MA Demo Investigation ---")
    # Use live tools to verify the real APIs we just checked
    env = ForgeEnv(use_live_tools=True)
    obs, info = env.reset()
    
    # Correct way to get root info from the environment
    root_text, root_domain, root_virality = env.get_graph_root_info()
    
    print(f"Target Claim ID: {info.get('graph_id')}")
    print(f"Claim Text: {root_text[:100]}...")
    print(f"Source Domain: {root_domain}")
    
    # We will run a fixed high-quality sequence
    TOOL_SEQUENCE = [
        "query_source", 
        "trace_origin", 
        "temporal_audit",
        "cross_reference", 
        "entity_link", 
        "network_cluster",
        "flag_manipulation"
    ]
    
    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    
    total_reward = 0
    steps = []
    
    for tool in TOOL_SEQUENCE:
        print(f"\n[Step {len(steps)+1}] Calling Tool: {tool}")
        idx = action_to_idx[tool]
        obs, rew, term, trunc, info = env.step(idx)
        total_reward += rew
        steps.append({
            "step": len(steps)+1,
            "tool": tool,
            "reward": rew,
            "obs_summary": f"Obs dim: {len(obs)}"
        })
        
        # Print a snippet of the tool output if available in info
        if "last_tool_output" in info:
            print(f"  Output: {str(info['last_tool_output'])[:150]}...")

    print("\n[Final Step] Submitting Verdict...")
    idx = action_to_idx["submit_verdict_misinfo"]
    obs, rew, term, trunc, info = env.step(idx)
    total_reward += rew
    
    print("\n--- Investigation Complete ---")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Verdict Correct: {info.get('verdict_correct')}")
    print(f"Final TED Score: {info.get('ted', 0.001):.4f}")
    
    # Save the trace for the frontend to potentially pick up
    trace_data = {
        "claim_id": info.get('claim_id'),
        "steps": steps,
        "total_reward": total_reward,
        "ted": info.get('ted', 0.001),
        "graph_nodes": len(env.graph.nodes) if hasattr(env, 'graph') else 0
    }
    
    os.makedirs("baselines", exist_ok=True)
    with open("baselines/demo_trace.json", "w") as f:
        json.dump(trace_data, f, indent=2)
    print("\nTrace saved to baselines/demo_trace.json")

if __name__ == "__main__":
    run_demo()
