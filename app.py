import gradio as gr
import sys
import time
from pathlib import Path

# Root is the directory containing this file (project root)
sys.path.insert(0, str(Path(__file__).parent))

import config
from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
from agents.llm_agent import LLMAgent
from env.tasks import TASK_REGISTRY
import logging

logging.getLogger("env").setLevel(logging.CRITICAL)
logging.getLogger("agents").setLevel(logging.CRITICAL)

def investigate(task_name, difficulty):
    agent = LLMAgent()
    
    # We instantiate a specific task environment
    env = MisInfoForensicsEnv(task_names=[task_name], difficulty=int(difficulty))
    ep_seed = int(time.time()) % 100000
    obs, info = env.reset(seed=ep_seed)
    
    if hasattr(agent, "reset"):
        agent.reset()

    logs = []
    claim_text = env.graph.root.text if env.graph else ""
    logs.append(f"Starting Investigation...")
    logs.append(f"Task: {info['task_id']}")
    logs.append(f"Claim: {claim_text}\n")
    yield "\n".join(logs)

    done = False
    verdict = None
    step_info = {}

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
        done = terminated or truncated
        
        logs.append(f"[Step {env.steps}] Action: {action_name}")
        yield "\n".join(logs)

        if step_info.get("verdict"):
            verdict = step_info["verdict"]

    true_label = env.graph.true_label if env.graph else "unknown"
    correct = (verdict == true_label)
    
    logs.append(f"\nInvestigation Complete.")
    logs.append(f"Predicted Verdict: {verdict}")
    logs.append(f"True Label: {true_label}")
    logs.append(f"Success: {correct}")
    
    yield "\n".join(logs)

with gr.Blocks(title="MisInfo Forensics Investigation AI") as demo:
    gr.Markdown("# MisInfo Forensics Investigation AI")
    gr.Markdown("Select a task scenario to have the LLM agent autonomously investigate it.")
    
    with gr.Row():
        task_dropdown = gr.Dropdown(choices=list(TASK_REGISTRY.keys()), value=list(TASK_REGISTRY.keys())[0], label="Select Task")
        difficulty_slider = gr.Slider(minimum=1, maximum=4, step=1, value=1, label="Difficulty")
        
    start_btn = gr.Button("Start Investigation")
    output_text = gr.Textbox(lines=20, label="Investigation Logs")
    
    start_btn.click(fn=investigate, inputs=[task_dropdown, difficulty_slider], outputs=output_text)

if __name__ == "__main__":
    demo.launch()
