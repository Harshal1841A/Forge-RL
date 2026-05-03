"""
Multi-Agent evaluation script matching the FORGE-MA architecture.
Loads REINFORCE Red (HAEModel) and PPO Blue (blue_ppo_agent).
"""
import os
import torch
import numpy as np
from env.forge_env import ForgeEnv
from red_team.hae_model import HAEModel
from agents.blue_ppo_agent import PPOAgent
import config

def evaluate_ma(red_ckpt: str, blue_ckpt: str, episodes: int = 10):
    env = ForgeEnv()
    
    # LOAD RED into environment
    if os.path.exists(red_ckpt):
        env.red_agent.hae.load_state_dict(torch.load(red_ckpt, map_location="cpu"))
        print(f"Loaded RED from {red_ckpt}")
    else:
        print("RED ckpt missing, using untrained.")
    env.red_agent.hae.eval()

    # LOAD BLUE policy
    blue_agent = PPOAgent(obs_dim=3859)
    if os.path.exists(blue_ckpt):
        blue_agent.load(blue_ckpt)
        print(f"Loaded BLUE from {blue_ckpt}")
    else:
        print("BLUE ckpt missing, using untrained.")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0
        
        while not done:
            # Blue agent observes the state and takes an action
            # (Note: ForgeEnv step currently ignores this action as Red acts autonomously)
            flat_obs = blue_agent._flatten_obs(obs)
            action, _, _ = blue_agent.act(flat_obs, deterministic=True)
            
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1
            
        print(f"Episode {ep+1}: Steps={step}, Final Blue Score={info.get('blue_score', 0)}, Reward={reward}")

if __name__ == "__main__":
    evaluate_ma(
        red_ckpt=os.path.join("checkpoints", "red_hae_final.pt"),
        blue_ckpt=os.path.join("checkpoints", "blue_ppo_final.pt"),
        episodes=5
    )
