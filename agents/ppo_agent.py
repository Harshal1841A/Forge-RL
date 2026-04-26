"""
PPOAgent — Proximal Policy Optimisation with:
  - GAE (Generalised Advantage Estimation)
  - Orthogonal initialisation
  - Gradient clipping
  - Entropy bonus for exploration
  - Supports MLPPolicy and GATPolicy

100% free — pure PyTorch.
"""

from __future__ import annotations
import logging
import os
from collections import deque
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   # moved to top — was erroneously imported at line 208 after first use
import torch.optim as optim

from agents.gnn_policy import build_policy
from env.misinfo_env import MisInfoForensicsEnv
import config

logger = logging.getLogger(__name__)


# ─── Rollout Buffer ───────────────────────────────────────────────────────────

class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.full = False

    def add(self, obs, action, reward, value, log_prob, done):
        i = self.ptr % self.size
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def compute_gae(
        self, last_value: float,
        gamma: float = config.PPO_GAMMA,
        lam: float = config.PPO_GAE_LAMBDA,
    ) -> None:
        n = self.size
        gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_val * not_done - self.values[t]
            gae = delta + gamma * lam * not_done * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]

    def get_batches(self, batch_size: int):
        idx = np.random.permutation(self.size)
        for start in range(0, self.size, batch_size):
            b = idx[start: start + batch_size]
            yield (
                torch.FloatTensor(self.obs[b]),
                torch.LongTensor(self.actions[b]),
                torch.FloatTensor(self.advantages[b]),
                torch.FloatTensor(self.returns[b]),
                torch.FloatTensor(self.log_probs[b]),
            )


# ─── PPO Agent ────────────────────────────────────────────────────────────────

class PPOAgent:
    name = "ppo"

    def __init__(
        self,
        obs_dim: int,
        use_gnn: bool = False,   # GNN needs graph data, MLP for flat obs
        lr: float = config.PPO_LR,
        device: str = "cpu",
    ):
        self.obs_dim = obs_dim
        self.device = torch.device(device)
        self.policy = build_policy(obs_dim, use_gnn=use_gnn).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=1000
        )

        self.buffer = RolloutBuffer(config.PPO_TRAIN_BATCH, obs_dim)
        self.ep_rewards: deque = deque(maxlen=100)
        self.total_steps = 0
        self.updates = 0

    # ── Action selection ──────────────────────────────────────────────────────

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Returns (action, log_prob, value)."""
        # Both MLPPolicy and GATPolicy dynamically handle flat obs (GAT falls back to a 0-node graph inside)
        return self.policy.get_action(obs, deterministic=deterministic)

    # ── Training ──────────────────────────────────────────────────────────────

    def collect_rollout(self, env: MisInfoForensicsEnv) -> Dict[str, float]:
        """Collect one full buffer of experience."""
        obs, _ = env.reset()
        ep_reward = 0.0
        completed_eps = 0

        for _ in range(config.PPO_TRAIN_BATCH):
            action, log_prob, value = self.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # FIXED: Only store 'terminated' as the done flag for GAE.
            # Truncated states are NOT terminal, so we must bootstrap their value.
            self.buffer.add(obs, action, reward, value, log_prob, terminated)
            self.total_steps += 1
            ep_reward += reward
            obs = next_obs

            if done:
                self.ep_rewards.append(ep_reward)
                ep_reward = 0.0
                completed_eps += 1
                obs, _ = env.reset()

        # Compute GAE using bootstrap value
        _, _, last_value = self.act(obs)
        self.buffer.compute_gae(last_value)

        return {
            "mean_reward": float(np.mean(self.ep_rewards)) if self.ep_rewards else 0.0,
            "total_steps": self.total_steps,
            "episodes": completed_eps,
        }

    def update(self) -> Dict[str, float]:
        """PPO update step over buffered experience."""
        adv = torch.FloatTensor(self.buffer.advantages)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # normalise advantages
        self.buffer.advantages[:] = adv.numpy()

        pg_losses, vf_losses, ent_losses, total_losses = [], [], [], []
        clip_fracs = []

        for _ in range(config.PPO_EPOCHS):
            for obs_b, act_b, adv_b, ret_b, old_lp_b in self.buffer.get_batches(
                config.PPO_MINI_BATCH
            ):
                obs_b = obs_b.to(self.device)
                act_b = act_b.to(self.device)
                adv_b = adv_b.to(self.device)
                ret_b = ret_b.to(self.device)
                old_lp_b = old_lp_b.to(self.device)

                # Route through forward correctly for both MLPPolicy and GATPolicy.
                # GATPolicy falls back to a zero-node graph when no graph data is
                # provided (flat-obs training mode).
                from agents.gnn_policy import GATPolicy
                if isinstance(self.policy, GATPolicy):
                    n = obs_b.shape[0]
                    node_feats = torch.zeros(n, self.policy.node_feat_dim)
                    edge_index = torch.zeros(2, 0, dtype=torch.long)
                    batch = torch.arange(n, dtype=torch.long)
                    logits, values = self.policy(obs_b, node_feats, edge_index, batch)
                else:
                    logits, values = self.policy(obs_b)
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_b).exp()
                pg1 = ratio * adv_b
                pg2 = ratio.clamp(1 - config.PPO_CLIP_EPS, 1 + config.PPO_CLIP_EPS) * adv_b
                pg_loss = -torch.min(pg1, pg2).mean()

                vf_loss = F.mse_loss(values, ret_b)

                loss = (
                    pg_loss
                    + config.PPO_VF_COEF * vf_loss
                    - config.PPO_ENTROPY_COEF * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                clip_frac = ((ratio - 1).abs() > config.PPO_CLIP_EPS).float().mean().item()
                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(entropy.item())
                total_losses.append(loss.item())
                clip_fracs.append(clip_frac)

        self.scheduler.step()
        self.updates += 1
        self.buffer.ptr = 0
        self.buffer.full = False

        return {
            "pg_loss": float(np.mean(pg_losses)),
            "vf_loss": float(np.mean(vf_losses)),
            "entropy": float(np.mean(ent_losses)),
            "total_loss": float(np.mean(total_losses)),
            "clip_frac": float(np.mean(clip_fracs)),
            "lr": self.scheduler.get_last_lr()[0],
            "updates": self.updates,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:  # FIXED: dirname("") → "" which causes makedirs to raise FileNotFoundError
            os.makedirs(parent, exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }, path)
        logger.info("Saved PPO checkpoint → %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:       # backward-compatible with old checkpoints
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.total_steps = ckpt.get("total_steps", 0)
        self.updates = ckpt.get("updates", 0)
        logger.info("Loaded PPO checkpoint ← %s (step %d)", path, self.total_steps)
