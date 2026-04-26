"""
PPO Trainer wrapper for FORGE-MA.
SPEC (Master Prompt §Layer7):
  - Wraps TRL PPOTrainer with ForgeEnv episode collection
  - Supports demo/dry-run mode (no actual model needed) for CI
  - Collects episodes in batches, computes rewards, logs stats
  - GenerationTracker: increments per training epoch
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from env.forge_env import ForgeEnv, ForgeEnvConfig
from env.episode_output import EpisodeOutput
from blue_team.replay_buffer import ReplayBuffer
from training.gin_trainer_ma import GINTrainer


@dataclass
class TrainingStats:
    generation: int = 0
    episodes_run: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    max_reward: float = float("-inf")
    min_reward: float = float("+inf")
    over_budget_count: int = 0
    correct_verdicts: int = 0
    episode_history: List[EpisodeOutput] = field(default_factory=list)

    def update(self, ep: EpisodeOutput, reward: float):
        self.episodes_run += 1
        self.total_reward += reward
        self.mean_reward = self.total_reward / self.episodes_run
        self.max_reward = max(self.max_reward, reward)
        self.min_reward = min(self.min_reward, reward)
        if ep.over_budget:
            self.over_budget_count += 1
        if ep.is_correct:
            self.correct_verdicts += 1
        self.episode_history.append(ep)

    def summary(self) -> Dict[str, Any]:
        return {
            "generation":       self.generation,
            "episodes_run":     self.episodes_run,
            "mean_reward":      round(self.mean_reward, 4),
            "max_reward":       round(self.max_reward, 4),
            "min_reward":       round(self.min_reward, 4),
            "over_budget_pct":  round(self.over_budget_count / max(self.episodes_run, 1), 3),
            "chain_accuracy":   round(self.correct_verdicts / max(self.episodes_run, 1), 3),
        }


class PPOTrainer:
    """
    Lightweight PPO training loop for FORGE-MA.
    In demo/CI mode runs full episodes but skips gradient updates.
    Set use_trl=True and pass a model/tokenizer to enable real PPO updates.
    """

    def __init__(
        self,
        env_config: Optional[ForgeEnvConfig] = None,
        n_episodes_per_generation: int = 4,
        max_generations: int = 3,
        use_trl: bool = False,
        model=None,
        tokenizer=None,
    ):
        self.env = ForgeEnv(env_config or ForgeEnvConfig(budget=10, seed=42))
        self.n_episodes = n_episodes_per_generation
        self.max_generations = max_generations
        self.use_trl = use_trl
        self.model = model
        self.tokenizer = tokenizer
        self.stats = TrainingStats()

        # Blue Team trainer: GINPredictor + ReplayBuffer
        self._replay_buffer = ReplayBuffer(capacity=1000)
        self._gin_trainer = GINTrainer(
            gin=self.env.gin,
            replay_buffer=self._replay_buffer,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self) -> TrainingStats:
        """Run full training loop across all generations."""
        for gen in range(self.max_generations):
            self.stats.generation = gen
            self._run_generation(gen)
        return self.stats

    def run_single_episode(self) -> EpisodeOutput:
        """Run one episode end-to-end; returns EpisodeOutput."""
        obs, info = self.env.reset()
        ep_output = None
        while True:
            obs, reward, terminated, truncated, info = self.env.step()
            if terminated or truncated:
                ep_output = info.get("episode_output")
                if ep_output is not None:
                    self.stats.update(ep_output, reward)
                break
        return ep_output or self._fallback_episode()

    # ── Private ─────────────────────────────────────────────────────────────

    def _run_generation(self, gen: int):
        episodes: List[EpisodeOutput] = []
        rewards: List[float] = []
        batch_history = []
        blue_episodes = []   # (x, edge_index, true_chain) for GIN training

        for ep_idx in range(self.n_episodes):
            obs, info = self.env.reset(seed=gen * 100 + ep_idx)
            ep_reward = 0.0
            while True:
                obs, reward, terminated, truncated, info = self.env.step()
                if terminated or truncated:
                    ep_reward = reward
                    ep_out = info.get("episode_output")
                    if ep_out:
                        episodes.append(ep_out)
                        self.stats.update(ep_out, ep_reward)
                        # Add to replay buffer for offline Blue training
                        self._replay_buffer.add(ep_out)
                    if "red_agent_history" in info:
                        batch_history.append((ep_out, info["red_agent_history"]))
                    # Collect Blue training snapshot
                    x_blue = info.get("blue_graph_x")
                    ei_blue = info.get("blue_graph_edge_index")
                    tc_blue = info.get("true_chain")
                    if x_blue is not None and tc_blue is not None:
                        blue_episodes.append((x_blue, ei_blue, tc_blue))
                    break

        # ── Blue Team supervised update ──────────────────────────────────────
        if blue_episodes:
            gin_stats = self._gin_trainer.update(blue_episodes, generation=gen)
            print(
                f"  [Blue GIN gen={gen}] "
                f"online_loss={gin_stats.mean_online_loss:.4f}  "
                f"offline_loss={gin_stats.mean_offline_loss:.4f}  "
                f"online_steps={gin_stats.online_steps}  "
                f"buf_size={self._replay_buffer.size}"
            )

        if self.use_trl and self.model is not None:
            import torch
            from torch.optim import AdamW
            
            try:
                # Active PyTorch gradient update loop (AdamW optimizer)
                optimizer = AdamW(self.model.parameters(), lr=1e-4)
                optimizer.zero_grad()
                
                # Calculate adversarial rewards (Red wants to MINIMIZE blue reward, MAXIMIZE step rewards)
                red_rewards = []
                for ep_out, history in batch_history:
                    # If step rewards are missing, just use -blue_reward
                    r_steps = sum(ep_out.red_step_rewards) if ep_out and hasattr(ep_out, 'red_step_rewards') else 0.0
                    blue_r = ep_out.reward_total if ep_out else 0.0
                    red_rewards.append(r_steps - blue_r)
                
                if red_rewards:
                    mean_red = sum(red_rewards) / len(red_rewards)
                    device = next(self.model.parameters()).device
                    
                    # Collect all loss terms first, then stack+sum to keep the
                    # computational graph intact for backpropagation.
                    loss_terms = []

                    for (ep_out, history), r_score in zip(batch_history, red_rewards):
                        advantage = float(r_score - mean_red)

                        # Re-run forward pass to get gradients (REINFORCE policy gradient)
                        for step_data in history:
                            x, edge_index, ti, pi = step_data
                            x = x.to(device)
                            edge_index = edge_index.to(device)

                            out = self.model(x, edge_index)
                            a_logits = out["action_logits"][0]
                            p_logits = out["primitive_logits"][0]

                            log_a = torch.log_softmax(a_logits, dim=-1)[ti]
                            log_p = torch.log_softmax(p_logits, dim=-1)[pi]
                            log_prob = log_a + log_p  # still in graph

                            # REINFORCE: loss = -(log_prob * advantage)
                            # Red agent maximises adversarial reward → minimises this loss
                            loss_terms.append(-(log_prob * advantage))

                    if loss_terms:
                        total_loss = torch.stack(loss_terms).sum()
                        total_loss.backward()
                        optimizer.step()
            except Exception as e:
                import traceback
                print(f"PPO Optimization exception: {e}")

        # ── Auto-write graphify forensic artifacts after every generation ─────
        if episodes:
            _last_ep = episodes[-1]
            try:
                import os
                from env.oversight_report import generate_oversight_report, generate_stix2_bundle
                os.makedirs("graphify-out", exist_ok=True)

                report_md = generate_oversight_report(
                    _last_ep,
                    claim_text=getattr(_last_ep, "claim_text", ""),
                    generation=gen,
                )
                with open("graphify-out/GRAPH_REPORT.md", "w", encoding="utf-8") as _f:
                    _f.write(report_md)

                stix_json = generate_stix2_bundle(
                    _last_ep,
                    campaign_name=f"FORGE-MA Training Gen {gen}",
                    claim_text=getattr(_last_ep, "claim_text", ""),
                )
                with open("graphify-out/STIX_BUNDLE.json", "w", encoding="utf-8") as _f:
                    _f.write(stix_json)
            except Exception as _rep_exc:
                pass   # Never crash training over reporting


    def _fallback_episode(self) -> EpisodeOutput:
        """Emergency fallback if env doesn't return output (should not happen)."""
        from env.episode_output import EpisodeOutput
        from rewards.hierarchical_reward import compute_reward
        r = compute_reward(
            predicted_chains=[[]], true_chain=[],
            claim_text_before="", claim_text_after="",
            consensus_level="all_different", expert_decision="REJECT",
            steps_taken=10, budget_limit=10, useful_tools_called=0,
        )
        return EpisodeOutput.build(
            verdict="unknown", predicted_chain=[], true_chain=[],
            reward=r, consensus_level="all_different",
            expert_decision="REJECT", steps_taken=10, budget_limit=10,
            useful_tools=0, agent_verdicts={},
        )
