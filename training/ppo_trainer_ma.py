"""
REINFORCE Trainer wrapper for FORGE-RL (formerly mis-labeled as PPO).

This module implements REINFORCE / Monte-Carlo policy gradient with a mean
baseline — NOT PPO. It has no clipped surrogate objective, no value head,
no GAE, and no KL constraint. Renamed from PPOTrainer to REINFORCETrainer
to avoid mis-representing the algorithm in code or external claims.

For backwards compatibility, the legacy `PPOTrainer` symbol still resolves
to this class (with a DeprecationWarning).

  - Wraps a Red-team policy with ForgeEnv episode collection
  - Supports demo/dry-run mode (no model needed) for CI
  - Collects episodes in batches, computes mean-baseline advantages, logs stats
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


class REINFORCETrainer:
    """
    Lightweight REINFORCE (Monte-Carlo policy gradient with mean baseline).

    NOT PPO: there is no clipped surrogate, no value head, no GAE, no KL
    constraint. The previous `PPOTrainer` name was a misnomer; the gradient
    update below is exactly REINFORCE with a mean-reward baseline.

    In demo/CI mode runs full episodes but skips gradient updates.
    Set use_trl=True and pass a model to enable the policy gradient update.
    """

    def __init__(
        self,
        env_config: Optional[ForgeEnvConfig] = None,
        n_episodes_per_generation: int = 50,
        max_generations: int = 10,
        use_trl: bool = False,
        model=None,
        tokenizer=None,
    ):
        self.env = ForgeEnv(env_config or ForgeEnvConfig(budget=10, seed=42))
        self.n_episodes = n_episodes_per_generation
        self.max_generations = max_generations
        self.use_trl = use_trl
        self.model = model or self.env.red_agent.hae
        self.tokenizer = tokenizer
        self.stats = TrainingStats()

        import torch as _t
        import os as _os
        self._red_optimizer = _t.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self._red_ckpt_path = "checkpoints/red_hae/model.pt"
        if _os.path.exists(self._red_ckpt_path):
            try:
                ckpt = _t.load(self._red_ckpt_path, map_location="cpu")
                self.model.load_state_dict(ckpt["model"])
                self._red_optimizer.load_state_dict(ckpt["optimizer"])
                print(f"[Red] Resumed from checkpoint gen={ckpt.get('generation')}")
            except Exception as _ce:
                print(f"[Red] Checkpoint load failed ({_ce}), starting fresh.")

        from training.curriculum import CurriculumManager
        self._curriculum = CurriculumManager()

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
        # Propagate generation to env and expert reviewer
        self.env.training_generation = gen

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
                    x_blue = info.get("blue_graph_x")
                    ei_blue = info.get("blue_graph_edge_index")
                    tc_blue = info.get("true_chain")
                    if ep_out:
                        episodes.append(ep_out)
                        self.stats.update(ep_out, ep_reward)
                        self._curriculum.record_episode_reward(ep_reward)
                        self._replay_buffer.add(
                            ep_out, x=x_blue, edge_index=ei_blue
                        )
                    if "red_agent_history" in info:
                        batch_history.append((ep_out, info["red_agent_history"]))
                    if x_blue is not None and tc_blue is not None:
                        blue_episodes.append((x_blue, ei_blue, tc_blue))
                    break

        advanced = self._curriculum.check_progression()
        if advanced:
            _budget = max(5, int(10 * self._curriculum.budget_multiplier))
            self.env = ForgeEnv(ForgeEnvConfig(budget=_budget, seed=gen))
            self.env.training_generation = gen
            self.model = self.env.red_agent.hae
            import torch
            self._red_optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=1e-4, weight_decay=1e-5
            )
            self._replay_buffer = ReplayBuffer(capacity=1000)
            self._gin_trainer = GINTrainer(
                gin=self.env.gin,
                replay_buffer=self._replay_buffer,
            )
            print(
                f"[Curriculum] Stage {self._curriculum.current_stage} "
                f"budget={_budget}"
            )

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
            import os as _os
            import torch

            try:
                red_rewards = []
                for ep_out, history in batch_history:
                    r_steps = (
                        sum(ep_out.red_step_rewards)
                        if ep_out and hasattr(ep_out, "red_step_rewards")
                        else 0.0
                    )
                    blue_r = ep_out.reward_total if ep_out else 0.0
                    red_rewards.append(r_steps - blue_r)

                loss_terms = []
                if red_rewards:
                    mean_red = sum(red_rewards) / len(red_rewards)
                    device = next(self.model.parameters()).device

                    for (ep_out, history), r_score in zip(
                        batch_history, red_rewards
                    ):
                        advantage = float(r_score - mean_red)

                        for step_data in history:
                            x, edge_index, ti, pi = step_data
                            x = x.to(device)
                            edge_index = edge_index.to(device)

                            out = self.model(x, edge_index)
                            a_logits = out["action_logits"][0]
                            p_logits = out["primitive_logits"][0]

                            log_a = torch.log_softmax(a_logits, dim=-1)[ti]
                            log_p = torch.log_softmax(p_logits, dim=-1)[pi]
                            log_prob = log_a + log_p

                            loss_terms.append(-(log_prob * advantage))

                if self.use_trl and self.model is not None and loss_terms:
                    total_loss = torch.stack(loss_terms).sum()
                    self._red_optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=0.5
                    )
                    self._red_optimizer.step()

                    _os.makedirs(
                        _os.path.dirname(self._red_ckpt_path), exist_ok=True
                    )
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "optimizer": self._red_optimizer.state_dict(),
                            "generation": gen,
                        },
                        self._red_ckpt_path,
                    )
                    print(
                        f"[Red] gen={gen} loss={total_loss.item():.4f} ckpt saved."
                    )
            except Exception as e:
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
                    campaign_name=f"FORGE-RL Training Gen {gen}",
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


# ── Backwards-compatibility alias ────────────────────────────────────────────
# The class was previously called `PPOTrainer`, but it implements REINFORCE.
# Existing imports (`from training.ppo_trainer_ma import PPOTrainer`) keep
# working through this alias — emit a DeprecationWarning when used.
class PPOTrainer(REINFORCETrainer):
    """Deprecated alias for REINFORCETrainer. Misleadingly named — this is
    REINFORCE with a mean baseline, not PPO. Use REINFORCETrainer instead.
    """

    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn(
            "PPOTrainer is a misnomer for REINFORCE — use REINFORCETrainer "
            "instead. This alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
