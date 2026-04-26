"""
GIN Trainer for FORGE-RL Blue Team.
SPEC (PRD v8.1 §Blue Team Training):

  The Blue Team (GINPredictor) learns via supervised signal:
    Loss = BCE(presence_probs, true_presence_labels)

  Two complementary training sources:
    1. Online:   fresh episode graphs from the current generation rollout.
    2. Offline:  high-quality episodes sampled from the ReplayBuffer.

  This is the symmetric counterpart to the Red Team's REINFORCE update:
    - Red Team  → maximises adversarial reward (confuse Blue)
    - Blue Team → minimises detection loss (correctly detect Red's primitives)

  Together they form the co-evolutionary self-play loop.

Usage:
    gin_trainer = GINTrainer(gin_predictor, replay_buffer)
    gin_trainer.update(blue_episodes)   # called after each PPO generation
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from blue_team.gin_predictor import GINPredictor
from blue_team.replay_buffer import ReplayBuffer
from env.episode_output import EpisodeOutput
from env.primitives import PrimitiveType


@dataclass
class GINTrainingStats:
    """Per-generation statistics for Blue Team learning."""
    generation: int = 0
    online_steps: int = 0
    offline_steps: int = 0
    mean_online_loss: float = 0.0
    mean_offline_loss: float = 0.0
    total_loss: float = 0.0
    history: List[dict] = field(default_factory=list)

    def record(self, online_loss: float, offline_loss: float, gen: int):
        self.generation = gen
        self.total_loss = online_loss + offline_loss
        self.history.append({
            "gen": gen,
            "online_loss": online_loss,
            "offline_loss": offline_loss,
            "total": self.total_loss,
        })

    def summary(self) -> dict:
        return {
            "generation":        self.generation,
            "online_steps":      self.online_steps,
            "offline_steps":     self.offline_steps,
            "mean_online_loss":  round(self.mean_online_loss, 6),
            "mean_offline_loss": round(self.mean_offline_loss, 6),
            "total_loss":        round(self.total_loss, 6),
        }


class GINTrainer:
    """
    Supervised trainer for the Blue Team (BlueGIN).

    Parameters
    ----------
    gin : GINPredictor
        The Blue Team GIN predictor (holds BlueGIN model + AdamW optimizer).
    replay_buffer : ReplayBuffer
        Stores high-quality episodes for off-policy training.
    offline_batch_size : int
        Number of replay episodes to train on each generation.
    online_weight : float
        Weight for the online loss contribution (default 1.0).
    offline_weight : float
        Weight for the offline replay loss contribution (default 0.5).
        Lower than 1.0 to reduce overfitting to old experience.
    """

    def __init__(
        self,
        gin: GINPredictor,
        replay_buffer: ReplayBuffer,
        offline_batch_size: int = 8,
        online_weight: float = 1.0,
        offline_weight: float = 0.5,
    ):
        self.gin = gin
        self.replay_buffer = replay_buffer
        self.offline_batch_size = offline_batch_size
        self.online_weight = online_weight
        self.offline_weight = offline_weight
        self.stats = GINTrainingStats()
        self._generation = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        blue_episodes: List[Tuple[torch.Tensor, torch.Tensor, List[PrimitiveType]]],
        generation: int,
    ) -> GINTrainingStats:
        """
        Run one Blue Team training update.

        Parameters
        ----------
        blue_episodes : list of (x, edge_index, true_chain)
            Fresh episode graphs from the current generation rollout.
            x           : node feature tensor
            edge_index  : edge connectivity tensor
            true_chain  : ground-truth list of PrimitiveType
        generation : int
            Current PPO generation number.

        Returns
        -------
        GINTrainingStats with loss breakdown.
        """
        self._generation = generation
        self.replay_buffer.set_generation(generation)

        # ── 1. Online training on fresh episodes ───────────────────────────────
        online_losses = []
        for x, edge_index, true_chain in blue_episodes:
            if x is None or edge_index is None or not true_chain:
                continue
            try:
                loss = self.gin.train_step(x, edge_index, true_chain)
                online_losses.append(loss)
            except Exception as e:
                print(f"⚠️ GINTrainer online step error: {e}")

        self.stats.online_steps += len(online_losses)
        mean_online = sum(online_losses) / len(online_losses) if online_losses else 0.0

        # ── 2. Offline training from ReplayBuffer ─────────────────────────────
        offline_losses = []
        if self.replay_buffer.size >= self.offline_batch_size:
            sampled = self.replay_buffer.sample(n=self.offline_batch_size)
            for ep_out in sampled:
                # EpisodeOutput stores true_chain as tuple of primitive *values*
                try:
                    true_chain = [PrimitiveType(v) for v in ep_out.true_chain]
                except (ValueError, KeyError):
                    continue  # skip malformed entries

                # Replay buffer only has EpisodeOutput, not raw tensors.
                # Reconstruct a minimal graph for training based on what GIN
                # originally saw: a single root node with zero features.
                # The signal is still valid: the labels come from the true chain.
                n_nodes = max(1, ep_out.steps_taken)
                from env.node_features import build_node_features
                from env.claim_graph_ma import ClaimNode
                dummy_node = ClaimNode(id="dummy", text="", domain="replay", trust_score=0.5, is_retrieved=False, injected=False, primitive=None, fingerprints={})
                feat = build_node_features(dummy_node, 10)
                x_replay = torch.tensor([feat for _ in range(n_nodes)], dtype=torch.float32)
                edge_index_replay = torch.zeros((2, 0), dtype=torch.long)

                try:
                    loss = self.gin.train_step(x_replay, edge_index_replay, true_chain)
                    offline_losses.append(loss)
                except Exception as e:
                    print(f"⚠️ GINTrainer offline step error: {e}")

        self.stats.offline_steps += len(offline_losses)
        mean_offline = sum(offline_losses) / len(offline_losses) if offline_losses else 0.0

        # ── 3. Record stats ───────────────────────────────────────────────────
        self.stats.mean_online_loss = mean_online
        self.stats.mean_offline_loss = mean_offline
        self.stats.record(mean_online, mean_offline, generation)

        # ── 4. Persist checkpoint after every generation. ──────────────────────
        # Without this, training is discarded on process restart and the
        # deployed server reverts to xavier-init weights on every boot.
        try:
            self.gin.save_checkpoint()
        except Exception as e:
            print(f"WARNING: GINTrainer failed to save checkpoint: {e}")

        return self.stats
