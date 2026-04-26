"""
PPO Training Script — full training loop with curriculum, logging, checkpointing.
Run: python -m training.train_ppo
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from agents.ppo_agent import PPOAgent
from env.misinfo_env import MisInfoForensicsEnv
from training.curriculum import CurriculumManager
from training.eval import evaluate_agent

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("train_ppo")


def train(args: argparse.Namespace) -> None:
    logger.info("=" * 60)
    logger.info("FORGE — PPO Training (Cost-Free PyTorch)")
    logger.info("=" * 60)

    curriculum = CurriculumManager()
    env = MisInfoForensicsEnv(
        difficulty=curriculum.difficulty,
        budget_multiplier=curriculum.budget_multiplier,
    )
    obs_dim = env.observation_space.shape[0]

    agent = PPOAgent(obs_dim=obs_dim, use_gnn=False, lr=config.PPO_LR, device=args.device)

    # Resume from checkpoint if provided
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        logger.info("Resumed from checkpoint: %s", args.resume)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_reward = -float("inf")
    start_time  = time.time()

    for iteration in range(1, args.iterations + 1):
        # ── Update env difficulty from curriculum ──────────────────────────────
        curr_status = curriculum.status()
        env.difficulty = curriculum.difficulty
        env.budget_multiplier = curriculum.budget_multiplier

        # ── Collect rollout ────────────────────────────────────────────────────
        rollout_stats = agent.collect_rollout(env)

        # Record ONLY the new episode rewards from this rollout — previously the entire
        # ep_rewards rolling window was pushed every iteration, causing duplicate values
        # in the curriculum window and premature stage advancement.
        n_new = rollout_stats.get("episodes", 1)
        new_ep_rewards = list(agent.ep_rewards)[-n_new:]
        for r in new_ep_rewards:
            curriculum.record_episode_reward(r)
        curriculum.check_progression()

        # ── PPO update ─────────────────────────────────────────────────────────
        update_stats = agent.update()

        # ── Logging ───────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        logger.info(
            "[Iter %04d] reward=%.3f | pg=%.4f | vf=%.4f | ent=%.4f | "
            "clip=%.3f | stage=%d | steps=%d | t=%.0fs",
            iteration,
            rollout_stats["mean_reward"],
            update_stats["pg_loss"],
            update_stats["vf_loss"],
            update_stats["entropy"],
            update_stats["clip_frac"],
            curr_status["current_stage"],
            agent.total_steps,
            elapsed,
        )

        # ── Evaluation every N iters ──────────────────────────────────────────
        if iteration % args.eval_every == 0:
            eval_stats = evaluate_agent(agent, n_episodes=args.eval_episodes)
            logger.info(
                "[EVAL %04d] accuracy=%.3f | f1=%.3f | efficiency=%.3f",
                iteration, eval_stats["accuracy"], eval_stats["macro_f1"],
                eval_stats["mean_efficiency"],
            )
            # Save best checkpoint
            if eval_stats["accuracy"] > best_reward:
                best_reward = eval_stats["accuracy"]
                agent.save(f"{args.checkpoint_dir}/best.pt")
                logger.info("  💾 New best checkpoint saved (acc=%.3f)", best_reward)

        # ── Regular checkpoint ────────────────────────────────────────────────
        if iteration % args.save_every == 0:
            agent.save(f"{args.checkpoint_dir}/iter_{iteration:04d}.pt")

    logger.info("Training complete. Best accuracy: %.3f", best_reward)
    agent.save(f"{args.checkpoint_dir}/final.pt")


def main():
    parser = argparse.ArgumentParser(description="FORGE PPO Training")
    parser.add_argument("--iterations",     type=int,   default=500)
    parser.add_argument("--eval-every",     type=int,   default=50)
    parser.add_argument("--eval-episodes",  type=int,   default=100)
    parser.add_argument("--save-every",     type=int,   default=100)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints/ppo")
    parser.add_argument("--resume",         type=str,   default="")
    parser.add_argument("--device",         type=str,   default="cpu")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
