"""
scripts/run_selfplay.py — Self-play training entry point for Docker/CLI.
Runs N generations of PBT adversarial training and saves population.
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("run_selfplay")


def main():
    parser = argparse.ArgumentParser(description="FORGE Self-Play Adversarial Training")
    parser.add_argument("--generations",       type=int, default=20)
    parser.add_argument("--episodes-per-gen",  type=int, default=30)
    parser.add_argument("--difficulty",        type=int, default=2, choices=[1,2,3,4])
    parser.add_argument("--population-size",   type=int, default=config.GENERATOR_POPULATION_SIZE)
    parser.add_argument("--save-dir",          type=str, default="checkpoints/selfplay")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()

    from env.misinfo_env import MisInfoForensicsEnv
    try:
        from agents.adversarial.self_play import SelfPlayTrainer
    except ImportError:
        logger.error(
            "SelfPlayTrainer not found — agents/adversarial/self_play.py has not been "
            "implemented yet. Stub it or implement the full class before running self-play. "
            "See scripts/run_selfplay.py for the expected API."
        )
        raise SystemExit(1)

    env = MisInfoForensicsEnv()
    obs_dim = env.observation_space.shape[0]

    trainer = SelfPlayTrainer(
        obs_dim=obs_dim,
        population_size=args.population_size,
        seed=args.seed,
    )

    logger.info("Starting self-play: %d generations × %d episodes @ difficulty=%d",
                args.generations, args.episodes_per_gen, args.difficulty)

    history = []
    for gen_idx in range(args.generations):
        stats = trainer.run_generation(
            episodes_per_generator=args.episodes_per_gen,
            difficulty=args.difficulty,
        )
        history.append(stats)
        logger.info(
            "Gen %02d/%02d | best_gen=%s elo=%d | inv_reward=%.3f | pg_loss=%.4f",
            gen_idx + 1, args.generations,
            stats["best_generator"], stats["best_gen_elo"],
            stats["inv_mean_reward"], stats["ppo_pg_loss"],
        )

    trainer.save_population(args.save_dir)

    # Write history
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    logger.info("Self-play complete. Results saved to %s", args.save_dir)


if __name__ == "__main__":
    main()
