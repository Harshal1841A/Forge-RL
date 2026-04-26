"""
pretrain.py  —  50-generation pre-training pipeline for HAE (Red) and GIN (Blue).

Usage:
    python pretrain.py                    # trains 50 gens, saves checkpoints
    python -c "from pretrain import load_checkpoints; load_checkpoints(env)"
"""
import os
import sys
import time
import logging
import torch
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from training.ppo_trainer_ma import PPOTrainer

logger = logging.getLogger(__name__)

HAE_CKPT = ROOT / "checkpoints" / "hae_model.pt"
GIN_CKPT = ROOT / "checkpoints" / "gin_model.pt"


# ── Public helpers ──────────────────────────────────────────────────────────

def load_checkpoints(forge_env) -> bool:
    """
    Load saved HAE + GIN weights into an existing ForgeEnv instance.
    Returns True if both checkpoints were loaded, False otherwise.
    Called at app startup so demo runs on trained weights.
    """
    loaded = False
    if HAE_CKPT.exists():
        try:
            forge_env.red_agent.hae.load_state_dict(
                torch.load(str(HAE_CKPT), map_location="cpu")
            )
            logger.info("[pretrain] HAE checkpoint loaded from %s", HAE_CKPT)
            loaded = True
        except Exception as exc:
            logger.warning("[pretrain] Failed to load HAE checkpoint: %s", exc)

    if GIN_CKPT.exists():
        try:
            forge_env.gin.model.load_state_dict(
                torch.load(str(GIN_CKPT), map_location="cpu")
            )
            logger.info("[pretrain] GIN checkpoint loaded from %s", GIN_CKPT)
            loaded = True
        except Exception as exc:
            logger.warning("[pretrain] Failed to load GIN checkpoint: %s", exc)

    return loaded


def checkpoints_exist() -> bool:
    """Return True if both checkpoint files exist on disk."""
    return HAE_CKPT.exists() and GIN_CKPT.exists()


# ── Training entry-point ────────────────────────────────────────────────────

def pretrain(n_generations: int = 50, n_episodes: int = 10):
    # n_episodes bumped from 4 → 10. With n_generations=50 this gives 500
    # total training episodes — the minimum viable volume per the review's P7.
    """Run full pretraining and write checkpoints to disk."""
    log_path = ROOT / "pretrain_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        def log(msg: str):
            print(msg, flush=True)
            f.write(msg + "\n")
            f.flush()

        log(f"Starting {n_generations} generations of pre-training "
            f"({n_episodes} episodes/gen)...")

        trainer = PPOTrainer(
            max_generations=n_generations,
            n_episodes_per_generation=n_episodes,
            use_trl=True,
        )
        trainer.model = trainer.env.red_agent.hae

        t0 = time.time()
        for gen in range(n_generations):
            log(f"--- Generation {gen + 1}/{n_generations} ---")
            trainer.stats.generation = gen
            trainer._run_generation(gen)
            log(
                f"Gen {gen + 1} complete. "
                f"min={trainer.stats.min_reward:.2f}  "
                f"mean={trainer.stats.mean_reward:.2f}"
            )

        elapsed = time.time() - t0
        log(f"Pre-training complete in {elapsed:.2f}s.")

        # ── Save checkpoints ────────────────────────────────────────────────
        os.makedirs(ROOT / "checkpoints", exist_ok=True)

        torch.save(trainer.env.gin.model.state_dict(), str(GIN_CKPT))
        log(f"GIN checkpoint saved -> {GIN_CKPT}")

        torch.save(trainer.env.red_agent.hae.state_dict(), str(HAE_CKPT))
        log(f"HAE checkpoint saved -> {HAE_CKPT}")

        # Log training curve
        import json
        with open(ROOT / "checkpoints" / "training_log.json", "w") as jf:
            json.dump({
                "n_generations": n_generations,
                "n_episodes": n_episodes,
                "min_rewards": trainer.stats.min_rewards if hasattr(trainer.stats, "min_rewards") else [],
                "mean_rewards": trainer.stats.mean_rewards if hasattr(trainer.stats, "mean_rewards") else [],
                "final_mean_reward": trainer.stats.mean_reward,
                "elapsed_seconds": elapsed
            }, jf, indent=2)
        log("Training curve logged to checkpoints/training_log.json")

if __name__ == "__main__":
    pretrain()
