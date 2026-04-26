"""
runtime/blue_gin.py — process-wide GINPredictor singleton.

Both ForgeEnv (training) and MisInfoForensicsEnv (deployment) used to
instantiate their own GINPredictor. The trainer's gradient updates therefore
never touched the deployed model. This module fixes that by exposing one
shared instance.

Behavior:
  * First call to `get_blue_gin()` constructs a GINPredictor that auto-loads
    `checkpoints/blue_gin/model.pt` if present.
  * Subsequent calls return the same instance — gradient updates on it
    are visible to every consumer in the process.
  * `reset_blue_gin()` is provided for tests that need a fresh instance.
"""
from __future__ import annotations
import logging
import threading
from typing import Optional

logger = logging.getLogger("forge.runtime.blue_gin")

_LOCK = threading.Lock()
_INSTANCE = None


def get_blue_gin():
    """Return the shared GINPredictor instance (constructed lazily)."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            from blue_team.gin_predictor import GINPredictor
            _INSTANCE = GINPredictor()
            logger.info(
                "Initialized shared Blue GIN (checkpoint_loaded=%s, path=%s)",
                _INSTANCE.checkpoint_loaded,
                _INSTANCE.checkpoint_path,
            )
    return _INSTANCE


def reset_blue_gin() -> None:
    """Drop the cached instance. Tests only — production code should not call this."""
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None
