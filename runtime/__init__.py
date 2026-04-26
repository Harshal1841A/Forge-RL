"""
runtime/ — process-wide singletons shared by training, the FastAPI server,
and both Gymnasium environments (ForgeEnv, MisInfoForensicsEnv).

The previous codebase had ForgeEnv (used for training) and MisInfoForensicsEnv
(used by the server's /reset endpoint) holding *separate* GINPredictor
instances. As a result, training updates in ForgeEnv never affected what the
deployed endpoint actually scored. This package exposes a single accessor —
get_blue_gin() — so both code paths share one model instance backed by the
same checkpoint on disk.
"""
from runtime.blue_gin import get_blue_gin, reset_blue_gin

__all__ = ["get_blue_gin", "reset_blue_gin"]
