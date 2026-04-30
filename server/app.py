# server/app.py
# Canonical entry point for openenv.yaml: app: server.app:app
# Re-exports the real app from server.main where all routes live.
from server.main import app  # noqa: F401
__all__ = ["app"]
