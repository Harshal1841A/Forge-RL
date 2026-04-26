"""
server/state.py — Shared mutable state for the server.
Imported by both main.py and route handlers to avoid circular imports.
"""
from typing import Dict

# In-memory episode store: episode_id → {"env", "obs", "agent_id", ...}
# In production: replace with Redis via aioredis
EPISODE_STORE: Dict[str, dict] = {}
