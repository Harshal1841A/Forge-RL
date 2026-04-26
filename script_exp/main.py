"""
FORGE FastAPI Server — OpenEnv-compatible REST API.

Endpoints:
  POST /episodes              → Create new episode
  POST /episodes/{id}/step    → Take action
  GET  /episodes/{id}/grade   → Get final grade
  GET  /episodes/{id}         → Get episode state
  DELETE /episodes/{id}       → Delete episode
  GET  /actions               → List all valid actions
  GET  /health                → Health check
  GET  /leaderboard           → Agent leaderboard
  GET  /grades/summary        → Aggregate grade summary
"""

from __future__ import annotations
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

import config  # noqa: E402
from server.state import EPISODE_STORE  # noqa: E402
from env.misinfo_env import ACTIONS  # noqa: E402

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("forge.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FORGE server starting — visit /docs for interactive API")
    yield
    logger.info("🛑 FORGE shutting down. Active episodes at close: %d", len(EPISODE_STORE))
    # Explicitly close tool registry DB connections before clearing store
    for record in EPISODE_STORE.values():
        if isinstance(record, dict):
            env = record.get("env")
            if env and hasattr(env, "tool_registry") and hasattr(env.tool_registry, "close"):
                env.tool_registry.close()
    EPISODE_STORE.clear()


def create_app() -> FastAPI:
    app = FastAPI(
        title="FORGE — MisInfo Forensics RL Environment",
        description=(
            "OpenEnv-compatible API for the FORGE misinformation investigation "
            "reinforcement learning environment. All APIs are 100% free to use."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers (imported here, after app is created, no circular dep) ────────
    from server.routes.episode import router as episode_router  # noqa: E402
    from server.routes.step import router as step_router  # noqa: E402
    from server.routes.grade import router as grade_router  # noqa: E402

    app.include_router(episode_router, prefix="", tags=["OpenEnv"])
    app.include_router(step_router, prefix="", tags=["OpenEnv"])
    app.include_router(grade_router, prefix="/episodes", tags=["Grading"])

    # ── Static endpoints ──────────────────────────────────────────────────────
    @app.get("/health", tags=["System"])
    async def health():
        return {
            "status": "ok",
            "env": "forge",
            "version": "1.0.0",
            "openenv_compliant": True,
            "tasks": 9,
            "action_space": 13,
            "observation_shape": 3859,
            "reward_range": [-1.0, 1.0],
        }

    @app.get("/tasks", tags=["System"])
    async def list_tasks():
        from env.tasks import TASK_REGISTRY
        difficulty_map = {
            "fabricated_stats": "easy", "out_of_context": "medium",
            "coordinated_campaign": "hard", "politifact_liar": "medium",
            "image_forensics": "hard", "sec_fraud": "hard",
            "verified_fact": "easy", "satire_news": "medium",
        }
        return {
            "tasks": [
                {
                    "id": name,
                    "difficulty": difficulty_map.get(name, "medium"),
                    "reward_range": [0.001, 0.999],
                }
                for name in TASK_REGISTRY
            ],
            "count": len(TASK_REGISTRY),
        }

    @app.get("/actions", tags=["System"])
    async def list_actions():
        _descriptions = {
            "query_source": "Query primary source (Wikipedia + Google FactCheck — free)",
            "trace_origin": "Trace origin via Wayback Machine + Wikidata (free)",
            "cross_reference": "Cross-check against multiple Wikipedia articles (free)",
            "request_context": "Request context from authoritative sources (free)",
            "entity_link": "Link named entities to Wikidata records (free)",
            "temporal_audit": "Audit timestamps via Wayback CDX API (free)",
            "network_cluster": "Detect bot amplification clusters (local, free)",
            "flag_manipulation": "Flag deliberate manipulation — FREE action (no step cost)",
            "submit_verdict_real": "Submit verdict: REAL",
            "submit_verdict_misinfo": "Submit verdict: MISINFORMATION",
            "submit_verdict_satire": "Submit verdict: SATIRE",
            "submit_verdict_out_of_context": "Submit verdict: OUT OF CONTEXT",
            "submit_verdict_fabricated": "Submit verdict: FABRICATED",
        }
        return {
            "actions": [
                {
                    "index": i,
                    "name": name,
                    "is_verdict": name.startswith("submit_verdict"),
                    "is_free": name == "flag_manipulation",
                    "description": _descriptions.get(name, ""),
                }
                for i, name in enumerate(ACTIONS)
            ]
        }

    @app.get("/leaderboard", tags=["System"])
    async def leaderboard():
        from server.routes.grade import GRADE_LOG
        from collections import defaultdict
        if not GRADE_LOG:
            return {"entries": [], "message": "No completed episodes yet."}
        stats: dict = defaultdict(lambda: {"rewards": [], "correct": [], "episodes": 0})
        for entry in GRADE_LOG:
            aid = entry.get("agent_id", "anonymous")
            stats[aid]["rewards"].append(entry["total_reward"])
            stats[aid]["correct"].append(entry["correct"])
            stats[aid]["episodes"] += 1
        board = [
            {
                "agent_id": aid,
                "accuracy": round(sum(s["correct"]) / len(s["correct"]), 4),
                "mean_reward": round(sum(s["rewards"]) / len(s["rewards"]), 4),
                "episodes_played": s["episodes"],
            }
            for aid, s in stats.items()
        ]
        board.sort(key=lambda x: x["accuracy"], reverse=True)
        return {"entries": board}

    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # ── Pre-warm Sentence Transformer ─────────────────────────────────────────
    # Pre-warm the sentence-transformer so the first investigation request
    # doesn't block the uvicorn worker for 30+ seconds.
    try:
        from sentence_transformers import SentenceTransformer
        from env.misinfo_env import MisInfoForensicsEnv
        if not hasattr(MisInfoForensicsEnv, '_shared_embedder') or \
           MisInfoForensicsEnv._shared_embedder is None:
            MisInfoForensicsEnv._shared_embedder = SentenceTransformer(
                config.HF_EMBEDDING_MODEL
            )
            logger.info("Sentence-transformer pre-warmed successfully.")
        else:
            logger.info("Sentence-transformer already loaded, skipping pre-warm.")
    except Exception as warm_exc:
        logger.warning("Embedder pre-warm failed: %s", warm_exc)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=True,
    )
