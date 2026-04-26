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
from pydantic import BaseModel  # noqa: E402

import config  # noqa: E402
from server.state import EPISODE_STORE  # noqa: E402
from env.misinfo_env import ACTIONS  # noqa: E402

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger("forge.server")


class FabricateRequest(BaseModel):
    seed_claim: str
    k_max: int = 4


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
    from server.routes.deepfake import router as deepfake_router  # noqa: E402

    app.include_router(episode_router, prefix="", tags=["OpenEnv"])
    app.include_router(step_router, prefix="", tags=["OpenEnv"])
    app.include_router(grade_router, prefix="/episodes", tags=["Grading"])
    app.include_router(deepfake_router, prefix="", tags=["Deepfake"])

    # ── Static endpoints ──────────────────────────────────────────────────────
    from fastapi.responses import RedirectResponse
    @app.get("/", tags=["System"])
    async def root():
        return RedirectResponse(url="/docs")

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


    @app.post("/fabricate", tags=["System"])
    async def fabricate_claim(request: FabricateRequest):
        """
        Given a seed claim, Red Team applies primitives and returns
        the fabricated graph + true chain for Blue Team to investigate.

        Real-news detection:
          - Claims that appear factual/real (no manipulation signals) return
            an empty true_chain so the UI correctly classifies them as REAL.
          - Claims with manipulation signals get the full Red Team treatment.
        """
        from env.forge_env import ForgeEnv
        import uuid, re

        seed = request.seed_claim.lower()

        # ── Lightweight real-news classifier ────────────────────────────────
        # Heuristics that strongly suggest a legitimate, factual claim:
        REAL_SIGNALS = [
            # Authoritative source domains
            r"\b(reuters|ap news|bbc|cnn|nytimes|guardian|npr|who\.int|cdc\.gov|nasa\.gov|"
            r"fda\.gov|whitehouse\.gov|un\.org|nature\.com|science\.org|pubmed)\b",
            # Factual phrasing patterns
            r"\b(according to|confirmed by|official(ly)?|announced|published|reported by|"
            r"study shows|research (shows|confirms|finds)|data (shows|reveals))\b",
            # Geopolitical/breaking-news verbs that are often factual
            r"\b(closed|opened|signed|approved|elected|deployed|arrested|launched|"
            r"summit|treaty|ceasefire|sanctions)\b",
        ]
        # Manipulation signals — if any present, run Red Team
        FAKE_SIGNALS = [
            r"\b(secret(ly)?|hidden|suppressed|cover.?up|they don'?t want you|"
            r"mainstream media (won'?t|refuses)|banned|censored|leaked documents|"
            r"shocking|you won'?t believe|wake up|sheeple)\b",
            r"\b(100%|proven|undeniable|irrefutable)\b",
            r"(!!!|\?\?\?)",
        ]

        real_score = sum(1 for pat in REAL_SIGNALS if re.search(pat, seed))
        fake_score = sum(1 for pat in FAKE_SIGNALS if re.search(pat, seed))

        # If claim looks real AND has no manipulation signals → classify REAL immediately
        if real_score >= 1 and fake_score == 0:
            episode_id = f"live-{uuid.uuid4().hex[:8]}"
            return {
                "seed_claim": request.seed_claim,
                "fabricated_claim": request.seed_claim,
                "true_chain": [],          # empty chain → frontend verdict = REAL
                "plausibility_score": 0.95,
                "graph_summary": {
                    "node_count": 1,
                    "suspicious_nodes": 0,
                    "steps_run": 0,
                },
                "episode_id": episode_id,
                "red_team_description": "No manipulation primitives detected — claim classified as REAL.",
            }

        # ── Otherwise run the Red Team ───────────────────────────────────────
        env = ForgeEnv()
        obs, info = env.reset()
        env._claim_text = request.seed_claim
        env._claim_text_initial = request.seed_claim
        env._true_chain = []
        env.red_agent.reset()

        steps_run = 0
        while steps_run < min(request.k_max, env.config.budget):
            try:
                obs, reward, terminated, truncated, step_info = env.step()
                steps_run += 1
                if terminated or truncated:
                    break
            except Exception:
                break

        final_chain = [p.value for p in env.red_agent.current_chain]
        graph_nodes = len(env._claim_graph.nodes) if env._claim_graph else 1
        suspicious = sum(1 for n in (env._claim_graph.nodes if env._claim_graph else []) if n.injected)
        episode_id = f"live-{uuid.uuid4().hex[:8]}"

        # ── Run the shared Blue GIN over the resulting claim graph ────────────
        # This replaces the prior pre-scripted animation: the verdict the
        # frontend now displays comes from the actual trained model (or
        # xavier-init if no checkpoint was loaded — either way, the same
        # codepath as the deployed predictor).
        gin_verdict = "unknown"
        gin_confidence = 0.0
        gin_predicted_chain: list = []
        try:
            import torch
            from runtime import get_blue_gin

            x_t, ei_t = env._graph_to_tensors()
            class _G:
                pass
            g = _G()
            g.x = x_t
            g.edge_index = ei_t
            g.batch = torch.zeros(x_t.size(0), dtype=torch.long)

            gin_pred = get_blue_gin().predict_chain(g)
            gin_verdict = gin_pred.get("verdict", "unknown")
            gin_confidence = float(gin_pred.get("confidence", 0.0))
            gin_predicted_chain = [
                p.name if hasattr(p, "name") else str(p)
                for p in gin_pred.get("ordered_chain", [])
            ]
        except Exception:
            # Never let inference failure break the API contract.
            pass

        return {
            "seed_claim": request.seed_claim,
            "fabricated_claim": request.seed_claim,
            "true_chain": final_chain,
            "plausibility_score": round(0.5 + 0.1 * len(final_chain), 2),
            "graph_summary": {
                "node_count": graph_nodes,
                "suspicious_nodes": suspicious,
                "steps_run": steps_run,
            },
            "episode_id": episode_id,
            "red_team_description": (
                f"Agent applied {len(final_chain)} primitives: {', '.join(final_chain)}. "
                f"Graph contains {suspicious} adversarial nodes."
            ),
            "gin_verdict": gin_verdict,
            "gin_confidence": gin_confidence,
            "gin_predicted_chain": gin_predicted_chain,
        }


    @app.get("/leaderboard", tags=["System"])
    async def leaderboard():
        from server.routes.grade import get_db
        with get_db() as conn:
            rows = conn.execute("""
                SELECT 
                    agent_id,
                    AVG(correct) as accuracy,
                    AVG(composite) as mean_reward,
                    COUNT(*) as episodes_played
                FROM grades
                GROUP BY agent_id
                ORDER BY accuracy DESC
            """).fetchall()
            
        if not rows:
            return {"entries": [], "message": "No completed episodes yet."}
            
        board = [
            {
                "agent_id": row["agent_id"],
                "accuracy": round(row["accuracy"], 4),
                "mean_reward": round(row["mean_reward"], 4),
                "episodes_played": row["episodes_played"]
            }
            for row in rows
        ]
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

    # ── Pre-warm Deepfake Detector ────────────────────────────────────────────
    # Loads EfficientNet-B4 weights once at startup. Missing weights or missing
    # deps (torchvision/timm/facenet-pytorch) are non-fatal; the route will
    # return 503 instead of crashing the app.
    try:
        from server.ml.deepfake_inference import init_detector
        det = init_detector()
        if det is not None and det.ready:
            logger.info("Deepfake detector pre-warmed on %s.", det.device)
        else:
            logger.info("Deepfake detector unavailable (missing weights or deps); endpoint will 503.")
    except Exception as df_exc:
        logger.warning("Deepfake pre-warm failed: %s", df_exc)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=True,
    )
