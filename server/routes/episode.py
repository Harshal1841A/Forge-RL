"""Episode creation and state routes (OpenEnv spec)."""

from __future__ import annotations
import logging
import random
from typing import Optional

from fastapi import APIRouter, HTTPException

from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
from server.schemas import ResetRequest, ResetResponse, StateResponse, Observation, Action, Reward
from server.state import EPISODE_STORE

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/observations/schema", tags=["OpenEnv"])
async def observations_schema():
    from server.schemas import Observation
    return Observation.model_json_schema()

@router.get("/actions/schema", tags=["OpenEnv"])
async def actions_schema():
    from server.schemas import Action
    return Action.model_json_schema()

@router.get("/rewards/schema", tags=["OpenEnv"])
async def rewards_schema():
    from server.schemas import Reward
    return Reward.model_json_schema()



@router.post("/reset", response_model=ResetResponse, status_code=200)
async def reset_env(req: Optional[ResetRequest] = None):
    req = req or ResetRequest()
    try:
        task_names = [req.task_name] if req.task_name else None
        env = MisInfoForensicsEnv(
            task_names=task_names,
            difficulty=req.difficulty,
            use_live_tools=req.use_live_tools,
        )
        seed = req.seed or random.randint(0, 2**31)
        obs, info = env.reset(seed=seed)

        # Max-size eviction to prevent memory leak on long-running servers.
        # In production, replace EPISODE_STORE with Redis (see server/state.py).
        MAX_EPISODES = 500
        non_meta_keys = [k for k in list(EPISODE_STORE.keys())
                         if k not in ("latest_id", "latest")]
        while len(non_meta_keys) >= MAX_EPISODES:
            oldest = non_meta_keys.pop(0)
            env_record = EPISODE_STORE.pop(oldest, None)
            # Close DB connection if live tools were used
            if env_record and hasattr(env_record.get("env"), "tool_registry"):
                tr = env_record["env"].tool_registry
                if hasattr(tr, "close"):
                    tr.close()
            logger.debug("Evicted episode %s from store (limit=%d)", oldest, MAX_EPISODES)

        episode_id = info["episode_id"]
        EPISODE_STORE[episode_id] = {
            "env": env,
            "obs": obs,
            "agent_id": req.agent_id or "anonymous",
            "total_reward": 0.0,
            "verdict": None,
            "done": False,
        }

        # In strict single-tenant, we cache the most recent ID instead of strong reference
        EPISODE_STORE["latest_id"] = episode_id

        return ResetResponse(
            episode_id=episode_id,
            observation=obs.tolist(),
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state", response_model=StateResponse)
async def get_state(episode_id: Optional[str] = None):
    eid = episode_id or EPISODE_STORE.get("latest_id")
    if not eid or eid not in EPISODE_STORE:
        raise HTTPException(status_code=404, detail="Active episode not found")

    record = EPISODE_STORE[eid]
    env = record["env"]
    info = env.get_episode_summary()
    if env.graph:
        info["graph_summary"] = env.graph.to_dict()

    graph = env.graph
    budget_remaining = round(1.0 - (env.steps / max(env.max_steps, 1)), 4)
    typed_obs = Observation(
        episode_id=eid,
        vector=record["obs"].tolist(),
        claim_text=graph.root.text if graph and graph.root else "",
        evidence_coverage=round(float(graph.evidence_coverage), 4) if graph else 0.0,
        source_diversity=round(float(graph.source_diversity_entropy), 4) if graph else 0.0,
        contradiction_count=int(graph.contradiction_surface_area) if graph else 0,
        manipulation_flagged=bool(env.manipulation_flagged),
        budget_remaining=budget_remaining,
        steps_used=int(env.steps),
    )

    human_readable = {
        "claim": graph.root.text if graph and graph.root else "no claim",
        "true_label_hidden": True,
        "task": info.get("task_id", "unknown"),
        "step": int(env.steps),
        "max_steps": int(env.max_steps),
        "evidence_coverage_pct": round(float(graph.evidence_coverage) * 100, 1) if graph else 0.0,
        "nodes_discovered": len(graph.nodes) if graph else 0,
        "contradictions": int(graph.contradiction_surface_area) if graph else 0,
        "manipulation_flagged": bool(env.manipulation_flagged),
        "verdict_submitted": record.get("verdict"),
        "actions_available": ACTIONS,
    }
    info["human_readable"] = human_readable

    return StateResponse(
        episode_id=eid,
        observation=record["obs"].tolist(),
        typed_observation=typed_obs,
        done=record["done"],
        total_reward=record["total_reward"],
        info=info,
    )
