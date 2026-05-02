"""Episode creation and state routes (OpenEnv spec)."""

from __future__ import annotations
import logging
import random
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException

from env.misinfo_env import MisInfoForensicsEnv, ACTIONS
from server.schemas import ResetRequest, ResetResponse, StateResponse, Observation, Action, Reward
from server.state import EPISODE_STORE

logger = logging.getLogger(__name__)
router = APIRouter()

# FORGE-MA task names — reset to ForgeEnv for these
_FORGE_MA_TASKS = {
    "fabricated_stats", "out_of_context", "coordinated_campaign",
    "politifact_liar", "satire_news", "plandemic",
    "sec_fraud", "verified_fact", "image_forensics",
}

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
        # ── Decide which env to use ───────────────────────────────────────────
        # Use ForgeEnv for FORGE-MA adversarial tasks (or when task_name is
        # unspecified — default behaviour). Fall back to MisInfoForensicsEnv
        # only when a task name is given that is NOT in the FORGE-MA set.
        use_forge_ma = (
            req.task_name is None                     # default: use ForgeEnv
            or req.task_name in _FORGE_MA_TASKS
            or getattr(req, "use_forge_ma", False)
        )

        seed = req.seed or random.randint(0, 2**31)

        if use_forge_ma:
            from env.forge_env import ForgeEnv, ForgeEnvConfig
            env = ForgeEnv(ForgeEnvConfig(budget=10, seed=seed))
            obs, info = env.reset(seed=seed)

            # ForgeEnv.reset() now returns np.ndarray after FIX 5C,
            # but guard against the old dict form in case of stale imports.
            import numpy as np
            if isinstance(obs, dict):
                from agents.ppo_agent import PPOAgent
                obs_arr = PPOAgent._flatten_obs_static(obs, 3859)
                obs_list = obs_arr.tolist()
                claim_text = obs.get("claim_text", "")
            else:
                obs_list = obs.tolist() if hasattr(obs, "tolist") else list(obs)
                claim_text = info.get("claim_text", "")
        else:
            task_names = [req.task_name]
            env = MisInfoForensicsEnv(
                task_names=task_names,
                difficulty=req.difficulty,
                use_live_tools=req.use_live_tools,
            )
            obs, info = env.reset(seed=seed)
            obs_list = obs.tolist()
            claim_text = ""

        # ── Evict old episodes to prevent memory leak ─────────────────────────
        MAX_EPISODES = 500
        non_meta_keys = [k for k in list(EPISODE_STORE.keys())
                         if k not in ("latest_id", "latest")]
        while len(non_meta_keys) >= MAX_EPISODES:
            oldest = non_meta_keys.pop(0)
            env_record = EPISODE_STORE.pop(oldest, None)
            if env_record and hasattr(env_record.get("env"), "tool_registry"):
                tr = env_record["env"].tool_registry
                if hasattr(tr, "close"):
                    tr.close()
            logger.debug("Evicted episode %s from store (limit=%d)", oldest, MAX_EPISODES)

        # ForgeEnv embeds episode_id in info; MisInfoForensicsEnv also does.
        # Fall back to a fresh UUID if neither provides one.
        episode_id = info.get("episode_id") or str(uuid.uuid4())[:8]

        EPISODE_STORE[episode_id] = {
            "env": env,
            "obs": obs,
            "agent_id": req.agent_id or "anonymous",
            "total_reward": 0.0,
            "verdict": None,
            "done": False,
            "claim_text": claim_text,
            "use_forge_ma": use_forge_ma,
        }

        EPISODE_STORE["latest_id"] = episode_id

        return ResetResponse(
            episode_id=episode_id,
            observation=obs_list,
            info=info,
        )
    except Exception as e:
        logger.exception("reset_env failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state", response_model=StateResponse)
async def get_state(episode_id: Optional[str] = None):
    eid = episode_id or EPISODE_STORE.get("latest_id")
    if not eid or eid not in EPISODE_STORE:
        raise HTTPException(status_code=404, detail="Active episode not found")

    record = EPISODE_STORE[eid]
    env = record["env"]

    # ── ForgeEnv state ────────────────────────────────────────────────────────
    if record.get("use_forge_ma"):
        import numpy as np
        raw_obs = record["obs"]
        if isinstance(raw_obs, np.ndarray):
            obs_list = raw_obs.tolist()
        else:
            obs_list = list(raw_obs)

        # Build a minimal typed observation from ForgeEnv internals
        budget_used = getattr(env, "_steps", 0)
        budget_total = getattr(env, "config", None)
        budget_total = budget_total.budget if budget_total else 10
        budget_remaining = round(1.0 - (budget_used / max(budget_total, 1)), 4)
        claim_graph = getattr(env, "_claim_graph", None)

        typed_obs = Observation(
            episode_id=eid,
            vector=obs_list,
            claim_text=record.get("claim_text", ""),
            evidence_coverage=round(float(claim_graph.evidence_coverage), 4) if claim_graph and hasattr(claim_graph, "evidence_coverage") else 0.0,
            source_diversity=round(float(claim_graph.source_diversity_entropy), 4) if claim_graph and hasattr(claim_graph, "source_diversity_entropy") else 0.0,
            contradiction_count=int(claim_graph.contradiction_surface_area) if claim_graph and hasattr(claim_graph, "contradiction_surface_area") else 0,
            manipulation_flagged=False,
            budget_remaining=budget_remaining,
            steps_used=budget_used,
        )
        info = {
            "task": "forge_ma_adversarial",
            "steps": budget_used,
            "budget": budget_total,
        }
        return StateResponse(
            episode_id=eid,
            observation=obs_list,
            typed_observation=typed_obs,
            done=record["done"],
            total_reward=record["total_reward"],
            info=info,
        )

    # ── MisInfoForensicsEnv state (legacy path) ───────────────────────────────
    info = env.get_episode_summary()
    if env.graph:
        info["graph_summary"] = env.graph.to_dict()

    graph = env.graph
    budget_remaining = round(1.0 - (env.steps / max(env.max_steps, 1)), 4)
    typed_obs = Observation(
        episode_id=eid,
        vector=record["obs"].tolist(),
        claim_text=(graph.root_claim.text if hasattr(graph, 'root_claim')
        else graph.root.text if hasattr(graph, 'root') else ""),
        evidence_coverage=round(float(graph.evidence_coverage), 4) if graph else 0.0,
        source_diversity=round(float(graph.source_diversity_entropy), 4) if graph else 0.0,
        contradiction_count=int(graph.contradiction_surface_area) if graph else 0,
        manipulation_flagged=bool(env.manipulation_flagged),
        budget_remaining=budget_remaining,
        steps_used=int(env.steps),
    )

    human_readable = {
        "claim": (graph.root_claim.text if hasattr(graph, 'root_claim')
        else graph.root.text if hasattr(graph, 'root') else ""),
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
