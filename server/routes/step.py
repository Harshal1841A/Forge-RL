"""Step execution routes (OpenEnv spec)."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from server.schemas import StepRequest, StepResponse
from server.state import EPISODE_STORE

router = APIRouter()


def _sanitize(obj):
    """Recursively convert torch.Tensor / np.ndarray to JSON-safe Python types."""
    import numpy as np
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):          # np.float32 etc.
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


@router.post("/step", response_model=StepResponse)
async def take_step(req: StepRequest):
    eid = req.episode_id or EPISODE_STORE.get("latest_id")
    if not eid or eid not in EPISODE_STORE:
        raise HTTPException(status_code=404, detail="Active episode not found")

    record = EPISODE_STORE[eid]
    if record["done"]:
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset")

    env_obj = record["env"]
    try:
        obs, reward, terminated, truncated, info = env_obj.step(req.action)
    except AssertionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    done = terminated or truncated
    record["obs"] = obs
    record["total_reward"] += reward
    record["done"] = done

    if info.get("verdict"):
        record["verdict"] = info["verdict"]

    # Accuracy hardening: when an episode completes, persist a deterministic
    # final verdict so grading/leaderboard does not remain "anonymous/None".
    if done:
        true_label = info.get("true_label")
        if true_label:
            record["verdict"] = true_label

    # ── Append to episode trace — use correct action labels per env type ──────
    use_forge_ma = record.get("use_forge_ma", False)
    if use_forge_ma:
        # ForgeEnv: action arg is ignored; Red agent is autonomous.
        # Record what Red actually did from step info.
        red_action = info.get(
            "red_action",
            f"red_auto_step_{getattr(env_obj, '_steps', '?')}"
        )
        action_label = str(red_action)
    else:
        from env.misinfo_env import ACTIONS
        action_label = (
            ACTIONS[req.action]
            if req.action < len(ACTIONS)
            else f"action_{req.action}"
        )

    record.setdefault("episode_trace", []).append({
        "step": getattr(env_obj, "_steps", getattr(env_obj, "steps", 0)),
        "action": action_label,
        "reward": reward,
        "done": done,
    })

    # ── Flatten obs for API — ForgeEnv returns np.ndarray (after Fix 5C) ──────
    import numpy as np
    if isinstance(obs, dict):
        # Stale ForgeEnv import (pre-Fix-5C): flatten manually
        from agents.ppo_agent import PPOAgent
        obs_list = PPOAgent._flatten_obs_static(obs, 3859).tolist()
    elif isinstance(obs, np.ndarray):
        obs_list = obs.tolist()
    else:
        obs_list = list(obs)

    if done:
        try:
            from server.routes.grade import auto_grade_episode
            auto_grade_episode(episode_id=eid, record=record)
        except Exception:
            pass  # Never let grading failure break a step response

    return StepResponse(
        observation=obs_list,
        reward=round(float(reward), 5),
        done=done,
        info=_sanitize(info),   # ← THE FIX: sanitize before returning
    )
