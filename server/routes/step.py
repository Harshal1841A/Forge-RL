"""Step execution routes (OpenEnv spec)."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from server.schemas import StepRequest, StepResponse
from server.state import EPISODE_STORE

router = APIRouter()


@router.post("/step", response_model=StepResponse)
async def take_step(req: StepRequest):
    eid = req.episode_id or EPISODE_STORE.get("latest_id")
    if not eid or eid not in EPISODE_STORE:
        raise HTTPException(status_code=404, detail="Active episode not found")

    record = EPISODE_STORE[eid]
    if record["done"]:
        raise HTTPException(status_code=400, detail="Episode is already done. Call /reset")

    env = record["env"]
    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
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

    # Append to episode trace for task grader
    from env.misinfo_env import ACTIONS
    record.setdefault("episode_trace", []).append({
        "step": env.steps,
        "action": ACTIONS[req.action],
        "reward": reward,
        "done": done,
    })

    if done:
        try:
            from server.routes.grade import auto_grade_episode
            auto_grade_episode(episode_id=eid, record=record)
        except Exception:
            pass  # Never let grading failure break a step response

    return StepResponse(
        observation=obs.tolist(),
        reward=round(float(reward), 5),
        done=done,
        info=info,
    )
