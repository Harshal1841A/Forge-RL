"""Grading routes — compute final episode score."""

from __future__ import annotations
from typing import List
from fastapi import APIRouter, HTTPException

from server.schemas import GradeResponse
from server.state import EPISODE_STORE
from env.misinfo_env import MisInfoForensicsEnv

router = APIRouter()

# Task score must stay strictly between 0 and 1.
TASK_SCORE_MIN = 0.001
TASK_SCORE_MAX = 0.999

def _clip_open_interval(value: float) -> float:
    return float(max(TASK_SCORE_MIN, min(TASK_SCORE_MAX, float(value))))

# In-memory grade log for leaderboard (replace with DB in production)
GRADE_LOG: List[dict] = []


# FIXED: /grades/summary must be registered BEFORE /{episode_id}/grade to prevent
# FastAPI's dynamic route from capturing "grades" as episode_id instead of routing
# to this fixed-path handler.
@router.get("/grades/summary", tags=["Grading"])
async def grade_summary():
    """Aggregate grade statistics across all completed episodes."""
    if not GRADE_LOG:
        return {"episodes": 0, "message": "No graded episodes yet."}
    n = len(GRADE_LOG)
    acc = sum(g["correct"] for g in GRADE_LOG) / n
    r_avg = sum(g["total_reward"] for g in GRADE_LOG) / n
    return {
        "total_episodes": n,
        "overall_accuracy": round(acc, 4),
        "mean_reward": round(r_avg, 4),
    }


@router.get("/{episode_id}/grade", response_model=GradeResponse)
async def get_grade(episode_id: str):
    """
    Compute and return the final grade for a completed episode.
    Episode must be done (terminated or truncated).
    """
    if episode_id not in EPISODE_STORE:
        raise HTTPException(status_code=404, detail="Episode not found")

    record = EPISODE_STORE[episode_id]
    if not record["done"]:
        raise HTTPException(
            status_code=400,
            detail="Episode not yet complete. Continue stepping until terminated/truncated."
        )

    env: MisInfoForensicsEnv = record["env"]
    graph = env.graph
    task = env.current_task

    verdict = record.get("verdict")
    true_label = graph.true_label if graph else "unknown"
    correct = (verdict == true_label) if verdict else False

    # Efficiency: oracle steps / actual steps (capped at 1.0)
    oracle_steps = task.oracle_steps(graph) if task and graph else 5
    efficiency = min(oracle_steps / max(env.steps, 1), 1.0)

    # Evidence coverage
    coverage = graph.evidence_coverage if graph else 0.0

    # Grade breakdown
    base_score = TASK_SCORE_MAX if correct else TASK_SCORE_MIN
    efficiency_score = round(efficiency * 0.2, 4)
    coverage_score = round(coverage * 0.1, 4)
    manip_score = 0.1 if (
        env.manipulation_flagged and task and task.has_manipulation(graph)
    ) else 0.0
    fp_penalty = -0.1 if (
        env.manipulation_flagged and task and not task.has_manipulation(graph)
    ) else 0.0

    import numpy as np
    total = round(float(np.clip(
        base_score + efficiency_score + coverage_score + manip_score + fp_penalty,
        TASK_SCORE_MIN, TASK_SCORE_MAX
    )), 4)

    # Run programmatic task grader for additional signal
    episode_trace = record.get("episode_trace", [])
    task_grade = TASK_SCORE_MIN
    if task and graph and episode_trace:
        try:
            task_grade = task.grade(episode_trace, graph)
        except Exception:
            task_grade = total  # fall back to reward-based score
    task_grade = _clip_open_interval(task_grade)

    grade_breakdown = {
        "base_correctness": base_score,
        "efficiency_bonus": efficiency_score,
        "coverage_bonus": coverage_score,
        "manipulation_bonus": manip_score,
        "false_positive_penalty": fp_penalty,
        "composite_score": total,
        "task_grader_score": task_grade,
        "combined_score": round(float(
            np.clip(0.6 * total + 0.4 * task_grade, TASK_SCORE_MIN, TASK_SCORE_MAX)
        ), 4),
    }

    grade = GradeResponse(
        episode_id=episode_id,
        verdict=verdict,
        true_label=true_label,
        correct=correct,
        accuracy=base_score,
        manipulation_detected=env.manipulation_flagged,
        evidence_coverage=round(coverage, 4),
        steps_used=env.steps,
        efficiency_score=round(efficiency, 4),
        total_reward=round(record["total_reward"], 4),
        grade_breakdown=grade_breakdown,
    )

    # Log for leaderboard
    GRADE_LOG.append({
        "episode_id": episode_id,
        "agent_id": record.get("agent_id", "anonymous"),
        "correct": correct,
        "total_reward": record["total_reward"],
        "composite": total,
    })

    return grade
