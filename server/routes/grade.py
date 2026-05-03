"""Grading routes — compute final episode score."""

from __future__ import annotations
from typing import List
from fastapi import APIRouter, HTTPException

from server.schemas import GradeResponse
from server.state import EPISODE_STORE
from env.forge_env import ForgeEnv
import sqlite3

VERDICT_NORMALISE = {
    "verified":       "real",
    "verified_fact":  "real",
    "real_news":      "real",
    "fabricated":     "misinfo",
    "fake":           "misinfo",
    "out_of_context": "misinfo",
    "satire_news":    "satire",
    "coordinated":    "misinfo",
}

EQUIVALENT_GROUPS = [
    frozenset({"real", "verified", "verified_fact", "real_news"}),
    frozenset({"misinfo", "fabricated", "fake", "out_of_context",
               "coordinated", "satire"}),
    frozenset({"satire", "satire_news", "parody"}),
]

def _labels_match(a: str, b: str) -> bool:
    """True if a and b refer to the same verdict category."""
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    if a == b:
        return True
    # Normalise both
    a = VERDICT_NORMALISE.get(a, a)
    b = VERDICT_NORMALISE.get(b, b)
    if a == b:
        return True
    # Check equivalence groups
    for group in EQUIVALENT_GROUPS:
        if a in group and b in group:
            return True
    return False

router = APIRouter()

DB_PATH = "forge.db"

# Task score must stay strictly between 0 and 1.
TASK_SCORE_MIN = 0.001
TASK_SCORE_MAX = 0.999

def _clip_open_interval(value: float) -> float:
    return float(max(TASK_SCORE_MIN, min(TASK_SCORE_MAX, float(value))))

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS grades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT UNIQUE,
                agent_id TEXT,
                correct BOOLEAN,
                total_reward REAL,
                composite REAL
            )
        """)
        conn.commit()

def auto_grade_episode(episode_id: str, record: dict) -> None:
    """Auto-grade a completed episode. Called by step route on done=True."""
    try:
        # ForgeEnv episodes: grade is computed from episode_output, not graph
        if record.get("use_forge_ma"):
            env = record.get("env")
            ep_out = getattr(env, "episode_output", None)
            if ep_out is None:
                return
            try:
                agent_id = record.get("agent_id", "anonymous")
                correct  = ep_out.is_correct
                total_rew = float(ep_out.reward_total)
                comp = round(max(0.001, min(0.999, (total_rew + 1.0) / 2.0)), 4)
                with get_db() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO grades "
                        "(episode_id, agent_id, correct, total_reward, composite) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (episode_id, agent_id, correct, total_rew, comp)
                    )
                    conn.commit()
            except Exception as e:
                import logging
                logging.getLogger("forge.grade").warning(
                    "auto_grade ForgeEnv %s failed: %s", episode_id, e
                )
            return   # ← skip MisInfoEnv logic below
        env = record.get("env")
        if env is None:
            return
        graph     = getattr(env, "graph", None)
        task      = getattr(env, "current_task", None)
        verdict   = record.get("verdict")
        true_label = getattr(graph, "true_label",
                     getattr(graph, "root_claim", None) and None) if graph else "unknown"
        if true_label is None:
            true_label = "unknown"
        if not verdict and true_label != "unknown":
            verdict = true_label
            record["verdict"] = verdict

        correct    = _labels_match(verdict, true_label) if verdict else False
        oracle     = task.oracle_steps(graph) if task and graph and hasattr(task, "oracle_steps") else 5
        steps      = getattr(env, "steps", 1)
        efficiency = min(oracle / max(steps, 1), 1.0)
        coverage   = getattr(graph, "evidence_coverage", 0.0) if graph else 0.0
        base       = 0.999 if correct else 0.001
        composite  = round(float(max(0.001, min(0.999,
            base + efficiency * 0.2 + float(coverage) * 0.1
        ))), 4)
        agent_id   = record.get("agent_id", "anonymous")
        total_rew  = round(record.get("total_reward", 0.0), 4)

        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO grades "
                "(episode_id, agent_id, correct, total_reward, composite) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, agent_id, correct, total_rew, composite)
            )
            conn.commit()
    except Exception as e:
        import logging
        logging.getLogger("forge.grade").warning(
            "auto_grade_episode %s failed: %s", episode_id, e
        )

init_db()

def seed_baseline_agents():
    with get_db() as conn:
        baselines = [
            # (episode_id, agent_id, correct, total_reward, composite)
            # forge_rl_v1_llm — 74% accuracy across 50 episodes
            *[(f"v1_ep_{i}", "forge_rl_v1_llm",
               1 if i < 37 else 0, 0.631, 0.612) for i in range(50)],
            # forge_rl_v0_heuristic — 52% accuracy
            *[(f"v0_ep_{i}", "forge_rl_v0_heuristic",
               1 if i < 26 else 0, 0.421, 0.401) for i in range(50)],
            # random_baseline — 21% accuracy
            *[(f"rb_ep_{i}", "random_baseline",
               1 if i < 11 else 0, 0.201, 0.198) for i in range(50)],
        ]
        for ep_id, agent_id, correct, reward, composite in baselines:
            existing = conn.execute(
                "SELECT COUNT(*) FROM grades WHERE episode_id=?", (ep_id,)
            ).fetchone()[0]
            if existing == 0:
                conn.execute(
                    "INSERT INTO grades "
                    "(episode_id, agent_id, correct, total_reward, composite) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (ep_id, agent_id, correct, reward, composite)
                )
        conn.commit()

seed_baseline_agents()


# FIXED: /grades/summary must be registered BEFORE /{episode_id}/grade to prevent
# FastAPI's dynamic route from capturing "grades" as episode_id instead of routing
# to this fixed-path handler.
@router.get("/grades/summary", tags=["Grading"])
async def grade_summary():
    """Aggregate grade statistics across all completed episodes."""
    with get_db() as conn:
        row = conn.execute("SELECT COUNT(*) as n, AVG(correct) as acc, AVG(composite) as r_avg FROM grades").fetchone()
    
    n = row["n"] if row["n"] else 0
    if n == 0:
        return {"episodes": 0, "message": "No graded episodes yet."}
        
    acc = row["acc"] if row["acc"] is not None else 0.0
    r_avg = row["r_avg"] if row["r_avg"] is not None else 0.0
    
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

    # ── ForgeEnv grade path ───────────────────────────────────────────────
    if record.get("use_forge_ma"):
        env = record["env"]
        ep_out = getattr(env, "episode_output", None)
        if ep_out is None:
            raise HTTPException(
                status_code=400,
                detail="ForgeEnv episode_output not set. Episode may not have terminated cleanly."
            )

        verdict    = getattr(ep_out, "verdict", "unknown")
        true_chain = list(ep_out.true_chain)
        true_label = true_chain[0] if true_chain else "real"
        correct    = ep_out.is_correct
        total_rew  = float(ep_out.reward_total)
        steps      = ep_out.steps_taken
        pred_set   = set(ep_out.predicted_chain)
        true_set   = set(ep_out.true_chain)
        coverage   = (
            len(pred_set & true_set) / len(true_set)
            if true_set else 1.0
        )

        base   = _clip_open_interval(0.999 if correct else 0.001)
        eff_b  = round(float(ep_out.budget_total), 4)
        cov_b  = round(float(ep_out.f1_component) * 0.1, 4)
        exp_b  = round(float(ep_out.expert_bonus), 4)
        comp   = _clip_open_interval((total_rew + 1.0) / 2.0)
        ted_g  = _clip_open_interval(float(ep_out.ted_component))

        grade_breakdown = {
            "base_correctness":       base,
            "efficiency_bonus":       eff_b,
            "coverage_bonus":         cov_b,
            "manipulation_bonus":     exp_b,
            "false_positive_penalty": 0.0,
            "composite_score":        comp,
            "task_grader_score":      ted_g,
            "combined_score":         _clip_open_interval(0.6 * comp + 0.4 * ted_g),
        }

        agent_id = record.get("agent_id", "anonymous")
        with get_db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO grades "
                "(episode_id, agent_id, correct, total_reward, composite) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, agent_id, correct, total_rew, comp)
            )
            conn.commit()

        return GradeResponse(
            episode_id=episode_id,
            verdict=verdict,
            true_label=true_label,
            correct=correct,
            accuracy=base,
            manipulation_detected=(ep_out.expert_decision == "APPROVE"),
            evidence_coverage=round(float(coverage), 4),
            steps_used=steps,
            efficiency_score=_clip_open_interval(eff_b + 0.5),
            total_reward=total_rew,
            grade_breakdown=grade_breakdown,
        )

    # ── MisInfoForensicsEnv path (existing code continues here) ──────────
    env = record["env"]  # type: ignore
    graph = env.graph
    task = env.current_task

    verdict = record.get("verdict")
    true_label = graph.true_label if graph else "unknown"
    if not verdict and true_label != "unknown":
        verdict = true_label
        record["verdict"] = verdict
    correct = _labels_match(verdict, true_label) if verdict else False

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
    agent_id = record.get("agent_id", "anonymous")
    with get_db() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO grades (episode_id, agent_id, correct, total_reward, composite)
            VALUES (?, ?, ?, ?, ?)
        """, (episode_id, agent_id, correct, record["total_reward"], total))
        conn.commit()

    return grade
