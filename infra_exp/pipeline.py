"""
pipeline.py — FORGE → FORGE-MA Two-Stage Pipeline Orchestrator.
Master Prompt v9.0 §3.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from env.bridge import convert_episode, BridgeResult
from env.primitives import PrimitiveType

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    r1_verdict:       str
    r1_true_label:    str
    r1_reward:        float
    r1_steps:         int
    r1_coverage:      float
    r1_correct:       bool
    bridge_padded:    bool
    true_chain:       List[str]
    claim_text:       str
    difficulty:       int
    r2_reward:        float
    r2_steps:         int
    r2_verdict:       str
    r2_ted_best:      float
    r2_consensus:     str
    r2_agent_verdicts: Dict[str, str]
    r1_elapsed_s:     float = 0.0
    bridge_elapsed_s: float = 0.0
    r2_elapsed_s:     float = 0.0
    total_elapsed_s:  float = 0.0
    pipeline_mode:    bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def total_correct(self) -> bool:
        return self.r1_correct and (self.r2_verdict == self.r1_true_label)


def run_pipeline_episode(
    r1_env, r1_agent, r2_env, r2_society,
    difficulty: int = 1, seed: int = 0, max_latency_s: float = 45.0,
) -> PipelineResult:
    """
    Full two-stage pipeline: R1 investigation → bridge → R2 adversarial episode.
    Falls back to R2 demo mode if R1 or bridge fails (§12).
    """
    pipeline_start = time.perf_counter()

    # ── Stage 1: R1 ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    r1_verdict, r1_true_label, r1_reward, r1_steps, r1_graph = _run_r1(r1_env, r1_agent, seed)
    r1_elapsed = time.perf_counter() - t0

    # ── Bridge ────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    bridge_result: Optional[BridgeResult] = None
    r2_pipeline_mode = False
    try:
        bridge_result = convert_episode(r1_graph)
        assert len(bridge_result.r2_graph.nodes) > 0
        r2_pipeline_mode = True
    except Exception as exc:
        logger.warning("[pipeline] Bridge failed (%s) — R2 demo mode fallback", exc)
    bridge_elapsed = time.perf_counter() - t0

    # ── Stage 2: R2 ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    r2_reward, r2_steps, r2_verdict, r2_ted, r2_consensus, r2_agents = _run_r2(
        r2_env, r2_society, bridge_result, seed
    )
    r2_elapsed = time.perf_counter() - t0

    total_elapsed = time.perf_counter() - pipeline_start
    if total_elapsed > max_latency_s:
        logger.warning("[pipeline] Latency %.1fs > %.1fs threshold", total_elapsed, max_latency_s)

    true_chain_strs = [p.value for p in bridge_result.true_chain] if bridge_result else []

    return PipelineResult(
        r1_verdict=r1_verdict, r1_true_label=r1_true_label, r1_reward=r1_reward,
        r1_steps=r1_steps, r1_coverage=bridge_result.r1_coverage if bridge_result else 0.0,
        r1_correct=(r1_verdict == r1_true_label),
        bridge_padded=bridge_result.padded if bridge_result else False,
        true_chain=true_chain_strs,
        claim_text=bridge_result.claim_text if bridge_result else "",
        difficulty=bridge_result.difficulty if bridge_result else difficulty,
        r2_reward=r2_reward, r2_steps=r2_steps, r2_verdict=r2_verdict,
        r2_ted_best=r2_ted, r2_consensus=r2_consensus, r2_agent_verdicts=r2_agents,
        r1_elapsed_s=round(r1_elapsed, 3), bridge_elapsed_s=round(bridge_elapsed, 3),
        r2_elapsed_s=round(r2_elapsed, 3), total_elapsed_s=round(total_elapsed, 3),
        pipeline_mode=r2_pipeline_mode,
    )


def run_pipeline_batch(
    r1_env, r1_agent, r2_env, r2_society,
    n_episodes: int = 10, difficulty: int = 1, base_seed: int = 0,
) -> List[PipelineResult]:
    """Run N pipeline episodes; return all PipelineResult objects."""
    results = []
    for i in range(n_episodes):
        try:
            result = run_pipeline_episode(r1_env, r1_agent, r2_env, r2_society,
                                          difficulty=difficulty, seed=base_seed + i)
            results.append(result)
            logger.info("[pipeline/batch] %d/%d — TED=%.3f total=%.1fs",
                        i + 1, n_episodes, result.r2_ted_best, result.total_elapsed_s)
        except Exception as exc:
            logger.error("[pipeline/batch] Episode %d crashed: %s", i + 1, exc)
    if results:
        logger.info("[pipeline/batch] R1_acc=%.2f R2_TED=%.3f R2_rew=%.3f",
                    sum(r.r1_correct for r in results) / len(results),
                    sum(r.r2_ted_best for r in results) / len(results),
                    sum(r.r2_reward for r in results) / len(results))
    return results


# ── Private helpers ────────────────────────────────────────────────────────────

def _run_r1(r1_env, r1_agent, seed: int):
    try:
        obs, info = r1_env.reset(seed=seed)
        done = False; total_reward = 0.0; steps = 0
        verdict = "unknown"; true_label = "unknown"; step_info = {}
        while not done:
            action = r1_agent.act(obs) if hasattr(r1_agent, "act") else r1_agent.predict(obs)[0]
            obs, reward, terminated, truncated, step_info = r1_env.step(action)
            total_reward += reward; steps += 1
            if terminated:
                verdict = step_info.get("verdict", "unknown")
                true_label = step_info.get("true_label", "unknown")
            done = terminated or truncated
        r1_graph = getattr(r1_env, "graph", None)
        return verdict, true_label, total_reward, steps, r1_graph
    except Exception as exc:
        logger.error("[pipeline/R1] %s", exc)
        from env.claim_graph import ClaimGraph as R1CG, ClaimNode as R1CN
        import uuid
        rid = "root-0"
        fb = R1CG(graph_id=str(uuid.uuid4()), root_claim_id=rid,
                  nodes={rid: R1CN(node_id=rid, text="R1 fallback", source_url="",
                                   domain="fallback", trust_score=0.5, retrieved=True)},
                  edges=[], applied_tactics=[], true_label="unknown", difficulty=1)
        return "unknown", "unknown", 0.0, 0, fb


def _run_r2(r2_env, r2_society, bridge_result: Optional[BridgeResult], seed: int):
    try:
        if bridge_result is not None and hasattr(r2_env, "reset_from_r1"):
            import copy
            obs, info = r2_env.reset_from_r1(
                initial_graph=bridge_result.r2_graph,
                true_chain=bridge_result.true_chain,
                claim_text=bridge_result.claim_text, seed=seed,
            )
        else:
            obs, info = r2_env.reset(seed=seed)

        claim_text = obs.get("claim_text", bridge_result.claim_text if bridge_result else "")
        true_chain = bridge_result.true_chain if bridge_result else []

        try:
            soc = r2_society.investigate(claim=claim_text, true_chain=true_chain,
                                         budget=r2_env.budget, claim_graph=r2_env.claim_graph)
            r2_verdict, r2_ted = soc.verdict, soc.ted_best
            r2_consensus, r2_agents = soc.consensus_level, soc.agent_verdicts
        except Exception as soc_exc:
            logger.warning("[pipeline/R2/Society] %s", soc_exc)
            r2_verdict, r2_ted, r2_consensus, r2_agents = "unknown", 0.001, "all_different", {}

        done = False; total_r2_reward = 0.0; steps = 0
        while not done:
            _, reward, terminated, truncated, _ = r2_env.step()
            total_r2_reward += reward; steps += 1
            done = terminated or truncated

        return total_r2_reward, steps, r2_verdict, r2_ted, r2_consensus, r2_agents
    except Exception as exc:
        logger.error("[pipeline/R2] %s", exc)
        return 0.0, 0, "unknown", 0.001, "all_different", {}
