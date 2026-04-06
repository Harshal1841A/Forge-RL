"""
Potential-based reward shaping for FORGE.
All shaping functions satisfy the policy-invariance theorem (Ng et al., 1999).
"""

from __future__ import annotations
from typing import Optional
import math
from env.claim_graph import ClaimGraph
import config


# ─── Potential Function Φ(s) ─────────────────────────────────────────────────

def compute_potential(graph: ClaimGraph) -> float:
    """
    State potential used for dense shaping:
      Φ(s) = w1·coverage + w2·diversity + w3·contradiction_area - w4·duplicate_waste
    """
    coverage   = graph.evidence_coverage                 # 0-1
    diversity  = min(graph.source_diversity_entropy / math.log2(max(len(graph.nodes), 2)), 1.0)
    contra     = min(graph.contradiction_surface_area / max(len(graph.edges), 1), 1.0)

    phi = (
        config.POTENTIAL_W1 * coverage
      + config.POTENTIAL_W2 * diversity
      + config.POTENTIAL_W3 * contra
    )
    return phi


def shaped_step_reward(
    prev_graph: ClaimGraph,
    curr_graph: ClaimGraph,
    base_reward: float,
    gamma: float = config.PPO_GAMMA,
) -> float:
    """
    r_shaped = r_base + γ·Φ(s') - Φ(s)
    Guarantees no change to optimal policy.
    """
    return base_reward + gamma * compute_potential(curr_graph) - compute_potential(prev_graph)


# ─── Terminal Verdict Reward ──────────────────────────────────────────────────

def verdict_reward(
    predicted_label: str,
    true_label: str,
    predicted_confidence: float,
    steps_used: int,
    max_steps: int,
    manipulation_flagged: bool,
    true_manipulation: bool,
) -> float:
    """
    Composite terminal reward:
      base + calibration_bonus + efficiency_bonus + manipulation_component
    """
    # ── Base ──────────────────────────────────────────────────────────────────
    correct = (predicted_label == true_label)
    base = config.REWARD_CORRECT_VERDICT if correct else config.REWARD_WRONG_VERDICT

    # Introduce partial credit for getting the macro category (misinfo vs real) correct
    # This prevents the grader from returning 0.0 for LLM agents that correctly identify it as fake,
    # but struggle to distinguish between 'satire' and 'fabricated' zero-shot.
    misinfo_categories = {"misinfo", "satire", "out_of_context", "fabricated"}
    
    if not correct:
        if predicted_label in misinfo_categories and true_label in misinfo_categories:
            base = config.REWARD_CORRECT_VERDICT * 0.5  # 50% partial credit
        elif predicted_label == "misinfo" and true_label == "real":
            base += config.REWARD_FALSE_POSITIVE   # additional penalty (negative)

    # ── Calibration bonus (rewards confidence alignment) ──────────────────────
    # Correct + high confidence → bonus; Wrong + high confidence → penalty
    confidence_alignment = predicted_confidence if correct else (1.0 - predicted_confidence)
    calibration_bonus = 0.1 * (confidence_alignment - 0.5)   # range [-0.05, +0.05]

    # ── Efficiency bonus ──────────────────────────────────────────────────────
    # Never penalise beyond the base reward; just reduce reward for wastefulness
    step_ratio = steps_used / max(max_steps, 1)
    efficiency_bonus = 0.1 * (1.0 - step_ratio)   # 0 if used all steps, +0.1 if minimal

    # ── Manipulation component ────────────────────────────────────────────────
    manip_reward = 0.0
    if true_manipulation:
        manip_reward = config.REWARD_MANIPULATION_FLAG if manipulation_flagged else -0.1
    else:
        # Penalise false manipulation flags
        manip_reward = -0.1 if manipulation_flagged else 0.0

    total = base + calibration_bonus + efficiency_bonus + manip_reward
    return round(total, 4)


# ─── Step-level Primitive Rewards ────────────────────────────────────────────

def tool_call_reward(
    tool_name: str,
    new_nodes_discovered: int,
    new_contradictions: int,
    is_duplicate_call: bool,
) -> float:
    """
    Immediate reward signal for a single tool call (before shaping).
    """
    if is_duplicate_call:
        return config.REWARD_DUPLICATE_TOOL_PENALTY

    # Information gain proxy
    info_gain = (new_nodes_discovered * 0.03) + (new_contradictions * 0.05)
    return config.REWARD_STEP_PENALTY + info_gain   # small negative + info gain


# ─── Difficulty-Normalised Efficiency Penalty ─────────────────────────────────

def efficiency_penalty(steps_used: int, difficulty: int) -> float:
    """
    Soft penalty for excessive steps, normalised by task difficulty.
    Harder tasks get less penalty per extra step.
    """
    base_budget = config.BASE_EPISODE_STEPS + difficulty * config.STEP_COMPLEXITY_BONUS
    excess = max(0, steps_used - base_budget)
    return -0.02 * excess / max(difficulty, 1)
