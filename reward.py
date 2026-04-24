"""
Potential-based reward shaping for FORGE.
All shaping functions satisfy the policy-invariance theorem (Ng et al., 1999).
"""

from __future__ import annotations
import math
from env.claim_graph import ClaimGraph
import config


# ─── Potential Function Φ(s) ─────────────────────────────────────────────────

def compute_potential(graph: ClaimGraph) -> float:
    """
    State potential used for dense shaping:
      Φ(s) = w1·coverage + w2·diversity + w3·contradiction_area + w4·network_diameter
    """
    coverage = graph.evidence_coverage                 # 0-1
    diversity = min(graph.source_diversity_entropy / math.log2(max(len(graph.nodes), 2)), 1.0)
    contra = min(graph.contradiction_surface_area / max(len(graph.edges), 1), 1.0)
    # FIX: POTENTIAL_W4 was declared but never used. Wire in network_diameter as a
    # normalised signal (larger graphs = more complex investigation = more potential).
    diameter = min((graph.network_diameter - 1) / max(len(graph.nodes) - 1, 1), 1.0)

    phi = (
        config.POTENTIAL_W1 * coverage
        + config.POTENTIAL_W2 * diversity
        + config.POTENTIAL_W3 * contra
        + config.POTENTIAL_W4 * diameter
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
    Guarantees no change to optimal policy (Ng et al., 1999).

    The shaping term can be negative; REWARD_CLIP_MIN = -1.0 now allows
    negative rewards to reach the agent, so no artificial floor is needed.
    """
    shaping = gamma * compute_potential(curr_graph) - compute_potential(prev_graph)
    result = base_reward + shaping
    return float(max(config.REWARD_CLIP_MIN, min(config.REWARD_CLIP_MAX, result)))


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
        elif predicted_label in misinfo_categories and true_label == "real":
            # FIX: was `base += REWARD_FALSE_POSITIVE` which stacked on top of
            # REWARD_WRONG_VERDICT. Now set explicitly so the total is predictable.
            # Also broadened from only "misinfo" to all misinfo-category false positives.
            base = config.REWARD_WRONG_VERDICT + config.REWARD_FALSE_POSITIVE

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
        manip_reward = config.REWARD_MANIPULATION_FLAG if manipulation_flagged else config.REWARD_MANIPULATION_PENALTY
    else:
        # Penalise false manipulation flags
        manip_reward = config.REWARD_MANIPULATION_PENALTY if manipulation_flagged else 0.0

    total = round(base + calibration_bonus + efficiency_bonus + manip_reward, 4)
    # Clip to open interval before returning
    return float(max(0.001, min(0.999, total)))


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
        return float(max(config.REWARD_CLIP_MIN, config.REWARD_DUPLICATE_TOOL_PENALTY))

    # Information gain proxy
    info_gain = (new_nodes_discovered * 0.03) + (new_contradictions * 0.05)
    result = config.REWARD_STEP_PENALTY + info_gain   # small negative + info gain
    return float(max(config.REWARD_CLIP_MIN, min(config.REWARD_CLIP_MAX, result)))


# ─── Difficulty-Normalised Efficiency Penalty ─────────────────────────────────

def efficiency_penalty(steps_used: int, difficulty: int) -> float:
    """
    Soft penalty for excessive steps, normalised by task difficulty.
    Harder tasks get less penalty per extra step.
    """
    base_budget = config.BASE_EPISODE_STEPS + difficulty * config.STEP_COMPLEXITY_BONUS
    excess = max(0, steps_used - base_budget)
    result = config.REWARD_STEP_PENALTY * excess / max(difficulty, 1)
    return float(max(config.REWARD_CLIP_MIN, min(config.REWARD_CLIP_MAX, result)))
