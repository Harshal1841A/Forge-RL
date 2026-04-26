"""
OversightReport — Dual-output report generator for FORGE-RL episode results.
SPEC (Master Prompt §Layer6 + PRD v8.1):
  - generate_oversight_report(): Markdown string for Gradio/Jupyter display
  - generate_stix2_bundle(): STIX2 Bundle JSON (required for Fleet AI prize)
    Contains: attack-pattern (per primitive), threat-actor, campaign objects
    All objects carry x_mitre_id DISARM fields.

No LLM calls — purely formatting.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from env.episode_output import EpisodeOutput
from env.primitives import PrimitiveType

# ── STIX2 support (optional install) ──────────────────────────────────────────
try:
    import stix2
    _HAS_STIX2 = True
except ImportError:
    _HAS_STIX2 = False

# ── DISARM tactic → STIX2 attack-pattern mapping ──────────────────────────────
_DISARM_MAP = {
    "SOURCE_LAUNDER":    {"disarm_id": "T0013.001", "name": "Source Laundering",
                          "phase": "Create Content"},
    "TEMPORAL_SHIFT":    {"disarm_id": "T0046",     "name": "Temporal Context Shift",
                          "phase": "Distort Facts"},
    "ENTITY_SUBSTITUTE": {"disarm_id": "T0075.001", "name": "Entity Substitution",
                          "phase": "Distort Facts"},
    "QUOTE_FABRICATE":   {"disarm_id": "T0006",     "name": "Quote Fabrication",
                          "phase": "Create Content"},
    "CONTEXT_STRIP":     {"disarm_id": "T0019.001", "name": "Context Stripping",
                          "phase": "Distort Facts"},
    "CITATION_FORGE":    {"disarm_id": "T0016",     "name": "Citation Forgery",
                          "phase": "Create Content"},
    "NETWORK_AMPLIFY":   {"disarm_id": "T0049",     "name": "Network Amplification",
                          "phase": "Maximise Exposure"},
    "SATIRE_REFRAME":    {"disarm_id": "T0085.001", "name": "Satire Reframing",
                          "phase": "Create Content"},
}

# ── Markdown helpers ───────────────────────────────────────────────────────────
_VERDICT_ICON = {
    "real":           "✅",
    "misinfo":        "🚨",
    "satire":         "🎭",
    "out_of_context": "⚠️",
    "fabricated":     "🔴",
    "unknown":        "❓",
    "trigger_expert": "🔍",
}

_CONSENSUS_ICON = {
    "unanimous":    "🟢 Unanimous (4/4)",
    "majority_3":   "🟡 Majority (3/4)",
    "split_2_2":    "🟠 Split (2/2)",
    "all_different": "🔴 No consensus",
}


def _chain_str(chain: tuple) -> str:
    if not chain:
        return "_empty_"
    return " → ".join(f"`{p}`" for p in chain)


def _reward_bar(value: float, width: int = 20) -> str:
    filled = int(abs(value) * width)
    filled = min(filled, width)
    bar = "█" * filled + "░" * (width - filled)
    sign = "+" if value >= 0 else "-"
    return f"{sign}[{bar}] {value:+.4f}"


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_oversight_report(episode: EpisodeOutput,
                               claim_text: Optional[str] = None,
                               generation: int = 0) -> str:
    """
    Build a full Markdown oversight report for one episode.
    Returns str: Markdown string suitable for Gradio/Jupyter rendering.
    """
    icon = _VERDICT_ICON.get(episode.verdict.lower(), "❓")
    consensus_label = _CONSENSUS_ICON.get(episode.consensus_level, episode.consensus_level)
    correct_mark = "✅ Chain Match" if episode.is_correct else "❌ Chain Mismatch"

    lines = [
        "# 🔬 FORGE-RL Oversight Report",
        f"**Episode:** `{episode.episode_id}` | "
        f"**Generation:** {generation} | "
        f"**Timestamp:** {episode.timestamp}",
        "",
        "---",
        "",
        "## 🏷️ Verdict",
        f"> {icon} **{episode.verdict.upper()}** — {correct_mark}",
        f"> Chain Accuracy: `{episode.chain_accuracy:.1%}`",
        "",
    ]

    if claim_text:
        lines += [
            "## 📰 Claim",
            f"> {claim_text}",
            "",
        ]

    lines += [
        "## 🔗 Tactic Chain",
        "",
        "| Agent | Chain |",
        "|-------|-------|",
        f"| **Predicted** | {_chain_str(episode.predicted_chain)} |",
        f"| **Ground Truth** | {_chain_str(episode.true_chain)} |",
        "",
    ]

    lines += ["## 🤖 Society of Thought — Agent Verdicts", ""]
    lines += ["| Agent | Verdict |", "|-------|---------|"]
    for agent_name, verdict in episode.agent_verdicts:
        v_icon = _VERDICT_ICON.get(verdict.lower(), "❓")
        lines.append(f"| **{agent_name.capitalize()}** | {v_icon} {verdict} |")
    lines += [""]

    lines += [
        "## 🗳️ Consensus",
        f"**Level:** {consensus_label}  ",
        f"**Bonus:** `{episode.consensus_bonus:+.3f}`",
        "",
    ]

    lines += [
        "## 🎓 Expert Panel",
        f"**Decision:** `{episode.expert_decision}`  ",
        f"**Bonus:** `{episode.expert_bonus:+.3f}`",
        "",
    ]

    lines += [
        "## 🏆 Reward Breakdown",
        "```",
        f"TED   {_reward_bar(episode.ted_component)}",
        f"F1    {_reward_bar(episode.f1_component)}",
        f"PLB   {_reward_bar(episode.plausibility_delta)}",
        f"CON   {_reward_bar(episode.consensus_bonus)}",
        f"EXP   {_reward_bar(episode.expert_bonus)}",
        f"BUD   {_reward_bar(episode.budget_total)}",
        f"      {'─'*36}",
        f"TOT   {_reward_bar(episode.reward_total)}",
        "```",
        "",
    ]

    over_flag = "⚠️ OVER BUDGET" if episode.over_budget else "✅ Within budget"
    lines += [
        "## ⏱️ Budget Audit",
        f"**Steps taken:** {episode.steps_taken} / {episode.budget_limit} ({over_flag})  ",
        f"**Useful tools called:** {episode.useful_tools}",
        "",
        "---",
        "",
        "## 📄 Raw JSON",
        "```json",
        episode.to_json(),
        "```",
        "",
    ]

    return "\n".join(lines)


def generate_stix2_bundle(episode: EpisodeOutput,
                           campaign_name: str = "FORGE-RL Episode",
                           claim_text: Optional[str] = None) -> str:
    """
    Generate a STIX2 Bundle for one episode.

    PRD v8.1 requirement: Bundle must contain —
      • attack-pattern objects per tactic (with x_mitre_id DISARM fields)
      • threat-actor object
      • campaign object linking all attack-patterns

    Returns str: JSON-serialised STIX2 Bundle.
    Falls back to hand-rolled STIX2-compatible JSON if stix2 library absent.
    """
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ep_id = episode.episode_id
    chain = list(episode.predicted_chain)

    if _HAS_STIX2:
        return _generate_stix2_with_library(episode, campaign_name, claim_text, now_iso, chain)
    else:
        return _generate_stix2_manual(episode, campaign_name, claim_text, now_iso, chain, ep_id)


def _generate_stix2_with_library(episode, campaign_name, claim_text, now_iso, chain) -> str:
    """Uses the stix2 library to produce a validated Bundle."""
    attack_patterns = []
    for prim_val in chain:
        meta = _DISARM_MAP.get(prim_val, {
            "disarm_id": "T0000", "name": prim_val, "phase": "Unknown"
        })
        ap = stix2.AttackPattern(
            name=meta["name"],
            description=f"DISARM tactic {meta['disarm_id']} detected in episode.",
            kill_chain_phases=[
                stix2.KillChainPhase(
                    kill_chain_name="DISARM",
                    phase_name=meta["phase"]
                )
            ],
            custom_properties={
                "x_mitre_id": meta["disarm_id"],
                "x_disarm_phase": meta["phase"],
                "x_forge_episode_id": episode.episode_id,
                "x_forge_primitive": prim_val,
            }
        )
        attack_patterns.append(ap)

    threat_actor = stix2.ThreatActor(
        name=f"FORGE-RL Red Agent (Episode {episode.episode_id})",
        description="Adversarial agent producing synthetic misinformation chains.",
        custom_properties={
            "x_forge_verdict": episode.verdict,
            "x_forge_chain_accuracy": episode.chain_accuracy,
        }
    )

    campaign = stix2.Campaign(
        name=campaign_name,
        description=claim_text or "No claim text provided.",
        custom_properties={
            "x_forge_episode_id": episode.episode_id,
            "x_forge_reward_total": episode.reward_total,
            "x_forge_consensus_level": episode.consensus_level,
            "x_forge_steps_taken": episode.steps_taken,
        }
    )

    # Relationships: threat-actor → uses → attack-patterns
    relationships = []
    for ap in attack_patterns:
        rel = stix2.Relationship(
            relationship_type="uses",
            source_ref=threat_actor.id,
            target_ref=ap.id,
        )
        relationships.append(rel)

    # Campaign → attributed-to → threat-actor
    camp_rel = stix2.Relationship(
        relationship_type="attributed-to",
        source_ref=campaign.id,
        target_ref=threat_actor.id,
    )

    bundle_objects = attack_patterns + [threat_actor, campaign, camp_rel] + relationships
    bundle = stix2.Bundle(objects=bundle_objects, allow_custom=True)
    return bundle.serialize(pretty=True)


def _generate_stix2_manual(episode, campaign_name, claim_text, now_iso, chain, ep_id) -> str:
    """
    Hand-rolled STIX2-compatible JSON when stix2 library is not installed.
    Produces a valid STIX2 Bundle schema without the library dependency.
    """
    objects = []

    ap_ids = []
    for prim_val in chain:
        meta = _DISARM_MAP.get(prim_val, {
            "disarm_id": "T0000", "name": prim_val, "phase": "Unknown"
        })
        ap_id = f"attack-pattern--{uuid.uuid4()}"
        ap_ids.append(ap_id)
        objects.append({
            "type": "attack-pattern",
            "spec_version": "2.1",
            "id": ap_id,
            "created": now_iso,
            "modified": now_iso,
            "name": meta["name"],
            "description": f"DISARM tactic {meta['disarm_id']} detected in episode {ep_id}.",
            "kill_chain_phases": [{
                "kill_chain_name": "DISARM",
                "phase_name": meta["phase"]
            }],
            "x_mitre_id": meta["disarm_id"],
            "x_disarm_phase": meta["phase"],
            "x_forge_episode_id": ep_id,
            "x_forge_primitive": prim_val,
        })

    ta_id = f"threat-actor--{uuid.uuid4()}"
    objects.append({
        "type": "threat-actor",
        "spec_version": "2.1",
        "id": ta_id,
        "created": now_iso,
        "modified": now_iso,
        "name": f"FORGE-RL Red Agent (Episode {ep_id})",
        "description": "Adversarial agent producing synthetic misinformation chains.",
        "threat_actor_types": ["activist"],
        "x_forge_verdict": episode.verdict,
        "x_forge_chain_accuracy": episode.chain_accuracy,
    })

    camp_id = f"campaign--{uuid.uuid4()}"
    objects.append({
        "type": "campaign",
        "spec_version": "2.1",
        "id": camp_id,
        "created": now_iso,
        "modified": now_iso,
        "name": campaign_name,
        "description": claim_text or "No claim text provided.",
        "x_forge_episode_id": ep_id,
        "x_forge_reward_total": episode.reward_total,
        "x_forge_consensus_level": episode.consensus_level,
        "x_forge_steps_taken": episode.steps_taken,
    })

    # Relationships
    for ap_id in ap_ids:
        objects.append({
            "type": "relationship",
            "spec_version": "2.1",
            "id": f"relationship--{uuid.uuid4()}",
            "created": now_iso,
            "modified": now_iso,
            "relationship_type": "uses",
            "source_ref": ta_id,
            "target_ref": ap_id,
        })

    objects.append({
        "type": "relationship",
        "spec_version": "2.1",
        "id": f"relationship--{uuid.uuid4()}",
        "created": now_iso,
        "modified": now_iso,
        "relationship_type": "attributed-to",
        "source_ref": camp_id,
        "target_ref": ta_id,
    })

    bundle = {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }
    return json.dumps(bundle, indent=2)
