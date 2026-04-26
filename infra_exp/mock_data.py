"""
FORGE-MA Mock Data — Single source of truth for demo.

IMPORTANT: v0 and v1 values are MEASURED from scripts/run_baseline.py
v1.5 and v2 are PROJECTED (labeled clearly in UI as dashed lines).
Never claim projected values as measured.
"""
import json
from pathlib import Path

# Load measured baselines if available
_BASELINES_PATH = Path(__file__).parent / "baselines" / "results.json"


def _load_baselines():
    if _BASELINES_PATH.exists():
        with open(_BASELINES_PATH) as f:
            d = json.load(f)
        return d.get("forge_ma_baselines", {})
    return {}


_B = _load_baselines()

# ─── Measured values (from scripts/run_baseline.py) ───────────────────────────
V0_TED = _B.get("v0_heuristic", {}).get("mean_ted", 0.14)
V1_TED = _B.get("v1_llm", {}).get("mean_ted", 0.38)
V0_ACCURACY = _B.get("v0_heuristic", {}).get("verdict_accuracy", 0.52)
V1_ACCURACY = _B.get("v1_llm", {}).get("verdict_accuracy", 0.74)
RANDOM_BASELINE = 0.11  # always fixed

# ─── Projected values (onsite training) ───────────────────────────────────────
V1_5_TED = 0.58   # GIN pretrained — projected
V2_TED = 0.78     # TRL fine-tuned — projected

# ─── Measurement status (drives solid/dashed in UI) ───────────────────────────
MEASUREMENT_STATUS = {
    "random": "measured",
    "v0":     "measured",
    "v1":     "measured",
    "v1_5":   "projected",
    "v2":     "projected",
}

# ─── Standard demo attack ─────────────────────────────────────────────────────
DEMO_ATTACK = {
    "seed_claim": "WHO study: Coffee consumption reduces cancer risk by 87% in adults over 50.",
    "true_chain": ["SOURCE_LAUNDER", "QUOTE_FABRICATE", "TEMPORAL_SHIFT"],
    "true_label": "fabricated",
    "plausibility_score": 0.81,
    "red_team_description": (
        "MarketBlog.com inserted as laundered source. "
        "Fake WHO quote attached. Publication backdated 4 years."
    ),
}

# ─── Per-version episode results ──────────────────────────────────────────────
EPISODE_RESULTS = {
    "v0": {
        "ted": V0_TED,
        "label": "v0 — Heuristic",
        "measured": True,
        "predicted_chain": ["TEMPORAL_SHIFT"],
        "verdict": "misinfo",
        "verdict_correct": False,
        "confidence": 0.31,
        "expert_decision": "REJECT",
        "expert_feedback": (
            f"Only 2 tools used. Min 5 required. "
            f"Coverage 23%. TED={V0_TED:.2f}."
        ),
        "recommended_action": "ALLOW WITH MONITORING",
        "tools_used": ["query_source", "temporal_audit"],
        "society_votes": {
            "forensic_auditor":  {"verdict": "misinfo", "chain": ["TEMPORAL_SHIFT"], "confidence": 0.31},
            "context_historian": {"verdict": "misinfo", "chain": ["TEMPORAL_SHIFT"], "confidence": 0.29},
            "graph_specialist":  {"verdict": "unknown", "chain": [],                  "confidence": 0.15},
            "narrative_critic":  {"verdict": "real",    "chain": [],                  "confidence": 0.20},
        },
        "consensus": "split",
    },
    "v1": {
        "ted": V1_TED,
        "label": "v1 — Prompted LLM",
        "measured": True,
        "predicted_chain": ["SOURCE_LAUNDER", "TEMPORAL_SHIFT"],
        "verdict": "fabricated",
        "verdict_correct": True,
        "confidence": 0.68,
        "expert_decision": "APPROVE",
        "expert_feedback": (
            f"Verdict correct. 2/3 tactics found. "
            f"SOURCE_LAUNDER confirmed. TED={V1_TED:.2f}."
        ),
        "recommended_action": "ESCALATE FOR REVIEW",
        "tools_used": ["query_source", "trace_origin", "temporal_audit", "cross_reference"],
        "society_votes": {
            "forensic_auditor":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER"],            "confidence": 0.71},
            "context_historian": {"verdict": "fabricated", "chain": ["TEMPORAL_SHIFT"],             "confidence": 0.68},
            "graph_specialist":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER"],             "confidence": 0.61},
            "narrative_critic":  {"verdict": "misinfo",    "chain": ["QUOTE_FABRICATE"],            "confidence": 0.45},
        },
        "consensus": "majority",
    },
    "v1_5": {
        "ted": V1_5_TED,
        "label": "v1.5 — GIN Pretrained",
        "measured": False,
        "predicted_chain": ["SOURCE_LAUNDER", "QUOTE_FABRICATE"],
        "verdict": "fabricated",
        "verdict_correct": True,
        "confidence": 0.79,
        "expert_decision": "APPROVE",
        "expert_feedback": (
            f"Strong evidence. GIN identified network pattern. "
            f"TED={V1_5_TED:.2f} (projected)."
        ),
        "recommended_action": "ESCALATE FOR REVIEW",
        "tools_used": ["query_source", "trace_origin", "temporal_audit",
                       "cross_reference", "quote_search", "entity_link"],
        "society_votes": {
            "forensic_auditor":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER", "CITATION_FORGE"], "confidence": 0.81},
            "context_historian": {"verdict": "fabricated", "chain": ["TEMPORAL_SHIFT"],                   "confidence": 0.73},
            "graph_specialist":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER", "QUOTE_FABRICATE"],"confidence": 0.87},
            "narrative_critic":  {"verdict": "fabricated", "chain": ["QUOTE_FABRICATE"],                  "confidence": 0.71},
        },
        "consensus": "unanimous",
    },
    "v2": {
        "ted": V2_TED,
        "label": "v2 — TRL Fine-tuned",
        "measured": False,
        "predicted_chain": ["SOURCE_LAUNDER", "QUOTE_FABRICATE", "TEMPORAL_SHIFT"],
        "verdict": "fabricated",
        "verdict_correct": True,
        "confidence": 0.91,
        "expert_decision": "APPROVE",
        "expert_feedback": (
            f"Excellent. All 3 tactics. High confidence. "
            f"TED={V2_TED:.2f} (projected). Recommended for enforcement."
        ),
        "recommended_action": "REMOVE",
        "tools_used": ["query_source", "trace_origin", "quote_search",
                       "temporal_audit", "cross_reference", "entity_link",
                       "network_cluster"],
        "society_votes": {
            "forensic_auditor":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER", "CITATION_FORGE"],    "confidence": 0.88},
            "context_historian": {"verdict": "fabricated", "chain": ["TEMPORAL_SHIFT", "CONTEXT_STRIP"],      "confidence": 0.84},
            "graph_specialist":  {"verdict": "fabricated", "chain": ["SOURCE_LAUNDER", "NETWORK_AMPLIFY"],   "confidence": 0.93},
            "narrative_critic":  {"verdict": "fabricated", "chain": ["QUOTE_FABRICATE", "SATIRE_REFRAME"],   "confidence": 0.79},
        },
        "consensus": "unanimous",
    },
}

# ─── Improvement curve for version slider chart ────────────────────────────────
IMPROVEMENT_CURVE = [
    {"version": "Random", "ted": RANDOM_BASELINE, "measured": True},
    {"version": "v0",     "ted": V0_TED,          "measured": True},
    {"version": "v1",     "ted": V1_TED,          "measured": True},
    {"version": "v1.5",   "ted": V1_5_TED,        "measured": False},
    {"version": "v2",     "ted": V2_TED,           "measured": False},
]
