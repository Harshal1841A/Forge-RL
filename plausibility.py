from typing import Any
import re as _re


def plausibility_score(text: str) -> float:
    """
    Text-only plausibility scorer for reward shaping (no graph object needed).
    Used by hierarchical_reward to compute Δplausibility = before − after.
    Deterministic, zero LM calls, <1ms.
    Returns float in (0.001, 0.999).
    """
    linguistic = _linguistic_coherence(text)
    entity     = _entity_coherence(text)
    # Source and structural default to 0.5 when no graph context available
    score = (linguistic * 0.5 * 0.5 * entity) ** 0.25
    return float(max(0.001, min(0.999, score)))

def compute_plausibility(graph: Any) -> float:
    linguistic = _linguistic_coherence(graph.root_claim.text)
    source     = _source_plausibility(graph.root_claim.trust_score)
    structural = _structural_coherence(graph)
    entity     = _entity_coherence(graph.root_claim.text)
    score = (linguistic * source * structural * entity) ** 0.25
    return float(max(0.001, min(0.999, score)))

def _linguistic_coherence(text: str) -> float:
    words = text.split()
    wc = len(words)
    if wc < 5 or wc > 100:
        wc_score = max(0.001, min(1.0, wc / 5 if wc < 5 else 100 / wc))
    else:
        wc_score = 1.0
    unique_ratio = len(set(words)) / max(wc, 1)
    diversity = unique_ratio / 0.3 if unique_ratio < 0.3 else 1.0
    return (wc_score + diversity) / 2

def _source_plausibility(trust_score: float) -> float:
    if trust_score <= 0.0 or trust_score >= 1.0:
        return 0.0  # Degenerate extremes kill the score
    if 0.05 <= trust_score <= 0.95:
        return 1.0
    return 0.5

def _structural_coherence(graph: Any) -> float:
    n_edges = len(graph.edges)
    n_relations = len(set(e.relation for e in graph.edges))
    if n_edges < 2 or n_edges > 200 or n_relations < 2:
        return 0.001
    return 1.0

def _entity_coherence(text: str) -> float:
    import re
    # Named entity = 2+ capitalized words in sequence
    pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
    return 1.0 if re.search(pattern, text) else 0.3
