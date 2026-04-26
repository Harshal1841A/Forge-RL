from typing import Any, Optional
import logging
import re as _re

import numpy as np

logger = logging.getLogger("forge.rewards.plausibility")


# ── Semantic-drift scorer (sentence-transformer cosine similarity) ────────────
# Replaces the previous regex/word-count plausibility heuristic with a real
# semantic signal. Model is lazy-loaded and reused across calls. If the
# sentence-transformer dependency is missing, we fall back to the regex
# heuristic so reward shaping never crashes the training loop.

_SENTENCE_MODEL = None      # lazy-init handle to SentenceTransformer
_SENTENCE_MODEL_TRIED = False


def _get_sentence_model():
    """Lazy-load the sentence-transformer used for semantic drift scoring."""
    global _SENTENCE_MODEL, _SENTENCE_MODEL_TRIED
    if _SENTENCE_MODEL is not None or _SENTENCE_MODEL_TRIED:
        return _SENTENCE_MODEL
    _SENTENCE_MODEL_TRIED = True
    try:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loaded sentence-transformer for plausibility scoring.")
    except Exception as e:
        logger.warning("Sentence-transformer unavailable; falling back to regex heuristic: %s", e)
        _SENTENCE_MODEL = None
    return _SENTENCE_MODEL


def semantic_drift_delta(text_before: str, text_after: str) -> Optional[float]:
    """Compute semantic drift between two claim strings as 1 - cos_sim.

    Returns a value in [0, 2] (0 = identical, 1 = orthogonal, 2 = opposite).
    Returns None if the sentence-transformer is unavailable, signalling the
    caller to fall back to the regex heuristic.
    """
    if not text_before or not text_after:
        return 0.0
    model = _get_sentence_model()
    if model is None:
        return None
    try:
        embs = model.encode([text_before, text_after], normalize_embeddings=True)
        cos = float(np.dot(embs[0], embs[1]))
        return float(max(0.0, 1.0 - cos))
    except Exception as e:
        logger.warning("semantic_drift_delta failed, falling back to regex: %s", e)
        return None


def plausibility_score(text: str) -> float:
    """
    Legacy text-only plausibility scorer (regex / word-count heuristic).

    Retained as a fallback signal when the sentence-transformer used by
    `semantic_drift_delta` is not available. Hierarchical reward shaping
    prefers `semantic_drift_delta(before, after)` because:
      - it gives a real semantic similarity signal
      - it is non-zero when Red perturbs the claim text
      - it is symmetric and bounded

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
