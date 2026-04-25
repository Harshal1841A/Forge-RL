"""
env/node_features.py — CANONICAL node feature encoder.
IMPORT THIS EVERYWHERE. Never build node features inline.
forge_env.py, society_of_thought.py, gin_trainer_ma.py
must all call build_node_features() from here.
"""
from env.primitives import FINGERPRINT_KEYS

NODE_FEAT_DIM = 10

# Fixed insertion-order list — NEVER sort this
_FP_KEYS = list(FINGERPRINT_KEYS.values())
# Insertion order: is_laundered_source, timestamp_shifted, entity_substituted,
# has_fabricated_quote, context_stripped, has_forged_citation,
# is_amplified, satire_reframed


def build_node_features(node, dim: int = NODE_FEAT_DIM) -> list:
    """
    Build 10-dim feature vector from a ClaimNode.
    [0] trust_score
    [1] is_retrieved
    [2] injected
    [3-10] fingerprint flags in FINGERPRINT_KEYS insertion order
    Truncated/padded to dim=10.
    """
    feat = [
        float(node.trust_score),
        1.0 if node.is_retrieved else 0.0,
        1.0 if node.injected else 0.0,
    ]
    for key in _FP_KEYS:
        feat.append(1.0 if node.fingerprints.get(key, False) else 0.0)
    feat = feat[:dim]
    feat += [0.0] * max(0, dim - len(feat))
    return feat
