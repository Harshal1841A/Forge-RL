from env.primitives import FINGERPRINT_KEYS

NODE_FEAT_DIM = 10
_FP_KEYS = list(FINGERPRINT_KEYS.values())


def build_node_features(node, dim: int = NODE_FEAT_DIM) -> list:
    """Build dim-length feature vector from R1 or R2 ClaimNode.

    Layout: [0] trust_score, [1] is_retrieved, [2] injected, [3-9] fingerprint flags.
    """
    if hasattr(node, "is_retrieved"):
        is_retrieved = float(node.is_retrieved)
    elif hasattr(node, "retrieved"):
        is_retrieved = float(node.retrieved)
    else:
        is_retrieved = 0.0

    injected = float(getattr(node, "injected", False))
    trust_score = float(getattr(node, "trust_score", 0.5))

    feat = [trust_score, is_retrieved, injected]

    fp = getattr(node, "fingerprints", {})
    for key in _FP_KEYS:
        feat.append(1.0 if fp.get(key, False) else 0.0)

    feat = feat[:dim]
    feat += [0.0] * max(0, dim - len(feat))
    return feat
