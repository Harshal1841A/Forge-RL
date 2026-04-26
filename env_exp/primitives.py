from enum import Enum

class PrimitiveType(str, Enum):
    SOURCE_LAUNDER   = "SOURCE_LAUNDER"    # DISARM T0013.001
    TEMPORAL_SHIFT   = "TEMPORAL_SHIFT"    # DISARM T0046
    ENTITY_SUBSTITUTE = "ENTITY_SUBSTITUTE" # DISARM T0075.001
    QUOTE_FABRICATE  = "QUOTE_FABRICATE"   # DISARM T0006
    CONTEXT_STRIP    = "CONTEXT_STRIP"     # DISARM T0019.001
    CITATION_FORGE   = "CITATION_FORGE"    # DISARM T0016
    NETWORK_AMPLIFY  = "NETWORK_AMPLIFY"   # DISARM T0049
    SATIRE_REFRAME   = "SATIRE_REFRAME"    # DISARM T0085.001

# Structural fingerprint keys — used by claim_graph.py and reward functions
FINGERPRINT_KEYS = {
    PrimitiveType.SOURCE_LAUNDER:    "is_laundered_source",
    PrimitiveType.TEMPORAL_SHIFT:    "timestamp_shifted",
    PrimitiveType.ENTITY_SUBSTITUTE: "entity_substituted",
    PrimitiveType.QUOTE_FABRICATE:   "has_fabricated_quote",
    PrimitiveType.CONTEXT_STRIP:     "context_stripped",
    PrimitiveType.CITATION_FORGE:    "has_forged_citation",
    PrimitiveType.NETWORK_AMPLIFY:   "is_amplified",
    PrimitiveType.SATIRE_REFRAME:    "satire_reframed",
}

K_MAX = 4  # Maximum chain length. This is a hard constant. Do not change.
