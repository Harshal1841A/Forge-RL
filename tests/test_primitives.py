from env.primitives import PrimitiveType, FINGERPRINT_KEYS, K_MAX

def test_primitives_enum():
    assert len(PrimitiveType) == 8
    names = [p.name for p in PrimitiveType]
    expected = [
        "SOURCE_LAUNDER", "TEMPORAL_SHIFT", "ENTITY_SUBSTITUTE", 
        "QUOTE_FABRICATE", "CONTEXT_STRIP", "CITATION_FORGE", 
        "NETWORK_AMPLIFY", "SATIRE_REFRAME"
    ]
    for e in expected:
        assert e in names

def test_k_max():
    assert K_MAX == 4, "k_max must be exactly 4 per specification"

def test_fingerprints():
    assert len(FINGERPRINT_KEYS) == 8
    for p in PrimitiveType:
        assert p in FINGERPRINT_KEYS

def test_primitive_values():
    assert PrimitiveType.SOURCE_LAUNDER.value == "SOURCE_LAUNDER"
