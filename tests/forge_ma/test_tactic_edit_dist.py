import math
import random
from rewards.tactic_edit_dist import tactic_edit_distance
from env.primitives import PrimitiveType

def test_tactic_edit_dist():
    P1 = PrimitiveType.SOURCE_LAUNDER
    P2 = PrimitiveType.TEMPORAL_SHIFT
    P3 = PrimitiveType.ENTITY_SUBSTITUTE
    P4 = PrimitiveType.QUOTE_FABRICATE
    P5 = PrimitiveType.CONTEXT_STRIP
    P6 = PrimitiveType.CITATION_FORGE
    P7 = PrimitiveType.NETWORK_AMPLIFY
    P8 = PrimitiveType.SATIRE_REFRAME

    prims = [P1, P2, P3, P4, P5, P6, P7, P8]

    # Perfect match combinations (1-4)
    assert math.isclose(tactic_edit_distance([P1], [P1]), 0.999)
    assert math.isclose(tactic_edit_distance([P1, P2], [P1, P2]), 0.999)
    assert math.isclose(tactic_edit_distance([P1, P2, P3], [P1, P2, P3]), 0.999)
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], [P1, P2, P3, P4]), 0.999)

    # Deletions / Insertions (5-8)
    assert math.isclose(tactic_edit_distance([P1, P2], [P1, P2, P3]), 1.0 - 1/3) # One deletion k=3 true -> 0.666...
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], [P1, P2, P3]), 1.0 - 1/4) # One insertion k=4 predicted -> 0.75
    assert math.isclose(tactic_edit_distance([P1, P2, P3], [P1, P2, P3, P4]), 1.0 - 1/4) # One deletion k=3 predicted, k=4 true -> 0.75
    assert math.isclose(tactic_edit_distance([P1], [P1, P2]), 0.5)

    # Substitutions (9-12)
    assert math.isclose(tactic_edit_distance([P1, P5, P3], [P1, P2, P3]), 1.0 - 1/3) # One substitution k=3 -> ~0.667
    assert math.isclose(tactic_edit_distance([P1, P5, P6, P4], [P1, P2, P3, P4]), 0.5) # Two substitutions k=4
    assert math.isclose(tactic_edit_distance([P5, P6, P7, P4], [P1, P2, P3, P4]), 0.25) # Three substitutions k=4
    assert math.isclose(tactic_edit_distance([P1, P2], [P3, P4]), 0.001)

    # Complete mismatch / Empty (13-17)
    assert math.isclose(tactic_edit_distance([P1, P2, P3], [P4, P5, P6]), 0.001)
    assert math.isclose(tactic_edit_distance([], [P1, P2, P3]), 0.001)
    assert math.isclose(tactic_edit_distance([P1, P2, P3], []), 0.001)
    assert math.isclose(tactic_edit_distance([], [P1, P2, P3, P4]), 0.001)
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], []), 0.001)

    # Empty vs Empty (18)
    assert math.isclose(tactic_edit_distance([], []), 0.999)

    # Transposition - standard Lev makes it 1 del + 1 ins = 2 edits or 2 sub (19)
    # [P1, P2, P3] vs [P1, P3, P2] -> 2 edits -> 1 - 2/3 = 1/3
    assert math.isclose(tactic_edit_distance([P1, P2, P3], [P1, P3, P2]), 1.0 - 2/3)

    # Various edge cases (20-29)
    assert math.isclose(tactic_edit_distance([P1, P1], [P1]), 0.5)
    assert math.isclose(tactic_edit_distance([P1], [P1, P1]), 0.5)
    assert math.isclose(tactic_edit_distance([P1, P2, P1], [P1, P1]), 1.0 - 1/3)
    assert math.isclose(tactic_edit_distance([P1, P1, P1, P1], [P2, P2, P2, P2]), 0.001)
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], [P4, P3, P2, P1]), 0.001) # 4 sub or some combo
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], [P1, P2, P4, P3]), 0.5) # 2 edits / max 4
    assert math.isclose(tactic_edit_distance([P1, P2, P3], [P2, P3, P4]), 1.0 - 2/3)   # 2 edits (ins P4, del P1) / max 3 = 1/3
    assert math.isclose(tactic_edit_distance([P1, P2, P3, P4], [P2, P3, P4, P5]), 0.5) # 2 edits / max 4 
    assert math.isclose(tactic_edit_distance([P1, P2], [P2, P1]), 0.001) # 2 edits / max 2 -> 0 -> clipped to 0.001
    assert math.isclose(tactic_edit_distance([P1, P2, P2], [P2, P2, P1]), 1.0 - 2/3) # 2 edits

    # 30. Bounds test
    assert 0.001 <= tactic_edit_distance([P1], [P2]) <= 0.999

    # 31. Random baseline k=3
    random.seed(42)
    s = 0
    for _ in range(1000):
        pred = [random.choice(prims) for _ in range(3)]
        true = [random.choice(prims) for _ in range(3)]
        s += tactic_edit_distance(pred, true)
    avg = s / 1000
    # True expected baseline is ~ 1 - (2.66 / 3) approximately 0.11...
    assert 0.10 <= avg <= 0.15, f"Avg random baseline k=3 should be ~0.11, was {avg}"

