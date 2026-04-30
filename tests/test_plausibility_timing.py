import time
import timeit
from collections import namedtuple
from rewards.plausibility import compute_plausibility

# Mock classes to mimic ClaimGraph structure
Claim = namedtuple('Claim', ['text', 'trust_score'])
Edge = namedtuple('Edge', ['relation'])

class MockGraph:
    def __init__(self, root_claim, edges):
        self.root_claim = root_claim
        self.edges = edges

def test_m3_plausibility_timing():
    graph = MockGraph(
        root_claim=Claim(text="According to the WHO report and National Institute of Health, coffee reduces cancer.", trust_score=0.8),
        edges=[Edge(relation='supports'), Edge(relation='contradicts'), Edge(relation='cites')]
    )

    start = time.perf_counter()
    for _ in range(100):
        score = compute_plausibility(graph)
    end = time.perf_counter()
    
    total_ms = (end - start) * 1000
    avg_ms = total_ms / 100
    print(f"Plausibility score: {score}")
    print(f"Total time for 100 calls: {total_ms:.2f} ms")
    print(f"Avg time per call: {avg_ms:.4f} ms")
    assert avg_ms < 1.0, f"Average time per call ({avg_ms:.4f} ms) exceeds 1.0 ms."

if __name__ == "__main__":
    test_m3_plausibility_timing()
