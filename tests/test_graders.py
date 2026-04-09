"""Smoke tests — verify all 8 task graders return valid scores."""
import pytest
import numpy as np
from env.tasks import TASK_REGISTRY

TASKS = list(TASK_REGISTRY.keys())


@pytest.mark.parametrize("task_name", TASKS)
def test_grader_returns_valid_score(task_name):
    task = TASK_REGISTRY[task_name]()
    graph = task.generate(difficulty=1, seed=42)

    # Empty trace — should return min score, not 0.0 or 1.0
    score_empty = task.grade([], graph)
    assert 0.001 <= score_empty <= 0.999, \
        f"{task_name}: empty trace score {score_empty} out of range"
    assert score_empty != 0.0 and score_empty != 1.0, \
        f"{task_name}: score is exactly 0.0 or 1.0"

    # Perfect trace — use oracle tool sequence
    oracle_trace = _build_oracle_trace(task_name, graph)
    score_oracle = task.grade(oracle_trace, graph)
    assert 0.001 <= score_oracle <= 0.999, \
        f"{task_name}: oracle trace score {score_oracle} out of range"

    # Oracle should score higher than empty
    assert score_oracle > score_empty, \
        f"{task_name}: oracle score {score_oracle} not > empty {score_empty}"


def _build_oracle_trace(task_name: str, graph) -> list:
    """Build a trace that uses the key tools and correct verdict."""
    tool_map = {
        "fabricated_stats": ["entity_link", "cross_reference"],
        "out_of_context": ["trace_origin", "temporal_audit"],
        "coordinated_campaign": ["query_source", "network_cluster", "flag_manipulation"],
        "politifact_liar": ["query_source", "cross_reference", "entity_link"],
        "image_forensics": ["trace_origin", "temporal_audit", "entity_link"],
        "sec_fraud": ["query_source", "cross_reference", "entity_link"],
        "verified_fact": ["cross_reference", "entity_link"],
        "satire_news": ["request_context", "cross_reference"],
    }
    tools = tool_map.get(task_name, ["cross_reference"])
    trace = [{"action": t} for t in tools]
    trace.append({"action": f"submit_verdict_{graph.true_label}"})
    return trace
