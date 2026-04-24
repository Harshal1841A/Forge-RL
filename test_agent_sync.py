import time

import numpy as np
import pytest

from blue_team.society_of_thought import SocietyOfThought
from env.misinfo_env import MisInfoForensicsEnv


class DummyAgent:
    def __init__(self, delay=0.0, verdict="real"):
        self.delay = delay
        self.verdict = verdict

    def analyze(self, claim, *args, **kwargs):
        time.sleep(self.delay)
        return {
            "verdict": self.verdict,
            "predicted_chain": [],
            "rationale": "ok",
            "confidence": 0.5,
        }


class DummyGin:
    def predict_chain(self, g):
        return {"ordered_chain": [], "confidence": 0.5, "presence_probs": np.zeros(8)}


def test_parse_observation_length_guard():
    env = MisInfoForensicsEnv(task_names=["fabricated_stats"], difficulty=1)
    obs, _ = env.reset()
    with pytest.raises(ValueError):
        MisInfoForensicsEnv.parse_observation(obs[:-1])


def test_graph_lock_context_manager():
    env = MisInfoForensicsEnv(task_names=["fabricated_stats"], difficulty=1)
    with env.graph_lock():
        assert env.has_graph() in (True, False)


def test_society_timeout_fallback():
    slow = DummyAgent(delay=0.2, verdict="misinfo")
    fast = DummyAgent(delay=0.0, verdict="real")
    gin = DummyGin()
    society = SocietyOfThought(
        auditor=slow,
        historian=fast,
        critic=fast,
        graph_specialist=None,
        gin=gin,
        agent_timeout_sec=0.05,
    )
    society.negotiated_search.generate_vectors = lambda claim, gin_model: None
    result = society.investigate("Test claim", claim_graph=None)
    assert result.agent_verdicts["auditor"] == "unknown"
