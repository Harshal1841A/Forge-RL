"""
Layer 4 — Red Team tests.
Covers: HAEModel architecture, ActionValidator gates, RedAgent K_MAX enforcement.
"""
import pytest
import torch
from env.primitives import PrimitiveType, K_MAX
from red_team.hae_model import HAEModel, HIDDEN_DIM, NUM_TOOLS, NUM_PRIMITIVES
from red_team.action_validator import (
    validate_chain, validate_no_consecutive_duplicate,
    validate_disarm_tags, validate_action
)
from red_team.red_agent import RedAgent

# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────
@pytest.fixture
def tiny_graph():
    """Single node, no edges — minimal valid graph for HAE."""
    x = torch.zeros(1, 10)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return x, edge_index


@pytest.fixture
def two_node_graph():
    """Two connected nodes."""
    x = torch.randn(2, 10)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    return x, edge_index


# ─────────────────────────────────────────────
# HAEModel architecture tests
# ─────────────────────────────────────────────
class TestHAEModel:
    def test_hidden_dim_is_32(self):
        assert HIDDEN_DIM == 32, "HAE hidden dim must be 32 (not 64)"

    def test_output_shapes_single_node(self, tiny_graph):
        x, ei = tiny_graph
        model = HAEModel(node_feat_dim=10)
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        assert out["action_logits"].shape == (1, NUM_TOOLS), (
            f"Expected action_logits (1, {NUM_TOOLS}), got {out['action_logits'].shape}")
        assert out["primitive_logits"].shape == (1, NUM_PRIMITIVES)
        assert out["graph_embed"].shape == (1, HIDDEN_DIM)
        assert out["node_embeds"].shape == (1, HIDDEN_DIM)

    def test_output_shapes_two_nodes(self, two_node_graph):
        x, ei = two_node_graph
        model = HAEModel(node_feat_dim=10)
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        # node_embeds should cover all nodes
        assert out["node_embeds"].shape[0] == 2
        assert out["node_embeds"].shape[1] == HIDDEN_DIM

    def test_num_tools_is_13(self):
        assert NUM_TOOLS == 13

    def test_num_primitives_is_8(self):
        assert NUM_PRIMITIVES == 8

    def test_score_action_returns_float(self, tiny_graph):
        x, ei = tiny_graph
        model = HAEModel(node_feat_dim=10)
        score = model.score_action(x, ei, action_idx=0, primitive_idx=0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_no_nan_in_outputs(self, two_node_graph):
        x, ei = two_node_graph
        model = HAEModel(node_feat_dim=10)
        model.eval()
        with torch.no_grad():
            out = model(x, ei)
        for key, tensor in out.items():
            assert not torch.isnan(tensor).any(), f"NaN in {key}"


# ─────────────────────────────────────────────
# ActionValidator tests
# ─────────────────────────────────────────────
class TestActionValidator:
    def test_chain_at_k_max_allowed(self):
        chain = list(PrimitiveType)[:K_MAX]
        assert validate_chain(chain) is True

    def test_chain_over_k_max_rejected(self):
        chain = list(PrimitiveType)[:K_MAX + 1]  # 5 elements
        assert validate_chain(chain) is False

    def test_empty_chain_allowed(self):
        assert validate_chain([]) is True

    def test_consecutive_duplicate_rejected(self):
        chain = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.TEMPORAL_SHIFT]
        assert validate_no_consecutive_duplicate(chain) is False

    def test_non_consecutive_repeat_allowed(self):
        chain = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.SOURCE_LAUNDER,
                 PrimitiveType.TEMPORAL_SHIFT]
        assert validate_no_consecutive_duplicate(chain) is True

    def test_single_element_chain_ok(self):
        chain = [PrimitiveType.QUOTE_FABRICATE]
        assert validate_no_consecutive_duplicate(chain) is True

    def test_t_prefix_valid(self):
        assert validate_disarm_tags(["T0001", "T0022.001"]) is True

    def test_ta_prefix_rejected(self):
        assert validate_disarm_tags(["TA0001"]) is False

    def test_mixed_ta_rejected(self):
        assert validate_disarm_tags(["T0001", "TA0022"]) is False

    def test_malformed_rejected(self):
        assert validate_disarm_tags(["INVALID", "001T"]) is False

    def test_validate_action_all_pass(self):
        chain = [PrimitiveType.TEMPORAL_SHIFT, PrimitiveType.SOURCE_LAUNDER]
        disarm = ["T0046", "T0013"]
        assert validate_action(chain, disarm) is True

    def test_validate_action_chain_too_long(self):
        chain = list(PrimitiveType)  # all 8 — exceeds K_MAX=4
        disarm = ["T0001"]
        assert validate_action(chain, disarm) is False

    def test_validate_action_bad_disarm(self):
        chain = [PrimitiveType.TEMPORAL_SHIFT]
        disarm = ["TA9999"]
        assert validate_action(chain, disarm) is False

    def test_validate_action_never_raises(self):
        # Even with garbage input, must return bool not exception
        result = validate_action(None, None)  # type: ignore
        assert isinstance(result, bool)


# ─────────────────────────────────────────────
# RedAgent tests
# ─────────────────────────────────────────────
class TestRedAgent:
    def test_reset_clears_chain(self):
        agent = RedAgent(mode="greedy")
        agent._current_chain = [PrimitiveType.TEMPORAL_SHIFT]
        agent.reset()
        assert agent.current_chain == []

    def test_propose_action_returns_action(self, tiny_graph):
        x, ei = tiny_graph
        agent = RedAgent(mode="greedy")
        agent.reset()
        action = agent.propose_action(x, ei, budget_remaining=5)
        assert action is not None
        assert action.primitive in list(PrimitiveType)

    def test_chain_grows_by_one_per_step(self, tiny_graph):
        x, ei = tiny_graph
        agent = RedAgent(mode="greedy")
        agent.reset()
        for step in range(K_MAX):
            action = agent.propose_action(x, ei, budget_remaining=K_MAX - step)
            assert action is not None, f"Expected action at step {step}"
        # Chain should now be at K_MAX
        assert len(agent.current_chain) == K_MAX

    def test_no_action_when_budget_zero(self, tiny_graph):
        x, ei = tiny_graph
        agent = RedAgent(mode="greedy")
        agent.reset()
        action = agent.propose_action(x, ei, budget_remaining=0)
        assert action is None

    def test_no_action_when_k_max_reached(self, tiny_graph):
        x, ei = tiny_graph
        agent = RedAgent(mode="greedy")
        agent.reset()
        # Fill up to K_MAX manually
        prims = list(PrimitiveType)
        for i in range(K_MAX):
            agent._current_chain.append(prims[i])
        action = agent.propose_action(x, ei, budget_remaining=10)
        assert action is None

    def test_disarm_ids_are_t_prefix(self, tiny_graph):
        x, ei = tiny_graph
        agent = RedAgent(mode="greedy")
        agent.reset()
        action = agent.propose_action(x, ei, budget_remaining=5)
        if action:
            for did in action.disarm_ids:
                assert did.startswith("T") and not did.startswith("TA"), \
                    f"Bad DISARM id: {did}"

    def test_epsilon_mode_still_valid(self, tiny_graph):
        """Epsilon-greedy random actions must still pass K_MAX gate."""
        x, ei = tiny_graph
        agent = RedAgent(epsilon=1.0, mode="epsilon")  # always random
        agent.reset()
        for _ in range(K_MAX):
            action = agent.propose_action(x, ei, budget_remaining=10)
            if action:
                assert len(agent.current_chain) <= K_MAX
