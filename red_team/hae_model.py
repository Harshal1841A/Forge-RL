"""
Red Team HAE (Heterogeneous Adversarial Encoder) GNN.
SPEC (Master Prompt §Layer4):
  - 1 GNN layer (not 2) — distinct from Blue Team GIN
  - Aggregation: MEAN (not SUM)
  - Hidden dim: 32 (not 64)
  - No MC Dropout (adversary doesn't expose uncertainty)
  - Output: action_logits for 13 tools + primitive_logits for 8 primitives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv  # MEAN aggregation via SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from env.primitives import K_MAX

HIDDEN_DIM = 32
NUM_PRIMITIVES = 8  # len(PrimitiveType)
NUM_TOOLS = 13


class _FallbackHAELayer(nn.Module):
    """Pure-PyTorch fallback when PyG not available."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Hand-rolled MEAN aggregation."""
        num_nodes = x.size(0)
        agg = torch.zeros(num_nodes, x.size(1), device=x.device)
        count = torch.zeros(num_nodes, 1, device=x.device)

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            agg.index_add_(0, dst, x[src])
            count.index_add_(0, dst, torch.ones(src.size(0), 1, device=x.device))
            mask = count > 0
            agg[mask.squeeze()] = agg[mask.squeeze()] / count[mask]

        out_self = self.lin_self(x)
        out_neigh = self.lin_neigh(agg)
        return F.relu(out_self + out_neigh)


class HAEModel(nn.Module):
    """
    1-layer Heterogeneous Adversarial Encoder.
    Used by Red Team agent to score adversarial edit actions.
    Input node features: raw 10-dim claim fingerprint.
    Outputs:
      - action_logits  : (N, NUM_TOOLS)
      - primitive_logits: (N, NUM_PRIMITIVES)
      - graph_embed    : (1, HIDDEN_DIM) — mean-pooled graph embedding
    """
    def __init__(self, node_feat_dim: int = 10):
        super().__init__()
        self.node_feat_dim = node_feat_dim

        if _HAS_PYG:
            # SAGEConv uses concat=True by default → output dim = hidden_dim
            self.conv = SAGEConv(node_feat_dim, HIDDEN_DIM, aggr="mean")
        else:
            self.conv = _FallbackHAELayer(node_feat_dim, HIDDEN_DIM)

        self.action_head = nn.Linear(HIDDEN_DIM, NUM_TOOLS)
        self.primitive_head = nn.Linear(HIDDEN_DIM, NUM_PRIMITIVES)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor = None):
        # 1-layer convolution
        if _HAS_PYG:
            h = self.conv(x, edge_index)
        else:
            h = self.conv(x, edge_index)
        h = F.relu(h)

        # Graph-level mean pooling
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_embed = global_mean_pool(h, batch)
        else:
            graph_embed = h.mean(dim=0, keepdim=True)

        action_logits = self.action_head(graph_embed)
        primitive_logits = self.primitive_head(graph_embed)

        return {
            "action_logits": action_logits,       # (B, NUM_TOOLS)
            "primitive_logits": primitive_logits,  # (B, NUM_PRIMITIVES)
            "graph_embed": graph_embed,            # (B, 32)
            "node_embeds": h,                      # (N, 32)
        }

    @torch.no_grad()
    def score_action(self, x: torch.Tensor, edge_index: torch.Tensor,
                     action_idx: int, primitive_idx: int) -> float:
        """
        Deterministic single-action score used by RedAgent for greedy selection.
        Returns scalar in [0, 1].
        """
        self.eval()
        out = self.forward(x, edge_index)
        a_prob = torch.softmax(out["action_logits"], dim=-1)[0, action_idx].item()
        p_prob = torch.softmax(out["primitive_logits"], dim=-1)[0, primitive_idx].item()
        return float(a_prob * p_prob) ** 0.5  # geometric mean
