"""
Red Team HAE (Hierarchical Adversarial Encoder) GNN.
SPEC (PRD v9.0 Section 7.2):
  - 1 GNN layer — distinct from Blue Team 2-layer GIN
  - Aggregation: SUM (GINConv) — NOT MEAN. SUM preserves neighbor
    counts which is critical for NETWORK_AMPLIFY detection.
  - Hidden dim: 32 (not 64 — lighter than Blue Team)
  - No MC Dropout — adversary does not expose uncertainty
  - Output: action_logits (13 tools) + primitive_logits (8 primitives)
  - Warm-start from Blue GIN weights, then fine-tune independently
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GINConv, global_add_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

from env.primitives import K_MAX

HIDDEN_DIM = 32
NUM_PRIMITIVES = 8
NUM_TOOLS = 13


class _FallbackSUMLayer(nn.Module):
    """
    Pure-PyTorch fallback implementing SUM aggregation when PyG unavailable.
    Matches GINConv semantics: h_v = MLP(h_v + sum(h_neighbours))
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        # SUM aggregation: accumulate neighbour features
        agg = torch.zeros_like(x)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            # scatter_add: for each dst node, sum src features
            agg.scatter_add_(0, dst.unsqueeze(1).expand_as(x[src]), x[src])
        # GIN update: MLP(self + sum_neighbours)
        return self.mlp(x + agg)


class HAEModel(nn.Module):
    """
    1-layer Hierarchical Adversarial Encoder using GINConv (SUM aggregation).
    
    Distinct from Blue Team GIN (2 layers, 64-dim, MC Dropout).
    Kept lightweight: 1 layer, 32-dim, no dropout.
    Runs after every Red Team action step for dense step reward.
    
    Outputs action_logits and primitive_logits for Red Team policy.
    """

    def __init__(self, node_feat_dim: int = 10):
        super().__init__()
        self.node_feat_dim = node_feat_dim

        if _HAS_PYG:
            # GINConv with SUM aggregation (aggr='add')
            # MLP inside GINConv: node_feat_dim -> HIDDEN_DIM -> HIDDEN_DIM
            gin_mlp = nn.Sequential(
                nn.Linear(node_feat_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            )
            self.conv = GINConv(gin_mlp, aggr='add')  # 'add' = SUM — not 'mean'
        else:
            self.conv = _FallbackSUMLayer(node_feat_dim, HIDDEN_DIM)

        self.action_head = nn.Linear(HIDDEN_DIM, NUM_TOOLS)
        self.primitive_head = nn.Linear(HIDDEN_DIM, NUM_PRIMITIVES)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            x: Node features [N, node_feat_dim]
            edge_index: Edge indices [2, E]
            batch: Batch vector [N] (defaults to all-zeros = single graph)
        Returns:
            dict with action_logits, primitive_logits, graph_embed, node_embeds
        """
        if x.size(0) == 0:
            zero = torch.zeros(1, HIDDEN_DIM, device=x.device)
            return {
                "action_logits": self.action_head(zero),
                "primitive_logits": self.primitive_head(zero),
                "graph_embed": zero,
                "node_embeds": zero,
            }

        h = self.conv(x, edge_index)   # [N, HIDDEN_DIM]
        h = F.relu(h)

        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        if _HAS_PYG:
            # SUM graph pooling — consistent with node-level SUM aggregation
            graph_embed = global_add_pool(h, batch)  # [B, HIDDEN_DIM]
        else:
            graph_embed = h.sum(dim=0, keepdim=True)  # [1, HIDDEN_DIM]

        return {
            "action_logits":   self.action_head(graph_embed),    # [B, NUM_TOOLS]
            "primitive_logits": self.primitive_head(graph_embed), # [B, NUM_PRIMITIVES]
            "graph_embed":     graph_embed,                       # [B, 32]
            "node_embeds":     h,                                 # [N, 32]
        }

    @torch.no_grad()
    def score_action(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        action_idx: int,
        primitive_idx: int,
    ) -> float:
        """
        Score a (tool, primitive) action pair for Red Team greedy selection.
        Returns scalar in [0, 1] via geometric mean of softmax probabilities.
        """
        self.eval()
        out = self.forward(x, edge_index)
        a_prob = torch.softmax(out["action_logits"], dim=-1)[0, action_idx].item()
        p_prob = torch.softmax(out["primitive_logits"], dim=-1)[0, primitive_idx].item()
        return float(a_prob * p_prob) ** 0.5

    @classmethod
    def warm_start_from_blue(
        cls,
        blue_gin_state_dict: dict,
        node_feat_dim: int = 10,
    ) -> "HAEModel":
        """
        Initialize HAE weights from Blue Team's pretrained GIN (warm start).
        Only copies layers where shapes match exactly.
        Red Team then fine-tunes independently via PPO.
        """
        model = cls(node_feat_dim=node_feat_dim)
        own_state = model.state_dict()
        compatible = {
            k: v for k, v in blue_gin_state_dict.items()
            if k in own_state and own_state[k].shape == v.shape
        }
        if compatible:
            model.load_state_dict(compatible, strict=False)
        return model
