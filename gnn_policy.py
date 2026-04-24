"""
GNN Policy Model — Graph Attention Network over ClaimGraph evidence structure.
Uses PyTorch + torch_geometric (both free/OSS).

Architecture:
  Input:  Flat obs vector (claim_embed + tool_history + scalars)
  Hidden: 3-layer GAT over evidence graph features
  Output: Policy logits (N_ACTIONS) + Value estimate (scalar)

Falls back to MLP if torch_geometric is unavailable.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from env.misinfo_env import N_ACTIONS
import config

logger = logging.getLogger(__name__)

# ─── Try to import torch_geometric, fall back to MLP gracefully ──────────────
try:
    from torch_geometric.nn import GATConv, global_mean_pool  # type: ignore
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    logger.warning("torch_geometric not installed — using MLP fallback policy.")


# ─── MLP fallback (always available) ─────────────────────────────────────────

class MLPPolicy(nn.Module):
    """Lightweight MLP actor-critic for environments without PyG."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self.forward(t)
            if deterministic:
                action = logits.argmax(dim=-1).item()
                log_prob = torch.tensor(0.0)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                action = action_t.item()
        return action, log_prob.item(), value.item()


# ─── GNN Policy (used when torch_geometric is available) ─────────────────────

class GATPolicy(nn.Module):
    """
    Graph Attention Network policy.
    Encodes the ClaimGraph structure via 3 GAT layers, then combines
    pooled graph embedding with the flat obs vector for actor-critic output.
    """

    def __init__(
        self,
        obs_dim: int,
        node_feat_dim: int = 8,        # per-node features: trust, virality, retrieved, etc.
        gnn_hidden: int = config.GNN_HIDDEN_DIM,
        gnn_heads: int = config.GNN_HEADS,
        gnn_layers: int = config.GNN_NUM_LAYERS,
        n_actions: int = N_ACTIONS,
        mlp_hidden: int = 256,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim

        # GAT encoder
        self.gat_layers = nn.ModuleList()
        in_dim = node_feat_dim
        for i in range(gnn_layers):
            out_dim = gnn_hidden if i < gnn_layers - 1 else gnn_hidden
            heads = gnn_heads if i < gnn_layers - 1 else 1
            concat = (i < gnn_layers - 1)
            self.gat_layers.append(
                GATConv(in_dim, out_dim, heads=heads, concat=concat, dropout=0.1)
            )
            in_dim = out_dim * heads if concat else out_dim

        graph_embed_dim = gnn_hidden   # after final GAT + global mean pool

        # Combine graph embedding with flat obs
        combined_dim = graph_embed_dim + obs_dim
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(mlp_hidden, n_actions)
        self.critic = nn.Linear(mlp_hidden, 1)

    def encode_graph(
        self,
        node_features: torch.Tensor,   # [N, node_feat_dim]
        edge_index: torch.Tensor,       # [2, E]
        batch: torch.Tensor,            # [N] — batch assignments
    ) -> torch.Tensor:
        import torch.nn.functional as F
        x = node_features
        for gat in self.gat_layers[:-1]:
            x = F.elu(gat(x, edge_index))
        x = self.gat_layers[-1](x, edge_index)
        return global_mean_pool(x, batch)   # [B, gnn_hidden]

    def forward(
        self,
        obs: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_emb = self.encode_graph(node_features, edge_index, batch)
        combined = torch.cat([graph_emb, obs], dim=-1)
        h = self.shared(combined)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        return logits, value

    def get_action(
        self,
        obs: np.ndarray,
        node_features: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        """Mirrors MLPPolicy.get_action() for API compatibility with PPOAgent.
        When graph data is not supplied (flat-obs mode), falls back to zero-graph embedding.
        """
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0)  # [1, obs_dim]
            n_nodes = 1
            if node_features is None:
                # Fallback: single zero-node graph (no structural info)
                node_features = torch.zeros(n_nodes, self.node_feat_dim)
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                batch = torch.zeros(n_nodes, dtype=torch.long)
            logits, value = self.forward(t, node_features, edge_index, batch)
            if deterministic:
                action = logits.argmax(dim=-1).item()
                log_prob = torch.tensor(0.0)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_t = dist.sample()
                log_prob = dist.log_prob(action_t)
                action = action_t.item()
        return action, log_prob.item(), value.squeeze(-1).item()


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_policy(obs_dim: int, use_gnn: bool = True) -> nn.Module:
    if use_gnn and HAS_PYGEOMETRIC:
        logger.info("Using GATPolicy (torch_geometric available)")
        return GATPolicy(obs_dim=obs_dim)
    logger.info("Using MLPPolicy (fallback)")
    return MLPPolicy(obs_dim=obs_dim, n_actions=N_ACTIONS)
