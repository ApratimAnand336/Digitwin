"""
model/network.py - Core Spatio-Temporal Model (LSTM + GCN).

Architecture:
1. Per-station LSTM encoder over recent observed history.
2. Hand-rolled Directed GCN layer to propagate hidden states along the physical line.
3. Multi-horizon output heads:
   - Short-horizon (next 5 steps, point estimate)
   - Long-horizon (next 30 steps, mean + std band)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DirectedGCNLayer(nn.Module):
    """
    A simple directed Graph Convolutional Layer.
    Propagates messages from source nodes to destination nodes.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_msg = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Nodes, Features)
        edge_index: (2, NumEdges)
        """
        B, N, _ = x.shape
        out = self.W_self(x)

        src, dst = edge_index[0], edge_index[1]
        
        # Compute messages from source nodes
        messages = self.W_msg(x[:, src, :])  # (B, NumEdges, out_dim)

        # Aggregate messages at destination nodes
        # Expand dst to match dimensions: (B, NumEdges, out_dim)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(B, -1, out.shape[-1])
        
        # Scatter add messages into the output tensor
        out.scatter_add_(1, dst_expanded, messages)

        return torch.relu(out)


class DigitalTwinModel(nn.Module):
    """
    Spatio-Temporal Model predicting future station states.
    
    If use_gcn=False, it ablates the graph communication, turning into
    independent per-station LSTMs.
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        short_horizon: int = 5,
        long_horizon: int = 30,
        use_gcn: bool = True,
    ) -> None:
        super().__init__()
        self.use_gcn = use_gcn
        
        # LSTM processes sequences (batch_first=True)
        self.lstm = nn.LSTM(in_features, hidden_dim, num_layers=2, batch_first=True)
        
        if self.use_gcn:
            # 2 layers of GCN allows information to flow up to 2 stations downstream
            self.gcn1 = DirectedGCNLayer(hidden_dim, hidden_dim)
            self.gcn2 = DirectedGCNLayer(hidden_dim, hidden_dim)
            
        # We only predict cycle_time_s to demonstrate the operational ripple effect
        self.short_head = nn.Linear(hidden_dim, short_horizon)
        
        # Long horizon outputs mean and log_std for a confidence band
        self.long_mean = nn.Linear(hidden_dim, long_horizon)
        self.long_logstd = nn.Linear(hidden_dim, long_horizon)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (Batch, Nodes, SeqLen, Features)
        edge_index: (2, NumEdges)
        
        Returns:
            short_pred: (Batch, Nodes, short_horizon)
            long_mean:  (Batch, Nodes, long_horizon)
            long_std:   (Batch, Nodes, long_horizon)
        """
        B, N, S, F = x.shape
        
        # 1. Per-station independent LSTM encoding
        x_flat = x.view(B * N, S, F)
        _, (h_n, _) = self.lstm(x_flat)
        # h_n shape: (num_layers, B*N, hidden_dim). Take the last layer.
        h = h_n[-1].view(B, N, -1)
        
        # 2. Graph propagation
        if self.use_gcn:
            h = self.gcn1(h, edge_index)
            h = self.gcn2(h, edge_index)
            
        # 3. Output heads
        short_pred = self.short_head(h)
        long_m = self.long_mean(h)
        # Constrain std to be positive
        long_s = torch.exp(torch.clamp(self.long_logstd(h), min=-5.0, max=2.0))
        
        return short_pred, long_m, long_s
