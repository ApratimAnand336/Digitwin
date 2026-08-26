"""
model/graph.py - Graph construction for the assembly line.

Loads the adjacency list from configs/graph.yaml and prepares it for
the PyTorch GCN layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import yaml

_DEFAULT_GRAPH = Path(__file__).parent.parent / "configs" / "graph.yaml"


def load_graph_data(config_path: Path | str | None = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, int]]]:
    """Load the raw nodes and edges from the YAML config."""
    if config_path is None:
        config_path = _DEFAULT_GRAPH
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["nodes"], data["edges"]


def get_edge_index(config_path: Path | str | None = None) -> torch.Tensor:
    """
    Returns a PyTorch tensor of shape (2, num_edges) representing the
    directed edges in the graph. This matches the standard format expected
    by graph neural networks (e.g., PyTorch Geometric).
    
    Index 0 contains the source nodes.
    Index 1 contains the destination nodes.
    """
    _, edges = load_graph_data(config_path)
    
    src = [e["src"] for e in edges]
    dst = [e["dst"] for e in edges]
    
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index
