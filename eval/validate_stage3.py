"""
eval/validate_stage3.py - Stage 3 validation: Graph Construction.

Checks:
  1. Load graph from configs/graph.yaml via model.graph.
  2. Verify it is a valid directed sequential chain.
  3. Plot the graph topology to confirm it visually.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model.graph import load_graph_data, get_edge_index

PLOT_DIR = ROOT / "data" / "plots"


def run_validation() -> bool:
    print("\n" + "=" * 65)
    print("STAGE 3 VALIDATION - Graph Construction")
    print("=" * 65)

    print("\n[1/3] Loading graph data ...")
    nodes, edges = load_graph_data()
    edge_index = get_edge_index()
    
    n_nodes = len(nodes)
    n_edges = edge_index.shape[1]
    print(f"  Loaded {n_nodes} nodes and {n_edges} directed edges.")
    print(f"  edge_index shape: {tuple(edge_index.shape)}")

    print("\n[2/3] Validating sequential topology ...")
    # A simple sequential line of N stations should have N-1 edges: (0->1, 1->2, ...)
    all_ok = True
    if n_edges != n_nodes - 1:
        print(f"  [FAIL] Expected {n_nodes - 1} edges, got {n_edges}")
        all_ok = False
        
    for i in range(n_edges):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if src != i or dst != i + 1:
            print(f"  [FAIL] Edge {i} is {src}->{dst}, expected {i}->{i+1}")
            all_ok = False

    if all_ok:
        print("  [OK] Graph is a perfect sequential chain.")

    print("\n[3/3] Plotting graph ...")
    G = nx.DiGraph()
    
    for n in nodes:
        G.add_node(n["id"], name=n["name"])
        
    for i in range(n_edges):
        G.add_edge(edge_index[0, i].item(), edge_index[1, i].item())

    # Create a linear layout
    pos = {node: (node, 0) for node in G.nodes()}
    
    fig, ax = plt.subplots(figsize=(12, 2))
    
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="skyblue", node_size=500)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=15)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")
    
    # Add names as annotations
    for node, (x, y) in pos.items():
        name = G.nodes[node]["name"]
        # Rotate text and put it below the nodes
        ax.text(x, y - 0.2, name, rotation=45, ha="right", va="top", fontsize=8)

    ax.set_ylim(-1.5, 0.5)
    ax.axis("off")
    fig.suptitle("Assembly Line Topology (Directed Sequential Chain)", fontsize=11, y=0.95)
    
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "stage3_graph.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved -> {out_path}")

    print("\n" + "=" * 65)
    if all_ok:
        print("VERDICT: PASS -- Graph topology is correctly structured.")
    else:
        print("VERDICT: STOP -- Graph topology has errors.")
    print("=" * 65 + "\n")
    
    return all_ok

if __name__ == "__main__":
    ok = run_validation()
    sys.exit(0 if ok else 1)
