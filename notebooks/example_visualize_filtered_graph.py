"""
Example: Visualize Causal Graph Filtered to Sink-Connected Nodes

This script demonstrates how to use the visualize_causal_graph_filtered_to_sink
function to display only the nodes and edges that are actually connected to 
the sink node (Y) in a causal DAG.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.utils import visualize_causal_graph_filtered_to_sink


def example_simple_chain():
    """Example 1: Simple causal chain with some disconnected nodes."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Simple Causal Chain with Disconnected Nodes")
    print("="*70)
    
    # Create a DAG: X0→X1→X2→Y, X3→X4 (disconnected), X5→Y
    # 7 nodes total: X0, X1, X2, X3, X4, X5, Y
    adj = np.array([
        [0, 1, 0, 0, 0, 0, 0],  # X0 → X1
        [0, 0, 1, 0, 0, 0, 0],  # X1 → X2
        [0, 0, 0, 0, 0, 0, 1],  # X2 → Y
        [0, 0, 0, 0, 1, 0, 0],  # X3 → X4 (disconnected!)
        [0, 0, 0, 0, 0, 0, 0],  # X4 (leaf, disconnected)
        [0, 0, 0, 0, 0, 0, 1],  # X5 → Y
        [0, 0, 0, 0, 0, 0, 0],  # Y (sink)
    ])
    
    features = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'Y']
    
    print(f"\nOriginal graph has {len(features)} nodes and {int(np.sum(adj))} edges")
    print("Expected: X3→X4 should be filtered out (not connected to Y)")
    
    fig, G_filtered, reachable_nodes, real_sources = visualize_causal_graph_filtered_to_sink(
        adjacency_matrix=adj,
        feature_names=features,
        sink_name='Y',
        title="Example 1: Simple Chain with Disconnected Component"
    )
    
    plt.savefig('example1_filtered_graph.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'example1_filtered_graph.png'")
    print(f"✓ Real source nodes that reach Y: {len(real_sources)}")
    plt.show()


def example_complex_dag():
    """Example 2: More complex DAG with multiple paths to Y."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Complex DAG with Multiple Paths")
    print("="*70)
    
    # Create a more complex DAG:
    # X0→X2→X5→Y, X1→X3→Y, X2→X4→Y, X6→X7 (disconnected), X8 (isolated)
    # 10 nodes: X0-X8 + Y
    adj = np.zeros((10, 10), dtype=int)
    
    # Connected paths to Y
    adj[0, 2] = 1  # X0 → X2
    adj[1, 3] = 1  # X1 → X3
    adj[2, 4] = 1  # X2 → X4
    adj[2, 5] = 1  # X2 → X5
    adj[3, 9] = 1  # X3 → Y
    adj[4, 9] = 1  # X4 → Y
    adj[5, 9] = 1  # X5 → Y
    
    # Disconnected component
    adj[6, 7] = 1  # X6 → X7 (not connected to Y!)
    
    # X8 is isolated (no edges)
    
    features = [f'X{i}' for i in range(9)] + ['Y']
    
    print(f"\nOriginal graph has {len(features)} nodes and {int(np.sum(adj))} edges")
    print("Expected: X6→X7 and X8 should be filtered out")
    print("Expected: X0, X1, X2, X3, X4, X5, Y should remain")
    
    fig, G_filtered, reachable_nodes, real_sources = visualize_causal_graph_filtered_to_sink(
        adjacency_matrix=adj,
        feature_names=features,
        sink_name='Y',
        title="Example 2: Complex DAG with Multiple Paths to Y",
        figsize=(16, 12)
    )
    
    plt.savefig('example2_filtered_graph.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'example2_filtered_graph.png'")
    print(f"✓ Real source nodes that reach Y: {len(real_sources)}")
    plt.show()


def example_real_world_style():
    """Example 3: Realistic style with many features."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Realistic Feature Set (20 features)")
    print("="*70)
    
    # Simulate a realistic scenario with 20 features + Y
    n_features = 20
    adj = np.zeros((n_features + 1, n_features + 1), dtype=int)
    
    # Create a realistic structure:
    # - 5 source nodes feeding into intermediate nodes
    # - Several paths to Y through intermediate nodes
    # - Some features completely disconnected
    
    # Source nodes: X0, X1, X2, X3, X4
    # Path 1: X0 → X5 → X10 → Y
    adj[0, 5] = 1
    adj[5, 10] = 1
    adj[10, 20] = 1
    
    # Path 2: X1 → X6 → X11 → Y
    adj[1, 6] = 1
    adj[6, 11] = 1
    adj[11, 20] = 1
    
    # Path 3: X2 → X7 → X10 (merges with path 1)
    adj[2, 7] = 1
    adj[7, 10] = 1
    
    # Path 4: X3 → X8 → Y (direct)
    adj[3, 8] = 1
    adj[8, 20] = 1
    
    # Path 5: X4 → X9 → X12 → Y
    adj[4, 9] = 1
    adj[9, 12] = 1
    adj[12, 20] = 1
    
    # Disconnected features: X13-X19 form their own subgraph
    adj[13, 14] = 1
    adj[14, 15] = 1
    adj[16, 17] = 1
    adj[17, 18] = 1
    adj[18, 19] = 1
    
    features = [f'X{i}' for i in range(n_features)] + ['Y']
    
    print(f"\nOriginal graph: {len(features)} nodes, {int(np.sum(adj))} edges")
    print("Expected: X13-X19 should be filtered out (disconnected from Y)")
    
    fig, G_filtered, reachable_nodes, real_sources = visualize_causal_graph_filtered_to_sink(
        adjacency_matrix=adj,
        feature_names=features,
        sink_name='Y',
        title="Example 3: Realistic Feature Set (Only Y-Connected Nodes Shown)",
        figsize=(18, 14),
        highlight_sources=True,
        show_node_layers=True
    )
    
    plt.savefig('example3_filtered_graph.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization to 'example3_filtered_graph.png'")
    print(f"✓ Real source nodes that reach Y: {len(real_sources)}")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CAUSAL GRAPH FILTERING EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the visualize_causal_graph_filtered_to_sink")
    print("function, which uses backward BFS to identify and display only the nodes")
    print("and edges that are actually connected to the sink node (Y).")
    print("\nThis is useful for:")
    print("  • Understanding which features actually influence the outcome")
    print("  • Identifying disconnected/irrelevant features")
    print("  • Simplifying complex causal graphs for interpretation")
    print("  • Debugging causal discovery results")
    
    # Run examples
    example_simple_chain()
    example_complex_dag()
    example_real_world_style()
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
    print("\nVisualization files saved:")
    print("  - example1_filtered_graph.png")
    print("  - example2_filtered_graph.png")
    print("  - example3_filtered_graph.png")
    print("\n")
