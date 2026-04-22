# Causal Graph Visualization with Filtering

## New Function: `visualize_causal_graph_filtered_to_sink`

**Location:** `utils/utils.py`

This function visualizes causal graphs by showing **only** the nodes and edges that are connected to a sink node (typically the outcome variable Y). It uses backward BFS (Breadth-First Search) to identify all nodes that can reach the sink, then filters the graph accordingly.

---

## Why Filter to Sink-Connected Nodes?

When working with causal graphs, especially in high-dimensional settings, not all features may actually contribute to the outcome. Reasons include:

1. **Causal discovery artifacts**: Some edges may be spurious or the algorithm may include disconnected components
2. **Feature engineering**: Some features might be derived but not directly in causal paths to Y
3. **Interpretability**: Focusing only on Y-connected nodes simplifies the visualization
4. **Debugging**: Quickly identify which features are truly relevant vs. isolated

---

## Function Signature

```python
def visualize_causal_graph_filtered_to_sink(
    adjacency_matrix: np.ndarray,
    feature_names: List[str],
    sink_name: str = 'Y',
    title: str = "Causal Graph (Filtered to Sink-Connected Nodes)",
    figsize=(14, 10),
    highlight_sources: bool = True,
    show_node_layers: bool = True
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adjacency_matrix` | `np.ndarray` | Required | Adjacency matrix (n_features × n_features) **including** sink node Y. Entry [i,j]=1 means feature i causes feature j |
| `feature_names` | `List[str]` | Required | Names of all features **INCLUDING** the sink node (e.g., `['X0', 'X1', ..., 'Y']`). Sink should be last element |
| `sink_name` | `str` | `'Y'` | Name of the sink/target node to filter connections to |
| `title` | `str` | Auto-generated | Title for the plot |
| `figsize` | `tuple` | `(14, 10)` | Figure size (width, height) |
| `highlight_sources` | `bool` | `True` | If True, highlight source nodes (no incoming edges) in green |
| `show_node_layers` | `bool` | `True` | If True, use hierarchical layout showing distance from sources |

---

## Returns

| Return Value | Type | Description |
|--------------|------|-------------|
| `fig` | `matplotlib.figure.Figure` | The figure object for further customization |
| `G_filtered` | `nx.DiGraph` | The filtered NetworkX graph (for analysis) |
| `reachable_nodes` | `set` | Set of node names that are connected to sink |
| `real_source_indices` | `list` | Indices of real source nodes (no parents AND can reach sink) |

---

## Algorithm: Backward BFS with Real Source Identification

The function uses **backward BFS** (same logic as in `ShapleyFlowWrapper`) to identify connected nodes AND real source nodes:

```python
# 1. Find potential source nodes (no incoming edges)
potential_sources = [i for i in range(n) if len(parents[i]) == 0]

# 2. Start backward BFS from sink node (Y)
queue = [sink_node]
visited = {sink_node}
reachable = {sink_node}

# 3. Walk backward through parents
while queue:
    current = queue.pop(0)
    reachable.add(current)
    
    # Add all parents (predecessors)
    for parent in parents[current]:
        if parent not in visited:
            visited.add(parent)
            queue.append(parent)

# 4. Filter sources to only those reachable from sink
real_sources = [s for s in potential_sources if s in reachable]

# 5. Filter graph to only reachable nodes
```

This ensures we identify:
- **All nodes connected to sink** (via backward BFS)
- **Real source nodes** = potential sources that can actually reach the sink

This is more accurate than just highlighting nodes with no incoming edges, because some source nodes might be disconnected from the sink.

---

## Visual Features

The visualization includes:

### Node Colors
- **Green (Light)**: Source nodes (no incoming edges)
- **Blue (Light)**: Intermediate nodes (have both incoming and outgoing edges)
- **Gold**: Sink node (Y)

### Edge Colors
- **Gray**: Edges between features
- **Green (Bold)**: Direct edges to sink node

### Statistics Box
Shows filtering results:
- Number of nodes before/after filtering
- Number of edges before/after filtering
- Number of source nodes identified

### Console Output
Prints detailed summary:
- List of source nodes
- List of intermediate nodes
- List of filtered-out nodes
- Percentage of graph retained

---

## Usage Examples

### Example 1: Basic Usage

```python
import numpy as np
from utils.utils import visualize_causal_graph_filtered_to_sink

# Adjacency matrix (5×5 including Y at index 4)
# X0→X1→Y, X2→Y, X3 (disconnected)
adj = np.array([
    [0, 1, 0, 0, 0],  # X0 → X1
    [0, 0, 0, 0, 1],  # X1 → Y
    [0, 0, 0, 0, 1],  # X2 → Y
    [0, 0, 0, 0, 0],  # X3 (isolated)
    [0, 0, 0, 0, 0],  # Y
])

features = ['X0', 'X1', 'X2', 'X3', 'Y']

fig, G, nodes, real_sources = visualize_causal_graph_filtered_to_sink(
    adjacency_matrix=adj,
    feature_names=features,
    sink_name='Y'
)

plt.show()
# Result: X3 will be filtered out, showing only X0→X1→Y and X2→Y
# real_sources will be [0, 2] (indices of X0 and X2)
```

### Example 2: With Real Causal Discovery Results

```python
from causal_discovery.causal_discovery import run_pc_algorithm
from utils.utils import visualize_causal_graph_filtered_to_sink

# Run causal discovery (returns adjacency for features only)
adjacency_features = run_pc_algorithm(X_train)

# Add Y to the adjacency matrix
n_features = adjacency_features.shape[0]
adj_with_y = np.zeros((n_features + 1, n_features + 1))
adj_with_y[:n_features, :n_features] = adjacency_features

# Add edges from important features to Y
# (based on model feature importance or Shapley values)
important_features = [5, 12, 23, 34]  # Example indices
for idx in important_features:
    adj_with_y[idx, n_features] = 1  # feature → Y

# Create feature names with Y
feature_names = [f'X{i}' for i in range(n_features)] + ['Y']

# Visualize filtered graph
fig, G, nodes, real_sources = visualize_causal_graph_filtered_to_sink(
    adjacency_matrix=adj_with_y,
    feature_names=feature_names,
    title="Discovered Causal Structure (Y-Connected Only)"
)

print(f"Real source nodes (indices): {real_sources}")
print(f"Real source nodes (names): {[feature_names[i] for i in real_sources]}")

plt.savefig('filtered_causal_graph.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Example 3: From Notebook

```python
# In your Jupyter notebook (after loading adjacency_matrix and feature_names)

# Assuming adjacency_matrix is 50×50 (features only)
# and feature_names has 50 elements

n_features = len(feature_names)
adj_with_y = np.zeros((n_features + 1, n_features + 1))
adj_with_y[:n_features, :n_features] = adjacency_matrix

# Use Shapley values to identify Y's parents
wrapper_importance = np.mean(np.abs(wrapper_shap_values), axis=0)
top_10_features = np.argsort(wrapper_importance)[-10:]

for idx in top_10_features:
    adj_with_y[idx, n_features] = 1  # Add edge: feature → Y

features_with_y = feature_names + ['Y']

# Visualize
fig, G, nodes, real_sources = visualize_causal_graph_filtered_to_sink(
    adjacency_matrix=adj_with_y,
    feature_names=features_with_y,
    sink_name='Y',
    title=f"Causal Graph - {dataset_name}",
    figsize=(16, 12)
)

print(f"\nReal source nodes that contribute to Y:")
for idx in real_sources:
    print(f"  - {features_with_y[idx]}")

plt.show()
```

---

## Running the Example Script

A standalone example script is provided: `example_visualize_filtered_graph.py`

```bash
cd /Users/juanrios/Documents/master_thesis
python example_visualize_filtered_graph.py
```

This will:
1. Run 3 different examples (simple, complex, realistic)
2. Save PNG files of each visualization
3. Print detailed statistics for each example

---

## Comparison with Existing Functions

| Function | Purpose | Filtering | Y Handling |
|----------|---------|-----------|------------|
| `visualize_causal_graph` | Show full DAG with confounders | None | Y shown separately |
| `visualize_comparison` | Compare true vs discovered | None | Y shown if provided |
| **`visualize_causal_graph_filtered_to_sink`** | **Show only Y-connected nodes** | **Backward BFS** | **Y integrated in adjacency** |

---

## Integration with ShapleyFlow

This function uses the **same backward BFS logic** as `ShapleyFlowWrapper`:

```python
# In ShapleyFlowWrapper.__init__():
# Filter sources to only those that can reach the sink
reachable_from_sink = set()
queue = [y_index]
visited = {y_index}

while queue:
    node = queue.pop(0)
    reachable_from_sink.add(node)
    for parent in parents[node]:
        if parent not in visited:
            visited.add(parent)
            queue.append(parent)

self.source_nodes = [s for s in potential_sources if s in reachable_from_sink]
```

This ensures consistency between visualization and Shapley Flow computation!

---

## Troubleshooting

### Issue: "feature_names length doesn't match adjacency_matrix"
**Solution**: Make sure your `feature_names` list includes the sink node Y. If your adjacency is 50×50, feature_names should have 51 elements (50 features + Y).

### Issue: "All nodes are filtered out"
**Solution**: Check that:
1. Y is actually in the adjacency matrix (not just feature-to-feature)
2. There are edges pointing TO Y (Y's parents)
3. The sink_name matches exactly (case-sensitive)

### Issue: "Graph looks cluttered"
**Solution**: 
- Increase `figsize` parameter: `figsize=(20, 16)`
- Set `show_node_layers=True` for hierarchical layout
- The function automatically filters disconnected nodes

---

## Performance

- **Time Complexity**: O(V + E) where V = nodes, E = edges (BFS)
- **Space Complexity**: O(V) for visited set
- **Typical Runtime**: < 1 second for graphs with 100+ nodes

---

## Notes

1. **Adjacency Matrix Format**: Binary (0/1), where `adj[i,j]=1` means `i → j`
2. **Sink Position**: Typically the last index in the adjacency matrix
3. **Feature Names**: Must include Y and match adjacency size exactly
4. **Disconnected Components**: Automatically filtered out (won't appear in visualization)
5. **Source Detection**: Nodes with no incoming edges are highlighted in green

---

## See Also

- `ShapleyFlow`: Edge-level Shapley attributions
- `ShapleyFlowWrapper`: Wrapper that uses same backward BFS logic
- `visualize_causal_graph`: Full graph visualization
- Example notebook: `shapflow_data_driven.ipynb` (see bottom cells)
