# Shapley Methods: Complete Documentation & Pseudocode

## Overview

This document provides comprehensive documentation of all Shapley value implementations, including:
- What each method does based on actual code implementation
- How coalitions and permutations work
- How the causal diagram is used in the game theory
- Where and how causal graph filtering occurs
- Complete pseudocode for each algorithm

---

## Table of Contents

1. [Vanilla Shapley (ShapleyFromScratch)](#1-vanilla-shapley)
2. [Asymmetric Shapley (AsymmetricShapley)](#2-asymmetric-shapley)
3. [Causal Shapley (CausalShapley)](#3-causal-shapley)
4. [Shapley Flow (ShapleyFlow + Wrapper)](#4-shapley-flow)
5. [Graph Explainer Wrapper](#5-graph-explainer-wrapper)
6. [Comparison Summary](#6-comparison-summary)

---

## 1. Vanilla Shapley (ShapleyFromScratch)

### What It Does

Computes **standard Shapley values** without any causal structure. Treats all features symmetrically, distributing credit based purely on marginal contributions across all possible coalitions.

### Game Theory Foundation

**Shapley Value Definition:**
```
φᵢ(f) = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v(S∪{i}) - v(S)]
```

Where:
- `N`: Set of all features
- `S`: A coalition (subset of features not including feature i)
- `v(S)`: Coalition value = E[f(X) | X_S = x_S] (expected model output when only S is observed)
- Weight term: `|S|!(|N|-|S|-1)! / |N|!` ensures fair credit distribution
- `v(S∪{i}) - v(S)`: Marginal contribution of feature i to coalition S

**Key Properties:**
- **Efficiency**: Σᵢ φᵢ = f(x) - f(baseline)
- **Symmetry**: If i and j contribute equally to all coalitions, φᵢ = φⱼ
- **Dummy**: If i never changes predictions, φᵢ = 0
- **Additivity**: For f = g + h, φᵢ^f = φᵢ^g + φᵢ^h

### Coalition & Permutation Mechanics

**Coalition Value Computation:**
```
v(S) = Average over background data:
    - Features IN S: Use instance's actual values (foreground)
    - Features NOT in S: Use background data values
    - Predict and average
```

**Monte Carlo Approximation:**
Instead of enumerating all 2^n coalitions, sample random permutations:
```
φᵢ = E_π [v(S_π^i ∪ {i}) - v(S_π^i)]
```
where `S_π^i` = features appearing before i in random permutation π

**Why permutations work:**
- Each permutation defines n different coalitions (one for each position)
- Averaging over random permutations approximates the exact Shapley formula
- Converges at O(1/√n_samples)

### Causal Diagram Usage

**NONE** - This method ignores causal structure entirely. All features are treated symmetrically.

### Pseudocode

```python
ALGORITHM: Vanilla Shapley (Monte Carlo)

INPUT:
    - instance x (features to explain)
    - model f (trained predictor)
    - background_data D (reference distribution)
    - n_samples (number of permutations)

OUTPUT:
    - φ ∈ R^n (Shapley values for n features)

INITIALIZE:
    φ ← zeros(n)
    baseline ← mean(f(D))

FOR trial = 1 to n_samples:
    # Sample random permutation
    π ← random_permutation([1, 2, ..., n])
    
    # Initialize incremental tracking
    S ← ∅  (empty coalition)
    v_prev ← baseline
    
    FOR each feature_i in π:
        # Add feature to coalition
        S ← S ∪ {feature_i}
        
        # Compute coalition value by marginalizing
        samples ← D.copy()
        FOR each feature_j in S:
            samples[:, feature_j] ← x[feature_j]
        v_curr ← mean(f(samples))
        
        # Marginal contribution
        Δ ← v_curr - v_prev
        φ[feature_i] += Δ
        
        # Update for next iteration
        v_prev ← v_curr
    
RETURN φ / n_samples
```

---

## 2. Asymmetric Shapley (AsymmetricShapley)

### What It Does

Computes **path-based Shapley values** that respect causal ordering. Instead of sampling from all permutations, it samples only from **causal paths** connecting source nodes to the outcome Y. Only features **on the sampled path** receive credit.

### Game Theory Foundation

**Modified Shapley for Causal Paths:**
```
φᵢ = E_π∈Π_causal [v(S_π^i ∪ {i}) - v(S_π^i)]
```

Where `Π_causal` is the restricted set of permutations that:
1. Only include nodes on a path from sources to outcome Y
2. Maintain topological order (ancestors before descendants)
3. Exclude nodes not on any path to Y

**Key Difference from Vanilla:**
- Vanilla: All features in every permutation
- Asymmetric: Only path features in each permutation
- Non-path features receive **zero attribution**

### Coalition & Permutation Mechanics

**Path Sampling Process:**
1. Identify source nodes (no incoming edges, can reach Y)
2. Randomly select ONE source node
3. Find all paths from that source to outcome Y
4. Randomly select ONE path from that source
5. Remove outcome Y from path (added conceptually at end)
6. Use path as the permutation order

**Coalition Formation:**
Same as vanilla Shapley, but:
- Coalitions only contain features from the sampled path
- Non-path features are always treated as "missing" (sampled from background)

**Current Implementation Detail:**
The code uses: `ordering = path` (line 908), meaning ONLY path nodes are included. Non-path nodes are completely excluded from the game.

### Causal Diagram Usage

**USED FOR: Path Identification & Ordering**

1. **Adjacency Matrix → Directed Graph:**
   - Extract binary directed graph from adjacency matrix
   - `directed_graph[i,j] = 1` means i → j

2. **Source Detection:**
   - Find nodes with no incoming edges
   - Filter to sources that have paths to outcome Y
   - `self.source_nodes = [filtered sources]`

3. **Path Finding (DFS):**
   ```python
   def _find_all_paths_to_outcome(source, outcome):
       # Depth-first search from source to outcome
       # Returns list of paths: [[node1, node2, ..., outcome], ...]
   ```

4. **Topological Constraint:**
   - Paths inherently respect topological order
   - Parents always appear before children on the path

**NO FILTERING OF ADJACENCY MATRIX** - Uses full causal graph to find paths.

### Pseudocode

```python
ALGORITHM: Asymmetric Shapley (Path-Based)

INPUT:
    - instance x (features to explain)
    - model f (trained predictor)
    - background_data D
    - causal_graph G (adjacency matrix including Y)
    - n_samples (number of path samples)

OUTPUT:
    - φ ∈ R^n (Shapley values for n features, excluding Y)

PREPROCESSING:
    # Extract directed graph
    directed_graph ← extract_directed_edges(causal_graph)
    
    # Build parent/child relationships
    parents ← {node: [p where directed_graph[p, node] = 1]}
    children ← {node: [c where directed_graph[node, c] = 1]}
    
    # Find source nodes (no parents, can reach Y)
    all_sources ← {node: len(parents[node]) == 0}
    
    # Filter sources via backward BFS from Y
    reachable_from_Y ← backward_BFS(Y, parents)
    source_nodes ← all_sources ∩ reachable_from_Y

INITIALIZE:
    φ ← zeros(n)
    baseline ← mean(f(D))
    outcome_idx ← n  # Y is last variable

FOR trial = 1 to n_samples:
    # Sample random source
    source ← random_choice(source_nodes)
    
    # Find all paths from source to Y
    paths ← DFS_find_paths(source, outcome_idx, directed_graph)
    
    IF paths is empty:
        CONTINUE  # Skip this trial
    
    # Sample random path
    path ← random_choice(paths)
    
    # Remove outcome from path (Y is always last conceptually)
    path ← path[:-1]  # Remove Y
    
    # Use path as permutation order
    π ← path
    
    # Compute marginal contributions
    S ← ∅
    v_prev ← baseline
    
    FOR each feature_i in π:
        S ← S ∪ {feature_i}
        
        # Coalition value
        samples ← D.copy()
        FOR each feature_j in S:
            samples[:, feature_j] ← x[feature_j]
        v_curr ← mean(f(samples))
        
        # Marginal contribution
        Δ ← v_curr - v_prev
        φ[feature_i] += Δ
        
        v_prev ← v_curr

RETURN φ / n_samples
```

**Key Observations:**
- Only path nodes accumulate contributions
- Non-path nodes remain at φ = 0
- Each trial focuses on ONE random path

---

## 3. Causal Shapley (CausalShapley)

### What It Does

Computes **interventional Shapley values** using do-calculus. Instead of conditioning `P(X | X_S = x_S)`, it **intervenes** `P(X | do(X_S = x_S))`, breaking incoming edges and removing confounding effects.

### Game Theory Foundation

**Causal Shapley Definition:**
```
φᵢ = Σ_{S⊆N\{i}} [|S|!(|N|-|S|-1)! / |N|!] × [v_do(S∪{i}) - v_do(S)]
```

Where:
```
v_do(S) = E[f(X) | do(X_S = x_S)]  (interventional distribution)
```

**Contrast with Standard Shapley:**
- Standard: `v(S) = E[f(X) | X_S = x_S]` (conditioning)
- Causal: `v_do(S) = E[f(X) | do(X_S = x_S)]` (intervention)

**Interventional Semantics:**
- `do(X_S = x_S)` cuts all incoming edges to variables in S
- Removes confounding from hidden common causes
- Isolates **direct causal effects** only

### Coalition & Permutation Mechanics

**Outer Loop (Permutations):**
Same as vanilla Shapley - sample random permutations of all features

**Inner Loop (Coalition Evaluation):**
For each coalition S, compute `v_do(S)` via **post-interventional sampling**:

1. Fix intervened variables: X_S = x_S (from instance)
2. Sample non-intervened variables from causal mechanism
3. Repeat M times, average predictions

### Post-Interventional Sampling Algorithm

**Key Data Structures:**

1. **Component Structure:**
   - Groups of features (confounded groups + individual features)
   - Ordered topologically (parents before children)
   
2. **Confounding Info:**
   - `confounded_info[component_idx]` = True if multiple features share hidden confounder
   - Built from bidirected edges in causal graph
   
3. **Parent Dictionary:**
   - `parents_dict[component_idx]` = parent feature indices for this component

**Sampling Process:**

```
FOR each component t in topological order:
    fixed_features ← component ∩ S (intervened)
    missing_features ← component \ S (to sample)
    parent_values ← values of parent features (already determined)
    
    IF component is CONFOUNDED:
        # Intervention breaks dependencies
        FOR each missing feature j:
            Sample X_j ~ P(X_j | Parents(X_j))  # INDEPENDENT
    ELSE:
        # Preserve within-component dependencies
        Sample X_missing ~ P(X_missing | Parents, X_fixed)  # JOINT
```

**Implementation:**
Uses conditional Gaussian sampling with precomputed covariance matrices.

### Causal Diagram Usage

**USED FOR: Component Structure & Sampling**

1. **Adjacency Matrix (discovered_adj):**
   - Binary DAG: `adj[i,j]=1` means i → j
   - Used to build parent relationships
   - Must be topologically sortable (no cycles)

2. **Confounders (discovered_conf):**
   - List of pairs: `[(X1, X2), (X3, X4), ...]`
   - `(X1, X2)` means bidirected edge X1 ↔ X2
   - Indicates shared hidden confounder

3. **Component Construction:**
   ```python
   # Group confounded features
   FOR each pair (i, j) in discovered_conf:
       Group i and j in same component
   
   # Individual components for non-confounded features
   FOR each remaining feature:
       Create singleton component
   
   # Topological sort of components
   components ← topological_order(components, parents_dict)
   ```

4. **Parent Dictionary:**
   ```python
   FOR each component C:
       parents_dict[C] ← {parent features of any node in C}
   ```

**NO ADJACENCY FILTERING** - Uses full causal graph for sampling.

### Pseudocode

```python
ALGORITHM: Causal Shapley (Post-Interventional)

INPUT:
    - instance x
    - model f
    - background_data D
    - discovered_adj (adjacency matrix)
    - discovered_conf (confounded pairs)
    - n_samples (outer permutations)
    - M_inner (samples per coalition)

OUTPUT:
    - φ ∈ R^n (Causal Shapley values)

PREPROCESSING:
    # Build component structure
    components, confounded_info, parents_dict ← 
        extract_causal_structure(discovered_adj, discovered_conf)
    
    # Precompute Gaussian statistics
    μ ← mean(D)
    Σ ← cov(D)

INITIALIZE:
    φ ← zeros(n)
    baseline ← mean(f(D))

FOR trial = 1 to n_samples:
    # Sample random permutation
    π ← random_permutation([1, ..., n])
    
    S ← ∅
    v_prev ← baseline
    
    FOR each feature_i in π:
        S ← S ∪ {feature_i}
        
        # ===== POST-INTERVENTIONAL SAMPLING =====
        v_curr ← 0
        FOR m = 1 to M_inner:
            sample ← zeros(n)
            
            # Fix interventions
            FOR j in S:
                sample[j] ← x[j]
            
            # Sample non-interventions by component
            FOR each component C in topological_order:
                fixed_C ← C ∩ S
                missing_C ← C \ S
                parent_vals ← sample[parents_dict[C]]
                
                IF confounded_info[C]:
                    # INDEPENDENT sampling
                    FOR each j in missing_C:
                        parent_vals_j ← sample[parents[j]]
                        sample[j] ~ Gaussian_conditional(
                            target=[j],
                            conditioning=parents[j],
                            values=parent_vals_j
                        )
                ELSE:
                    # JOINT sampling
                    sample[missing_C] ~ Gaussian_conditional(
                        target=missing_C,
                        conditioning=parents_dict[C] ∪ fixed_C,
                        values=[parent_vals, sample[fixed_C]]
                    )
            
            v_curr += f(sample)
        
        v_curr ← v_curr / M_inner
        
        # Marginal contribution
        Δ ← v_curr - v_prev
        φ[feature_i] += Δ
        
        v_prev ← v_curr

RETURN φ / n_samples
```

**Computational Cost:**
- Outer: O(n_samples × n)
- Inner: O(M_inner × n_components)
- **Total**: O(n_samples × n × M_inner × n_features)
- Much more expensive than vanilla due to M_inner samples per coalition

---

## 4. Shapley Flow (ShapleyFlow + ShapleyFlowWrapper)

### What It Does

Computes **edge-level attributions** instead of feature-level. Explains which **causal edges** (X_i → X_j) contribute most to predictions. Aggregates edge credits to get node (feature) importance.

### Game Theory Foundation

**Edge Shapley Values:**
```
φ_{i→j} = E_π∈Π_edges [v(E_π^{i→j} ∪ {i→j}) - v(E_π^{i→j})]
```

Where:
- `Π_edges`: Random permutations of EDGES (not features)
- `E_π^{i→j}`: Edges appearing before (i→j) in permutation π
- `v(E)`: Model prediction when edge set E is active

**Coalition of Edges:**
- Edge ACTIVE: Child uses foreground value from instance
- Edge INACTIVE: Child sampled as "missing" from P(child | active_parents)

**Node Attribution (aggregation):**
```
φ_i = Σ_{j: i→j exists} φ_{i→j}
```

### Coalition & Permutation Mechanics

**Two Modes:**

#### Mode 1: Exhaustive DFS (use_path_sampling=False)

1. Start DFS from each source node
2. At each node, randomly permute children
3. For each child edge in random order:
   - Compute v(current_edges)
   - Add edge
   - Compute v(current_edges ∪ {edge})
   - Marginal = difference
   - Recurse to child

#### Mode 2: Path Sampling (use_path_sampling=True, **DEFAULT**)

1. **Backward sampling** from sink to source:
   - Start at sink (Y)
   - Randomly select one parent
   - Move to parent, repeat
   - Stop when reaching a source node
   - Reverse to get source→sink path

2. For each sampled path:
   - Extract edges in path
   - Random permutation of these edges
   - Evaluate incrementally: ∅ → +edge1 → +edge2 → ...
   - Compute marginal for each edge

**Why Backward Sampling?**
- Sources are pre-filtered (can reach Y)
- Starting from Y and walking backward ALWAYS succeeds
- No retries needed, every attempt is valid
- More efficient than forward random walks

### Causal Diagram Usage

**FILTERING IN WRAPPER (ShapleyFlowWrapper):**

**Important: The core ShapleyFlow class does NOT filter - filtering happens in the wrapper!**

1. **Source Filtering (Backward BFS):**
   ```python
   # Build parent dict from adjacency
   parents[i] = [j where adj[j,i] ≠ 0]
   
   # Find potential sources (no parents)
   potential_sources = [i where len(parents[i]) == 0]
   
   # Backward BFS from Y
   reachable_from_Y = set()
   queue = [Y]
   WHILE queue not empty:
       current = queue.pop()
       reachable_from_Y.add(current)
       FOR each parent in parents[current]:
           IF parent not visited:
               queue.append(parent)
   
   # Filter sources
   source_nodes = potential_sources ∩ reachable_from_Y
   ```

2. **Graph Structure (Full Adjacency):**
   ```python
   # NO adjacency filtering - uses full graph!
   graph_structure[i] = [j where adj[i,j] ≠ 0]  # All edges
   ```

**Key Point:** ShapleyFlowWrapper filters SOURCES but NOT edges. All edges in the original causal graph are used.

### Pseudocode

```python
ALGORITHM: Shapley Flow (Path Sampling Mode)

INPUT:
    - x_foreground (instance to explain)
    - x_background (baseline instance)
    - graph_structure (adjacency list: node → children)
    - source_nodes (filtered to Y-reachable)
    - sink_node (outcome Y)
    - n_samples (number of trials)
    - paths_per_source (paths to sample per source)

OUTPUT:
    - edge_attributions: {(u,v): attribution score}

PREPROCESSING:
    # Build parent dict for backward sampling
    parents ← {node: [] for all nodes}
    FOR each (parent, child) in graph_structure:
        parents[child].append(parent)

INITIALIZE:
    edge_attributions ← {(u,v): 0.0 for all edges}
    total_paths ← len(source_nodes) × paths_per_source

FOR trial = 1 to n_samples:
    
    FOR path_idx = 1 to total_paths:
        # ===== BACKWARD PATH SAMPLING =====
        path_edges ← []
        current ← sink_node
        visited ← {sink_node}
        
        WHILE current not in source_nodes:
            # Get unvisited parents
            candidates ← [p in parents[current] if p not in visited]
            
            IF candidates is empty:
                BREAK  # Dead end
            
            # Random parent selection
            parent ← random_choice(candidates)
            
            # Store edge in forward direction
            path_edges.append((parent, current))
            
            visited.add(parent)
            current ← parent
        
        # Reverse to get source→sink direction
        path_edges ← reverse(path_edges)
        
        # ===== MARGINAL CONTRIBUTION COMPUTATION =====
        # Random permutation of edges in this path
        perm_edges ← random_permutation(path_edges)
        
        # Baseline (no edges active)
        v_prev ← evaluate_system([], x_foreground, x_background)
        
        history ← []
        FOR each edge in perm_edges:
            history.append(edge)
            
            # Evaluate with this edge active
            v_curr ← evaluate_system(history, x_foreground, x_background)
            
            # Marginal contribution
            marginal ← v_curr - v_prev
            edge_attributions[edge] += marginal
            
            v_prev ← v_curr

# Average across all path samples
FOR each edge in edge_attributions:
    edge_attributions[edge] /= (n_samples × total_paths)

RETURN edge_attributions


HELPER: evaluate_system(active_edges, x_fg, x_bg)
    """Evaluate system state given active edges"""
    
    node_values ← {}
    
    IF active_edges is empty:
        # Baseline: all features at background
        RETURN predict(x_bg)
    
    # Determine which nodes have active incoming edges
    FOR each node:
        has_active_incoming ← any((p, node) in active_edges)
        
        IF node is source:
            has_active_outgoing ← any((node, c) in active_edges)
            IF has_active_outgoing:
                node_values[node] ← x_fg[node]
            ELSE:
                node_values[node] ← x_bg[node]
        
        ELSE IF has_active_incoming:
            node_values[node] ← x_fg[node]
        ELSE:
            node_values[node] ← x_bg[node]
    
    RETURN predict(node_values)
```

**Node Attribution (Post-Processing):**
```python
node_attributions[i] ← Σ_{j: (i,j) is edge} edge_attributions[(i,j)]
```

---

## 5. Graph Explainer Wrapper (GraphExplainerWrapper)

### What It Does

Wrapper for the external **shapflow library's GraphExplainer**. Key difference: **Filters the adjacency matrix** before constructing the graph, removing edges that cannot reach the target Y.

### Filtering Approach

**ADJACENCY MATRIX FILTERING (not just sources!):**

```python
ALGORITHM: Filter Adjacency to Sink-Reachable

INPUT:
    - adjacency_matrix (n × n)
    - feature_names
    - sink_name (target variable Y)

OUTPUT:
    - filtered_adjacency (same size, with irrelevant edges zeroed)
    - reachable_indices
    - filter_stats

PROCEDURE:
    n ← adjacency_matrix.shape[0]
    sink_idx ← index of sink_name
    
    # Build parent dictionary
    parents ← {}
    FOR j = 0 to n-1:
        parents[j] ← [i where adjacency[i,j] ≠ 0]
    
    # Backward BFS from sink
    reachable_indices ← set()
    queue ← [sink_idx]
    visited ← {sink_idx}
    
    WHILE queue not empty:
        current ← queue.pop(0)
        reachable_indices.add(current)
        
        FOR each parent in parents[current]:
            IF parent not visited:
                visited.add(parent)
                queue.append(parent)
    
    # Zero out edges NOT between reachable nodes
    filtered_adjacency ← adjacency_matrix.copy()
    FOR i = 0 to n-1:
        FOR j = 0 to n-1:
            IF i not in reachable_indices OR j not in reachable_indices:
                filtered_adjacency[i,j] ← 0
    
    RETURN filtered_adjacency, feature_names, reachable_indices, stats
```

**Key Difference from ShapleyFlowWrapper:**
- ShapleyFlowWrapper: Filters SOURCES only
- GraphExplainerWrapper: Filters EDGES in adjacency matrix
- Result: GraphExplainer works with smaller graph → faster computation

### Graph Construction

After filtering, converts adjacency matrix to shapflow Graph object:

1. Create Node objects for each feature
2. Add parent relationships from filtered adjacency
3. Fit causal functions from training data (using XGBoost/linear regression)
4. Return Graph with learned mechanisms

### Causal Diagram Usage

**USED FOR: Full Graph Structure with Edge Filtering**

1. **Filter adjacency matrix** (as shown above)
2. **Build Node objects** with parent relationships
3. **Learn causal functions**: `f_j = g_j(Parents(X_j))` from data
4. **GraphExplainer** uses learned graph for SHAP computation

---

## 6. Comparison Summary

| Method | Causal Structure | Filtering | Coalition Type | Computational Cost |
|--------|------------------|-----------|----------------|-------------------|
| **Vanilla Shapley** | None | None | All features, random order | O(n_samples × n × M_bg) |
| **Asymmetric Shapley** | Paths to Y | Sources (backward BFS) | Only path features | O(n_samples × path_length × M_bg) |
| **Causal Shapley** | Components + Confounders | None | All features, interventional | O(n_samples × n × M_inner × n) |
| **Shapley Flow** | Full DAG | Sources (backward BFS) | Edges, path sampling | O(n_samples × n_paths × path_length × M_bg) |
| **GraphExplainer** | Full DAG | Edges (backward BFS) | Edges (library-dependent) | Depends on shapflow implementation |

**Filtering Locations:**

1. **AsymmetricShapley**: 
   - `__init__`: Backward BFS to find Y-reachable sources
   - `_sample_causal_paths_to_outcome`: Samples paths from filtered sources

2. **ShapleyFlowWrapper**:
   - `__init__`: Backward BFS to filter sources
   - Does NOT filter adjacency matrix

3. **GraphExplainerWrapper**:
   - `__init__`: Calls `_filter_adjacency_to_sink_reachable`
   - Zeros out edges not between Y-reachable nodes
   - **Most aggressive filtering**

**Coalition Formation:**

- **Vanilla**: Random permutations of ALL features
- **Asymmetric**: Random paths, only path features
- **Causal**: Random permutations, post-interventional sampling
- **ShapleyFlow**: Random paths of EDGES (not features)
- **GraphExplainer**: Library-dependent (likely similar to ShapleyFlow)

**Causal Graph Interpretation:**

- **Vanilla**: Ignored
- **Asymmetric**: Paths define valid orderings
- **Causal**: Components define intervention semantics
- **ShapleyFlow**: Edges define game players
- **GraphExplainer**: Full graph with learned mechanisms

---

## Appendix: Key Implementation Details

### Backward BFS (Used in Multiple Methods)

```python
def backward_BFS(sink_idx, parents):
    """Find all nodes that can reach the sink"""
    reachable = set()
    queue = [sink_idx]
    visited = {sink_idx}
    
    while queue:
        current = queue.pop(0)
        reachable.add(current)
        
        for parent in parents[current]:
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
    
    return reachable
```

**Used by:**
- AsymmetricShapley: Filter sources
- ShapleyFlowWrapper: Filter sources  
- GraphExplainerWrapper: Filter adjacency matrix

### Conditional Gaussian Sampling (CausalShapley)

```python
def sample_conditional_gaussian(target, conditioning, values):
    """Sample X_target | X_cond = values using Gaussian formula"""
    
    Σ_AA = Σ[target, target]
    Σ_BB = Σ[conditioning, conditioning]
    Σ_AB = Σ[target, conditioning]
    
    μ_cond = μ_A + Σ_AB @ inv(Σ_BB) @ (values - μ_B)
    Σ_cond = Σ_AA - Σ_AB @ inv(Σ_BB) @ Σ_BA
    
    return sample(Normal(μ_cond, Σ_cond))
```

### Path Finding (AsymmetricShapley)

```python
def find_all_paths_DFS(source, target, graph, max_paths=20):
    """Find paths from source to target via DFS"""
    paths = []
    stack = [(source, [source])]
    
    while stack and len(paths) < max_paths:
        node, path = stack.pop()
        
        if node == target:
            paths.append(path)
            continue
        
        for child in graph[node]:
            if child not in path:  # Avoid cycles
                stack.append((child, path + [child]))
    
    return paths
```

---

## Conclusion

Each method uses the causal graph differently:

1. **Vanilla**: Ignores causality
2. **Asymmetric**: Uses paths for ordering
3. **Causal**: Uses structure for intervention
4. **ShapleyFlow**: Uses edges as game players
5. **GraphExplainer**: Uses filtered graph with learned mechanisms

**Filtering Summary:**
- AsymmetricShapley & ShapleyFlowWrapper: Filter **sources** only
- GraphExplainerWrapper: Filters **adjacency matrix** (most aggressive)
- CausalShapley & Vanilla: No filtering

The choice of method depends on:
- Available causal knowledge
- Computational budget
- Interpretation needs (features vs edges)
- Presence of confounding
