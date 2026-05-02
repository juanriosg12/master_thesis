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

Computes **Shapley values with causal ordering constraints** (Frye et al. 2021). Permutations are sampled uniformly from the set of all valid topological orderings of the X-feature DAG (its linear extensions). All features appear in every permutation; no feature receives structural zero attribution.

### Game Theory Foundation

**Modified Shapley for Causal Orderings (Frye et al. 2021):**
```
φᵢ = E_π∈Π_topo [v(S_π^i ∪ {i}) - v(S_π^i)]
```

Where `Π_topo` is the set of all valid topological orderings of the X-feature DAG:
- Every permutation respects the causal partial order: if i → j in the DAG then i precedes j in π
- All n_features features appear in every permutation
- Coalition value v(S) uses observational background marginalization (same as vanilla)

**Key Difference from Vanilla:**
- Vanilla: uniform over all n! permutations (ignores causal structure)
- Asymmetric: uniform over valid topological orderings only (set of linear extensions)

### Coalition & Permutation Mechanics

**Topological Ordering (Kahn-style uniform sampling):**
1. Initialise a pool of *ready* X-features — those whose in-degree among X-only edges is zero
2. At each step pick **uniformly at random** from the pool (swap-remove, O(1))
3. Decrement in-degrees of the selected node's X-children; add newly-ready children to pool
4. Repeat until all n_features are placed
5. Safety fallback: if the graph has unexpected cycles, unplaced nodes are appended in random order

**Coalition Formation:**
Same as vanilla Shapley:
- All features are included; coalitions grow incrementally along the sampled ordering
- Features IN coalition S: use instance's actual values (foreground)
- Features NOT in coalition S: sample from background data

### Causal Diagram Usage

**USED FOR: X-Only Topological Ordering**

1. **Adjacency Matrix → Directed Graph:**
   - Extract binary directed graph from adjacency matrix
   - `directed_graph[i,j] = 1` means i → j

2. **X-Only Ordering Structures (precomputed in `__init__`):**
   - `_x_children[i]`: children of node i among X features (Y excluded)
   - `_x_in_degree[i]`: number of X-feature parents of node i
   - Used by `_sample_topological_ordering()` at every trial; O(n) per call

3. **Cycle Detection:**
   - If DAG has cycles, `directed_graph` is zeroed out (falls back to unconstrained)

**NO FILTERING OF ADJACENCY MATRIX** — uses the full X-feature subgraph.

### Pseudocode

```python
ALGORITHM: Asymmetric Shapley (Random Topological Ordering, Frye et al. 2021)

INPUT:
    - instance x (features to explain)
    - model f (trained predictor)
    - background_data D
    - causal_graph G (adjacency matrix including Y, shape n+1 × n+1)
    - n_samples (number of random topological orderings)

OUTPUT:
    - φ ∈ R^n (Shapley values for n X-features)

PREPROCESSING:
    # Extract directed graph; detect and disable cycles
    directed_graph ← extract_directed_edges(causal_graph)
    
    # Build X-only adjacency structures (Y = last node, excluded)
    FOR i = 0 to n-1:
        FOR j = 0 to n-1:
            IF directed_graph[i, j] ≠ 0:
                _x_children[i].append(j)
                _x_in_degree[j] += 1

INITIALIZE:
    φ ← zeros(n)
    baseline ← mean(f(D))

FOR trial = 1 to n_samples:
    # ===== SAMPLE RANDOM VALID TOPOLOGICAL ORDERING =====
    remaining_in_degree ← copy(_x_in_degree)
    ready ← [i for i in 0..n-1 if remaining_in_degree[i] == 0]
    π ← []
    
    WHILE ready is not empty:
        # Uniform random pick via swap-remove (O(1))
        idx ← randint(len(ready))
        node ← ready[idx]; ready[idx] ← ready[-1]; ready.pop()
        π.append(node)
        FOR child in _x_children[node]:
            remaining_in_degree[child] -= 1
            IF remaining_in_degree[child] == 0:
                ready.append(child)
    
    # ===== COMPUTE MARGINAL CONTRIBUTIONS =====
    S ← ∅
    v_prev ← baseline
    
    FOR each feature_i in π:
        S ← S ∪ {feature_i}
        
        # Coalition value: marginalize non-S features from background
        samples ← D.copy()
        FOR each feature_j in S:
            samples[:, feature_j] ← x[feature_j]
        v_curr ← mean(f(samples))
        
        φ[feature_i] += v_curr - v_prev
        v_prev ← v_curr

RETURN φ / n_samples
```

**Key Properties:**
- All features appear in every permutation — no structural zeros
- Causal partial order is respected for every ancestor-descendant pair
- O(n) sampling cost per permutation
- Reference: Frye, Rowat & Feige (2021), NeurIPS

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

**Outer Loop (Permutations) — Component Topological Ordering:**

Instead of a fully random permutation, the outer loop draws from the uniform distribution over valid topological orderings of the **component DAG** in two stages:

*Stage 1 — Kahn-style random sort of components:*
- Maintain a pool of ready components (in-degree 0 in the component DAG)
- At each step pick uniformly at random from the pool (swap-remove, O(1))
- Decrement child in-degrees; add newly-ready children to pool

*Stage 2 — Expand to features:*
- For each component in the sampled order, expand to its features
- Within a confounded component the features are shuffled randomly
- Single-feature (non-confounded) components are trivially ordered

The result is a flat feature list where every ancestor always precedes every descendant.

**Inner Loop (Coalition Evaluation):**
For each coalition S, compute `v_do(S)` via **post-interventional sampling**:

1. Fix intervened variables: X_S = x_S (from instance)
2. Sample non-intervened variables from causal mechanism, traversing components in the **fixed** deterministic topological order (not randomised — do-calculus requires consistent structural propagation)
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
    # ===== SAMPLE RANDOM COMPONENT TOPOLOGICAL ORDERING =====
    remaining_in_degree ← copy(_comp_in_degree)
    ready ← [c for c in 0..n_comps-1 if remaining_in_degree[c] == 0]
    comp_order ← []
    
    WHILE ready is not empty:
        idx ← randint(len(ready))
        comp ← ready[idx]; ready[idx] ← ready[-1]; ready.pop()
        comp_order.append(comp)
        FOR child_comp in _comp_children[comp]:
            remaining_in_degree[child_comp] -= 1
            IF remaining_in_degree[child_comp] == 0:
                ready.append(child_comp)
    
    # Expand component order to flat feature list
    π ← []
    FOR comp_idx in comp_order:
        features ← causal_graph_components[comp_idx]
        IF len(features) > 1: shuffle(features)  # confounded component
        π.extend(features)
    
    S ← ∅
    v_prev ← baseline
    
    FOR each feature_i in π:
        S ← S ∪ {feature_i}
        
        # ===== POST-INTERVENTIONAL SAMPLING (fixed topological order inside) =====
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
    """Evaluate system state given active edges.
    Y (sink_node) is always stripped from the feature vector before predict(),
    so observed Y values never reach the model."""
    
    node_values ← {}
    
    IF active_edges is empty:
        # Baseline: all features at background
        RETURN predict(x_bg)
    
    history_set ← set(active_edges)
    
    # Determine value for each node
    FOR each node:
        has_active_incoming ← any((p, node) in history_set
                                   for p in parents[node])
        
        IF node is source:
            has_active_outgoing ← any((node, c) in history_set
                                       for c in graph[node])
            IF has_active_outgoing:
                node_values[node] ← x_fg[node]   # source activated
            ELSE:
                node_values[node] ← x_bg[node]   # source at background
        
        ELSE IF has_active_incoming:
            node_values[node] ← x_fg[node]        # downstream node activated
        
        ELSE IF (node, sink_node) in history_set:
            # Direct X_k → Y edge is active: bring X_k to foreground so the
            # model sees its actual feature value. Y itself is excluded from
            # the prediction input (see feature_indices below), so no Y data
            # leakage occurs.
            node_values[node] ← x_fg[node]
        
        ELSE:
            node_values[node] ← x_bg[node]        # node at background
    
    # Exclude Y (sink_node) from model input — Y is never fed to the predictor
    feature_indices ← [i for i in range(n_features) if i != sink_node]
    x_features ← [node_values[i] for i in feature_indices]
    RETURN predict(x_features)
```

**Node Attribution (Post-Processing):**
```python
node_attributions[i] ← Σ_{j: (i,j) is edge} edge_attributions[(i,j)]
```

---

## 5. Graph Explainer Wrapper (GraphExplainerWrapper)

### What It Does

Wrapper for the external **shapflow library's GraphExplainer**. 

**Two key differences from ShapleyFlow:**

1. **Filters the adjacency matrix** (not just sources) - removes edges that cannot reach target Y
2. **Learns causal functions** from training data before computing attributions

### How It Works (Two-Phase Process)

**PHASE 1: Graph Construction with Learned Causal Functions**

Unlike ShapleyFlow (which samples background/foreground data), GraphExplainer:
1. Learns causal mechanisms: `f_j = g_j(Parents(X_j))` from training data
2. Uses learned functions (XGBoost/linear) to model relationships
3. Builds explicit causal graph with these functions

**PHASE 2: Edge Attribution Using Learned Functions**

Edge attributions are computed using:
- The learned causal functions (not data sampling)
- Interventional semantics on the graph structure
- GraphExplainer's algorithm from shapflow library

**Key Difference:**
- **ShapleyFlow**: Edge (i→j) active/inactive → use foreground/background data
- **GraphExplainer**: Edge (i→j) active/inactive → use learned function f_j or not

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
     (`_adjacency_to_graph`)

**CRITICAL PREPARATION STEP:** After filtering, converts adjacency to shapflow Graph with learned causal functions:

```python
ALGORITHM: Build Graph with Learned Functions

INPUT:
    - filtered_adjacency (already filtered to Y-reachable edges)
    - feature_names
    - train_data (for learning functions)
    - fit_method ('xgboost', 'linear', etc.)

STEP 1: Create Node objects
    FOR each feature in feature_names:
        Create Node(name, f=None, args=[], is_target=?)
        Store in nodes_dict[name]

STEP 2: Add parent relationships from filtered adjacency
    FOR each child_node:
        parent_indices ← [i where filtered_adjacency[i, child] ≠ 0]
        FOR each parent_idx:
            child_node.args.append(parent_node)
            parent_node.children.append(child_node)

STEP 3: Create Graph object
    graph ← Graph(nodes=list(nodes_dict.values()))

STEP 4: Learn causal functions from training data
    graph.fit_missing_links(train_data, method=fit_method)
    # This learns: f_j = g_j(Parents(X_j)) for each node j
    # Using XGBoost/linear regression on train_data
    
RETURN graph (with learned functions embedded)
```

**What `fit_missing_links` does:**
- For each node j: Fit `X_j = f_j(Parents(X_j), noise)` using training data
- XGBoost: Learns non-linear functions
- Linear: Learns linear coefficients
- Stores learned function in each Node object

### Causal Diagram Usage

**USED FOR: Full Graph Structure with Edge Filtering + Learned Mechanisms**

1. **Filter adjacency matrix** (backward BFS to zero out non-Y-reachable edges)
2. **Build Node objects** with parent relationships from filtered adjacency
3. **Learn causal functions**: `f_j = g_j(Parents(X_j))` from training data
   - Uses actual data to fit relationships
   - Each node stores its learned function
4. **GraphExplainer** uses learned graph for SHAP computation
   - Edge attributions use learned functions (not data sampling)
   - Can evaluate counterfactuals using learned mechanisms

### Key Algorithmic Difference: GraphExplainer vs ShapleyFlow

| Aspect | ShapleyFlow | GraphExplainer |
|--------|-------------|----------------|
| **Edge Semantics** | Active: use foreground data<br>Inactive: use background data | Active: use learned function f_j<br>Inactive: marginalize out |
| **Data Usage** | Samples from background data at runtime | Learns functions from train data beforehand |
| **Graph Representation** | Adjacency list only | Full causal graph with learned f_j |
| **Prediction** | Direct model evaluation | Can simulate via learned mechanisms |
| **Computational** | Needs background data each time | Upfront cost to learn functions |
    RETURN filtered_adjacency, feature_names, reachable_indices, stats
```

**Key Difference from ShapleyFlowWrapper:**
- ShapleyFlowWrapper: Filters SOURCES only
- GraphExplainerWrapper: Filters EDGES in adjacency matrix
- Result: GraphExplainer works with smaller graph → faster computation

### Graph Construction

After filtering, converts adjacency matrix to shapflow Graph object:
, data sampling
- **GraphExplainer**: Random paths of EDGES, learned function evaluation
2. Add parent relationships from filtered adjacency
3. Fit causal functions from training data (using XGBoost/linear regression)
4. Return Graph with learned mechanisms

### Causal Diagram Usage

**USED FOR: Full Graph Structure with Edge F (data sampling)
- **GraphExplainer**: Edges define game players (learned mechanisms)

**Data vs Function-Based:**

- **ShapleyFlow**: Runtime data sampling
  - Edge active: child uses foreground value
  - Edge inactive: child uses background value
  - No learned functions

- **GraphExplainer**: Learned causal functions
  - Edge active: evaluate learned f_j(parents)
  - Edge inactive: marginalize using learned distribution
  - Functions fit beforehand from training data
1. **Filter adjacency matrix** (as shown above)
2. **Build Node objects** with parent relationships
3. **Learn causal functions**: `f_j = g_j(Parents(X_j))` from data
4. **GraphExplainer** uses learned graph for SHAP computation

---

## 6. Comparison Summary

| Method | Causal Structure | Filtering | Coalition Type | Computational Cost |
|--------|------------------|-----------|----------------|-------------------|
| **Vanilla Shapley** | None | None | All features, uniform random order | O(n_samples × n × M_bg) |
| **Asymmetric Shapley** | X-feature DAG | None | All features, uniform over topological orderings | O(n_samples × n × M_bg) |
| **Causal Shapley** | Components + Confounders | None | All features, component topo. ordering, interventional values | O(n_samples × n × M_inner × n) |
| **Shapley Flow** | Full DAG | Sources (backward BFS) | Edges, path sampling | O(n_samples × n_paths × path_length × M_bg) |
| **GraphExplainer** | Full DAG | Edges (backward BFS) | Edges (library-dependent) | Depends on shapflow implementation |

**Filtering Locations:**

1. **AsymmetricShapley**: 
   - No source or adjacency filtering — uses the full X-feature subgraph
   - Cycle detection in `__init__`: zeros out directed_graph if cycles found

2. **ShapleyFlowWrapper**:
   - `__init__`: Backward BFS to filter sources
   - Does NOT filter adjacency matrix

3. **GraphExplainerWrapper**:
   - `__init__`: Calls `_filter_adjacency_to_sink_reachable`
   - Zeros out edges not between Y-reachable nodes
   - **Most aggressive filtering**

**Coalition Formation:**

- **Vanilla**: Random permutations of ALL features (uniform over n!)
- **Asymmetric**: ALL features in every permutation; orderings drawn uniformly from topological orderings of the X-DAG
- **Causal**: ALL features; outer permutation drawn from topological orderings of component DAG; inner sampling is post-interventional
- **ShapleyFlow**: Random paths of EDGES (not features)
- **GraphExplainer**: Library-dependent (likely similar to ShapleyFlow)

**Causal Graph Interpretation:**

- **Vanilla**: Ignored
- **Asymmetric**: Defines valid topological orderings (linear extensions of the DAG)
- **Causal**: Components define intervention semantics; component DAG constrains outer permutation
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
