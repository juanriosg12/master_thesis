# Shapley Values Algorithms: Pseudo Code and Detailed Explanations

This document provides detailed pseudo code explanations for all Shapley value implementations in `shapley_values.py`.

---

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [vanilla Shapley (ShapleyFromScratch)](#1-vanilla-shapley-shapleyfromscratch)
3. [Asymmetric Shapley (AsymmetricShapley)](#2-asymmetric-shapley-asymmetricshapley)
4. [Causal Shapley (CausalShapley)](#3-causal-shapley-causalshapley)
5. [Shapley Flow (ShapleyFlow)](#4-shapley-flow-shapleyflow)
6. [Comparison Table](#5-comparison-table)

---

## Core Concepts

### What are Shapley Values?

Shapley values decompose a model's prediction into contributions from each feature:

```
f(x) = baseline + φ₁ + φ₂ + ... + φₙ
```

where φᵢ is the Shapley value (contribution) of feature i.

### Key Properties:

1. **Efficiency**: Sum of contributions equals total prediction difference
   ```
   Σᵢ φᵢ = f(x) - f(baseline)
   ```

2. **Symmetry**: Identical features get identical contributions

3. **Dummy**: Features that don't affect output get zero contribution

4. **Additivity**: For sum of models, Shapley values sum

### The Coalition Value Function v(S)

For a coalition (subset) S of features, we need to compute v(S) = "expected model output when only features in S are known"

**Three main approaches**:

1. **Marginalization (Vanilla Shapley)**:
   ```
   v(S) = E[f(X) | X_S = x_S]
   ```
   Condition on observed features, average over missing features from background data

2. **Intervention (Causal Shapley)**:
   ```
   v(S) = E[f(X) | do(X_S = x_S)]
   ```
   Intervene on observed features (cut incoming edges in causal graph)

3. **Graph-based (Shapley Flow)**:
   ```
   v(E) = E[f(X) | active_edges = E]
   ```
   Value of edge set E (features get values based on edge activation)

---

## 1. Vanilla Shapley (ShapleyFromScratch)

### Overview

Computes standard Shapley values using the original game-theoretic definition.

### Mathematical Definition

```
φᵢ = Σ_{S⊆N\{i}} [|S|!(n-|S|-1)! / n!] * [v(S∪{i}) - v(S)]
```

where:
- N = set of all features
- S = coalition not containing feature i
- v(S) = expected prediction when only features in S are observed
- Weight term ensures fair credit distribution

### Pseudo Code: Monte Carlo Approximation

```
ALGORITHM: ComputeShapley_MonteCarlo(instance x, background_data B, n_samples)

INPUT:
  - x: instance to explain (n_features)
  - B: background dataset (m_samples × n_features) 
  - model: trained predictive model
  - n_samples: number of random permutations

OUTPUT:
  - φ: Shapley values (n_features)

PROCEDURE:

1. Compute baseline:
   baseline = mean(model.predict(B))

2. Initialize:
   φ = zeros(n_features)

3. FOR iter = 1 to n_samples:

   a. Sample random permutation π of features:
      π = random_permutation([0, 1, 2, ..., n-1])
   
   b. Initialize empty coalition:
      S = ∅
      v_prev = baseline
   
   c. FOR each feature i in order π:
      
      i. Add feature to coalition:
         S = S ∪ {i}
      
      ii. Compute coalition value v(S):
         v_curr = ComputeCoalitionValue(x, S, B)
      
      iii. Marginal contribution:
         Δᵢ = v_curr - v_prev
      
      iv. Accumulate:
         φ[i] += Δᵢ
      
      v. Update:
         v_prev = v_curr

4. Average over permutations:
   φ = φ / n_samples

5. RETURN φ
```

### Pseudo Code: Coalition Value Computation

```
FUNCTION: ComputeCoalitionValue(instance x, coalition S, background_data B)

INPUT:
  - x: instance values (n_features)
  - S: set of feature indices in coalition
  - B: background data (m × n_features)

OUTPUT:
  - value: expected prediction E[f(x) | X_S = x_S]

PROCEDURE:

1. Create m synthetic samples:
   FOR j = 1 to m:
      sample[j] = B[j]  // Start with background
      
      FOR each feature i in S:
         sample[j][i] = x[i]  // Replace with instance values

2. Predict on all samples:
   predictions = model.predict(samples)

3. Average:
   value = mean(predictions)

4. RETURN value
```

### Background vs Foreground Usage

| Component | Source | Purpose |
|-----------|--------|---------|
| **Foreground** | Instance x to explain | Provides values for features IN the coalition |
| **Background** | Training/reference data | Provides values for features NOT in the coalition |
| **Result** | Average over background | Marginalizes out missing features |

### Example Walkthrough

**Setup**: 
- Features: X₀, X₁, X₂
- Instance: x = [5, 10, 15]
- Background has 100 samples

**Permutation**: π = [X₁, X₀, X₂]

| Step | Coalition S | Synthetic Samples | v(S) | Marginal |
|------|-------------|-------------------|------|----------|
| 0 | ∅ | All from background | baseline | - |
| 1 | {X₁} | X₁=10, others from bg (100 samples) | v₁ | φ₁ += v₁ - baseline |
| 2 | {X₁, X₀} | X₁=10, X₀=5, X₂ from bg (100 samples) | v₂ | φ₀ += v₂ - v₁ |
| 3 | {X₁, X₀, X₂} | All from instance: [5,10,15] | v₃ | φ₂ += v₃ - v₂ |

Repeat for many random permutations and average.

---

## 2. Asymmetric Shapley (AsymmetricShapley)

### Overview

Modifies vanilla Shapley to respect causal ordering: a feature can only contribute if its causal parents are already in the coalition.

### Key Insight

In causal graphs X₁ → X₂ → Y:
- Invalid coalition: {X₂} without X₁ (cuts causal path)
- Valid coalition: {X₁, X₂} (respects causal order)

### Causal Constraint

A coalition S is valid if:
```
∀i ∈ S: Parents(i) ⊆ S
```

### Pseudo Code: Asymmetric Shapley (Frye Method)

```
ALGORITHM: ComputeAsymmetricShapley_Frye(instance x, causal_graph G, background B, n_samples)

INPUT:
  - x: instance to explain
  - G: directed acyclic graph (adjacency matrix)
  - B: background dataset
  - n_samples: number of causal permutations

OUTPUT:
  - φ: Asymmetric Shapley values

PROCEDURE:

1. Extract causal structure from G:
   parents = ExtractParents(G)
   // parents[i] = set of parent features of feature i

2. Initialize:
   φ = zeros(n_features)
   baseline = mean(model.predict(B))

3. FOR iter = 1 to n_samples:

   a. Sample causal permutation:
      π = SampleCausalPermutation_Frye(parents)
   
   b. Initialize:
      S = ∅
      v_prev = baseline
   
   c. FOR each feature i in order π:
      
      i. Add to coalition:
         S = S ∪ {i}
      
      ii. Compute value:
         v_curr = ComputeCoalitionValue(x, S, B)
      
      iii. Marginal contribution:
         φ[i] += v_curr - v_prev
      
      iv. Update:
         v_prev = v_curr

4. Average:
   φ = φ / n_samples

5. RETURN φ
```

### Pseudo Code: Sampling Causal Permutations (Frye Method)

```
FUNCTION: SampleCausalPermutation_Frye(parents)

INPUT:
  - parents: dictionary mapping feature → set of parent features

OUTPUT:
  - π: valid causal permutation (list of feature indices)

PROCEDURE:

1. Initialize:
   π = []  // Result permutation
   remaining = {0, 1, ..., n-1}  // Features not yet added

2. WHILE remaining is not empty:

   a. Find candidates (features whose parents are already in π):
      candidates = []
      FOR each feature i in remaining:
         IF parents[i] ⊆ set(π):
            candidates.append(i)
   
   b. Randomly select one candidate:
      selected = random_choice(candidates)
   
   c. Add to permutation:
      π.append(selected)
      remaining.remove(selected)

3. RETURN π
```

### Example: Causal Permutation Sampling

**Causal Graph**: X₀ → X₂, X₁ → X₂

```
Parents:
  parents[X₀] = ∅
  parents[X₁] = ∅
  parents[X₂] = {X₀, X₁}

Sampling Process:

Step 1: π = [], remaining = {X₀, X₁, X₂}
  - Candidates: {X₀, X₁} (both have no parents)
  - Select: X₁ (random)
  - π = [X₁], remaining = {X₀, X₂}

Step 2: π = [X₁], remaining = {X₀, X₂}
  - Candidates: {X₀} (X₂ needs both parents)
  - Select: X₀
  - π = [X₁, X₀], remaining = {X₂}

Step 3: π = [X₁, X₀], remaining = {X₂}
  - Candidates: {X₂} (both parents now in π)
  - Select: X₂
  - π = [X₁, X₀, X₂]

Result: π = [X₁, X₀, X₂] ✓ Valid causal ordering
```

### How Causal DAG is Used

1. **Extract directed edges**: 
   ```
   G[i,j] = 1  =>  edge i → j exists
   ```

2. **Build parent dictionary**:
   ```
   parents[j] = {i : G[i,j] ≠ 0}
   ```

3. **Constrain permutations**:
   - Only sample permutations where each feature appears after ALL its parents
   - This ensures coalitions respect causal dependencies

---

## 3. Causal Shapley (CausalShapley)

### Overview

Most rigorous causal method using **post-interventional distributions** to correctly handle confounding.

### Key Distinction

| Method | Formula | When to Use |
|--------|---------|-------------|
| **Vanilla** | v(S) = E[f(X) \| X_S = x_S] | No confounding |
| **Causal** | v(S) = E[f(X) \| do(X_S = x_S)] | Presence of confounding |

### Confounding Example

```
True Structure:
    U (hidden confounder)
   / \
  X₁  X₂
   \  /
    Y

Problem with Conditioning:
- P(X₂ | X₁=x₁) depends on U
- X₁ gets credit for effects from U
- OVERESTIMATES X₁'s causal effect

Solution with Intervention:
- P(X₂ | do(X₁=x₁)) independent of U
- Cuts edge U → X₁
- CORRECT causal effect
```

### Causal Graph Components

The algorithm partitions features into **components**:

1. **Confounded components**: Features sharing a hidden confounder
   - Indicated by bidirected edge: X₁ ↔ X₂
   - Sample features INDEPENDENTLY given parents
   
2. **Non-confounded components**: Individual features
   - Sample features JOINTLY given parents + siblings
   - Preserves mutual interactions

### Pseudo Code: Causal Shapley

```
ALGORITHM: ComputeCausalShapley(instance x, causal_structure, background B, 
                                 n_samples, M_inner)

INPUT:
  - x: instance to explain
  - causal_structure: (adjacency_matrix, confounders, feature_names)
  - B: background data for conditional distributions
  - n_samples: number of outer permutations
  - M_inner: number of inner samples per coalition

OUTPUT:
  - φ: Causal Shapley values

PROCEDURE:

1. Extract causal structure:
   (components, confounded_info, parents) = 
       ExtractCausalComponents(adjacency_matrix, confounders)
   
   // components: list of feature groups in topological order
   // confounded_info[comp_idx] = True if component has confounding
   // parents[comp_idx] = parent features for component

2. Precompute statistics for Gaussian sampling:
   μ = mean(B, axis=0)
   Σ = covariance(B)

3. Initialize:
   φ = zeros(n_features)
   baseline = mean(model.predict(B))

4. FOR iter = 1 to n_samples:

   a. Sample random permutation of ALL features:
      π = random_permutation([0, 1, ..., n-1])
   
   b. Initialize:
      S = ∅
      v_prev = baseline
   
   c. FOR each feature i in order π:
      
      i. Add to coalition:
         S = S ∪ {i}
      
      ii. Compute interventional value:
         v_curr = ComputeInterventionalValue(x, S, components, 
                                             confounded_info, parents, 
                                             μ, Σ, M_inner)
      
      iii. Marginal:
         φ[i] += v_curr - v_prev
      
      iv. Update:
         v_prev = v_curr

5. Average:
   φ = φ / n_samples

6. RETURN φ
```

### Pseudo Code: Post-Interventional Sampling

```
FUNCTION: ComputeInterventionalValue(x, S, components, confounded_info, 
                                     parents, μ, Σ, M_inner)

INPUT:
  - x: instance values
  - S: coalition (set of intervened features)
  - components: causal components in topological order
  - confounded_info: which components are confounded
  - parents: parent features for each component
  - μ, Σ: mean and covariance of background data
  - M_inner: number of samples to draw

OUTPUT:
  - value: E[f(X) | do(X_S = x_S)]

PROCEDURE:

1. Initialize result:
   predictions = []

2. FOR m = 1 to M_inner:

   a. Sample from do-distribution:
      sample = SamplePostInterventional(x, S, components, confounded_info,
                                        parents, μ, Σ)
   
   b. Predict:
      y_pred = model.predict(sample)
      predictions.append(y_pred)

3. Average over samples:
   value = mean(predictions)

4. RETURN value
```

### Pseudo Code: Sampling from P(X | do(X_S = x_S))

```
FUNCTION: SamplePostInterventional(x, S, components, confounded_info, 
                                   parents, μ, Σ)

INPUT:
  - x: instance values
  - S: intervened features (coalition)
  - components: causal components in topo order
  - confounded_info: confounding status per component
  - parents: parent features per component
  - μ, Σ: Gaussian parameters

OUTPUT:
  - sample: one sample from P(X | do(X_S = x_S))

PROCEDURE:

1. Initialize sample:
   sample = zeros(n_features)

2. Step 1: Fix intervened features
   FOR feature i in S:
      sample[i] = x[i]  // Intervention: set to instance value

3. Step 2: Process components in topological order
   FOR each component_t in components:
   
      a. Identify features:
         fixed_features = component_t ∩ S
         missing_features = component_t \ S
      
      b. IF missing_features is empty:
         CONTINUE  // All features in component are fixed
      
      c. Get parent values (already determined):
         parent_features = parents[component_t]
         parent_values = sample[parent_features]
      
      d. *** CRITICAL BRANCHING ***
      
      IF confounded_info[component_t] == True:
         // CONFOUNDED COMPONENT
         // Sample each missing feature INDEPENDENTLY given parents only
         
         FOR each feature j in missing_features:
            sample[j] = SampleConditionalGaussian(
                target = [j],
                conditioning = parent_features,
                conditioning_values = parent_values,
                μ, Σ
            )
      
      ELSE:
         // NON-CONFOUNDED COMPONENT  
         // Sample missing features JOINTLY given parents + fixed siblings
         
         conditioning = parent_features + fixed_features
         conditioning_values = sample[conditioning]
         
         sample[missing_features] = SampleConditionalGaussian(
             target = missing_features,
             conditioning = conditioning,
             conditioning_values = conditioning_values,
             μ, Σ
         )

4. RETURN sample
```

### Conditional Gaussian Sampling

```
FUNCTION: SampleConditionalGaussian(target, conditioning, values, μ, Σ)

INPUT:
  - target: indices of features to sample
  - conditioning: indices of features to condition on
  - values: observed values of conditioning features
  - μ: mean vector of joint distribution
  - Σ: covariance matrix of joint distribution

OUTPUT:
  - samples: sampled values for target features

FORMULA:
  Given X = [X_A, X_B] ~ N(μ, Σ), compute:
  
  X_A | X_B = b ~ N(μ_{A|B}, Σ_{A|B})
  
  where:
    μ_{A|B} = μ_A + Σ_{AB} Σ_{BB}^{-1} (b - μ_B)
    Σ_{A|B} = Σ_{AA} - Σ_{AB} Σ_{BB}^{-1} Σ_{BA}

PROCEDURE:

1. Extract submatrices:
   Σ_AA = Σ[target, target]
   Σ_BB = Σ[conditioning, conditioning]
   Σ_AB = Σ[target, conditioning]

2. Compute conditional mean:
   μ_cond = μ[target] + Σ_AB @ inv(Σ_BB) @ (values - μ[conditioning])

3. Compute conditional covariance:
   Σ_cond = Σ_AA - Σ_AB @ inv(Σ_BB) @ Σ_AB.T

4. Sample:
   samples = multivariate_normal(μ_cond, Σ_cond)

5. RETURN samples
```

### Example Walkthrough

**Setup**:
```
Features: X₀, X₁, X₂
Confounders: X₀ ↔ X₁ (shared hidden U)
DAG: X₀ → X₂, X₁ → X₂

Components (topological order):
  Component 0: {X₀, X₁} (CONFOUNDED, no parents)
  Component 1: {X₂} (NON-CONFOUNDED, parents={X₀,X₁})
```

**Coalition S = {X₀}**:

Sample from P(X | do(X₀ = x₀)):

```
Step 1: Fix intervention
  sample[X₀] = x[X₀] = 5

Step 2: Process Component 0 = {X₀, X₁}
  - fixed_features = {X₀}
  - missing_features = {X₁}
  - parents = ∅
  - CONFOUNDED = True
  
  → Sample X₁ INDEPENDENTLY (no conditioning):
    sample[X₁] ~ N(μ[X₁], Σ[X₁,X₁])
    // X₁ is independent of X₀ after intervention!

Step 3: Process Component 1 = {X₂}
  - fixed_features = ∅
  - missing_features = {X₂}
  - parents = {X₀, X₁}
  - CONFOUNDED = False
  
  → Sample X₂ given parents (already determined):
    sample[X₂] ~ P(X₂ | X₀=5, X₁=sample[X₁])

Result: sample = [5, sample[X₁], sample[X₂]]
```

**Coalition S = {X₀, X₁}**:

```
Step 1: Fix interventions
  sample[X₀] = x[X₀] = 5
  sample[X₁] = x[X₁] = 10

Step 2: Process Component 0 = {X₀, X₁}
  - All features fixed, SKIP

Step 3: Process Component 1 = {X₂}
  - missing_features = {X₂}
  - parents = {X₀, X₁}
  
  → Sample X₂ given parents:
    sample[X₂] ~ P(X₂ | X₀=5, X₁=10)

Result: sample = [5, 10, sample[X₂]]
```

---

## 4. Shapley Flow (ShapleyFlow)

### Overview

Graph-based approach that attributes importance to **edges** rather than features. Uses on-manifold perturbation with conditional sampling.

### Key Ideas

1. **Edge-level explanations**: How important is edge Xᵢ → Xⱼ?
2. **On-manifold**: All perturbed samples respect data distribution
3. **Conditional sampling**: Missing nodes sampled from P(node | active_parents)

### Edge Activation Model

For a set of active edges E:
- Node is **observed** if it has ≥1 active incoming edge (or is a source with active outgoing)
- Node is **missing** if it has no active edges
- Missing nodes are sampled conditionally

### Pseudo Code: Shapley Flow (Exhaustive DFS)

```
ALGORITHM: ComputeShapleyFlow_ExhaustiveDFS(x_foreground, x_background, 
                                             graph, n_samples)

INPUT:
  - x_foreground: instance to explain (dict: node → value)
  - x_background: background values (dict: node → value)
  - graph: adjacency list (dict: node → [children])
  - source_nodes: nodes with no parents
  - n_samples: number of Monte Carlo trials

OUTPUT:
  - edge_attributions: dict mapping (u,v) → importance score

PROCEDURE:

1. Initialize:
   edge_attributions = {(u,v): 0.0 for all edges}

2. FOR trial = 1 to n_samples:

   a. FOR each source_node in source_nodes:
      
      DFS_RecursiveShapley(source_node, history=[], 
                          x_foreground, x_background, 
                          graph, edge_attributions)

3. Average over trials:
   FOR each edge in edge_attributions:
      edge_attributions[edge] /= n_samples

4. RETURN edge_attributions
```

### Pseudo Code: Recursive DFS

```
FUNCTION: DFS_RecursiveShapley(node, history, x_foreground, x_background, 
                               graph, edge_attributions)

INPUT:
  - node: current node in DFS
  - history: list of active edges so far
  - x_foreground, x_background: foreground and background values
  - graph: adjacency list
  - edge_attributions: accumulated edge scores (modified in-place)

PROCEDURE:

1. IF node is sink:
   RETURN  // Base case

2. Get children of current node:
   children = graph[node]
   
   IF children is empty:
      RETURN  // Leaf node

3. *** KEY STEP: Random permutation of children ***
   children_permuted = random_shuffle(children)

4. Compute current value:
   v_current = EvaluateSystem(history, x_foreground, x_background)

5. FOR each child in children_permuted:

   a. Create new edge:
      edge = (node, child)
      new_history = history + [edge]
   
   b. Evaluate with new edge:
      v_after = EvaluateSystem(new_history, x_foreground, x_background)
   
   c. Marginal contribution:
      marginal = v_after - v_current
   
   d. Accumulate to edge attribution:
      edge_attributions[edge] += marginal
   
   e. Update current value for next iteration:
      v_current = v_after
   
   f. Recurse to child:
      DFS_RecursiveShapley(child, new_history, x_foreground, x_background,
                          graph, edge_attributions)
```

### Pseudo Code: System Evaluation

```
FUNCTION: EvaluateSystem(active_edges, x_foreground, x_background)

INPUT:
  - active_edges: set of edges currently active
  - x_foreground: instance values (dict)
  - x_background: background values (dict)

OUTPUT:
  - prediction: model prediction given active edges

PROCEDURE:

1. Determine node values based on edge activation:
   
   node_values = {}
   
   FOR each node in graph:
      
      IF node is source_node:
         // Check if node has active outgoing edge
         has_active_outgoing = any((node, child) in active_edges 
                                    for child in graph[node])
         
         IF has_active_outgoing:
            node_values[node] = x_foreground[node]
         ELSE:
            node_values[node] = x_background[node]
      
      ELSE:
         // Check if node has active incoming edge
         has_active_incoming = any((parent, node) in active_edges
                                    for parent in parents[node])
         
         IF has_active_incoming:
            node_values[node] = x_foreground[node]
         ELSE:
            // Node is "missing" - sample from conditional distribution
            node_values[node] = x_background[node]  
            // In practice: could use P(node | active_parents)

2. Construct feature vector (exclude sink/Y node):
   features = [node_values[i] for i in feature_indices]

3. Predict:
   prediction = model.predict(features)

4. RETURN prediction
```

### Pseudo Code: Path Sampling (Faster Alternative)

```
ALGORITHM: ComputeShapleyFlow_PathSampling(x_foreground, x_background, 
                                           graph, n_samples, K_paths)

INPUT:
  - x_foreground, x_background: foreground/background values
  - graph: adjacency list
  - n_samples: number of trials
  - K_paths: number of random paths per source

OUTPUT:
  - edge_attributions: edge importance scores

PROCEDURE:

1. Initialize:
   edge_attributions = {edge: 0.0 for all edges}

2. FOR trial = 1 to n_samples:

   a. FOR each source in source_nodes:
      
      FOR k = 1 to K_paths:
         
         i. Sample random path:
            path_edges = SampleRandomPath(source, sink, graph)
            // Random walk: at each node, randomly pick one child
         
         ii. Evaluate path contribution:
            marginals = EvaluatePathContribution(path_edges, 
                                                 x_foreground, 
                                                 x_background)
            // Random permutation within path + incremental evaluation
         
         iii. Accumulate:
            FOR edge, marginal in marginals:
               edge_attributions[edge] += marginal

3. Average:
   total_paths = n_samples * len(source_nodes) * K_paths
   FOR each edge:
      edge_attributions[edge] /= total_paths

4. RETURN edge_attributions
```

### Example Walkthrough

**Graph**:
```
X₀ → X₂
X₁ → X₂
X₂ → Y

Sources: {X₀, X₁}
Sink: Y
```

**Trial 1: DFS from X₀**

```
Start at X₀, history = []

Step 1: At X₀
  - Children: [X₂]
  - Current value: v₀ = f(x_bg[X₀], x_bg[X₁], x_bg[X₂])
  - Add edge X₀→X₂:
    * history = [(X₀,X₂)]
    * v₁ = f(x_fg[X₀], x_bg[X₁], x_fg[X₂])
    * marginal = v₁ - v₀
    * attribution[X₀→X₂] += marginal
  - Recurse to X₂

Step 2: At X₂, history = [(X₀,X₂)]
  - Children: [Y]
  - Current value: v₁
  - Add edge X₂→Y:
    * history = [(X₀,X₂), (X₂,Y)]
    * v₂ = model.predict(x_fg)  // All features active
    * marginal = v₂ - v₁
    * attribution[X₂→Y] += marginal
  - Recurse to Y (sink, done)
```

**Trial 1: DFS from X₁**

```
Start at X₁, history = []

Step 1: At X₁
  - Children: [X₂]
  - Add edge X₁→X₂:
    * attribution[X₁→X₂] += (marginal)
  - Recurse to X₂

Step 2: At X₂ (same as above)
  - Add edge X₂→Y
  - attribution[X₂→Y] += (marginal)
```

After many trials with random child orderings, edge attributions converge.

### Aggregating to Node Importance

```
node_importance[X_i] = Σ_{(X_i, v) ∈ edges} edge_attribution[(X_i, v)]
```

Sum of outgoing edge attributions = feature importance.

---

## 5. Comparison Table

### Summary of Methods

| Method | Formula | Causal DAG Usage | Handles Confounding | Complexity |
|--------|---------|------------------|---------------------|------------|
| **Vanilla Shapley** | E[f(X) \| X_S=x_S] | None | ❌ No | O(n²·m) per instance |
| **Asymmetric Shapley** | E[f(X) \| X_S=x_S] with valid S | Constrains coalitions | ❌ No | O(n²·m) per instance |
| **Causal Shapley** | E[f(X) \| do(X_S=x_S)] | Components + topo order | ✅ Yes | O(n²·M·m) per instance |
| **Shapley Flow** | Edge-based v(E) | Graph structure, sources, sink | Partial (on-manifold) | O(E·k·m) per instance |

Where:
- n = number of features
- m = background data size
- M = inner samples (Causal Shapley)
- E = number of edges
- k = number of trials

### When to Use Each Method

| Method | Best For | Limitations |
|--------|----------|-------------|
| **Vanilla** | Tabular data, no known causal structure | Doesn't respect causality |
| **Asymmetric** | Known DAG, no confounding | Doesn't handle hidden confounders |
| **Causal** | Known DAG with confounding | Requires causal discovery, slower |
| **Shapley Flow** | Interpretable causal paths, fine-grained explanations | Needs complete causal graph including Y |

### Background Data Usage Comparison

| Method | How Background is Used |
|--------|------------------------|
| **Vanilla** | Sample missing features uniformly from background |
| **Asymmetric** | Same as Vanilla, but constrains which features can be missing |
| **Causal** | Estimate conditional distributions P(X\|Parents) from background |
| **Shapley Flow** | Provide baseline values for inactive nodes |

---

## Additional Notes

### Computational Trade-offs

1. **Exact vs Approximate**:
   - Exact: O(2ⁿ) coalitions, only feasible for n ≤ 10
   - Monte Carlo: O(n_samples) permutations, works for any n

2. **Inner Sampling** (Causal Shapley):
   - Trades accuracy for computational cost
   - M_inner = 50-100 typically sufficient
   - Higher M_inner → more accurate do-calculus approximation

3. **Path Sampling** (Shapley Flow):
   - Dense graphs: use path sampling
   - Sparse graphs: exhaustive DFS may be feasible

### Implementation Details

1. **Feature Alignment**:
   - Models may use subset of features (regularization, feature selection)
   - Always convert arrays to DataFrames with proper column names
   - Let model.predict() handle feature selection

2. **Numerical Stability**:
   - Add regularization to covariance matrices (±1e-6)
   - Use pseudo-inverse for singular matrices
   - Fall back to marginal distributions on failure

3. **Progress Tracking**:
   - Log every 10% of instances
   - Track evaluation counts for debugging
   - Set max limits to prevent infinite loops

---

## References

1. **Shapley, L. S. (1953)**. "A value for n-person games." Contributions to the Theory of Games.

2. **Lundberg, S. M., & Lee, S. I. (2017)**. "A unified approach to interpreting model predictions." NeurIPS.

3. **Frye, C., et al. (2021)**. "Asymmetric Shapley values: incorporating causal knowledge into model-agnostic explainability." NeurIPS.

4. **Heskes, T., et al. (2020)**. "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models." NeurIPS.

5. **Wang, J., & Venkatasubramanian, S. (2021)**. "Shapley Flow: A Graph-based Approach to Interpreting Model Predictions." AISTATS.

---

*End of Pseudo Code Documentation*
