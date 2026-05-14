# Shapley Methods: Documentation & Pseudocode

## Overview

This document describes all four Shapley value implementations used in the
experiment: what each method computes, how the causal graph is used, and the
complete algorithm with pseudocode.

---

## Table of Contents

1. [Vanilla Shapley (ShapleyFromScratch)](#1-vanilla-shapley-shapleyfromscratch)
2. [Asymmetric Shapley (AsymmetricShapley)](#2-asymmetric-shapley-asymmetricshapley)
3. [Causal Shapley (CausalShapley)](#3-causal-shapley-causalshapley)
4. [Shapley Flow (ShapleyFlowWrapper)](#4-shapley-flow-shapleyflowwrapper)
5. [Comparison Summary](#5-comparison-summary)

---

## 1. Vanilla Shapley (ShapleyFromScratch)

### What It Does

Computes standard Shapley values with no causal structure.  All features are
treated symmetrically; credit is distributed based purely on marginal
contributions across all possible coalitions (approximated via Monte Carlo
permutation sampling).

### Game Theory Foundation

The Shapley value of feature $i$:

$$\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!}\,\bigl[v(S \cup \{i\}) - v(S)\bigr]$$

| Symbol | Meaning |
|--------|---------|
| $N$ | Set of all features |
| $S$ | Coalition (subset of features not including $i$) |
| $v(S) = \mathbb{E}[f(X) \mid X_S = x_S]$ | Coalition value (marginalise non-$S$ features over background data) |

**Properties:** Efficiency ($\sum_i \phi_i = f(x) - \mathbb{E}[f]$), Symmetry,
Dummy, Additivity.

### Coalition Value Computation

```
v(S):
    FOR each row in background_data:
        Replace features in S with instance values x_S
        Keep non-S features at background row values
    v(S) = mean of model predictions over all modified rows
```

### Monte Carlo Permutation Estimator

Instead of enumerating all $2^n$ coalitions, sample $T$ random permutations:

$$\hat{\phi}_i = \frac{1}{T} \sum_{t=1}^{T} \bigl[v(S_\pi^i \cup \{i\}) - v(S_\pi^i)\bigr]$$

where $S_\pi^i$ = features appearing before $i$ in permutation $\pi_t$.

### Pseudocode

```
INPUT : instance x, model f, background_data D, n_samples T
OUTPUT: φ ∈ ℝⁿ

φ ← zeros(n)
baseline ← mean(f(D))

FOR t = 1 … T:
    π ← random_permutation(0 … n-1)
    S ← ∅,  v_prev ← baseline
    FOR i in π:
        S ← S ∪ {i}
        samples ← D.copy()
        FOR j in S: samples[:, j] ← x[j]
        v_curr ← mean(f(samples))
        φ[i] += v_curr − v_prev
        v_prev ← v_curr

RETURN φ / T
```

### Causal Diagram Usage

**None.** The causal graph is ignored entirely.

---

## 2. Asymmetric Shapley (AsymmetricShapley)

### What It Does

Computes Shapley values where permutations are drawn uniformly from the set of
all valid **topological orderings** of the X-feature DAG (its linear
extensions), following Frye et al. 2021.  All features appear in every
permutation; no feature receives a structural zero attribution.

The coalition value $v(S)$ uses the same **observational marginalization** as
Vanilla Shapley — the only difference is the restricted permutation space.

### Game Theory Foundation

$$\phi_i = \mathbb{E}_{\pi \in \Pi_\text{topo}} \bigl[v(S_\pi^i \cup \{i\}) - v(S_\pi^i)\bigr]$$

where $\Pi_\text{topo}$ is the set of all linear extensions of the X-feature
DAG (topological orderings that respect every ancestor–descendant pair).

### Topological Ordering Sampler (Kahn-style, O(n) per sample)

```
INIT (called once in __init__):
    Build _x_children[i]   = children of i in X-only subgraph (Y excluded)
    Build _x_in_degree[i]  = number of X-feature parents of i
    If DAG has cycles → zero out directed_graph (fall back to unconstrained)

SAMPLE_TOPOLOGICAL_ORDERING():
    remaining_in_degree ← copy(_x_in_degree)
    ready ← [i where remaining_in_degree[i] == 0]   # sources
    π ← []
    WHILE ready not empty:
        idx ← randint(len(ready))
        node ← ready[idx];  ready[idx] ← ready[-1];  ready.pop()   # swap-remove O(1)
        π.append(node)
        FOR child in _x_children[node]:
            remaining_in_degree[child] -= 1
            IF remaining_in_degree[child] == 0: ready.append(child)
    # Safety: append any unplaced nodes (cycle fallback)
    RETURN π
```

### Pseudocode

```
INPUT : instance x, model f, background_data D,
        causal_graph G (shape n+1 × n+1, last index = Y), n_samples T
OUTPUT: φ ∈ ℝⁿ

INIT: precompute _x_children, _x_in_degree from G[:n, :n]

φ ← zeros(n),  baseline ← mean(f(D))

FOR t = 1 … T:
    π ← SAMPLE_TOPOLOGICAL_ORDERING()
    S ← ∅,  v_prev ← baseline
    FOR i in π:
        S ← S ∪ {i}
        samples ← D.copy()
        FOR j in S: samples[:, j] ← x[j]
        v_curr ← mean(f(samples))
        φ[i] += v_curr − v_prev
        v_prev ← v_curr

RETURN φ / T
```

### Causal Diagram Usage

The causal graph (shape $n+1 \times n+1$, last node = Y) is used exclusively
to build the **X-only topological ordering structures** at initialisation.  The
Y column/row is ignored.  No adjacency filtering is performed; the full X DAG
(including nodes that do not connect to Y) constrains the permutation.

---

## 3. Causal Shapley (CausalShapley)

### What It Does

Computes **interventional** Shapley values using do-calculus (Heskes et al.
2020).  The coalition value replaces observational conditioning with an
intervention:

$$v_\text{do}(S) = \mathbb{E}\bigl[f(X) \mid \mathrm{do}(X_S = x_S)\bigr]$$

The $\mathrm{do}(\cdot)$ operator cuts all incoming edges to variables in $S$,
removing confounding effects and isolating direct causal contributions.

### Outer Loop: Random Component Topological Ordering

Features are grouped into **components** before sampling permutations:

* Each confounder pair $(X_i, X_j)$ (a bidirected edge) forms a joint component.
* All remaining features are singleton components.
* A random topological ordering is drawn over the **component DAG** using the
  same Kahn-style sampler as Asymmetric Shapley.
* Within a confounded (multi-feature) component, features are shuffled randomly.
* The result is a flat feature list respecting the causal partial order.

### Inner Loop: Post-Interventional Sampling

For each coalition $S$, $v_\text{do}(S)$ is estimated by sampling $M$
post-interventional draws and averaging model predictions:

```
FOR m = 1 … M:
    sample ← zeros(n)
    FOR j in S: sample[j] ← x[j]       # fix interventions

    FOR each component C in deterministic topological order:
        fixed_C   ← C ∩ S
        missing_C ← C \ S
        IF confounded_info[C]:
            # Intervention breaks dependencies → sample each missing feature independently
            FOR j in missing_C:
                sample[j] ~ Gaussian_conditional(
                    target=j, cond=parents(j), vals=sample[parents(j)])
        ELSE:
            # Preserve within-component dependencies → joint conditional
            sample[missing_C] ~ Gaussian_conditional(
                target=missing_C,
                cond=parents_dict[C] ∪ fixed_C,
                vals=[sample[parents_dict[C]], sample[fixed_C]])

    v_curr += f(sample)

v_do(S) = v_curr / M
```

Conditional Gaussian sampling uses the closed-form formula:

$$\mu_{A|B} = \mu_A + \Sigma_{AB}\,\Sigma_{BB}^{-1}(x_B - \mu_B), \qquad
\Sigma_{A|B} = \Sigma_{AA} - \Sigma_{AB}\,\Sigma_{BB}^{-1}\,\Sigma_{BA}$$

with $\mu$ and $\Sigma$ estimated from the background data.

### Pseudocode

```
INPUT : instance x, model f, background_data D,
        discovered_adj (n×n X-only adjacency),
        discovered_conf (list of confounder pairs),
        n_samples T, M_inner_samples M
OUTPUT: φ ∈ ℝⁿ

INIT:
    components, confounded_info, parents_dict ←
        extract_causal_structure(discovered_adj, discovered_conf)
    μ ← mean(D),  Σ ← cov(D)
    precompute _comp_children, _comp_in_degree for component DAG

φ ← zeros(n),  baseline ← mean(f(D))

FOR t = 1 … T:
    comp_order ← SAMPLE_COMPONENT_TOPOLOGICAL_ORDERING()
    π ← expand_components_to_features(comp_order)   # shuffle within confounded comps
    S ← ∅,  v_prev ← baseline

    FOR i in π:
        S ← S ∪ {i}
        v_curr ← 0
        FOR m = 1 … M:
            sample ← POST_INTERVENTIONAL_SAMPLE(
                x, S, components, confounded_info, parents_dict, μ, Σ)
            v_curr += f(sample)
        v_curr /= M
        φ[i] += v_curr − v_prev
        v_prev ← v_curr

RETURN φ / T
```

**Computational cost:** $O(T \times n \times M \times n)$ — significantly more
expensive than Vanilla / Asymmetric due to the inner sampling loop.  The
pipeline uses $T = 100$ outer samples and $M = 10$ inner samples.

### Causal Diagram Usage

* **discovered_adj** ($n \times n$, X only) — defines parent relationships for
  post-interventional sampling and the component DAG topology.
* **discovered_conf** — list of confounder pairs; each pair forms a joint
  component where the $\mathrm{do}$ operator breaks within-component
  dependencies.
* No adjacency filtering is applied; the full X graph is used.

---

## 4. Shapley Flow (ShapleyFlowWrapper)

### What It Does

Computes **edge-level** Shapley attributions instead of feature-level ones
(Wang & Joshi 2021).  The game players are *edges* in the causal graph; the
edge credits are then summed over outgoing edges per node to produce feature
importance scores.

$$\phi_{i \to j} = \mathbb{E}_{\pi \in \Pi_\text{edges}}\bigl[v(E_\pi^{i\to j} \cup \{i\to j\}) - v(E_\pi^{i\to j})\bigr]$$

Node attribution: $\displaystyle\phi_i = \sum_{j:\, i\to j \text{ exists}} \phi_{i\to j}$

### Source Filtering (Wrapper)

`ShapleyFlowWrapper.__init__` filters source nodes to those that can reach Y
via a backward BFS:

```
parents[i] ← [j where adj[j, i] ≠ 0]
potential_sources ← [i where len(parents[i]) == 0]

reachable_from_Y ← backward_BFS(start=Y, parents=parents)
source_nodes ← potential_sources ∩ reachable_from_Y
```

The adjacency matrix itself is **not** filtered; all edges are passed to the
core `ShapleyFlow` class unchanged.

**Note — no-op in this pipeline.** The graph-augmentation rules in Step 4
add X→Y edges for every feature in `model.selected_features` (all 50, because
`feature_selection=False`) plus every sink node.  As a result, every X feature
has a direct edge to Y, so the backward BFS from Y reaches every X feature in
a single step.  The intersection `potential_sources ∩ reachable_from_Y` always
equals `potential_sources` and no source is ever excluded.

### MC Edge Permutation (Default Mode, `use_mc_permutation=True`)

Every trial draws a **uniform random permutation of all edges** in the graph
and evaluates them incrementally.  No path sampling of any kind occurs.

```
FOR t = 1 … T:
    perm ← random_permutation(all_edges)
    v_prev ← evaluate_system(∅, x_fg, x_bg)
    history ← []
    FOR edge in perm:
        history.append(edge)
        v_curr ← evaluate_system(history, x_fg, x_bg)
        edge_attributions[edge] += v_curr − v_prev
        v_prev ← v_curr

edge_attributions /= T
```

Why unconstrained ordering (not causal-depth order): if parent edges always
precede child edges, a non-source node $X_i$ is already at foreground by the
time $(X_i \to Y)$ fires, making its marginal ≈ 0.  The unconstrained random
ordering ensures every edge, including direct X→Y edges, gets a non-zero
unbiased estimate.  Cost: $T \times (|E|+1)$ model calls (e.g. $50 \times 276
= 13\,800$).

### System Evaluation

For a set of active edges $E$, each node's value is determined by whether it
has an active incoming edge (or, for source nodes, an active outgoing edge):

```
evaluate_system(active_edges, x_fg, x_bg):
    FOR each node i:
        IF source node AND has active outgoing edge:
            node_values[i] ← x_fg[i]
        ELSE IF has active incoming edge:
            node_values[i] ← x_fg[i]
        ELSE:
            node_values[i] ← x_bg[i]

    # Y (sink_node) is always stripped before model.predict()
    x_features ← node_values excluding Y
    RETURN model.predict(x_features)
```

### Pseudocode

```
INPUT : x_fg (instance), x_bg (background row), adjacency (n+1 × n+1),
        sink_node (Y index), n_samples T
OUTPUT: node_attributions ∈ ℝⁿ

all_edges ← [(u,v) for every non-zero entry in adjacency]
edge_attributions ← {e: 0.0  for e in all_edges}

FOR t = 1 … T:
    perm ← random_permutation(all_edges)
    v_prev ← evaluate_system(∅, x_fg, x_bg)
    history ← []
    FOR edge in perm:
        history.append(edge)
        v_curr ← evaluate_system(history, x_fg, x_bg)
        edge_attributions[edge] += v_curr − v_prev
        v_prev ← v_curr

edge_attributions /= T

# Aggregate to features
FOR i = 0 … n-1:
    node_attributions[i] ← Σ_{j: (i,j) ∈ edges} edge_attributions[(i,j)]

RETURN node_attributions
```

### Causal Diagram Usage

The full $(n+1) \times (n+1)$ adjacency (including Y as the last node) is
passed to `ShapleyFlowWrapper`.  The wrapper uses it to:

1. Identify source nodes (no incoming edges) and check reachability to Y via
   backward BFS — this check is a no-op in the current pipeline (see note above).
2. Build a children dictionary (`graph_structure`) for the core `ShapleyFlow`
   object.

No edges are removed from the adjacency.

---

## 5. Comparison Summary

| Method | Causal graph role | Coalition players | Marginalization | Relative cost |
|--------|-------------------|-------------------|-----------------|---------------|
| **Vanilla Shapley** | Not used | All features, uniform random order | Observational (background data) | Low |
| **Asymmetric Shapley** | Constrains permutations to topological orderings of X DAG | All features | Observational (background data) | Low |
| **Causal Shapley** | Defines component structure + outer permutation + inner sampling | All features | Interventional (post-interventional Gaussian) | High |
| **Shapley Flow** | Defines edge players; BFS source check (no-op when all X→Y) | Edges → aggregated to features | Foreground/background by edge activation | Medium |

### Permutation Space

| Method | Permutation space | Features per permutation |
|--------|-------------------|--------------------------|
| Vanilla | Uniform over all $n!$ orderings | All $n$ |
| Asymmetric | Uniform over linear extensions of X DAG | All $n$ |
| Causal | Uniform over topological orderings of component DAG | All $n$ |
| Shapley Flow | Uniform random permutation of all edges (MC) | All edges |

### Causal Graph Format Per Method

| Method | Input adjacency | Shape | Y included? |
|--------|----------------|-------|-------------|
| Asymmetric Shapley | `causal_graph` | $(n+1) \times (n+1)$ | Yes (last index), ignored in ordering |
| Causal Shapley | `discovered_adj` | $n \times n$ (X only) | No |
| Shapley Flow | `causal_graph` | $(n+1) \times (n+1)$ | Yes (sink node = Y) |

---

## Appendix: Key Shared Utilities

### Backward BFS

Used by `ShapleyFlowWrapper` to identify source nodes reachable from Y
(a no-op in this pipeline — see §4 note):

```python
def backward_BFS(sink_idx, parents):
    reachable, visited = set(), {sink_idx}
    queue = [sink_idx]
    while queue:
        current = queue.pop(0)
        reachable.add(current)
        for parent in parents[current]:
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)
    return reachable
```

### Conditional Gaussian Sampling (CausalShapley)

```python
def sample_conditional_gaussian(target_idx, cond_idx, cond_values, μ, Σ):
    Σ_AA = Σ[np.ix_(target_idx, target_idx)]
    Σ_BB = Σ[np.ix_(cond_idx, cond_idx)]
    Σ_AB = Σ[np.ix_(target_idx, cond_idx)]
    μ_cond = μ[target_idx] + Σ_AB @ np.linalg.inv(Σ_BB) @ (cond_values - μ[cond_idx])
    Σ_cond = Σ_AA - Σ_AB @ np.linalg.inv(Σ_BB) @ Σ_AB.T
    return np.random.multivariate_normal(μ_cond, Σ_cond)
```

### Kahn-Style Uniform Topological Sort (Asymmetric & Causal Shapley)

```python
def sample_topological_ordering(children, in_degree):
    remaining = in_degree.copy()
    ready = [i for i, d in enumerate(remaining) if d == 0]
    order = []
    while ready:
        idx = random.randrange(len(ready))
        node = ready[idx]; ready[idx] = ready[-1]; ready.pop()  # O(1) swap-remove
        order.append(node)
        for child in children[node]:
            remaining[child] -= 1
            if remaining[child] == 0:
                ready.append(child)
    return order
```
