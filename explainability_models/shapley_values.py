import logging

import numpy as np
import math
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Any
from itertools import chain, combinations, permutations
import warnings
# warnings.filterwarnings('ignore')

import shap

from sklearn.base import BaseEstimator
import networkx as nx

    
class ShapleyFromScratch:
    """Vanilla Shapley value computation from first principles.
    
    Implements Shapley values without using external libraries, following the
    original game-theoretic definition from Lloyd Shapley (1953).
    
    SHAPLEY VALUE DEFINITION:
    For feature i, the Shapley value is the weighted average of its marginal
    contributions across all possible coalitions (subsets) of other features:
    
    φᵢ(f) = Σ_S⊆N\{i} [|S|!(|N|-|S|-1)! / |N|!] * [f(S∪{i}) - f(S)]
    
    where:
    - N = set of all features
    - S = coalition (subset) of features not including i
    - f(S) = expected model output when only features in S are observed
    - The weight term ensures fair credit distribution
    
    BACKGROUND DATA USAGE:
    Background data is used to marginalize over (fill in) missing features:
    - Features IN coalition S: Use instance's actual values
    - Features NOT in coalition S: Sample from background data
    - Average predictions over all background samples
    
    COMPUTATION METHODS:
    1. Exact: Enumerate all 2^n coalitions (exponential, only feasible for n≤10)
    2. Monte Carlo: Sample random permutations and compute marginal contributions
    
    SAMPLING FOR EACH INSTANCE:
    For each instance x:
    1. Sample random permutation π of features: (f_π(1), f_π(2), ..., f_π(n))
    2. Build coalitions incrementally: ∅ → {f_π(1)} → {f_π(1), f_π(2)} → ...
    3. For each feature i, compute marginal: v(S∪{i}) - v(S) where S = features before i in π
    4. Average marginal contributions over many permutations
    
    Parameters
    ----------
    model : BaseEstimator
        Trained model to explain
    background_data : pd.DataFrame
        Reference dataset for marginalizing over missing features
        Should be representative of the data distribution (typically training data sample)
    n_samples : int, default=1000
        Number of random permutations for Monte Carlo approximation
    random_state : int or None
        Random seed for reproducibility
    
    Attributes
    ----------
    baseline_value : float
        Expected prediction over background data (serves as reference point)
    shap_values : np.ndarray or None
        Computed Shapley values after calling explain()
    
    Examples
    --------
    >>> explainer = ShapleyFromScratch(model, X_train.sample(100), n_samples=500)
    >>> shap_values = explainer.explain(X_test, method='monte_carlo')
    """

    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 n_samples: int = 1000, random_state: Optional[int] = None ):
        
        self.model = model
        self.background_data = background_data.values
        self.background_data_df = background_data  # Keep DataFrame for feature alignment
        self.feature_names = background_data.columns.tolist()
        self.n_features  = len(self.feature_names)
        self.n_samples = n_samples
        self.random_state = random_state
        self.shap_values = None

        self.rng = np.random.RandomState(random_state)

        self.baseline_value = self._predict_with_feature_alignment(self.background_data_df).mean()
    
    def _predict_with_feature_alignment(self, X):
        """
        Predict with proper feature alignment.
        
        If X is numpy array, convert to DataFrame with correct column names.
        The model's predict() method will then handle feature selection.
        
        Parameters:
        -----------
        X : np.ndarray or pd.DataFrame
            Input samples
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        if isinstance(X, np.ndarray):
            # Convert numpy array to DataFrame with correct column names
            if X.ndim == 1:
                X_df = pd.DataFrame([X], columns=self.feature_names)
            else:
                X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        # Model's predict() will handle feature selection if needed
        return self.model.predict(X_df)

    def _predict_coalition(self, instance: np.ndarray, coalition: List[int]) -> float:
        """Compute coalition value v(S) by marginalizing over missing features.
        
        Core operation for Shapley value computation:
        - Features IN coalition: Use instance's values (foreground)
        - Features NOT in coalition: Sample from background data
        - Return: Average of model predictions over background samples
        
        This implements the conditional expectation:
        v(S) = E[f(x) | X_S = x_S] where X_S are features in coalition S
        
        Parameters
        ----------
        instance : np.ndarray
            Instance to explain (1D array of shape n_features)
        coalition : List[int]
            Indices of features in the coalition
        
        Returns
        -------
        value : float
            Expected prediction when only coalition features are observed
            v(S) = (1/M) Σ_m f(x_S, X_{-S}^(m)) where X_{-S}^(m) ~ background
        """

        # Create samples with coalition features from instance, others from background
        samples = self.background_data.copy()

        for feature_idx in coalition:
            samples[:, feature_idx] = instance[feature_idx]

        # Average predictions with proper feature alignment
        predictions = self._predict_with_feature_alignment(samples)
        return predictions.mean()
    
    def _compute_exact_shapley(self, instance: np.ndarray) -> np.ndarray:
        """Compute exact Shapley values by enumerating all coalitions.
        
        ALGORITHM:
        For each feature i:
            1. Consider all subsets S of other features (there are 2^(n-1) subsets)
            2. For each subset S:
                a. Compute v(S∪{i}) - v(S) = marginal contribution of feature i
                b. Weight by: |S|!(n-|S|-1)! / n!
                c. Accumulate weighted marginal
            3. Sum all weighted marginals to get φᵢ
        
        COMPLEXITY:
        - Time: O(n * 2^n * M) where M = background data size
        - Space: O(2^n)
        - Only feasible for n ≤ 10 features
        
        Parameters
        ----------
        instance : np.ndarray
            Instance to explain
        
        Returns
        -------
        shapley_values : np.ndarray
            Exact Shapley values (shape: n_features)
        
        Warnings
        --------
        - Extremely slow for more than 10 features
        - Automatically falls back to Monte Carlo if n > 10
        """
        shapley_values = np.zeros(self.n_features)

        for i in range(self.n_features):

            other_features = [j for j in range(self.n_features) if j!=i]

            marginal_contributions = []

            # All subsets of other features
            for coalition_size in range(self.n_features):
                for coalition in combinations(other_features, coalition_size):
                    coalition = list(coalition)

                    coalition_with_i = coalition + [i]
                    v_with = self._predict_coalition(instance, coalition_with_i)

                    v_without = self._predict_coalition(instance, coalition)

                    marginal = v_with - v_without

                    weight = 1.0/ (self.n_features * math.comb(self.n_features -1, coalition_size))

                    marginal_contributions.append(weight * marginal)
            
            shapley_values[i] =sum(marginal_contributions)

        return shapley_values
    
    def _compute_monte_carlo_shapley(self, instance: np.ndarray) -> np.ndarray:
        """Compute approximate Shapley values using Monte Carlo sampling.
        
        PERMUTATION-BASED ALGORITHM:
        Repeat n_samples times:
            1. Sample random permutation π of all features
            2. Initialize: S = ∅, v_prev = baseline
            3. For each feature f_i in order π:
                a. Add f_i to coalition: S = S ∪ {f_i}
                b. Compute: v_curr = v(S)
                c. Marginal contribution: Δᵢ = v_curr - v_prev
                d. Accumulate: φᵢ += Δᵢ
                e. Update: v_prev = v_curr
        Return: φ / n_samples (average over all permutations)
        
        WHY THIS WORKS:
        The Shapley value can be written as an expectation over random permutations:
        φᵢ = E_π[f(S_π^i ∪ {i}) - f(S_π^i)]
        where S_π^i = features appearing before i in permutation π
        
        COMPLEXITY:
        - Time: O(n_samples * n * M) where M = background data size
        - Much faster than exact method for n > 10
        - Convergence: error decreases as O(1/√n_samples)
        
        Parameters
        ----------
        instance : np.ndarray
            Instance to explain
        
        Returns
        -------
        shapley_values : np.ndarray
            Approximate Shapley values (shape: n_features)
        """
        shapley_values = np.zeros(self.n_features)

        # Sample random permutations
        for sample_idx in range(self.n_samples):
            # Random permutation of features
            perm = self.rng.permutation(self.n_features)

            # Track value as we add features

            prev_value = self.baseline_value
            coalition = []

            for feature_idx in perm:
                # add feature to coalition
                coalition.append(feature_idx)

                curr_value = self._predict_coalition(instance,coalition)

                marginal = curr_value - prev_value

                shapley_values[feature_idx] += marginal

                prev_value = curr_value
            
            # Log progress every 100 samples
            if (sample_idx + 1) % 100 == 0:
                logging.info(f"    Sample {sample_idx + 1}/{self.n_samples} completed")
        
        # Average over samples 
        shapley_values /= self.n_samples

        return shapley_values
    
    def explain(self, X: pd.DataFrame, method: str = 'monte_carlo') -> np.ndarray:
        """Calculate Shapley values for given instances.
        
        Computes Shapley values that decompose each prediction as:
        f(x) = baseline + Σᵢ φᵢ(x)
        
        where φᵢ(x) represents the contribution of feature i.
        
        Parameters
        ----------
        X : pd.DataFrame
            Instances to explain (shape: n_samples x n_features)
        method : str, default='monte_carlo'
            Computation method:
            - 'exact': Enumerate all coalitions (slow, only for n_features ≤ 10)
            - 'monte_carlo': Random permutation sampling (fast, recommended)
        
        Returns
        -------
        shapley_values : np.ndarray
            Shapley values (shape: n_samples x n_features)
            Row i contains Shapley values for instance i
            Column j contains contributions of feature j
        
        Notes
        -----
        Automatically switches to Monte Carlo if method='exact' but n_features > 10
        """
        print(f"Computing Shapley values from scratch using {method} method...")
        logging.info(f"Starting ShapleyFromScratch computation for {len(X)} instances")
        logging.info(f"Using {self.n_samples} samples per instance")

        X_values = X.values
        n_instances = len(X_values)

        self.shap_values = np.zeros((n_instances, self.n_features))

        for i, instance in enumerate(X_values):
            if method == 'exact':
                if self.n_features > 10:
                    warnings.warn("Exact Shapley calculation with >10 features is very slow. Using Monte Carlo instead.")
                    self.shap_values[i] = self._compute_monte_carlo_shapley(instance)
                else:
                    self.shap_values[i] = self._compute_exact_shapley(instance)
            elif method == 'monte_carlo':
                self.shap_values[i] = self._compute_monte_carlo_shapley(instance)
            else: 
                raise ValueError(f"Unknown method: {method}")
            
            # Log progress every 20 instances
            if (i + 1) % 20 == 0:
                logging.info(f"  Completed {i + 1}/{n_instances} instances")
            
            # Progress indicator removed for cleaner output
            # if (i + 1) % max(1, n_instances //10) == 0:
            #     print(f" Progress: {i + 1}/{n_instances} instances")

        logging.info(f"Completed all {n_instances} instances")
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from Shapley values.
        
        Returns
        -------
        importance : pd.DataFrame
            Mean absolute Shapley values per feature, sorted descending
        
        Raises
        ------
        ValueError
            If explain() has not been called yet
        """
        if self.shap_values is None:
            raise ValueError("Must call expalin( first to compute Shapley values")
        
        importance_scores = np.abs(self.shap_values).mean(axis=0)

        importance = pd.DataFrame({
            "feature": self.feature_names,
            'importance' : importance_scores
        }).sort_values('importance', ascending=False)

        return importance

    def get_shap_values_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get Shapley values as DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame
            Instances (if values not already computed)
        
        Returns
        -------
        shap_df : pd.DataFrame
            Shapley values with feature names and indices
        """
        if self.shap_values is None:
            self.explain(X)

        return pd.DataFrame(self.shap_values, columns=self.feature_names, index= X.abs)

class AsymmetricShapley(ShapleyFromScratch):
    """Asymmetric Shapley values with causal ordering constraints (Frye et al. 2021).
    
    ═══════════════════════════════════════════════════════════════════════════════
    WHAT THIS METHOD ACTUALLY DOES (based on code implementation):
    ═══════════════════════════════════════════════════════════════════════════════
    
    Computes Shapley values using random topological orderings of ALL features,
    sampled uniformly from the set of valid linear extensions of the causal DAG.
    
    KEY BEHAVIORS:
    1. Every permutation is a valid topological ordering of the X-feature DAG
    2. ALL features appear in every permutation — no structural zeros
    3. Causal partial order is respected for every pair (ancestor before descendant)
    4. Y (outcome node) is excluded from the X-only ordering; it is never a player
    
    ═══════════════════════════════════════════════════════════════════════════════
    ALGORITHM (Monte Carlo, Random Topological Ordering):
    ═══════════════════════════════════════════════════════════════════════════════
    
    PREPROCESSING:
        1. Extract directed graph from causal_graph adjacency matrix
        2. Build X-only children list and initial in-degrees (outcome Y excluded)
        3. Detect and disable cycles (DAG required)
    
    FOR trial = 1 to n_samples:
        1. Sample a random valid topological ordering π of all X features:
               - Kahn-style: maintain a pool of ready nodes (in-degree 0)
               - At each step pick uniformly at random from the pool (swap-remove)
               - Decrement in-degrees of children; add newly-ready children to pool
               - This generates a sample from the uniform distribution over all
                 linear extensions of the DAG (Frye et al. 2021)
        2. Compute marginal contributions:
               S ← ∅, v_prev ← baseline
               FOR each feature i in π:
                   S ← S ∪ {i}
                   v_curr ← v(S)  # coalition value via background marginalization
                   φᵢ += v_curr − v_prev
                   v_prev ← v_curr
    
    RETURN φ / n_samples
    
    ═══════════════════════════════════════════════════════════════════════════════
    CAUSAL GRAPH USAGE:
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. **Graph extraction**:
       - Adjacency matrix → binary directed graph (directed_graph)
       - Outcome node Y is always the last index (n_features)
    
    2. **X-only ordering structures** (precomputed in __init__):
       - _x_children[i]: children of node i among X features (excludes Y)
       - _x_in_degree[i]: number of X-feature parents of node i
       - Used by _sample_topological_ordering() at every trial; O(n) per call
    
    3. **NO ADJACENCY FILTERING** — uses the full X-feature subgraph
    
    ═══════════════════════════════════════════════════════════════════════════════
    COALITION & PERMUTATION MECHANICS:
    ═══════════════════════════════════════════════════════════════════════════════
    
    **Coalition Formation:**
    - All features are included in every permutation
    - Coalition value v(S): same as vanilla Shapley
      (condition on S features, marginalize the rest from background)
    
    **Permutation Space:**
    - The set of all valid topological orderings of the X-feature DAG
      (linear extensions of the partial order defined by the causal graph)
    - Strictly larger than path-based orderings and strictly smaller than n!
    - Every feature receives non-zero expected attribution
    
    ═══════════════════════════════════════════════════════════════════════════════
    DIFFERENCE FROM OTHER METHODS:
    ═══════════════════════════════════════════════════════════════════════════════
    
    vs Vanilla Shapley:
    - Vanilla: uniform over all n! permutations (ignores causal order)
    - Asymmetric: uniform over valid topological orderings only

    vs Causal Shapley (Heskes et al.):
    - Causal: uses do-calculus, interventional coalition values
    - Asymmetric: uses observational coalition values (background marginalization)
    
    ═══════════════════════════════════════════════════════════════════════════════
    BACKGROUND DATA USAGE:
    ═══════════════════════════════════════════════════════════════════════════════
    
    Same as vanilla Shapley:
    - Features IN coalition S: use instance's actual values (foreground)
    - Features NOT in coalition S: sample from background data
    - Average predictions over all background samples
    
    No interventional sampling (unlike CausalShapley).
    
    ═══════════════════════════════════════════════════════════════════════════════
    PARAMETERS:
    ═══════════════════════════════════════════════════════════════════════════════
    
    Parameters
    ----------
    model : BaseEstimator
        Trained model to explain
    background_data : pd.DataFrame
        Reference dataset for marginalizing over missing features
    causal_graph : np.ndarray
        Adjacency matrix INCLUDING outcome node Y
        Shape: (n_features+1, n_features+1) where last row/col is outcome Y
        Entry [i,j]=1 means i causes j (binary directed graph)
    n_samples : int, default=1000
        Number of random topological orderings to sample
    random_state : int or None
        Random seed for reproducibility
    
    Attributes
    ----------
    directed_graph : np.ndarray
        Binary directed adjacency matrix (n_features+1 × n_features+1)
    outcome_node : int
        Index of outcome variable Y (typically n_features)
    parents : Dict[int, Set[int]]
        Parent nodes for each node (full graph including Y)
    _x_children : Dict[int, List[int]]
        Children of each X feature within the X-only subgraph (excludes Y)
    _x_in_degree : List[int]
        Initial in-degree of each X feature within the X-only subgraph
    shap_values : np.ndarray or None
        Computed Shapley values after calling explain()
    
    References
    ----------
    Frye, C., Rowat, C., & Feige, I. (2021). "Asymmetric Shapley Values:
    Incorporating Causal Knowledge into Model-Agnostic Explainability."
    NeurIPS 2021.
    
    Examples
    --------
    >>> # Create causal DAG: X0→X1→X2→Y (Y at index 3)
    >>> causal_dag = np.array([[0,1,0,0], 
    ...                         [0,0,1,0], 
    ...                         [0,0,0,1],
    ...                         [0,0,0,0]])
    >>> explainer = AsymmetricShapley(model, X_train, causal_dag)
    >>> shap_values = explainer.explain(X_test)
    """

    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 causal_graph: np.ndarray, n_samples: int = 1000,
                 random_state: Optional [int] = None):
        """Initialize Asymmetric Shapley explainer.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to explain
        background_data : pd.DataFrame
            Reference dataset for marginalizing over missing features
        causal_graph : np.ndarray
            Adjacency matrix encoding causal structure INCLUDING outcome node
            Shape should be (n_features+1, n_features+1) where last row/col is outcome Y
        n_samples : int, default=1000
            Number of random causal path orderings to sample
        random_state : int or None
            Random seed
        """
        
        super().__init__(model, background_data, n_samples, random_state)

        self.causal_graph = causal_graph
        self.directed_graph = self._extract_directed_graph(causal_graph)
        self.outcome_node = self.directed_graph.shape[0] - 1
        if self._has_cycle(self.directed_graph):
            warnings.warn(
                "Directed causal constrains contain cycles."
                "Causal Shapley requires a DAG; disabling constraints."
            )
            self.directed_graph = np.zeros_like(self.directed_graph)

        # Build parents dictionary after extracting directed graph
        self.parents = {}
        n_nodes_total = self.directed_graph.shape[0]
        for j in range(n_nodes_total):
            self.parents[j] = set(np.where(self.directed_graph[:, j] != 0)[0])

        # Identify source nodes (no incoming edges, excluding outcome)
        n_features_total = self.directed_graph.shape[0]
        all_source_nodes = []
        for i in range(n_features_total):
            if i == self.outcome_node:
                continue
            has_incoming = any(self.directed_graph[j, i] != 0 for j in range(n_features_total))
            if not has_incoming:
                all_source_nodes.append(i)

        # Precompute X-only children list and initial in-degrees for the
        # random topological ordering sampler (excludes the outcome/Y node).
        self._x_children: Dict[int, List[int]] = {i: [] for i in range(self.n_features)}
        self._x_in_degree: List[int] = [0] * self.n_features
        for i in range(self.n_features):
            for j in range(self.n_features):
                if self.directed_graph[i, j] != 0:
                    self._x_children[i].append(j)
                    self._x_in_degree[j] += 1


    def _extract_directed_graph(self, causal_graph: np.ndarray) -> np.ndarray:
        """Convert causal graph matrix into binary directed adjacency matrix.
        
        Supports two common encodings:
        1. Binary adjacency: causal_graph[i, j] = 1 means i → j (and [j,i]=0)
        2. Causal-learn endpoint encoding: 
           - causal_graph[i,j]=-1, causal_graph[j,i]=1 means i → j (tail at i, arrow at j)
           - Other combinations (undirected, bidirected) are ignored
        
        Parameters
        ----------
        causal_graph : np.ndarray
            Input causal graph matrix INCLUDING outcome node
            Expected shape: (n_features+1, n_features+1)
        
        Returns
        -------
        directed : np.ndarray
            Binary adjacency matrix where directed[i,j]=1 means i → j
            Shape: (n_features+1, n_features+1)
        """
        expected_shape = (self.n_features + 1, self.n_features + 1)
        if causal_graph.shape != expected_shape:
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match expected shape "
                f"{expected_shape} (n_features+1 to include outcome node)"
            )
        
        n_nodes_total = self.n_features + 1
        directed = causal_graph.astype(int)

        n_raw_edged = int(np.sum(causal_graph != 0))
        n_directed_edges = int(np.sum(directed != 0))
        if n_raw_edged > 0 and n_directed_edges == 0:
            warnings.warn(
                "No directed edges could be extracted from causal_graph."
                "Falling back to unconstrained Causal Shapley behavior"
            )
        return directed
        
    def _has_cycle(self, directed_graph: np.ndarray) -> bool: 
        """ Return True if the directed graph contains at least one cycle."""
        n_nodes_total = directed_graph.shape[0]
        in_degree = np.sum(directed_graph != 0, axis = 0).astype(int)
        queue = [i for i in range(n_nodes_total) if in_degree[i] == 0]
        visited = 0

        while queue :
            node = queue.pop(0)
            visited+=1
            for child in range(n_nodes_total):
                if directed_graph[node, child] !=0:
                    in_degree[child] -=1
                    if in_degree[child] ==0:
                        queue.append(child)
        return visited != n_nodes_total

    
    
    def _sample_topological_ordering(self) -> List[int]:
        """Sample a uniformly random topological ordering of all X features.

        Implements the standard random topological sort (Knuth 1997):
        1. Initialise a pool of *ready* nodes — those whose in-degree
           among X-only edges has fallen to zero.
        2. Repeatedly pick **uniformly at random** from the pool, append
           to the ordering, and decrement the in-degrees of its children;
           add any newly-ready child to the pool.
        3. Y (outcome node) is NOT included here; the caller decides
           whether to append it.

        This generates a sample from the **uniform distribution over all
        valid topological orderings** (linear extensions) of the DAG,
        which is exactly the permutation distribution assumed by Frye et
        al. (2021) Asymmetric Shapley values.

        Differences from the old path-sampling approach
        -----------------------------------------------
        - The causal partial order is always respected for *all* pairs,
          not just pairs on a single source-to-Y path.
        - O(n) per sample instead of DFS path enumeration.

        Returns
        -------
        ordering : List[int]
            A random valid topological ordering of feature indices 0…n_x-1.
        """
        n_x = self.n_features
        remaining_in_degree = self._x_in_degree[:]
        ready = [i for i in range(n_x) if remaining_in_degree[i] == 0]
        ordering: List[int] = []

        while ready:
            # Uniform random pick via swap-remove (O(1))
            idx = self.rng.randint(len(ready))
            node = ready[idx]
            ready[idx] = ready[-1]
            ready.pop()

            ordering.append(node)

            for child in self._x_children[node]:
                remaining_in_degree[child] -= 1
                if remaining_in_degree[child] == 0:
                    ready.append(child)

        # Safety fallback if the graph still has cycles (should not happen
        # after cycle removal in causal discovery, but be defensive).
        if len(ordering) < n_x:
            placed = set(ordering)
            remaining = [i for i in range(n_x) if i not in placed]
            self.rng.shuffle(remaining)
            ordering.extend(remaining)

        return ordering

    def _compute_monte_carlo_causal_shapley(self, instance: np.ndarray) -> np.ndarray:
        """Compute Asymmetric Shapley values via random topological orderings.

        ALGORITHM (Frye et al. 2021):
        Repeat n_samples times:
            1. Draw a uniformly random valid topological ordering π of all X features.
               A valid ordering respects the DAG partial order:
               if i → j then i precedes j in π.
            2. Initialise: S = ∅, v_prev = baseline
            3. For each feature f_i in order π:
                a. S ← S ∪ {f_i}
                b. v_curr ← v(S)   # coalition value (background marginalisation)
                c. φᵢ += v_curr − v_prev
                d. v_prev ← v_curr
        Return: φ / n_samples

        Advantages over path sampling
        -----------------------------
        - All features contribute in every permutation (no structural zeros).
        - Respects the DAG partial order for every pair of features, not
          just for pairs on a sampled source-to-Y path.
        - O(n) sampling cost per permutation instead of DFS.

        Parameters
        ----------
        instance : np.ndarray
            Instance to explain (shape: n_features,)

        Returns
        -------
        shapley_values : np.ndarray
            Asymmetric Shapley values (shape: n_features,)
        """
        shapley_values = np.zeros(self.n_features)

        for sample_idx in range(self.n_samples):
            # Sample a random valid topological ordering of all X features
            perm = self._sample_topological_ordering()

            prev_value = self.baseline_value
            coalition: List[int] = []

            for feature_idx in perm:
                coalition.append(feature_idx)
                curr_value = self._predict_coalition(instance, coalition)
                shapley_values[feature_idx] += curr_value - prev_value
                prev_value = curr_value

            if (sample_idx + 1) % 10 == 0:
                logging.info(f"    Sample {sample_idx + 1}/{self.n_samples} completed")

        shapley_values /= self.n_samples
        return shapley_values
    
    def explain(self, X: pd.DataFrame, method: str = 'monte_carlo') -> np.ndarray:

        print(f"Computing Asymmetric Shapley values (random topological ordering, {method} mode)...")
        logging.info(f"Starting Asymmetric Shapley (topo-ordering) computation for {len(X)} instances")
        logging.info(f"Using {self.n_samples} samples per instance")
        # Detailed configuration info removed for cleaner output
        # print(f"Sampling approach: Path-based (focuses on causal paths to outcome)")
        # print(f"Respecting causal graph with {np.sum(self.causal_graph != 0)} edges")

        X_values = X.values
        n_instances = len(X_values)


        self.shap_values = np.zeros((n_instances, self.n_features))

        for i, instance in enumerate(X_values):
    
            self.shap_values[i] = self._compute_monte_carlo_causal_shapley(instance)

            # # Log progress every 20 instances
            # if (i + 1) % 20 == 0:
            #     logging.info(f"  Completed {i + 1}/{n_instances} instances")
            
            # Progress indicator removed for cleaner output
            if (i + 1) % max(1, n_instances//10) == 0:
                print(f" Progress: {i + 1}/{n_instances} instances")
        
        logging.info(f"Completed all {n_instances} instances")

        return self.shap_values

class CausalShapley(ShapleyFromScratch):
    """Causal Shapley values using post-interventional sampling (Heskes et al. 2020).
    
    ═══════════════════════════════════════════════════════════════════════════════
    WHAT THIS METHOD ACTUALLY DOES (based on code implementation):
    ═══════════════════════════════════════════════════════════════════════════════
    
    Computes Shapley values using INTERVENTIONAL semantics: replaces conditional
    distributions P(X | X_S = x_S) with do-calculus P(X | do(X_S = x_S)).
    
    KEY BEHAVIORS:
    1. Uses ALL features in every permutation
    2. Outer permutation loop draws from the uniform distribution over valid
       topological orderings of the component DAG (causal-constrained)
    3. For each coalition, inner loop samples from the do-distribution
    4. Handles confounding via component-based independent/joint sampling
    5. Much more computationally expensive than vanilla (nested sampling loops)
    
    ═══════════════════════════════════════════════════════════════════════
    KEY CONCEPTUAL DIFFERENCE FROM OTHER METHODS:
    ═══════════════════════════════════════════════════════════════════════
    
    Standard Shapley: v(S) = E[f(X) | X_S = x_S]
        → Conditions on observed features (preserves correlations/confounding)
    
    Causal Shapley: v(S) = E[f(X) | do(X_S = x_S)]
        → Intervenes on features (breaks incoming edges, removes confounding)
    
    ═══════════════════════════════════════════════════════════════════════
    CONFOUNDING EXAMPLE:
    ═══════════════════════════════════════════════════════════════════════
    
    Consider: U → X1, U → X2, X1 → Y, X2 → Y  (U is hidden confounder)
    
    Question: What is the contribution of X1 to Y?
    
    Standard Shapley (conditioning):
        - When we condition X1=x1, we also implicitly condition on U
        - This affects X2's distribution through the confounding path
        - X1 gets credit for effects that actually come from U
        - OVERESTIMATES X1's causal effect
    
    Causal Shapley (intervention):
        - When we intervene do(X1=x1), we cut U → X1 edge
        - X2 is sampled independently of X1 (confounding removed)
        - X1 only gets credit for direct causal effects
        - CORRECT causal interpretation
    
    ═══════════════════════════════════════════════════════════════════════
    HOW CAUSAL STRUCTURE IS USED:
    ═══════════════════════════════════════════════════════════════════════
    
    1. ADJACENCY MATRIX (discovered_adj):
       - Binary DAG: adj[i,j]=1 means i → j (parent-child relationships)
       - Used to determine topological ordering and parent dependencies
    
    2. CONFOUNDERS (discovered_conf):
       - List of feature pairs: [(X1, X2), (X3, X4), ...]
       - Means X1 ↔ X2 (bidirected edge = shared hidden confounder U)
       - Groups confounded features into components
    
    3. COMPONENT STRUCTURE:
       - Features are partitioned into causal components
       - CONFOUNDED component: features share a hidden confounder
       - NON-CONFOUNDED component: individual feature or causally related features
    
    4. COMPONENT DAG (precomputed in __init__):
       - _comp_children[c]: child components of component c
       - _comp_in_degree[c]: number of parent components of component c
       - Used by _sample_component_topological_ordering() at every outer iteration
    
    ═══════════════════════════════════════════════════════════════════════
    OUTER PERMUTATION — COMPONENT TOPOLOGICAL ORDERING:
    ═══════════════════════════════════════════════════════════════════════
    
    The outer loop samples a random valid topological ordering of all features
    in two stages:
    
    Stage 1 — Kahn-style random sort of components:
        - Maintain a pool of ready components (in-degree 0 in the component DAG)
        - At each step pick uniformly at random from the pool (swap-remove, O(1))
        - Decrement child in-degrees; add newly-ready children to pool
        - Produces a uniform sample from the linear extensions of the component DAG
    
    Stage 2 — Expand to features:
        - For each component in the sampled order, expand to its features
        - Within a confounded component the features are shuffled randomly
        - Single-feature (non-confounded) components are trivially ordered
    
    ═══════════════════════════════════════════════════════════════════════
    POST-INTERVENTIONAL SAMPLING ALGORITHM (inner loop):
    ═══════════════════════════════════════════════════════════════════════
    
    To sample from P(X | do(X_S = x_S)):
    
    1. Fix intervened features: X_S = x_S (from instance)
    
    2. For each component t in FIXED topological order:
       a. Identify: fixed_features = component ∩ S (intervened)
                   missing_features = component \\ S (to sample)
       
       b. Get parent values (already determined from topological order)
       
       c. IF component is CONFOUNDED:
             → Sample each missing feature INDEPENDENTLY given parents only
             → Intervention breaks dependencies between confounded features
             → For each missing feature j:
                 X_j ~ P(X_j | Parents(X_j))
       
       d. ELSE (component is NON-CONFOUNDED):
             → Sample missing features JOINTLY given parents + fixed siblings
             → Preserves mutual interactions within component
             → (X_missing) ~ P(X_missing | Parents, X_fixed)
    
    Note: the inner loop always traverses components in the FIXED deterministic
    topological order (not random), because the do-calculus requires a consistent
    structural ordering to propagate interventions correctly.
    
    3. Repeat M times to get samples from the do-distribution
    
    ═══════════════════════════════════════════════════════════════════════
    BACKGROUND DATA USAGE:
    ═══════════════════════════════════════════════════════════════════════
    
    Background data is used to estimate conditional distributions:
    - Fit Gaussian approximation: X ~ N(μ, Σ)
    - Use conditional Gaussian formulas for sampling:
      X_A | X_B = b ~ N(μ_A + Σ_AB Σ_BB^{-1}(b - μ_B), Σ_AA - Σ_AB Σ_BB^{-1} Σ_BA)
    
    Note: Current implementation assumes Gaussian distributions.
    For non-Gaussian: could use Gibbs sampling or other methods.
    
    ═══════════════════════════════════════════════════════════════════════
    PARAMETERS:
    ═══════════════════════════════════════════════════════════════════════
    
    Parameters
    ----------
    model : BaseEstimator
        Trained model to explain
    background_data : pd.DataFrame
        Reference dataset for estimating conditional distributions
    discovered_adj : np.ndarray
        Adjacency matrix (n_features x n_features) encoding causal DAG
    discovered_conf : List[Tuple[str,str]]
        List of confounded feature pairs: [(feature1, feature2), ...]
    feature_names : List[str]
        Feature names (must match background_data columns)
    n_samples : int, default=100
        Number of outer permutations (Monte Carlo samples)
    M_inner_samples : int, default=50
        Number of inner samples from do-distribution per coalition
    random_state : int or None
        Random seed
    
    Attributes
    ----------
    causal_graph_components : List[List[int]]
        Groups of features (confounded groups + individual features) in topological order
    confounded_info : Dict[int, bool]
        Maps component_idx → is_confounded (True if multiple features in component)
    parents_dict : Dict[int, List[int]]
        Maps component_idx → list of parent feature indices
    feature_to_component : Dict[int, int]
        Maps feature_idx → component_idx
    _comp_children : Dict[int, List[int]]
        Children of each component in the component DAG
    _comp_in_degree : List[int]
        Initial in-degree of each component in the component DAG
    
    ═══════════════════════════════════════════════════════════════════════
    REFERENCES:
    ═══════════════════════════════════════════════════════════════════════
    
    Heskes, T., Sijben, E., Bucur, I. G., & Claassen, T. (2020).
    "Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual
    Predictions of Complex Models." NeurIPS 2020.
    
    Examples
    --------
    >>> adj = np.array([[0,1,0], [0,0,1], [0,0,0]])  # X0→X1→X2
    >>> conf = [('X0', 'X1')]  # X0 and X1 confounded
    >>> explainer = CausalShapley(model, X_train, adj, conf, 
    ...                           feature_names=['X0','X1','X2'])
    >>> shap_values = explainer.explain(X_test)
    """

    def __init__(self, model: BaseEstimator,
                 background_data: pd.DataFrame,
                 discovered_adj: np.ndarray,
                 discovered_conf: List[Tuple[str,str]],
                 feature_names: List[str],
                 n_samples: int = 100,
                 M_inner_samples: int = 100,
                 random_state: Optional[int] = None):
        """Initialize CausalShapley with post-interventional sampling.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model
        background_data : pd.DataFrame
            Reference data for conditional sampling
        discovered_adj : np.ndarray
            Adjacency matrix of causal DAG
        discovered_conf : List[Tuple[str,str]]
            Confounded feature pairs
        feature_names : List[str]
            Feature names
        n_samples : int, default=100
            Outer permutations
        M_inner_samples : int, default=50
            Inner samples per coalition
        random_state : int or None
            Random seed
        """
        
        super().__init__(model, background_data, n_samples,random_state)

        # Store causal structure
        # Verbose initialization removed for cleaner output
        # print(f"Initialized components")

        self.causal_graph_components,self.confounded_info, self.parents_dict = self._extract_causal_structure_for_shapley(discovered_adj,discovered_conf,feature_names)
        
        # print("Components initialized")
        
        self.n_samples = n_samples
        self.M_inner_samples = M_inner_samples

        # Build component membership map for quick lookup
        self.feature_to_component = {}
        for comp_idx, features in enumerate(self.causal_graph_components):
            for feature in features:
                self.feature_to_component[feature] = comp_idx

        # Precompute statistics for Gaussian conditional sampling
        self._precompute_statistics()

        # Precompute component-level children and in-degrees so we can draw a
        # random valid topological ordering of components at each outer iteration.
        n_comps = len(self.causal_graph_components)
        self._comp_children: Dict[int, List[int]] = {i: [] for i in range(n_comps)}
        self._comp_in_degree: List[int] = [0] * n_comps
        for comp_idx in range(n_comps):
            parent_feat_indices = self.parents_dict.get(comp_idx, [])
            # Map each parent feature to its component, deduplicate
            parent_comp_indices = set(
                self.feature_to_component[f] for f in parent_feat_indices
            )
            for parent_comp in parent_comp_indices:
                self._comp_children[parent_comp].append(comp_idx)
                self._comp_in_degree[comp_idx] += 1

        # Verbose initialization removed for cleaner output
        # print("CausalShapleyPostInterventional Initialzied:")
        # print(f" Features: {self.n_features}")
        # print(f" Components: {len(self.causal_graph_components)} (in topological order)")
        # for comp_idx, comp in enumerate(self.causal_graph_components):
        #     conf_status = "CONFOUNDED" if self.confounded_info.get(comp_idx, False) else "NON-CONFOUNDED"
        #     parents = self.parents_dict.get(comp_idx, [])
        #     print(f" Component {comp_idx}: features {comp}, {conf_status}, parents {parents}")
        # print(f" Outer samples (permutations): {n_samples}")
        # print(f" Inner samples (per coalition): {M_inner_samples}")

    def _extract_causal_structure_for_shapley(self,
                                               causal_graph: np.ndarray,
                                               confounders: List[Tuple[str,str]],
                                               feature_names: List[str]
                               ) -> Tuple[List[List[int]],Dict[int,bool],Dict[int,List[int]]]:
        """Extract causal structure components from binary adjacency matrix.
        
        Note: Assumes causal_graph is already a binary adjacency matrix (0s and 1s)
        as guaranteed by the causal discovery step.
        """
        if causal_graph.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match features"
            )
        
        # Validate binary adjacency matrix
        unique_vals = set(np.unique(causal_graph).tolist())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"causal_graph must be binary (0s and 1s), got values: {unique_vals}"
            )
        
        directed = causal_graph.astype(int)
        n_features = len(feature_names)

        confounded_pairs = set()
        for feat1,feat2 in confounders:
            try:
                idx1 = feature_names.index(feat1)
                idx2 = feature_names.index(feat2)
                confounded_pairs.add((min(idx1, idx2),max(idx1,idx2)))
            except ValueError:
                continue

        G = nx.DiGraph()
        G.add_nodes_from(range(n_features))
        for i in range(n_features):
            for j in range(n_features):
                if directed[i,j] !=0:
                    G.add_edge(i,j)

        #Topological sort
        try: 
            topo_order = list(nx.topological_sort(G))
        except nx.NetworkXError as e:
            # Topological sort fails when graph has cycles
            print(f"WARNING: Topological sort failed - graph contains cycles!")
            print(f"         This violates the DAG assumption for Causal Shapley.")
            print(f"         Using simple ordering (results may be unreliable).")
            print(f"         Error: {e}")
            topo_order = list(range(n_features))
        except Exception as e:
            print(f"WARNING: Topological sort failed with unexpected error: {e}")
            print(f"         Using simple ordering.")
            topo_order = list(range(n_features))
        
        # Group confounded features into components
        # Features are confounded if they share a bidirected edge
        confounded_groups = []
        remaining_features = set(topo_order)

        for idx1, idx2 in confounded_pairs:
            # Check if feature are already in a group
            found_group = None
            for group in confounded_groups:
                if idx1 in group or idx2 in group:
                    found_group = group
                    break

            if found_group is not None:
                found_group.add(idx1)
                found_group.add(idx2)
            else:
                confounded_groups.append({idx1,idx2})

            remaining_features.discard(idx1)
            remaining_features.discard(idx2)

        # Build componets: confounded groups + individual features
        components = []
        component_map = {} # Maps feature idx to component idx

        # Add confounded groups first (in topological order)
        for group in confounded_groups:
            group_list = sorted(group, key= lambda x: topo_order.index(x))
            comp_idx = len(components)
            components.append(group_list)
            for feat_idx in group_list:
                component_map[feat_idx] = comp_idx

        # Add remaining individual features
        for feat_idx in sorted(remaining_features, key = lambda x : topo_order.index(x)):
            comp_idx = len(components)
            components.append([feat_idx])
            component_map[feat_idx] = comp_idx

        # Build parent dictionary for each component
        confounded_info = {}
        for comp_idx, comp in enumerate(components):
            # component is confounded if it has multple features (shared confounder)
            confounded_info[comp_idx] = len(comp) > 1

        # Build parent dictionary for each component
        parents_dict = {}
        for comp_idx, comp_features in enumerate(components):
            parents = set()
            for feat_idx in comp_features:
                # Find parents of this feature
                for parent_idx in range(n_features):
                    if directed[parent_idx, feat_idx] != 0:
                        if parent_idx in component_map and component_map[parent_idx] < comp_idx:
                            parents.add(parent_idx)
            parents_dict[comp_idx] = sorted(list(parents))

        return components, confounded_info, parents_dict
    
    def _precompute_statistics(self):
        """Precompute mean and covariance for Gaussian condtional sampling"""
        self.mean = np.mean(self.background_data,axis=0)
        self.cov = np.cov(self.background_data.T)

        # Add small regularization to ensure postive definitness
        self.cov += np.eye(self.n_features) * 1e-6

    def _sample_conditional_gaussian(self, target_features: List[int],
                                     conditioning_features: List[int],
                                     conditioning_values: np.ndarray) -> np.ndarray:
        """Sample from conditional Gaussian distribution P(X_target | X_cond = values).
        
        Uses the closed-form conditional Gaussian formula:
        
        Given X = [X_A, X_B] ~ N(μ, Σ), then:
        X_A | X_B = b ~ N(μ_A|B, Σ_A|B) where:
            μ_A|B = μ_A + Σ_AB Σ_BB^{-1} (b - μ_B)
            Σ_A|B = Σ_AA - Σ_AB Σ_BB^{-1} Σ_BA
        
        This is a key subroutine for post-interventional sampling.
        
        Parameters
        ----------
        target_features : List[int]
            Feature indices to sample
        conditioning_features : List[int]
            Feature indices being conditioned on
        conditioning_values : np.ndarray
            Observed values of conditioning features
        
        Returns
        -------
        samples : np.ndarray
            Sampled values for target features (shape: len(target_features))
        
        Notes
        -----
        - If no conditioning features: samples from marginal P(X_target)
        - Adds regularization to ensure positive definiteness
        - Falls back to marginal if conditioning fails (singular covariance)
        """

        if len(conditioning_features) == 0:
            # No conditioning: sample from marginal
            if len(target_features) == 1:
                return np.array([self.rng.normal(self.mean[target_features[0]],
                                                 np.sqrt(self.cov[target_features[0],target_features[0]]))])
            else:
                cov_target = self.cov[np.ix_(target_features,target_features)]
                return self.rng.multivariate_normal(self.mean[target_features],cov_target)
            
        try:
            #Extract covariance submatrices
            Sigma_AA = self.cov[np.ix_(target_features,target_features)]
            Sigma_BB = self.cov[np.ix_(conditioning_features,conditioning_features)]
            Sigma_AB = self.cov[np.ix_(target_features,conditioning_features)]

            # Conditional mean
            Sigma_BB_inv = np.linalg.pinv(Sigma_BB)
            conditional_mean = (self.mean[target_features]+
                                Sigma_AB @ Sigma_BB_inv @ (conditioning_values - self.mean[conditioning_features]))
            
            # Conditional covariance
            conditional_cov = Sigma_AA - Sigma_AB @ Sigma_BB_inv @ Sigma_AB.T

            # Ensure positive definiteness
            conditional_cov = (conditional_cov + conditional_cov.T) / 2
            eigvals = np.linalg.eigvalsh(conditional_cov)
            if eigvals.min() < 1e-6:
                conditional_cov += np.eye(len(target_features)) * (1e-6 - eigvals.min())

            # Sample
            if len(target_features) == 1:
                return np.array([self.rng.normal(conditional_mean[0],np.sqrt(conditional_cov[0,0]))])
            else: 
                return self.rng.multivariate_normal(conditional_mean,conditional_cov)
        except:
            # Fallback: sample from marginal if conditioning fails
            print("Warning: Gaussian conditioning failed, sampling from marginal distribution")
            if len(target_features) == 1:
                return np.array([self.mean[target_features[0]]])
            else:
                return self.mean[target_features]
            
    def _sample_post_interventional(self, S: List[int], x_instance: np.ndarray) -> np.ndarray:
        """
        Sample from post-interventional distribution P(X | do(X_S = x_S))
        
        This is the core algorithm from Heskes et al. (2020)

        Algorithm:
        1. Fix X_S = X_S (interventional values from instance)
        2. For each component t in topological order:
            - Identify missing features (not in S) within component
            - Get parents values (already determined from topological ordering)
            - IF component is CONFOUNDED:
                Sample missing features INDEPENDENTLY conditional on parents only 
                (Intervention breaks dependencies between features in component)
            - ELSE (component is NON-CONFOUNDED):
                Sample missing features JOINTLY conditional on parents + sibligs in S
                (Features have mutual intereactions, not confounding)

        Parameters:
        ----------
        S: List[int]
            Coaliton of features (intervened features)
        x_instance: np.ndarray
            Instance values to explain
        Returns:
        --------
        samples : np.ndarray
            M_inner_samples complete features vectores samples from do-distribution
        """

        S_set = set(S)
        samples = []
        for _ in range(self.M_inner_samples):
            # Initialize: fix intervened features, zero out the rest
            sample = np.zeros(self.n_features)
            sample[list(S)] = x_instance[list(S)]

            # Iterate through components in topological order
            for comp_idx, component in enumerate(self.causal_graph_components):
                missing_in_comp = [f for f in component if f not in S_set]
                if not missing_in_comp:
                    continue

                # Parent values are already determined (topological order guarantees this)
                parent_features = self.parents_dict.get(comp_idx, [])
                parent_values = sample[parent_features]  # empty array when no parents

                # CRITICAL DISTINCTION: Confounded vs Non-confounded
                if self.confounded_info.get(comp_idx, False):
                    # CONFOUNDED COMPONENT: sample each missing feature INDEPENDENTLY
                    # given parents only — this is what breaks the confounded correlation.
                    # (Joint sampling would preserve it, hence the per-feature loop.)
                    for feature in missing_in_comp:
                        sample[feature] = self._sample_conditional_gaussian(
                            target_features=[feature],
                            conditioning_features=parent_features,
                            conditioning_values=parent_values
                        )[0]
                else:
                    # NON-CONFOUNDED COMPONENT: sample missing features JOINTLY given
                    # parents + fixed siblings — preserves within-component interactions.
                    # NOTE: for non-Gaussian distributions this should use Gibbs sampling.
                    fixed_in_comp = [f for f in component if f in S_set]
                    conditioning_features = parent_features + fixed_in_comp
                    conditioning_values = sample[conditioning_features]
                    sampled_values = self._sample_conditional_gaussian(
                        target_features=missing_in_comp,
                        conditioning_features=conditioning_features,
                        conditioning_values=conditioning_values
                    )
                    sample[missing_in_comp] = sampled_values

            samples.append(sample)

        return np.array(samples)
    
    def _compute_value_function(self, S: List[int], x_instance: np.ndarray) -> float:
        """
        Compute causal value fucntion v(S) = E[f(x) | do(X_S = x_S)]

        Samples M_inner_samples from the post-interventional distribution and
        averages the model predictions

        Parameters:
        -----------
        S : List[int]
            Coalition of features
        x_instance: np.ndarray
            Instance to explain

        Returns:
        -------
        value : float
            Expected prediction E[f(x) | do(X_S = x_S)]
        """
        # Sample from post-interventional distribution
        samples = self._sample_post_interventional(S,x_instance)

        # Predict on all samples with proper feature alignment
        predictions = self._predict_with_feature_alignment(samples)

        return predictions.mean()

    def _sample_component_topological_ordering(self) -> List[int]:
        """Sample a random topological ordering of features via component-level randomisation.

        Two-stage algorithm:

        Stage 1 — random topological sort of **components**:
            Uses the same Kahn-style uniform random pick as AsymmetricShapley
            (swap-remove from the *ready* pool at each step).  This generates
            a sample from the uniform distribution over all linear extensions
            of the component DAG.

        Stage 2 — expand to features:
            Within each component the features are shuffled randomly.  For a
            single-feature (non-confounded) component this is a no-op.

        The resulting flat list satisfies the causal partial order: every
        feature always appears after **all** features in its ancestor components.
        Confounded siblings within a component may appear in any relative order.

        Returns
        -------
        ordering : List[int]
            Feature indices in a random valid topological order (length n_features).
        """
        n_comps = len(self.causal_graph_components)
        remaining_in_degree = self._comp_in_degree[:]
        ready = [i for i in range(n_comps) if remaining_in_degree[i] == 0]
        comp_order: List[int] = []

        while ready:
            # Uniform random pick via swap-remove — O(1)
            idx = self.rng.randint(len(ready))
            comp_idx = ready[idx]
            ready[idx] = ready[-1]
            ready.pop()
            comp_order.append(comp_idx)
            for child_comp in self._comp_children[comp_idx]:
                remaining_in_degree[child_comp] -= 1
                if remaining_in_degree[child_comp] == 0:
                    ready.append(child_comp)

        # Safety fallback — only reached if the component DAG has unexpected cycles
        if len(comp_order) < n_comps:
            placed = set(comp_order)
            remaining = [i for i in range(n_comps) if i not in placed]
            comp_order.extend(remaining)

        # Expand components → flat feature list, shuffling within each component
        feature_ordering: List[int] = []
        for comp_idx in comp_order:
            features = list(self.causal_graph_components[comp_idx])
            if len(features) > 1:
                self.rng.shuffle(features)
            feature_ordering.extend(features)

        return feature_ordering

    def _compute_monte_carlo_causal_shapley(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute Causal Shapley values using Monte Carlo with causal-order-constrained permutations.

        Outer Loop (n_samples iterations):
        - Sample a **random valid topological ordering** of all features,
          respecting the component-level DAG partial order.
        - Initialize empty coalition S = ∅
        - For each feature j in that order:
            * S ← S ∪ {j}
            * v_curr ← E[f(X) | do(X_S = x_S)]  (post-interventional, inner loop)
            * φ_j += v_curr − v_prev
            * v_prev ← v_curr
        Return: φ / n_samples

        Why this fixes the original implementation
        ------------------------------------------
        The original code used `rng.permutation(n_features)` — a fully random
        shuffle with no causal constraints.  The topological order existed only
        *inside* `_sample_post_interventional` (for building the interventional
        sample), but the *outer* Shapley permutation was unconstrained.

        With this fix both layers respect causality:
          - Outer permutation: random linear extension of the component DAG
            (ancestors always precede descendants in the Shapley sum)
          - Inner sampling: fixed topological component traversal for do-calculus

        Parameters:
        ----------
        instance: np.ndarray
            Instance to explain

        Returns:
        --------
        shapley_values : np.ndarray
            Causal Shapley values for each feature
        """

        shapley_values = np.zeros(self.n_features)

        for perm_idx in range(self.n_samples):
            # Sample a random valid topological ordering of all features,
            # respecting the causal partial order defined by the component DAG.
            perm = self._sample_component_topological_ordering()

            # Initialize empty coalition
            coalition = []

            # Track previous value to compute marginals efficiently
            prev_value = self.baseline_value

            # Iterate through features in permutation order
            for feature in perm:
                coalition.append(feature)

                # Compute v(S u {j}) using post-interventional sampling (inner loop)
                curr_value= self._compute_value_function(coalition,instance)

                marginal_contribution = curr_value - prev_value

                shapley_values[feature] +=marginal_contribution

                prev_value = curr_value
            
            # Progress indicator every 50% of permutations
            if (perm_idx + 1) % max(1, self.n_samples // 5) == 0:
                print(f"     Permutations: {perm_idx + 1}/{self.n_samples}")
        
        # Average over all permutation
        shapley_values /= self.n_samples

        return shapley_values
    
    def explain(self, X: pd.DataFrame) -> np.ndarray :
        """
        Compute Causal Shapley values with post interventional sampling.

        Parameters:
        ----------
        X : pd.DataFrame
            Instances to explain

        Returns:
        --------
        shap_values : np.ndarray
            Causal Shapley values (shape: n_samples x n_features)
        """
        print(f"Computing Causal Shapley values (post-interventional sampling)...")
        print(f"  Instances: {len(X)}, Permutations: {self.n_samples}, Inner samples: {self.M_inner_samples}")
        print(f"  Estimated predictions: {len(X) * self.n_samples * self.n_features * self.M_inner_samples:,}")

        X_values = X.values
        n_instances = len(X_values)

        self.shap_values = np.zeros((n_instances, self.n_features))
        for i, instance in enumerate(X_values):
            print(f"  Instance {i+1}/{n_instances}...")
            self.shap_values[i] = self._compute_monte_carlo_causal_shapley(instance)

        return self.shap_values

class ShapleyFlow:
    """Shapley Flow: Edge-level Shapley attributions on causal graphs.
    
    ═══════════════════════════════════════════════════════════════════════════════
    WHAT THIS METHOD ACTUALLY DOES (based on code implementation):
    ═══════════════════════════════════════════════════════════════════════════════
    
    Computes Shapley values for EDGES (not features) in a causal DAG.
    
    KEY BEHAVIORS:
    1. Game players are EDGES (i→j) rather than features
    2. Two modes: MC edge permutation (default) or exhaustive DFS
    3. MC mode permutes ALL edges uniformly — no path sampling is performed
    4. Edge attributions are aggregated to get node (feature) importance
    5. Core class does NOT filter - wrapper handles source detection
    
    ═══════════════════════════════════════════════════════════════════════════════
    ALGORITHM (MC Permutation Mode - DEFAULT, use_mc_permutation=True):
    ═══════════════════════════════════════════════════════════════════════════════

    FOR trial = 1 to n_samples:
        perm_edges ← random_permutation(ALL edges in graph)
        v_prev ← evaluate_system([], x_foreground, x_background)
        history ← []
        FOR each edge in perm_edges:
            history.append(edge)
            v_curr ← evaluate_system(history, x_fg, x_bg)
            edge_attributions[edge] += v_curr − v_prev
            v_prev ← v_curr

    RETURN edge_attributions / n_samples
    
    ═══════════════════════════════════════════════════════════════════════════════
    WHY ALL-EDGE MC PERMUTATION? (Correct Shapley Estimator)
    ═══════════════════════════════════════════════════════════════════════════════

    Path-sampling (backward walk) is INCORRECT for edge Shapley values because:
    - Each backward path covers only ~5 of the 275+ edges
    - Deep edges appear in <1% of paths → accumulated marginals ≈ 0
    - But ALL edges are divided by the same denominator → deep edges undervalued 50-100×
    - Result: efficiency ratio ≈ 0.17× instead of 1.0×

    Correct estimator: random permutation of ALL edges
    - Every edge participates in every permutation → no visitation-frequency bias
    - Σ marginals in one permutation = v(all_edges) − v({}) = f(x) − f(bg) exactly
    - Efficiency axiom holds at 1.00× by construction
    - Cost: n_samples × (n_edges + 1) model calls (e.g. 50 × 276 = 13,800)
    
    ═══════════════════════════════════════════════════════════════════════════════
    EDGE COALITION SEMANTICS:
    ═══════════════════════════════════════════════════════════════════════════════
    
    For each edge (i→j):
    - Edge ACTIVE: Node j uses foreground value from instance
    - Edge INACTIVE: Node j treated as "missing", uses background value
    
    evaluate_system(active_edges, x_foreground, x_background):
        FOR each node:
            IF node is source AND has active outgoing edges:
                node_value ← x_foreground[node]
            ELSE IF node has active incoming edges:
                node_value ← x_foreground[node]
            ELSE:
                node_value ← x_background[node]
        RETURN predict(node_values)
    
    ═══════════════════════════════════════════════════════════════════════════════
    CAUSAL GRAPH USAGE (IMPORTANT - NO FILTERING IN CORE CLASS):
    ═══════════════════════════════════════════════════════════════════════════════
    
    ShapleyFlow class itself does NOT filter:
    - Takes graph_structure as input (adjacency list)
    - Takes source_nodes as input (must be pre-filtered)
    - Uses ALL edges in graph_structure
    
    FILTERING HAPPENS IN WRAPPER (ShapleyFlowWrapper):
    - Wrapper filters sources via backward BFS
    - Wrapper does NOT filter adjacency matrix (uses all edges)
    - See ShapleyFlowWrapper docstring for details
    
    ═══════════════════════════════════════════════════════════════════════
    BACKGROUND vs FOREGROUND DATA:
    ═══════════════════════════════════════════════════════════════════════
    
    - FOREGROUND: The instance x to explain
        * Used for source nodes when their edges are active
        * Used for downstream nodes when their incoming edges are active
    
    - BACKGROUND: Reference dataset
        * Used to sample "missing" node values when edges are inactive
        * Provides conditional distribution for on-manifold sampling
    
    ═══════════════════════════════════════════════════════════════════════
    ALGORITHM (Recursive DFS with Random Permutations):
    ═══════════════════════════════════════════════════════════════════════
    
    For n_samples trials:
        1. Start DFS from each source node
        2. At each node u:
            a. Get children: {v1, v2, ..., vk}
            b. Random permutation: shuffle children order
            c. For each child vi in shuffled order:
                - Compute v_before = value(current_edge_set)
                - Add edge (u → vi) to edge_set
                - Compute v_after = value(current_edge_set ∪ {(u,vi)})
                - Marginal contribution = v_after - v_before
                - Accumulate to edge attribution
                - Recurse to child vi
    
    Average edge attributions over all trials.
    
    ═══════════════════════════════════════════════════════════════════════
    PATH SAMPLING MODE (FASTER):
    ═══════════════════════════════════════════════════════════════════════
    
    For dense graphs, exhaustive DFS explores too many edge combinations.
    Path sampling mode is more efficient:
    
    1. Sample K random paths from each source to sink
    2. For each path:
        a. Extract edges in path: [(u1,v1), (u2,v2), ..., (uk,vk)]
        b. Random permutation of these edges
        c. Evaluate incrementally: ∅ → +edge1 → +edge2 → ... → complete_path
        d. Compute marginal for each edge
    3. Average over all sampled paths
    
    This focuses sampling on relevant causal paths (ignores irrelevant edges).
    
    ═══════════════════════════════════════════════════════════════════════
    HOW CAUSAL DAG IS USED:
    ═══════════════════════════════════════════════════════════════════════
    
    - graph_structure: Dict[node → children]
        Defines which edges exist in the DAG
        Only edges in this structure receive attributions
    
    - source_nodes: Nodes with no parents (inputs)
        DFS/path sampling starts from these nodes
    
    - sink_node: Final output node (Y)
        Model prediction is read from this node
    
    ═══════════════════════════════════════════════════════════════════════
    EFFICIENCY AXIOM:
    ═══════════════════════════════════════════════════════════════════════
    
    Shapley Flow satisfies:
    Σ_{edges} attribution(edge) = f(x_foreground) - f(x_background)
    
    Total edge importance = prediction difference between foreground and background
    
    ═══════════════════════════════════════════════════════════════════════
    
    Parameters
    ----------
    graph_structure : Dict[int, List[int]]
        Adjacency list: graph[u] = [v1, v2, ...] means edges u→v1, u→v2, ...
    background_data : np.ndarray
        Reference dataset for conditional sampling (n_samples x n_features)
    model : BaseEstimator or None
        Trained model (required if sink_node is specified)
    source_nodes : List[int] or None
        Input nodes (auto-detected as nodes with no parents if None)
    sink_node : int or None
        Output node for predictions
    n_samples : int, default=100
        Number of Monte Carlo trials (permutations)
    random_state : int or None
        Random seed
    feature_names : List[str] or None
        Feature names for model alignment
    use_mc_permutation : bool, default=True
        If True, permute ALL edges uniformly each trial (unbiased Shapley estimator).
        If False, use exhaustive DFS traversal (original algorithm).
    paths_per_source : int, default=100
        Unused when use_mc_permutation=True; retained for the DFS fallback mode.
    
    Attributes
    ----------
    edge_attributions : Dict[Tuple[int,int], float]
        Importance score for each edge (u,v)
    parents : Dict[int, List[int]]
        Reverse graph: parents[v] = [u1, u2, ...] for edges u1→v, u2→v, ...
    
    References
    ----------
    Wang, J., & Venkatasubramanian, S. (2021). "Shapley Flow: A Graph-based
    Approach to Interpreting Model Predictions." AISTATS 2021.
    
    Examples
    --------
    >>> graph = {0: [2], 1: [2], 2: [3], 3: []}  # X0→X2←X1, X2→X3
    >>> flow = ShapleyFlow(graph, background_data, model, 
    ...                    source_nodes=[0,1], sink_node=3)
    >>> edge_attrs = flow.compute(x_foreground, x_background)
    >>> node_attrs = flow.get_node_attributions()
    """
    def __init__(self, graph_structure: Dict[int, List[int]],
                 background_data: np.ndarray,
                 model: Optional[BaseEstimator] = None,
                 source_nodes: Optional[List[int]] = None,
                 sink_node: Optional[int] = None,
                 n_samples: int = 100,
                 random_state: Optional[int] = None,
                 feature_names: Optional[List[str]] = None,
                 use_mc_permutation: bool = True,
                 paths_per_source: int = 100):
        """Initialize Shapley Flow calculator."""

        self.graph = graph_structure
        self.background_data = background_data
        self.model = model
        self.source_nodes = source_nodes
        self.sink_node = sink_node
        self.n_samples = n_samples
        self.feature_names = feature_names
        self.use_mc_permutation = use_mc_permutation
        self.paths_per_source = paths_per_source

        self.rng = np.random.RandomState(random_state)

        # Build reverse graph (parents for each node)
        self.parents = {node:[] for node in graph_structure.keys()}
        for parent, children in graph_structure.items():
            for child in children:
                if child not in self.parents:
                    self.parents[child] = []
                self.parents[child].append(parent)

        # Source nodes must be provided (computed by wrapper with proper filtering)
        if source_nodes is None:
            raise ValueError(
                "source_nodes must be provided. Use ShapleyFlowWrapper for automatic "
                "source node detection with proper filtering to sink-reachable nodes."
            )
        self.source_nodes = source_nodes

        # Edge attributions (will be computed)
        self.edge_attributions = {}
        for parent, children in graph_structure.items():
            for child in children:
                self.edge_attributions[(parent,child)]= 0.0
        
        # Track evaluation count for debugging
        self.eval_count = 0
        self.max_evals_per_trial = 100000  # Safety limit
    
    def _predict_with_feature_alignment(self, X):
        """
        Predict with proper feature alignment.
        
        Converts numpy arrays to model's expected input format.
        For wrapper models, need to provide feature names.
        
        Parameters:
        -----------
        X : np.ndarray
            Input samples (excluding sink/Y node)
            
        Returns:
        --------
        predictions : np.ndarray
            Model predictions
        """
        if isinstance(X, np.ndarray):
            # If we have feature names, convert to DataFrame
            if self.feature_names is not None:
                # X contains features excluding Y (sink node)
                # Get feature names excluding Y
                feature_names_no_y = [name for i, name in enumerate(self.feature_names) 
                                     if i != self.sink_node]
                
                if X.ndim == 1:
                    X_df = pd.DataFrame([X], columns=feature_names_no_y)
                else:
                    X_df = pd.DataFrame(X, columns=feature_names_no_y)
                return self.model.predict(X_df)
            else:
                # No feature names, hope model can handle raw arrays
                return self.model.predict(X)
        else:
            return self.model.predict(X)

    # def _sample_conditional(self, node: int, observed_nodes: Dict[int, float]) -> float:

    #     if len(observed_nodes) == 0:
    #         # No conditioning information : sample from marginal
    #         return self.background_data[self.rng.randint(len(self.background_data)), node]
        
    #     # Extract conditioning features and values
    #     cond_indices = list(observed_nodes.keys())
    #     cond_values = np.array([observed_nodes[i] for i in cond_indices])

    #     bg_cond = self.background_data[:, cond_indices]
    #     distances = np.sum((bg_cond-cond_values) ** 2, axis =1)

    #     # Use K nearest neighbors (k=10 or 10% of data, whichever is smaller)
    #     k = min(10, max(1, len(self.background_data) // 10 ))
    #     nearest_indices = np.argpartition(distances,k)[:k]

    #     candidate_values = self.background_data[nearest_indices,node]
    #     sampled_values = candidate_values[self.rng.randint(len(candidate_values))]

    #     return sampled_values

    def _evaluate_system(self, history: List[Tuple[int,int]],
                         x_foreground: Dict[int,float],
                         x_background: Dict[int,float]) -> float:
        """
        Evaluate the system state given active edges (history)

       

        Parameters:
        -----------
        history: List[Tuple[int,int]]
            List of active edges (u,v)
        x_foreground: Dict[int, float]
            Foreground values for all nodes
        x_background: Dict[int, float]
            Background values for all nodes
        
        Returns:
        output: float
            Value at the sink node (model prediction or computed value)
        """
        self.eval_count += 1
        history_set = set(history)
        node_values = {}

        # Special case: empty history means baseline (all features at background)
        if len(history) == 0:
            for idx in range(self.background_data.shape[1]):
                node_values[idx] = x_background.get(idx, 0.0)
        else:
            # Identify which nodes are observed (connected by active edges)
            observed_nodes = {}

            for node in self.graph.keys():
                # Check if this node has any active incoming edge
                has_active_incoming = any((parent,node) in history_set
                                          for parent in self.parents.get(node,[]))
                
                # Source nodes: foreground if active in outgoing, backgroung otherwise
                if node in self.source_nodes:
                    has_active_outgoing = any((node, child) in history
                                              for child in self.graph.get(node,[]))
                    if has_active_outgoing:
                        node_values[node] = x_foreground.get(node, 0.0)
                        observed_nodes[node] = node_values[node]
                    else:
                        node_values[node] = x_background.get(node, 0.0)
                        observed_nodes[node] = node_values[node]
                elif has_active_incoming:
                    node_values[node] = x_foreground.get(node, 0.0)
                    observed_nodes[node] = node_values[node]
                elif self.sink_node is not None and (node, self.sink_node) in history_set:
                    # Edge X_k → Y is active: X_k must take its foreground value so
                    # the model sees the real feature value for this direct Y-parent.
                    # Note: Y itself is always stripped from the prediction input via
                    # `feature_indices` below, so no observed Y data ever reaches the
                    # model — the user's requirement is satisfied.
                    node_values[node] = x_foreground.get(node, 0.0)
                    observed_nodes[node] = node_values[node]
            # For missing nodes, use background values
            all_nodes = set(self.graph.keys() | set(child for children in self.graph.values() for child in children))
            for node in all_nodes:
                if node not in node_values:
                    node_values[node] = x_background.get(node, 0.0)

        # Predict using the model with proper feature alignment
        n_features = self.background_data.shape[1]
        feature_indices = [ i for i in range(n_features) if i!= self.sink_node]
        x_features = np.array([node_values.get(idx,0.0) for idx in feature_indices])

        result = self._predict_with_feature_alignment(x_features.reshape(1,-1))[0]
        logging.debug(f"    Result  at evaluate system {result}")

        return result
    
    def _sample_random_path_backward(self, max_depth: int = 100) -> List[Tuple[int, int]]:
        """
        Sample ONE random path by walking backward from sink to a source node.
        
        BACKWARD SAMPLING STRATEGY (more efficient than forward):
        Instead of: Source → random walk → (hope to reach Sink with retries)
        We do:      Sink → walk backward via parents → (guaranteed to reach a Source)
        
        WHY THIS IS BETTER:
        1. self.source_nodes is pre-filtered (wrapper's backward BFS) to only
           sources that CAN reach the sink
        2. Starting from sink and following parents backward MUST eventually
           hit one of these pre-validated sources
        3. NO RETRIES NEEDED - every attempt succeeds!
        4. NO WASTED SAMPLES - every path is valid
        
        Algorithm:
        - Start at sink node (outcome Y)
        - Randomly select one parent
        - Move to that parent, repeat
        - Stop when we reach any node in self.source_nodes
        - Reverse path to get source → sink direction
        
        Parameters:
        -----------
        max_depth: int
            Maximum path length to prevent infinite loops in cyclic graphs
        
        Returns:
        --------
        path_edges: List[Tuple[int, int]]
            List of edges [(u1, v1), (u2, v2), ...] forming complete path source → sink
            Empty list only if sink is None or max_depth exceeded (rare)
        """
        if self.sink_node is None:
            return []
        
        # Walk backward from sink to source
        backward_path = []  # Will store edges in reverse: [(parent, child), ...]
        current_node = self.sink_node
        depth = 0
        visited = {self.sink_node}
        
        # Keep walking backward until we hit a source node
        while current_node not in self.source_nodes and depth < max_depth:
            # Get parents of current node
            node_parents = self.parents.get(current_node, [])
            
            # Filter out visited nodes to avoid cycles
            unvisited_parents = [p for p in node_parents if p not in visited]
            
            if not unvisited_parents:
                # Dead end - shouldn't happen with proper source filtering
                # but we handle it gracefully
                return []
            
            # Randomly select ONE parent (this is the sampling part)
            parent = self.rng.choice(unvisited_parents)
            
            # Store edge in forward direction: parent → current_node
            backward_path.append((parent, current_node))
            
            visited.add(parent)
            current_node = parent
            depth += 1
        
        # Check if we successfully reached a source
        if current_node not in self.source_nodes:
            # Max depth exceeded - should be very rare
            return []
        
        # Reverse the path to get source → sink direction
        path_edges = list(reversed(backward_path))
        
        return path_edges
    
    def _evaluate_path_contribution(self, path_edges: List[Tuple[int, int]],
                                    x_foreground: Dict[int, float],
                                    x_background: Dict[int, float]) -> Dict[Tuple[int, int], float]:
        """
        Compute Shapley marginal contributions for edges in a sampled path.
        
        Uses random permutation of edges in the path (maintains Shapley property).
        
        Strategy:
        1. Generate random permutation of path edges
        2. Evaluate incrementally: empty -> +edge1 -> +edge2 -> ... -> complete path
        3. Marginal of edge_i = value(path[:i+1]) - value(path[:i])
        
        Parameters:
        -----------
        path_edges: List[Tuple[int, int]]
            Ordered list of edges forming a path
        x_foreground: Dict[int, float]
            Foreground values for all nodes
        x_background: Dict[int, float]
            Background values for all nodes
        
        Returns:
        --------
        edge_marginals: Dict[Tuple[int, int], float]
            Dictionary mapping edge -> marginal contribution
        """
        edge_marginals = {}
        
        if not path_edges:
            return edge_marginals
        
        # Random permutation of edges in this path (key for Shapley property!)
        # Use index permutation to preserve tuples (not convert to lists)
        indices = self.rng.permutation(len(path_edges))
        perm_edges = [path_edges[i] for i in indices]
        
        # Evaluate baseline (empty path)
        prev_value = self._evaluate_system([], x_foreground, x_background)
        
        # Build up the path incrementally in random order
        history = []
        for edge in perm_edges:
            history.append(edge)
            current_value = self._evaluate_system(history, x_foreground, x_background)
            marginal = current_value - prev_value
            
            # Store marginal for this edge
            edge_marginals[edge] = marginal
            
            prev_value = current_value
        
        return edge_marginals
    
    def _dfs(self, node: int, history: List[Tuple[int, int]],
             x_foreground: Dict[int, float],
             x_background: Dict[int, float],
             depth: int = 0) -> None:
        """
        Recursive DFS for one trial (one permutation path)

        Processes children in random order, computes marginal contributions,
        and accumulates to global edge attributions.

        Parameters:
        -----------
        node: int
            Current node
        history: List[Tuple[int, int]]
        x_foreground: Dict[int, float]
            Foreground values for source nodes
        x_background: Dict[int, float]
            Background values for source nodes
        depth: int
            Current recursion depth (for limiting)
        """
        # Safety limits
        if depth > 50:  # Max depth to prevent infinite recursion
            return
        if self.eval_count > self.max_evals_per_trial:
            return
            
        # Base case: reached sink
        if node == self.sink_node:
            return
        
        # Get children of current node
        children = self.graph.get(node,[])

        if len(children) == 0:
            return

        # Extract visited nodes from current path to detect cycles
        visited_in_path = set([node])
        for edge in history:
            visited_in_path.add(edge[0])
            visited_in_path.add(edge[1])

        # Random permutation of children for THIS trial
        perm_children = self.rng.permutation(children).tolist()

        # Optization: compute value_beofre once and reuse
        current_value = self._evaluate_system(history,x_foreground,x_background)

        for child in perm_children:
            # Skip if child creates a cycle (already in current path)
            if child in visited_in_path:
                continue
                
            edge = (node, child)
            new_history = history + [edge]

            # Compute marginal contribution of adding this edge
            value_after = self._evaluate_system(new_history, x_foreground, x_background)
            marginal = value_after - current_value

            # Accumulate to edge attributio (will average later)
            logging.debug(f"    Edge {edge} marginal contribution: {marginal}")
            self.edge_attributions[edge] += marginal

            # Update current value for next iteration (sequentail reuse)
            current_value = value_after

            # Recurse with edge in history
            self._dfs(child, new_history, x_foreground, x_background)


    def compute(self, x_foreground: Dict[int, float],
                x_background: Dict[int, float]) -> Dict[Tuple[ int, int], float]:
        """
        Compute Shapley Flow Edge attributions.

        Runs n_samples trials, where each trial does a DFS traversal
        with random permutations of children at each node.

        Parameters:
        -----------
        x_foreground: Dict[int, float]
            Foreground values for source nodes
        x_background: Dict[int, float]
            Background values for source nodes
        
        Returns:
        --------
        attributions: Dict[Tuple[int, int], float]
            Edge attribution mapping (u,v) -> importance score
        """

        for edge in self.edge_attributions:
            self.edge_attributions[edge] = 0.0

        # Progress logging for debugging
        import sys
        import logging
        n_edges = len(self.edge_attributions)
        
        if self.use_mc_permutation:
            logging.info(f"    → ShapleyFlow (MC Edge Permutation): {n_edges} edges, {self.n_samples} trials")
        else:
            logging.info(f"    → ShapleyFlow (Exhaustive DFS): {n_edges} edges, {len(self.source_nodes)} sources, {self.n_samples} trials")
        sys.stdout.flush()

        if self.use_mc_permutation:
            # CORRECT MC PERMUTATION MODE
            # Permute ALL edges randomly every trial → every edge participates in
            # every permutation → no visitation-frequency bias → efficiency = 1.00×.
            #
            # WHY unconstrained (not causal-depth-ordered):
            # _evaluate_system determines a node's fg/bg status from its INCOMING
            # edges.  If we enforce causal order (parents' edges before children's),
            # a non-source node Xi is already foreground (from its parent edge) by
            # the time (Xi→Y) fires.  The model already sees Xi at fg → marginal of
            # (Xi→Y) ≈ 0 → every non-source direct Y-parent gets zero attribution.
            #
            # With unconstrained ordering, roughly half the permutations will place
            # (Xi→Y) BEFORE (Xparent→Xi).  In those trials Xi is still at background
            # when (Xi→Y) fires, so the special-case branch in _evaluate_system sets
            # Xi=fg and the marginal captures Xi's full contribution to Y.  Averaged
            # over N trials, every edge gets a non-zero, unbiased estimate.
            #
            # Cost: n_samples × (n_edges + 1) model calls (e.g. 50 × 276 = 13,800).
            all_edges = [(u, v) for u, children in self.graph.items() for v in children]
            logging.info(
                f"    → ShapleyFlow (MC Permutation): {len(all_edges)} edges, "
                f"{self.n_samples} trials = {self.n_samples * (len(all_edges) + 1):,} model calls"
            )
            sys.stdout.flush()

            for trial in range(self.n_samples):
                self.eval_count = 0

                # Uniform random permutation of ALL edges (unbiased Shapley estimator)
                perm_idx = self.rng.permutation(len(all_edges))
                perm_edges = [all_edges[i] for i in perm_idx]

                prev_v = self._evaluate_system([], x_foreground, x_background)
                history = []
                for edge in perm_edges:
                    history.append(edge)
                    curr_v = self._evaluate_system(history, x_foreground, x_background)
                    self.edge_attributions[edge] += curr_v - prev_v
                    prev_v = curr_v

                if (trial + 1) % 10 == 0 or trial == 0:
                    logging.info(f"    → Trial {trial + 1}/{self.n_samples} completed ({self.eval_count} evaluations)")
                    sys.stdout.flush()

            # Average across trials (one permutation per trial → divide by n_samples)
            for edge in self.edge_attributions:
                self.edge_attributions[edge] /= self.n_samples
                
        else:
            # EXHAUSTIVE DFS MODE - original algorithm
            for trial in range(self.n_samples):
                # Reset eval count for this trial
                self.eval_count = 0
                
                # Each trial: DFS form each source with random permutations
                for source in self.source_nodes:
                    self._dfs(source,[], x_foreground, x_background)
                
                # Progress indicator every 10 trials
                if (trial + 1) % 10 == 0 or trial == 0:
                    logging.info(f"    → Trial {trial + 1}/{self.n_samples} completed ({self.eval_count} evaluations)")
                    sys.stdout.flush()
            
            # Average across trials
            for edge in self.edge_attributions:
                self.edge_attributions[edge] /= self.n_samples


        # Sanity check: verify efficiency axiom 
        total_attribution = sum(self.edge_attributions.values())
        n_eval_samples = self.n_samples
        f_x_samples = []
        f_x_prime_samples = []
        for _ in range(n_eval_samples):
            f_x_samples.append(self._evaluate_system(list(self.edge_attributions.keys()),
                                                     x_foreground, x_background))
            f_x_prime_samples.append(self._evaluate_system([],x_foreground, x_background))
        f_x = np.mean(f_x_samples)
        f_x_prime = np.mean(f_x_prime_samples)
        expected_total = f_x - f_x_prime
        relative_error = abs(total_attribution - expected_total) / (abs(expected_total) + 1e-10)

        # Debug output removed for cleaner output
        logging.info(f" Total edge attributions: {total_attribution:.6f}")
        logging.info(f" Expected (f(x) - f(x')) : {expected_total:.6f}")
        logging.info(f" Difference: {abs(total_attribution - expected_total)}" 
              f" using {n_eval_samples} samples to calculate f(x) and f(x')")
        logging.info(f" Relative error: {relative_error:.4f}")

        return self.edge_attributions
    
    def get_node_attributions(self) -> Dict[int, float]:
        """
        Aggregate edge attributions to get node-level importance

        Returns: 
        --------
        node_attr : Dict[int, float]
            Node importance scores (sum of outgoing edge attributions)
        """

        node_attr = {}
        for (u, v), score in self.edge_attributions.items():
            node_attr[u] = node_attr.get(u, 0.0) + score
        return node_attr
    
class ShapleyFlowWrapper:
    """Wrapper for ShapleyFlow with source filtering (but NOT edge filtering).
    
    ═══════════════════════════════════════════════════════════════════════════════
    WHAT THIS WRAPPER ACTUALLY DOES (based on code implementation):
    ═══════════════════════════════════════════════════════════════════════════════
    
    1. Converts causal adjacency matrix → graph structure (adjacency list)
    2. **FILTERS SOURCES** using backward BFS from sink (Y-reachable sources only)
    3. **DOES NOT FILTER EDGES** - uses full adjacency matrix
    4. Creates ShapleyFlow instance with filtered sources
    5. Aggregates edge attributions to node-level importance
    
    ═══════════════════════════════════════════════════════════════════════════════
    FILTERING STRATEGY (Source Filtering Only):
    ═══════════════════════════════════════════════════════════════════════════════
    
    WHAT IS FILTERED:
    - Source nodes: Only keeps sources that can reach sink Y (backward BFS)
    
    WHAT IS NOT FILTERED:
    - Adjacency matrix: Uses ALL edges from original causal graph
    - Intermediate nodes: All nodes in graph are kept
    
    ALGORITHM (lines 2408-2437):
    ```
    # Build parent dict
    parents[i] ← [j where causal_graph[j,i] ≠ 0]
    
    # Find potential sources (no parents)
    potential_sources ← [i where len(parents[i]) == 0]
    
    # Backward BFS from sink Y
    reachable_from_Y ← backward_BFS(Y, parents)
    
    # Filter sources
    source_nodes ← potential_sources ∩ reachable_from_Y
    
    # Build FULL graph structure (NO edge filtering!)
    graph_structure[i] ← [j where causal_graph[i,j] ≠ 0]  # All edges kept
    ```
    
    ═══════════════════════════════════════════════════════════════════════════════
    COMPARISON WITH GraphExplainerWrapper:
    ═══════════════════════════════════════════════════════════════════════════════
    
    ShapleyFlowWrapper:
    - Filters: SOURCES only
    - Adjacency: Full original matrix
    - Graph size: Same as input
    
    GraphExplainerWrapper:
    - Filters: ADJACENCY MATRIX (zeros out non-reachable edges)
    - Graph size: Potentially much smaller
    - More aggressive filtering → faster computation
    
    ═══════════════════════════════════════════════════════════════════════
    KEY PARAMETERS:
    ═══════════════════════════════════════════════════════════════════════
    
    - causal_graph: Adjacency matrix including ALL variables (features + Y)
        * Shape: (n_features+1, n_features+1) where last index is Y
        * causal_graph[i,j]=1 means variable i causes variable j
    
    - y_index: Position of outcome variable Y in the causal graph
        * Typically the last index: y_index = n_features
        * Needed to identify which node is the prediction target
    
    ═══════════════════════════════════════════════════════════════════════
    NODE vs FEATURE IMPORTANCE:
    ═══════════════════════════════════════════════════════════════════════
    
    Shapley Flow produces EDGE attributions: importance(u → v)
    
    To get feature-level importance (compatible with other explainers):
    - Aggregate outgoing edges: importance(X_i) = Σ_{j} importance(X_i → X_j)
    - Excludes Y node (Y always has 0 importance since it's the outcome)
    
    ═══════════════════════════════════════════════════════════════════════
    
    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn compatible model
    background_data : pd.DataFrame
        Reference dataset for conditional sampling
    causal_graph : np.ndarray
        Adjacency matrix (n_features+1, n_features+1) including Y
        Entry [i,j]=1 means variable i causes variable j
    y_index : int
        Index of outcome variable Y in the causal graph
    n_samples : int, default=100
        Number of Monte Carlo trials
    random_state : int or None
        Random seed
    use_mc_permutation : bool, default=True
        If True, permute ALL edges uniformly each trial (unbiased Shapley estimator).
        If False, use exhaustive DFS traversal.
    paths_per_source : int, default=100
        Unused when use_mc_permutation=True; retained for the DFS fallback mode.
    
    Attributes
    ----------
    directed_graph : np.ndarray
        Binary adjacency matrix extracted from causal_graph
    graph_structure : Dict[int, List[int]]
        Adjacency list representation
    source_nodes : List[int]
        Nodes with no parents (input features)
    shap_values : np.ndarray
        Node-level importance scores (n_samples x n_input_features)
        Excludes Y since it always has 0 importance
    
    Examples
    --------
    >>> # Causal graph: X0→X1→Y, X2→Y (Y is last variable)
    >>> causal_dag = np.array([[0,1,0,0], [0,0,0,1], [0,0,0,1], [0,0,0,0]])
    >>> wrapper = ShapleyFlowWrapper(model, X_train, causal_dag, y_index=3)
    >>> shap_values = wrapper.explain(X_test)
    >>> importance = wrapper.get_feature_importance()
    """

    def __init__(self, model: BaseEstimator,
                background_data: pd.DataFrame,
                causal_graph: np.ndarray,
                y_index: int,
                n_samples: int = 100,
                random_state: Optional[int] = None,
                use_mc_permutation: bool = True,
                paths_per_source: int = 100):
        """Initialize Shapley Flow wrapper for ML models."""
        self.model = model
        self.background_data_df = background_data
        self.background_data = background_data.values
        self.features_names = background_data.columns.tolist()
        self.n_features = len(self.features_names)
        self.y_index = y_index
        self.n_samples = n_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.use_mc_permutation = use_mc_permutation
        self.paths_per_source = paths_per_source

        # Extract directed graph 
        self.directed_graph = self._extract_directed_graph(causal_graph)

        # Build graph structure for ShapleyFlow
        self.graph_structure = {}
        for i in range(self.n_features):
            children = [j for j in range(self.n_features) if self.directed_graph[i,j] !=0]
            self.graph_structure[i] = children

        # Build reverse graph (parents for each node) for backward BFS
        parents = {}
        for i in range(self.n_features):
            parents[i] = [j for j in range(self.n_features) if self.directed_graph[j, i] != 0]

        # Find all potential source nodes (no parents)
        potential_sources = []
        for i in range(self.n_features):
            if len(parents[i]) == 0:
                potential_sources.append(i)

        # Filter sources to only those that can reach the sink (y_index)
        # Use backward BFS from sink (more efficient than forward path finding)
        if y_index is not None and potential_sources:
            # Backward BFS from sink to find all nodes that can reach it
            reachable_from_sink = set()
            queue = [y_index]
            visited = {y_index}
            
            while queue:
                current = queue.pop(0)
                reachable_from_sink.add(current)
                # Add all parents (nodes that have edges to current)
                for parent in parents.get(current, []):
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)
            
            # Filter sources to only those reachable from sink
            self.source_nodes = [s for s in potential_sources if s in reachable_from_sink]
            
            if not self.source_nodes:
                warnings.warn(f"No source nodes can reach sink node {y_index}. Using all potential sources.")
                self.source_nodes = potential_sources
        elif not potential_sources:
            # No sources found (e.g., cycles), use all non-Y nodes as sources
            warnings.warn("No source nodes found (graph may have cycles). Using all non-Y nodes as sources.")
            self.source_nodes = [i for i in range(self.n_features) if i != y_index]
        else:
            # No sink specified, use all potential sources
            self.source_nodes = potential_sources

        # Log edge count for diagnostics
        import sys
        import logging
        n_edges = sum(len(v) for v in self.graph_structure.values())
        logging.info(f"    → ShapleyFlowWrapper: {self.n_features} features, {n_edges} edges, {len(self.source_nodes)} sources")
        sys.stdout.flush()

    def _extract_directed_graph(self, causal_graph: np.ndarray) -> np.ndarray:
        """Validate and return binary adjacency matrix.
        
        Note: Assumes causal_graph is already a binary adjacency matrix (0s and 1s)
        as guaranteed by the causal discovery step.
        """
        if causal_graph.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match features"
            )
        
        # Validate binary adjacency matrix
        unique_vals = set(np.unique(causal_graph).tolist())
        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"causal_graph must be binary (0s and 1s), got values: {unique_vals}"
            )
        
        return causal_graph.astype(int)


    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute Shapley Flow values for instance.

        Parameters:
        ----------
        X : pd.DataFrame
            Instances to explain

        Return:
        ------
        shap_values : np.ndarray
            Node-level importance scores (n_samples x n_features)
            Only includes input features, excludes Y since it always has 0 attribution
        """
        X_values = X.values
        n_instances = len(X_values)

        # Exclude Y from shap_Values since it always has 0 attribution and match other class outputs
        n_input_features = self.n_features - 1
        self.shap_values = np.zeros((n_instances, n_input_features))
        
        # Create mapping from node idex to shap_values column index
        # All features before Y keep their. index, features after Y shift down by 1
        self.node_to_shap_idx = {}
        shap_idx = 0
        for node_idx in range(self.n_features):
            if node_idx != self.y_index:
                self.node_to_shap_idx[node_idx] = shap_idx
                shap_idx +=1

        # Progress logging
        import sys
        import logging
        logging.info(f"    → Computing Shapley Flow for {n_instances} instances...")
        sys.stdout.flush()

        # Create ShapleyFlow object once (reused across all instances)
        flow = ShapleyFlow(
            graph_structure=self.graph_structure,
            background_data=self.background_data,
            model=self.model,
            source_nodes=self.source_nodes,
            sink_node=self.y_index,
            n_samples=self.n_samples,
            random_state=self.random_state,
            feature_names=self.features_names,  # Pass feature names for alignment
            use_mc_permutation=self.use_mc_permutation,
            paths_per_source=self.paths_per_source
        )

        for i, instance in enumerate(X_values):
            # Progress indicator
            if n_instances > 1:
                logging.info(f"    → Instance {i + 1}/{n_instances}")
                sys.stdout.flush()

            # Prepare forground and background
            x_foreground = {j: instance[j] for j in range(self.n_features)}

            # Sample random background 
            bg_idx = self.rng.randint(0, len(self.background_data))
            x_background = {j: self.background_data[bg_idx,j] for j in range(self.n_features)}

            # Compute edge attributions
            edge_attrs = flow.compute(x_foreground,x_background)

            node_attrs = flow.get_node_attributions()

            for node_idx, score in node_attrs.items():
                if node_idx != self.y_index:
                    self.shap_values[i, self.node_to_shap_idx[node_idx]] = score
        
        logging.info(f"    → All instances completed")
        sys.stdout.flush()
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get mean absolute importance per feature (excluding Y)"""
        if not hasattr(self, 'shap_values'):
            raise ValueError("Must call explain() first")

        mean_abs_values = np.abs(self.shap_values).mean(axis=0)

        features_names_without_y = [name for i, name in enumerate(self.features_names)
                                    if i!=self.y_index]

        importance_df = pd.DataFrame({
            'feature' : features_names_without_y,
            'importance' : mean_abs_values
        })

        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)


class GraphExplainerWrapper:
    """Wrapper for shapflow's GraphExplainer with automatic adjacency filtering.
    
    This class integrates the complete pipeline for causal Shapley value computation:
    1. Filters adjacency matrix to only Y-reachable edges (backward BFS)
    2. Constructs causal graph from filtered adjacency
    3. Learns causal functions from training data
    4. Computes Shapley values using GraphExplainer
    5. Returns values in standard array format compatible with other methods
    
    The filtering step zeros out edges that don't contribute to the target variable,
    significantly reducing computational cost while maintaining data compatibility.
    
    IMPORTANT: Requires shapflow library with Node, Graph, and GraphExplainer classes.
    
    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Adjacency matrix (n × n) where adj[i,j] != 0 means feature i causes feature j.
        Must include the target variable Y as the last row/column.
    feature_names : list of str
        List of all feature names INCLUDING the target (e.g., ['X0', 'X1', ..., 'Y']).
        Must match the order in adjacency_matrix.
    train_data : pd.DataFrame
        Training data for learning causal functions. Must contain all features in feature_names.
    background_data : pd.DataFrame
        Background data for GraphExplainer baseline. Typically 100-1000 samples from train data.
    sink_name : str, default='Y'
        Name of the target/sink node to explain.
    nruns : int, default=100
        Number of Monte Carlo samples for GraphExplainer.
    silent : bool, default=False
        If True, suppress GraphExplainer output.
    method : str, default='divide_and_conquer'
        GraphExplainer method: 'divide_and_conquer', 'bruteforce_sampling', etc.
    fit_method : str, default='xgboost'
        Method for learning causal functions: 'xgboost', 'linear', etc.
    
    Attributes
    ----------
    adjacency_matrix : np.ndarray
        Original adjacency matrix (unfiltered)
    filtered_adjacency : np.ndarray
        Filtered adjacency with irrelevant edges zeroed
    feature_names : list
        List of all feature names (including target)
    features_names_without_y : list
        List of feature names excluding target (for SHAP array output)
    graph : shapflow.flow.Graph
        Constructed causal graph with learned functions
    explainer : shapflow.flow.GraphExplainer
        Initialized GraphExplainer instance
    filter_stats : dict
        Statistics about adjacency filtering
    shap_values : np.ndarray or None
        Computed SHAP values after calling explain()
    
    Examples
    --------
    >>> # Initialize with adjacency matrix and data
    >>> wrapper = GraphExplainerWrapper(
    ...     adjacency_matrix=adj_matrix,
    ...     feature_names=['X0', 'X1', 'X2', 'Y'],
    ...     train_data=train_df,
    ...     background_data=background_df,
    ...     nruns=100
    ... )
    >>> 
    >>> # Compute SHAP values for test instances
    >>> shap_values = wrapper.explain(test_df)
    >>> # Returns array of shape (n_test_instances, n_features)
    >>> 
    >>> # Get feature importance
    >>> importance = wrapper.get_feature_importance()
    """
    
    def __init__(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        train_data: pd.DataFrame,
        background_data: pd.DataFrame,
        sink_name: str = 'Y',
        nruns: int = 100,
        silent: bool = False,
        method: str = 'bruteforce_sampling',
        fit_method: str = 'xgboost'
    ):
        """Initialize GraphExplainerWrapper with filtered adjacency and learned causal graph."""
        
        # Import shapflow here to avoid hard dependency
        try:
            from shapflow.flow import Node, Graph, GraphExplainer
            self.Node = Node
            self.Graph = Graph
            self.GraphExplainer = GraphExplainer
        except ImportError:
            raise ImportError(
                "shapflow library is required for GraphExplainerWrapper. "
                "Please install it or add it to your Python path."
            )
        
        # Store original inputs
        self.adjacency_matrix = adjacency_matrix.copy()
        self.feature_names = feature_names.copy()
        self.sink_name = sink_name
        self.nruns = nruns
        self.silent = silent
        self.method = method
        self.fit_method = fit_method
        self.shap_values = None
        
        # Validate inputs
        if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
            raise ValueError(f"adjacency_matrix must be square, got {adjacency_matrix.shape}")
        
        if len(feature_names) != adjacency_matrix.shape[0]:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"adjacency_matrix size ({adjacency_matrix.shape[0]})"
            )
        
        if sink_name not in feature_names:
            raise ValueError(f"sink_name '{sink_name}' not found in feature_names")
        
        # Get feature names without target (for SHAP array output)
        self.features_names_without_y = [f for f in feature_names if f != sink_name]
        
        # Step 1: Filter adjacency matrix to only Y-reachable edges
        logging.info("Filtering adjacency matrix using backward BFS...")
        self.filtered_adjacency, _, self.reachable_indices, self.filter_stats = \
            self._filter_adjacency_to_sink_reachable(adjacency_matrix, feature_names, sink_name)
        
        # Step 2: Convert filtered adjacency to causal graph
        logging.info("Building causal graph from filtered adjacency...")
        self.graph = self._adjacency_to_graph(
            self.filtered_adjacency, 
            feature_names, 
            train_data,
            fit_method
        )
        
        # Step 3: Initialize GraphExplainer
        logging.info("Initializing GraphExplainer...")
        self.explainer = self.GraphExplainer(
            graph=self.graph,
            bg=background_data,
            nruns=nruns,
            silent=silent
        )
        
        logging.info("GraphExplainerWrapper initialized successfully!")
        logging.info(f"  - Total edges: {self.filter_stats['total_edges']}")
        logging.info(f"  - Edges kept: {self.filter_stats['edges_kept']} "
                    f"({self.filter_stats['percentage_edges_kept']:.1f}%)")
        logging.info(f"  - Computational savings: ~{100 - self.filter_stats['percentage_edges_kept']:.1f}%")
    
    def _filter_adjacency_to_sink_reachable(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        sink_name: str
    ) -> Tuple[np.ndarray, List[str], List[int], Dict]:
        """Filter adjacency matrix by zeroing out edges not reachable from sink.
        
        ═══════════════════════════════════════════════════════════════════════════
        CRITICAL: This is ADJACENCY MATRIX filtering (not just source filtering)
        ═══════════════════════════════════════════════════════════════════════════
        
        ALGORITHM:
        1. Backward BFS from sink to find all Y-reachable nodes
        2. Zero out edges where EITHER endpoint is not Y-reachable
        3. Return filtered matrix (same size, but with zeroed edges)
        
        PSEUDOCODE:
        ```
        # Build parent dict
        parents[j] ← [i where adjacency[i,j] ≠ 0]
        
        # Backward BFS from sink
        reachable ← {sink}
        queue ← [sink]
        WHILE queue not empty:
            current ← queue.pop()
            FOR each parent in parents[current]:
                IF parent not in reachable:
                    reachable.add(parent)
                    queue.append(parent)
        
        # Zero out edges NOT between reachable nodes
        filtered ← adjacency.copy()
        FOR i, j in all_edges:
            IF i not in reachable OR j not in reachable:
                filtered[i,j] ← 0
        ```
        
        USED BY:
        - GraphExplainerWrapper: Filters adjacency before building graph
        - NOT used by ShapleyFlowWrapper (which only filters sources)
        
        Returns
        -------
        filtered_adjacency : np.ndarray
            Adjacency matrix with non-reachable edges zeroed (same size as input)
        feature_names : list
            Same feature names as input (unchanged for data compatibility)
        reachable_indices : list
            Indices of nodes reachable from sink via backward traversal
        stats : dict
            Filtering statistics (edges removed, nodes kept, etc.)
        """
        n_features = adjacency_matrix.shape[0]
        sink_idx = feature_names.index(sink_name)
        
        # Build parent dictionary from adjacency matrix
        parents = {}
        for j in range(n_features):
            parents[j] = [i for i in range(n_features) if adjacency_matrix[i, j] != 0]
        
        # Backward BFS from sink - find all nodes that can reach the sink
        reachable_indices = set()
        queue = [sink_idx]
        visited = {sink_idx}
        
        while queue:
            current_idx = queue.pop(0)
            reachable_indices.add(current_idx)
            
            # Add all parents (predecessors) of current node
            for parent_idx in parents.get(current_idx, []):
                if parent_idx not in visited:
                    visited.add(parent_idx)
                    queue.append(parent_idx)
        
        # Create filtered adjacency matrix (same size as original)
        # Copy the original, then zero out edges NOT between reachable nodes
        filtered_adjacency = adjacency_matrix.copy()
        reachable_set = set(reachable_indices)
        
        for i in range(n_features):
            for j in range(n_features):
                # If either node is not reachable from Y, zero out the edge
                if i not in reachable_set or j not in reachable_set:
                    filtered_adjacency[i, j] = 0
        
        # Calculate statistics
        n_total_edges = int(np.sum(adjacency_matrix != 0))
        n_filtered_edges = int(np.sum(filtered_adjacency != 0))
        n_reachable = len(reachable_indices)
        
        stats = {
            'total_nodes': n_features,
            'nodes_kept': n_reachable,
            'nodes_filtered': n_features - n_reachable,
            'total_edges': n_total_edges,
            'edges_kept': n_filtered_edges,
            'edges_filtered': n_total_edges - n_filtered_edges,
            'percentage_nodes_kept': 100 * n_reachable / n_features,
            'percentage_edges_kept': 100 * n_filtered_edges / max(1, n_total_edges)
        }
        
        return filtered_adjacency, feature_names, sorted(list(reachable_indices)), stats
    
    def _adjacency_to_graph(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        train_data: pd.DataFrame,
        fit_method: str = 'xgboost'
    ):
        """Convert adjacency matrix to shapflow Graph object with learned causal functions.
        
        Parameters
        ----------
        adjacency_matrix : np.ndarray
            Adjacency matrix where adj[i, j] != 0 means feature i is parent of feature j
        feature_names : list
            List of feature names
        train_data : pd.DataFrame
            Training data to fit causal functions
        fit_method : str
            Method for learning causal functions ('xgboost', 'linear', etc.)
        
        Returns
        -------
        graph : shapflow.flow.Graph
            Graph object with learned causal functions
        """
        n_features = len(feature_names)
        
        # Step 1: Create all nodes first (without parent relationships)
        nodes_dict = {}
        for i, name in enumerate(feature_names):
            # Determine if this is the target node
            is_target = (name == self.sink_name)
            
            # Create node
            node = self.Node(
                name=name,
                f=None,
                args=[],
                is_target_node=is_target,
                is_noise_node=False,
                is_dummy_node=False,
                is_categorical=False
            )
            nodes_dict[name] = node
        
        # Step 2: Add parent relationships based on adjacency matrix
        for j, child_name in enumerate(feature_names):
            child_node = nodes_dict[child_name]
            
            # Find parents: where adjacency[i, j] != 0 means i is parent of j
            parent_indices = np.where(adjacency_matrix[:, j] != 0)[0]
            
            # Add each parent to the child's args
            for parent_idx in parent_indices:
                parent_name = feature_names[parent_idx]
                parent_node = nodes_dict[parent_name]
                child_node.args.append(parent_node)
                # Also add child to parent's children list
                if child_node not in parent_node.children:
                    parent_node.children.append(child_node)
        
        # Step 3: Create Graph object
        graph = self.Graph(nodes=list(nodes_dict.values()))
        
        # Step 4: Fit missing links (learn causal functions from data)
        if not self.silent:
            logging.info(f"Learning causal functions using {fit_method}...")
        graph.fit_missing_links(train_data, method=fit_method)
        
        return graph
    
    def _get_node_attributions(self, edge_credit: Dict) -> Dict[str, np.ndarray]:
        """Aggregate edge attributions to get node-level importance.
        
        Parameters
        ----------
        edge_credit : dict
            Dictionary of edge credits from GraphExplainer
        
        Returns
        -------
        node_attr : dict
            Dictionary mapping feature names to arrays of SHAP values
        """
        node_attr = {}
        for node1, d in edge_credit.items():
            if "noise" not in node1.name:
                for node2, val in d.items():
                    node_attr[node1.name] = node_attr.get(node1.name, 0.0) + val
        return node_attr
    
    def _convert_node_attributions_to_array(
        self,
        node_attributions: Dict[str, np.ndarray],
        feature_names: List[str]
    ) -> np.ndarray:
        """Convert node attributions dictionary to standard SHAP format array.
        
        Parameters
        ----------
        node_attributions : dict
            Dictionary {feature_name: array_of_shap_values}
        feature_names : list
            List of all feature names in correct order (excluding target)
        
        Returns
        -------
        shap_array : np.ndarray
            Array of shape (n_instances, n_features) with SHAP values
        """
        # Get number of instances from any feature in the dictionary
        if len(node_attributions) > 0:
            first_feature = list(node_attributions.keys())[0]
            n_instances = len(node_attributions[first_feature])
        else:
            raise ValueError("node_attributions is empty")
        
        n_features = len(feature_names)
        
        # Initialize array with zeros
        shap_array = np.zeros((n_instances, n_features))
        
        # Fill in SHAP values for features that have them
        for i, feature_name in enumerate(feature_names):
            if feature_name in node_attributions:
                shap_array[:, i] = node_attributions[feature_name]
            # else: remains 0
        
        return shap_array
    
    def explain(self, foreground_data: pd.DataFrame) -> np.ndarray:
        """Compute Shapley values for foreground instances.
        
        Parameters
        ----------
        foreground_data : pd.DataFrame
            Instances to explain. Must contain all features in feature_names.
        
        Returns
        -------
        shap_values : np.ndarray
            Array of shape (n_instances, n_features) with SHAP values.
            Features are in the same order as the original data (excluding target).
        """
        logging.info(f"Computing Shapley values for {len(foreground_data)} instances...")
        
        # Compute SHAP values using GraphExplainer
        cf = self.explainer.shap_values(
            X=foreground_data,
            method=self.method
        )
        
        # Aggregate edge credits to node-level attributions
        node_attributions = self._get_node_attributions(cf.edge_credit)
        
        # Convert to standard array format (n_instances × n_features)
        self.shap_values = self._convert_node_attributions_to_array(
            node_attributions,
            self.features_names_without_y
        )
        
        logging.info(f"✅ SHAP values computed: {self.shap_values.shape}")
        logging.info(f"   Features with non-zero values: "
                    f"{np.sum(np.any(self.shap_values != 0, axis=0))}/{self.shap_values.shape[1]}")
        
        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get mean absolute importance per feature (excluding target).
        
        Returns
        -------
        importance_df : pd.DataFrame
            DataFrame with 'feature' and 'importance' columns, sorted by importance
        """
        if self.shap_values is None:
            raise ValueError("Must call explain() first")
        
        mean_abs_values = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.features_names_without_y,
            'importance': mean_abs_values
        })
        
        return importance_df.sort_values('importance', ascending=False).reset_index(drop=True)


