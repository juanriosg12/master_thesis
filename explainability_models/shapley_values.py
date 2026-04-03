import numpy as np
import math
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple, Any
from itertools import chain, combinations, permutations
import warnings
warnings.filterwarnings('ignore')

import shap

from sklearn.base import BaseEstimator
import networkx as nx


class ShapleyExplainer: 
    
    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 method: str ='auto'):
        
        self.model = model 
        self.background_data = background_data
        self.feature_names = background_data.columns.tolist()
        self.method = method
        self.explainer = None
        self.shap_values = None

        self._initialize_explainer()

    def _initialize_explainer(self):

        model_type = type(self.model).__name__

        if self.method == 'auto':

            if 'LGBM' in model_type or 'LightGBM' in model_type:

                self.explainer = shap.TreeExplainer(self.model)
                self.method = 'tree'
            elif hasattr(self.model, 'tree_') or 'RandomForest' in model_type or 'GradientBoosting' in model_type:
                
                self.explainer = shap.TreeExplainer(self.model)
                self.method = 'tree'
            elif hasattr(self.model, 'coef_'):
                # linear model
                self.explainer = shap.LinearExplainer(self.model, self.background_data)
                self.method = 'linear'
            else:
                background_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
                self.explainer = shap.KernelExplainer(self.model.predct, background_sample)
                self.method = 'kernel'
        elif self.method == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        elif self.method == 'kernel':
            background_sample = shap.sample(self.background_data, min(100, len(self.background_data)))
            self.explainer = shap.KernelExplainer(self.model.predct, background_sample)
        elif self.method == 'linear':
            self.explainer= shap.LinearExplainer(self.model, self.background_data)
        else:
            raise ValueError(f"Unknown SHAP method: {self.method}")
        
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        
        print(f"Computing SHAP values using {self.method} explainer...")

        self.shap_values = self.explainer.shap_values(X)

        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        return self.shap_values
        
    def get_feature_importance(self) -> pd.DataFrame:

        if self.shap_values is None:
            raise ValueError("Must call explain() first to compute SHAP values")
        
        importance_scores = np.abs(self.shap_values).mean(axis=0)

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance',ascending=False)

        return importance
    
    def get_shap_values_df(self, X: pd.DataFrame) ->pd.DataFrame:

        if self.shap_values is None:
            self.explain(X)

        return pd.DataFrame(self.shap_values,columns= self.feature_names, index=X.index)
    
class ShapleyFromScratch:

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

        """
        Predict with only features in coalition, marginalizing over others

        Parameters: 
        -----------
        instance : np.ndarray
            Instance to explain (1D array)
        coalition: List[int]
            Indices of features in the coalition

        Returns:
        -------
        prediction : float
            Average prediction over background samples
        """
        # Create samples with coalition features from instance, others from background
        samples = self.background_data.copy()

        for feature_idx in coalition:
            samples[:, feature_idx] = instance[feature_idx]

        # Average predictions with proper feature alignment
        predictions = self._predict_with_feature_alignment(samples)
        return predictions.mean()
    
    def _compute_exact_shapley(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute exact Shapley values using all possible coalitions.

        Warning: Exponential complexity O(2^n). Only use for small n (<=10)
        
        Parameters:
        -----------
        instance : np.ndarray
            Instance to explain
        Returns:
        ----------
        shapley_values : np.ndarray
            Exact Shapley values
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
        """
        Compute approximate Shapley values using Monte Carlo sampling.

        Uses random permutations to estimate Shapley values efficiently

        Parameters:
        -----------
        instance: nd.ndarray
            Instance to explain

        Returns:
        ----------
        shapley_values : np.ndarray
            Approximate Shapley values
        """
        shapley_values = np.zeros(self.n_features)

        # Sample random permutations
        for _ in range(self.n_samples):
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
        
        # Average over samples 
        shapley_values /= self.n_samples

        return shapley_values
    
    def explain(self, X: pd.DataFrame, method: str = 'monte_carlo') -> np.ndarray:
        """
        Calculate Shapley values for given instances.

        Parameters:
        ----------
        X : pd.DataFrame
            Instances to explain
        method : str
            'exact' for exact calculation (slow, only for n_features <=10)
            'monte_carlo' for approximation (fast)

        Returns:
        --------
        shapley_values : np.ndarray
            Shapley values (shape: n_samples x n_features)
        """
        print(f"Computing Shapley values from scratch using {method} method...")

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
            
            # Progress indicator removed for cleaner output
            # if (i + 1) % max(1, n_instances //10) == 0:
            #     print(f" Progress: {i + 1}/{n_instances} instances")

        return self.shap_values
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance."""
        if self.shap_values is None:
            raise ValueError("Must call expalin( first to compute Shapley values")
        
        importance_scores = np.abs(self.shap_values).mean(axis=0)

        importance = pd.DataFrame({
            "feature": self.feature_names,
            'importance' : importance_scores
        }).sort_values('importance', ascending=False)

        return importance

    def get_shap_values_df(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get Shapley values as DataFrame"""
        if self.shap_values is None:
            self.explain(X)

        return pd.DataFrame(self.shap_values, columns=self.feature_names, index= X.abs)
    

class TrueShapley(ShapleyFromScratch):
    """
    Compute Shapley values using the true data generation function instead of a model.
    This provides ground truth feature importance for validation purposes.
    """
    
    def __init__(self, true_generator: Callable, background_data: pd.DataFrame,
                 n_samples: int = 1000, random_state: Optional[int] = None):
        """
        Args:
            true_generator: Callable that takes X (n_samples, n_features) and returns Y (n_samples,)
            background_data: Background dataset for Shapley computation
            n_samples: Number of Monte Carlo samples for Shapley approximation
            random_state: Random seed for reproducibility
        """
        # We don't have a model, so we'll pass None and override _predict_coalition
        self.true_generator = true_generator
        self.background_data = background_data
        self.feature_names = background_data.columns.tolist()
        self.n_features = len(self.feature_names)
        self.n_samples = n_samples
        
        # Set random state
        if random_state is not None:
            np.random.seed(random_state)
            self.rng = np.random.RandomState(random_state)
        else:
            self.rng = np.random.RandomState()
        
        # Compute baseline value as expected output over background data
        self.baseline_value = np.mean(self.true_generator(background_data.values))
        
        self.shap_values = None
    
    def _predict_coalition(self, instance: np.ndarray, coalition: list) -> float:
        """
        Predict using true generator for a coalition of features.
        
        Args:
            instance: Single instance to explain (n_features,)
            coalition: List of feature indices in the coalition
        
        Returns:
            Expected Y value for this coalition
        """
        # Create samples by combining instance features (in coalition) with random background features (not in coalition)
        samples = np.tile(instance, (self.n_samples, 1))
        
        # Convert coalition list to set for faster lookup
        coalition_set = set(coalition)
        
        # For features not in coalition, sample from background
        for j in range(self.n_features):
            if j not in coalition_set:
                # Sample random values from background for feature j
                random_indices = self.rng.choice(len(self.background_data), size=self.n_samples, replace=True)
                samples[:, j] = self.background_data.values[random_indices, j]
        
        # Compute Y using true generator and average
        predictions = self.true_generator(samples)
        return np.mean(predictions)


class AsymmetricShapley(ShapleyFromScratch):

    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 causal_graph: np.ndarray, n_samples: int = 1000,
                 random_sate: Optional [int] = None,
                 asymmetric_method: str = 'frye'):
        """
        Initialize Asymmetric Shapley explainer.
        
        Parameters:
        -----------
        asymmetric_method : str
            Method for sampling causal permutations:
            - 'frye': True Asymmetric Shapley (Frye et al.) - only enforces direct parent constraints
            - 'strict': Strict topological ordering by depth layers
        """
        
        super().__init__(model, background_data, n_samples,random_sate)
        
        if asymmetric_method not in ['frye', 'strict']:
            raise ValueError(f"asymmetric_method must be 'frye' or 'strict', got {asymmetric_method}")
        self.asymmetric_method = asymmetric_method

        self.causal_graph = causal_graph
        self.directed_graph = self._extract_directed_graph(causal_graph)
        if self._has_cycle(self.directed_graph):
            warnings.warn(
                "Directed causal constrains contain cycles."
                "Causal Shapley requires a DAG; disabling constraints."
            )
            self.directed_graph = np.zeros_like(self.directed_graph)

        self.parents = {}
        for j in range(self.n_features):
            self.parents[j] = set(np.where(self.directed_graph[:, j] != 0)[0])
            # Detailed parent info removed for cleaner output
            # print(f"Feature {self.feature_names[j]} has parents: {[self.feature_names[p] for p in self.parents[j]]}")

        self.topological_order = self._topological_sort()
        # Detailed structure info removed for cleaner output
        # print(f"Topological order of features: {self.topological_order}")
        # print(f"Using asymmetric method: {self.asymmetric_method}")

    def _extract_directed_graph(self, causal_graph: np.ndarray) -> np.ndarray:
        """
        Convert a generic causal graph matrix into directed adjacency (0/1)

        Support two common encodings:
        1) Binary adjacency where causal_graph[i, j] = 1 means i -> j
        2) causal-learn endpoint encoding in graph.graph where edege direction is
            encoded by asymmetric endpoint marks (e.g, -1/1 for tail/arrow)

        Ambiguous/non-directed edges (undirected, bidirected,circle endpoints,
            or summetric 1/1 edges) are ignored for parent constraints
        """

        if causal_graph.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match number"
                f"of features ({self.n_features}, {self.n_features})"
            )
        
        directed = np.zeros((self.n_features, self.n_features), dtype=int)

        # Case 1: likely plain binary adjacency matrix
        unique_vals = set(np.unique(causal_graph).tolist())
        if unique_vals.issubset({0,1}):
            for i in range(self.n_features):
                for j in range(self.n_features):
                    if i!=j and causal_graph[i, j]!=0 and causal_graph[j, i] ==0:
                        directed[i, j] =1
            return directed
        
        # Case 2: causal-learn endpoint encoding matrix
        for i in range(self.n_features):
            for j in range(i + 1, self.n_features):
                a = causal_graph[i, j] 
                b = causal_graph[j, i]

                #Directed edges: i -> j or j -> i
                if a == -1 and b == 1:
                    directed[i, j] = 1
                elif a == 1 and b == -1:
                    directed[j, i] = 1

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
        in_degree = np.sum(directed_graph != 0, axis = 0).astype(int)
        queue = [i for i in range(self.n_features) if in_degree[i] == 0]
        visited = 0

        while queue :
            node = queue.pop(0)
            visited+=1
            for child in range(self.n_features):
                if directed_graph[node, child] !=0:
                    in_degree[child] -=1
                    if in_degree[child] ==0:
                        queue.append(child)
        return visited != self.n_features

    def _topological_sort(self) -> List[int]:
        
        # Kahn's algorithm for topological sort
        in_degree = np.sum(self.directed_graph != 0 , axis= 0) # Count incoming edges
        queue = [i for i in range(self.n_features) if in_degree[i] == 0]
        order = []

        while queue:
            # sort for determinism
            queue.sort()
            node = queue.pop(0)
            order.append(node)

            for child in range(self.n_features):
                if self.directed_graph[node,child]!=0:
                    in_degree[child] -=1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(order) != self.n_features:
            warnings.warn("Causal graph contains cycles. Using arbitrary ordering.")
            order = list(range(self.n_features))

        return order
    
    def _is_valid_coalition(self, coalition: List[int]) -> bool:
        
        coalition_set = set(coalition)

        for feature in coalition:

            if not self.parents[feature].issubset(coalition_set):
                return False
        return True
    
    def _get_valid_coalitions(self, features: List[int]) -> List[List[int]]:

        all_subsets = chain.from_iterable(
            combinations(features,r) for r in range(len(features) + 1)
        )

        valid_coalitions = [
            list(subset) for subset in all_subsets
            if self._is_valid_coalition(list(subset))
        ]

        return valid_coalitions
    
    def _compute_exact_causal_shapley( self, instance: np.ndarray) -> np.ndarray:


        shapley_values = np.zeros(self.n_features)

        for i in range(self.n_features):
            other_features = [j for j in range(self.n_features) if j !=i ]

            valid_coalitions = self._get_valid_coalitions(other_features)

            marginal_contributions = []

            for coalition in valid_coalitions:

                coalition_with_i = coalition + [i]
                
                if not self._is_valid_coalition(coalition_with_i):

                    continue

                v_with = self._predict_coalition(instance, coalition_with_i)
                v_without = self._predict_coalition(instance, coalition)

                marginal = v_with - v_without

                coalition_size = len(coalition)
                weight = 1.0 / (self.n_features * math.comb(self.n_features -1, coalition_size))

                marginal_contributions.append(weight * marginal)

            shapley_values[i] = sum(marginal_contributions) if marginal_contributions else 0.0

        return shapley_values
    
    def _sample_causal_permutation_strict(self) -> List[int]:
        """
        Sample causal permutation using STRICT topological ordering by depth layers.
        
        This enforces full topological ordering: all features at depth k come before
        all features at depth k+1. More restrictive than necessary for causal validity.
        
        Returns:
        --------
        perm : List[int]
            Valid causal permutation
        """
        # start with topological order
        perm = self.topological_order.copy()

        # Shuffle within layers (features with same topological depth )
        # This maintains causal validity while adding randomness
        
        depths = np.zeros(self.n_features, dtype = int)
        for node in self.topological_order:
            if self.parents[node]:
                depths[node] = max(depths[p] for p in self.parents[node]) + 1

        # Group by depth and shuffle within groups
        by_depth = {}
        for node in perm:
            depth = depths[node]
            if depth not in by_depth:
                by_depth[depth] = []
            by_depth[depth].append(node)


        result = []
        for depth in sorted(by_depth.keys()):
            level = by_depth[depth]
            self.rng.shuffle(level)
            result.extend(level)

        return result
    
    def _sample_causal_permutation_frye(self) -> List[int]:
        """
        Sample causal permutation using TRUE Asymmetric Shapley (Frye et al.).
        
        Only enforces direct parent constraints: each feature i must appear after
        its direct parents. Unrelated features can appear in any order, allowing
        more permutations than strict topological ordering.
        
        Algorithm: Greedily build permutation by randomly selecting from features
        whose parents are already in the permutation.
        
        Returns:
        --------
        perm : List[int]
            Valid causal permutation (uniform over all valid permutations)
        """
        perm = []
        remaining = set(range(self.n_features))
        
        while remaining:
            # Find all features whose parents are already in the permutation
            candidates = []
            for feature in remaining:
                # Check if all parents of this feature are already in perm
                if self.parents[feature].issubset(set(perm)):
                    candidates.append(feature)
            
            # If no candidates, we have a cycle (shouldn't happen with valid DAG)
            if not candidates:
                warnings.warn("No valid candidates found - possible cycle in graph. Using arbitrary order.")
                candidates = list(remaining)
            
            # Randomly select one candidate (uniform sampling)
            selected = candidates[self.rng.randint(len(candidates))]
            perm.append(selected)
            remaining.remove(selected)
        
        return perm
    
    def _compute_monte_carlo_causal_shapley(self, instance: np.ndarray) -> np.ndarray:

        shapley_values = np.zeros(self.n_features)

        for _ in range(self.n_samples):
            # Sample permutation using selected method
            if self.asymmetric_method == 'frye':
                perm = self._sample_causal_permutation_frye()
            else:  # 'strict'
                perm = self._sample_causal_permutation_strict()

            prev_value = self.baseline_value
            coalition = []

            for feature_idx in perm:

                coalition.append(feature_idx)

                curr_value = self._predict_coalition(instance, coalition)

                marginal = curr_value - prev_value

                shapley_values[feature_idx]+= marginal

                prev_value = curr_value

        shapley_values /= self.n_samples


        return shapley_values
    
    def explain(self, X: pd.DataFrame, method: str = 'monte_carlo') -> np.ndarray:

        print(f"Computing Causal Shapley values using {method} method...")
        # Detailed configuration info removed for cleaner output
        # print(f"Asymmetric sampling: {self.asymmetric_method}")
        # print(f"Respecting causal graph with {np.sum(self.causal_graph != 0)} edges")

        X_values = X.values
        n_instances = len(X_values)


        self.shap_values = np.zeros((n_instances, self.n_features))

        for i, instance in enumerate(X_values):
            if method == 'exact':
                if self.n_features >10:
                    warnings.warn("Exact Causal Shapley calculation with >10 features is very slow. Using Monte Carlo instead.")
                    self.shap_values[i] = self._compute_monte_carlo_causal_shapley(instance)
                else: 
                    self.shap_values[i] = self._compute_exact_causal_shapley(instance)
            elif method == 'monte_carlo':
                self.shap_values[i] = self._compute_monte_carlo_causal_shapley(instance)
            else:
                raise ValueError(f"Unkown method: {method}. Use 'exact' or 'monte_carlo' ")
            
            # Progress indicator removed for cleaner output
            # if (i + 1) % max(1, n_instances//10) == 0:
            #     print(f" Progress: {i + 1}/{n_instances} instances")

        return self.shap_values


class CausalShapley(ShapleyFromScratch):
    """
    Compute Causal Shapley values using post-interventional sampling

    Implements the rigorous alogrithm from Heskes et at. (2020):
    "Causal Shapley Values: Exploiting Causal Knowledge to Explain
    Individual Predictions of Complex Models"

    This correctly handeles confounding by distinguishing:
    - CONFOUNDED components: sample features independently
    - NON-CONFOUNDED components: sample features jointly (preserve mutual interactions)
    """

    def __init__(self, model: BaseEstimator,
                 background_data: pd.DataFrame,
                 discovered_adj: np.ndarray,
                 discovered_conf: List[Tuple[str,str]],
                 feature_names: List[str],
                 n_samples: int = 100,
                 M_inner_samples: int = 50,
                 random_state: Optional[int] = None):
        
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
        """Extract directed edges from causal graph."""
        if causal_graph.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match features"
            )
        directed = np.zeros((self.n_features, self.n_features), dtype=int)

        # Check if binary adjacency matrix
        unique_vals = set(np.unique(causal_graph).tolist())
        if unique_vals.issubset({0,1}):
            directed = causal_graph.astype(int)
        else:
            for i in range(self.n_features):
                for j in range(i+1, self.n_features):
                    a = causal_graph[i, j]
                    b = causal_graph[j, i]

                    if a == -1 and b == 1:
                        directed[i, j] = 1
                    elif a == 1 and b == -1:
                        directed[j, i] = 1

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
        """
        Sample P(X_target | X_conditioning = values) using Gaussian assumption.

        Uses the conditional Gaussian formula for X_A | X_B ~ N

        Parameters:
        -----------
        target_features : List[int]
            Feature indices to sample
        conditioning_features: List[int]
            Features indices being conditioned on
        conditioning_values: np.ndarray
            Values of conditioning features
        
        Returns:
        --------
        samples : np.ndarray
            Sampled values for target features
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
        2. For each compontent t in topological roder:
            - Identify missing geatures (not in S) withing component
            - Get parents vales ( already determined from topoloical ordering)
            - IF component is CONFOUNDED:
                Sample missing features IDEPENDENLTY conditional on parents only 
                (Intervention breaks dependencies between features in componen)
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
            # Initialize sample vector
            sample = np.zeros(self.n_features)

            # Step 1: Fix interventional values X_s = X_s
            for feature in S:
                sample[feature] = x_instance[feature]
            
            # Step 2: Iterate through components in topological order
            for comp_idx, component in enumerate(self.causal_graph_components):
                # Identify fixed vs missing features in this component
                fixed_in_comp = [f for f in component if f in S_set]
                missing_in_comp = [f for f in component if f not in S_set]

                if len(missing_in_comp) == 0:
                    # All features in component are fixed, nothing to sample
                    continue

                # Get parent values (parents are in earlier components, already determined)
                parent_features = self.parents_dict.get(comp_idx, [])
                parent_values = sample[parent_features] if len(parent_features) > 0 else np.array([])

                # CRITICAL DISTINTION: Confounded vs Non-confounded
                if self.confounded_info.get(comp_idx, False):
                    # CONFOUNDED COMPONENT
                    # Sample each missing feature INDEPENDENTLY conditional on parents only
                    for feature in missing_in_comp:
                        if len(parent_features)>0:
                            sampled_value = self._sample_conditional_gaussian(
                                target_features=[feature],
                                conditioning_features=parent_features,
                                conditioning_values=parent_values
                            )[0]
                        else:
                            # No parents: sample from marginal
                            sampled_value = self.rng.normal(self.mean[feature],
                                                            np.sqrt(self.cov[feature,feature]))
                        sample[feature] = sampled_value
                else:
                    # NON-CONFOUNDED COMPONENT
                    # Sample missing features JOINTLY conditional on parents + siblings in S

                    # NOTE : For non-Gaussian or complex distributions, this step should 
                    # be refined using Gibbs sampling to properly capture joint dependencies

                    # Conditional set: parents + fixed siblings in component
                    conditioning_features = list(parent_features) + fixed_in_comp
                    conditioning_values = sample[conditioning_features] if len(conditioning_features) > 0 else np.array([])

                    if len(conditioning_features) > 0:
                        sampled_values = self._sample_conditional_gaussian(
                            target_features=missing_in_comp,
                            conditioning_features=conditioning_features,
                            conditioning_values=conditioning_values
                        )
                    else:
                        # No conditioning: sample from joint marginal
                        if len(missing_in_comp) == 1:
                            sampled_values = np.array([self.rng.normal(self.mean[missing_in_comp[0]],
                                                                      np.sqrt(self.cov[missing_in_comp[0],missing_in_comp[0]]))])
                        else:
                            cov_missing = self.cov[np.ix_(missing_in_comp,missing_in_comp)]
                            sampled_values = self.rng.multivariate_normal(self.mean[missing_in_comp],cov_missing)
                    
                    for feature, value in zip(missing_in_comp,sampled_values):
                        sample[feature] = value
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
    
    def _compute_monte_carlo_causal_shapley(self, instance: np.ndarray) -> np.ndarray:
        """
        Compute Causal Shapley values using Monte Carlo permutation Sampling

        Outer Loop (n_samples iterations):
        - Sample random permutation of features
        - Initialize empty coalition S = 0
        - For each feature J in permutation order:
            * Compute v(S) using inner sampling loop
            * compute v(S u {j} ) using inner sampling loop
            * Marginal contribution = V(S u {j}) - V(S)
            * Add to features j's Shapley value
            * Add j to coalition: S = S u {j}

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
            # Sample random permutation of all features
            # TODO: For Asymmetric Causal Shaply, enforce topological ordering
            perm = self.rng.permutation(self.n_features).tolist()

            # Initialize empty coaliton
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
    """
    Shapley Flow implementatiojn based on Wang et al. (2021)
    Compute edge attributions in a DAG using recursive DFS with random
    permutations of children, following Algorithm 1 from the paper

    Use on-manifold perturbation with conditional expectatios:
    - Features are smpled from conditional distributions P(X-i | non-missing predecessors)
    - When an edge is active, the feature uses its forground value
    - When and edge is not active, the feature is treated as missing and sampled conditionally


    Reference: Wang & Venkatasubramanian (2021) "Shapley Flow: A Graph-based
    Approach to Interpreting Model Predictions
    """
    def __init__(self, graph_structure: Dict[int, List[int]],
                 background_data: np.ndarray,
                 model: Optional[BaseEstimator] = None,
                 source_nodes: Optional[List[int]] = None,
                 sink_node: Optional[int] = None,
                 n_samples: int = 100,
                 random_state: Optional[int] = None,
                 feature_names: Optional[List[str]] = None):
        """
        Initiliaze Shapley Flow Calculator

        Parameters:
        -----------
        graph_structure: Dict[int, List[int]]
            Adjacency list where graph_structure[u] = list of children of u
        backgourd_data: np.ndarray
            Background data for conditional sampling (n_samples x n_features)
        model: BaseEstimator, optional
            Model for predcition (required if sin_node is specified)
        source_nodes: List[int]
            list of source/input node identifiers (auto-detected if None)
        sink_node: int
            Identifeier for the final output node (for model prediction)
        n_samples: int
            Number of Monte Carlo samples (random permutations)
        random_state: int, optional
            Randome seed for reproducibility
        feature_names: List[str], optional
            Feature names for proper alignment with model (needed for wrapper models)
        """

        self.graph = graph_structure
        self.background_data = background_data
        self.model = model
        self.source_nodes = source_nodes
        self.sink_node = sink_node
        self.n_samples = n_samples
        self.feature_names = feature_names

        self.rng = np.random.RandomState(random_state)

        # Build reverse graph (parents for each node)
        self.parents = {node:[] for node in graph_structure.keys()}
        for parent, children in graph_structure.items():
            for child in children:
                if child not in self.parents:
                    self.parents[child] = []
                self.parents[child].append(parent)

        # Auto-detect source nodes if not provided
        if source_nodes is None: 
            self.source_nodes = [node for node in graph_structure.keys()
                                if len(self.parents.get(node,[])) == 0]
        else:
            self.source_nodes = source_nodes

        # Edge attributions (will be computed)
        self.edge_attributions = {}
        for parent, children in graph_structure.items():
            for child in children:
                self.edge_attributions[(parent,child)]= 0.0
    
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

    def _sample_conditional(self, node: int, observed_nodes: Dict[int, float]) -> float:

        if len(observed_nodes) == 0:
            # No conditioning information : sample from marginal
            return self.background_data[self.rng.randint(len(self.background_data)), node]
        
        # Extract conditioning features and values
        cond_indices = list(observed_nodes.keys())
        cond_values = np.array([observed_nodes[i] for i in cond_indices])

        bg_cond = self.background_data[:, cond_indices]
        distances = np.sum((bg_cond-cond_values) ** 2, axis =1)

        # Use K nearest neighbors (k=10 or 10% of data, whichever is smaller)
        k = min(10, max(1, len(self.background_data) // 10 ))
        nearest_indices = np.argpartition(distances,k)[:k]

        candidate_values = self.background_data[nearest_indices,node]
        sampled_values = candidate_values[self.rng.randint(len(candidate_values))]

        return sampled_values

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

        return result
    
    def _dfs(self, node: int, history: List[Tuple[int, int]],
             x_foreground: Dict[int, float],
             x_background: Dict[int, float]) -> None:
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
        """
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

        # Verbose trial info removed for cleaner output
        # print(f"Computing Shapley Flow with { self.n_samples} trials...")

        #Run n_samples Monte Carlo trials
        for trial in range(self.n_samples):
            # Each trial: DFS form each source with random permutations
            for source in self.source_nodes:
                self._dfs(source,[], x_foreground, x_background)
            # # Pick ONE random source per trial
            # source = self.source_nodes[self.rng.randint(len(self.source_nodes))]
            # self._dfs(source,[], x_foreground,x_background)

        # Average acrross trials
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
        # print(f" Total edge attributions: {total_attribution:.6f}")
        # print(f" Expected (f(x) - f(x')) : {expected_total:.6f}")
        # print(f" Difference: {abs(total_attribution - expected_total)}" 
        #       f" using {n_eval_samples} samples to calculate f(x) and f(x')")
        # print(f" Relative error: {relative_error:.4f}")

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
    """
    Wrapper for ShapleyFlow to work with trained ML models

    Converts a trained model + causal DAG into the graph structure 
    required by the core ShapleyFlow alrgorithm, using on-manifold
    perturbatiuon with conditional expectations.
    """

    def __init__(self, model: BaseEstimator,
                background_data: pd.DataFrame,
                causal_graph: np.ndarray,
                y_index: int,
                n_samples: int = 100,
                random_state: Optional[int] = None):
        """
        Initialize wrapper.

        Parameters:
        -----------
        model: BaseEstimator
            Trained predictive model
        background_data : pd.DataFrame
            Reference dataset for conditional sampling
        causal_graph : np.ndarray
            Adjancency matrix where causal_graph[i,j]=1 means i causes j
        y_index : int
            Index of the outcome variable Y
        n_samples : int
            Number of Monte Carlo samples
        random_state : int , optional
            Random seed
        """
        self.model = model
        self.background_data_df = background_data
        self.background_data = background_data.values
        self.features_names = background_data.columns.tolist()
        self.n_features = len(self.features_names)
        self.y_index = y_index
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)

        # Extract directed graph 
        self.directed_graph = self._extract_directed_graph(causal_graph)

        # Build graph structure for ShapleyFlow
        self.graph_structure = {}
        for i in range(self.n_features):
            children = [j for j in range(self.n_features) if self.directed_graph[i,j] !=0]
            self.graph_structure[i] = children

        self.source_nodes = []
        for i in range(self.n_features):
            has_parent = any(self.directed_graph[j, i] != 0 for j in range(self.n_features))
            if not has_parent:
                self.source_nodes.append(i)

        # if no sources found (e.g., cycles), use all non-Y nodes as sources
        if not self.source_nodes:
            self.source_nodes = [i for i in range(self.n_features) if i!= y_index]

        # Verbose initialization removed for cleaner output
        # print(f"ShapleyFlowWrapper initialized:")
        # print(f" Features: {self.n_features}")
        # print(f" Y index: {self.y_index}")
        # print(f" Source nodes: {self.source_nodes}")
        # print(f" Edges: {sum(len(v) for v in self.graph_structure.values())}")
        # print(f" Using on-manifold perturbation with condtional expectatiosn")

    def _extract_directed_graph(self, causal_graph: np.ndarray) -> np.ndarray:
        """Extract directed edges from causal graph."""
        if causal_graph.shape != (self.n_features, self.n_features):
            raise ValueError(
                f"causal_graph shape {causal_graph.shape} does not match features"
            )
        directed = np.zeros((self.n_features, self.n_features), dtype=int)

        # Check if binary adjacency matrix
        unique_vals = set(np.unique(causal_graph).tolist())
        if unique_vals.issubset({0,1}):
            return causal_graph.astype(int)

        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                a = causal_graph[i, j]
                b = causal_graph[j, i]

                if a == -1 and b == 1:
                    directed[i, j] = 1
                elif a == 1 and b == -1:
                    directed[j, i] = 1

        return directed


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

        # Removed for cleaner output
        # print(f"Computing Shapley Flow for {n_instances} instances...")

        for i, instance in enumerate(X_values):
            # Create value functionss for this instance

            flow = ShapleyFlow(
                graph_structure=self.graph_structure,
                background_data=self.background_data,
                model=self.model,
                source_nodes=self.source_nodes,
                sink_node=self.y_index,
                n_samples=self.n_samples,
                random_state=self.rng.randint(0,100000),
                feature_names=self.features_names  # Pass feature names for alignment
            )
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

            # Progress indicator removed for cleaner output
            # if (i +1) % max(1, n_instances// 10) == 0:
            #     print(f" Progress: {i + 1}/{n_instances}")
        
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


