import numpy as np
import math
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from itertools import chain, combinations, permutations
import warnings
warnings.filterwarnings('ignore')

import shap

from sklearn.base import BaseEstimator


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
        self.feature_names = background_data.columns.tolist()
        self.n_features  = len(self.feature_names)
        self.n_samples = n_samples
        self.random_state = random_state
        self.shap_values = None

        self.rng = np.random.RandomState(random_state)

        self.baseline_value = self.model.predict(self.background_data).mean()

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

        # Average predictions 
        predictions = self.model.predict(samples)
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
        print(f"Computin Shapley values from scratch using {method} method...")

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
            
            if (i + 1) % max(1, n_instances //10) == 0:
                print(f" Progress: {i + 1}/{n_instances} instances")

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
    

class CausalShapley(ShapleyFromScratch):

    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 causal_graph: np.ndarray, n_samples: int = 1000,
                 random_sate: Optional [int] = None ):
        
        super().__init__(model, background_data, n_samples,random_sate)

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
            print(f"Feature {self.feature_names[j]} has parents: {[self.feature_names[p] for p in self.parents[j]]}")

        self.topological_order = self._topological_sort()
        print(f"Topological order of features: {self.topological_order}")

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
    
    def _sample_causal_permutation(self) -> List[int]:

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
    
    def _compute_monte_carlo_causal_shapley(self, instance: np.ndarray) -> np.ndarray:

        shapley_values = np.zeros(self.n_features)

        for _ in range(self.n_samples):
            perm = self._sample_causal_permutation()

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

        print(f"Computin Causal Shapley values using {method} method...")
        print(f"Respecting causal graph with {np.sum(self.causal_graph != 0)} edges")

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
            
            if (i + 1) % max(1, n_instances//10) == 0:
                print(f" Progress: {i + 1}/{n_instances} instances")

        return self.shap_values




