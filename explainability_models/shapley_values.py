import logging

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
    """Wrapper around SHAP library for model-specific Shapley value explanations.
    
    This class provides an easy-to-use interface to the official SHAP library,
    automatically selecting the most appropriate explainer (Tree, Linear, or Kernel)
    based on the model type.
    
    BACKGROUND DATA USAGE:
    - For TreeExplainer: Background data is used to estimate missing features during tree traversal
    - For LinearExplainer: Background data defines the baseline (reference point) for explanations
    - For KernelExplainer: Background data is sampled to marginalize over missing features
    
    SHAP VALUES CALCULATION:
    - Uses model-specific optimized algorithms from the SHAP library
    - TreeExplainer: Polynomial-time algorithm for tree-based models (exact)
    - LinearExplainer: Closed-form solution for linear models (exact)
    - KernelExplainer: Model-agnostic weighted linear regression (approximate)
    
    Parameters
    ----------
    model : BaseEstimator
        Trained scikit-learn compatible model to explain
    background_data : pd.DataFrame
        Reference dataset for computing baseline and marginalizing over missing features.
        Typically a sample (~100-1000 instances) from the training data
    method : str, default='auto'
        Explainer method: 'auto' (auto-detect), 'tree', 'linear', or 'kernel'
    
    Attributes
    ----------
    explainer : shap.Explainer
        Initialized SHAP explainer instance
    shap_values : np.ndarray or None
        Computed SHAP values after calling explain()
    feature_names : List[str]
        Feature names from background data
    
    Examples
    --------
    >>> explainer = ShapleyExplainer(rf_model, X_train.sample(100))
    >>> shap_values = explainer.explain(X_test)
    >>> importance = explainer.get_feature_importance()
    """
    
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
        """Initialize the appropriate SHAP explainer based on model type.
        
        Auto-detection logic:
        1. Check for tree-based models (LightGBM, RandomForest, GradientBoosting)
        2. Check for linear models (models with coef_ attribute)
        3. Fall back to model-agnostic KernelExplainer
        
        Returns
        -------
        None
            Sets self.explainer and self.method
        """

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
        """Compute SHAP values for given instances.
        
        For each feature i and instance x, computes the contribution of feature i
        to the model's prediction f(x) relative to the baseline prediction.
        
        The SHAP value satisfies:
        f(x) = baseline + sum(shap_values)
        
        Parameters
        ----------
        X : pd.DataFrame
            Instances to explain (shape: n_samples x n_features)
        
        Returns
        -------
        shap_values : np.ndarray
            SHAP values (shape: n_samples x n_features)
            shap_values[i, j] = contribution of feature j to prediction for instance i
        """
        
        print(f"Computing SHAP values using {self.method} explainer...")

        self.shap_values = self.explainer.shap_values(X)

        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[0]

        return self.shap_values
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Compute global feature importance from SHAP values.
        
        Aggregates SHAP values across all instances using mean absolute value,
        which measures the average impact of each feature on predictions.
        
        Returns
        -------
        importance : pd.DataFrame
            Feature importance scores sorted in descending order
            Columns: ['feature', 'importance']
        
        Raises
        ------
        ValueError
            If explain() has not been called yet
        """

        if self.shap_values is None:
            raise ValueError("Must call explain() first to compute SHAP values")
        
        importance_scores = np.abs(self.shap_values).mean(axis=0)

        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance',ascending=False)

        return importance
    
    def get_shap_values_df(self, X: pd.DataFrame) ->pd.DataFrame:
        """Get SHAP values as a DataFrame.
        
        Converts the numpy array of SHAP values to a pandas DataFrame
        with proper feature names and indices for easier analysis.
        
        Parameters
        ----------
        X : pd.DataFrame
            Instances to explain (if not already computed)
        
        Returns
        -------
        shap_df : pd.DataFrame
            SHAP values with feature names as columns and same index as X
        """

        if self.shap_values is None:
            self.explain(X)

        return pd.DataFrame(self.shap_values,columns= self.feature_names, index=X.index)
    
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
    

class TrueShapley(ShapleyFromScratch):
    """Ground truth Shapley values using the true data generation function.
    
    Instead of explaining a trained model, this class computes Shapley values
    directly from the true data generating process (if known). This provides
    ground truth feature importance for validation and benchmarking purposes.
    
    USE CASE:
    Primarily used for synthetic data experiments where the true function is known:
    - Validate that Shapley implementations are correct
    - Compare model-based explanations against ground truth
    - Understand theoretical properties of Shapley values
    
    DIFFERENCE FROM ShapleyFromScratch:
    - Uses true_generator(X) instead of model.predict(X)
    - Provides "oracle" explanations (no model approximation error)
    - Useful for debugging and validation
    
    Parameters
    ----------
    true_generator : Callable
        True data generation function: f(X) → Y
        Takes array of shape (n_samples, n_features) and returns (n_samples,)
    background_data : pd.DataFrame
        Background dataset for marginalizing over missing features
    n_samples : int, default=1000
        Number of Monte Carlo samples for approximation
    random_state : int or None
        Random seed for reproducibility
    
    Examples
    --------
    >>> def true_func(X):
    ...     return X[:, 0] * 2 + X[:, 1] * 3  # Linear function
    >>> explainer = TrueShapley(true_func, background_data)
    >>> true_shap = explainer.explain(X_test, method='monte_carlo')
    """
    
    def __init__(self, true_generator: Callable, background_data: pd.DataFrame,
                 n_samples: int = 1000, random_state: Optional[int] = None):
        """Initialize TrueShapley explainer.
        
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
        """Predict using true generator for a coalition of features.
        
        Overrides parent method to use true data generator instead of a model.
        
        SAMPLING STRATEGY:
        - Features IN coalition: Use instance's values (foreground)
        - Features NOT in coalition: Sample from background (n_samples times)
        - Compute Y using true generator for all samples
        - Return average Y
        
        Args:
            instance: Single instance to explain (n_features,)
            coalition: List of feature indices in the coalition
        
        Returns:
            Expected Y value for this coalition: E[Y | X_coalition = x_coalition]
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
    """Asymmetric Shapley values that respect causal ordering constraints.
    
    This class implements Asymmetric Shapley values, which modify the standard
    Shapley formula by restricting the set of valid coalitions to those that
    respect causal dependencies in a directed acyclic graph (DAG).
    
    KEY INSIGHT:
    Standard Shapley values treat all features symmetrically, but in causal systems,
    a feature can only contribute if its causal parents are also present. For example,
    if X1 → X2 → Y, then X2 cannot contribute to Y without X1 being observed.
    
    CAUSAL CONSTRAINT:
    A coalition S is valid only if: for every feature i ∈ S, all parents of i are in S
    This ensures we don't "cut" causal paths in the graph.
    
    HOW CAUSAL DAG IS USED:
    1. Extract directed edges from adjacency matrix: dag[i,j]=1 means i → j
    2. Build parent dictionary: parents[j] = {all i where i → j}
    3. During sampling, only consider permutations where each feature appears
       after all its parents (topological ordering)
    
    TWO SAMPLING METHODS:
    - 'frye': True Asymmetric Shapley (Frye et al. 2021)
        * Only enforces DIRECT parent constraints
        * Feature i must appear after its parents, but unrelated features can be in any order
        * Samples uniformly from all valid causal orderings
        * More permutations → better exploration
    
    - 'strict': Strict topological layering
        * Groups features by depth: depth[j] = max(depth[parent]) + 1
        * All features at depth k must appear before all features at depth k+1
        * More restrictive than necessary
        * Fewer permutations → may underestimate interactions
    
    BACKGROUND DATA USAGE:
    Same as vanilla Shapley - used to marginalize over missing features
    
    Parameters
    ----------
    model : BaseEstimator
        Trained model to explain
    background_data : pd.DataFrame
        Reference dataset for marginalizing over missing features
    causal_graph : np.ndarray
        Adjacency matrix (n_features x n_features) where:
        - causal_graph[i,j]=1 means i causes j (binary format)
        - OR causal-learn format: causal_graph[i,j]=-1, causal_graph[j,i]=1 means i → j
    n_samples : int, default=1000
        Number of random causal permutations to sample
    random_state : int or None
        Random seed for reproducibility
    asymmetric_method : str, default='frye'
        Sampling method: 'frye' (recommended) or 'strict'
    
    Attributes
    ----------
    directed_graph : np.ndarray
        Extracted directed adjacency matrix (binary)
    parents : Dict[int, Set[int]]
        Parent features for each feature
    topological_order : List[int]
        Valid topological sort of the DAG
    
    References
    ----------
    Frye, C., et al. (2021). "Asymmetric Shapley values: incorporating causal 
    knowledge into model-agnostic explainability." NeurIPS.
    
    Examples
    --------
    >>> causal_dag = np.array([[0,1,0], [0,0,1], [0,0,0]])  # X0→X1→X2
    >>> explainer = AsymmetricShapley(model, X_train, causal_dag, 
    ...                                asymmetric_method='frye')
    >>> shap_values = explainer.explain(X_test)
    """

    def __init__(self, model: BaseEstimator, background_data: pd.DataFrame,
                 causal_graph: np.ndarray, n_samples: int = 1000,
                 random_state: Optional [int] = None,
                 asymmetric_method: str = 'frye'):
        """Initialize Asymmetric Shapley explainer.
        
        Parameters
        ----------
        model : BaseEstimator
            Trained model to explain
        background_data : pd.DataFrame
            Reference dataset for marginalizing over missing features
        causal_graph : np.ndarray
            Adjacency matrix encoding causal structure
        n_samples : int, default=1000
            Number of random causal permutations
        random_state : int or None
            Random seed
        asymmetric_method : str, default='frye'
            Method for sampling causal permutations:
            - 'frye': True Asymmetric Shapley (Frye et al.) - only enforces direct parent constraints
            - 'strict': Strict topological ordering by depth layers
        """
        
        super().__init__(model, background_data, n_samples,random_state)
        
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
        """Convert causal graph matrix into binary directed adjacency matrix.
        
        Supports two common encodings:
        1. Binary adjacency: causal_graph[i, j] = 1 means i → j (and [j,i]=0)
        2. Causal-learn endpoint encoding: 
           - causal_graph[i,j]=-1, causal_graph[j,i]=1 means i → j (tail at i, arrow at j)
           - Other combinations (undirected, bidirected) are ignored
        
        Parameters
        ----------
        causal_graph : np.ndarray
            Input causal graph matrix
        
        Returns
        -------
        directed : np.ndarray
            Binary adjacency matrix where directed[i,j]=1 means i → j
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
        """Sample causal permutation using TRUE Asymmetric Shapley (Frye et al.).
        
        ALGORITHM (Greedy Valid Extension):
        1. Start with empty permutation: π = []
        2. Build available set: candidates = {features whose parents are all in π}
        3. Randomly select one candidate and append to π
        4. Repeat until all features are in π
        
        KEY PROPERTY:
        - Only enforces direct parent constraints
        - Samples uniformly from all valid causal orderings
        - More flexible than strict topological ordering
        
        EXAMPLE:
        For DAG: X1 → X3, X2 → X3
        Valid permutations include:
        - [X1, X2, X3] ✓
        - [X2, X1, X3] ✓  (X1 and X2 can be in any order)
        - [X1, X3, X2] ✗  (X3 before its parent X2)
        
        Returns
        -------
        perm : List[int]
            Valid causal permutation (uniform distribution over valid orderings)
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
        """Compute Asymmetric Shapley values using Monte Carlo with causal permutations.
        
        ALGORITHM:
        Repeat n_samples times:
            1. Sample valid causal permutation π (using selected method)
            2. Initialize: S = ∅, v_prev = baseline
            3. For each feature f_i in order π:
                a. Add f_i to coalition: S = S ∪ {f_i}
                b. Compute: v_curr = v(S)
                c. Marginal: Δ_i = v_curr - v_prev
                d. Accumulate: φ_i += Δ_i
                e. Update: v_prev = v_curr
        Return: φ / n_samples
        
        DIFFERENCE FROM VANILLA SHAPLEY:
        - Standard Shapley: all n! permutations are valid
        - Asymmetric Shapley: only causal permutations are valid
        - This restricts the set of coalitions considered
        
        Parameters
        ----------
        instance : np.ndarray
            Instance to explain
        
        Returns
        -------
        shapley_values : np.ndarray
            Asymmetric Shapley values respecting causal constraints
        """

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
    """Causal Shapley values using post-interventional sampling (Heskes et al. 2020).
    
    This is the most rigorous causal Shapley method that correctly handles confounding
    by using post-interventional distributions P(X | do(X_S = x_S)) instead of
    conditional distributions P(X | X_S = x_S).
    
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
       - CONFOUNDED component: Features share a hidden confounder
       - NON-CONFOUNDED component: Individual features or causally related features
    
    ═══════════════════════════════════════════════════════════════════════
    POST-INTERVENTIONAL SAMPLING ALGORITHM:
    ═══════════════════════════════════════════════════════════════════════
    
    To sample from P(X | do(X_S = x_S)):
    
    1. Fix intervened features: X_S = x_S (from instance)
    
    2. For each component t in topological order:
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
    
    3. Repeat M times to get samples from the do-distribution
    
    ═══════════════════════════════════════════════════════════════════════
    BACKGROUND DATA USAGE:
    ═══════════════════════════════════════════════════════════════════════
    
    Background data is used to estimate conditional distributions:
    - Fit Gaussian approximation: X ~ N(μ, Σ)
    - Use conditional Gaussian formulas for sampling:
      X_A | X_B = b ~ N(μ_A + Σ_AB Σ_BB^{-1}(b - μ_B), Σ_AA - Σ_AB Σ_BB^{-1} Σ_BA)
    
    Note: Current implementation assumes Gaussian distributions
    For non-Gaussian: could use Gibbs sampling or other methods
    
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
                 M_inner_samples: int = 50,
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
    """Shapley Flow: Graph-based edge attributions using on-manifold perturbations.
    
    Implements the Shapley Flow algorithm from Wang & Venkatasubramanian (2021),
    which extends Shapley values from features to EDGES in a causal DAG.
    
    ═══════════════════════════════════════════════════════════════════════
    KEY IDEA:
    ═══════════════════════════════════════════════════════════════════════
    
    Instead of asking "How important is feature X_i?", Shapley Flow asks:
    "How important is the causal edge X_i → X_j?"
    
    This provides FINE-GRAINED explanations:
    - Which causal paths contribute most to predictions?
    - How does information flow through the causal graph?
    - More interpretable for domain experts who know the causal structure
    
    ═══════════════════════════════════════════════════════════════════════
    ON-MANIFOLD PERTURBATION:
    ═══════════════════════════════════════════════════════════════════════
    
    Unlike standard Shapley (which uses arbitrary feature masking), Shapley Flow
    uses CONDITIONAL SAMPLING to stay on the data manifold:
    
    For each edge (u → v):
    - Edge ACTIVE: v uses its foreground value from the instance
    - Edge NOT ACTIVE: v is treated as "missing" and sampled from P(v | active_parents)
    
    This ensures all sampled instances are realistic (respect data distribution).
    
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
    use_path_sampling : bool, default=True
        If True, use efficient path sampling (recommended for dense graphs)
        If False, use exhaustive DFS (original algorithm)
    paths_per_source : int, default=100
        Number of random paths to sample per source (only if use_path_sampling=True)
    
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
                 use_path_sampling: bool = True,
                 paths_per_source: int = 100):
        """Initialize Shapley Flow calculator."""

        self.graph = graph_structure
        self.background_data = background_data
        self.model = model
        self.source_nodes = source_nodes
        self.sink_node = sink_node
        self.n_samples = n_samples
        self.feature_names = feature_names
        self.use_path_sampling = use_path_sampling
        self.paths_per_source = paths_per_source

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
    
    def _sample_random_path(self, start_node: int, max_depth: int = 100) -> List[Tuple[int, int]]:
        """
        Sample ONE random path from start_node to sink (or leaf/max_depth).
        
        Uses random walk: at each node, randomly select one child.
        Avoids cycles by tracking visited nodes.
        
        Parameters:
        -----------
        start_node: int
            Starting node for the path
        max_depth: int
            Maximum path length to prevent infinite loops
        
        Returns:
        --------
        path_edges: List[Tuple[int, int]]
            List of edges [(u1, v1), (u2, v2), ...] forming a path
        """
        path_edges = []
        current_node = start_node
        depth = 0
        visited = {start_node}
        
        while current_node != self.sink_node and depth < max_depth:
            children = self.graph.get(current_node, [])
            
            # Filter out visited nodes to avoid cycles
            unvisited_children = [c for c in children if c not in visited]
            
            if not unvisited_children:
                break  # Reached a leaf or dead end
            
            # Randomly select ONE child (this is the sampling part)
            child = self.rng.choice(unvisited_children)
            edge = (current_node, child)
            path_edges.append(edge)
            
            visited.add(child)
            current_node = child
            depth += 1
        
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
        
        if self.use_path_sampling:
            logging.info(f"    → ShapleyFlow (Path Sampling): {n_edges} edges, {len(self.source_nodes)} sources")
            logging.info(f"    → Sampling {self.paths_per_source} paths/source × {self.n_samples} trials")
        else:
            logging.info(f"    → ShapleyFlow (Exhaustive DFS): {n_edges} edges, {len(self.source_nodes)} sources, {self.n_samples} trials")
        sys.stdout.flush()

        if self.use_path_sampling:
            # PATH SAMPLING MODE - faster for dense graphs
            for trial in range(self.n_samples):
                self.eval_count = 0
                
                # Sample K random paths from each source
                for source in self.source_nodes:
                    for _ in range(self.paths_per_source):
                        # Sample one random path
                        path_edges = self._sample_random_path(source)
                        
                        if not path_edges:
                            continue  # Empty path, skip
                        
                        # Compute marginal contributions for edges in this path
                        edge_marginals = self._evaluate_path_contribution(path_edges, x_foreground, x_background)
                        
                        # Accumulate to global edge attributions
                        for edge, marginal in edge_marginals.items():
                            self.edge_attributions[edge] += marginal
                
                # Progress indicator
                if (trial + 1) % 10 == 0 or trial == 0:
                    logging.info(f"    → Trial {trial + 1}/{self.n_samples} completed (~{self.eval_count} evaluations)")
                    sys.stdout.flush()
            
            # Average across trials and sources (not individual paths)
            for edge in self.edge_attributions:
                self.edge_attributions[edge] /= (self.n_samples * len(self.source_nodes))
                
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
    """Convenience wrapper for applying Shapley Flow to trained ML models.
    
    This class bridges trained scikit-learn models with the Shapley Flow algorithm.
    It handles the conversion from causal DAG to graph structure and provides
    a familiar interface similar to other Shapley explainers.
    
    ═══════════════════════════════════════════════════════════════════════
    WHAT THIS WRAPPER DOES:
    ═══════════════════════════════════════════════════════════════════════
    
    1. Converts causal adjacency matrix → graph structure (adjacency list)
    2. Identifies source nodes (features with no parents)
    3. Sets up sink node (outcome variable Y)
    4. Creates ShapleyFlow instance with proper configuration
    5. Computes edge attributions and aggregates to node-level importance
    6. Returns feature importance scores compatible with other explainers
    
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
    use_path_sampling : bool, default=True
        Use fast path sampling (recommended for dense graphs)
    paths_per_source : int, default=100
        Paths to sample per source (if use_path_sampling=True)
    
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
                use_path_sampling: bool = True,
                paths_per_source: int = 100):
        """Initialize Shapley Flow wrapper for ML models."""
        self.model = model
        self.background_data_df = background_data
        self.background_data = background_data.values
        self.features_names = background_data.columns.tolist()
        self.n_features = len(self.features_names)
        self.y_index = y_index
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)
        self.use_path_sampling = use_path_sampling
        self.paths_per_source = paths_per_source

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

        # Log edge count for diagnostics
        import sys
        import logging
        n_edges = sum(len(v) for v in self.graph_structure.values())
        logging.info(f"    → ShapleyFlowWrapper: {self.n_features} features, {n_edges} edges, {len(self.source_nodes)} sources")
        sys.stdout.flush()

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

        # Progress logging
        import sys
        import logging
        logging.info(f"    → Computing Shapley Flow for {n_instances} instances...")
        sys.stdout.flush()

        for i, instance in enumerate(X_values):
            # Progress indicator
            if n_instances > 1:
                logging.info(f"    → Instance {i + 1}/{n_instances}")
                sys.stdout.flush()

            flow = ShapleyFlow(
                graph_structure=self.graph_structure,
                background_data=self.background_data,
                model=self.model,
                source_nodes=self.source_nodes,
                sink_node=self.y_index,
                n_samples=self.n_samples,
                random_state=self.rng.randint(0,100000),
                feature_names=self.features_names,  # Pass feature names for alignment
                use_path_sampling=self.use_path_sampling,  # Use path sampling mode
                paths_per_source=self.paths_per_source  # Number of paths to sample
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


