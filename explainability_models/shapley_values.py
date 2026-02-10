import numpy as np
import pandas as pd
from typing import Callabe, Dict, List, Optional, Tuple, Any
from itertools import combinations, permutations
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
                 n_samples: int = 1000):
        
        self.model = model
        self.background_data = background_data.values
        self.feature_names = background_data.columns.tolist()
        self.n_features  = len(self.feature_names)
        self.n_samples = n_samples
        self.shap_values = None

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
            for coalition_size in (self.n_features):
                for coalition in combinations(other_features, coalition_size):
                    coalition = list(coalition)

                    coalition_with_i = coalition + [i]
                    v_with = self._predict_coalition(instance, coalition_with_i)

                    v_without = self._predict_coalition(instance, coalition)

                    marginal = v_with - v_without

                    weight = 1.0/ (self.n_features * np.math.comb(self.n_features -1, coalition_size))

                    marginal_contributions.append(weight*marginal)
            
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
            perm = np.random.permutation(self.n_features)

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

