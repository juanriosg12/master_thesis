"""
Predictive Modeling Module

This module implements regression models for predictiv Y from X features:
1. LightGBM Regressor with hyperparameter optimization
2. Neural Netwok (MLP) with hyperparameter optimization

Includes Feature selection and comprehensive evaluatuon metrics
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# ML libraries 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, mean_absolute_percentage_error)
from sklearn.neural_network import MLPRegressor

# LightGBM
import lightgbm as lgb

# Neuoral network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameter optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

class PredictiveModel: 
    """
    Base class for predictive models.
    """

    def __init__(self, random_state: int =42):
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.metrics = {}

    def select_features(self, X: pd.DataFrame, y : pd.Series,
                        n_features: Optional[int] = None,
                        method: str = 'mutual_info') -> pd.DataFrame:
        if method =='all' or X.shape[1] <=5:
            self.selected_features = X.columns.tolist()
            return X
        
        if n_features is None:
            n_features = min(10, X.shape[1])

        n_features = min(n_features, X.shape[1])

        if method == 'mutual_info':
            selector = SelectKBest(mutual_info_regression, k=n_features)
        else: 
            selector = SelectKBest(f_regression, k=n_features)

        X_selected = selector.fit_transform(X,y)

        # Get selected feature names

        feature_mask = selector.get_support()
        self.selected_features = X.columns[feature_mask].tolist()
        self.feature_selector = selector

        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def evaluate(self,y_true: np.ndarray, y_pred: np.ndarray,
                 prefix: str = '') -> Dict[str, float]:
        metrics = {
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true,y_pred)),
            f'{prefix}mae': mean_absolute_error(y_true,y_pred),
            f'{prefix}r2': r2_score(y_true,y_pred),
            f'{prefix}mse': mean_squared_error(y_true,y_pred),
        }

        return metrics
    
class LGBMRegressor(PredictiveModel):
    
    def __init__(self, random_state: int= 42, n_trials: int = 50):
        super().__init__(random_state=random_state)
        self.n_trials = n_trials
        self.best_params = None

    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        

        def objective(trial):
            params ={
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_estimators': trial.suggest_int('n_estimators',50, 500),
                'learning_rate': trial.suggest_float('learning_rate',0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves',20, 150),
                'max_depth': trial.suggest_int('max_depth',3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples',5,100),
                'subsample': trial.suggest_float('subsample',0.6,1.0),
                'colsample_bytree':trial.suggest_float('colsample_bytree',0.6,1.0),
                'reg_alpha': trial.suggest_float('reg_alpha',1e-8,10.0, log=True),
                'reg_lambda':  trial.suggest_float('reg_lambda',1e-8,10.0, log=True),
            }

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train,y_train,
                      eval_set=[(X_val,y_val)],
                      callbacks=[lgb.early_stopping(stopping_rounds=20,verbose=False)])
            
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val,y_pred))

            return rmse
    
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed = self.random_state))
        study.optimize(objective,n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = study.best_params
        self.best_params['objective'] = 'regression'
        self.best_params['metric'] = 'rmse'
        self.best_params['verbosity'] = -1
        self.best_params['random_state'] = self.random_state

        return self.best_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            n_features: Optional[int] = None,
            feature_selection: bool = True,
            optimize: bool = True) -> 'LGBMRegressor':
        if feature_selection:
            X = self.select_features(X,y, n_features=n_features)
        else:
            self.selected_features = X.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        if optimize:
            print(f"Optimizing LightGBM hyperparameters ({self.n_trials} trials)...")
            self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            params = self.best_params
        else:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': self.random_state,
                'n_estimators': 200,
            }

        # Train final model
        print("Training LightGBM model...")
        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(X_train,y_train,
                       eval_set=[(X_val,y_val)],
                       callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        
        # Evaluate
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        train_metrics = self.evaluate(y_train,y_train_pred, prefix='train_')
        val_metrics = self.evaluate(y_val,y_val_pred, prefix='val_')

        self.metrics = {**train_metrics,**val_metrics}

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.selected_features is not None:
            X = X[self.selected_features]
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return None
        
        importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance',ascending=False)

        return importance
    
class NeuralNetRegressor(PredictiveModel):

    def __init__(self, random_state: int = 42, n_trials: int =20):
        super().__init__(random_state=random_state)
        self.n_trials = n_trials
        self.best_params = None
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_val: np.ndarray, y_val: np.ndarray) -> Dict:
        def objective(trial):
            n_layers = trial.suggest_int('n_layers',1,3)
            hidden_layer_sizes = tuple([
                trial.suggest_int(f'hidden_dim_{i}',16,128)
                for i in range(n_layers)
            ])

            # Training parameters 
            alpha = trial.suggest_float('alpha,', 1e-5, 1e-2, log=True) #L2 regularization
            learning_rate_init = trial.suggest_float('learning_rate_init',1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 'auto'])
            activation = trial.suggest_categorical('activation', ['relu','tanh'])

            model = MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver='adam',
                alpha=alpha,
                batch_size=batch_size,
                learning_rate_init=learning_rate_init,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=self.random_state,
                verbose=False
            )

            #Train
            model.fit(X_train, y_train)

            # Evaluate on validation set
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val,y_pred)

            return mse
        
        study = optuna.create_study(direction='minimize',
                                    sampler= optuna.samplers.TPESampler(seed=self.random_state))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        # Extract best parameters
        best_params = study.best_params
        n_layers = best_params['n_layers']
        hidden_layer_sizes = tuple([
            best_params[f'hidden_dim_{i}'] for i in range(n_layers)
        ])

        self.best_params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': best_params['activation'],
            'alpha': best_params['alpha'],
            'batch_size': best_params['batch_size'],
            'learning_rate_init': best_params['learning_rate_init'],

        }

        return self.best_params
    
    def fit(self, X: pd.DataFrame, y: pd.Series,
            n_features: Optional[int] = None,
            feature_selection: bool = True,
            optimize: bool = True) -> 'NeuralNetRegressor':
        
        # Feature selection
        if feature_selection:
            X = self.select_features(X,y,n_features=n_features)
        else:
            self.select_features = X.columns.to_list()

        # split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Hyperparameter optimization
        if optimize:
            print(f"Optimizing Neural Network hyperparameters ({self.n_trials} trials)...")
            self.optimize_hyperparameters(X_train_scaled, y_train.values,
                                          X_val_scaled,y_val.values)
            params = self.best_params
        else:
            params = {
                'hidden_layer_sizes': (64,32),
                'activation': 'relu',
                'alpha': 0.0001,
                'batch_size': 'auto',
                'learning_rate_init': 0.001,
                }
            
        # Train final model
        print("Training Neural Network model...")
        self.model = MLPRegressor(
            hidden_layer_sizes= params['hidden_layer_sizes'],
            activation= params['activation'],
            solver='adam',
            alpha=params["alpha"],
            batch_size=params["batch_size"],
            learning_rate_init=params["learning_rate_init"],
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state= self.random_state,
            verbose=False
        )
        
        self.model.fit(X_train_scaled, y_train.values)

        #Evaluate

        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)

        train_metrics = self.evaluate(y_train.values, y_train_pred.values, prefix='train_')
        val_metrics = self.evaluate(y_val.values, y_val_pred.values, prefix='val_')

        self.metrics = {**train_metrics, **val_metrics}

        return self
    
    def predict( self, X: pd.DataFrame) -> np.ndarray:
        if self.selected_features is not None:
            X = X[self.selected_features]

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        return predictions


def train_predictive_models(data: pd.DataFrame,
                     y_column: str = 'Y',
                     features_selection: bool =True,
                     models: List[str] = ['lgbm','nn'],
                     n_features: Optional[int] = None,
                     optimize: bool =True,
                     random_state: int = 42,
                     n_trials: int =50) -> Dict:
    
    X = data.drop(columns=[y_column])
    y = data[y_column]

    print(f"Training models to predict {y_column}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print("=" * 70)

    results = {
        'models': {},
        'metrics': {},
        'selected_features': {}
    }

    if 'lgbm' in models:
        print("\b[1/2] Training LightGBM Regressor")
        print("-" * 70)
        lgbm_model = LGBMRegressor(random_state=random_state, n_trials=n_trials)
        lgbm_model.fit(X,y, n_features=n_features,
                    feature_selection=features_selection,
                    optimize=optimize)
        
        results["models"]["lgbm"] = lgbm_model
        results["metrics"]["lgbm"] = lgbm_model.metrics
        results["selected_features"]["lgbm"]  = lgbm_model.selected_features

    if 'nn' in models:
        print("\b[1/2] Training Neural Network Regressor")
        print("-" * 70)
        nn_model = NeuralNetRegressor(random_state=random_state, n_trials=n_trials)
        nn_model.fit(X,y, n_features=n_features,
                     feature_selection=features_selection,
                     optimize=optimize)
        
        results["models"]["nn"] = nn_model
        results["metrics"]["nn"] = nn_model.metrics
        results["selected_features"]["nn"] = nn_model.selected_features

    
    comparison_df = pd.DataFrame({
        model_name: metrics
        for model_name, metrics in results['metrics'].items()
    }).T

    print(comparison_df[['val_rmse','val_mae','val_r2','val_mape']].to_string())

    return results