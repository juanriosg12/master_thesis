"""
Comprehensive Experimental Pipeline for Causal Feature Importance Methods

This script runs a complete experimental pipeline comparing different Shapley value
methods on synthetic causal data.

Steps:
1. Generate synthetic datasets (6 systems: linear/nonlinear/mixed × confounders/no-confounders)
2. Create standardized train/test splits
3. Run causal discovery (PC and LiNGAM)
4. Train predictive models (LGBM and NN)
5. Calculate Shapley values using all methods and compare across causal discovery algorithms

Author: Juan Rios
"""

# ============================================================================
# Thread Control - MUST BE FIRST
# ============================================================================
# Limit threads to avoid contention (set BEFORE importing numpy/sklearn/etc.)
import os
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'
os.environ['VECLIB_MAXIMUM_THREADS'] = '8'

import sys
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Add parent directory to path
sys.path.append('/Users/juanrios/Documents/master_thesis')

from syntethic_data.syntethic_causal_data import SyntheticCausalSystem
from causal_discovery.causal_discovery import LiNGAMWithFCI, PCWithFCI, compare_with_ground_truth
from predictive_models.predictive_models import (
    LGBMRegressor, NeuralNetRegressor, train_test_split
)
from explainability_models import (
    ShapleyExplainer, ShapleyFromScratch, AsymmetricShapley,
    ShapleyFlowWrapper, CausalShapley
)
from utils.utils import visualize_comparison, visualize_causal_graph, plot_shapley_feature_comparison


# ============================================================================
# Configuration
# ============================================================================

# Experimental parameters
N_FEATURES = 50
N_SAMPLES = 1000
Y_PARENTS_RATIO = 0.5
NOISE_STD = 0.5
EDGE_PROBABILITY = 0.2
MIN_CONNECTED_EDGES = 2
RANDOM_STATE = 42

# Model parameters
TEST_SIZE = 0.2
BACKGROUND_RATIO = 0.3
TEST_INSTANCES_RATIO = 0.5
N_SHAPLEY_SAMPLES = 50
M_INNER_SAMPLES_CAUSAL = 10  # Reduced for CausalShapley performance (was 50 default)

# Discovery parameters
LINGAM_ALPHA = 0.01
PC_ALPHA = 0.01
INDEP_TEST = "fisherz"

# Directories
BASE_DIR = Path('/Users/juanrios/Documents/master_thesis')
DATA_DIR = BASE_DIR / 'data'
SYNTHETIC_DIR = DATA_DIR / 'synthetic'
PROCESSED_DIR = DATA_DIR / 'processed'
CAUSAL_DIR = DATA_DIR / 'causal'
EXPLAINABILITY_DIR = DATA_DIR / 'explainability'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'
TARGET_DATASET = "mixed_conf_f50_s1000_p50"

# Create directories
for directory in [SYNTHETIC_DIR, PROCESSED_DIR, CAUSAL_DIR, EXPLAINABILITY_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Setup simple logging configuration."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = LOGS_DIR / f'pipeline_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing loggers
    )
    
    # Force unbuffered output
    for handler in logging.getLogger().handlers:
        handler.flush = lambda: sys.stdout.flush()
    
    logging.info(f"Log file: {log_file}")
    return logging.getLogger(__name__)


# ============================================================================
# Step 1: Generate Synthetic Datasets
# ============================================================================

def generate_all_datasets():
    """
    Generate all 6 synthetic datasets and save them with metadata.
    
    Returns:
    --------
    dataset_configs : List[Dict]
        Configuration for each generated dataset
    """
    logging.info("=" * 80)
    logging.info("STEP 1: GENERATING SYNTHETIC DATASETS")
    logging.info("=" * 80)
    
    systems = [
        ('linear', 'no_conf', False, 'generate_linear_system'),
        ('linear', 'conf', True, 'generate_linear_system'),
        ('nonlinear', 'no_conf', False, 'generate_nonlinear_system'),
        ('nonlinear', 'conf', True, 'generate_nonlinear_system'),
        ('mixed', 'no_conf', False, 'generate_mixed_system'),
        ('mixed', 'conf', True, 'generate_mixed_system'),
    ]
    
    dataset_configs = []
    
    for idx, (system_type, conf_label, with_confounders, generator_method) in enumerate(systems, 1):
        logging.info(f"\n[Dataset {idx}/6] Generating {system_type} system {'with' if with_confounders else 'without'} confounders")
        
        # Create system
        gen_system = SyntheticCausalSystem(
            n_features=N_FEATURES,
            random_state=RANDOM_STATE,
            min_num_connected_edges=MIN_CONNECTED_EDGES,
            edge_probability=EDGE_PROBABILITY
        )
        
        # Generate data
        generator = getattr(gen_system, generator_method)
        if system_type == 'mixed':
            result = generator(
                n_samples=N_SAMPLES,
                with_confounders=with_confounders,
                noise_std=NOISE_STD,
                linear_ratio=0.5,
                y_parents_ratio=Y_PARENTS_RATIO
            )
            data, adj, conf, edge_types = result
        else:
            result = generator(
                n_samples=N_SAMPLES,
                with_confounders=with_confounders,
                noise_std=NOISE_STD,
                y_parents_ratio=Y_PARENTS_RATIO
            )
            data, adj, conf = result
        
        # Create filename
        filename = f"{system_type}_{conf_label}_f{N_FEATURES}_s{N_SAMPLES}_p{int(Y_PARENTS_RATIO*100)}"
        
        # Save data
        data_path = SYNTHETIC_DIR / f"{filename}.parquet"
        data.to_parquet(data_path)
        
        # Save adjacency matrix
        adj_path = SYNTHETIC_DIR / f"{filename}_adjacency.npy"
        np.save(adj_path, adj)
        
        # Save confounders
        conf_path = SYNTHETIC_DIR / f"{filename}_confounders.json"
        with open(conf_path, 'w') as f:
            json.dump(conf, f, indent=2)
        
        # Save Y generation parameters (for reconstructing the true generator later)
        generator_params_path = SYNTHETIC_DIR / f"{filename}_y_params.json"
        with open(generator_params_path, 'w') as f:
            # Convert numpy types to Python native types for JSON serialization
            y_params = gen_system.y_generation_params.copy()
            if 'weights' in y_params:
                y_params['weights'] = y_params['weights'].tolist()
            json.dump(y_params, f, indent=2)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'system_type': system_type,
            'with_confounders': with_confounders,
            'n_features': N_FEATURES,
            'n_samples': N_SAMPLES,
            'y_parents_ratio': Y_PARENTS_RATIO,
            'y_parent_indices': gen_system.y_parent_indices if isinstance(gen_system.y_parent_indices, list) else gen_system.y_parent_indices.tolist(),
            'noise_std': NOISE_STD,
            'edge_probability': EDGE_PROBABILITY,
            'min_connected_edges': MIN_CONNECTED_EDGES,
            'random_state': RANDOM_STATE,
            'n_edges': int(np.sum(adj != 0)),
            'n_confounders': len(conf) if conf else 0
        }
        
        metadata_path = SYNTHETIC_DIR / f"{filename}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        dataset_configs.append(metadata)
        
        logging.info(f"  ✓ Saved: {filename} | Features: {N_FEATURES}, Samples: {N_SAMPLES}, Y parents: {len(gen_system.y_parent_indices)}, Edges: {int(np.sum(adj != 0))}, Confounders: {len(conf) if conf else 0}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 1 Complete: {len(dataset_configs)} datasets generated")
    logging.info(f"{'='*80}\n")
    
    return dataset_configs


# ============================================================================
# Step 2: Create Train/Test Splits
# ============================================================================

def create_train_test_splits(dataset_configs: List[Dict]):
    """
    Create standardized train/test splits for all datasets.
    
    Parameters:
    -----------
    dataset_configs : List[Dict]
        Dataset configurations from Step 1
    """
    logging.info("=" * 80)
    logging.info("STEP 2: CREATING TRAIN/TEST SPLITS")
    logging.info("=" * 80)
    
    for idx, config in enumerate(dataset_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {idx}/6] Processing {filename}...")
        
        # Load data
        data_path = SYNTHETIC_DIR / f"{filename}.parquet"
        data = pd.read_parquet(data_path)
        
        # Split features and target
        X = data.drop(columns=['Y'])
        y = data['Y']
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        # Save splits
        train_data = X_train.copy()
        train_data['Y'] = y_train
        test_data = X_test.copy()
        test_data['Y'] = y_test
        
        train_path = PROCESSED_DIR / f"{filename}_train.parquet"
        test_path = PROCESSED_DIR / f"{filename}_test.parquet"
        
        train_data.to_parquet(train_path)
        test_data.to_parquet(test_path)
        
        logging.info(f"  ✓ Train: {train_data.shape}, Test: {test_data.shape}")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 2 Complete: Train/test splits created for {len(dataset_configs)} datasets")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 3: Run Causal Discovery
# ============================================================================

def run_causal_discovery(dataset_configs: List[Dict]):
    """
    Run PC and LiNGAM causal discovery on all datasets.
    
    Parameters:
    -----------
    dataset_configs : List[Dict]
        Dataset configurations from Step 1
    """
    logging.info("=" * 80)
    logging.info("STEP 3: RUNNING CAUSAL DISCOVERY")
    logging.info("=" * 80)
    
    for idx, config in enumerate(dataset_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {idx}/6] Processing {filename}")
        
        # Load data and ground truth
        data_path = PROCESSED_DIR / f"{filename}_train.parquet"
        data = pd.read_parquet(data_path)

        subsample_size = min(500, len(data))  # Limit to 500 samples for discovery to speed up
        subsample_data = data.sample(n=subsample_size, random_state=RANDOM_STATE)
        
        adj_path = SYNTHETIC_DIR / f"{filename}_adjacency.npy"
        true_adj = np.load(adj_path)
        
        conf_path = SYNTHETIC_DIR / f"{filename}_confounders.json"
        with open(conf_path, 'r') as f:
            true_conf = json.load(f)
        
        # Initialize discovery methods
        lingam_fci = LiNGAMWithFCI(alpha=LINGAM_ALPHA, indep_test=INDEP_TEST)
        pc_fci = PCWithFCI(alpha=PC_ALPHA, indep_test=INDEP_TEST)
        
        # Run discovery
        logging.info("  Running LiNGAM...")
        results_lingam = lingam_fci.get_causal_relationships(subsample_data)
        
        logging.info("  Running PC...")
        results_pc = pc_fci.get_causal_relationships(subsample_data)
        
        # Also run discovery on X_train for use in Step 5
        
        logging.info("  Extracting train adjacency matrices for Step 5...")
        lingam_train_adj = results_lingam['adjacency_matrix'][:-1, :-1]  # Exclude Y
        pc_train_adj = results_pc['adjacency_matrix'][:-1, :-1]  # Exclude Y
        
        # Save discovery results
        for method_name, results in [('lingam', results_lingam), ('pc', results_pc)]:
            # Save results as JSON
            results_json = {
                'adjacency_matrix': results['adjacency_matrix'].tolist(),
                'feature_names': results['feature_names'],
                'confounders': results['confounders']
            }
            
            results_path = CAUSAL_DIR / f"{filename}_{method_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            # Save train adjacency matrix for Step 5
            if method_name == 'lingam':
                train_adj_path = CAUSAL_DIR / f"{filename}_{method_name}_train_adjacency.npy"
                np.save(train_adj_path, lingam_train_adj)
            else:  # pc
                train_adj_path = CAUSAL_DIR / f"{filename}_{method_name}_train_adjacency.npy"
                np.save(train_adj_path, pc_train_adj)
            
            # Compare with ground truth
            comparison = compare_with_ground_truth(
                discovered_adj=results['adjacency_matrix'],
                true_adj=true_adj,
                discovered_confounders=results['confounders'],
                true_confounders=true_conf
            )
            
            # Save comparison metrics
            comparison_path = CAUSAL_DIR / f"{filename}_{method_name}_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logging.info(f"  ✓ {method_name.upper()}: F1={comparison.get('f1_score', 0):.3f}, Precision={comparison.get('precision', 0):.3f}, Recall={comparison.get('recall', 0):.3f}")
            
            # Create visualization
            fig = visualize_comparison(
                true_adj=true_adj,
                discovered_adj=results['adjacency_matrix'],
                features_names=data.columns.tolist(),
                y_parent_indices=config['y_parent_indices'],
                true_confounders=true_conf,
                discovered_confounders=results['confounders'],
                title_preix=f"{method_name.upper()} Algorithm"
            )
            
            # Save visualization
            fig_path = CAUSAL_DIR / f"{filename}_{method_name}_visualization.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 3 Complete: Causal discovery completed for {len(dataset_configs)} datasets")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 4: Train Predictive Models
# ============================================================================

def train_all_models(dataset_configs: List[Dict]):
    """
    Train LGBM model on target dataset only.
    
    Parameters:
    -----------
    dataset_configs : List[Dict]
        Dataset configurations from Step 1
    """
    logging.info("=" * 80)
    logging.info(f"STEP 4: TRAINING PREDICTIVE MODELS (LGBM only, {TARGET_DATASET} dataset)")
    logging.info("=" * 80)
    
    # Filter to only target dataset
    target_dataset = TARGET_DATASET
    filtered_configs = [c for c in dataset_configs if c['filename'] == target_dataset]
    
    if not filtered_configs:
        logging.warning(f"Target dataset {target_dataset} not found!")
        return
    
    for idx, config in enumerate(filtered_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {idx}/1] Training LGBM model for {filename}")
        
        # Load train/test data
        train_path = PROCESSED_DIR / f"{filename}_train.parquet"
        test_path = PROCESSED_DIR / f"{filename}_test.parquet"
        
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        
        X_train = train_data.drop(columns=['Y'])
        y_train = train_data['Y']
        X_test = test_data.drop(columns=['Y'])
        y_test = test_data['Y']
        
        # Train LGBM
        logging.info("  Training LGBM...")
        lgbm_model = LGBMRegressor()
        lgbm_model.fit(X_train, y_train)
        
        # Evaluate LGBM
        lgbm_pred = lgbm_model.predict(X_test)
        lgbm_score = r2_score(y_test, lgbm_pred)
        lgbm_mse = np.mean((y_test - lgbm_pred) ** 2)
        lgbm_mae = np.mean(np.abs(y_test - lgbm_pred))
        lgbm_rmse = np.sqrt(lgbm_mse)
        
        # Save LGBM model
        lgbm_path = MODELS_DIR / f"{filename}_lgbm"
        lgbm_model.save(str(lgbm_path))
        logging.info(f"  ✓ LGBM: R²={lgbm_score:.4f}, MSE={lgbm_mse:.4f}, MAE={lgbm_mae:.4f}, RMSE={lgbm_rmse:.4f}")
        
        # NN training skipped for focused experiment
        # # Train NN
        # logging.info("  Training NN...")
        # nn_model = NeuralNetRegressor()
        # nn_model.fit(X_train, y_train)
        # 
        # # Evaluate NN
        # nn_pred = nn_model.predict(X_test)
        # nn_score = r2_score(y_test, nn_pred)
        # nn_mse = np.mean((y_test - nn_pred) ** 2)
        # nn_mae = np.mean(np.abs(y_test - nn_pred))
        # nn_rmse = np.sqrt(nn_mse)
        # 
        # # Save NN model
        # nn_path = MODELS_DIR / f"{filename}_nn"
        # nn_model.save(str(nn_path))
        # logging.info(f"  ✓ NN: R²={nn_score:.4f}, MSE={nn_mse:.4f}, MAE={nn_mae:.4f}, RMSE={nn_rmse:.4f}")
        
        # Save evaluation metrics
        metrics = {
            'lgbm': {
                'r2': float(lgbm_score),
                'mse': float(lgbm_mse),
                'mae': float(lgbm_mae),
                'rmse': float(lgbm_rmse)
            }
            # 'nn': {
            #     'r2': float(nn_score),
            #     'mse': float(nn_mse),
            #     'mae': float(nn_mae),
            #     'rmse': float(nn_rmse)
            # }
        }
        
        metrics_path = MODELS_DIR / f"{filename}_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 4 Complete: LGBM model trained for {target_dataset}")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 5: Calculate Shapley Values
# ============================================================================

def calculate_all_shapley_values(dataset_configs: List[Dict]):
    """
    Calculate and save Shapley values using explainability methods.
    
    For mixed_no_conf_f50_s1000_p50 dataset with LGBM model:
      - ShapleyFromScratch (no causal graph)
      - For each discovery method (PC, LiNGAM):
        - AsymmetricShapley
        - CausalShapley
        - ShapleyFlowWrapper
    
    Saves:
    ------
    - shapley_values.npy: SHAP values for each instance
    - feature_importance.csv: Average feature importance
    
    Parameters:
    -----------
    dataset_configs : List[Dict]
        Dataset configurations from Step 1
    """
    logging.info("=" * 80)
    logging.info("STEP 5: CALCULATING SHAPLEY VALUES (mixed_no_conf + LGBM only)")
    logging.info("=" * 80)
    
    # Filter to only mixed_no_conf dataset
    target_dataset = TARGET_DATASET
    filtered_configs = [c for c in dataset_configs if c['filename'] == target_dataset]
    
    if not filtered_configs:
        logging.warning(f"Target dataset {target_dataset} not found!")
        return
    
    for ds_idx, config in enumerate(filtered_configs, 1):
        filename = config['filename']
        start_time = time.time()
        logging.info(f"\n[Dataset {ds_idx}/1] Processing {filename}")
        
        # Load data
        train_path = PROCESSED_DIR / f"{filename}_train.parquet"
        test_path = PROCESSED_DIR / f"{filename}_test.parquet"
        
        train_data = pd.read_parquet(train_path)
        test_data = pd.read_parquet(test_path)
        
        X_train = train_data.drop(columns=['Y'])
        y_train = train_data['Y']
        X_test = test_data.drop(columns=['Y'])
        y_test = test_data['Y']
        
        # Prepare background data and test instances (standardized)
        background_data = X_train.sample(n=int(len(X_train) * BACKGROUND_RATIO), random_state=RANDOM_STATE)
        test_instances = X_test.sample(n=int(len(X_test) * TEST_INSTANCES_RATIO), random_state=RANDOM_STATE)
        
        # Load Y parent indices (needed for parent identification metrics)
        metadata_path = SYNTHETIC_DIR / f"{filename}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        y_parent_indices = metadata['y_parent_indices']
        
        # Process LGBM model only
        total_combinations = 1 + 2 * 3  # 1 scratch + 2 discovery methods * 3 causal methods (Asym, Causal, Flow)
        progress_counter = 1
        
        for model_name in ['lgbm']:  # Only LGBM
            
            # Load model
            model_path = MODELS_DIR / f"{filename}_{model_name}"
            if model_name == 'lgbm':
                model = LGBMRegressor.load(str(model_path))
            else:
                model = NeuralNetRegressor.load(str(model_path))
            
            # Calculate ShapleyFromScratch
            logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} - ShapleyFromScratch")
            progress_counter += 1
            scratch_explainer = ShapleyFromScratch(
                model,  # Pass wrapper object, not model.model
                background_data,
                n_samples=N_SHAPLEY_SAMPLES,
                random_state=RANDOM_STATE
            )
            scratch_values = scratch_explainer.explain(test_instances, method='monte_carlo')
            scratch_importance = scratch_explainer.get_feature_importance()
            
            # Save Scratch results
            scratch_dir = EXPLAINABILITY_DIR / filename / model_name / 'scratch'
            scratch_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(scratch_dir / 'shapley_values.npy', scratch_values)
            scratch_importance.to_csv(scratch_dir / 'feature_importance.csv', index=False)
            
            # Process each discovery method
            for discovery_method in ['pc', 'lingam']:
                
                # Load discovery results
                results_path = CAUSAL_DIR / f"{filename}_{discovery_method}_results.json"
                with open(results_path, 'r') as f:
                    discovery_results = json.load(f)
                
                causal_graph = np.array(discovery_results['adjacency_matrix'])
                confounders = discovery_results['confounders']
                feature_names = discovery_results['feature_names']
                
                # Load discovered adjacency matrix from X_train (computed in Step 3)
                train_adj_path = CAUSAL_DIR / f"{filename}_{discovery_method}_train_adjacency.npy"
                discovered_adj = np.load(train_adj_path)
                
                # Calculate AsymmetricShapley
                logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + {discovery_method.upper()} - AsymmetricShapley")
                progress_counter += 1
                asymmetric_explainer = AsymmetricShapley(
                    model,  # Pass wrapper object, not model.model
                    background_data,
                    causal_graph=discovered_adj,
                    n_samples=N_SHAPLEY_SAMPLES,
                    random_sate=RANDOM_STATE,
                    asymmetric_method='strict'
                )
                asymmetric_values = asymmetric_explainer.explain(test_instances, method='monte_carlo')
                asymmetric_importance = asymmetric_explainer.get_feature_importance()
                
                # Save Asymmetric results
                asym_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'asymmetric'
                asym_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(asym_dir / 'shapley_values.npy', asymmetric_values)
                asymmetric_importance.to_csv(asym_dir / 'feature_importance.csv', index=False)
                
                # Calculate CausalShapley
                logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + {discovery_method.upper()} - CausalShapley (SLOW - uses {M_INNER_SAMPLES_CAUSAL} inner samples)")
                progress_counter += 1
                causal_explainer = CausalShapley(
                    model,  # Pass wrapper object, not model.model,
                    background_data=background_data,
                    discovered_adj=discovered_adj,
                    discovered_conf=confounders,
                    feature_names=[f for f in feature_names if f != 'Y'],
                    n_samples=N_SHAPLEY_SAMPLES,
                    M_inner_samples=M_INNER_SAMPLES_CAUSAL,  # Use reduced value
                    random_state=RANDOM_STATE
                )
                causal_values = causal_explainer.explain(test_instances)
                causal_importance = causal_explainer.get_feature_importance()
                
                # Save Causal results
                causal_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'causal'
                causal_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(causal_dir / 'shapley_values.npy', causal_values)
                causal_importance.to_csv(causal_dir / 'feature_importance.csv', index=False)
                
                # Calculate ShapleyFlow
                logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + {discovery_method.upper()} - ShapleyFlow")
                sys.stdout.flush()  # Ensure output is visible
                progress_counter += 1
                
                # Prepare data with Y for ShapleyFlow
                background_data_flow = X_train.copy()
                background_data_flow['Y'] = y_train
                test_instances_flow = test_instances.copy()
                test_instances_flow['Y'] = y_test.loc[test_instances.index]
                
                flow_explainer = ShapleyFlowWrapper(
                    model=model,  # Pass wrapper object, not model.model
                    background_data=background_data_flow,
                    causal_graph=causal_graph,
                    y_index=len(feature_names) - 1,  # Y is last column
                    n_samples=N_SHAPLEY_SAMPLES,
                    random_state=RANDOM_STATE
                )
                flow_values = flow_explainer.explain(test_instances_flow)
                flow_importance = flow_explainer.get_feature_importance()
                
                # Save Flow results
                flow_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'flow'
                flow_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(flow_dir / 'shapley_values.npy', flow_values)
                flow_importance.to_csv(flow_dir / 'feature_importance.csv', index=False)
        
        elapsed = time.time() - start_time
        logging.info(f"  ✓ Completed {filename} in {elapsed/60:.1f} minutes")
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 5 Complete: Shapley values calculated for all combinations")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 6: Calculate Comparison Metrics
# ============================================================================

def calculate_comparison_metrics(dataset_configs: List[Dict]):
    """
    Create PC vs LiNGAM comparison visualizations for mixed_no_conf dataset.
    
    This step loads the saved SHAP values from Step 5 and creates:
    - Visualization comparing PC vs LiNGAM for top 10 features from Scratch
    - Shows Scratch, Asymmetric, and Causal methods
    
    Parameters:
    -----------
    dataset_configs : List[Dict]
        Dataset configurations from Step 1
    """
    logging.info("=" * 80)
    logging.info("STEP 6: CREATING PC vs LiNGAM COMPARISON (mixed_no_conf + LGBM)")
    logging.info("=" * 80)
    
    # Filter to only target dataset
    target_dataset = TARGET_DATASET
    filtered_configs = [c for c in dataset_configs if c['filename'] == target_dataset]

    logging.info("=" * 80)
    logging.info(f"STEP 6: CREATING PC vs LiNGAM COMPARISON ({TARGET_DATASET} + LGBM)")
    logging.info("=" * 80)
    
    if not filtered_configs:
        logging.warning(f"Target dataset {target_dataset} not found!")
        return
    
    for ds_idx, config in enumerate(filtered_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {ds_idx}/1] Processing {filename}")
        
        # Load metadata for y_parent_indices
        metadata_path = SYNTHETIC_DIR / f"{filename}_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        y_parent_indices = metadata['y_parent_indices']
        true_parents = set(y_parent_indices)
        
        # Load train data for feature names
        train_path = PROCESSED_DIR / f"{filename}_train.parquet"
        train_data = pd.read_parquet(train_path)
        X_train = train_data.drop(columns=['Y'])
        feature_names = X_train.columns.tolist()
        
        # Process LGBM model only
        model_name = 'lgbm'
        logging.info(f"  Creating PC vs LiNGAM comparison for {model_name.upper()}...")
        
        # Load Scratch importance to get top 10 features
        scratch_dir = EXPLAINABILITY_DIR / filename / model_name / 'scratch'
        scratch_values = np.load(scratch_dir / 'shapley_values.npy')
        scratch_importance = np.abs(scratch_values).mean(axis=0)
        top_10_indices = np.argsort(scratch_importance)[-10:][::-1]  # Top 10 in descending order
        top_10_features = [feature_names[i] for i in top_10_indices]
        
        logging.info(f"  Top 10 features from Scratch: {top_10_features}")
        
        # Load SHAP values for both discovery methods
        methods_data = {}
        for discovery_method in ['pc', 'lingam']:
            # Load SHAP values for this discovery method
            asym_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'asymmetric'
            causal_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'causal'
            flow_dir = EXPLAINABILITY_DIR / filename / model_name / discovery_method / 'flow'
            
            asymmetric_values = np.load(asym_dir / 'shapley_values.npy')
            causal_values = np.load(causal_dir / 'shapley_values.npy')
            flow_values = np.load(flow_dir / 'shapley_values.npy')
            
            # Calculate mean importance for top 10 features
            asym_importance_top10 = np.abs(asymmetric_values)[:, top_10_indices].mean(axis=0)
            causal_importance_top10 = np.abs(causal_values)[:, top_10_indices].mean(axis=0)
            flow_importance_top10 = np.abs(flow_values)[:, top_10_indices].mean(axis=0)
            scratch_importance_top10 = scratch_importance[top_10_indices]
            
            methods_data[discovery_method] = {
                'Scratch': scratch_importance_top10,
                'Asymmetric': asym_importance_top10,
                'Causal': causal_importance_top10,
                'Flow': flow_importance_top10
            }
        
        # Create PC vs LiNGAM comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        
        x_pos = np.arange(len(top_10_features))
        width = 0.2  # Slightly narrower to fit 4 methods
        
        for ax_idx, discovery_method in enumerate(['pc', 'lingam']):
            ax = axes[ax_idx]
            data = methods_data[discovery_method]
            
            # Plot bars for each method (4 methods now)
            ax.barh(x_pos - 1.5*width, data['Scratch'], width, label='Scratch', alpha=0.8)
            ax.barh(x_pos - 0.5*width, data['Asymmetric'], width, label='Asymmetric', alpha=0.8)
            ax.barh(x_pos + 0.5*width, data['Causal'], width, label='Causal', alpha=0.8)
            ax.barh(x_pos + 1.5*width, data['Flow'], width, label='Flow', alpha=0.8)
            
            ax.set_yticks(x_pos)
            ax.set_yticklabels(top_10_features)
            ax.set_xlabel('Mean Absolute SHAP Value', fontsize=12)
            ax.set_title(f'{discovery_method.upper()} Discovery', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='x')
        
        axes[0].invert_yaxis()  # Highest importance at top
        fig.suptitle(f'{filename} - LGBM Model\nTop 10 Features: PC vs LiNGAM Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save visualization
        comparison_dir = EXPLAINABILITY_DIR / filename / model_name
        comparison_dir.mkdir(parents=True, exist_ok=True)
        viz_path = comparison_dir / 'pc_vs_lingam_comparison.png'
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logging.info(f"  ✓ Saved PC vs LiNGAM comparison visualization")
        
        # COMMENTED OUT: Parent identification metrics (for future use)
        # for discovery_method in ['pc', 'lingam']:
        #     # Load discovery results for discovered parents
        #     results_path = CAUSAL_DIR / f"{filename}_{discovery_method}_results.json"
        #     with open(results_path, 'r') as f:
        #         discovery_results = json.load(f)
        #     
        #     causal_graph = np.array(discovery_results['adjacency_matrix'])
        #     discovered_parents_full = set()
        #     for i in range(causal_graph.shape[0] - 1):
        #         if causal_graph[i, -1] == 1:
        #             discovered_parents_full.add(i)
        #     
        #     # Calculate parent identification metrics...
        #     # (code omitted for brevity)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 6 Complete: PC vs LiNGAM comparison visualization created")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run the complete experimental pipeline."""
    setup_logging()
    
    logging.info("\n" + "="*80)
    logging.info("EXPERIMENTAL PIPELINE: CAUSAL FEATURE IMPORTANCE COMPARISON")
    logging.info("="*80)
    
    logging.info(f"\n{'='*80}")
    logging.info("CONFIGURATION PARAMETERS")
    logging.info(f"{'='*80}")
    
    logging.info("\n1. DATASET PARAMETERS:")
    logging.info(f"   - N_FEATURES: {N_FEATURES}")
    logging.info(f"   - N_SAMPLES: {N_SAMPLES}")
    logging.info(f"   - Y_PARENTS_RATIO: {Y_PARENTS_RATIO}")
    logging.info(f"   - NOISE_STD: {NOISE_STD}")
    logging.info(f"   - EDGE_PROBABILITY: {EDGE_PROBABILITY}")
    logging.info(f"   - MIN_CONNECTED_EDGES: {MIN_CONNECTED_EDGES}")
    logging.info(f"   - RANDOM_STATE: {RANDOM_STATE}")
    
    logging.info("\n2. MODEL TRAINING PARAMETERS:")
    logging.info(f"   - TEST_SIZE: {TEST_SIZE}")
    logging.info(f"   - BACKGROUND_RATIO: {BACKGROUND_RATIO}")
    logging.info(f"   - TEST_INSTANCES_RATIO: {TEST_INSTANCES_RATIO}")
    
    logging.info("\n3. EXPLAINABILITY PARAMETERS:")
    logging.info(f"   - N_SHAPLEY_SAMPLES: {N_SHAPLEY_SAMPLES}")
    logging.info(f"   - M_INNER_SAMPLES_CAUSAL: {M_INNER_SAMPLES_CAUSAL}")
    
    logging.info("\n4. CAUSAL DISCOVERY PARAMETERS:")
    logging.info(f"   - LINGAM_ALPHA: {LINGAM_ALPHA}")
    logging.info(f"   - PC_ALPHA: {PC_ALPHA}")
    logging.info(f"   - INDEP_TEST: {INDEP_TEST}")
    
    logging.info("\n5. TARGET CONFIGURATION:")
    logging.info(f"   - TARGET_DATASET: {TARGET_DATASET}")
    logging.info(f"   - MODEL: LGBM (only)")
    
    logging.info("\n6. DIRECTORIES:")
    logging.info(f"   - BASE_DIR: {BASE_DIR}")
    logging.info(f"   - SYNTHETIC_DIR: {SYNTHETIC_DIR}")
    logging.info(f"   - PROCESSED_DIR: {PROCESSED_DIR}")
    logging.info(f"   - CAUSAL_DIR: {CAUSAL_DIR}")
    logging.info(f"   - EXPLAINABILITY_DIR: {EXPLAINABILITY_DIR}")
    logging.info(f"   - MODELS_DIR: {MODELS_DIR}")
    logging.info(f"   - LOGS_DIR: {LOGS_DIR}")
    
    logging.info(f"\n{'='*80}\n")
    
    pipeline_start = time.time()
    
    try:
        # Step 1: Generate datasets
        dataset_configs = generate_all_datasets()
        
        # Step 2: Create train/test splits
        create_train_test_splits(dataset_configs)
        
        # Step 3: Run causal discovery
        run_causal_discovery(dataset_configs)
        
        # Step 4: Train models
        train_all_models(dataset_configs)
        
        # Step 5: Calculate Shapley values
        calculate_all_shapley_values(dataset_configs)
        
        # Step 6: Calculate comparison metrics
        calculate_comparison_metrics(dataset_configs)
        
        total_time = time.time() - pipeline_start
        
        logging.info("\n" + "="*80)
        logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logging.info("="*80)
        logging.info(f"\nTotal execution time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
        logging.info(f"\nResults saved in:")
        logging.info(f"  - Synthetic data: {SYNTHETIC_DIR}")
        logging.info(f"  - Processed data: {PROCESSED_DIR}")
        logging.info(f"  - Causal discovery: {CAUSAL_DIR}")
        logging.info(f"  - Models: {MODELS_DIR}")
        logging.info(f"  - Explainability: {EXPLAINABILITY_DIR}")
        logging.info(f"  - Logs: {LOGS_DIR}")
        logging.info("\n" + "="*80 + "\n")
        
    except Exception as e:
        logging.error(f"\n{'='*80}")
        logging.error(f"ERROR: Pipeline failed!")
        logging.error(f"{'='*80}")
        logging.error(f"Error message: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
