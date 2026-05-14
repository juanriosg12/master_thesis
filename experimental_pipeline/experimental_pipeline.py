"""
Experimental Pipeline: Causal Feature Importance Comparison

Compares three causal Shapley value methods — Asymmetric Shapley, Causal Shapley,
and Shapley Flow — across three causal graph sources (PC, LiNGAM, True DAG) on
six synthetic datasets that vary in functional form (linear / nonlinear / mixed)
and confounding structure (with / without confounders).

Execution order
---------------
1. generate_all_datasets      — produce 6 synthetic datasets (X features + Y)
2. create_train_test_splits   — 80 / 20 stratified random split, saved to parquet
3. train_all_models           — fit one LGBMRegressor per dataset (all 6)
4. run_causal_discovery       — run DirectLiNGAM and PC on X-only train data;
                                augment with X→Y edges; build and persist true_full_adj
5. calculate_all_shapley_values — compute 10 method × graph combinations per
                                  target dataset and save shapley_values.npy

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
    ShapleyFromScratch, AsymmetricShapley,
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
N_SHAPLEY_SAMPLES = 100  # this could be increased if shapley flow performs well and we want more stable estimates (currently set low for faster debugging)
M_INNER_SAMPLES_CAUSAL = 10  # Reduced for CausalShapley performance (was 50 default)

# Discovery parameters
LINGAM_ALPHA = 0.05
PC_ALPHA = 0.05
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
# TARGET_DATASETS = ["mixed_conf_f50_s1000_p50", "mixed_no_conf_f50_s1000_p50"]
TARGET_DATASETS = ["mixed_no_conf_f50_s1000_p50","linear_conf_f50_s1000_p50"]
# Create directories
for directory in [SYNTHETIC_DIR, PROCESSED_DIR, CAUSAL_DIR, EXPLAINABILITY_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Configure file + stdout logging with timestamps.

    Creates a timestamped log file under LOGS_DIR (e.g.
    ``pipeline_20260511_142300.log``) and simultaneously streams all output to
    stdout so progress is visible during a long run.

    Returns
    -------
    logging.Logger
        Module-level logger (not strictly needed; callers use the root logger).
    """
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
    Generate the 6 synthetic causal datasets used in the experiment.

    The six datasets are the Cartesian product of:
      * functional form : linear | nonlinear | mixed (50 % linear / 50 % nonlinear)
      * confounding     : no confounders | with confounders

    Each dataset is produced by ``SyntheticCausalSystem`` with a shared random
    seed (``RANDOM_STATE``) and the following global parameters:

      * N_FEATURES = 50 input features (X1 … X50) + 1 target (Y)
      * N_SAMPLES  = 1 000 total observations
      * Y_PARENTS_RATIO = 0.50  → 25 of the 50 features are direct parents of Y
      * NOISE_STD = 0.5  added to every structural equation
      * EDGE_PROBABILITY = 0.2  (Erdős–Rényi probability for X→X edges)
      * MIN_CONNECTED_EDGES = 2  (minimum degree for connectivity guarantee)
      * mixed systems use linear_ratio = 0.5

    For each dataset the following files are written to ``data/synthetic/``:
      * ``{filename}.parquet``           — full data (X columns + Y column)
      * ``{filename}_adjacency.npy``     — true adjacency matrix, shape (51, 51),
                                           adj[i,j]=1 means i→j
      * ``{filename}_confounders.json``  — list of confounder pairs
      * ``{filename}_y_params.json``     — weights / parameters used to generate Y
      * ``{filename}_metadata.json``     — experiment parameters + y_parent_indices

    Returns
    -------
    dataset_configs : List[Dict]
        One metadata dict per dataset (same content as the saved JSON files).
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
    Create and persist 80 / 20 train–test splits for all 6 datasets.

    The split is performed with ``sklearn.model_selection.train_test_split``
    using ``TEST_SIZE = 0.20`` and ``RANDOM_STATE = 42`` so the partition is
    identical across all runs.

    With N_SAMPLES = 1 000 this yields:
      * 800 training rows   (used for model training and causal discovery)
      * 200 test rows       (used for SHAP evaluation)

    Each split is saved as a parquet file under ``data/processed/``:
      * ``{filename}_train.parquet``  — 800 rows, X columns + Y column
      * ``{filename}_test.parquet``   — 200 rows, X columns + Y column

    Parameters
    ----------
    dataset_configs : List[Dict]
        Dataset configurations returned by ``generate_all_datasets``.
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
# Step 4: Run Causal Discovery
# ============================================================================

def run_causal_discovery(dataset_configs: List[Dict]):
    """
    Run DirectLiNGAM and PC causal discovery on all 6 datasets and build the
    ground-truth reference graph (True DAG) used for both accuracy evaluation
    and True-DAG Shapley computation.

    Discovery algorithms
    --------------------
    Both algorithms operate on the **X-only training set** (Y excluded from
    the input; 800 rows, full dataset — no subsampling):

    * **DirectLiNGAM** (``causal-learn``)
        - Fits a linear non-Gaussian acyclic model.
        - ``adjacency_matrix_[i,j]`` is the coefficient of X_j *on* X_i, so
          the matrix is transposed before binarisation to obtain the convention
          ``adj[i,j] = 1 ↔ edge i → j``.
        - Edges with |coefficient| < 0.10 are pruned to zero.

    * **PC** (``causal-learn``, Fisher-z conditional independence test,
      ``PC_ALPHA = 0.05``)
        - Produces a CPDAG.
        - Directed edges (``graph[i,j] == -1 and graph[j,i] == 1``) are
          preserved as-is.
        - Undirected edges (``graph[i,j] == -1 and graph[j,i] == -1``) are
          arbitrarily oriented as i→j (lower index to higher).

    X→Y edge augmentation
    ---------------------
    After X-only discovery, edges to Y are added so that the full (n+1)×(n+1)
    adjacency matrix reflects what the LGBM model actually uses:
      1. X_i → Y for every feature in ``model.selected_features`` (for datasets
         in TARGET_DATASETS where a trained model exists).
      2. X_i → Y for every sink node — any X_i with no outgoing X→X edges —
         so that Y remains the terminal sink for all datasets.

    True DAG reference (true_full_adj)
    -----------------------------------
    The raw data-generation adjacency (X→X only, shape 50×50) is extended to a
    (51×51) matrix using the same two rules above.  This becomes the canonical
    ground-truth graph used for:
      * Accuracy metrics (F1, Precision, Recall) of PC and LiNGAM.
      * True-DAG Shapley values in Step 5.
    The matrix is persisted to ``data/causal/{filename}_true_full_adjacency.npy``
    so that Step 5 can load it without recomputing.

    Saved outputs (``data/causal/``)
    ----------------------------------
    For each dataset × method (lingam / pc):
      * ``{filename}_{method}_results.json``          — full (n+1)×(n+1) adj + feature names
      * ``{filename}_{method}_train_adjacency.npy``   — X-only (n×n) adj for CausalShapley
      * ``{filename}_{method}_comparison.json``       — F1 / Precision / Recall vs true_full_adj
      * ``{filename}_{method}_visualization.png``     — full graph comparison plot
      * ``{filename}_{method}_visualization_filtered.png`` — sink-filtered comparison plot
    Additionally:
      * ``{filename}_true_full_adjacency.npy``        — True DAG reference matrix

    Parameters
    ----------
    dataset_configs : List[Dict]
        Dataset configurations returned by ``generate_all_datasets``.
    """
    logging.info("=" * 80)
    logging.info("STEP 4: RUNNING CAUSAL DISCOVERY")
    logging.info("=" * 80)
    
    for idx, config in enumerate(dataset_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {idx}/{len(dataset_configs)}] Processing {filename}")
        
        # Load train data and drop Y (discovery runs on X only)
        data_path = PROCESSED_DIR / f"{filename}_train.parquet"
        data = pd.read_parquet(data_path)
        X_train = data.drop(columns=['Y'])

        adj_path = SYNTHETIC_DIR / f"{filename}_adjacency.npy"
        true_adj = np.load(adj_path)
        
        conf_path = SYNTHETIC_DIR / f"{filename}_confounders.json"
        with open(conf_path, 'r') as f:
            true_conf = json.load(f)
        
        # Load trained LGBM model if available (TARGET_DATASETS only)
        trained_model = None
        if filename in TARGET_DATASETS:
            model_path = MODELS_DIR / f"{filename}_lgbm.pkl"
            if model_path.exists():
                trained_model = LGBMRegressor.load(str(MODELS_DIR / f"{filename}_lgbm"))
                logging.info(f"  Loaded LGBM model ({len(trained_model.selected_features)} features used for Y edges)")
            else:
                logging.warning(f"  No trained model found for {filename}; Y edges determined by sink nodes only")

        # Build true_full_adj: canonical reference graph for accuracy metrics.
        # Matches the construction used in Step 5 (calculate_all_shapley_values):
        #   • X→Y for every feature used by the LGBM model (if available)
        #   • X→Y for every sink node in true_adj_xx (no outgoing X→X edges)
        _feature_names_list = X_train.columns.tolist()
        _n_feat = len(_feature_names_list)
        _true_adj_xx = true_adj[:_n_feat, :_n_feat]
        true_full_adj = np.zeros((_n_feat + 1, _n_feat + 1), dtype=int)
        true_full_adj[:_n_feat, :_n_feat] = _true_adj_xx
        if trained_model is not None:
            for _feat in trained_model.selected_features:
                if _feat in _feature_names_list:
                    _fi = _feature_names_list.index(_feat)
                    true_full_adj[_fi, _n_feat] = 1
        for _fi in range(_n_feat):
            if _true_adj_xx[_fi, :].sum() == 0:
                true_full_adj[_fi, _n_feat] = 1
        _y_edges_ref = int(true_full_adj[:_n_feat, _n_feat].sum())
        logging.info(f"  True DAG reference: {int(_true_adj_xx.sum())} X→X edges + {_y_edges_ref} X→Y edges")

        # Persist true_full_adj so Step 5 can load it without recomputing
        true_full_adj_path = CAUSAL_DIR / f"{filename}_true_full_adjacency.npy"
        np.save(true_full_adj_path, true_full_adj)
        logging.info(f"  True DAG reference saved to {true_full_adj_path.name}")

        # Initialize discovery methods
        lingam_fci = LiNGAMWithFCI(alpha=LINGAM_ALPHA, indep_test=INDEP_TEST)
        pc_fci = PCWithFCI(alpha=PC_ALPHA, indep_test=INDEP_TEST)
        
        # Run discovery (X only; model determines Y edges inside the method)
        logging.info(f"  Running LiNGAM on full training set ({len(X_train)} rows)...")
        results_lingam = lingam_fci.get_causal_relationships(X_train, model=trained_model)
        
        logging.info(f"  Running PC on full training set ({len(X_train)} rows)...")
        results_pc = pc_fci.get_causal_relationships(X_train, model=trained_model)
        
        logging.info("  Extracting X-only train adjacency matrices for Step 5...")
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
            
            # Save X-only train adjacency matrix for Step 5 (CausalShapley)
            if method_name == 'lingam':
                train_adj_path = CAUSAL_DIR / f"{filename}_{method_name}_train_adjacency.npy"
                np.save(train_adj_path, lingam_train_adj)
            else:  # pc
                train_adj_path = CAUSAL_DIR / f"{filename}_{method_name}_train_adjacency.npy"
                np.save(train_adj_path, pc_train_adj)
            
            # Compare with ground truth using true_full_adj (LGBM features + sink
            # nodes as X→Y edges) so that accuracy metrics account for the model's
            # view of which features matter, not just the raw data-generation parents.
            comparison = compare_with_ground_truth(
                discovered_adj=results['adjacency_matrix'],
                true_adj=true_full_adj,
                discovered_confounders=results['confounders'],
                true_confounders=true_conf
            )
            
            # Save comparison metrics
            comparison_path = CAUSAL_DIR / f"{filename}_{method_name}_comparison.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logging.info(f"  ✓ {method_name.upper()}: F1={comparison.get('f1_score', 0):.3f}, Precision={comparison.get('precision', 0):.3f}, Recall={comparison.get('recall', 0):.3f}")
            
            # Visualizations — feature_names from results already includes 'Y'
            all_feature_names = results['feature_names']
            fig = visualize_comparison(
                true_adj=true_full_adj,
                discovered_adj=results['adjacency_matrix'],
                features_names=all_feature_names,
                y_parent_indices=config['y_parent_indices'],
                true_confounders=true_conf,
                discovered_confounders=results['confounders'],
                title_preix=f"{method_name.upper()} Algorithm"
            )

            fig_filtered = visualize_comparison(
                true_adj=true_full_adj,
                discovered_adj=results['adjacency_matrix'],
                features_names=all_feature_names,
                y_parent_indices=config['y_parent_indices'],
                true_confounders=true_conf,
                discovered_confounders=results['confounders'],
                title_preix=f"{method_name.upper()} Algorithm",
                filter_to_sink=True
            )
            
            # Save visualization
            fig_path = CAUSAL_DIR / f"{filename}_{method_name}_visualization.png"
            fig_filtered_path = CAUSAL_DIR / f"{filename}_{method_name}_visualization_filtered.png"
            fig.savefig(fig_path, dpi=300, bbox_inches='tight')
            fig_filtered.savefig(fig_filtered_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plt.close(fig_filtered)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Step 4 Complete: Causal discovery completed for {len(dataset_configs)} datasets")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 3: Train Predictive Models
# ============================================================================

def train_all_models(dataset_configs: List[Dict]):
    """
    Fit one LightGBM regressor per dataset (all 6 datasets).

    Feature selection is **disabled** (``feature_selection=False``) so that
    the model uses all 50 input features, which keeps the feature space
    consistent with the causal graph and SHAP computation.

    For each dataset the following are computed on the 200-row test split and
    logged / saved to ``models/{filename}_metrics.json``:
      * R²  (coefficient of determination)
      * MSE (mean squared error)
      * MAE (mean absolute error)
      * RMSE (root mean squared error)

    The trained model is serialised to:
      * ``models/{filename}_lgbm.pkl``  (via ``LGBMRegressor.save``)

    ``model.selected_features`` contains the list of all feature names (all 50)
    and is used downstream by ``run_causal_discovery`` to assign X→Y edges in
    the reference graph.

    Parameters
    ----------
    dataset_configs : List[Dict]
        Dataset configurations returned by ``generate_all_datasets``.
    """
    logging.info("=" * 80)
    logging.info(f"STEP 3: TRAINING PREDICTIVE MODELS (LGBM only, {len(dataset_configs)} datasets — all)")
    logging.info("=" * 80)

    for idx, config in enumerate(dataset_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {idx}/{len(dataset_configs)}] Training LGBM model for {filename}")
        
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
        lgbm_model.fit(X_train, y_train,feature_selection=False)  # Disable feature selection for consistency with explainability step
        
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
        logging.info(f"  ✓ LGBM model saved with the following number of features: {len(lgbm_model.selected_features)}")
        # NN training skipped for focused experiment
        # # Train NN
        # logging.info("  Training NN...")
        # nn_model = NeuralNetRegressor()
        # nn_model.fit(X_train, y_train, feature_selection=False)
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
    logging.info(f"Step 3 Complete: LGBM model trained for {len(dataset_configs)} datasets")
    logging.info(f"{'='*80}\n")


# ============================================================================
# Step 5: Calculate Shapley Values
# ============================================================================

def calculate_all_shapley_values(dataset_configs: List[Dict]):
    """
    Compute and persist Shapley values for all method × causal-graph combinations
    on the TARGET_DATASETS.

    Only datasets listed in ``TARGET_DATASETS`` are processed.

    Data setup
    ----------
    For each target dataset:
      * Background data : random sample of 30 % of training rows
        (``BACKGROUND_RATIO = 0.30`` → ~240 rows), used as the reference
        distribution for all methods.
      * Test instances  : random sample of 50 % of test rows
        (``TEST_INSTANCES_RATIO = 0.50`` → ~100 rows), the instances whose
        predictions are explained.
    Both samples are drawn with ``RANDOM_STATE = 42`` for reproducibility.

    Combinations computed  (10 total per dataset)
    ----------------------------------------------
    1. **Scratch**                — ``ShapleyFromScratch`` (no causal graph),
                                   Monte Carlo permutation, N_SHAPLEY_SAMPLES = 100
    2. **PC + Asymmetric**        — ``AsymmetricShapley`` with PC full_adj (n+1 × n+1)
    3. **PC + Causal**            — ``CausalShapley`` with PC X-only adj (n × n)
    4. **PC + Flow**              — ``ShapleyFlowWrapper`` with PC full_adj
    5. **LiNGAM + Asymmetric**    — ``AsymmetricShapley`` with LiNGAM full_adj
    6. **LiNGAM + Causal**        — ``CausalShapley`` with LiNGAM X-only adj
    7. **LiNGAM + Flow**          — ``ShapleyFlowWrapper`` with LiNGAM full_adj
    8. **True + Asymmetric**      — ``AsymmetricShapley`` with true_full_adj
    9. **True + Causal**          — ``CausalShapley`` with true X-only adj (raw 50 × 50)
    10. **True + Flow**           — ``ShapleyFlowWrapper`` with true_full_adj

    Causal graph sources
    --------------------
    * PC / LiNGAM : loaded from ``data/causal/{filename}_{method}_results.json``
      (full n+1 × n+1 adj) and ``{method}_train_adjacency.npy`` (X-only adj).
    * True DAG    : ``true_full_adj`` loaded from
      ``data/causal/{filename}_true_full_adjacency.npy`` (produced by Step 4);
      the raw X-only block is re-extracted for CausalShapley.

    ShapleyFlow data format
    -----------------------
    ShapleyFlow requires Y in the data frame.  Background and test frames are
    extended with the corresponding Y column before being passed to
    ``ShapleyFlowWrapper``.

    Saved outputs (``data/explainability/{dataset}/lgbm/``)
    --------------------------------------------------------
    Each combination writes two files:
      * ``{graph}/{method}/shapley_values.npy``      — shape (n_test_instances, n_features)
      * ``{graph}/{method}/feature_importance.csv``  — mean |SHAP| per feature

    Parameters
    ----------
    dataset_configs : List[Dict]
        Dataset configurations returned by ``generate_all_datasets``.
    """
    logging.info("=" * 80)
    logging.info(f"STEP 5: CALCULATING SHAPLEY VALUES ({len(TARGET_DATASETS)} datasets + LGBM only)")
    logging.info("=" * 80)
    
    # Filter to only target datasets
    filtered_configs = [c for c in dataset_configs if c['filename'] in TARGET_DATASETS]
    
    if not filtered_configs:
        logging.warning(f"Target datasets {TARGET_DATASETS} not found!")
        return
    
    for ds_idx, config in enumerate(filtered_configs, 1):
        filename = config['filename']
        start_time = time.time()
        logging.info(f"\n[Dataset {ds_idx}/{len(filtered_configs)}] Processing {filename}")
        
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
        total_combinations = 1 + 2 * 3 + 3  # 1 scratch + 2 disc methods × 3 causal methods + 3 True-graph methods
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
                    causal_graph=causal_graph,
                    n_samples=N_SHAPLEY_SAMPLES,
                    random_state=RANDOM_STATE,
                )
                logging.info(" Calculating asymmetric SHAP values")
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

            # ── True DAG Shapley values ────────────────────────────────────────────
            # Load true_full_adj saved by run_causal_discovery (Step 4):
            # that step builds X→X from raw adjacency + X→Y for LGBM features
            # and sink nodes, then persists it as {filename}_true_full_adjacency.npy.
            logging.info(f"\n  ─── True DAG (ground-truth causal structure) ───")
            feature_names_xx = X_train.columns.tolist()
            n_feat_xx = len(feature_names_xx)
            true_adj_xx_pipe = np.load(SYNTHETIC_DIR / f"{filename}_adjacency.npy")[:n_feat_xx, :n_feat_xx]

            true_full_adj = np.load(CAUSAL_DIR / f"{filename}_true_full_adjacency.npy")
            y_edges_true = int(true_full_adj[:n_feat_xx, n_feat_xx].sum())
            logging.info(f"  True DAG loaded: {int(true_adj_xx_pipe.sum())} X→X edges + {y_edges_true} X→Y edges")

            # AsymmetricShapley (True)
            logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + TRUE - AsymmetricShapley")
            progress_counter += 1
            asym_true_exp = AsymmetricShapley(
                model, background_data,
                causal_graph=true_full_adj,
                n_samples=N_SHAPLEY_SAMPLES,
                random_state=RANDOM_STATE,
            )
            asym_true_values = asym_true_exp.explain(test_instances, method='monte_carlo')
            asym_true_dir = EXPLAINABILITY_DIR / filename / model_name / 'true' / 'asymmetric'
            asym_true_dir.mkdir(parents=True, exist_ok=True)
            np.save(asym_true_dir / 'shapley_values.npy', asym_true_values)
            asym_true_exp.get_feature_importance().to_csv(asym_true_dir / 'feature_importance.csv', index=False)

            # CausalShapley (True) — X-only adj, no confounders
            logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + TRUE - CausalShapley")
            progress_counter += 1
            causal_true_exp = CausalShapley(
                model,
                background_data=background_data,
                discovered_adj=true_adj_xx_pipe,
                discovered_conf=[],
                feature_names=feature_names_xx,
                n_samples=N_SHAPLEY_SAMPLES,
                M_inner_samples=M_INNER_SAMPLES_CAUSAL,
                random_state=RANDOM_STATE,
            )
            causal_true_values = causal_true_exp.explain(test_instances)
            causal_true_dir = EXPLAINABILITY_DIR / filename / model_name / 'true' / 'causal'
            causal_true_dir.mkdir(parents=True, exist_ok=True)
            np.save(causal_true_dir / 'shapley_values.npy', causal_true_values)
            causal_true_exp.get_feature_importance().to_csv(causal_true_dir / 'feature_importance.csv', index=False)

            # ShapleyFlow (True)
            logging.info(f"  [Progress: {progress_counter}/{total_combinations}] {model_name.upper()} + TRUE - ShapleyFlow")
            progress_counter += 1
            true_bg_flow = background_data.copy()
            true_bg_flow['Y'] = y_train.loc[background_data.index]
            true_test_flow = test_instances.copy()
            true_test_flow['Y'] = y_test.loc[test_instances.index]
            flow_true_exp = ShapleyFlowWrapper(
                model=model,
                background_data=true_bg_flow,
                causal_graph=true_full_adj,
                y_index=n_feat_xx,
                n_samples=N_SHAPLEY_SAMPLES,
                random_state=RANDOM_STATE,
            )
            flow_true_values = flow_true_exp.explain(true_test_flow)
            flow_true_dir = EXPLAINABILITY_DIR / filename / model_name / 'true' / 'flow'
            flow_true_dir.mkdir(parents=True, exist_ok=True)
            np.save(flow_true_dir / 'shapley_values.npy', flow_true_values)
            flow_true_exp.get_feature_importance().to_csv(flow_true_dir / 'feature_importance.csv', index=False)
            logging.info(f"  ✓ True DAG Shapley values saved ({len(test_instances)} instances)")


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
    Produce a side-by-side PC vs LiNGAM bar chart for the top-10 features.

    Operates only on TARGET_DATASETS.  For each target dataset:

    1. Load mean |SHAP| from ``scratch/shapley_values.npy`` and rank all 50
       features by importance.
    2. Restrict to features that appear in at least one edge of the PC *or*
       LiNGAM discovered graph.
    3. Select the top 10 among those filtered features.
    4. For each discovery method (PC, LiNGAM), display four bars per feature:
         Scratch | Asymmetric | Causal | Flow

    The resulting figure is saved to:
      ``data/explainability/{dataset}/lgbm/pc_vs_lingam_comparison.png``

    Parameters
    ----------
    dataset_configs : List[Dict]
        Dataset configurations returned by ``generate_all_datasets``.
    """
    logging.info("=" * 80)
    logging.info(f"STEP 6: CREATING PC vs LiNGAM COMPARISON ({len(TARGET_DATASETS)} datasets + LGBM)")
    logging.info("=" * 80)
    
    # Filter to only target datasets
    filtered_configs = [c for c in dataset_configs if c['filename'] in TARGET_DATASETS]
    
    if not filtered_configs:
        logging.warning(f"Target datasets {TARGET_DATASETS} not found!")
        return
    
    for ds_idx, config in enumerate(filtered_configs, 1):
        filename = config['filename']
        logging.info(f"\n[Dataset {ds_idx}/{len(filtered_configs)}] Processing {filename}")
        
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
        
        # Load adjacency matrices for both discovery methods to get features in causal graph
        features_in_causal_graph = set()
        
        for discovery_method in ['pc', 'lingam']:
            results_path = CAUSAL_DIR / f"{filename}_{discovery_method}_results.json"
            with open(results_path, 'r') as f:
                discovery_results = json.load(f)
            
            causal_graph_adj = np.array(discovery_results['adjacency_matrix'])
            
            # Features that have at least one edge (either as parent or child, including edges to/from Y)
            has_edge = (causal_graph_adj.sum(axis=0) > 0) | (causal_graph_adj.sum(axis=1) > 0)
            
            # Remove Y (last element) from has_edge to only keep feature indices
            has_edge = has_edge[:-1]
            
            # Add feature indices to the set
            features_in_causal_graph.update(np.where(has_edge)[0])
        
        # Convert to sorted list for consistent ordering
        features_in_causal_graph = sorted(list(features_in_causal_graph))
        
        logging.info(f"  Features in causal graph (PC or LiNGAM): {len(features_in_causal_graph)} out of {len(feature_names)} features")
        
        # Filter scratch_importance to only features in causal graph
        if len(features_in_causal_graph) == 0:
            logging.warning(f"  No features found in causal graph! Using all features instead.")
            features_in_causal_graph = list(range(len(feature_names)))
        
        scratch_importance_filtered = scratch_importance[features_in_causal_graph]
        
        # Get top 10 from filtered features
        top_10_filtered_indices = np.argsort(scratch_importance_filtered)[-10:][::-1]
        top_10_indices = [features_in_causal_graph[i] for i in top_10_filtered_indices]
        top_10_features = [feature_names[i] for i in top_10_indices]
        
        logging.info(f"  Top 10 features from Scratch (filtered to causal graph): {top_10_features}")
        
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
                'Flow': flow_importance_top10,
            }
            
        
        # Create PC vs LiNGAM comparison visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
        
        x_pos = np.arange(len(top_10_features))
        width = 0.2
        
        for ax_idx, discovery_method in enumerate(['pc', 'lingam']):
            ax = axes[ax_idx]
            data = methods_data[discovery_method]
            
            # 4 methods: Scratch, Asymmetric, Causal, Flow
            ax.barh(x_pos - 1.5*width, data['Scratch'], width, label='Scratch', alpha=0.8)
            ax.barh(x_pos - 0.5*width, data['Asymmetric'], width, label='Asymmetric', alpha=0.8)
            ax.barh(x_pos + 0.5*width, data['Causal'], width, label='Causal', alpha=0.8)
            ax.barh(x_pos + 1.5*width, data['Flow'], width, label='Flow (Path)', alpha=0.8)
            
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
    logging.info(f"   - TARGET_DATASETS: {TARGET_DATASETS}")
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
        
        # Step 3: Train models (must run before causal discovery so the model
        #          can guide which features connect to Y in the causal graph)
        train_all_models(dataset_configs)

        # Step 4: Run causal discovery (uses trained models for Y-edge assignment)
        run_causal_discovery(dataset_configs)
        
        # Step 5: Calculate Shapley values
        calculate_all_shapley_values(dataset_configs)
        
        # # Step 6: Calculate comparison metrics
        # calculate_comparison_metrics(dataset_configs)
        
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
