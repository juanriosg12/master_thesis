# Evaluating the Impact of Discovered Causal Structures on Structure-Aware Shapley Methods

Experimental codebase for the master's thesis: *"Evaluating the Impact of Discovered Causal Structures on Structure-Aware Shapley Methods: An Experimental Pipeline"*.

---

## Folder Structure

| Folder | Description |
|---|---|
| `syntethic_data/` | Synthetic causal dataset generator (linear, nonlinear, mixed; with/without confounders) |
| `causal_discovery/` | PC and DirectLiNGAM wrappers for causal graph discovery |
| `predictive_models/` | LightGBM and Neural Network regressors |
| `explainability_models/` | Shapley value implementations: Asymmetric, Causal, ShapleyFlow, and Scratch |
| `utils/` | Plotting utilities, style helpers, and analysis functions |
| `experimental_pipeline/` | Main pipeline script (see below) |
| `notebooks/` | Analysis and result summary notebooks |
| `data/` | Generated datasets, causal graphs, and Shapley outputs (not tracked in git) |
| `models/` | Trained model files and metrics JSON |
| `logs/` | Pipeline run logs |

---

## Running the Pipeline

**Requirements**

```bash
# macOS: LightGBM requires OpenMP
brew install libomp

pip install -r requirements.txt
```

**Run**

```bash
cd /path/to/master_thesis
python experimental_pipeline/experimental_pipeline.py
```

The pipeline executes 5 steps in order:
1. **Generate datasets** — 6 synthetic causal datasets (linear/nonlinear/mixed × confounded/clean)
2. **Train/test splits** — 80/20 split saved to `data/processed/`
3. **Train models** — one LightGBM regressor per dataset, saved to `models/`
4. **Causal discovery** — PC and LiNGAM on training data, saved to `data/causal/`
5. **Shapley values** — 10 method × graph combinations per target dataset, saved to `data/explainability/`

Target datasets and other parameters can be configured at the top of `experimental_pipeline/experimental_pipeline.py`.
