# Dataset: `linear_conf_f50_s1000_p50`

## 1. Dataset Overview

The name encodes all key parameters:

| Token | Meaning | Value |
|---|---|---|
| `linear` | Functional form between features | Strictly additive linear |
| `conf` | Confounding structure | 5 hidden confounders present |
| `f50` | Number of input features | 50 (X0 … X49) |
| `s1000` | Number of observations | 1,000 |
| `p50` | Y-parents ratio | 0.5 → 25 direct causes of Y |

The full design matrix is 51 columns: 50 feature variables + 1 continuous target Y, with 1,000 rows → 80/20 split → **800 training / 200 test** observations. Random seed is fixed at 42.

---

## 2. Feature Relationships (Linear Structure)

Each feature $X_j$ is generated following the causal order of a DAG:

$$X_j = \sum_{i \in \text{Pa}(j)} w_{ij} \cdot X_i + \varepsilon_j, \quad \varepsilon_j \sim \mathcal{N}(0,\, 0.5)$$

- Structural coefficients $w_{ij} \sim \text{Uniform}(0.5,\, 2.0) \times \{-1, +1\}$ — randomly signed, bounded away from zero to avoid weak effects.
- After accumulating parent contributions, each variable is **normalized** (outliers clipped at ±5σ, then standardized) to prevent signal explosion along deep paths.
- The target Y is generated the same way from its 25 parents: $Y = \sum_{k \in \text{Pa}(Y)} \beta_k \cdot X_k + \varepsilon_Y$.

---

## 3. Graph Structure and Connectivity

The DAG skeleton over the 50 features is sampled from the **upper-triangular Erdős–Rényi model** (acyclicity by construction) with edge probability $p = 0.2$:

$$\text{Max possible X→X directed edges} = \binom{50}{2} = 1{,}225$$

| Quantity | Value |
|---|---|
| Actual X→X edges | **266** |
| Connectivity rate (X→X only) | $266 / 1{,}225 \approx$ **21.7%** |
| Average in-degree per feature | $266 / 50 \approx$ **5.3 parents/node** |
| Direct causes of Y (X→Y) | **25** (y_parents_ratio = 0.5) |
| Total edges in true full DAG | **316** (266 X→X + 50 X→Y\*) |

\* The true full adjacency used for causal discovery evaluation records **50 X→Y edges** because, with confounders present, the pipeline loads the trained LGBM model and uses all 50 features as potential Y predictors in the evaluation reference graph — reflecting the fact that confounders create indirect statistical paths from non-parents to Y.

---

## 4. What Confounders Mean in This Dataset

The dataset contains **5 hidden (latent) confounders**. Each confounder $H_c$ is a standard normal variable not included in the feature matrix, but it **additively influences 2–3 observed features simultaneously**:

$$X_i \mathrel{+}= \alpha_{ci} \cdot H_c, \quad \alpha_{ci} \sim \text{Uniform}(0.5,\, 1.5) \times \{-1, +1\}$$

**In practice this means:**
- Two observed features that share a hidden common cause will appear correlated in the data, even if there is no direct causal edge between them.
- Both **LiNGAM** and **PC** assume *causal sufficiency* (no hidden common causes). When this assumption is violated, both algorithms are expected to hallucinate spurious directed edges between the affected feature pairs.
- The comparison files confirm this: LiNGAM **over-discovers** with 445 found vs. 316 true edges; PC **under-discovers** with only 138 found edges but is more conservative.

---

## 5. Predictive Model: LightGBM Regressor

The model is a **LightGBMRegressor** with full **Optuna-based hyperparameter optimization** (50 TPE trials), using RMSE as the objective. The search space is:

| Hyperparameter | Range |
|---|---|
| `n_estimators` | [50, 500] |
| `learning_rate` | [0.01, 0.30] (log scale) |
| `num_leaves` | [20, 150] |
| `max_depth` | [3, 12] |
| `min_child_samples` | [5, 100] |
| `subsample` | [0.60, 1.0] |
| `colsample_bytree` | [0.60, 1.0] |
| `reg_alpha`, `reg_lambda` | [1e-8, 10.0] (log scale) |

Training uses an internal 80/20 validation split with **early stopping** (20 rounds). All 50 features were used (no feature selection was applied for this dataset).

**Test-set performance:**

| Metric | Value |
|---|---|
| R² | **0.9187** |
| MSE | 3.7283 |
| MAE | 1.5601 |
| RMSE | 1.9309 |

This is the **highest R² across all 6 datasets** in the experiment, which is expected: linear relationships are the easiest to capture with a gradient boosting model, and the confounders add signal (more variance is "explainable" through indirect paths) without disrupting the linear structure.

---

## 6. Causal Discovery Performance

Both algorithms were run on the **800-sample training set** with significance level $\alpha = 0.05$. The true DAG has **316 total edges** to recover.

| Algorithm | TP | FP | FN | Precision | Recall | F1 | SHD | Edges found |
|---|---|---|---|---|---|---|---|---|
| **LiNGAM** | 159 | 286 | 157 | 0.357 | **0.503** | **0.418** | 443 | 445 |
| **PC** | 83 | 55 | 233 | **0.601** | 0.263 | 0.366 | 288 | 138 |

**Reading the results:**

- **LiNGAM** recovers more true edges (higher recall = 0.503) but at the cost of many false positives (286 spurious edges). It over-estimates the graph density — discovering 445 edges vs. 316 true. The confounders cause LiNGAM to insert false directed edges between feature pairs with a shared hidden cause, since it cannot distinguish a direct path from a confounder-induced correlation.

- **PC** is more conservative and precise (precision = 0.601): when it reports an edge it is correct 60% of the time, but it misses 74% of the true edges (recall = 0.263). Its lower Structural Hamming Distance (SHD = 288 vs. 443) indicates the overall skeleton is less wrong, even though recall is poor.

- Both algorithms report detecting **46 "confounders"** via their FCI extensions — far above the ground truth of 5 — illustrating how difficult latent variable detection is in high-dimensional settings.

---

## Summary

`linear_conf_f50_s1000_p50` is a **medium-complexity, well-conditioned benchmark**: 1,000 samples over 50 linearly-related features with a moderately dense graph (~22% connectivity). The linear structure makes it the most learnable dataset for the LGBM model (R² = 0.92), providing a high-quality black-box to explain. The 5 hidden confounders represent the central challenge: they induce spurious correlations that violate the causal sufficiency assumptions of both discovery algorithms. LiNGAM exploits the linear non-Gaussian signal well (best recall = 0.50) but is contaminated by false edges; PC is selective but under-recovers. These imperfect graphs are then used as structural priors for causal Shapley methods, and the quality gap between them and the true DAG is precisely what the thesis aims to quantify.
