# Dataset: `sachs` — Sachs Cell Signalling (Real Data)

## 1. Dataset Overview

The Sachs et al. (2005) *Science* dataset contains simultaneous measurements of
11 protein and phospholipid concentrations in stimulated human T-cells, collected
under multiple experimental interventions. It is one of the most widely used
benchmarks for causal discovery because a consensus reference DAG is available
(Sachs 2005, Figure 3).

| Property | Value |
|---|---|
| Source | Sachs et al. (2005), *Science* 308(5721):523–529 |
| Observations | 7,466 |
| Variables | 11 protein/lipid concentrations |
| Target Y | `akt` (Akt kinase — downstream node in PI3K and PKA signalling pathways) |
| X features | 10: `raf`, `mek`, `plc`, `pip2`, `pip3`, `erk`, `pka`, `pkc`, `p38`, `jnk` |
| True DAG (X→X edges) | 16 directed edges |
| True DAG (X→Y edges) | 3 (`pip3→akt`, `pka→akt`, `erk→akt`) |
| True DAG (total) | **19 directed edges** |
| Train / Test split | 80 / 20 → **5,972 train / 1,494 test** (seed 42) |

---

## 2. Raw Data Characteristics

All 11 proteins are strictly positive, right-skewed (log-normal) concentrations
measured in fluorescence units. Minimum value is 1.0 across all variables.

| Protein | Mean | Std | Median | Max | Role |
|---|---|---|---|---|---|
| `raf` | 124.1 | 247.5 | 53.8 | 4,614 | MAPK kinase kinase |
| `mek` | 145.4 | 377.1 | 26.7 | 7,105 | MAPK kinase |
| `plc` | 54.9 | 173.9 | 16.5 | 6,208 | Phospholipase C |
| `pip2` | 151.1 | 299.3 | 52.8 | 9,058 | Phospholipid |
| `pip3` | 27.0 | 43.0 | 17.8 | 1,275 | Phospholipid |
| `erk` | 26.6 | 45.8 | 17.2 | 2,571 | ERK kinase |
| `pka` | 625.8 | 644.5 | 449.0 | 8,896 | Protein kinase A |
| `pkc` | 30.3 | 92.9 | 12.7 | 1,611 | Protein kinase C |
| `p38` | 135.0 | 494.8 | 30.5 | 7,499 | p38 MAP kinase |
| `jnk` | 73.3 | 215.7 | 18.4 | 4,740 | JNK kinase |
| **`akt`** (Y) | **81.2** | **137.8** | **37.2** | **3,555** | **Akt kinase (target)** |

**Target Y range (test set, 1,494 observations, raw scale):**

| Statistic | Value |
|---|---|
| Min | 1.67 |
| Max | 3,555.00 |
| Mean | 80.73 |
| Std | 156.66 |

---

## 3. Preprocessing Pipeline

Two transformations are applied to the X features before modelling and causal
discovery. `akt` is kept in its **raw (untransformed) scale** as the regression
target Y.

### Step 1 — log1p transform (before train/test split)

```python
X = np.log1p(X_raw)
```

- Stabilises the right-skewed log-normal distributions of all 10 proteins.
- Converts multiplicative relationships to approximately additive ones — a
  requirement for LiNGAM's linear SEM assumption and Fisher-z in PC.
- Applied globally before the split (deterministic transform, no data leakage).

### Step 2 — StandardScaler (fit on train only)

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit here
X_test  = scaler.transform(X_test)        # transform only
```

- Brings all proteins to μ=0, σ=1 so LiNGAM's |coefficient| < 0.10 pruning
  threshold is scale-consistent across proteins with very different raw ranges
  (e.g. `pka` mean = 626 vs. `pip3` mean = 27).
- Fitted **only on training data** to prevent leakage.

---

## 4. True Reference DAG (Sachs 2005 Consensus)

The consensus reference graph from the original paper encodes the known signalling
relationships among the 11 proteins. After removing `akt` (target Y), the
reference has:

**X→X edges (16):**

| From | To | Pathway |
|---|---|---|
| raf | mek | MAPK cascade |
| mek | erk | MAPK cascade |
| plc | pip2 | PIP metabolism |
| plc | pip3 | PIP metabolism |
| plc | pkc | PKC activation |
| pip2 | pkc | PKC activation |
| pip3 | plc | PIP feedback |
| pka | raf | PKA regulation |
| pka | mek | PKA regulation |
| pka | erk | PKA regulation |
| pka | p38 | PKA regulation |
| pka | jnk | PKA regulation |
| pkc | raf | PKC regulation |
| pkc | mek | PKC regulation |
| pkc | p38 | PKC regulation |
| pkc | jnk | PKC regulation |

**X→Y edges (3):** `pip3→akt`, `pka→akt`, `erk→akt`

The network is a **sparse 10-node DAG** with 16 directed edges — much smaller
and sparser than the 50-node synthetic DAGs (266 edges). This makes discovery
harder per-edge because spurious correlations from the complex multi-pathway
biology are not controlled, and the sample size (7,466) is much larger but the
true signal is entangled across pathways.

---

## 5. Predictive Model: LightGBM Regressor

The model is a **LightGBMRegressor** with full **Optuna-based hyperparameter
optimization** (50 TPE trials), using RMSE as the objective. All 10 X features
are used (`feature_selection=False`). Target Y (`akt`) is kept in raw scale.

**Test-set performance (on 1,494 held-out observations, Y in raw units):**

| Metric | Value |
|---|---|
| R² | **0.6786** |
| MSE | 7,881.47 |
| MAE | 17.82 |
| RMSE | 88.78 |

The moderate R² (0.68) reflects the inherent difficulty of predicting Akt kinase
activity from the other 10 proteins: the relationships are noisy, the data comes
from a mixture of different experimental interventions, and the raw-scale target
(mean = 81, max = 3,555) has high variability. This is substantially lower than
the synthetic linear datasets (R² ≈ 0.84–0.91), which is expected for real
biological data where the true generative process is far more complex than a
linear SEM.

---

## 6. Causal Discovery Performance

Both algorithms were run on the **5,972-row training set** (log1p + scaled) with
significance level $\alpha = 0.05$.

### 6.1 Full Graph (X→X + X→Y edges)

The pipeline appends X→Y edges for all features in `model.selected_features`
(all 10) plus any sink nodes, resulting in a full reference graph with
**19 true edges**. The X→Y portion (3 edges: `pip3→akt`, `pka→akt`, `erk→akt`)
is fixed by the pipeline heuristic and does not test discovery quality.

| Algorithm | TP | FP | FN | Precision | Recall | F1 | SHD | Edges found |
|---|---|---|---|---|---|---|---|---|
| **LiNGAM** | 9 | 27 | 10 | 0.250 | **0.474** | **0.327** | 37 | 36 |
| **PC** | 6 | 23 | 13 | 0.207 | 0.316 | 0.250 | 36 | 29 |

### 6.2 X→X Edges Only (actual causal structure recovered)

PC and LiNGAM only discover **X→X edges**. The X→X block has **16 true edges**.

| Algorithm | TP | FP | FN | Precision | Recall | F1 | SHD | Edges found |
|---|---|---|---|---|---|---|---|---|
| **LiNGAM** | 6 | 20 | 10 | **0.231** | **0.375** | **0.286** | 30 | 26 |
| **PC** | 3 | 16 | 13 | 0.158 | 0.188 | 0.171 | 29 | 19 |

**Reading the results:**

- **LiNGAM** recovers more true X→X edges (recall = 0.375) but at the cost of
  many false positives (20 spurious edges out of 26 reported). The linear
  non-Gaussian signal is present in the log-transformed data, but the multiple
  overlapping biological pathways and intervention-induced non-stationarity create
  false associations. F1 = 0.286.

- **PC** is again the more conservative algorithm (only 19 edges reported vs. 16
  true), but with very low precision (0.158): only 3 of its 19 reported X→X edges
  are correct. F1 = 0.171. The Fisher-z test at $\alpha = 0.05$ struggles with
  the non-linear, non-Gaussian residuals that persist even after log-transform.

- Both algorithms perform **substantially worse** on this real dataset than on
  the synthetic linear no-conf dataset (best F1 = 0.289 vs. 0.286 here), despite
  the much larger sample size (5,972 vs. 800). This confirms that real biological
  data violates the algorithmic assumptions more severely than the synthetic
  confounded case.

- Note: PC removed 2 edges during post-processing to break cycles (`jnk→mek`
  and `pip3→plc`), indicating the PC output was not a valid DAG before cycle
  removal.

---

## 7. Shapley Computation Settings

| Parameter | Value |
|---|---|
| Background rows (30 % of train) | **1,791** |
| Test instances (50 % of test, capped at 100) | **100** |
| Monte Carlo permutations (`N_SHAPLEY_SAMPLES`) | 100 |
| Inner samples (`M_INNER_SAMPLES_CAUSAL`) | 10 |
| Methods computed | 7: Scratch, Asymmetric (PC), Causal (PC), Flow (PC), Asymmetric (LiNGAM), Causal (LiNGAM), Flow (LiNGAM) |

No True-DAG variants are computed for the real dataset (no ground-truth
adjacency file on disk). The Sachs consensus graph is used only for evaluation
of discovery quality, not as a Shapley input.

---

## Summary

The Sachs dataset is the **real-data benchmark** for the thesis: 7,466 simultaneous
measurements of 10 signalling proteins, with Akt kinase as the regression target.
After log1p + StandardScaler preprocessing, LightGBM achieves R² = 0.68 — a
realistic level for complex biological signal. The true DAG (16 X→X edges) is
extremely sparse compared to the synthetic datasets, yet both PC and LiNGAM
struggle (best F1 = 0.286 for LiNGAM), confirming that real biological noise,
multi-pathway interference, and intervention heterogeneity make causal discovery
qualitatively harder than the synthetic confounded setting. This dataset tests
whether causal Shapley methods add value over Scratch even when the discovered
graph is heavily imprecise.
