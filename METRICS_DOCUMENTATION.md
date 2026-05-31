# Evaluation Metric Definitions

This document formalises all evaluation metrics used in the thesis to compare causal Shapley methods (Asymmetric, Causal, ShapleyFlow) across causal graph variants (PC, LiNGAM, True DAG) and the graph-free Scratch baseline.

Metrics are implemented in `analysis_utils.py` and computed in `notebooks/shapley_summary.ipynb`.

---

## Notation

| Symbol | Description |
|--------|-------------|
| $m$ | Shapley method $\in$ {Asymmetric, Causal, Flow} |
| $\mathcal{G}$ | Causal graph used by the method $\in$ {PC, LiNGAM, True, Scratch} |
| $\phi^{(m,\mathcal{G})}_{i,f}$ | SHAP attribution for instance $i$, feature $f$, method $m$, graph $\mathcal{G}$ |
| $N$ | Number of test instances |
| $F$ | Number of features |
| $K$ | Cardinality of top-feature set |
| $\text{Pa}(Y)$ | Set of true causal parents of the target variable $Y$ |

---

## 1. Feature Importance Baseline

### Mean Absolute SHAP

The standard feature importance score for a method/graph combination, averaged over all test instances.

$$\bar{\phi}^{(m,\mathcal{G})}_f = \frac{1}{N} \sum_{i=1}^{N} \left| \phi^{(m,\mathcal{G})}_{i,f} \right|$$

- **Range:** $[0, +\infty)$
- **Interpretation:** Higher = feature $f$ is more important on average under this method/graph.
- **Used in:** Spearman ρ, Jaccard, Precision@K, GSS, TGA.

---

## 2. Scratch-Baseline Comparison

These metrics quantify how much each causal method diverges from **Scratch** — the graph-free SHAP baseline that uses an empty (fully-connected) causal graph. A score close to the Scratch value indicates that incorporating the causal graph had little effect.

### 2.1 Spearman Rank Correlation (ρ)

Measures whether the feature importance *ranking* of a causal method agrees with the Scratch ranking.

$$\rho\!\left(m, \mathcal{G}\right) = \text{Spearman}\!\left(\bar{\phi}^{(m,\mathcal{G})}, \bar{\phi}^{(\text{Scratch})}\right)$$

- **Range:** $[-1, 1]$; higher = more similar ranking to Scratch.
- **Interpretation:** $\rho = 1$ means the causal graph did not change the feature importance order.

---

### 2.2 Top-$K$ Jaccard Overlap

Measures whether the *set* of the $K$ most important features is the same as Scratch's top-$K$ set.

$$\text{Jaccard}_K\!\left(m, \mathcal{G}\right) = \frac{\left| \mathcal{T}_K^{(m,\mathcal{G})} \cap \mathcal{T}_K^{(\text{Scratch})} \right|}{\left| \mathcal{T}_K^{(m,\mathcal{G})} \cup \mathcal{T}_K^{(\text{Scratch})} \right|}$$

where $\mathcal{T}_K^{(m,\mathcal{G})} = \operatorname*{arg\,top}_K \bar{\phi}^{(m,\mathcal{G})}$ is the index set of the $K$ features with highest mean absolute SHAP.

- **Range:** $[0, 1]$; higher = more overlap with Scratch top-$K$.
- **Evaluated at:** $K \in \{5, 10, 20\}$.

---

### 2.3 True-Parent Precision@$K$

Measures whether the top-$K$ most important features are true causal parents of $Y$.

$$\text{Prec@}K\!\left(m, \mathcal{G}\right) = \frac{\left| \mathcal{T}_K^{(m,\mathcal{G})} \cap \text{Pa}(Y) \right|}{K}$$

- **Range:** $[0, 1]$; higher = top-$K$ features are more enriched for true parents.
- **Baseline:** Random precision = $|\text{Pa}(Y)| / F$.
- **Evaluated at:** $K \in \{5, 10, 20\}$.

---

## 3. Graph Stability (PC vs LiNGAM)

These metrics measure the *sensitivity* of a method's attributions to the choice of causal discovery algorithm (PC vs LiNGAM) on the same dataset. Lower instability = more robust to graph uncertainty.

### 3.1 Graph Sensitivity Score (GSS)

Per-feature signed magnitude difference between PC and LiNGAM outputs, averaged over instances then over features.

$$\text{GSS}^{(m)}_f = \frac{1}{N} \sum_{i=1}^{N} \left( \left|\phi^{(m,\text{PC})}_{i,f}\right| - \left|\phi^{(m,\text{LiNGAM})}_{i,f}\right| \right)$$

The sign encodes which graph assigns higher importance for that feature:

- **Positive** → PC assigns higher mean absolute importance than LiNGAM for feature $f$.
- **Negative** → LiNGAM assigns higher mean absolute importance than PC for feature $f$.
- **Near zero** → both graphs agree on magnitude for that feature.

**Aggregate over features:**

$$\overline{\text{GSS}}^{(m)} = \frac{1}{F} \sum_{f=1}^{F} \text{GSS}^{(m)}_f$$

- **Range:** $(-\infty, +\infty)$; **near 0 = both graphs agree on average**.
- $\overline{\text{GSS}} > 0$ means PC systematically assigns higher feature importance than LiNGAM for this method.
- $\overline{\text{GSS}} < 0$ means LiNGAM systematically assigns higher feature importance than PC.
- The magnitude of $\text{GSS}^{(m)}_f$ (in SHAP units) reflects how large the disagreement is for that feature.

---

### 3.2 Sign Stability Score (SSS)

Per-feature sign agreement rate between PC and LiNGAM attributions for the same method.

$$\text{SSS}^{(m)}_f = \frac{\displaystyle\sum_{i=1}^{N} \mathbf{1}\!\left[\operatorname{sign}\!\left(\phi^{(m,\text{PC})}_{i,f}\right) = \operatorname{sign}\!\left(\phi^{(m,\text{LiNGAM})}_{i,f}\right)\right] \cdot v_{i,f}}{\displaystyle\sum_{i=1}^{N} v_{i,f}}$$

where $v_{i,f} = \mathbf{1}\!\left[\phi^{(m,\text{PC})}_{i,f} \neq 0 \;\wedge\; \phi^{(m,\text{LiNGAM})}_{i,f} \neq 0\right]$ excludes zero attributions.

**Aggregated:**

$$\overline{\text{SSS}}^{(m)} = \frac{1}{F} \sum_{f=1}^{F} \text{SSS}^{(m)}_f$$

- **Range:** $[0, 1]$; **higher = more stable** (signs agree more often).
- **Complementary to GSS:** GSS captures the signed directional magnitude bias (PC vs LiNGAM); SSS captures sign (direction) instability.
- **Cross-evaluation:** GSS vs $(1 - \overline{\text{SSS}})$ places each method in a 2-D space: x shows directional magnitude bias (positive = PC higher, negative = LiNGAM higher), y shows sign flip rate between the two graphs.

---

## 4. True-Graph Alignment

Measures how closely a method using a *discovered* graph (PC, LiNGAM) matches the oracle method using the **True DAG**. Higher alignment = the discovered graph led to attributions consistent with the ground-truth causal structure.

### 4.1 Sign Alignment vs Reference

Per-feature sign agreement rate between a (method, graph) pair and a reference attribution.

$$\text{SA}^{(m,\mathcal{G})}_f\!\left(\text{ref}\right) = \frac{\displaystyle\sum_{i : \phi^{(\text{ref})}_{i,f} \neq 0} \mathbf{1}\!\left[\operatorname{sign}\!\left(\phi^{(m,\mathcal{G})}_{i,f}\right) = \operatorname{sign}\!\left(\phi^{(\text{ref})}_{i,f}\right)\right]}{\displaystyle\left|\left\{i : \phi^{(\text{ref})}_{i,f} \neq 0\right\}\right|}$$

**Aggregated:**

$$\overline{\text{SA}}^{(m,\mathcal{G})}\!\left(\text{ref}\right) = \frac{1}{F} \sum_{f=1}^{F} \text{SA}^{(m,\mathcal{G})}_f\!\left(\text{ref}\right)$$

- **Range:** $[0, 1]$; **higher = more aligned with reference**.
- **Instances where $\phi^{(\text{ref})}_{i,f} = 0$ are excluded** to avoid division by zero and to focus on features where the reference provides a non-trivial signal.
- **Reference options:**
  - `"True"` → compares against $\phi^{(m, \text{True})}$ (per-method oracle; used in Section 5).
  - `"Scratch"` → compares against the single $\phi^{(\text{Scratch})}$ array for all methods (used in Section 6).

---

### 4.2 Magnitude Total Graph Alignment (TGA)

Per-feature signed magnitude difference between discovered-graph and reference attributions, averaged over instances.

$$\text{TGA}^{(m,\mathcal{G})}_f\!\left(\text{ref}\right) = \frac{1}{N}\displaystyle\sum_{i=1}^{N} \left( \left|\phi^{(m,\mathcal{G})}_{i,f}\right| - \left|\phi^{(\text{ref})}_{i,f}\right| \right)$$

No outer absolute and no normalisation. The sign encodes which attribution is larger:

- **Positive** → discovered graph assigns higher mean importance than the reference for feature $f$.
- **Negative** → discovered graph assigns lower mean importance than the reference for feature $f$.
- **Near zero** → discovered graph and reference agree in magnitude for feature $f$.

**Aggregated:**

$$\overline{\text{TGA}}^{(m,\mathcal{G})}\!\left(\text{ref}\right) = \frac{1}{F} \sum_{f=1}^{F} \text{TGA}^{(m,\mathcal{G})}_f\!\left(\text{ref}\right)$$

- **Range:** $(-\infty, +\infty)$; **near 0 = magnitudes match the reference on average**.
- **Reference options:** same as Sign Alignment (`"True"` or `"Scratch"`).

---

## 5. Multi-Method Sign Agreement

Measures how consistently all methods agree on the *direction* of attribution for each (instance, feature), independently of which graph is used.

$$\text{Agr}_{i,f} = \frac{\max\!\left(n^+_{i,f},\; n^-_{i,f}\right)}{M}$$

where $n^+_{i,f} = \sum_m \mathbf{1}[\phi^{(m)}_{i,f} > 0]$, $n^-_{i,f} = \sum_m \mathbf{1}[\phi^{(m)}_{i,f} < 0]$, and $M$ is the number of methods compared.

- **Range:** $[0.5, 1.0]$ (by construction, the majority sign always holds at least half the votes).
- **Interpretation:** $\text{Agr}_{i,f} = 1$ means all methods agree on sign; $= 0.5$ means a perfect split.

---

## 6. Causal Graph Quality (Edge Recovery)

Evaluates how well the discovered graph (PC, LiNGAM) recovers the edges of the True DAG, using *direction-aware* precision, recall, and F1.

$$\text{Precision} = \frac{|\{(u,v) : \hat{A}_{uv}=1 \wedge A_{uv}=1\}|}{|\{(u,v) : \hat{A}_{uv}=1\}|}$$

$$\text{Recall} = \frac{|\{(u,v) : \hat{A}_{uv}=1 \wedge A_{uv}=1\}|}{|\{(u,v) : A_{uv}=1\}|}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

where $A$ is the true adjacency matrix and $\hat{A}$ is the predicted adjacency matrix. An edge $(u \to v)$ is counted as correct only if both the edge and its direction match.

---

## Summary Table

| Metric | Symbol | Range | Better | Reference | Section |
|--------|--------|-------|--------|-----------|---------|
| Mean Absolute SHAP | $\bar{\phi}_f$ | $[0,\infty)$ | — | — | Baseline |
| Spearman ρ | $\rho$ | $[-1,1]$ | Higher | Scratch | 2.1 |
| Top-$K$ Jaccard | $\text{Jaccard}_K$ | $[0,1]$ | Higher | Scratch | 2.2 |
| Precision@$K$ | $\text{Prec@}K$ | $[0,1]$ | Higher | True parents | 2.3 |
| Graph Sensitivity Score | $\overline{\text{GSS}}$ | $(-\infty,+\infty)$ | **Near 0** | PC vs LiNGAM | 3.1 |
| Sign Stability Score | $\overline{\text{SSS}}$ | $[0,1]$ | Higher | PC vs LiNGAM | 3.2 |
| Sign Alignment | $\overline{\text{SA}}$ | $[0,1]$ | Higher | True or Scratch | 4.1 |
| Magnitude TGA | $\overline{\text{TGA}}$ | $(-\infty,+\infty)$ | **Near 0** | True or Scratch | 4.2 |
| Multi-method Agreement | $\text{Agr}_{i,f}$ | $[0.5,1]$ | Higher | — | 5 |
| Edge F1 | $F_1$ | $[0,1]$ | Higher | True DAG | 6 |

---

## Implementation Notes

All metrics are implemented as pure functions in `analysis_utils.py`:

| Function | Metric |
|----------|--------|
| `mean_abs_shap(shap_data, key)` | Mean Absolute SHAP (§1) |
| `top_k_jaccard(arr1, arr2, k)` | Top-$K$ Jaccard (§2.2) |
| `compute_sign_agreement(method_list, shap_data, feat_idx)` | Multi-method sign agreement (§5) |
| `compute_sign_alignment(shap_data, base_methods, disc_graphs, reference="True")` | Sign Alignment (§4.1) |
| `compute_tga(shap_data, feature_names, base_methods, disc_graphs, reference="True")` | Magnitude TGA (§4.2) |
| `compute_sss(shap_data, base_methods)` | Sign Stability Score (§3.2) |
| `edge_recovery(true_adj, pred_adj)` | Edge precision / recall / F1 (§6) |

`compute_sign_alignment` and `compute_tga` accept a `reference` keyword argument (default `"True"`). Passing `reference="Scratch"` compares each method/graph pair against the graph-free Scratch baseline using the identical formula.
