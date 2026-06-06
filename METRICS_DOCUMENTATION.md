# Evaluation Metric Definitions

This document formalises all evaluation metrics used in the thesis to compare structure-aware Shapley methods (Asymmetric, Causal, ShapleyFlow) across causal graph variants (PC, LiNGAM, True/consensus DAG) and the graph-free Traditional (Scratch) baseline.

The two primary metrics — **Magnitude Divergence** and **Sign Disagreement** — are implemented in `analysis_utils.py` and `assessment_extras.py` and computed in `notebooks/shapley_summary.ipynb`. The scalar values they produce are tabulated per dataset in `notebooks/plots_claude/<dataset>/method_level_metrics_<dataset>.md`.

> **Naming.** Earlier drafts used four acronyms for what are really two metrics under different references. This document uses the standardised scheme: a single magnitude metric (ΔM) and a single sign metric (D), each carried under a reference subscript. The legacy mapping is: TGA → ΔM, GSS → ΔM_disc, (1 − SA) → D, (1 − SSS) → D_disc.

---

## Metric Framework

Every comparison asks the same two questions of a `(method, graph)` result against a reference: did the attribution change in **magnitude**, and did it change in **direction (sign)**? This gives two dimensions, each measured by exactly one metric, reported under three references and at three aggregation levels.

**Two metrics (one per dimension):**

| Dimension | Metric | Symbol | Range | Units |
|-----------|--------|--------|-------|-------|
| Magnitude | Magnitude Divergence | ΔM | $[0,\infty)$ | % of model-output std $\hat\sigma$ |
| Sign / direction | Sign Disagreement | D | $[0,1]$ | fraction of attributions |

**Three reference comparisons (the subscript):**

| Subscript | Reference | Question | Thesis section |
|-----------|-----------|----------|----------------|
| `base` | Traditional Shapley (graph-free) | How far does a graph move attributions from the graph-free baseline? | 4.2 |
| `oracle` | True DAG (synthetic) / consensus DAG (Sachs) | How faithfully does a discovered graph recover the oracle attributions? | 4.3 |
| `disc` | the opposite discovered graph (PC ↔ LiNGAM) | How much does the choice of discovery algorithm alone perturb attributions? | 4.4 |

In all three, the **subject** is a discovered-graph result $(m,\mathcal{G})$ with $\mathcal{G}\in\{\text{PC},\text{LiNGAM}\}$. For `disc`, the subject is fixed to PC and the reference to LiNGAM.

**Three measurement levels.** Both metrics are defined once at the atomic instance level, then lifted by fixed aggregation operators. Convention: tables report the **global** level, heatmaps the **feature** level, violin/scatter plots the **instance** level.

---

## Notation

| Symbol | Description |
|--------|-------------|
| $m$ | Shapley method $\in$ {Asymmetric, Causal, Flow} |
| $\mathcal{G}$ | Causal graph used by the method $\in$ {PC, LiNGAM, True, Scratch} |
| $\text{ref}$ | Reference configuration $\in$ {base (Traditional/Scratch), oracle (True/consensus), disc (opposite graph)} |
| $\phi^{(m,\mathcal{G})}_{i,f}$ | SHAP attribution for instance $i$, feature $f$, method $m$, graph $\mathcal{G}$ |
| $N$ | Number of test instances |
| $F$ | Number of features |
| $\hat\sigma$ | Std of the model prediction $f(x)$ over the test set (the scale magnitudes are normalised by) |
| $K$ | Cardinality of top-feature set |
| $\text{Pa}(Y)$ | Set of true causal parents of the target variable $Y$ |

---

## 1. Magnitude Divergence (ΔM)

How much the absolute attribution moves relative to the reference, expressed as a fraction of $\hat\sigma$. Non-negative.

**Instance level (signed).** The atomic quantity keeps its sign so that the *direction* of a magnitude change can be recovered when needed (positive = the discovered graph inflates the feature's local importance vs the reference; negative = it suppresses it):

$$\delta^{(m,\mathcal{G},\text{ref})}_{i,f} = \left|\phi^{(m,\mathcal{G})}_{i,f}\right| - \left|\phi^{(\text{ref})}_{i,f}\right|$$

**Feature level (non-negative, normalised).** Root-mean-square over instances, divided by $\hat\sigma$:

$$\Delta M^{(m,\mathcal{G},\text{ref})}_f = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(\delta^{(m,\mathcal{G},\text{ref})}_{i,f}\right)^2}}{\hat\sigma}$$

The RMS (rather than a plain mean) measures the typical *size* of the per-instance change without letting positive and negative gaps cancel; dividing by $\hat\sigma$ puts every feature and dataset on one interpretable scale.

> **Why $\hat\sigma$ is the natural scale.** By the efficiency axiom, every Shapley method here satisfies $\sum_f \phi^{(m,\mathcal{G})}_{i,f} = f(x_i) - \mathbb{E}[f(x)]$ — the attributions sum to the centred prediction. The whole SHAP space therefore lives on the model-output scale, and $\hat\sigma = \operatorname{std}_i f(x_i)$ is the total dispersion it distributes. Normalising by $\hat\sigma$ expresses a magnitude change as a fraction of the attribution budget that actually exists, so the metric is comparable across models and across systems of different nature (units, feature counts, and prediction magnitudes all cancel).

**Global level.** Mean over features:

$$\Delta M^{(m,\mathcal{G},\text{ref})} = \frac{1}{F}\sum_{f=1}^{F}\Delta M^{(m,\mathcal{G},\text{ref})}_f$$

- **Range:** $[0,\infty)$; **near 0 = magnitudes match the reference**.
- **Interpretation:** $\Delta M = 0.10$ means the typical attribution shifted by 10% of $\hat\sigma$.
- **References:** `base` → ref $=$ Traditional/Scratch; `oracle` → ref $=$ True/consensus DAG; `disc` → subject PC, ref LiNGAM (denoted $\Delta M_{\text{disc}}$).

---

## 2. Sign Disagreement (D)

The fraction of `(instance, feature)` attributions that point in the opposite direction to the reference.

**Instance level.** The atomic disagreement indicator, defined only where the reference attribution is non-zero:

$$d^{(m,\mathcal{G},\text{ref})}_{i,f} = \mathbf{1}\!\left[\operatorname{sign}\!\left(\phi^{(m,\mathcal{G})}_{i,f}\right) \neq \operatorname{sign}\!\left(\phi^{(\text{ref})}_{i,f}\right)\right], \quad i : \phi^{(\text{ref})}_{i,f}\neq 0$$

**Feature level.** Disagreement rate over valid instances:

$$D^{(m,\mathcal{G},\text{ref})}_f = \frac{\sum_{i:\phi^{(\text{ref})}_{i,f}\neq 0} d^{(m,\mathcal{G},\text{ref})}_{i,f}}{\left|\{i : \phi^{(\text{ref})}_{i,f}\neq 0\}\right|}$$

**Global level.** Mean over features:

$$D^{(m,\mathcal{G},\text{ref})} = \frac{1}{F}\sum_{f=1}^{F} D^{(m,\mathcal{G},\text{ref})}_f$$

- **Range:** $[0,1]$; **0 = perfect directional agreement** with the reference.
- **Instances where $\phi^{(\text{ref})}_{i,f}=0$ are excluded** to avoid an undefined reference sign.
- **References:** same three as ΔM. $D_{\text{disc}}$ uses the opposite discovered graph.
- **Relation to the code's agreement scores:** `analysis_utils.py` returns sign *agreement* rates (SA, SSS). Disagreement is their complement: $D = 1 - \text{SA}$ and $D_{\text{disc}} = 1 - \text{SSS}$.

---

## 3. Auxiliary Feature-Importance Metrics

These rank- and set-based metrics complement the two primary metrics and are reported in the supporting analyses. They operate on the mean absolute attribution $\bar{\phi}^{(m,\mathcal{G})}_f = \frac{1}{N}\sum_i |\phi^{(m,\mathcal{G})}_{i,f}|$.

### 3.1 Spearman Rank Correlation (ρ)

Whether the feature-importance *ranking* of a configuration agrees with a reference (Scratch) ranking.

$$\rho\!\left(m,\mathcal{G}\right) = \text{Spearman}\!\left(\bar{\phi}^{(m,\mathcal{G})}, \bar{\phi}^{(\text{Scratch})}\right)$$

- **Range:** $[-1,1]$; $\rho=1$ means the graph did not change the importance order.

### 3.2 Top-$K$ Jaccard Overlap

Whether the set of the $K$ most important features matches the reference top-$K$ set.

$$\text{Jaccard}_K\!\left(m,\mathcal{G}\right) = \frac{\left|\mathcal{T}_K^{(m,\mathcal{G})}\cap\mathcal{T}_K^{(\text{Scratch})}\right|}{\left|\mathcal{T}_K^{(m,\mathcal{G})}\cup\mathcal{T}_K^{(\text{Scratch})}\right|}$$

where $\mathcal{T}_K^{(m,\mathcal{G})}=\operatorname*{arg\,top}_K \bar{\phi}^{(m,\mathcal{G})}$. **Range:** $[0,1]$. Evaluated at $K\in\{5,10,20\}$.

### 3.3 True-Parent Precision@$K$

Whether the top-$K$ features are true causal parents of $Y$.

$$\text{Prec@}K\!\left(m,\mathcal{G}\right) = \frac{\left|\mathcal{T}_K^{(m,\mathcal{G})}\cap\text{Pa}(Y)\right|}{K}$$

- **Range:** $[0,1]$; random baseline $=|\text{Pa}(Y)|/F$. Evaluated at $K\in\{5,10,20\}$.

### 3.4 Multi-Method Sign Agreement

How consistently all methods agree on the direction of an attribution for each `(instance, feature)`, independent of graph.

$$\text{Agr}_{i,f} = \frac{\max\!\left(n^+_{i,f}, n^-_{i,f}\right)}{M}$$

where $n^+_{i,f}=\sum_m\mathbf{1}[\phi^{(m)}_{i,f}>0]$, $n^-_{i,f}=\sum_m\mathbf{1}[\phi^{(m)}_{i,f}<0]$, $M$ = number of methods. **Range:** $[0.5,1]$; $1$ = all methods agree on sign.

---

## 4. Causal Graph Quality (Edge Recovery)

How well the discovered graph (PC, LiNGAM) recovers the True-DAG edges, using *direction-aware* precision, recall, and F1.

$$\text{Precision} = \frac{|\{(u,v) : \hat{A}_{uv}=1 \wedge A_{uv}=1\}|}{|\{(u,v) : \hat{A}_{uv}=1\}|}, \quad \text{Recall} = \frac{|\{(u,v) : \hat{A}_{uv}=1 \wedge A_{uv}=1\}|}{|\{(u,v) : A_{uv}=1\}|}$$

$$F_1 = \frac{2\cdot\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$

$A$ is the true adjacency matrix, $\hat{A}$ the predicted one. An edge $(u\to v)$ counts as correct only if both the edge and its direction match.

---

## Summary Table

| Metric | Symbol | Levels | Range | Better | Reference | Section |
|--------|--------|--------|-------|--------|-----------|---------|
| Magnitude Divergence | $\Delta M_{\text{base/oracle/disc}}$ | instance / feature / global | $[0,\infty)$ (% of $\hat\sigma$) | **Near 0** | base, oracle, or disc | 1 |
| Sign Disagreement | $D_{\text{base/oracle/disc}}$ | instance / feature / global | $[0,1]$ | **Near 0** | base, oracle, or disc | 2 |
| Mean Absolute SHAP | $\bar{\phi}_f$ | feature | $[0,\infty)$ | — | — | 3 |
| Spearman ρ | $\rho$ | global | $[-1,1]$ | Higher | Scratch | 3.1 |
| Top-$K$ Jaccard | $\text{Jaccard}_K$ | global | $[0,1]$ | Higher | Scratch | 3.2 |
| Precision@$K$ | $\text{Prec@}K$ | global | $[0,1]$ | Higher | True parents | 3.3 |
| Multi-method Agreement | $\text{Agr}_{i,f}$ | instance | $[0.5,1]$ | Higher | — | 3.4 |
| Edge F1 | $F_1$ | global | $[0,1]$ | Higher | True DAG | 4 |

---

## Implementation Notes

The two primary metrics are pure functions in `analysis_utils.py`; the distribution-preserving / structure-conditioned views live in `assessment_extras.py`. Both divide the RMS by `output_std` ($\hat\sigma$, the std of model predictions on the test set), so feature- and global-level magnitudes are returned as fractions of $\hat\sigma$.

| Function | Metric | Notes |
|----------|--------|-------|
| `compute_tga(shap_data, feature_names, base_methods, disc_graphs, reference="True", output_std=...)` | Magnitude Divergence ΔM (§1) | `reference="True"` → ΔM_oracle; `reference="Scratch"` → ΔM_base. RMS over instances ÷ `output_std`. |
| `compute_gss(shap_data, base_methods, output_std=...)` | Cross-discovery ΔM_disc (§1) | PC vs LiNGAM, same RMS ÷ `output_std`. |
| `compute_sign_alignment(shap_data, base_methods, disc_graphs, reference="True")` | Sign agreement; $D = 1 - \text{SA}$ (§2) | `reference` in {"True", "Scratch"}. |
| `compute_sss(shap_data, base_methods)` | Sign agreement PC vs LiNGAM; $D_{\text{disc}} = 1 - \text{SSS}$ (§2) | — |
| `mean_abs_shap(shap_data, key)` | Mean Absolute SHAP (§3) | — |
| `top_k_jaccard(arr1, arr2, k)` | Top-$K$ Jaccard (§3.2) | — |
| `compute_sign_agreement(method_list, shap_data, feat_idx)` | Multi-method sign agreement (§3.4) | — |
| `edge_recovery(true_adj, pred_adj)` | Edge precision / recall / F1 (§4) | — |

`compute_tga` and `compute_sign_alignment` accept a `reference` keyword (default `"True"`). Passing `reference="Scratch"` compares each `(method, graph)` pair against the graph-free Traditional baseline using the identical formula — i.e. the only difference between ΔM_oracle and ΔM_base (and between $D_{\text{oracle}}$ and $D_{\text{base}}$) is the reference array. Pass `output_std` to obtain the normalised (% of $\hat\sigma$) values reported in the thesis; omit it to get raw attribution units.
