"""
shapley_plots.py
================

Centralized, ``ctx``-driven plotting for the structure-aware Shapley evaluation.

Consolidates the figures that were previously scattered inline in
``notebooks/shapley_results_analysis.ipynb`` / ``shapley_summary.ipynb`` (and the
real-data twins) into one module. **All figures are matplotlib** and share the single
style/palette defined in ``plot_style.py``, so every plot — here and in
``assessment_extras.py`` — looks the same.

Metric definitions are inherited from ``assessment_extras`` (refined):
    tga_instance(method, graph, ref) = |phi_disc| - |phi_ref|           (signed, per instance)
    tga_feature                       = sqrt( mean_i tga_instance² )     (RMS over instances)
                                        / output_std                      (÷ model pred std)
    global TGA                        = mean_f( tga_feature )
    gss_instance(method)              = |phi_PC| - |phi_LiNGAM|          (signed, per instance)
    gss_feature                       = sqrt( mean_i gss_instance² )     (RMS over instances)
                                        / output_std                      (÷ model pred std)
    global GSS                        = mean_f( gss_feature )

    RMS aggregation handles sparsity better than mean(|diff|): large deviations in a
    minority of instances are not washed out by near-zero ones.  Dividing by
    ``output_std`` (std of model predictions on test set) makes both metrics
    dimensionless and directly comparable across datasets with different target scales.

Scope: subject = discovered graph (PC/LiNGAM); references = True / Traditional(Scratch) /
opposite graph.

Figures
-------
Feature level   : plot_gss_delta_box, plot_gss_heatmap, plot_sss_heatmap
Disc vs ref     : plot_sign_alignment_heatmap(reference=...), plot_tga_heatmap(reference=...)
Structure       : plot_adjacency_comparison
Method level    : plot_gss_sss_scatter, plot_tga_sa_scatter(reference=...)

    import assessment_extras as ax
    import shapley_plots as sp
    ctx = ax.load_context("linear_conf_f50_s1000_p30")
    sp.run_all(ctx)            # all figures -> notebooks/plots_claude/<dataset>/
    python shapley_plots.py    # run_all for the focus datasets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import assessment_extras as ax
from assessment_extras import (
    CAUSAL_DIR, DEFAULT_PLOTS_DIR, DEFAULT_DATASETS,
    METHODS, DISC_GRAPHS, SCRATCH,
    load_context, tga_feature, gss_instance, gss_feature,
)
import plot_style as st
from plot_style import (
    METHOD_COLORS, METHOD_ORDER, REFERENCE_COLORS, GRAPH_MARKERS,
    SEQ_CMAP, DISAGREE_CMAP, DIV_CMAP, savefig,
)

ORDER = METHOD_ORDER                     # ["Asymmetric", "Causal", "Flow"]
REF_TOKEN = {"True": "True", "Traditional": SCRATCH, "Scratch": SCRATCH}

# reference token -> (metric subscript, display name) for the standardised ΔM / D naming.
#   base   = Traditional Shapley (graph-free)   oracle = True / consensus DAG   disc = opposite discovered graph
REF_DISPLAY = {"True": ("oracle", "True DAG"),
               "Traditional": ("base", "Traditional"),
               "Scratch": ("base", "Traditional")}


# ============================================================================= #
# metric dictionaries (ctx -> per-feature / per-instance arrays)
# ============================================================================= #
def _methods_present(ctx, graphs):
    return [m for m in ORDER if all(f"{m} ({g})" in ctx["shap"] for g in graphs)]


def gss_delta_dict(ctx):
    """method -> (N, F) signed |phi_PC| - |phi_LiNGAM|."""
    return {m: gss_instance(ctx, m) for m in _methods_present(ctx, ["PC", "LiNGAM"])}


def gss_feature_dict(ctx):
    """method -> (F,) per-feature GSS.

    Each value is the RMS of the per-instance signed magnitude difference
    (|φ_PC| − |φ_LiNGAM|), divided by the model prediction std:

        GSS(m, f) = sqrt(mean_i(|φ^PC_{i,f}| − |φ^LiNGAM_{i,f}|)²) / output_std
    """
    return {m: gss_feature(ctx, m) for m in _methods_present(ctx, ["PC", "LiNGAM"])}


def sss_feature_dict(ctx):
    """method -> (F,) Sign Stability Score (PC vs LiNGAM), matching compute_sss."""
    out = {}
    for m in _methods_present(ctx, ["PC", "LiNGAM"]):
        pc, lg = ctx["shap"][f"{m} (PC)"], ctx["shap"][f"{m} (LiNGAM)"]
        valid = (pc != 0) & (lg != 0)
        agree = (np.sign(pc) == np.sign(lg)).astype(float)
        agree[~valid] = np.nan
        with np.errstate(all="ignore"):
            out[m] = np.nanmean(agree, axis=0)
    return out


def sign_alignment_dict(ctx, reference="True"):
    """(method, graph) -> (F,) sign alignment vs reference, matching compute_sign_alignment."""
    ref = REF_TOKEN[reference]
    out = {}
    for m in ORDER:
        rk = ax._ref_key(ctx["shap"], m, ref)
        if rk is None:
            continue
        phi_ref = ctx["shap"][rk]
        for g in DISC_GRAPHS:
            dk = f"{m} ({g})"
            if dk not in ctx["shap"]:
                continue
            agree = (np.sign(ctx["shap"][dk]) == np.sign(phi_ref)).astype(float)
            agree[phi_ref == 0] = np.nan
            with np.errstate(all="ignore"):
                out[(m, g)] = np.nanmean(agree, axis=0)
    return out


def tga_feature_dict(ctx, reference="True"):
    """(method, graph) -> (F,) per-feature TGA vs reference.

    Each value is the RMS of the per-instance signed magnitude difference
    (|φ_disc| − |φ_ref|), divided by the model prediction std:

        TGA(m, g, f) = sqrt(mean_i(|φ^disc_{i,f}| − |φ^ref_{i,f}|)²) / output_std
    """
    ref = REF_TOKEN[reference]
    out = {}
    for m in ORDER:
        for g in DISC_GRAPHS:
            t = tga_feature(ctx, m, g, ref)
            if t is not None:
                out[(m, g)] = t
    return out


# ============================================================================= #
# feature names + adjacency
# ============================================================================= #
def feature_names(ctx):
    p = CAUSAL_DIR / f"{ctx['dataset']}_pc_results.json"
    if p.exists():
        names = json.load(open(p))["feature_names"]
        return [n for n in names if n != "Y"]
    return [f"X{i}" for i in range(ctx["F"])]


def load_adjacency(ctx):
    ds, F = ctx["dataset"], ctx["F"]
    try:
        true_full = np.load(CAUSAL_DIR / f"{ds}_true_full_adjacency.npy")
        pc = np.load(CAUSAL_DIR / f"{ds}_pc_train_adjacency.npy")
        lg = np.load(CAUSAL_DIR / f"{ds}_lingam_train_adjacency.npy")
    except FileNotFoundError:
        return None
    names = feature_names(ctx)
    return ((true_full[:F, :F] != 0).astype(int), (pc[:F, :F] != 0).astype(int),
            (lg[:F, :F] != 0).astype(int), names[:F])


# ============================================================================= #
# shared heatmap helper
# ============================================================================= #
def _heatmap(ax_, Z, row_labels, col_labels, cmap, vmin, vmax, cbar_label,
             annotate=False, cbar_ticks=None, cbar_ticklabels=None):
    im = ax_.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_.set_xticks(range(len(col_labels)))
    ax_.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    ax_.set_yticks(range(len(row_labels)))
    ax_.set_yticklabels(row_labels, fontsize=9)
    ax_.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax_.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax_.grid(which="minor", color="white", linewidth=0.6)
    ax_.grid(which="major", visible=False)
    ax_.tick_params(which="minor", length=0)
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                v = Z[i, j]
                if not np.isnan(v):
                    ax_.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=6,
                             color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.55 else "black")
    cbar = ax_.figure.colorbar(im, ax=ax_, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
        if cbar_ticklabels is not None:
            cbar.set_ticklabels(cbar_ticklabels)
    return im


# ============================================================================= #
# FEATURE LEVEL
# ============================================================================= #
def plot_gss_delta_box(ctx, n_top_features=15, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Per-instance magnitude delta |phi_PC| - |phi_LiNGAM| box-plots for the top-N features
    by GSS. Positive box = PC assigns more importance; tight = consistent, wide = sensitive."""
    feat_delta = gss_delta_dict(ctx)
    if not feat_delta:
        return None
    names = feature_names(ctx)
    methods = [m for m in ORDER if m in feat_delta]
    all_d = np.stack([feat_delta[m] for m in methods], axis=0)        # (M, N, F)
    rank = np.argsort(np.abs(all_d).mean(axis=(0, 1)))[::-1][:n_top_features]

    nM = len(methods)
    group_w = 0.8
    box_w = group_w / nM
    fig, ax_ = plt.subplots(figsize=(max(9, 0.75 * len(rank) + 2), 5.2))
    for j, m in enumerate(methods):
        positions = np.arange(len(rank)) + (j - (nM - 1) / 2) * box_w
        data = [feat_delta[m][:, fi] for fi in rank]
        bp = ax_.boxplot(data, positions=positions, widths=box_w * 0.9,
                         patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(METHOD_COLORS[m]); patch.set_alpha(0.75)
            patch.set_edgecolor("#333333"); patch.set_linewidth(0.6)
        for med in bp["medians"]:
            med.set_color("#222222"); med.set_linewidth(1.0)
        for w in bp["whiskers"] + bp["caps"]:
            w.set_color("#666666"); w.set_linewidth(0.7)
    ax_.axhline(0, color=st.C_BASELINE_LINE, lw=1.0, ls="--")
    ax_.set_xticks(range(len(rank)))
    ax_.set_xticklabels([names[i] for i in rank], rotation=40, ha="right")
    ax_.set_ylabel("|φ(PC)| − |φ(LiNGAM)|  (instance-level gap)")
    ax_.legend(handles=[Patch(facecolor=METHOD_COLORS[m], alpha=0.75, label=m) for m in methods],
               title="Method", loc="upper left", ncol=nM)
    st.style_title(ax_, f"Per-instance magnitude delta (PC − LiNGAM) — {ctx['dataset']}",
                   f"Top {len(rank)} features by ΔM_disc · positive = PC higher · tight box = consistent across instances")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_delta_box_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_gss_heatmap(ctx, top_n=10, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × method Graph Sensitivity Score heatmap (PC vs LiNGAM). Sequential (cividis):
    brighter = larger RMS magnitude difference between PC and LiNGAM, normalised by output range."""
    gss = gss_feature_dict(ctx)
    if not gss:
        return None
    names = feature_names(ctx)
    labels = [m for m in ORDER if m in gss]
    mat = np.stack([gss[m] for m in labels], axis=1) * 100           # (F, M) → %
    top = np.argsort(mat.mean(axis=1))[::-1][:min(top_n, len(names))]
    Z = mat[top].T                                                   # (M, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 1.0 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], SEQ_CMAP, 0, float(mat.max()),
             "ΔM_disc  (% of model-output std)", annotate=len(top) <= 14)
    st.style_title(ax_, f"Feature-level Magnitude Divergence — PC vs LiNGAM\n{ctx['dataset']}",
                   f"ΔM_disc = RMS_i(|φ(PC)|−|φ(LiNGAM)|) / model-output std · top {len(top)} features by mean ΔM_disc")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_heatmap_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_sss_heatmap(ctx, top_n=10, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × method Sign Disagreement heatmap (PC vs LiNGAM). Dark orange = signs flip
    across the two discovered graphs, white = stable. Sequential from 0 (OK) upward."""
    sss = sss_feature_dict(ctx)
    if not sss:
        return None
    names = feature_names(ctx)
    labels = [m for m in ORDER if m in sss]
    mat = np.stack([sss[m] for m in labels], axis=1) * 100           # (F, M) agreement → %
    dis_mat = 100 - mat                                              # (F, M) disagreement → %
    top = np.argsort(np.nanmean(dis_mat, axis=1))[::-1][:min(top_n, len(names))]
    Z = dis_mat[top].T                                               # (M, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 1.0 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], DISAGREE_CMAP, 0, float(dis_mat.max()),
             "D_disc  (Sign Disagreement, %)", annotate=len(top) <= 14)
    st.style_title(ax_, f"Feature-level Sign Disagreement — PC vs LiNGAM\n{ctx['dataset']}",
                   f"top {len(top)} most-unstable features · white = stable")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"sss_heatmap_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# DISCOVERED (PC, LiNGAM) vs REFERENCE (True / Traditional)
# ============================================================================= #
def plot_sign_alignment_heatmap(ctx, reference="True", top_n=10,
                                plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × (method, graph) sign-disagreement heatmap vs the reference. Dark orange =
    sign opposes reference, white = agrees. Sequential from 0 (OK) upward."""
    sa = sign_alignment_dict(ctx, reference)
    if not sa:
        return None
    names = feature_names(ctx)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in sa]
    labels = [f"{m} ({g})" for m, g in keys]
    mat = np.stack([sa[k] for k in keys], axis=1) * 100             # (F, K) agreement → %
    dis_mat = 100 - mat                                             # (F, K) disagreement → %
    top = np.argsort(np.nanmean(dis_mat, axis=1))[::-1][:min(top_n, len(names))]
    Z = dis_mat[top].T                                             # (K, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 0.55 * len(labels) + 1.8))
    sub, disp = REF_DISPLAY.get(reference, ("ref", reference))
    _heatmap(ax_, Z, labels, [names[i] for i in top], DISAGREE_CMAP, 0, float(dis_mat.max()),
             f"Sign Disagreement  (%, vs {disp})", annotate=len(top) <= 14)
    st.style_title(ax_, f"Sign Disagreement vs {disp}\n{ctx['dataset']}",
                   f"sorted by highest D_{sub} · white = agrees with reference")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"],
                  f"sign_alignment_heatmap_{reference.lower()}_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_tga_heatmap(ctx, reference="True", top_n=10, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × (method, graph) magnitude-TGA heatmap vs the reference. Sequential (cividis):
    brighter = larger RMS magnitude difference from the reference, normalised by output range."""
    tga = tga_feature_dict(ctx, reference)
    if not tga:
        return None
    names = feature_names(ctx)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in tga]
    labels = [f"{m} ({g})" for m, g in keys]
    mat = np.stack([tga[k] for k in keys], axis=1) * 100            # (F, K) → %
    top = np.argsort(mat.mean(axis=1))[::-1][:min(top_n, len(names))]
    Z = mat[top].T                                                 # (K, n_top)
    sub, disp = REF_DISPLAY.get(reference, ("ref", reference))
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 0.55 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], SEQ_CMAP, 0, float(mat.max()),
             f"ΔM_{sub}  (% of model-output std)", annotate=len(top) <= 14)
    st.style_title(ax_, f"Feature-level Magnitude Divergence vs {disp}\n{ctx['dataset']}",
                   f"ΔM_{sub} = RMS_i(|φ(disc)|−|φ({disp})|) / model-output std · top {len(top)} features by mean ΔM_{sub}")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"],
                  f"tga_heatmap_{reference.lower()}_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# DISCOVERED MATRICES (structure)
# ============================================================================= #
def plot_adjacency_comparison(ctx, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Side-by-side True / PC / LiNGAM X→X adjacency heatmaps. adj[i,j]=1 means edge i→j.
    The True panel is greyscale; PC and LiNGAM panels are colour-coded vs the True graph:
      green  = True Positive  (edge in both)
      red    = False Positive (edge in discovered, not in true)
      blue   = False Negative (edge in true, not in discovered)
      white  = True Negative  (no edge in either)
    """
    from matplotlib.colors import ListedColormap, BoundaryNorm

    adj = load_adjacency(ctx)
    if adj is None:
        return None
    true_xx, pc_xx, lg_xx, names = adj
    n = len(names)
    annotate = n <= 16

    # 0=TN (white), 1=TP (green), 2=FP (red), 3=FN (blue)
    _CMP_COLORS = ["#f5f5f5", "#2ca02c", "#d62728", "#1f77b4"]
    cmp_cmap = ListedColormap(_CMP_COLORS)
    cmp_norm = BoundaryNorm([0, 1, 2, 3, 4], cmp_cmap.N)

    def _diff_mat(disc):
        m = np.zeros_like(true_xx)
        m[(true_xx == 1) & (disc == 1)] = 1   # TP
        m[(true_xx == 0) & (disc == 1)] = 2   # FP
        m[(true_xx == 1) & (disc == 0)] = 3   # FN
        return m

    def _setup_ticks(axx):
        axx.set_xticks(range(n)); axx.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axx.set_yticks(range(n)); axx.set_yticklabels(names, fontsize=7)
        axx.set_xticks(np.arange(-.5, n, 1), minor=True)
        axx.set_yticks(np.arange(-.5, n, 1), minor=True)
        axx.grid(which="minor", color="#dddddd", linewidth=0.5)
        axx.tick_params(which="minor", length=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    # --- Panel 0: True graph (greyscale reference) ---
    axes[0].imshow(true_xx, cmap="Greys", vmin=0, vmax=1, aspect="auto")
    _setup_ticks(axes[0])
    axes[0].set_title(f"True X→X  ({int(true_xx.sum())} edges)", fontsize=11, fontweight="bold")
    if annotate:
        for i in range(n):
            for j in range(n):
                if true_xx[i, j]:
                    axes[0].text(j, i, "→", ha="center", va="center", fontsize=8, color="#f5f5f5")

    # --- Panels 1 & 2: discovered graphs colour-coded vs True ---
    for axx, disc, label in [(axes[1], pc_xx, "PC"), (axes[2], lg_xx, "LiNGAM")]:
        diff = _diff_mat(disc)
        tp = int((diff == 1).sum()); fp = int((diff == 2).sum()); fn = int((diff == 3).sum())
        axx.imshow(diff, cmap=cmp_cmap, norm=cmp_norm, aspect="auto")
        _setup_ticks(axx)
        axx.set_title(f"{label} X→X  (TP={tp}  FP={fp}  FN={fn})", fontsize=11, fontweight="bold")
        if annotate:
            _sym = {1: "→", 2: "→", 3: "○"}
            for i in range(n):
                for j in range(n):
                    v = int(diff[i, j])
                    if v != 0:
                        axx.text(j, i, _sym[v], ha="center", va="center",
                                 fontsize=8, color="white")

    # --- shared legend ---
    legend_handles = [
        Patch(facecolor="#2ca02c", label="True Positive  (TP)"),
        Patch(facecolor="#d62728", label="False Positive (FP)"),
        Patch(facecolor="#1f77b4", label="False Negative (FN)"),
        Patch(facecolor="#f5f5f5", edgecolor="#aaaaaa", label="True Negative  (TN)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Adjacency comparison (X-only) — {ctx['dataset']}   ·   adj[i,j]=1 means edge i→j",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    out = savefig(fig, plots_dir, ctx["dataset"], f"adjacency_comparison_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# CROSS-DATASET COMPARISON
# ============================================================================= #

def _r2_for_dataset(dataset, base_dir):
    """Load LGBM R² from a saved metrics JSON; load the saved model and score if absent."""
    metrics_path = base_dir / "models" / f"{dataset}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as _f:
            return json.load(_f)["lgbm"]["r2"]
    # Fallback: load saved model and score on saved test parquet (no retraining)
    try:
        import sys as _sys
        import pandas as _pd
        from sklearn.metrics import r2_score as _r2
        _sys.path.insert(0, str(base_dir))
        from predictive_models.predictive_models import LGBMRegressor as _LGBM
        _model_path = base_dir / "models" / f"{dataset}_lgbm"
        _m = _LGBM.load(str(_model_path))
        _te = _pd.read_parquet(base_dir / "data" / "processed" / f"{dataset}_test.parquet")
        return float(_r2(_te["Y"], _m.predict(_te.drop(columns=["Y"]))))
    except Exception as _e:
        print(f"[warn] could not compute R² for {dataset}: {_e}")
        return float("nan")


def _xonly_f1_for_dataset(dataset, causal_dir, synth_dir):
    """Return (pc_f1, lingam_f1) using X-only adjacency (Y node excluded)."""
    try:
        pc_xx = np.load(causal_dir / f"{dataset}_pc_train_adjacency.npy")
        lg_xx = np.load(causal_dir / f"{dataset}_lingam_train_adjacency.npy")
    except FileNotFoundError:
        return float("nan"), float("nan")
    n = pc_xx.shape[0]
    # True reference: prefer raw synthetic adjacency first (n+1 × n+1 with Y at index n),
    # then fall back to true_full_adjacency (e.g. for Sachs, already saved as full adj).
    synth_path = synth_dir / f"{dataset}_adjacency.npy"
    full_path  = causal_dir / f"{dataset}_true_full_adjacency.npy"
    if synth_path.exists():
        true_xx = np.load(synth_path)[:n, :n]
    elif full_path.exists():
        true_xx = np.load(full_path)[:n, :n]
    else:
        return float("nan"), float("nan")

    def _f1(ref, disc):
        r = (ref != 0).astype(int);  d = (disc != 0).astype(int)
        tp = int((r & d).sum());  fp = int(((d == 1) & (r == 0)).sum())
        fn = int(((r == 1) & (d == 0)).sum())
        pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rc = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0

    return _f1(true_xx, pc_xx), _f1(true_xx, lg_xx)


def plot_discovery_vs_r2(
    datasets=None,
    plots_dir=DEFAULT_PLOTS_DIR,
    show=False,
):
    """
    Dual-axis chart: causal discovery F1 (PC vs LiNGAM, X-only edges) on the
    left y-axis and LGBM test R² on the right y-axis. X-axis = dataset group.

    Integrates ``discovery_vs_r2_plot.py`` into the centralised plotting module.
    F1 is computed on X→X edges only (Y node excluded from both reference and
    discovered graphs) so the score reflects true causal structure recovery.

    Parameters
    ----------
    datasets : list of (dataset_name, display_label), optional
        Defaults to the synthetic-linear-confounded + Sachs pair used in the
        thesis::

            [("linear_conf_f50_s1000_p30", "Synthetic\\n(Linear, Confounded)"),
             ("sachs", "Sachs")]

    plots_dir : str or Path
        Output root. Saved to ``plots_dir/cross_dataset/discovery_vs_r2.png``.
    show : bool
        Call ``plt.show()`` if True.
    """
    if datasets is None:
        datasets = [
            ("linear_conf_f50_s1000_p30", "Synthetic\n(Linear, Confounded)"),
            ("sachs",                       "Sachs"),
        ]

    # Derive project root from the already-imported CAUSAL_DIR
    # CAUSAL_DIR = <root>/data/causal  ->  .parent.parent = <root>
    base_dir   = CAUSAL_DIR.parent.parent
    causal_dir = CAUSAL_DIR
    synth_dir  = base_dir / "data" / "synthetic"

    labels, f1_pc_list, f1_lg_list, r2_list = [], [], [], []
    for ds, label in datasets:
        pc_f1, lg_f1 = _xonly_f1_for_dataset(ds, causal_dir, synth_dir)
        r2 = _r2_for_dataset(ds, base_dir)
        labels.append(label)
        f1_pc_list.append(pc_f1)
        f1_lg_list.append(lg_f1)
        r2_list.append(r2)
        print(f"  {ds}: PC F1={pc_f1:.3f}  LiNGAM F1={lg_f1:.3f}  R²={r2:.4f}")

    x = np.arange(len(labels))
    width, offset = 0.28, 0.15
    PC_COLOR     = "#1f77b4"
    LINGAM_COLOR = "#ff7f0e"
    R2_COLOR     = "#2ca02c"

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars_pc = ax1.bar(x - offset, f1_pc_list, width,
                      color=PC_COLOR, alpha=0.85, zorder=3)
    bars_lg = ax1.bar(x + offset, f1_lg_list, width,
                      color=LINGAM_COLOR, alpha=0.85, zorder=3)
    (line_r2,) = ax2.plot(x, r2_list, color=R2_COLOR, marker="D",
                          markersize=9, linewidth=2.0, linestyle="--",
                          label="LGBM R²", zorder=4)

    ax1.set_ylabel("F1 Score (causal discovery, X-only edges)", fontsize=11)
    ax2.set_ylabel("R² (LGBM test performance)", fontsize=11, color=R2_COLOR)
    ax2.tick_params(axis="y", labelcolor=R2_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 1.05)
    ax1.set_xlim(-0.6, len(labels) - 0.4)

    for bar, val in zip(bars_pc, f1_pc_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                 color=PC_COLOR, fontweight="bold")
    for bar, val in zip(bars_lg, f1_lg_list):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9,
                 color=LINGAM_COLOR, fontweight="bold")
    for xi, val in zip(x, r2_list):
        if not np.isnan(val):
            ax2.text(xi + 0.07, val + 0.02, f"{val:.2f}",
                     ha="left", va="bottom", fontsize=9,
                     color=R2_COLOR, fontweight="bold")

    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    legend_handles = [
        Patch(color=PC_COLOR,     alpha=0.85, label="PC F1"),
        Patch(color=LINGAM_COLOR, alpha=0.85, label="LiNGAM F1"),
        line_r2,
    ]
    ax1.legend(handles=legend_handles, loc="upper left", fontsize=10, framealpha=0.9)

    st.style_title(ax1,
                   "Causal Discovery F1 vs Model R²",
                   "X-only edges (Y excluded) · left axis = F1 · right axis = R²")
    fig.tight_layout()
    out = savefig(fig, plots_dir, "cross_dataset", "discovery_vs_r2.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# METHOD LEVEL
# ============================================================================= #
def plot_gss_sss_scatter(ctx, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Per-method global GSS (x) vs sign-disagreement 1−SSS (y). Both are PC↔LiNGAM instability
    measures (≥0); lower-left = most robust to the discovery algorithm.

    Global GSS = mean_f( RMS_i(|φ(PC)|−|φ(LiNGAM)|) / output_std ).
    """
    gss, sss = gss_feature_dict(ctx), sss_feature_dict(ctx)
    methods = [m for m in ORDER if m in gss and m in sss]
    if not methods:
        return None
    fig, ax_ = plt.subplots(figsize=(6, 5))
    for m in methods:
        x = float(np.abs(gss[m]).mean()) * 100
        y = (1 - float(np.nanmean(sss[m]))) * 100
        ax_.scatter(x, y, s=150, color=METHOD_COLORS[m], edgecolor="white", lw=1.2, zorder=3)
        ax_.annotate(m, (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10, color=METHOD_COLORS[m], fontweight="bold")
    ax_.set_xlim(left=0)
    ax_.set_ylim(bottom=0, top=ax_.get_ylim()[1] * 1.15)
    ax_.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax_.set_xlabel("Magnitude Divergence (PC vs LiNGAM) [% of model-output std]")
    ax_.set_ylabel("Sign Disagreement [%]")
    st.style_title(ax_, f"Graph-discovery instability — {ctx['dataset']}",
                   "Lower-left = most stable across PC / LiNGAM (both axes ≥ 0)")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_sss_scatter_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_tga_sa_scatter(ctx, reference="True", plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Per (method, graph): global TGA (x) vs sign-disagreement 1−SA (y) vs the reference.
    Colour = method, marker = graph. Lower-left / TGA→0 = closer to the reference.

    Global TGA = mean_f( RMS_i(|φ(disc)|−|φ(ref)|) / output_std ).
    """
    tga, sa = tga_feature_dict(ctx, reference), sign_alignment_dict(ctx, reference)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in tga and (m, g) in sa]
    if not keys:
        return None
    fig, ax_ = plt.subplots(figsize=(6.6, 5))
    for m, g in keys:
        x = float(np.abs(tga[(m, g)]).mean()) * 100
        y = (1 - float(np.nanmean(sa[(m, g)]))) * 100
        ax_.scatter(x, y, s=130, color=METHOD_COLORS[m], marker=GRAPH_MARKERS.get(g, "o"),
                    edgecolor="white", lw=1.1, zorder=3)
    sub, disp = REF_DISPLAY.get(reference, ("ref", reference))
    ax_.set_xlim(left=0); ax_.set_ylim(0, 50)
    ax_.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax_.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax_.set_xlabel(f"Magnitude Divergence vs {disp} [% of model-output std]")
    ax_.set_ylabel(f"Sign Disagreement vs {disp} [%]")
    method_handles = [Line2D([0], [0], marker="o", ls="", color=METHOD_COLORS[m],
                             markersize=10, label=m) for m in ORDER if any(k[0] == m for k in keys)]
    graph_handles = [Line2D([0], [0], marker=GRAPH_MARKERS[g], ls="", color="#666666",
                            markersize=10, label=g) for g in DISC_GRAPHS]
    leg1 = ax_.legend(handles=method_handles, title="Method", loc="upper left")
    ax_.add_artist(leg1)
    ax_.legend(handles=graph_handles, title="Graph", loc="lower right")
    st.style_title(ax_, f"Alignment to {disp} — {ctx['dataset']}",
                   f"Lower-left = closer to {disp} (both axes ≥ 0)")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"],
                  f"tga_sa_scatter_{reference.lower()}_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# GROUP-METRIC EXPORT (markdown tables behind the method-level scatters)
# ============================================================================= #
def gss_sss_table(ctx):
    """Per Shapley method: global GSS and SSS (the coordinates of the GSS–SSS scatter).

    GSS = mean_f( RMS_i(|φ(PC)|−|φ(LiNGAM)|) / output_std ).
    """
    gss, sss = gss_feature_dict(ctx), sss_feature_dict(ctx)
    rows = []
    for m in ORDER:
        if m in gss and m in sss:
            s = float(np.nanmean(sss[m])) * 100
            rows.append(dict(Method=m, GSS_pct=float(np.abs(gss[m]).mean()) * 100,
                             SSS_pct=s, SignDisagree_pct=100 - s))
    return rows


def tga_sa_table(ctx, reference="True"):
    """Per (Shapley method, discovered graph): global TGA and Sign Alignment vs the reference
    (the coordinates of the TGA–SA scatter).

    TGA = mean_f( RMS_i(|φ(disc)|−|φ(ref)|) / output_std ).
    """
    tga, sa = tga_feature_dict(ctx, reference), sign_alignment_dict(ctx, reference)
    rows = []
    for m in ORDER:
        for g in DISC_GRAPHS:
            if (m, g) in tga and (m, g) in sa:
                a = float(np.nanmean(sa[(m, g)])) * 100
                rows.append(dict(Method=m, Graph=g, Reference=reference,
                                 TGA_pct=float(np.abs(tga[(m, g)]).mean()) * 100,
                                 SignAlign_pct=a, SignDisagree_pct=100 - a))
    return rows


def _md_table(rows, floatfmt="{:.2f} %"):
    if not rows:
        return "_(no data)_\n"
    cols = list(rows[0])
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = []
    for r in rows:
        cells = [floatfmt.format(r[c]) if isinstance(r[c], float) else str(r[c]) for c in cols]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def export_method_level_metrics(ctx, plots_dir=DEFAULT_PLOTS_DIR):
    """Write the group metrics behind the GSS–SSS and TGA–SA scatters to one markdown file,
    grouped by Shapley method, discovered graph, and reference (True / Traditional)."""
    ds = ctx["dataset"]
    parts = [f"# Method-level group metrics — {ds}\n",
             "Scalar summaries behind the method-level scatter plots, using the standardised metric names: "
             "**Magnitude Divergence ΔM** (column `GSS_pct`/`TGA_pct`) = mean over features of the per-feature "
             "RMS(|φ_subject|−|φ_ref|)/model-output std, in %; **Sign Disagreement D** (column `SignDisagree_pct`) "
             "= mean over features of the % of instances whose sign disagrees with the reference. "
             "Reference subscripts: disc = PC↔LiNGAM, oracle = True/consensus DAG, base = Traditional.\n",
             "## Cross-discovery instability — ΔM_disc vs D_disc (per Shapley method)",
             "`GSS_pct` = ΔM_disc = RMS_i(|φ(PC)|−|φ(LiNGAM)|)/model-output std per feature × 100, averaged over features; "
             "`SSS_pct` = % of instances with matching sign PC vs LiNGAM; `SignDisagree_pct` = D_disc = 100 − SSS (the scatter's y-axis).\n",
             _md_table(gss_sss_table(ctx))]
    for ref in ("True", "Traditional"):
        rows = tga_sa_table(ctx, ref)
        if rows:
            sub = REF_DISPLAY.get(ref, ("ref", ref))[0]
            parts += [f"\n## Alignment to {ref} — ΔM_{sub} vs D_{sub} "
                      f"(per Shapley method × discovered graph)",
                      f"`TGA_pct` = ΔM_{sub} = RMS_i(|φ(disc)|−|φ({ref})|)/model-output std per feature × 100, averaged over features; "
                      f"`SignAlign_pct` = % of instances with matching sign vs {ref}; `SignDisagree_pct` = D_{sub} = 100 − SignAlign (the scatter's y-axis).\n",
                      _md_table(rows)]
    out_dir = Path(plots_dir) / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"method_level_metrics_{ds}.md"
    dest.write_text("\n".join(parts))
    return dest


# ============================================================================= #
# driver
# ============================================================================= #
def run_all(ctx, plots_dir=DEFAULT_PLOTS_DIR):
    outs = [
        plot_gss_delta_box(ctx, plots_dir=plots_dir),
        plot_gss_heatmap(ctx, plots_dir=plots_dir),
        plot_sss_heatmap(ctx, plots_dir=plots_dir),
        plot_sign_alignment_heatmap(ctx, reference="True", plots_dir=plots_dir),
        plot_sign_alignment_heatmap(ctx, reference="Traditional", plots_dir=plots_dir),
        plot_tga_heatmap(ctx, reference="True", plots_dir=plots_dir),
        plot_tga_heatmap(ctx, reference="Traditional", plots_dir=plots_dir),
        plot_adjacency_comparison(ctx, plots_dir=plots_dir),
        plot_gss_sss_scatter(ctx, plots_dir=plots_dir),
        plot_tga_sa_scatter(ctx, reference="True", plots_dir=plots_dir),
        plot_tga_sa_scatter(ctx, reference="Traditional", plots_dir=plots_dir),
        export_method_level_metrics(ctx, plots_dir=plots_dir),
    ]
    return [str(o) for o in outs if o is not None]


if __name__ == "__main__":
    import sys

    for ds in (sys.argv[1:] or list(DEFAULT_DATASETS)):
        try:
            made = run_all(load_context(ds))
            print(f"[ok] {ds}: {len(made)} figures")
            for m in made:
                print("     ", m)
        except Exception as e:
            print(f"[skip] {ds}: {type(e).__name__}: {e}")

    # Cross-dataset comparison (always runs regardless of argv)
    try:
        out = plot_discovery_vs_r2()
        print(f"[ok] cross_dataset: {out}")
    except Exception as e:
        print(f"[skip] cross_dataset: {type(e).__name__}: {e}")
