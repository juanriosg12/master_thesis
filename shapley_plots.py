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
    tga_instance(method, graph, ref) = |phi_disc| - |phi_ref|     (signed, per instance)
    tga_feature                       = mean_i | tga_instance |     (absolute average)
    global TGA                        = mean_f | tga_feature |
    gss_*                             = same logic, PC subject, LiNGAM reference

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
    SEQ_CMAP, DIV_CMAP, savefig,
)

ORDER = METHOD_ORDER                     # ["Asymmetric", "Causal", "Flow"]
REF_TOKEN = {"True": "True", "Traditional": SCRATCH, "Scratch": SCRATCH}


# ============================================================================= #
# metric dictionaries (ctx -> per-feature / per-instance arrays)
# ============================================================================= #
def _methods_present(ctx, graphs):
    return [m for m in ORDER if all(f"{m} ({g})" in ctx["shap"] for g in graphs)]


def gss_delta_dict(ctx):
    """method -> (N, F) signed |phi_PC| - |phi_LiNGAM|."""
    return {m: gss_instance(ctx, m) for m in _methods_present(ctx, ["PC", "LiNGAM"])}


def gss_feature_dict(ctx):
    """method -> (F,) per-feature GSS."""
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
    """(method, graph) -> (F,) per-feature TGA vs reference."""
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
                    ax_.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
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
    ax_.set_ylabel("|φ(PC)| − |φ(LiNGAM)|")
    ax_.legend(handles=[Patch(facecolor=METHOD_COLORS[m], alpha=0.75, label=m) for m in methods],
               title="Method", loc="upper left", ncol=nM)
    st.style_title(ax_, f"Per-instance magnitude delta (PC − LiNGAM) — {ctx['dataset']}",
                   f"Top {len(rank)} features by GSS · positive = PC higher · tight box = consistent across instances")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_delta_box_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_gss_heatmap(ctx, top_n=40, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × method Graph Sensitivity Score heatmap (PC vs LiNGAM). Sequential (cividis):
    brighter = larger magnitude difference between PC and LiNGAM graphs."""
    gss = gss_feature_dict(ctx)
    if not gss:
        return None
    names = feature_names(ctx)
    labels = [m for m in ORDER if m in gss]
    mat = np.stack([gss[m] for m in labels], axis=1)                 # (F, M)
    top = np.argsort(mat.mean(axis=1))[::-1][:min(top_n, len(names))]
    Z = mat[top].T                                                   # (M, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 1.0 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], SEQ_CMAP, 0, float(mat.max()), "GSS",
             annotate=len(top) <= 14)
    st.style_title(ax_, f"Feature-level graph sensitivity — PC vs LiNGAM — {ctx['dataset']}",
                   f"GSS = mean_i ||φ(PC)|−|φ(LiNGAM)|| · top {len(top)} features by mean GSS")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_heatmap_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_sss_heatmap(ctx, top_n=40, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × method Sign Stability heatmap (PC vs LiNGAM). Blue = signs agree across the
    two discovered graphs, red = they flip. Diverging around 0.5 (random)."""
    sss = sss_feature_dict(ctx)
    if not sss:
        return None
    names = feature_names(ctx)
    labels = [m for m in ORDER if m in sss]
    mat = np.stack([sss[m] for m in labels], axis=1)                 # (F, M)
    top = np.argsort(np.nanmean(mat, axis=1))[:min(top_n, len(names))]
    Z = mat[top].T                                                   # (M, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 1.0 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], DIV_CMAP, 0, 1, "SSS",
             annotate=len(top) <= 14,
             cbar_ticks=[0, 0.5, 1], cbar_ticklabels=["0 flips", "0.5", "1 agrees"])
    st.style_title(ax_, f"Feature-level sign stability — PC vs LiNGAM — {ctx['dataset']}",
                   f"Fraction of instances with matching SHAP sign · top {len(top)} most-unstable features")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"sss_heatmap_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# DISCOVERED (PC, LiNGAM) vs REFERENCE (True / Traditional)
# ============================================================================= #
def plot_sign_alignment_heatmap(ctx, reference="True", top_n=40,
                                plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × (method, graph) sign-alignment heatmap vs the reference. Blue = sign matches
    reference, red = opposite. Diverging around 0.5."""
    sa = sign_alignment_dict(ctx, reference)
    if not sa:
        return None
    names = feature_names(ctx)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in sa]
    labels = [f"{m} ({g})" for m, g in keys]
    mat = np.stack([sa[k] for k in keys], axis=1)                   # (F, K)
    top = np.argsort(np.nanmean(mat, axis=1))[:min(top_n, len(names))]
    Z = mat[top].T                                                  # (K, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 0.55 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], DIV_CMAP, 0, 1, "Sign align",
             annotate=len(top) <= 14,
             cbar_ticks=[0, 0.5, 1], cbar_ticklabels=["0 opp.", "0.5", "1 match"])
    st.style_title(ax_, f"Sign alignment vs {reference} — {ctx['dataset']}",
                   f"sign(φ_disc)=sign(φ_{reference}) per instance · sorted by worst mean alignment")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"],
                  f"sign_alignment_heatmap_{reference.lower()}_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_tga_heatmap(ctx, reference="True", top_n=40, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Feature × (method, graph) magnitude-TGA heatmap vs the reference. Sequential (cividis):
    brighter = larger magnitude difference from the reference."""
    tga = tga_feature_dict(ctx, reference)
    if not tga:
        return None
    names = feature_names(ctx)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in tga]
    labels = [f"{m} ({g})" for m, g in keys]
    mat = np.stack([tga[k] for k in keys], axis=1)                  # (F, K)
    top = np.argsort(mat.mean(axis=1))[::-1][:min(top_n, len(names))]
    Z = mat[top].T                                                 # (K, n_top)
    fig, ax_ = plt.subplots(figsize=(max(8, 0.22 * len(top) + 2), 0.55 * len(labels) + 1.8))
    _heatmap(ax_, Z, labels, [names[i] for i in top], SEQ_CMAP, 0, float(mat.max()), "TGA",
             annotate=len(top) <= 14)
    st.style_title(ax_, f"Feature-level magnitude TGA vs {reference} — {ctx['dataset']}",
                   f"TGA = mean_i ||φ(disc)|−|φ({reference})|| · top {len(top)} features by mean TGA")
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
    Each panel uses the reference colour of its graph."""
    adj = load_adjacency(ctx)
    if adj is None:
        return None
    true_xx, pc_xx, lg_xx, names = adj
    n = len(names)
    annotate = n <= 16
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    panels = [(axes[0], true_xx, "True X→X", "Greys"),
              (axes[1], pc_xx, "PC X→X", "Blues"),
              (axes[2], lg_xx, "LiNGAM X→X", "Oranges")]
    for axx, mat, title, cmap in panels:
        axx.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        axx.set_xticks(range(n)); axx.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        axx.set_yticks(range(n)); axx.set_yticklabels(names, fontsize=7)
        axx.set_xticks(np.arange(-.5, n, 1), minor=True)
        axx.set_yticks(np.arange(-.5, n, 1), minor=True)
        axx.grid(which="minor", color="#dddddd", linewidth=0.5)
        axx.tick_params(which="minor", length=0)
        axx.set_title(f"{title}  ({int(mat.sum())} edges)", fontsize=11, fontweight="bold")
        if annotate:
            for i in range(n):
                for j in range(n):
                    if mat[i, j]:
                        axx.text(j, i, "→", ha="center", va="center", fontsize=8, color="#222222")
    fig.suptitle(f"Adjacency comparison (X-only) — {ctx['dataset']}   ·   adj[i,j]=1 means edge i→j",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"adjacency_comparison_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# METHOD LEVEL
# ============================================================================= #
def plot_gss_sss_scatter(ctx, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Per-method |GSS| (x) vs sign-disagreement 1−SSS (y). Both are PC↔LiNGAM instability
    measures (≥0); lower-left = most robust to the discovery algorithm."""
    gss, sss = gss_feature_dict(ctx), sss_feature_dict(ctx)
    methods = [m for m in ORDER if m in gss and m in sss]
    if not methods:
        return None
    fig, ax_ = plt.subplots(figsize=(6, 5))
    for m in methods:
        x = float(np.abs(gss[m]).mean())
        y = 1 - float(np.nanmean(sss[m]))
        ax_.scatter(x, y, s=150, color=METHOD_COLORS[m], edgecolor="white", lw=1.2, zorder=3)
        ax_.annotate(m, (x, y), textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=10, color=METHOD_COLORS[m], fontweight="bold")
    ax_.set_xlim(left=0); ax_.set_ylim(bottom=0)
    ax_.set_xlabel("|GSS| — magnitude difference (PC vs LiNGAM)")
    ax_.set_ylabel("1 − SSS — sign disagreement rate")
    st.style_title(ax_, f"Graph-discovery instability — {ctx['dataset']}",
                   "Lower-left = most stable across PC / LiNGAM (both axes ≥ 0)")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"], f"gss_sss_scatter_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


def plot_tga_sa_scatter(ctx, reference="True", plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Per (method, graph): global TGA (x) vs sign-disagreement 1−SA (y) vs the reference.
    Colour = method, marker = graph. Lower-left / TGA→0 = closer to the reference."""
    tga, sa = tga_feature_dict(ctx, reference), sign_alignment_dict(ctx, reference)
    keys = [(m, g) for m in ORDER for g in DISC_GRAPHS if (m, g) in tga and (m, g) in sa]
    if not keys:
        return None
    fig, ax_ = plt.subplots(figsize=(6.6, 5))
    for m, g in keys:
        x = float(np.abs(tga[(m, g)]).mean())
        y = 1 - float(np.nanmean(sa[(m, g)]))
        ax_.scatter(x, y, s=130, color=METHOD_COLORS[m], marker=GRAPH_MARKERS.get(g, "o"),
                    edgecolor="white", lw=1.1, zorder=3)
    ax_.set_xlim(left=0); ax_.set_ylim(0, 0.5)
    ax_.set_xlabel(f"magnitude TGA vs {reference}")
    ax_.set_ylabel("1 − Sign Alignment")
    method_handles = [Line2D([0], [0], marker="o", ls="", color=METHOD_COLORS[m],
                             markersize=10, label=m) for m in ORDER if any(k[0] == m for k in keys)]
    graph_handles = [Line2D([0], [0], marker=GRAPH_MARKERS[g], ls="", color="#666666",
                            markersize=10, label=g) for g in DISC_GRAPHS]
    leg1 = ax_.legend(handles=method_handles, title="Method", loc="upper left")
    ax_.add_artist(leg1)
    ax_.legend(handles=graph_handles, title="Graph", loc="lower right")
    st.style_title(ax_, f"Alignment to {reference} — {ctx['dataset']}",
                   "Colour = method · marker = graph · lower-left = closer to reference")
    fig.tight_layout()
    out = savefig(fig, plots_dir, ctx["dataset"],
                  f"tga_sa_scatter_{reference.lower()}_{ctx['dataset']}.png")
    plt.show() if show else plt.close(fig)
    return out


# ============================================================================= #
# GROUP-METRIC EXPORT (markdown tables behind the method-level scatters)
# ============================================================================= #
def gss_sss_table(ctx):
    """Per Shapley method: global |GSS| and SSS (the coordinates of the GSS–SSS scatter)."""
    gss, sss = gss_feature_dict(ctx), sss_feature_dict(ctx)
    rows = []
    for m in ORDER:
        if m in gss and m in sss:
            s = float(np.nanmean(sss[m]))
            rows.append(dict(Method=m, GSS=float(np.abs(gss[m]).mean()),
                             SSS=s, SignDisagree=1 - s))
    return rows


def tga_sa_table(ctx, reference="True"):
    """Per (Shapley method, discovered graph): global TGA and Sign Alignment vs the reference
    (the coordinates of the TGA–SA scatter)."""
    tga, sa = tga_feature_dict(ctx, reference), sign_alignment_dict(ctx, reference)
    rows = []
    for m in ORDER:
        for g in DISC_GRAPHS:
            if (m, g) in tga and (m, g) in sa:
                a = float(np.nanmean(sa[(m, g)]))
                rows.append(dict(Method=m, Graph=g, Reference=reference,
                                 TGA=float(np.abs(tga[(m, g)]).mean()),
                                 SignAlign=a, SignDisagree=1 - a))
    return rows


def _md_table(rows, floatfmt="{:.4f}"):
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
             "Scalar summaries behind the method-level scatter plots. Magnitude metrics "
             "(GSS, TGA) are the absolute average of the per-feature metric; sign metrics "
             "(SSS, Sign Alignment) are the mean over features.\n",
             "## Graph-discovery instability — GSS vs SSS (per Shapley method)",
             "`|GSS|` = magnitude difference PC vs LiNGAM; `SSS` = sign agreement PC vs LiNGAM; "
             "`SignDisagree` = 1 − SSS (the scatter's y-axis).\n",
             _md_table(gss_sss_table(ctx))]
    for ref in ("True", "Traditional"):
        rows = tga_sa_table(ctx, ref)
        if rows:
            parts += [f"\n## Alignment to {ref} — TGA vs Sign Alignment "
                      f"(per Shapley method × discovered graph)",
                      f"`TGA` = magnitude difference vs {ref}; `SignAlign` = sign agreement vs "
                      f"{ref}; `SignDisagree` = 1 − SignAlign (the scatter's y-axis).\n",
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
