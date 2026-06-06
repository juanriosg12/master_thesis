"""
analysis_utils.py
-----------------
Shared metric and plot utilities for the master thesis SHAP analysis.

Usage from a notebook in notebooks/:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path("..").resolve()))
    from analysis_utils import (
        resolve_features,
        mean_abs_shap, top_k_jaccard,
        compute_sign_agreement, compute_sign_alignment, compute_tga, compute_sss,
        compute_gss, top_k_features, rank_shift_top_k,
        edge_recovery, flow_graph_stats, parse_pipeline_timing,
        build_dag_graph, make_dag_pos, draw_dag, PROX_PALETTE,
        make_shap_scatter_figure, make_instance_shap_figure,
        plot_shap_scatter_pc_vs_lingam, plot_instance_shap_pc_vs_lingam,
        plot_dag_highlight_3panel, plot_gss_sss_scatter, plot_tga_sa_scatter,
        DEFAULT_METHOD_COLORS,
    )

Sections:
  1. Constants (colour palettes)
  2. Feature helpers
  3. Metric helpers (sign agreement, sign alignment, TGA, SSS, edge recovery, Jaccard)
     Note: compute_sign_alignment and compute_tga accept a ``reference`` kwarg
     (default ``"True"``). Pass ``"Scratch"`` to compare vs the graph-free baseline.
  3b. Per-feature diagnostic helpers (compute_gss, top_k_features, rank_shift_top_k)
  4. Graph (DAG) helpers
  5. Plot builders (Plotly figure constructors — pure functions, return go.Figure)
  6. Pipeline helpers (log parsing, graph stats)
"""
from __future__ import annotations

import collections
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

PROX_PALETTE: list[str] = [
    "#e67e22", "#f39c12", "#f1c40f", "#1abc9c", "#2ecc71", "#27ae60",
]

DEFAULT_METHOD_COLORS: dict[str, str] = {
    "Scratch":     "#636363",
    "Asymmetric":  "#2166ac",
    "Causal":      "#4dac26",
    "ShapleyFlow": "#d01c8b",
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_features(
    feature_names: list[str],
    scratch_rank:  np.ndarray,
    top_features:  list[str],
    features:      list[str] | None = None,
) -> tuple[list[int], list[str]]:
    """
    Resolve a feature list to (indices, labels).

    Parameters
    ----------
    feature_names : list of str
        Full ordered list of feature names (X0 … Xn-1).
    scratch_rank : ndarray of int
        Indices of the top-N features sorted by Scratch importance (descending).
    top_features : list of str
        Corresponding feature names for scratch_rank.
    features : list of str or None
        Specific feature names to use. If None, returns the Scratch top-N.

    Returns
    -------
    feat_idx    : list of int
    feat_labels : list of str
    """
    if features is None:
        return scratch_rank.tolist(), list(top_features)
    idx = [feature_names.index(f) for f in features]
    return idx, list(features)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def mean_abs_shap(shap_data: dict[str, np.ndarray], key: str) -> np.ndarray:
    """Return mean |SHAP| per feature for one method. Shape: (n_features,)."""
    return np.abs(shap_data[key]).mean(axis=0)


def top_k_jaccard(arr1: np.ndarray, arr2: np.ndarray, k: int) -> float:
    """Jaccard overlap of the top-k indices of arr1 and arr2."""
    s1 = set(np.argsort(arr1)[-k:])
    s2 = set(np.argsort(arr2)[-k:])
    return len(s1 & s2) / len(s1 | s2)


def compute_sign_agreement(
    method_list: list[str],
    shap_data:   dict[str, np.ndarray],
    feat_idx:    np.ndarray,
) -> tuple[np.ndarray | None, list[str]]:
    """
    For each (instance i, feature f): max(n_pos, n_neg) / n_methods.

    Returns
    -------
    agreement : ndarray (n_instances, n_feats) in [0.5, 1.0], or None if < 2 methods.
    available : list of method names actually used.
    """
    available = [m for m in method_list if m in shap_data]
    if len(available) < 2:
        return None, available
    stack     = np.stack([shap_data[m][:, feat_idx] for m in available], axis=0)
    n_methods = stack.shape[0]
    signs     = np.sign(stack)
    n_pos     = (signs > 0).sum(axis=0)
    n_neg     = (signs < 0).sum(axis=0)
    return np.maximum(n_pos, n_neg) / n_methods, available


def compute_sign_alignment(
    shap_data:    dict[str, np.ndarray],
    base_methods: list[str],
    disc_graphs:  list[str],
    reference:    str = "True",
) -> dict[tuple[str, str], np.ndarray]:
    """
    Per-feature sign alignment rate of each (method, graph) vs a reference.

    sign(φ_disc) == sign(φ_ref) → 1, else 0.
    Instances where φ_ref == 0 are excluded (set to NaN).

    Reference resolution (first match wins):
    1. ``f"{method} ({reference})"`` — per-method key, e.g. ``"Asymmetric (True)"``.
    2. ``reference`` as a direct key — method-independent, e.g. ``"Scratch"``.

    Parameters
    ----------
    shap_data    : dict with keys like ``"{method} ({graph})"`` or ``"Scratch"``.
    base_methods : list of method names, e.g. ``["Asymmetric", "Causal", "Flow"]``.
    disc_graphs  : graphs to evaluate, e.g. ``["PC", "LiNGAM"]``.
    reference    : reference variant. Default ``"True"`` (True-DAG alignment).
                   Pass ``"Scratch"`` for graph-free baseline alignment.

    Returns
    -------
    sign_align : dict mapping (method, graph) → ndarray (n_features,) in [0, 1]
    """
    sign_align: dict[tuple[str, str], np.ndarray] = {}
    for meth in base_methods:
        ref_key = f"{meth} ({reference})"
        if ref_key not in shap_data:
            ref_key = reference
        if ref_key not in shap_data:
            continue
        phi_ref = shap_data[ref_key]
        for g in disc_graphs:
            disc_key = f"{meth} ({g})"
            if disc_key not in shap_data:
                continue
            phi_disc     = shap_data[disc_key]
            nonzero_mask = phi_ref != 0
            agree        = (np.sign(phi_disc) == np.sign(phi_ref)).astype(float)
            agree[~nonzero_mask] = np.nan
            with np.errstate(all="ignore"):
                sign_align[(meth, g)] = np.nanmean(agree, axis=0)
    return sign_align


def compute_tga(
    shap_data:    dict[str, np.ndarray],
    feature_names: list[str],
    base_methods: list[str],
    disc_graphs:  list[str],
    reference:    str   = "True",
    output_range: float | None = None,
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, set[int]]]:
    """
    RMS magnitude difference between discovered-graph and reference attributions,
    normalised by model output range.

    TGA(method, graph, feature) = sqrt( mean_i (|φ(disc,i,f)| − |φ(ref,i,f)|)² )
                                   ÷ output_range

    RMS aggregation (instead of mean of abs) handles sparsity better: large
    deviations in a minority of instances are not washed out by near-zero ones.
    Dividing by output_range makes the metric dimensionless and comparable
    across datasets with different target scales.

    Reference is resolved the same way as in ``compute_sign_alignment``.

    Parameters
    ----------
    shap_data     : dict with keys like ``"{method} ({graph})"`` or ``"Scratch"``.
    feature_names : list of feature name strings (kept for API compatibility).
    base_methods  : list of method names.
    disc_graphs   : graphs to evaluate.
    reference     : reference variant. Default ``"True"``. Pass ``"Scratch"``
                    for graph-free baseline alignment.
    output_range  : model prediction range (max − min on test set). When provided
                    the per-feature RMS is divided by this value.

    Returns
    -------
    tga_data       : dict mapping (method, graph) → ndarray (n_features,) in [0, +∞)
    zero_feat_sets : dict mapping method → empty set (kept for API compatibility)
    """
    tga_data:       dict[tuple[str, str], np.ndarray] = {}
    zero_feat_sets: dict[str, set[int]]               = {}
    for meth in base_methods:
        ref_key = f"{meth} ({reference})"
        if ref_key not in shap_data:
            ref_key = reference
        if ref_key not in shap_data:
            continue
        phi_ref = shap_data[ref_key]
        abs_ref = np.abs(phi_ref)
        zero_feat_sets[meth] = set()  # no NaN features — absolute metric, no division
        for g in disc_graphs:
            disc_key = f"{meth} ({g})"
            if disc_key not in shap_data:
                continue
            phi_disc = shap_data[disc_key]
            diff     = np.abs(phi_disc) - abs_ref          # signed per-instance diff
            feat_rms = np.sqrt(np.mean(np.square(diff), axis=0))  # RMS over instances
            if output_range:
                feat_rms = feat_rms / output_range
            tga_data[(meth, g)] = feat_rms
    return tga_data, zero_feat_sets


def compute_sss(
    shap_data:    dict[str, np.ndarray],
    base_methods: list[str],
) -> dict[str, np.ndarray]:
    """
    Sign Stability Score (SSS): per-feature sign agreement rate between
    PC and LiNGAM attributions for the same method.

    Analogous to ``compute_sign_alignment``, but comparing two discovered
    graphs (PC vs LiNGAM) rather than a discovered graph vs the True DAG.

    .. math::
        \\text{SSS}(m, f) = \\frac{1}{|\\{i : \\phi^{PC}_{i,f} \\neq 0 \\;\\wedge\\; \\phi^{LiNGAM}_{i,f} \\neq 0\\}|}
        \\sum_i \\mathbf{1}[\\text{sign}(\\phi^{PC}_{i,f}) = \\text{sign}(\\phi^{LiNGAM}_{i,f})]

    Instances where either attribution is zero are excluded (set to NaN).

    Parameters
    ----------
    shap_data    : dict mapping key strings to (n_instances, n_features) arrays.
                   Expected keys: ``"{method} (PC)"`` and ``"{method} (LiNGAM)"``.
    base_methods : list of method names, e.g. ``["Asymmetric", "Causal", "Flow"]``.

    Returns
    -------
    sss : dict mapping method → ndarray (n_features,) in [0, 1]
          Higher = signs more consistent between PC and LiNGAM graphs.
    """
    sss: dict[str, np.ndarray] = {}
    for meth in base_methods:
        pc_key = f"{meth} (PC)"
        lg_key = f"{meth} (LiNGAM)"
        if pc_key not in shap_data or lg_key not in shap_data:
            continue
        phi_pc = shap_data[pc_key]
        phi_lg = shap_data[lg_key]
        valid  = (phi_pc != 0) & (phi_lg != 0)
        agree  = (np.sign(phi_pc) == np.sign(phi_lg)).astype(float)
        agree[~valid] = np.nan
        with np.errstate(all="ignore"):
            sss[meth] = np.nanmean(agree, axis=0)
    return sss


def edge_recovery(
    true_adj: np.ndarray,
    pred_adj: np.ndarray,
) -> tuple[float, float, float]:
    """
    Direction-aware edge precision, recall, and F1 score.

    Returns
    -------
    precision, recall, f1
    """
    true_edges = set(zip(*np.where(true_adj != 0)))
    pred_edges = set(zip(*np.where(pred_adj != 0)))
    tp   = len(true_edges & pred_edges)
    prec = tp / len(pred_edges) if pred_edges else 0.0
    rec  = tp / len(true_edges) if true_edges  else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Per-feature diagnostic helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_gss(
    shap_data:    dict[str, np.ndarray],
    base_methods: list[str],
    output_range: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Per-feature RMS magnitude difference between PC and LiNGAM outputs,
    normalised by model output range.

    GSS(method, feature) = sqrt( mean_i (|φ^PC_{i,f}| − |φ^LiNGAM_{i,f}|)² )
                           ÷ output_range

    RMS aggregation handles sparsity better than mean(|diff|): large per-instance
    deviations dominate rather than being averaged away. Dividing by output_range
    makes the metric dimensionless and comparable across datasets.

    Parameters
    ----------
    shap_data    : dict with keys like ``"{method} (PC)"`` and ``"{method} (LiNGAM)"``.
    base_methods : list of method names.
    output_range : model prediction range (max − min on test set). When provided
                   the per-feature RMS is divided by this value.

    Returns
    -------
    gss : dict mapping method → ndarray (n_features,) in [0, +∞)
          Near 0 = PC and LiNGAM agree; higher = larger magnitude difference.
    """
    gss: dict[str, np.ndarray] = {}
    for meth in base_methods:
        pc_key = f"{meth} (PC)"
        lg_key = f"{meth} (LiNGAM)"
        if pc_key not in shap_data or lg_key not in shap_data:
            continue
        diff     = np.abs(shap_data[pc_key]) - np.abs(shap_data[lg_key])  # signed
        feat_rms = np.sqrt(np.mean(np.square(diff), axis=0))
        if output_range:
            feat_rms = feat_rms / output_range
        gss[meth] = feat_rms
    return gss


def top_k_features(
    score_arr:     np.ndarray,
    feature_names: list[str],
    k:             int  = 5,
    ascending:     bool = False,
    score_col:     str  = "score",
) -> pd.DataFrame:
    """
    Return the top-k (or bottom-k if ascending=True) features ranked by magnitude.

    Ranking is always performed on ``|score_arr|`` so that signed metrics (e.g. GSS,
    TGA) are ordered by the size of the change regardless of direction.  The returned
    DataFrame contains the **original signed values**, allowing the caller to see
    whether the shift is positive or negative (e.g. PC > LiNGAM vs LiNGAM > PC).

    - ``ascending=False`` → top-k by **highest |score|** (largest magnitude change).
    - ``ascending=True``  → bottom-k by **lowest |score|**  (closest to zero / most stable).

    NaN entries are silently ignored.

    Parameters
    ----------
    score_arr     : per-feature score array of shape (n_features,).  May be signed.
    feature_names : list of feature name strings aligned with score_arr.
    k             : number of features to return.
    ascending     : if True return the k features with *smallest* absolute score.
    score_col     : column name for the score in the returned DataFrame.

    Returns
    -------
    DataFrame with columns [feature, <score_col>] (signed values) indexed by rank (1-based).
    """
    valid    = ~np.isnan(score_arr)
    v_idx    = np.where(valid)[0]
    v_abs    = np.abs(score_arr[v_idx])          # rank by magnitude
    order    = np.argsort(v_abs) if ascending else np.argsort(v_abs)[::-1]
    sel      = order[:k]
    top_idx  = v_idx[sel]
    return pd.DataFrame(
        {"feature": [feature_names[i] for i in top_idx], score_col: score_arr[top_idx]},
        index=pd.RangeIndex(1, len(top_idx) + 1, name="rank"),
    )


def rank_shift_top_k(
    shap_data:     dict[str, np.ndarray],
    feature_names: list[str],
    query_keys:    list[str],
    reference_key: str  = "Scratch",
    k:             int  = 5,
    ascending:     bool = False,
) -> pd.DataFrame:
    """
    For each query key, find the top-k features with the largest (or smallest
    if ascending=True) absolute shift in importance rank (mean |SHAP|) relative
    to a reference key.

    Parameters
    ----------
    shap_data     : dict mapping key strings to (n_instances, n_features) arrays.
    feature_names : list of feature name strings.
    query_keys    : keys to evaluate, e.g. ``["Asymmetric (PC)", "Flow (PC)"]``.
    reference_key : reference key, default ``"Scratch"``.
    k             : number of features to return per query key.
    ascending     : if True, return the k features with the *smallest* rank shift
                    (most stable importance ordering). Default False (largest shift).

    Returns
    -------
    DataFrame with columns [feature, ref_rank, query_rank, delta_rank]
    with a 2-level index (key, rank).  Rank 1 = largest (or smallest) rank shift.
    """
    if reference_key not in shap_data:
        raise KeyError(f"Reference key '{reference_key}' not found in shap_data.")

    ref_imp  = mean_abs_shap(shap_data, reference_key)
    ref_rank = np.argsort(np.argsort(-ref_imp))   # 0 = most important feature

    rows = []
    for qkey in query_keys:
        if qkey not in shap_data:
            continue
        q_rank  = np.argsort(np.argsort(-mean_abs_shap(shap_data, qkey)))
        delta   = np.abs(ref_rank - q_rank)
        top_idx = (np.argsort(delta) if ascending else np.argsort(delta)[::-1])[:k]
        for rk, fi in enumerate(top_idx, 1):
            rows.append({
                "key":        qkey,
                "rank":       rk,
                "feature":    feature_names[fi],
                "ref_rank":   int(ref_rank[fi]) + 1,
                "query_rank": int(q_rank[fi]) + 1,
                "delta_rank": int(delta[fi]),
            })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index(["key", "rank"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Graph (DAG) helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_dag_graph(
    adj:   np.ndarray,
    names: list[str],
) -> tuple:
    """
    Build a NetworkX DiGraph from an adjacency matrix and compute topological
    depth from source nodes (excluding Y as an intermediate).

    Returns
    -------
    G, topo_depth, level_groups, sources, y_idx, max_depth
    """
    n     = len(names)
    y_idx = names.index("Y")

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i, label=names[i])
    for i in range(n):
        for j in range(n):
            if adj[i, j] != 0:
                G.add_edge(i, j)

    parents_full = {i: [j for j in range(n) if adj[j, i] != 0] for i in range(n)}
    sources      = {i for i in range(n) if len(parents_full[i]) == 0 and i != y_idx}

    non_y         = [i for i in range(n) if i != y_idx]
    children_no_y = {
        u: [v for v in range(n) if adj[u, v] != 0 and v != y_idx]
        for u in non_y
    }
    parents_no_y  = {v: [] for v in non_y}
    for u, ch in children_no_y.items():
        for v in ch:
            parents_no_y[v].append(u)

    in_deg     = {nd: len(parents_no_y[nd]) for nd in non_y}
    topo_depth = {nd: 0 for nd in non_y}
    queue      = [nd for nd in non_y if in_deg[nd] == 0]
    while queue:
        node = queue.pop(0)
        for child in children_no_y.get(node, []):
            topo_depth[child] = max(topo_depth[child], topo_depth[node] + 1)
            in_deg[child] -= 1
            if in_deg[child] == 0:
                queue.append(child)

    max_depth = max(topo_depth.values()) if topo_depth else 0
    topo_depth[y_idx] = max_depth + 1

    level_groups: dict[int, list[int]] = {}
    for nd, d in topo_depth.items():
        level_groups.setdefault(d, []).append(nd)
    for d in level_groups:
        level_groups[d].sort()

    return G, topo_depth, level_groups, sources, y_idx, max_depth


def make_dag_pos(
    level_groups: dict[int, list[int]],
    max_depth:    int,
) -> dict[int, tuple[float, float]]:
    """Compute a layered (x, y) layout for a DAG."""
    Y_STEP  = 2.5
    H_SCALE = 1.4
    pos: dict[int, tuple[float, float]] = {}
    for d, nodes in level_groups.items():
        n     = len(nodes)
        h_gap = max(0.6, H_SCALE * 22 / max(n, 1))
        y_coord = -(max_depth + 1 - d) * Y_STEP
        for k, node in enumerate(nodes):
            pos[node] = ((k - (n - 1) / 2) * h_gap, y_coord)
    return pos


def _dag_node_color(
    node_idx:   int,
    topo_depth: dict[int, int],
    max_depth:  int,
    y_idx:      int,
) -> str:
    """Colour a DAG node by its proximity to Y."""
    if node_idx == y_idx:
        return "#e74c3c"
    prox = max_depth - topo_depth.get(node_idx, 0)
    if max_depth == 0:
        return PROX_PALETTE[0]
    idx = int(round(prox / max_depth * (len(PROX_PALETTE) - 1)))
    return PROX_PALETTE[min(idx, len(PROX_PALETTE) - 1)]


def draw_dag(
    ax:         Any,
    G:          nx.DiGraph,
    pos:        dict,
    topo_depth: dict[int, int],
    sources:    set[int],
    y_idx:      int,
    max_depth:  int,
    title:      str,
) -> None:
    """Draw a single DAG panel onto a Matplotlib Axes object."""
    nc = [_dag_node_color(nd, topo_depth, max_depth, y_idx) for nd in G.nodes()]
    ns = [1200 if nd == y_idx else (650 if nd in sources else 280) for nd in G.nodes()]
    ec = ["#c0392b" if v == y_idx else "#bdc3c7" for u, v in G.edges()]
    ew = [2.2    if v == y_idx else 0.55         for u, v in G.edges()]

    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=ec, width=ew, alpha=0.5,
        arrows=True, arrowstyle="->", arrowsize=7,
        connectionstyle="arc3,rad=0.05",
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=nc, node_size=ns,
        alpha=0.92, linewidths=0.7, edgecolors="#2c3e50",
    )
    nx.draw_networkx_labels(
        G, pos, {nd: G.nodes[nd]["label"] for nd in G.nodes()},
        ax=ax, font_size=5.8, font_weight="bold",
    )

    n_direct   = sum(1 for u, v in G.edges() if v == y_idx)
    depth_dist = collections.Counter(v for k, v in topo_depth.items() if k != y_idx)
    stats_str  = (
        f"{G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges\n"
        f"{n_direct} direct Y-parents  |  {len(sources)} sources\n"
        + "  ".join(f"D{d}:{depth_dist[d]}" for d in sorted(depth_dist))
    )
    ax.text(
        0.01, 0.01, stats_str, transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")


def draw_dag_highlight(
    ax,
    G,
    pos:            dict,
    topo_depth:     dict,
    sources:        set,
    y_idx:          int,
    max_depth:      int,
    title:          str,
    highlight_name: str,
    names:          list,
) -> None:
    """
    Draw a DAG with one node highlighted; all other nodes and edges are muted.

    The highlighted node is drawn in a vivid colour with a thick border.
    Its incoming edges are drawn in blue (#2166ac) and outgoing edges in orange-red (#d6604d).
    All other nodes and edges are rendered in light grey so the focus node stands out.

    Parameters
    ----------
    ax             : Matplotlib Axes to draw on.
    G              : NetworkX DiGraph (from build_dag_graph).
    pos            : node-position dict (from make_dag_pos).
    topo_depth     : topological depth per node (from build_dag_graph).
    sources        : set of source node indices.
    y_idx          : index of Y (sink) node.
    max_depth      : maximum topological depth.
    title          : axes title string.
    highlight_name : name of the node to highlight, e.g. "X0".
    names          : ordered list of all node names (same order as adjacency matrix).
    """
    from matplotlib.lines import Line2D

    if highlight_name not in names:
        raise ValueError(f"highlight_name '{highlight_name}' not found in names.")
    h_idx = names.index(highlight_name)

    in_edges  = [(u, v) for u, v in G.edges() if v == h_idx]
    out_edges = [(u, v) for u, v in G.edges() if u == h_idx]
    bg_edges  = [(u, v) for u, v in G.edges() if u != h_idx and v != h_idx]

    n_in  = G.in_degree(h_idx)
    n_out = G.out_degree(h_idx)

    # Background edges (muted grey)
    if bg_edges:
        nx.draw_networkx_edges(G, pos, edgelist=bg_edges, ax=ax,
            edge_color="#dcdcdc", width=0.35, alpha=0.45,
            arrows=True, arrowstyle="->", arrowsize=5,
            connectionstyle="arc3,rad=0.05")

    # Incoming edges (blue)
    if in_edges:
        nx.draw_networkx_edges(G, pos, edgelist=in_edges, ax=ax,
            edge_color="#2166ac", width=2.2, alpha=0.9,
            arrows=True, arrowstyle="->", arrowsize=14,
            connectionstyle="arc3,rad=0.05")

    # Outgoing edges (orange-red)
    if out_edges:
        nx.draw_networkx_edges(G, pos, edgelist=out_edges, ax=ax,
            edge_color="#d6604d", width=2.2, alpha=0.9,
            arrows=True, arrowstyle="->", arrowsize=14,
            connectionstyle="arc3,rad=0.05")

    # Node colors / sizes / borders
    nc, ns, ne, nlw = [], [], [], []
    for nd in G.nodes():
        if nd == h_idx:
            nc.append("#e74c3c" if nd == y_idx else "#f39c12")
            ns.append(1500)
            ne.append("#922b21" if nd == y_idx else "#b7770d")
            nlw.append(2.5)
        elif nd == y_idx:
            nc.append("#f1948a")
            ns.append(700)
            ne.append("#c0392b")
            nlw.append(1.0)
        else:
            nc.append("#eeeeee")
            ns.append(160)
            ne.append("#cccccc")
            nlw.append(0.4)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=ns,
        alpha=0.95, linewidths=nlw, edgecolors=ne)

    # Labels: background grey/small; neighbours bolder; highlight bold/large
    bg_lbl = {nd: G.nodes[nd]["label"] for nd in G.nodes() if nd != h_idx}
    if bg_lbl:
        nx.draw_networkx_labels(G, pos, bg_lbl, ax=ax,
            font_size=5.0, font_color="#aaaaaa")

    nbr_lbl = {}
    for u, _ in in_edges:
        nbr_lbl[u] = G.nodes[u]["label"]
    for _, v in out_edges:
        nbr_lbl[v] = G.nodes[v]["label"]
    if nbr_lbl:
        nx.draw_networkx_labels(G, pos, nbr_lbl, ax=ax,
            font_size=6.5, font_color="#2c3e50", font_weight="bold")

    nx.draw_networkx_labels(G, pos, {h_idx: G.nodes[h_idx]["label"]}, ax=ax,
        font_size=9.0, font_color="#1a1a1a", font_weight="bold")

    # Stats annotation (bottom-left)
    depth_h  = topo_depth.get(h_idx, "?")
    src_note = "  (source — no parents)" if h_idx in sources else ""
    stats_str = (
        f"{highlight_name}  ·  depth {depth_h}{src_note}\n"
        f"in-edges : {n_in}   out-edges : {n_out}\n"
        f"graph    : {G.number_of_nodes()} nodes · {G.number_of_edges()} edges"
    )
    ax.text(0.01, 0.01, stats_str, transform=ax.transAxes,
        fontsize=8.0, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.88, ec="#cccccc"))

    # Edge legend (upper-right)
    legend_handles = [
        Line2D([0], [0], color="#2166ac", linewidth=2.0, label=f"Incoming ({n_in})"),
        Line2D([0], [0], color="#d6604d", linewidth=2.0, label=f"Outgoing ({n_out})"),
        Line2D([0], [0], color="#dcdcdc", linewidth=1.0, label="Other edges"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.0, framealpha=0.85)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.axis("off")
    ax.set_facecolor("#f8f9fa")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plot builders  (pure functions — data in, Figure out)
# ─────────────────────────────────────────────────────────────────────────────

def make_shap_scatter_figure(
    methods:        list[str],
    col_labels:     list[str],
    feature_idx:    list[int],
    feature_labels: list[str],
    x_test:         np.ndarray,
    shap_data:      dict[str, np.ndarray],
    title:          str,
    colors:         dict | None = None,
) -> go.Figure:
    """
    Build an (n_features × n_methods) scatter grid.
    Each cell: feature value (x) vs SHAP value (y). Y-axis shared within rows.
    """
    n_rows = len(feature_idx)
    n_cols = len(methods)

    vertical_spacing   = max(0.005, min(0.06, 0.25 / max(n_rows - 1, 1)))
    horizontal_spacing = 0.06
    cell_h = 220  # fixed per-row height so fewer rows don't stretch the subplots

    subplot_titles = [
        col_labels[ci] if ri == 0 else ""
        for ri in range(n_rows)
        for ci in range(n_cols)
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        shared_yaxes="rows",
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    for ann in fig.layout.annotations:
        if ann.text:
            ann.update(font=dict(size=11, color="#2c3e50"))

    for ri, (fi, feat) in enumerate(zip(feature_idx, feature_labels)):
        x_vals = x_test[:, fi]
        for ci, method in enumerate(methods):
            if method not in shap_data:
                continue
            shap_vals = shap_data[method][:, fi]

            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=shap_vals, mode="markers",
                    marker=dict(
                        size=4, color=x_vals, colorscale="Viridis",
                        showscale=False, opacity=0.7, line=dict(width=0),
                    ),
                    name=method, showlegend=False,
                    hovertemplate=(
                        f"<b>{feat}</b> — {method}<br>"
                        "x = %{x:.3f}<br>SHAP = %{y:.4f}<extra></extra>"
                    ),
                ),
                row=ri + 1, col=ci + 1,
            )
            x_range = [float(x_vals.min()), float(x_vals.max())]
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=[0, 0], mode="lines",
                    line=dict(color="gray", width=0.8, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ),
                row=ri + 1, col=ci + 1,
            )

        fig.update_yaxes(
            title_text=feat, title_font=dict(size=9),
            title_standoff=4, row=ri + 1, col=1,
        )
        for ci in range(1, n_cols + 1):
            fig.update_xaxes(
                showticklabels=(ri == n_rows - 1),
                tickfont=dict(size=8), row=ri + 1, col=ci,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=cell_h * n_rows + 80,
        width=240 * n_cols + 120,
        margin=dict(l=90, r=20, t=80, b=40),
    )
    return fig


def make_instance_shap_figure(
    methods:        list[str],
    col_labels:     list[str],
    feature_idx:    list[int],
    feature_labels: list[str],
    instance_idx:   list[int],
    shap_data:      dict[str, np.ndarray],
    title:          str,
) -> go.Figure:
    """
    Build an (n_instances × n_methods) grid of horizontal bar charts.
    Blue = positive SHAP, red = negative SHAP.
    """
    n_rows  = len(instance_idx)
    n_cols  = len(methods)
    n_feats = len(feature_idx)   # noqa: F841

    cell_h             = max(160, 22 * n_feats + 40)
    vertical_spacing   = max(0.005, min(0.04, 0.15 / max(n_rows - 1, 1)))
    horizontal_spacing = 0.06

    subplot_titles = [
        col_labels[ci] if ri == 0 else ""
        for ri in range(n_rows) for ci in range(n_cols)
    ]

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=False, shared_yaxes=False,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    for ann in fig.layout.annotations:
        if ann.text:
            ann.update(font=dict(size=11, color="#2c3e50"))

    POS_COLOR     = "#2196F3"
    NEG_COLOR     = "#EF5350"
    feat_labels_r = feature_labels[::-1]
    feat_idx_r    = feature_idx[::-1]

    # Collect per-subplot x-ranges so we can add label padding afterwards
    subplot_xranges: dict[tuple[int, int], tuple[float, float]] = {}

    for ri, inst in enumerate(instance_idx):
        for ci, method in enumerate(methods):
            if method not in shap_data:
                continue
            sv     = shap_data[method][inst]
            vals   = np.array([sv[fi] for fi in feat_idx_r])
            colors = [POS_COLOR if v >= 0 else NEG_COLOR for v in vals]
            fig.add_trace(
                go.Bar(
                    x=vals, y=feat_labels_r, orientation="h",
                    marker=dict(color=colors, line=dict(width=0)),
                    showlegend=False,
                    text=[f"{v:.3f}" for v in vals],
                    textposition="outside",
                    textfont=dict(size=7),
                    constraintext="none",
                    hovertemplate=(
                        f"<b>Instance {inst}</b><br>%{{y}}: %{{x:.4f}}<extra></extra>"
                    ),
                ),
                row=ri + 1, col=ci + 1,
            )
            fig.add_vline(
                x=0,
                line=dict(color="rgba(80,80,80,0.40)", width=1, dash="dot"),
                row=ri + 1, col=ci + 1,
            )
            subplot_xranges[(ri, ci)] = (float(vals.min()), float(vals.max()))

        fig.update_yaxes(
            title_text=f"#{inst}", title_font=dict(size=9),
            title_standoff=3, showticklabels=True,
            tickfont=dict(size=8), row=ri + 1, col=1,
        )
        for ci in range(2, n_cols + 1):
            fig.update_yaxes(showticklabels=False, row=ri + 1, col=ci)
        for ci in range(1, n_cols + 1):
            fig.update_xaxes(
                showticklabels=(ri == n_rows - 1),
                tickfont=dict(size=8), row=ri + 1, col=ci,
            )

    # Apply padded x-axis ranges so outside text labels are never clipped.
    # Padding = 30 % of the span on each side, with a minimum absolute buffer.
    for (ri, ci), (xmin, xmax) in subplot_xranges.items():
        span = xmax - xmin if xmax != xmin else max(abs(xmax), 1e-6)
        pad  = max(0.30 * span, 0.05 * max(abs(xmin), abs(xmax), 1e-6))
        fig.update_xaxes(
            range=[xmin - pad, xmax + pad],
            row=ri + 1, col=ci + 1,
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=cell_h * n_rows + 80,
        width=290 * n_cols + 140,
        margin=dict(l=90, r=20, t=80, b=40),
        bargap=0.10,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def flow_graph_stats(adj_list: list, n_x: int) -> tuple[int, int]:
    """
    Return (total_edges, n_source_nodes) for a (n_x+1)×(n_x+1) adjacency.
    Source = in-degree 0 in the X-only subgraph AND reachable from Y via backward BFS.
    """
    adj   = np.array(adj_list)
    total = int(np.sum(adj != 0))
    n     = adj.shape[0]
    parents = {i: np.where(adj[:, i] != 0)[0].tolist() for i in range(n)}
    reachable: set[int] = set()
    queue = [n - 1]
    while queue:
        node = queue.pop()
        if node not in reachable:
            reachable.add(node)
            queue.extend(parents[node])
    x_adj   = adj[:n_x, :n_x]
    in_deg  = np.sum(x_adj != 0, axis=0)
    sources = [i for i in range(n_x) if in_deg[i] == 0 and i in reachable]
    return total, len(sources)


def parse_pipeline_timing(
    logs_dir: Path,
    dataset:  str,
    verbose:  bool = True,
) -> tuple[pd.DataFrame | None, str]:
    """
    Parse the latest pipeline log file and return per-method timing info.

    Returns
    -------
    timing_df  : DataFrame with per-method timing rows, or None if no log found.
    total_str  : Human-readable total execution time string.
    """
    log_files = sorted(logs_dir.glob("pipeline_*.log"))
    if not log_files:
        if verbose:
            print("No log files found in", logs_dir)
        return None, "—"

    log_file = log_files[-1]
    if verbose:
        print(f"Log file: {log_file.name}")

    ts_re   = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (.*)$")
    prog_re = re.compile(r"\[Progress:\s*(\d+)/(\d+)\]\s+(.+)")
    flow_re = re.compile(r"ShapleyFlowWrapper:.*?(\d+) edges,\s*(\d+) sources")

    events: list[tuple] = []
    with open(log_file) as f:
        for line in f:
            m = ts_re.match(line.strip())
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                events.append((ts, m.group(2).strip()))

    flow_stats: dict[str, tuple[int, int]] = {}
    current_key = None
    for ts, content in events:
        pm = prog_re.search(content)
        if pm:
            current_key = pm.group(3).strip()
        fm = flow_re.search(content)
        if fm and current_key and current_key not in flow_stats:
            flow_stats[current_key] = (int(fm.group(1)), int(fm.group(2)))

    total_min    = None
    pipeline_end = None
    for ts, content in events:
        m = re.search(r"Total execution time:\s*([\d.]+)\s*minutes", content)
        if m:
            total_min    = float(m.group(1))
            pipeline_end = ts
    if pipeline_end is None and events:
        pipeline_end = events[-1][0]

    prog_evts = []
    for ts, content in events:
        m = prog_re.search(content)
        if m:
            prog_evts.append({"ts": ts, "idx": int(m.group(1)), "name": m.group(3).strip()})

    if not prog_evts:
        return None, "—"

    rows = []
    for i, pe in enumerate(prog_evts):
        end_ts     = prog_evts[i + 1]["ts"] if i + 1 < len(prog_evts) else pipeline_end
        dur_s      = max(0.0, (end_ts - pe["ts"]).total_seconds()) if end_ts else 0.0
        dur_min    = dur_s / 60
        n_inst     = 100

        raw_name   = pe["name"]
        clean_name = re.sub(r"\s*\(SLOW[^)]*\)", "", raw_name).strip()
        is_pc      = "PC"     in raw_name.upper()
        is_lingam  = "LINGAM" in raw_name.upper()

        if raw_name in flow_stats:
            edges, srcs = flow_stats[raw_name]
            graph_str = f"{edges} edges, {srcs} sources"
        elif is_pc:
            graph_str = "PC graph"
        elif is_lingam:
            graph_str = "LiNGAM graph"
        else:
            graph_str = "—"

        if dur_s < 5:
            dur_str = "< 5 s (skipped)"
            per_str = "—"
        elif dur_min >= 60:
            dur_str = f"{dur_min/60:.1f} h  ({dur_min:.0f} min)"
            per_str = f"~{dur_s/n_inst:.0f} s"
        elif dur_min >= 1:
            dur_str = f"{dur_min:.1f} min"
            per_str = f"~{dur_s/n_inst:.0f} s"
        else:
            dur_str = f"{dur_s:.0f} s"
            per_str = f"~{dur_s/n_inst:.1f} s"

        rows.append({
            "#":             pe["idx"],
            "Method":        clean_name,
            "Start":         pe["ts"].strftime("%H:%M:%S"),
            "End":           end_ts.strftime("%H:%M:%S") if end_ts else "—",
            "Duration":      dur_str,
            "Per Instance":  per_str,
            "Graph Details": graph_str,
        })

    total_str = (
        f"{total_min:.1f} min  ({total_min/60:.2f} h)" if total_min else "—"
    )
    return pd.DataFrame(rows), total_str


# ─────────────────────────────────────────────────────────────────────────────
# 7. Comparison plot helpers (Scratch | PC | LiNGAM | True DAG)
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_scatter_pc_vs_lingam(
    shap_data: dict,
    feature_names: list,
    scratch_rank: np.ndarray,
    top_features: list,
    x_test: np.ndarray,
    pc_edges: int,
    lingam_edges: int,
    dataset: str,
    n_top_features: int,
    method: str = "Asymmetric",
    features: list | None = None,
    colors: dict | None = None,
):
    """
    SHAP scatter plot (feature value vs SHAP) for ONE method across all graph sources.

    Produces a grid where:
      • Rows    : selected features
      • Columns : Traditional (no graph) | <method> (PC) | <method> (LiNGAM) | <method> (True DAG)

    Parameters
    ----------
    shap_data : dict
        Mapping of method key → SHAP array (n_instances × n_features).
    feature_names : list of str
        Full list of feature names.
    scratch_rank : np.ndarray
        Feature indices sorted by descending mean |SHAP| for Scratch.
    top_features : list of str
        Pre-computed top-N feature names (used as default when features=None).
    x_test : np.ndarray
        Feature matrix for test instances (n_instances × n_features).
    pc_edges : int
        Number of edges in the PC discovered graph.
    lingam_edges : int
        Number of edges in the LiNGAM discovered graph.
    dataset : str
        Dataset name used in the plot title.
    n_top_features : int
        Number of top features to show when features=None.
    method : str
        Which Shapley method to compare. One of "Asymmetric", "Causal", "Flow".
    features : list of str, optional
        Specific feature names to display. Defaults to top-N by Scratch.
    colors : dict, optional
        Method colour mapping. Defaults to DEFAULT_METHOD_COLORS.

    Examples
    --------
    >>> plot_shap_scatter_pc_vs_lingam(shap_data, feature_names, scratch_rank,
    ...     top_features, x_test, pc_edges, lingam_edges, dataset, n_top_features)
    >>> plot_shap_scatter_pc_vs_lingam(..., method="Causal", features=["X33", "X7"])
    """
    valid_methods = ["Asymmetric", "Causal", "Flow"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    pc_key     = f"{method} (PC)"
    lingam_key = f"{method} (LiNGAM)"
    true_key   = f"{method} (True)"

    for key in [pc_key, lingam_key]:
        if key not in shap_data:
            raise KeyError(f"'{key}' not found in shap_data. Available: {list(shap_data)}")
    if "Scratch" not in shap_data:
        raise KeyError("'Scratch' not found in shap_data.")

    methods_list = ["Scratch", pc_key, lingam_key]
    col_labels   = ["Traditional (no graph)", f"{method} (PC)", f"{method} (LiNGAM)"]
    if true_key in shap_data:
        methods_list.append(true_key)
        col_labels.append(f"{method} (True DAG)")

    feat_idx, feat_labels = resolve_features(feature_names, scratch_rank, top_features, features)
    title_tag = f"Top {n_top_features}" if features is None else f"{len(feat_labels)}"

    fig = make_shap_scatter_figure(
        methods        = methods_list,
        col_labels     = col_labels,
        feature_idx    = feat_idx,
        feature_labels = feat_labels,
        x_test         = x_test,
        shap_data      = shap_data,
        title          = (
            f"SHAP Scatter — {method}: Traditional | PC ({pc_edges} edges) | "
            f"LiNGAM ({lingam_edges} edges) | True DAG — "
            f"{title_tag} Features — {dataset}"
        ),
        colors         = colors or DEFAULT_METHOD_COLORS,
    )
    fig.show()
    return fig


def plot_instance_shap_pc_vs_lingam(
    shap_data: dict,
    feature_names: list,
    scratch_rank: np.ndarray,
    top_features: list,
    instance_idx: list,
    pc_edges: int,
    lingam_edges: int,
    dataset: str,
    n_top_features: int,
    method: str = "Asymmetric",
    features: list | None = None,
):
    """
    Instance-level SHAP bar charts for ONE method, comparing all graph sources.

    Produces a figure with four columns per instance:
      • Col 1 : Traditional (no graph, graph-free baseline)
      • Col 2 : <method> (PC)
      • Col 3 : <method> (LiNGAM)
      • Col 4 : <method> (True DAG)

    Parameters
    ----------
    shap_data : dict
        Mapping of method key → SHAP array (n_instances × n_features).
    feature_names : list of str
        Full list of feature names.
    scratch_rank : np.ndarray
        Feature indices sorted by descending mean |SHAP| for Scratch.
    top_features : list of str
        Pre-computed top-N feature names (used as default when features=None).
    instance_idx : list of int
        Row indices into the SHAP arrays to display.
    pc_edges : int
        Number of edges in the PC discovered graph.
    lingam_edges : int
        Number of edges in the LiNGAM discovered graph.
    dataset : str
        Dataset name used in the plot title.
    n_top_features : int
        Number of top features to show when features=None.
    method : str
        Which Shapley method to compare. One of "Asymmetric", "Causal", "Flow".
    features : list of str, optional
        Specific feature names to display. Defaults to top-N by Scratch.

    Examples
    --------
    >>> plot_instance_shap_pc_vs_lingam(shap_data, feature_names, scratch_rank,
    ...     top_features, instance_idx, pc_edges, lingam_edges, dataset, n_top_features)
    >>> plot_instance_shap_pc_vs_lingam(..., method="Causal", features=["X33", "X7"])
    """
    valid_methods = ["Asymmetric", "Causal", "Flow"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")

    pc_key     = f"{method} (PC)"
    lingam_key = f"{method} (LiNGAM)"
    true_key   = f"{method} (True)"

    for key in [pc_key, lingam_key]:
        if key not in shap_data:
            raise KeyError(f"'{key}' not found in shap_data. Available: {list(shap_data)}")
    if "Scratch" not in shap_data:
        raise KeyError("'Scratch' not found in shap_data.")

    methods_list = ["Scratch", pc_key, lingam_key]
    col_labels   = ["Traditional (no graph)", f"{method} (PC)", f"{method} (LiNGAM)"]
    if true_key in shap_data:
        methods_list.append(true_key)
        col_labels.append(f"{method} (True DAG)")

    feat_idx, feat_labels = resolve_features(feature_names, scratch_rank, top_features, features)
    title_tag = f"Top {n_top_features}" if features is None else f"{len(feat_labels)}"

    fig = make_instance_shap_figure(
        methods        = methods_list,
        col_labels     = col_labels,
        feature_idx    = feat_idx,
        feature_labels = feat_labels,
        instance_idx   = instance_idx,
        shap_data      = shap_data,
        title          = (
            f"Instance-Level SHAP — {method}: Traditional | PC ({pc_edges} edges) | "
            f"LiNGAM ({lingam_edges} edges) | True DAG — "
            f"{title_tag} Features — {dataset}"
        ),
    )
    fig.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 8. Stand-alone plot functions (save to plots/ folder + show)
#    Each function accepts an optional ``save_dir`` (Path or str).
#    If supplied the figure is saved there; otherwise nothing is written.
# ─────────────────────────────────────────────────────────────────────────────

def _save_fig(fig, save_dir, filename):
    """Save fig (plotly or matplotlib) to save_dir/filename if save_dir is set."""
    if save_dir is None:
        return
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / filename
    if hasattr(fig, "write_image"):          # Plotly figure
        try:
            fig.write_image(str(dest))
        except Exception:
            fig.write_html(str(dest.with_suffix(".html")))
    else:                                    # Matplotlib figure
        fig.savefig(str(dest), dpi=150, bbox_inches="tight")
    print(f"  ✅  Saved → {dest}")


def plot_timing_table(
    timing_df: "pd.DataFrame",
    total_time: str,
    dataset: str,
    save_dir=None,
) -> "go.Figure":
    """
    Plotly table of pipeline timing summary.

    Parameters
    ----------
    timing_df  : DataFrame from ``parse_pipeline_timing``.
    total_time : Human-readable total time string.
    dataset    : Dataset name for the title.
    save_dir   : Optional directory to save the figure.

    Returns
    -------
    go.Figure
    """
    header_vals = list(timing_df.columns)
    cell_vals   = [timing_df[c].tolist() for c in timing_df.columns]
    n = len(timing_df)
    row_colors = [["#f0f4f8" if i % 2 == 0 else "white" for i in range(n)]] * len(header_vals)

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{v}</b>" for v in header_vals],
            fill_color="#2c3e50", font=dict(color="white", size=12),
            align="left", height=30,
        ),
        cells=dict(
            values=cell_vals,
            fill_color=row_colors,
            align="left", height=26,
            font=dict(size=11),
        ),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Pipeline Timing Summary — {dataset}<br>"
                f"<sup>Total execution time: {total_time}</sup>"
            ),
            font=dict(size=14),
        ),
        margin=dict(l=10, r=10, t=60, b=10),
        height=90 + 30 + n * 50,
    )
    fig.show()
    _save_fig(fig, save_dir, f"timing_table_{dataset}.png")
    return fig


def plot_feature_importance_bar(
    shap_data:      dict,
    feature_names:  list,
    scratch_rank:   "np.ndarray",
    top_features:   list,
    mean_abs:       dict,
    pc_edges:       int,
    pc_sources:     int,
    lingam_edges:   int,
    lingam_sources: int,
    method_colors:  dict,
    dataset:        str,
    n_top_features: int,
    features:       list | None = None,
    save_dir=None,
) -> "go.Figure":
    """
    Grouped bar chart of mean |SHAP| per feature × method, split by graph type.

    Parameters
    ----------
    shap_data      : method key → SHAP array.
    feature_names  : ordered list of feature name strings.
    scratch_rank   : top-N feature indices sorted by Scratch importance.
    top_features   : corresponding feature name list.
    mean_abs       : method key → mean |SHAP| array (pre-computed).
    pc_edges       : number of edges in the PC graph.
    pc_sources     : number of source nodes in the PC graph.
    lingam_edges   : number of edges in the LiNGAM graph.
    lingam_sources : number of source nodes in the LiNGAM graph.
    method_colors  : dict mapping method short-name → hex colour.
    dataset        : dataset name for the title.
    n_top_features : number of top features shown when ``features`` is None.
    features       : optional explicit list of feature names to display.
    save_dir       : optional directory to save the figure.
    """
    feat_idx, feat_labels = resolve_features(feature_names, scratch_rank, top_features, features)
    title_tag = f"Top {n_top_features}" if features is None else f"{len(feat_labels)}"

    pc_methods     = ["Scratch", "Asymmetric (PC)",     "Causal (PC)",     "Flow (PC)"]
    lingam_methods = ["Scratch", "Asymmetric (LiNGAM)", "Causal (LiNGAM)", "Flow (LiNGAM)"]

    rows = []
    for method in pc_methods:
        src_key = "Scratch" if method == "Scratch" else method
        if src_key not in mean_abs:
            continue
        label = method.replace(" (PC)", "")
        for fi in feat_idx:
            rows.append(dict(
                Feature    = feature_names[fi],
                Method     = label,
                Graph      = f"PC  ({pc_edges} edges, {pc_sources} sources)",
                Importance = float(mean_abs[src_key][fi]),
            ))
    for method in lingam_methods:
        src_key = "Scratch" if method == "Scratch" else method
        if src_key not in mean_abs:
            continue
        label = method.replace(" (LiNGAM)", "")
        for fi in feat_idx:
            rows.append(dict(
                Feature    = feature_names[fi],
                Method     = label,
                Graph      = f"LiNGAM  ({lingam_edges} edges, {lingam_sources} sources)",
                Importance = float(mean_abs[src_key][fi]),
            ))

    df_imp = pd.DataFrame(rows)
    df_imp["Feature"] = pd.Categorical(df_imp["Feature"], categories=feat_labels, ordered=True)

    fig = px.bar(
        df_imp,
        x="Feature", y="Importance", color="Method",
        barmode="group",
        facet_col="Graph",
        facet_col_spacing=0.06,
        category_orders={
            "Feature": feat_labels,
            "Method":  ["Scratch", "Asymmetric", "Causal", "Flow"],
        },
        color_discrete_map={
            "Scratch":    method_colors.get("Scratch",     "#636363"),
            "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
            "Causal":     method_colors.get("Causal",      "#4dac26"),
            "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
        },
        labels={"Importance": "Mean |SHAP|", "Feature": ""},
        title=(
            f"Global Feature Importance — {title_tag} Features "
            f"(sorted by Traditional) — {dataset}"
        ),
        height=480,
        width=1300,
    )
    fig.update_yaxes(matches="y")
    fig.update_layout(legend_title_text="Method", margin=dict(t=80, b=60))
    fig.show()
    _save_fig(fig, save_dir, f"feature_importance_{dataset}.png")
    return fig


def plot_dag_comparison(
    G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc,
    G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
    dataset: str,
    prox_palette: list,
    save_dir=None,
):
    """
    Draw discovered PC and LiNGAM DAGs side by side.

    Parameters
    ----------
    G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc         : PC graph from build_dag_graph / make_dag_pos.
    G_lingam, pos_lingam, td_lingam, src_lingam, ...  : LiNGAM graph from same.
    dataset     : dataset name for the title.
    prox_palette: list of hex colours for depth colouring.
    save_dir    : optional directory to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(26, 14))
    fig.patch.set_facecolor("#f0f0f0")

    draw_dag(axes[0], G_pc,     pos_pc,     td_pc,     src_pc,     y_pc,     ml_pc,
             f"PC Discovered — {dataset}")
    draw_dag(axes[1], G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
             f"LiNGAM Discovered — {dataset}")

    max_lv_disc    = max(ml_pc, ml_lingam)
    legend_entries = [mpatches.Patch(color="#e74c3c", label="Y (sink)")]
    for prox in range(min(len(prox_palette), max_lv_disc + 1)):
        idx = int(round(prox / max(max_lv_disc, 1) * (len(prox_palette) - 1)))
        col = prox_palette[min(idx, len(prox_palette) - 1)]
        d   = max_lv_disc - prox
        lbl = (f"Depth {d} — sources (no parents)" if prox == max_lv_disc
               else f"Depth {d}" + (" ← closest to Y" if prox == 0 else ""))
        legend_entries.append(mpatches.Patch(color=col, label=lbl))
    legend_entries += [mpatches.Patch(color="#c0392b", label="Edges → Y"),
                       mpatches.Patch(color="#bdc3c7", label="Other edges")]
    fig.legend(handles=legend_entries, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Discovered Causal Graphs — {dataset}\n"
        f"Node colour = topological depth from sources (Y excluded as intermediate)",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    _save_fig(fig, save_dir, f"dag_comparison_{dataset}.png")
    plt.show()
    return fig


def plot_true_dag(
    G_true, pos_true, td_true, src_true, y_true, ml_true,
    dataset: str,
    prox_palette: list,
    save_dir=None,
):
    """
    Draw the ground-truth causal DAG as a standalone panel.

    Parameters
    ----------
    G_true, pos_true, td_true, src_true, y_true, ml_true : True graph.
    dataset      : dataset name for the title.
    prox_palette : list of hex colours for depth colouring.
    save_dir     : optional directory to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(18, 14))
    fig.patch.set_facecolor("#f0f0f0")

    draw_dag(ax, G_true, pos_true, td_true, src_true, y_true, ml_true,
             f"True Causal Graph — {dataset}")

    legend_entries = [mpatches.Patch(color="#e74c3c", label="Y (sink)")]
    for prox in range(min(len(prox_palette), ml_true + 1)):
        idx = int(round(prox / max(ml_true, 1) * (len(prox_palette) - 1)))
        col = prox_palette[min(idx, len(prox_palette) - 1)]
        d   = ml_true - prox
        lbl = (f"Depth {d} — sources (no parents)" if prox == ml_true
               else f"Depth {d}" + (" ← closest to Y" if prox == 0 else ""))
        legend_entries.append(mpatches.Patch(color=col, label=lbl))
    legend_entries += [mpatches.Patch(color="#c0392b", label="Edges → Y"),
                       mpatches.Patch(color="#bdc3c7", label="Other edges")]
    fig.legend(handles=legend_entries, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Ground-Truth Causal Graph — {dataset}\n"
        f"Node colour = topological depth from sources (Y excluded as intermediate)\n"
        f"{G_true.number_of_nodes()} nodes  ·  {G_true.number_of_edges()} edges  ·  "
        f"{sum(1 for u,v in G_true.edges() if v==y_true)} direct Y-parents  ·  "
        f"{len(src_true)} sources  ·  max depth {ml_true}",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    _save_fig(fig, save_dir, f"dag_true_{dataset}.png")
    plt.show()
    return fig


def plot_dag_highlight(
    G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc,
    G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
    feature_name: str,
    feature_names_list: list,
    dataset: str,
    figsize=(26, 14),
    save_dir=None,
):
    """
    Draw PC and LiNGAM DAGs side by side with one feature node highlighted.

    Parameters
    ----------
    feature_name       : Name of the node to highlight, e.g. ``"X5"``.
    feature_names_list : Ordered list of all node names.
    dataset            : Dataset name for the title.
    figsize            : Figure size.
    save_dir           : Optional directory to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor("#f0f0f0")

    draw_dag_highlight(
        axes[0], G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc,
        title=f"PC Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    draw_dag_highlight(
        axes[1], G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
        title=f"LiNGAM Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    fig.suptitle(
        f"DAG Node Highlight — {feature_name} — {dataset}\n"
        f"Blue = incoming edges (parents of {feature_name})  ·  "
        f"Orange-red = outgoing edges (children of {feature_name})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_fig(fig, save_dir, f"dag_highlight_{feature_name}_{dataset}.png")
    plt.show()
    return fig


def plot_dag_highlight_true(
    G_true, pos_true, td_true, src_true, y_true, ml_true,
    feature_name: str,
    feature_names_list: list,
    dataset: str,
    figsize=(16, 14),
    save_dir=None,
):
    """
    Draw the True (ground-truth) causal DAG with one feature node highlighted.

    Parameters
    ----------
    feature_name       : Name of the node to highlight.
    feature_names_list : Ordered list of all node names.
    dataset            : Dataset name for the title.
    figsize            : Figure size.
    save_dir           : Optional directory to save the figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor("#f0f0f0")

    draw_dag_highlight(
        ax, G_true, pos_true, td_true, src_true, y_true, ml_true,
        title=f"True Causal Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    fig.suptitle(
        f"True DAG Node Highlight — {feature_name} — {dataset}\n"
        f"Blue = incoming edges (parents of {feature_name})  ·  "
        f"Orange-red = outgoing edges (children of {feature_name})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_fig(fig, save_dir, f"dag_highlight_true_{feature_name}_{dataset}.png")
    plt.show()
    return fig


def plot_dag_highlight_3panel(
    G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc,
    G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
    G_true, pos_true, td_true, src_true, y_true, ml_true,
    feature_name: str,
    feature_names_list: list,
    dataset: str,
    figsize=(26, 10),
    save_dir=None,
):
    """
    Draw PC, LiNGAM, and True DAGs side by side with one feature node highlighted.

    Parameters
    ----------
    feature_name       : Name of the node to highlight, e.g. ``"pip3"``.
    feature_names_list : Ordered list of all node names.
    dataset            : Dataset name for the title.
    figsize            : Figure size.
    save_dir           : Optional directory to save the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor("#f0f0f0")

    draw_dag_highlight(
        axes[0], G_pc, pos_pc, td_pc, src_pc, y_pc, ml_pc,
        title=f"PC Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    draw_dag_highlight(
        axes[1], G_lingam, pos_lingam, td_lingam, src_lingam, y_lingam, ml_lingam,
        title=f"LiNGAM Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    draw_dag_highlight(
        axes[2], G_true, pos_true, td_true, src_true, y_true, ml_true,
        title=f"True Graph — {dataset}  ·  highlight: {feature_name}",
        highlight_name=feature_name,
        names=feature_names_list,
    )
    fig.suptitle(
        f"DAG Node Highlight — {feature_name} — {dataset}\n"
        f"Blue = incoming edges (parents of {feature_name})  ·  "
        f"Orange-red = outgoing edges (children of {feature_name})",
        fontsize=13, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    _save_fig(fig, save_dir, f"dag_highlight_3panel_{feature_name}_{dataset}.png")
    plt.show()
    return fig


def plot_gss_sss_scatter(
    df_gss:        "pd.DataFrame",
    df_sss:        "pd.DataFrame",
    method_colors: dict,
    dataset:       str,
    height:        int  = 420,
    width:         int  = 560,
    save_dir=None,
) -> "go.Figure":
    """
    Scatter plot of |GSS| (x) vs 1 − SSS (y) per method.

    Both axes are instability measures (≥ 0); lower-left = most stable.

    Parameters
    ----------
    df_gss        : DataFrame with columns [Method, MeanGSS].
    df_sss        : DataFrame with columns [Method, MeanSSS].
    method_colors : method name → hex colour.
    dataset       : dataset label for the title.
    height, width : figure dimensions.
    save_dir      : optional directory to save the figure.

    Returns
    -------
    go.Figure
    """
    df_cross = df_gss[["Method", "MeanGSS"]].merge(
        df_sss[["Method", "MeanSSS"]], on="Method"
    )
    df_cross["SignFlipRate"] = 1 - df_cross["MeanSSS"]

    fig = go.Figure()
    for _, row in df_cross.iterrows():
        meth = row["Method"]
        fig.add_trace(go.Scatter(
            x    = [row["MeanGSS"]],
            y    = [row["SignFlipRate"]],
            mode = "markers+text",
            name = meth,
            text = [meth],
            textposition = "top center",
            marker = dict(
                color  = method_colors.get(meth, "#888888"),
                symbol = "circle",
                size   = 14,
                line   = dict(width=1.2, color="white"),
            ),
            hovertemplate=(
                f"<b>{meth}</b><br>"
                "|GSS|: %{x:.4f}<br>"
                "1−SSS: %{y:.3f}<extra></extra>"
            ),
        ))

    x_max = df_cross["MeanGSS"].max() * 1.35
    y_max = df_cross["SignFlipRate"].max() * 1.35

    fig.update_layout(
        title=dict(
            text=(
                f"|GSS| vs Sign Disagreement Rate (1 − SSS) — {dataset}<br>"
                "<sup>Lower-left = most stable; both axes are instability measures (≥ 0)</sup>"
            ),
            font=dict(size=12),
        ),
        xaxis=dict(
            title="|GSS| — Absolute Magnitude Difference (PC vs LiNGAM)",
            range=[0, x_max],
            zeroline=False,
        ),
        yaxis=dict(
            title="1 − SSS — Sign Disagreement Rate",
            range=[0, y_max],
            zeroline=False,
        ),
        height=height, width=width,
        showlegend=False,
        margin=dict(l=70, r=30, t=80, b=60),
    )
    fig.show()
    _save_fig(fig, save_dir, f"gss_sss_scatter_{dataset}.png")
    return fig


def plot_tga_sa_scatter(
    df_sa:         "pd.DataFrame",
    df_tga:        "pd.DataFrame",
    method_colors: dict,
    dataset:       str,
    reference:     str = "True",
    height:        int = 480,
    width:         int = 660,
    save_dir=None,
) -> "go.Figure":
    """
    Scatter plot of TGA (x) vs 1 − Sign Alignment (y) per method and graph.

    Parameters
    ----------
    df_sa         : DataFrame with columns [Dataset, Method, Graph, MeanSignAlign].
    df_tga        : DataFrame with columns [Dataset, Method, Graph, MeanTGA].
    method_colors : method name → hex colour.
    dataset       : dataset label for the title.
    reference     : "True" or "Traditional" — determines title and axis labels.
    height, width : figure dimensions.
    save_dir      : optional directory to save the figure.

    Returns
    -------
    go.Figure
    """
    GRAPH_MARKERS = {"PC": "circle", "LiNGAM": "diamond", "True": "square"}
    
    # Merge dataframes - handle both single and multi-dataset cases
    has_dataset = "Dataset" in df_sa.columns
    if has_dataset:
        merge_cols = ["Dataset", "Method", "Graph"]
        sa_cols = ["Dataset", "Method", "Graph", "MeanSignAlign"]
        tga_cols = ["Dataset", "Method", "Graph", "MeanTGA"]
    else:
        merge_cols = ["Method", "Graph"]
        sa_cols = ["Method", "Graph", "MeanSignAlign"]
        tga_cols = ["Method", "Graph", "MeanTGA"]
    
    df_summary = df_sa[sa_cols].merge(df_tga[tga_cols], on=merge_cols)
    
    fig = go.Figure()
    SEEN_METHODS = set()
    
    for _, row in df_summary.iterrows():
        meth = row["Method"]
        g    = row["Graph"]
        ds   = row.get("Dataset", dataset)  # Use dataset param if no Dataset column
        show_meth = meth not in SEEN_METHODS
        SEEN_METHODS.add(meth)
        
        sign_disagree = 1 - row["MeanSignAlign"]
        
        # Build hover template conditionally
        if has_dataset:
            hover_text = (
                f"<b>{meth}</b> — {g}<br>"
                f"Dataset: {ds}<br>"
                "TGA: %{x:.4f}<br>"
                "Sign disagreement (1−align): %{y:.3f}<extra></extra>"
            )
        else:
            hover_text = (
                f"<b>{meth}</b> — {g}<br>"
                "TGA: %{x:.4f}<br>"
                "Sign disagreement (1−align): %{y:.3f}<extra></extra>"
            )
        
        fig.add_trace(go.Scatter(
            x    = [row["MeanTGA"]],
            y    = [sign_disagree],
            mode = "markers",
            name = meth,
            legendgroup = meth,
            showlegend  = show_meth,
            marker = dict(
                color  = method_colors.get(meth, "#888888"),
                symbol = GRAPH_MARKERS.get(g, "circle"),
                size   = 11,
                line   = dict(width=1.0, color="white"),
            ),
            hovertemplate=hover_text,
        ))
    
    # Graph shape legend
    for g_label, symbol in GRAPH_MARKERS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=g_label,
            legendgroup=f"graph_{g_label}",
            showlegend=True,
            marker=dict(color="gray", symbol=symbol, size=11),
        ))
    
    # Reference-specific configuration
    if reference == "True":
        title_text = (
            f"True-Graph Alignment Summary — {dataset}<br>"
            "<sup>Colour = method, Shape = graph | "
            "Lower disagreement + TGA near 0 = closer to True oracle</sup>"
        )
        x_title = "Magnitude TGA vs True DAG"
        x_range = None  # auto
    else:  # Traditional
        title_text = (
            f"Traditional Baseline Alignment Summary — {dataset}<br>"
            "<sup>Colour = method, Shape = graph | "
            "Lower-left = closer to Traditional baseline</sup>"
        )
        x_title = "Absolute TGA vs Traditional"
        x_max = df_summary["MeanTGA"].max() * 1.2
        x_range = [0, x_max]
    
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=12),
        ),
        xaxis=dict(
            title=x_title,
            range=x_range,
            zeroline=(reference == "True"),
            zerolinewidth=1.5 if reference == "True" else 1,
            zerolinecolor="gray" if reference == "True" else None,
        ),
        yaxis=dict(
            title="1 − Sign Alignment (sign disagreement rate)",
            range=[0, 0.5],
        ),
        height=height, width=width,
        legend=dict(x=1.02, y=1, xanchor="left", font=dict(size=11)),
        margin=dict(l=70, r=180, t=90, b=70),
    )
    
    fig.show()
    ref_label = reference.lower()
    _save_fig(fig, save_dir, f"tga_sa_scatter_{ref_label}_{dataset}.png")
    return fig


def plot_gss_heatmap(
    gss_feat:      dict,
    feature_names: list,
    base_methods:  list,
    dataset:       str,
    top_n:         int = 40,
    save_dir=None,
) -> "go.Figure":
    """
    Feature-level GSS heatmap (methods = rows, features = columns).

    Parameters
    ----------
    gss_feat      : method → 1-D np.ndarray of signed GSS per feature.
    feature_names : ordered list of feature names.
    base_methods  : ordered list of method names, e.g. ["Asymmetric","Causal","Flow"].
    dataset       : dataset name for the title.
    top_n         : number of features to show (sorted by mean |GSS| descending).
    save_dir      : optional directory to save the figure.
    """
    gss_matrix = np.stack([gss_feat[m] for m in base_methods if m in gss_feat], axis=1)
    feat_order  = np.argsort(gss_matrix.mean(axis=1))[::-1]  # GSS is now absolute, no need for abs()
    top_n       = min(top_n, len(feature_names))
    top_idx     = feat_order[:top_n]
    gss_z_T     = gss_matrix[top_idx].T
    max_val     = float(np.nanmax(gss_matrix))
    labels      = [m for m in base_methods if m in gss_feat]

    fig = go.Figure(go.Heatmap(
        z=gss_z_T, y=labels,
        x=[feature_names[i] for i in top_idx],
        colorscale="Blues",
        zmin=0, zmax=max_val,
        colorbar=dict(title=dict(text="GSS", font=dict(size=11))),
        hovertemplate="<b>%{y}</b> — %{x}<br>GSS = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Feature-Level Graph Sensitivity Score — PC vs LiNGAM — {dataset}<br>"
                f"<sup>GSS = mean_i ||φ(PC)| − |φ(LiNGAM)||.  "
                f"Higher = larger magnitude difference between PC and LiNGAM.  "
                f"Top {top_n} features sorted by mean GSS descending.</sup>"
            ),
            font=dict(size=13),
        ),
        height=310, width=max(800, 20 * top_n + 200),
        xaxis=dict(tickfont=dict(size=8), tickangle=-45, side="bottom"),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=110, r=80, t=85, b=110),
    )
    fig.show()
    _save_fig(fig, save_dir, f"gss_heatmap_{dataset}.png")
    return fig


def plot_gss_delta_box(
    feat_delta:    dict,
    feature_names: list,
    method_colors: dict,
    dataset:       str,
    n_top_features: int,
    save_dir=None,
) -> "go.Figure":
    """
    Per-instance importance magnitude delta box-plots for the top-N features.

    Parameters
    ----------
    feat_delta     : method → (n_instances, n_features) array of |φ(PC)| − |φ(LiNGAM)|.
    feature_names  : ordered list of feature names.
    method_colors  : method short-name → hex colour.
    dataset        : dataset name.
    n_top_features : N shown in the title.
    save_dir       : optional save directory.
    """
    # Compute GSS-based ranking: mean absolute delta across all methods and instances
    all_deltas = np.stack([arr for arr in feat_delta.values()], axis=0)  # (n_methods, n_instances, n_features)
    mean_abs_gss = np.abs(all_deltas).mean(axis=(0, 1))  # (n_features,)
    gss_rank = np.argsort(mean_abs_gss)[::-1][:n_top_features]  # Top N features by GSS
    
    rows = []
    for label, arr in feat_delta.items():
        for fi in gss_rank:
            for v in arr[:, fi]:
                rows.append(dict(Feature=feature_names[fi], Method=label, Delta=float(v)))
    df = pd.DataFrame(rows)
    df["Feature"] = pd.Categorical(
        df["Feature"],
        categories=[feature_names[i] for i in gss_rank], ordered=True,
    )
    fig = px.box(
        df, x="Feature", y="Delta", color="Method",
        color_discrete_map={
            "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
            "Causal":     method_colors.get("Causal",      "#4dac26"),
            "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
        },
        points="outliers",
        title=(
            f"Per-Instance Importance Magnitude Delta |φ(PC)| − |φ(LiNGAM)| — "
            f"Top {n_top_features} Features by Highest GSS — {dataset}<br>"
            f"<sup>Positive = PC assigns higher importance; negative = LiNGAM.  "
            f"Tight boxes = consistent effect; wide = instance-specific sensitivity.</sup>"
        ),
        labels={"Delta": "|φ(PC)| − |φ(LiNGAM)|", "Feature": ""},
        height=480, width=1300,
    )
    fig.add_hline(y=0, line=dict(color="black", width=1, dash="dash"))
    fig.update_layout(xaxis_tickangle=-40, legend_title_text="Method", margin=dict(t=90, b=80))
    fig.show()
    _save_fig(fig, save_dir, f"gss_delta_box_{dataset}.png")
    return fig


def plot_spearman_heatmap(
    spearman_data: dict,
    base_methods:  list,
    disc_graphs:   list,
    dataset:       str,
    n_shap:        int,
    save_dir=None,
) -> "go.Figure":
    """
    Heat-map of Spearman ρ (each method × discovered graph vs Scratch).

    Parameters
    ----------
    spearman_data : (method, graph) → float.
    base_methods  : e.g. ["Asymmetric", "Causal", "Flow"].
    disc_graphs   : e.g. ["PC", "LiNGAM"].
    dataset       : dataset name.
    n_shap        : number of SHAP instances (for subtitle).
    save_dir      : optional save directory.
    """
    matrix = np.full((len(base_methods), len(disc_graphs)), np.nan)
    for mi, meth in enumerate(base_methods):
        for gi, g in enumerate(disc_graphs):
            if (meth, g) in spearman_data:
                matrix[mi, gi] = spearman_data[(meth, g)]

    fig = go.Figure(go.Heatmap(
        z=matrix, x=disc_graphs, y=base_methods,
        colorscale="RdYlGn", zmin=0.5, zmax=1.0,
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}", textfont=dict(size=14, color="black"),
        colorbar=dict(title=dict(text="Spearman ρ", font=dict(size=11)),
                      tickvals=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        hovertemplate="<b>%{y} (%{x})</b> vs Traditional<br>ρ = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Feature Ranking Correlation vs Traditional (Spearman ρ) — {dataset}<br>"
                f"<sup>1.0 = identical ranking to the graph-free baseline.  "
                f"Lower = graph reshuffles importance order.  n={n_shap} instances.</sup>"
            ),
            font=dict(size=13),
        ),
        height=250, width=480,
        margin=dict(l=90, r=60, t=90, b=60),
        yaxis=dict(tickfont=dict(size=12)),
        xaxis=dict(tickfont=dict(size=11)),
    )
    fig.show()
    _save_fig(fig, save_dir, f"spearman_heatmap_{dataset}.png")
    return fig


def plot_jaccard_bar(
    df_jacc:       "pd.DataFrame",
    method_colors: dict,
    dataset:       str,
    save_dir=None,
) -> "go.Figure":
    """
    Grouped bar chart of top-K Jaccard overlap vs Traditional baseline.

    Parameters
    ----------
    df_jacc       : DataFrame with columns [Method, Graph, K, Jaccard].
    method_colors : method short-name → hex colour.
    dataset       : dataset name.
    save_dir      : optional save directory.
    """
    fig = px.bar(
        df_jacc, x="Graph", y="Jaccard", color="Method", facet_col="K",
        barmode="group",
        color_discrete_map={
            "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
            "Causal":     method_colors.get("Causal",      "#4dac26"),
            "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
        },
        range_y=[0, 1.05],
        title=(
            f"Top-K Feature Set Overlap vs Traditional (Jaccard) — {dataset}<br>"
            f"<sup>Fraction of top-K features shared with Traditional's top-K.  "
            f"1.0 = method selects the exact same features as the graph-free baseline.</sup>"
        ),
        labels={"Jaccard": "Jaccard vs Traditional", "Graph": "Discovered Graph"},
        height=380, width=900,
    )
    fig.add_hline(y=1.0, line=dict(color="gray", width=1, dash="dot"))
    fig.update_layout(legend_title_text="Method", margin=dict(t=90, b=60))
    fig.show()
    _save_fig(fig, save_dir, f"jaccard_bar_{dataset}.png")
    return fig


def plot_parent_precision_bar(
    df_parent:       "pd.DataFrame",
    method_colors:   dict,
    dataset:         str,
    disc_graphs:     list,
    random_baseline: float,
    save_dir=None,
) -> "go.Figure":
    """
    Bar chart of true Y-parent Precision@K.

    Parameters
    ----------
    df_parent        : DataFrame with columns [Method, K, Precision].
    method_colors    : method short-name → hex colour.
    dataset          : dataset name.
    disc_graphs      : list of discovered graph names, e.g. ["PC", "LiNGAM"].
    random_baseline  : fraction of features that are true Y-parents.
    save_dir         : optional save directory.
    """
    fig = px.bar(
        df_parent, x="Method", y="Precision", color="Method",
        facet_col="K", barmode="group",
        color_discrete_map={
            "Scratch":               method_colors.get("Scratch",    "#636363"),
            **{f"Asymmetric ({g})": method_colors.get("Asymmetric", "#2166ac") for g in disc_graphs},
            **{f"Causal ({g})":     method_colors.get("Causal",     "#4dac26") for g in disc_graphs},
            **{f"Flow ({g})":       method_colors.get("ShapleyFlow","#d01c8b") for g in disc_graphs},
        },
        range_y=[0, 1.05],
        title=(
            f"True Y-Parent Precision@K — {dataset}<br>"
            f"<sup>Red dashed = random baseline ({random_baseline:.2f}).</sup>"
        ),
        labels={"Precision": "Precision@K (true Y-parents)", "Method": ""},
        height=440, width=1150,
    )
    fig.add_hline(
        y=random_baseline,
        line=dict(color="#d62728", width=1.5, dash="dash"),
        annotation_text=f"random ({random_baseline:.2f})",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(t=90, b=100))
    fig.update_xaxes(tickfont=dict(size=9))
    fig.show()
    _save_fig(fig, save_dir, f"parent_precision_bar_{dataset}.png")
    return fig


def plot_sss_heatmap(
    sss_feat:      dict,
    feature_names: list,
    base_methods:  list,
    dataset:       str,
    top_n:         int = 40,
    save_dir=None,
) -> "go.Figure":
    """
    Feature-level Sign Stability Score heat-map.

    Parameters
    ----------
    sss_feat      : method → 1-D np.ndarray of SSS per feature.
    feature_names : ordered list of feature names.
    base_methods  : ordered list of method names.
    dataset       : dataset name.
    top_n         : number of features to show (sorted by mean SSS ascending).
    save_dir      : optional save directory.
    """
    labels     = [m for m in base_methods if m in sss_feat]
    matrix     = np.stack([sss_feat[m] for m in labels], axis=1)
    feat_order = np.argsort(np.nanmean(matrix, axis=1))
    top_n      = min(top_n, len(feature_names))
    top_idx    = feat_order[:top_n]
    sss_z_T    = matrix[top_idx].T

    fig = go.Figure(go.Heatmap(
        z=sss_z_T, y=labels,
        x=[feature_names[i] for i in top_idx],
        colorscale="RdYlGn", zmin=0.0, zmax=1.0,
        colorbar=dict(
            title=dict(text="SSS", font=dict(size=11)),
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0 (always flips)", "0.25", "0.5", "0.75", "1 (always agrees)"],
        ),
        hovertemplate="<b>%{y}</b> — %{x}<br>SSS = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Feature-Level Sign Stability Score — PC vs LiNGAM — {dataset}<br>"
                f"<sup>SSS = fraction of instances where SHAP sign agrees between PC and LiNGAM.  "
                f"Top {top_n} features sorted by mean SSS ascending (most unstable first).</sup>"
            ),
            font=dict(size=13),
        ),
        height=310, width=max(800, 20 * top_n + 200),
        xaxis=dict(tickfont=dict(size=8), tickangle=-45, side="bottom"),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=110, r=80, t=85, b=110),
    )
    fig.show()
    _save_fig(fig, save_dir, f"sss_heatmap_{dataset}.png")
    return fig


def plot_sss_bar(
    sss_feat:      dict,
    base_methods:  list,
    method_colors: dict,
    dataset:       str,
    save_dir=None,
) -> "go.Figure":
    """
    Bar chart of overall mean SSS per method.

    Parameters
    ----------
    sss_feat      : method → 1-D np.ndarray of SSS per feature.
    base_methods  : ordered method list.
    method_colors : method short-name → hex colour.
    dataset       : dataset name.
    save_dir      : optional save directory.
    """
    color_map = {
        "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
        "Causal":     method_colors.get("Causal",      "#4dac26"),
        "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
    }
    fig = go.Figure()
    for meth in base_methods:
        if meth not in sss_feat:
            continue
        arr = sss_feat[meth]
        fig.add_trace(go.Bar(
            x=[meth], y=[float(np.nanmean(arr))],
            name=meth, marker_color=color_map.get(meth, "#888888"),
            hovertemplate=f"<b>{meth}</b><br>Mean SSS = %{{y:.4f}}<extra></extra>",
        ))
    fig.add_hline(y=1.0, line=dict(color="gray",    dash="dot", width=1))
    fig.add_hline(y=0.5, line=dict(color="#d62728", dash="dash", width=1),
                  annotation_text="0.5 (random)", annotation_font_size=9)
    fig.update_layout(
        title=dict(
            text=(
                f"Overall Sign Stability Score (Mean SSS) — {dataset}<br>"
                f"<sup>1 = all features preserve sign; 0.5 = random; 0 = always flips.</sup>"
            ),
            font=dict(size=13),
        ),
        yaxis=dict(title="Mean SSS", range=[0, 1.05]),
        showlegend=False,
        height=340, width=420,
        margin=dict(l=70, r=30, t=90, b=60),
        xaxis=dict(tickfont=dict(size=11)),
    )
    fig.show()
    _save_fig(fig, save_dir, f"sss_bar_{dataset}.png")
    return fig


def plot_sign_alignment_heatmap(
    sign_align:    dict,
    base_methods:  list,
    disc_graphs:   list,
    feature_names: list,
    dataset:       str,
    n_instances:   int,
    top_n:         int = 40,
    save_dir=None,
) -> "go.Figure":
    """
    Heat-map of per-feature sign alignment rate vs True graph.

    Parameters
    ----------
    sign_align    : (method, graph) → 1-D np.ndarray.
    base_methods  : ordered method list.
    disc_graphs   : discovered graph names.
    feature_names : ordered feature names.
    dataset       : dataset name.
    n_instances   : number of instances used.
    top_n         : features to show (sorted by worst mean alignment).
    save_dir      : optional save directory.
    """
    sa_keys   = [(m, g) for m in base_methods for g in disc_graphs if (m, g) in sign_align]
    sa_labels = [f"{m} ({g})" for m, g in sa_keys]
    matrix    = np.stack([sign_align[k] for k in sa_keys], axis=1)

    feat_order = np.argsort(matrix.mean(axis=1))
    top_n      = min(top_n, len(feature_names))
    top_idx    = feat_order[:top_n]

    fig = go.Figure(go.Heatmap(
        z=matrix[top_idx], x=sa_labels,
        y=[feature_names[i] for i in top_idx],
        colorscale="RdYlGn", zmin=0, zmax=1.0,
        colorbar=dict(
            title=dict(text="Sign Align", font=dict(size=11)),
            tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0 (always opp.)", "0.25", "0.5 (random)", "0.75", "1.0 (perfect)"],
        ),
        hovertemplate="<b>%{y}</b> — %{x}<br>Sign Align = %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Sign Alignment vs True Graph — {dataset}<br>"
                f"<sup>Fraction of instances where sign(φ_discovered) = sign(φ_true).  "
                f"Sorted by worst mean alignment.  N={n_instances} instances.</sup>"
            ),
            font=dict(size=13),
        ),
        height=max(420, 18 * top_n + 120), width=760,
        yaxis=dict(tickfont=dict(size=9)),
        xaxis=dict(tickfont=dict(size=9), tickangle=-25),
        margin=dict(l=80, r=80, t=90, b=80),
    )
    fig.show()
    _save_fig(fig, save_dir, f"sign_alignment_heatmap_{dataset}.png")
    return fig


def plot_sign_alignment_bar(
    df_sa:         "pd.DataFrame",
    sign_align:    dict,
    method_colors: dict,
    dataset:       str,
    save_dir=None,
) -> "go.Figure":
    """
    Bar chart of overall sign alignment per method × graph.

    Parameters
    ----------
    df_sa         : DataFrame with columns [Method, Graph, MeanAlign, label].
    sign_align    : (method, graph) → np.ndarray (needed for annotation).
    method_colors : method short-name → hex colour.
    dataset       : dataset name.
    save_dir      : optional save directory.
    """
    fig = px.bar(
        df_sa, x="label", y="MeanAlign", color="Method", facet_col="Graph",
        barmode="group",
        color_discrete_map={
            "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
            "Causal":     method_colors.get("Causal",      "#4dac26"),
            "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
        },
        range_y=[0, 1.05],
        title=(
            f"Overall Sign Alignment vs True Graph — {dataset}<br>"
            f"<sup>Mean across all features of per-feature sign agreement rate.  "
            f"1.0 = all signs agree with True-graph attributions.</sup>"
        ),
        labels={"MeanAlign": "Mean Sign Alignment", "label": ""},
        height=380, width=700,
    )
    fig.add_hline(y=0.5, line=dict(color="gray",  width=1, dash="dot"),
                  annotation_text="random (0.5)", annotation_font_size=9)
    fig.add_hline(y=1.0, line=dict(color="green", width=1, dash="dot"))
    fig.update_layout(showlegend=True, xaxis_tickangle=-20, margin=dict(t=90, b=80))
    fig.show()
    _save_fig(fig, save_dir, f"sign_alignment_bar_{dataset}.png")
    return fig


def plot_tga_heatmap(
    tga_data:      dict,
    feature_names: list,
    base_methods:  list,
    disc_graphs:   list,
    dataset:       str,
    n_instances:   int,
    top_n:         int = 40,
    save_dir=None,
) -> "go.Figure":
    """
    Feature-level magnitude TGA heat-map vs True DAG.

    Parameters
    ----------
    tga_data      : (method, graph) → 1-D np.ndarray of signed TGA per feature.
    feature_names : ordered feature names.
    base_methods  : ordered method list.
    disc_graphs   : discovered graph names.
    dataset       : dataset name.
    n_instances   : number of instances used (for subtitle).
    top_n         : features to show.
    save_dir      : optional save directory.
    """
    tga_keys   = [(m, g) for m in base_methods for g in disc_graphs if (m, g) in tga_data]
    tga_labels = [f"{m} ({g})" for m, g in tga_keys]
    matrix     = np.stack([tga_data[k] for k in tga_keys], axis=1)
    max_val    = float(np.max(matrix))

    sort_idx   = np.argsort(matrix.mean(axis=1))[::-1]  # TGA is now absolute, no need for abs()
    top_n      = min(top_n, len(feature_names))
    top_idx    = sort_idx[:top_n]

    fig = go.Figure(go.Heatmap(
        z=matrix[top_idx].T,
        x=[feature_names[i] for i in top_idx],
        y=tga_labels,
        colorscale="Blues",
        zmin=0, zmax=max_val,
        colorbar=dict(title=dict(text="TGA", font=dict(size=11))),
        hovertemplate="<b>%{y}</b> — %{x}<br>TGA = %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Feature-Level Magnitude TGA vs True DAG — {dataset}<br>"
                f"<sup>TGA = mean_i ||φ(disc)| − |φ(true)||.  "
                f"Higher = larger magnitude difference from True DAG.  "
                f"Top {top_n} features sorted by mean TGA descending.  N={n_instances} instances.</sup>"
            ),
            font=dict(size=13),
        ),
        height=310, width=max(800, 20 * top_n + 300),
        xaxis=dict(tickfont=dict(size=8), tickangle=-45, side="bottom"),
        yaxis=dict(tickfont=dict(size=10)),
        margin=dict(l=130, r=80, t=85, b=110),
    )
    fig.show()
    _save_fig(fig, save_dir, f"tga_heatmap_{dataset}.png")
    return fig


def plot_tga_bar(
    df_tga:        "pd.DataFrame",
    method_colors: dict,
    dataset:       str,
    save_dir=None,
) -> "go.Figure":
    """
    Bar chart of overall magnitude TGA (absolute deviation) per method × graph.

    Parameters
    ----------
    df_tga        : DataFrame with columns [Method, Graph, MeanTGA, label].
    method_colors : method short-name → hex colour.
    dataset       : dataset name.
    save_dir      : optional save directory.
    """
    fig = px.bar(
        df_tga, x="label", y="MeanTGA", color="Method", facet_col="Graph",
        barmode="group",
        color_discrete_map={
            "Asymmetric": method_colors.get("Asymmetric",  "#2166ac"),
            "Causal":     method_colors.get("Causal",      "#4dac26"),
            "Flow":       method_colors.get("ShapleyFlow",  "#d01c8b"),
        },
        title=(
            f"Overall Magnitude TGA vs True DAG — {dataset}<br>"
            f"<sup>Mean absolute magnitude difference from True DAG attributions.  "
            f"Lower = better alignment with the True graph.</sup>"
        ),
        labels={"MeanTGA": "Mean |TGA|", "label": ""},
        height=380, width=700,
    )
    fig.update_layout(
        showlegend=True, xaxis_tickangle=-20, margin=dict(t=90, b=80),
        yaxis=dict(rangemode="tozero"),
    )
    fig.show()
    _save_fig(fig, save_dir, f"tga_bar_{dataset}.png")
    return fig


def plot_summary_table(
    df_summary:      "pd.DataFrame",
    dataset:         str,
    n_shap:          int,
    n_true_instances: int,
    random_baseline: float,
    tga_data:        dict,
    k_summary:       int = 20,
    save_dir=None,
) -> "go.Figure":
    """
    Aggregated method recommendation summary table.

    Parameters
    ----------
    df_summary       : DataFrame with one row per method.
    dataset          : dataset name.
    n_shap           : number of SHAP instances.
    n_true_instances : number of true-graph instances.
    random_baseline  : fraction of features that are true Y-parents.
    tga_data         : TGA dict (used only to determine label suffix).
    k_summary        : K used for Jaccard and Precision@K columns (default 20).
    save_dir         : optional save directory.
    """
    k = k_summary
    _sss = "SSS" in df_summary.columns and df_summary["SSS"].notna().any()
    cols = [
        ("Method",              "Method"),
        ("GSS",                 "GSS\n(PC↔LiNGAM) ↓"),
    ]
    if _sss:
        cols.append(("SSS", "SSS\n(PC↔LiNGAM) ↑"))
    cols += [
        ("rho_pc",              "ρ(PC–Traditional) ↑"),
        ("rho_lg",              "ρ(LiNGAM–Traditional) ↑"),
        (f"J{k}_pc",            f"J{k}\n(PC–Traditional) ↑"),
        (f"J{k}_lg",            f"J{k}\n(LiNGAM–Traditional) ↑"),
        (f"PP{k}_pc",           f"P@{k}\n(PC) ↑"),
        (f"PP{k}_lg",           f"P@{k}\n(LiNGAM) ↑"),
        ("TGA_pc",              "TGA\n(PC→True) ↓"),
        ("TGA_lg",              "TGA\n(LiNGAM→True) ↓"),
    ]
    # Keep only columns that exist in df_summary
    cols = [(ck, ch) for ck, ch in cols if ck in df_summary.columns]
    col_keys    = [c[0] for c in cols]
    col_headers = [c[1] for c in cols]

    LOWER_BETTER  = {"GSS", "TGA_pc", "TGA_lg"}
    HIGHER_BETTER = {"SSS", "rho_pc", "rho_lg",
                     f"J{k}_pc", f"J{k}_lg", f"PP{k}_pc", f"PP{k}_lg"}

    def fmt(v):
        if isinstance(v, str):  return v
        if isinstance(v, float) and np.isnan(v): return "—"
        return f"{v:.3f}"

    n_rows = len(df_summary)
    cell_vals   = []
    cell_colors = []
    for key in col_keys:
        vals    = df_summary[key].tolist()
        fmtd    = [fmt(v) for v in vals]
        cell_vals.append(fmtd)
        numeric     = [v if (not isinstance(v, str) and not (isinstance(v, float) and np.isnan(v))) else None for v in vals]
        causal_vals = [(i, v) for i, v in enumerate(numeric) if v is not None and i > 0]
        best_i = None
        if causal_vals:
            if key in LOWER_BETTER:
                best_i = min(causal_vals, key=lambda x: x[1])[0]
            elif key in HIGHER_BETTER:
                best_i = max(causal_vals, key=lambda x: x[1])[0]
        row_colors = []
        for i in range(n_rows):
            if i == 0:
                row_colors.append("#eaf4fb")
            elif i == best_i:
                row_colors.append("#d5f5e3")
            else:
                row_colors.append("#f8f9fa" if i % 2 == 0 else "white")
        cell_colors.append(row_colors)

    tga_note = f"N_true={n_true_instances}" if tga_data else "TGA needs Section 10.1"

    fig = go.Figure(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in col_headers],
            fill_color="#2c3e50",
            font=dict(color="white", size=10),
            align="center", height=52,
        ),
        cells=dict(
            values=cell_vals, fill_color=cell_colors,
            align="center", height=40, font=dict(size=13),
        ),
    ))
    fig.update_layout(
        title=dict(
            text=(
                f"Graph Sensitivity — Method Recommendation Summary — {dataset}<br>"
                f"<sup>Blue = Traditional (reference).  Green = best causal method per column.  "
                f"↓ lower better  ↑ higher better.  "
                f"GSS/Spearman/Jaccard: n={n_shap} instances.  {tga_note}.  "
                f"P@{k} random baseline = {random_baseline:.2f}</sup>"
            ),
            font=dict(size=13),
        ),
        height=340, width=1300,
        margin=dict(l=10, r=10, t=85, b=10),
    )
    fig.show()
    _save_fig(fig, save_dir, f"summary_table_{dataset}.png")
    return fig


def plot_adjacency_comparison(
    true_adj_xx:         "np.ndarray",
    pc_train_adj_xx:     "np.ndarray",
    lingam_train_adj_xx: "np.ndarray",
    feature_names:       list,
    dataset:             str,
    save_dir=None,
):
    """
    Side-by-side heat-maps of True, PC, and LiNGAM X→X adjacency matrices.

    Parameters
    ----------
    true_adj_xx         : n×n ground-truth X→X adjacency.
    pc_train_adj_xx     : n×n PC discovered X→X adjacency.
    lingam_train_adj_xx : n×n LiNGAM discovered X→X adjacency.
    feature_names       : ordered list of feature names.
    dataset             : dataset name.
    save_dir            : optional save directory.
    """
    n = len(feature_names)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, mat, title, cmap in [
        (axes[0], true_adj_xx,         "True X→X",    "Blues"),
        (axes[1], pc_train_adj_xx,     "PC X→X",      "Oranges"),
        (axes[2], lingam_train_adj_xx, "LiNGAM X→X",  "Greens"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels(feature_names, fontsize=7)
        ax.set_title(f"{title}\n({int(mat.sum())} edges)", fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Adjacency matrix comparison (X-only block) — {dataset}\nadj[i,j]=1 means edge i→j",
        fontweight="bold", fontsize=11,
    )
    plt.tight_layout()
    _save_fig(fig, save_dir, f"adjacency_comparison_{dataset}.png")
    plt.show()
    return fig
