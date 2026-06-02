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
    reference:    str = "True",
) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, set[int]]]:
    """
    Signed magnitude difference between discovered-graph and reference attributions.

    TGA(method, graph, feature) = mean_i ( |φ(disc,i,f)| − |φ(ref,i,f)| )

    No outer absolute and no normalisation.  The sign encodes direction:
      positive → discovered graph over-estimates the reference magnitude;
      negative → discovered graph under-estimates the reference magnitude;
      near 0   → discovered graph and reference agree in magnitude.

    Reference is resolved the same way as in ``compute_sign_alignment``.

    Parameters
    ----------
    shap_data     : dict with keys like ``"{method} ({graph})"`` or ``"Scratch"``.
    feature_names : list of feature name strings (kept for API compatibility).
    base_methods  : list of method names.
    disc_graphs   : graphs to evaluate.
    reference     : reference variant. Default ``"True"``. Pass ``"Scratch"``
                    for graph-free baseline alignment.

    Returns
    -------
    tga_data       : dict mapping (method, graph) → ndarray (n_features,) in (-∞, +∞)
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
        zero_feat_sets[meth] = set()  # no NaN features — signed metric, no division
        for g in disc_graphs:
            disc_key = f"{meth} ({g})"
            if disc_key not in shap_data:
                continue
            phi_disc = shap_data[disc_key]
            diff     = (np.abs(phi_disc) - abs_ref).mean(axis=0)  # signed, per feature
            tga_data[(meth, g)] = diff
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
) -> dict[str, np.ndarray]:
    """
    Per-feature signed magnitude difference between PC and LiNGAM outputs.

    d_f(m) = mean_i ( |φ^PC_{i,f}| − |φ^LiNGAM_{i,f}| )

    No outer absolute and no normalisation.  The sign encodes direction:
      positive → PC assigns higher mean absolute importance than LiNGAM;
      negative → LiNGAM assigns higher mean absolute importance;
      near 0   → both graphs agree on magnitude for that feature.

    Averaging d_f over features gives the overall directional bias of a method.

    Parameters
    ----------
    shap_data    : dict with keys like ``"{method} (PC)"`` and ``"{method} (LiNGAM)"``.
    base_methods : list of method names.

    Returns
    -------
    gss : dict mapping method → ndarray (n_features,) in (-∞, +∞)
          Near 0 = PC and LiNGAM agree; positive = PC > LiNGAM; negative = LiNGAM > PC.
    """
    gss: dict[str, np.ndarray] = {}
    for meth in base_methods:
        pc_key = f"{meth} (PC)"
        lg_key = f"{meth} (LiNGAM)"
        if pc_key not in shap_data or lg_key not in shap_data:
            continue
        gss[meth] = (np.abs(shap_data[pc_key]) - np.abs(shap_data[lg_key])).mean(axis=0)
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
