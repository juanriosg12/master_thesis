"""
shapley_comparison_plots.py
───────────────────────────
Two focused comparison plots for pairs of Shapley methods.

Functions
---------
plot_shap_vs_feature(...)
    SHAP value vs feature value scatter — two methods overlaid in one subplot
    per feature.  Good for seeing how the attribution *shape* changes.

plot_shap_vs_instance(...)
    SHAP value vs instance index — two methods overlaid in one subplot per
    feature.  Good for seeing per-instance attribution *differences*.

Both functions accept any method keys that exist in ``shap_data``, e.g.:
    "Scratch", "Asymmetric (PC)", "Causal (LiNGAM)", "Flow (True)", …

Quick usage
-----------
    from shapley_comparison_plots import plot_shap_vs_feature, plot_shap_vs_instance

    # Feature-value scatter
    plot_shap_vs_feature(
        shap_data, feature_names, x_test,
        method_1="Scratch", method_2="Flow (PC)",
        features=["X3", "X21", "X47"],
        dataset=DATASET,
    )

    # Instance-level scatter
    plot_shap_vs_instance(
        shap_data, feature_names,
        method_1="Scratch", method_2="Flow (PC)",
        features=["X3", "X21", "X47"],
        instance_idx=instance_idx,
        dataset=DATASET,
    )
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────────────────────
# Colour defaults
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_COLORS: dict[str, str] = {
    "Scratch":              "#636363",
    "Asymmetric (PC)":      "#2166ac",
    "Causal (PC)":          "#4dac26",
    "Flow (PC)":            "#d01c8b",
    "Asymmetric (LiNGAM)": "#b2182b",
    "Causal (LiNGAM)":     "#1a9850",
    "Flow (LiNGAM)":       "#f46d43",
    "Asymmetric (True)":   "#74add1",
    "Causal (True)":       "#a6d96a",
    "Flow (True)":         "#fdae61",
}

# Fallback pair when a method key is not in the table above
_FALLBACK = ("#1f77b4", "#ff7f0e")


def _resolve_colors(
    method_1: str,
    method_2: str,
    colors:   dict | None,
) -> tuple[str, str]:
    c = _DEFAULT_COLORS.copy()
    if colors:
        c.update(colors)
    return c.get(method_1, _FALLBACK[0]), c.get(method_2, _FALLBACK[1])


def _safe_name(method: str) -> str:
    """Turn a method key into a safe filename fragment."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", method)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  SHAP vs Feature Value
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_vs_feature(
    shap_data:     dict[str, np.ndarray],
    feature_names: list[str],
    x_test:        np.ndarray,
    method_1:      str,
    method_2:      str,
    features:      list[str] | None = None,
    dataset:       str = "",
    colors:        dict | None = None,
    marker_size:   int = 5,
    opacity:       float = 0.65,
    save_dir=None,
) -> go.Figure:
    """
    SHAP value vs feature value scatter — two methods overlaid.

    One row per feature.  Both methods share the same subplot so the
    difference in attribution shape is immediately visible.

    Parameters
    ----------
    shap_data     : dict mapping method key → ndarray (n_instances, n_features).
    feature_names : full ordered list of feature names (X0 … Xn-1, Y excluded).
    x_test        : feature matrix (n_instances, n_features).
    method_1      : first method key, e.g. ``"Scratch"``.
    method_2      : second method key, e.g. ``"Flow (PC)"``.
    features      : feature names to plot.  If None, all features are plotted.
    dataset       : dataset name shown in the title.
    colors        : optional colour overrides ``{method_key: hex_string}``.
    marker_size   : scatter dot size (default 5).
    opacity       : scatter dot opacity (default 0.65).
    save_dir      : ``pathlib.Path`` (or str) to write the HTML file.
                    None = skip saving.

    Returns
    -------
    go.Figure
    """
    for key in (method_1, method_2):
        if key not in shap_data:
            raise KeyError(
                f"'{key}' not found in shap_data.  "
                f"Available keys: {sorted(shap_data)}"
            )

    feat_list = list(features) if features is not None else list(feature_names)
    feat_idx  = [feature_names.index(f) for f in feat_list]
    n_rows    = len(feat_list)

    c1, c2    = _resolve_colors(method_1, method_2, colors)
    v_spacing = max(0.005, min(0.06, 0.25 / max(n_rows - 1, 1)))

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=v_spacing,
    )

    _specs = [
        (method_1, c1, "circle"),
        (method_2, c2, "diamond"),
    ]

    for ri, (fi, feat) in enumerate(zip(feat_idx, feat_list)):
        x_vals      = x_test[:, fi]
        show_legend = ri == 0

        for method, color, symbol in _specs:
            sv = shap_data[method][:, fi]
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=sv,
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=color,
                        symbol=symbol,
                        opacity=opacity,
                        line=dict(width=0),
                    ),
                    name=method,
                    showlegend=show_legend,
                    legendgroup=method,
                    hovertemplate=(
                        f"<b>{feat}</b> — {method}<br>"
                        "feature value = %{x:.4f}<br>"
                        "SHAP = %{y:.4f}"
                        "<extra></extra>"
                    ),
                ),
                row=ri + 1, col=1,
            )

        # Zero reference line
        x_min, x_max = float(x_vals.min()), float(x_vals.max())
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[0, 0],
                mode="lines",
                line=dict(color="gray", width=0.8, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=ri + 1, col=1,
        )

        fig.update_yaxes(
            title_text=f"SHAP ({feat})",
            title_font=dict(size=9),
            title_standoff=4,
            row=ri + 1, col=1,
        )
        fig.update_xaxes(
            title_text="Feature value" if ri == n_rows - 1 else "",
            tickfont=dict(size=8),
            row=ri + 1, col=1,
        )

    title_parts = [f"SHAP vs Feature Value — {method_1}  ·  {method_2}"]
    if dataset:
        title_parts.append(dataset)
    if len(feat_list) < len(feature_names):
        title_parts.append(f"{len(feat_list)} features")
    title_str = " — ".join(title_parts)

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14)),
        height=240 * n_rows + 80,
        width=600,
        margin=dict(l=110, r=20, t=80, b=50),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.02, yanchor="bottom",
            font=dict(size=11),
        ),
    )

    if save_dir is not None:
        out = Path(save_dir) / (
            f"shap_vs_feature_{_safe_name(method_1)}_vs_{_safe_name(method_2)}.html"
        )
        fig.write_html(str(out))
        print(f"Saved → {out}")

    fig.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SHAP vs Instance
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_vs_instance(
    shap_data:     dict[str, np.ndarray],
    feature_names: list[str],
    method_1:      str,
    method_2:      str,
    features:      list[str] | None = None,
    instance_idx:  list[int] | None = None,
    dataset:       str = "",
    colors:        dict | None = None,
    mode:          str = "markers",
    marker_size:   int = 5,
    opacity:       float = 0.70,
    save_dir=None,
) -> go.Figure:
    """
    SHAP value vs instance index — two methods overlaid.

    One row per feature.  Both methods share the same subplot so per-instance
    attribution differences are directly visible.

    Parameters
    ----------
    shap_data     : dict mapping method key → ndarray (n_instances, n_features).
    feature_names : full ordered list of feature names.
    method_1      : first method key, e.g. ``"Scratch"``.
    method_2      : second method key, e.g. ``"Flow (PC)"``.
    features      : feature names to plot.  If None, all features are plotted.
    instance_idx  : subset of instance indices to display.  None = all instances.
    dataset       : dataset name shown in the title.
    colors        : optional colour overrides ``{method_key: hex_string}``.
    mode          : plotly scatter mode.  One of ``"markers"``, ``"lines"``,
                    ``"lines+markers"`` (default: ``"markers"``).
    marker_size   : dot / line marker size.
    opacity       : marker opacity.
    save_dir      : ``pathlib.Path`` (or str) to write the HTML file.
                    None = skip saving.

    Returns
    -------
    go.Figure
    """
    for key in (method_1, method_2):
        if key not in shap_data:
            raise KeyError(
                f"'{key}' not found in shap_data.  "
                f"Available keys: {sorted(shap_data)}"
            )

    n_total  = shap_data[method_1].shape[0]
    inst_idx = list(instance_idx) if instance_idx is not None else list(range(n_total))

    feat_list = list(features) if features is not None else list(feature_names)
    feat_idx  = [feature_names.index(f) for f in feat_list]
    n_rows    = len(feat_list)

    c1, c2    = _resolve_colors(method_1, method_2, colors)
    v_spacing = max(0.005, min(0.06, 0.25 / max(n_rows - 1, 1)))

    use_lines = "lines" in mode
    _specs = [
        (method_1, c1, "circle",  "solid"),
        (method_2, c2, "diamond", "dot"),
    ]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=v_spacing,
    )

    for ri, (fi, feat) in enumerate(zip(feat_idx, feat_list)):
        show_legend = ri == 0

        for method, color, symbol, dash in _specs:
            sv = shap_data[method][inst_idx, fi]
            trace_kwargs: dict = dict(
                x=inst_idx,
                y=sv,
                mode=mode,
                marker=dict(
                    size=marker_size,
                    color=color,
                    symbol=symbol,
                    opacity=opacity,
                    line=dict(width=0),
                ),
                name=method,
                showlegend=show_legend,
                legendgroup=method,
                hovertemplate=(
                    f"<b>{feat}</b> — {method}<br>"
                    "instance = %{x}<br>"
                    "SHAP = %{y:.4f}"
                    "<extra></extra>"
                ),
            )
            if use_lines:
                trace_kwargs["line"] = dict(color=color, width=1.5, dash=dash)

            fig.add_trace(go.Scatter(**trace_kwargs), row=ri + 1, col=1)

        # Zero reference line
        fig.add_trace(
            go.Scatter(
                x=[inst_idx[0], inst_idx[-1]],
                y=[0, 0],
                mode="lines",
                line=dict(color="gray", width=0.8, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=ri + 1, col=1,
        )

        fig.update_yaxes(
            title_text=f"SHAP ({feat})",
            title_font=dict(size=9),
            title_standoff=4,
            row=ri + 1, col=1,
        )

    fig.update_xaxes(
        title_text="Instance index",
        tickfont=dict(size=8),
        row=n_rows, col=1,
    )

    title_parts = [f"SHAP vs Instance — {method_1}  ·  {method_2}"]
    if dataset:
        title_parts.append(dataset)
    if len(inst_idx) < n_total:
        title_parts.append(f"n={len(inst_idx)} instances")
    title_str = " — ".join(title_parts)

    fig.update_layout(
        title=dict(text=title_str, font=dict(size=14)),
        height=220 * n_rows + 80,
        width=700,
        margin=dict(l=110, r=20, t=80, b=50),
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=1.02, yanchor="bottom",
            font=dict(size=11),
        ),
    )

    if save_dir is not None:
        out = Path(save_dir) / (
            f"shap_vs_instance_{_safe_name(method_1)}_vs_{_safe_name(method_2)}.html"
        )
        fig.write_html(str(out))
        print(f"Saved → {out}")

    fig.show()
    return fig
