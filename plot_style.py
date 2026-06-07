"""
plot_style.py
=============

Single source of truth for the visual style of every figure in the analysis
(``assessment_extras.py`` and ``shapley_plots.py`` both import from here). One palette,
one matplotlib style, so the thesis figures are consistent.

Design choices (research / print friendly)
-----------------------------------------
* Method colours: the **Okabe–Ito** colourblind-safe qualitative palette.
* Magnitude metrics (TGA, GSS — non-negative): a **sequential blue** colormap
  (``Blues``), where white = 0 and deeper blue = larger magnitude deviation.
* Sign-disagreement metrics (D — non-negative): a **sequential light-coral** colormap
  (white → #F08080), where white = 0 (OK / stable) and deeper coral = higher concern.
  Coral vs blue visually separates the two metric families at a glance.
* Direction / agreement metrics kept in ``DIV_CMAP`` (``RdBu``) for any legacy uses
  that explicitly pass the diverging map.
* Graphs are encoded by **marker shape** (scatter) or **hatch** (bars), never by colour,
  so colour always means *method*.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ----------------------------------------------------------------------------- #
# palette
# ----------------------------------------------------------------------------- #
METHOD_ORDER = ["Asymmetric", "Causal", "Flow"]

# Okabe–Ito colourblind-safe qualitative colours
METHOD_COLORS = {
    "Asymmetric": "#0072B2",   # blue
    "Causal":     "#D55E00",   # vermillion
    "Flow":       "#009E73",   # bluish green
}

# references / baselines (used for anchor lines, scratch bars, etc.)
REFERENCE_COLORS = {
    "True":        "#000000",   # oracle — black
    "Scratch":     "#7F7F7F",   # traditional baseline — grey
    "Traditional": "#7F7F7F",
    "PC":          "#56B4E9",   # sky blue   (only when a graph must be coloured)
    "LiNGAM":      "#E69F00",   # orange
}

# graph encodings (colour is reserved for method)
GRAPH_MARKERS = {"PC": "o", "LiNGAM": "D", "True": "s"}
GRAPH_HATCH = {"PC": "", "LiNGAM": "//", "True": ".."}

# colormaps
SEQ_CMAP      = "Blues"    # magnitude (ΔM), >= 0 : white (0) -> deep blue (high)
DISAGREE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "LightCoral", ["#ffffff", "#F08080"]  # white → light coral
)
DIV_CMAP      = "RdBu"     # agreement in [0,1], diverging around 0.5

# semantic accents
C_BASELINE_LINE = "#444444"   # dashed reference / zero lines
C_ACCENT = "#D55E00"          # marker for the reported (aggregated) metric
C_DIST = "#4C72B0"            # instance scatter / distribution fill
C_DIST_FILL = "#9ecae1"


def method_color(name, default="#888888"):
    return METHOD_COLORS.get(name, default)


# ----------------------------------------------------------------------------- #
# matplotlib style
# ----------------------------------------------------------------------------- #
def apply_style():
    """Apply the shared rcParams. Call once at import time of the plotting modules."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.dpi":      150,
        "savefig.bbox":     "tight",
        "figure.dpi":       120,
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.titleweight": "bold",
        "axes.labelsize":   11,
        "axes.edgecolor":   "#444444",
        "axes.linewidth":   0.8,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "xtick.color":      "#333333",
        "ytick.color":      "#333333",
        "legend.fontsize":  9,
        "legend.frameon":   False,
        "grid.alpha":       0.25,
        "grid.linewidth":   0.6,
    })


def style_title(ax_or_fig, title, subtitle=None):
    """Bold title with an optional smaller grey subtitle (two-line research caption)."""
    if hasattr(ax_or_fig, "set_title"):
        ax_or_fig.set_title(title, loc="left", pad=20 if subtitle else 6)
        if subtitle:
            ax_or_fig.annotate(subtitle, xy=(0, 1.0), xycoords="axes fraction",
                               xytext=(0, 4), textcoords="offset points",
                               fontsize=8.5, color="#666666", va="bottom")
    else:
        ax_or_fig.suptitle(title, fontweight="bold")


def savefig(fig, plots_dir, dataset, filename, method=None):
    """Save figure to <plots_dir>/<dataset>[/<method>]/<filename> and return the path.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    plots_dir : str or Path
        Base plots directory.
    dataset : str
        Dataset name for subdirectory.
    filename : str
        Output filename (e.g., 'plot.png').
    method : str, optional
        If provided, creates a method-specific subdirectory (e.g., 'asymmetric').
        Method name is lowercased for directory name.
    
    Returns
    -------
    Path
        Path to the saved figure.
    """
    out_dir = Path(plots_dir) / dataset
    if method:
        # Create method-specific subdirectory (lowercase)
        out_dir = out_dir / method.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / filename
    fig.savefig(str(dest))
    return dest


# apply on import so simply importing the style is enough
apply_style()
