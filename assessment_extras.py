"""
assessment_extras.py
====================

Distribution-preserving and structure-conditioned views of the structure-aware
Shapley results. Complements ``analysis_utils.py`` (which collapses to per-feature
means) by keeping the *local* (instance) axis and connecting changes to graph
structure.

IMPORTANT — metric definitions and scope
---------------------------------------
* The per-feature metrics here are copied VERBATIM from ``analysis_utils.py`` so the
  numbers match ``notebooks/shapley_summary.ipynb`` exactly:

      TGA(method, graph, feature) = mean_i | |phi_disc| - |phi_ref| |        (inner abs)
      GSS(method, feature)        = mean_i | |phi_PC|   - |phi_LiNGAM| |      (inner abs)

  Both are NON-NEGATIVE (no sign, no cancellation). Global value = mean over features.
  (Verified: Causal (PC) vs True -> 0.1960 on linear_conf_f50_s1000_p30.)

* SCOPE. The *subject* is always a DISCOVERED-graph result: (method, graph) with
  graph in {PC, LiNGAM}. The only admissible references are:
      - "True"   : same method on the True DAG          -> compute_tga(reference="True")
      - "Scratch": graph-free baseline                  -> compute_tga(reference="Scratch")
      - opposite discovered graph, same method (PC<->LiNGAM) -> GSS
  We never use True-vs-Scratch (oracle vs baseline) — that is out of scope.

Usage
-----
    import assessment_extras as ax
    ctx = ax.load_context("linear_conf_f50_s1000_p30")
    ax.plot_instance_distributions(ctx, method="Causal", graph="PC", reference="True")
    ax.plot_mass_budget(ctx)

    python assessment_extras.py            # regenerates the two focus datasets
"""

from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------- #
# CONFIG
# ----------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
EXPLAIN_DIR = BASE_DIR / "data" / "explainability"
CAUSAL_DIR = BASE_DIR / "data" / "causal"
DEFAULT_PLOTS_DIR = BASE_DIR / "notebooks" / "plots_claude"

DEFAULT_MODEL = "lgbm"
DEFAULT_DATASETS = ("linear_conf_f50_s1000_p30", "sachs")

# display label -> folder name
METHODS = {"Causal": "causal", "Asymmetric": "asymmetric", "Flow": "flow", "Traditional": "Scratch"}
GRAPHS = {"True": "true", "PC": "pc", "LiNGAM": "lingam"}
DISC_GRAPHS = ("PC", "LiNGAM")          # admissible subjects
SCRATCH = "Scratch"

# colour grammar — single source of truth in plot_style.py
from plot_style import (  # noqa: E402
    METHOD_COLORS, GRAPH_HATCH, REFERENCE_COLORS,
    C_ACCENT as C_MEAN, C_DIST, C_DIST_FILL, C_BASELINE_LINE,
)
C_BASE = REFERENCE_COLORS["Scratch"]

UPSTREAM_MIN_DESC = 3


# ----------------------------------------------------------------------------- #
# DATA LOADING  (keys use the notebook convention "Method (Graph)" + "Scratch")
# ----------------------------------------------------------------------------- #
def _load_npy(dataset, model, *parts):
    p = EXPLAIN_DIR / dataset / model / Path(*parts) / "shapley_values.npy"
    return np.load(p) if p.exists() else None


def _load_feature_names(dataset, F):
    """Load feature names from causal discovery results, or return generic names."""
    p = CAUSAL_DIR / f"{dataset}_pc_results.json"
    if p.exists():
        try:
            data = json.load(open(p))
            names = data.get("feature_names", [])
            # Filter out 'Y' (target) and return only feature names
            names = [n for n in names if n != "Y"]
            if len(names) == F:
                return names
        except Exception:
            pass
    return [f"X{i}" for i in range(F)]


def load_context(dataset: str, model: str = DEFAULT_MODEL) -> dict:
    scratch = _load_npy(dataset, model, "scratch")
    if scratch is None:
        raise FileNotFoundError(f"No scratch SHAP for {dataset}/{model}")
    F = scratch.shape[1]

    shap = {SCRATCH: scratch}
    for M, mdir in METHODS.items():
        for G, gdir in GRAPHS.items():
            arr = _load_npy(dataset, model, gdir, mdir)
            if arr is not None:
                shap[f"{M} ({G})"] = arr

    struct = _graph_struct(dataset, F)
    feature_names = _load_feature_names(dataset, F)
    return dict(dataset=dataset, model=model, F=F, shap=shap, struct=struct, feature_names=feature_names)


def _graph_struct(dataset, F):
    p = CAUSAL_DIR / f"{dataset}_true_full_adjacency.npy"
    if not p.exists():
        return None
    adj = np.load(p)
    if adj.shape[0] < F:
        return None
    A = (adj[:F, :F] != 0).astype(int)

    def desc(j):
        seen, st = set(), [j]
        while st:
            u = st.pop()
            for v in np.where(A[u] != 0)[0]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return len(seen)

    return dict(indeg=A.sum(0), outdeg=A.sum(1),
                ndesc=np.array([desc(j) for j in range(F)]))


# ----------------------------------------------------------------------------- #
# METRICS — copied verbatim from analysis_utils.py (inner absolute, non-negative)
# ----------------------------------------------------------------------------- #
def _ref_key(shap, method, reference):
    """Resolve reference the same way analysis_utils does: per-method key, else direct."""
    # Handle Traditional -> Scratch alias
    if reference == "Traditional":
        reference = SCRATCH
    
    rk = f"{method} ({reference})"
    if rk in shap:
        return rk
    return reference if reference in shap else None


def tga_instance(ctx, method, graph, reference="True"):
    """(N, F) per-instance contribution  | |phi_disc| - |phi_ref| |  (what TGA averages)."""
    shap = ctx["shap"]
    dk = f"{method} ({graph})"
    rk = _ref_key(shap, method, reference)
    if dk not in shap or rk is None:
        return None
    return np.abs(shap[dk]) - np.abs(shap[rk])


def tga_feature(ctx, method, graph, reference="True"):
    """(F,) per-feature TGA = mean over instances. Matches compute_tga."""
    inst = tga_instance(ctx, method, graph, reference)
    return None if inst is None else np.abs(inst).mean(0)


def gss_instance(ctx, method):
    """(N, F) signed per-instance  |phi_PC| - |phi_LiNGAM|  (same logic as tga_instance,
    with PC as subject and LiNGAM as reference). Sign encodes which graph is larger."""
    return tga_instance(ctx, method, "PC", "LiNGAM")


def gss_feature(ctx, method):
    """(F,) per-feature GSS = mean_i | |phi_PC| - |phi_LiNGAM| |. Matches compute_gss."""
    inst = gss_instance(ctx, method)
    return None if inst is None else np.abs(inst).mean(0)


def global_tga(ctx, method, graph, reference="True"):
    """Scalar global TGA = absolute average of the per-feature TGA."""
    feat = tga_feature(ctx, method, graph, reference)
    return None if feat is None else float(np.abs(feat).mean())


def global_gss(ctx, method):
    """Scalar global GSS = absolute average of the per-feature GSS."""
    feat = gss_feature(ctx, method)
    return None if feat is None else float(np.abs(feat).mean())


def _spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _save(fig, ctx, name, plots_dir):
    out_dir = Path(plots_dir) / ctx["dataset"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    return out


# ----------------------------------------------------------------------------- #
# LEVEL 2 — instance distribution behind the per-feature TGA
# ----------------------------------------------------------------------------- #
def plot_instance_distributions(ctx, method="Causal", graph="PC", reference="True",
                                top_n=8, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Violin + strip of the per-instance TGA contribution  | |phi_disc|-|phi_ref| |
    for the most-changed features. Crimson diamond = per-feature TGA (the reported
    number). Shows that the per-feature TGA is driven by a skewed instance distribution
    — a few instances dominate — which the double mean (over instances, then features)
    hides.  (Magnitude only; sign behaviour is a separate metric, Sign Alignment.)
    """
    inst = tga_instance(ctx, method, graph, reference)
    if inst is None:
        return None
    tga = np.median(inst, 0)  # median is more representative of the distribution's centre (mean can be skewed by outliers)

    tga_abs_feat = np.abs(inst).mean(0)
    order = np.argsort(tga_abs_feat)[-top_n:]
    N = inst.shape[0]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    parts = ax.violinplot([inst[:, j] for j in order], vert=False,
                          showmeans=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#9ecae1")
        pc.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, j in enumerate(order):
        ax.scatter(inst[:, j], np.full(N, i + 1) + rng.uniform(-0.12, 0.12, N),
                   s=6, color=C_DIST, alpha=0.35)
        ax.scatter(tga[j], i + 1, color=C_MEAN, s=70, marker="D", zorder=5)
    ax.set_yticks(range(1, len(order) + 1))
    # Use actual feature names if available
    feature_names = ctx.get("feature_names", [f"X{j}" for j in range(ctx["F"])])
    ax.set_yticklabels([feature_names[j] for j in order])
    ax.axvline(0, ls="--", c="k", lw=0.8)
    ax.set_xlabel(rf"per-instance TGA contribution  $|\phi_{{{graph}}}| - |\phi_{{{reference}}}|$")
    ax.set_title(f"{method} ({graph}) vs ({reference}) — instance distribution behind TGA\n"
                 "Median ◆ = reported per-feature TGA")
    fig.tight_layout()
    out = _save(fig, ctx, f"instance_tga_{method}_{graph}_vs_{reference}", plots_dir)
    if not show:
        plt.close(fig)
    return out


# ----------------------------------------------------------------------------- #
# LEVEL 1 — attribution mass budget (total |phi| per result, ratio vs Scratch)
# ----------------------------------------------------------------------------- #
def plot_mass_budget(ctx, plots_dir=DEFAULT_PLOTS_DIR, show=False):
    """Total |phi| budget for every (method, graph) result, as a ratio to Scratch.
    Subjects are the discovered graphs (PC, LiNGAM); True is shown as the oracle
    anchor. Reveals which Shapley method inflates the attribution budget vs which
    merely re-slices it — the explanation for the *level* of TGA.
    """
    sct = np.abs(ctx["shap"][SCRATCH]).sum()
    methods = [m for m in METHODS if any(f"{m} ({g})" in ctx["shap"] for g in GRAPHS)]
    graphs = list(GRAPHS)  # PC, LiNGAM, True

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(methods))
    w = 0.26
    for k, g in enumerate(graphs):
        vals = [np.abs(ctx["shap"][f"{m} ({g})"]).sum() / sct
                if f"{m} ({g})" in ctx["shap"] else np.nan for m in methods]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=g,
                      color=[METHOD_COLORS[m] for m in methods],
                      hatch=GRAPH_HATCH[g], edgecolor="k")
        for xi, v in zip(x + (k - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, ls="--", c="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("total |φ| budget  (÷ Scratch)")
    ax.set_title(f"Attribution mass budget — {ctx['dataset']}\n"
                 "bars = graph (PC / LiNGAM / True); dashed = Scratch (×1.0)")
    ax.legend(title="graph")
    fig.tight_layout()
    out = _save(fig, ctx, "mass_budget", plots_dir)
    if not show:
        plt.close(fig)
    return out


# ----------------------------------------------------------------------------- #
# driver
# ----------------------------------------------------------------------------- #
def run_all(dataset, model=DEFAULT_MODEL, plots_dir=DEFAULT_PLOTS_DIR):
    ctx = load_context(dataset, model)
    outs = [
        plot_instance_distributions(ctx, method="Causal", graph="PC", reference="LiNGAM"),
        plot_instance_distributions(ctx, method="Asymmetric", graph="PC", reference="LiNGAM"),
        plot_instance_distributions(ctx, method="Flow", graph="PC", reference="LiNGAM"),
        plot_instance_distributions(ctx, method="Asymmetric", graph="PC", reference="Traditional"),
        plot_instance_distributions(ctx, method="Asymmetric", graph="LiNGAM", reference="Traditional"),
        plot_mass_budget(ctx, plots_dir=plots_dir),
    ]
    return [str(o) for o in outs if o is not None]


def discover_datasets():
    return sorted(d.name for d in EXPLAIN_DIR.iterdir() if d.is_dir()) \
        if EXPLAIN_DIR.exists() else []


if __name__ == "__main__":
    import sys

    for ds in (sys.argv[1:] or list(DEFAULT_DATASETS)):
        try:
            made = run_all(ds)
            print(f"[ok] {ds}: {len(made)} figures")
            for m in made:
                print("     ", m)
        except Exception as e:
            print(f"[skip] {ds}: {type(e).__name__}: {e}")
