"""
shapley_comparison_plots.py
───────────────────────────
Thin re-export wrapper.

``plot_shap_vs_feature`` and ``plot_shap_vs_instance`` have been consolidated
into ``analysis_utils.py`` so that all comparison plots share the same colour
palette, marker style, and PNG-save logic.

Import from ``analysis_utils`` directly, or continue using this module via
``scp.plot_shap_vs_feature(...)`` — both work.
"""
from utils.analysis_utils import (  # noqa: F401
    plot_shap_vs_feature,
    plot_shap_vs_instance,
)

__all__ = ["plot_shap_vs_feature", "plot_shap_vs_instance"]
