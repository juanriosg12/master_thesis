"""
regen_figs.py
=============
Convenience wrapper to regenerate every figure in ``notebooks/plots_clean/``
for the two focus datasets after editing plot labels in ``shapley_plots.py``.

In a normal environment (joblib + pyarrow + lightgbm installed) this is
equivalent to ``python shapley_plots.py``: ``output_std`` is read from the saved
model's predictions on the test set. Run from the project root:

    python regen_figs.py
"""
from . import assessment_extras as ax
from . import shapley_plots as sp
from .assessment_extras import DEFAULT_DATASETS

if __name__ == "__main__":
    for ds in DEFAULT_DATASETS:
        ctx = ax.load_context(ds)
        made = sp.run_all(ctx)
        print(f"[ok] {ds}: {len(made)} outputs (output_std = {ctx['output_std']:.4f})")
