"""
Plot: Causal Discovery F1 (PC vs LiNGAM) + Model R²
Datasets: Synthetic Linear Confounded p30  |  Sachs

Two y-axes on the same figure:
  - Left  y-axis : F1 score (bars, one per method per dataset)
  - Right y-axis : R² of the LGBM model (line + markers)
  - x-axis       : dataset groups
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.metrics import r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent))

from predictive_models.predictive_models import LGBMRegressor

# ── Paths ─────────────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
CAUSAL  = BASE / 'data' / 'causal'
MODELS  = BASE / 'models'
PROC    = BASE / 'data' / 'processed'

SYNTH_DATASET = 'linear_conf_f50_s1000_p30'

# ── Helpers ────────────────────────────────────────────────────────────────────
def edge_recovery(ref_adj: np.ndarray, disc_adj: np.ndarray):
    """Return (precision, recall, f1) of disc_adj vs ref_adj."""
    ref  = (ref_adj  != 0).astype(int)
    disc = (disc_adj != 0).astype(int)
    tp = int((ref & disc).sum())
    fp = int(((disc == 1) & (ref == 0)).sum())
    fn = int(((ref  == 1) & (disc == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# =============================================================================
# 1.  Synthetic dataset metrics — X-only edges (Y node excluded)
# =============================================================================
# The raw adjacency from data generation is (51×51): rows/cols 0..49 = X, 50 = Y.
# Discovered train adjacency files are already X-only (50×50).
# We compare only X→X edges so Y-links do not inflate F1.
synth_true_full = np.load(BASE / 'data' / 'synthetic' / f'{SYNTH_DATASET}_adjacency.npy')
synth_true_xx   = synth_true_full[:50, :50]            # drop Y row/col

synth_pc_xx     = np.load(CAUSAL / f'{SYNTH_DATASET}_pc_train_adjacency.npy')
synth_lingam_xx = np.load(CAUSAL / f'{SYNTH_DATASET}_lingam_train_adjacency.npy')

_, _, synth_f1_pc     = edge_recovery(synth_true_xx, synth_pc_xx)
_, _, synth_f1_lingam = edge_recovery(synth_true_xx, synth_lingam_xx)

with open(MODELS / f'{SYNTH_DATASET}_metrics.json') as f:
    synth_metrics = json.load(f)
synth_r2 = synth_metrics['lgbm']['r2']

print(f"Synthetic | PC F1={synth_f1_pc:.3f}  LiNGAM F1={synth_f1_lingam:.3f}  R²={synth_r2:.4f}")


# =============================================================================
# 2.  Sachs metrics — X-only edges (akt/Y node excluded)
# =============================================================================
# true_full_adj is (11×11); the X-only block is [:10, :10] (akt excluded).
# sachs_{pc,lingam}_train_adjacency.npy are already X-only (10×10).
sachs_true_full = np.load(CAUSAL / 'sachs_true_full_adjacency.npy')
n_sachs_x       = 10                                   # 10 proteins, akt excluded
sachs_true_xx   = sachs_true_full[:n_sachs_x, :n_sachs_x]

sachs_pc_xx     = np.load(CAUSAL / 'sachs_pc_train_adjacency.npy')
sachs_lingam_xx = np.load(CAUSAL / 'sachs_lingam_train_adjacency.npy')

_, _, sachs_f1_pc     = edge_recovery(sachs_true_xx, sachs_pc_xx)
_, _, sachs_f1_lingam = edge_recovery(sachs_true_xx, sachs_lingam_xx)

# ── 2b. Train LGBM on sachs_train and evaluate R² on sachs_test ───────────────
sachs_train = pd.read_parquet(PROC / 'sachs_train.parquet')
sachs_test  = pd.read_parquet(PROC / 'sachs_test.parquet')

X_train = sachs_train.drop(columns=['Y'])
y_train = sachs_train['Y']
X_test  = sachs_test.drop(columns=['Y'])
y_test  = sachs_test['Y']

print("Training LGBM on Sachs …")
sachs_model = LGBMRegressor()
sachs_model.fit(X_train, y_train, feature_selection=False)
y_pred = sachs_model.predict(X_test)
sachs_r2 = r2_score(y_test, y_pred)

print(f"Sachs     | PC F1={sachs_f1_pc:.3f}  LiNGAM F1={sachs_f1_lingam:.3f}  R²={sachs_r2:.4f}")


# =============================================================================
# 3.  Build the figure
# =============================================================================
datasets   = ['Synthetic\n(Linear, Confounded)', 'Sachs']
f1_pc      = [synth_f1_pc,     sachs_f1_pc]
f1_lingam  = [synth_f1_lingam, sachs_f1_lingam]
r2_vals    = [synth_r2,        sachs_r2]

x      = np.arange(len(datasets))
width  = 0.28          # bar width
offset = 0.15          # separation between the two bars per group

# Colours
PC_COLOR     = '#1f77b4'   # blue
LINGAM_COLOR = '#ff7f0e'   # orange
R2_COLOR     = '#2ca02c'   # green

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# ── F1 bars ───────────────────────────────────────────────────────────────────
bars_pc     = ax1.bar(x - offset, f1_pc,     width, label='PC F1',     color=PC_COLOR,     alpha=0.85, zorder=3)
bars_lingam = ax1.bar(x + offset, f1_lingam, width, label='LiNGAM F1', color=LINGAM_COLOR, alpha=0.85, zorder=3)

# ── R² line + markers ─────────────────────────────────────────────────────────
line_r2, = ax2.plot(x, r2_vals, color=R2_COLOR, marker='D', markersize=9,
                    linewidth=2.0, linestyle='--', label='LGBM R²', zorder=4)

# ── Axis labels and limits ────────────────────────────────────────────────────
ax1.set_ylabel('F1 Score (causal discovery)', fontsize=12)
ax2.set_ylabel('R² (LGBM test performance)', fontsize=12, color=R2_COLOR)
ax2.tick_params(axis='y', labelcolor=R2_COLOR)

ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=12)
ax1.set_ylim(0, 1.05)
ax2.set_ylim(0, 1.05)
ax1.set_xlim(-0.6, len(datasets) - 0.4)

# ── Value annotations on bars ─────────────────────────────────────────────────
for bar, val in zip(bars_pc, f1_pc):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9, color=PC_COLOR, fontweight='bold')
for bar, val in zip(bars_lingam, f1_lingam):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=9, color=LINGAM_COLOR, fontweight='bold')
for xi, val in zip(x, r2_vals):
    ax2.text(xi + 0.07, val + 0.02, f'{val:.2f}', ha='left', va='bottom',
             fontsize=9, color=R2_COLOR, fontweight='bold')

# ── Grid ──────────────────────────────────────────────────────────────────────
ax1.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=PC_COLOR,     alpha=0.85, label='PC F1'),
    mpatches.Patch(color=LINGAM_COLOR, alpha=0.85, label='LiNGAM F1'),
    line_r2,
]
ax1.legend(handles=handles, loc='upper left', fontsize=10, framealpha=0.9)

ax1.set_title('Causal Discovery F1 vs Model R²\n'
              'Synthetic (Linear, Confounded, p=30) · Sachs Dataset',
              fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
out_path = BASE / 'notebooks' / 'plots' / 'discovery_vs_r2.png'
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved → {out_path}")
plt.show()
