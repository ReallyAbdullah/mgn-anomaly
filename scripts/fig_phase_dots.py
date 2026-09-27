"""Figure: onset vs sustained frame AUROC per failure type and detector (validation sims 20-99, protocol v3)."""
import pickle
import sys

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/valid_v3/records.pkl", "rb"))["records"]
TYPES = ["hourglass", "penetration", "inversion", "instability", "frozen"]
DETS = [("gnn", "GNN surrogate", "#2a78d6", "o"), ("constvel", "Constant velocity", "#eb6834", "s"),
        ("laplacian", "Position Laplacian", "#1baf7a", "^"), ("vlap", "Velocity Laplacian", "#4a3aa7", "D")]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def auc(kind, phase, det):
    rs = [r for r in R if r["type"] == kind and r["phase"] in (phase, "clean")]
    return np.mean([roc_auc_score([r["label"] for r in rs if r["sev"] == s], [r[f"frame_{det}"] for r in rs if r["sev"] == s])
                    for s in (1, 2, 3)])


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": INK})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True, facecolor=SURFACE)
for ax, phase in zip(axes, ["onset", "sustained"]):
    ax.set_facecolor(SURFACE)
    for k, kind in enumerate(TYPES):
        y = len(TYPES) - 1 - k
        ax.axhline(y, color="#e6e5e0", lw=0.8, zorder=0)
        for j, (det, label, col, mk) in enumerate(DETS):
            ax.scatter(auc(kind, phase, det), y + (j - 1.5) * 0.14, s=46, marker=mk, color=col,
                       edgecolors=SURFACE, linewidths=1.2, zorder=3, label=label if k == 0 else None)
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(0.4, 1.02)
    ax.set_title(f"{phase.capitalize()} frames", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Frame AUROC (mean over 3 severities)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bdbcb6")
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(range(len(TYPES)), TYPES[::-1])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.55, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig("docs/report/figs/phase_dots.png", dpi=220, facecolor=SURFACE)
print("saved")
