"""Regenerate the README's onset-vs-sustained table from the test records (frame AUROC, mean over 3 severities).
Usage: uv run python scripts/phase_table.py [results/test/records.pkl]"""
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/test/records.pkl", "rb"))["records"]
DETS = ["gnn", "constvel", "laplacian", "velocity", "jacobian", "clip_knn"]
print("| anomaly | phase | " + " | ".join(DETS) + " |\n|---|---|" + "---|" * len(DETS))
for kind in ["hourglass", "penetration", "inversion", "instability", "frozen"]:
    for phase in ["onset", "sustained"]:
        rs = [r for r in R if r["type"] == kind and r["phase"] in (phase, "clean")]
        vals = [np.mean([roc_auc_score([r["label"] for r in rs if r["sev"] == s], [r[f"frame_{d}"] for r in rs if r["sev"] == s])
                         for s in (1, 2, 3)]) for d in DETS]
        print(f"| {kind} | {phase} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
