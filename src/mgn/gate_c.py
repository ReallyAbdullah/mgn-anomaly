"""Gate C (pre-registered in docs/gate_c.md): does the learned surrogate beat the best simple baseline?

Primary, per GNN variant (non-causal `gnn` vs. best non-causal baseline; `gnn_causal` vs. best causal baseline):
  family = sustained-phase frame AUROC on {hourglass, inversion, frozen} x severity {1, 2, 3};
  the best baseline per cell is chosen on VALIDATION (sustained phase); statistic = mean over the 9 cells of
  AUROC(GNN) - AUROC(best baseline) on TEST, with a paired bootstrap over test simulations (2000 resamples).
  Claim "learned physics helps" only if the 95% CI lower bound is > 0.
Secondary (exploratory): all 15 type x severity cells, all event frames, one-sided paired-bootstrap p-values,
  Holm-corrected at 0.05.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from mgn.evaluate import RESULTS, best_baselines
from mgn.inject import SEVERITY, TYPES

PRIMARY = [(k, s) for k in ("hourglass", "inversion", "frozen") for s in SEVERITY]


def auc(recs, det, phase):
    R = [r for r in recs if r["phase"] == "clean" or phase == "all" or r["phase"] == phase]
    return roc_auc_score([r["label"] for r in R], [r[f"frame_{det}"] for r in R])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, default=RESULTS / "test" / "records.pkl")
    p.add_argument("--valid-metrics", type=Path, default=RESULTS / "valid" / "metrics.csv")
    p.add_argument("--n-boot", type=int, default=2000)
    a = p.parse_args()
    d = pickle.load(open(a.test, "rb"))
    assert d["meta"]["split"] == "test" and d["meta"]["protocol"] == "v2"
    sims = np.array(d["meta"]["sims"])
    cells = {}
    for r in d["records"]:
        cells.setdefault((r["type"], r["sev"]), {}).setdefault(r["traj"], []).append(r)
    rng = np.random.default_rng(0)
    boots = [rng.choice(sims, len(sims)) for _ in range(a.n_boot)]
    gather = lambda cell, take: [r for s in take for r in cells[cell][s]]
    lines = [f"Gate C on {len(sims)} test sims (checkpoint {d['meta']['ckpt_sha256'][:12]}), "
             f"best baselines chosen on {a.valid_metrics}\n"]

    for variant in ("gnn", "gnn_causal"):
        if f"frame_{variant}" not in d["records"][0]:
            lines.append(f"({variant} not in records — skipped)\n")
            continue
        best = best_baselines(a.valid_metrics, "sustained")
        diff = lambda take: np.mean([auc(gather(c, take), variant, "sustained") - auc(gather(c, take), best[c][variant], "sustained")
                                     for c in PRIMARY])
        point, bs = diff(sims), [diff(t) for t in boots]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        lines.append(f"**Primary — {variant}** (sustained phase, hourglass/inversion/frozen x 3 severities): "
                     f"mean AUROC difference vs best baseline {point:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] -> "
                     + ("**learned physics helps**" if lo > 0 else "**no demonstrated benefit**"))
        lines.append("  best baselines per cell: " + ", ".join(f"{k}/{s}={best[k, s][variant]}" for k, s in PRIMARY) + "\n")

        # secondary: every cell, all event frames, Holm-corrected one-sided p-values
        best_all = best_baselines(a.valid_metrics, "all")
        rows = []
        for c in [(k, s) for k in TYPES for s in SEVERITY]:
            f = lambda take: auc(gather(c, take), variant, "all") - auc(gather(c, take), best_all[c][variant], "all")
            bs = np.array([f(t) for t in boots])
            rows.append((c, best_all[c][variant], f(sims), (np.sum(bs <= 0) + 1) / (len(bs) + 1)))
        order = np.argsort([r[3] for r in rows])
        m, holm_ok, sig = len(rows), True, set()
        for rank, i in enumerate(order):  # Holm step-down
            if holm_ok and rows[i][3] <= 0.05 / (m - rank):
                sig.add(i)
            else:
                holm_ok = False
        lines.append(f"Secondary — {variant} vs best baseline per cell (all event frames; * = Holm-significant):\n")
        lines.append("| cell | baseline | Δ AUROC | one-sided p |\n|---|---|---|---|")
        for i, (c, b, dlt, pv) in enumerate(rows):
            lines.append(f"| {c[0]}/{c[1]} | {b} | {dlt:+.3f}{' *' if i in sig else ''} | {pv:.4f} |")
        lines.append("")
    out = a.test.parent / "gate_c.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
