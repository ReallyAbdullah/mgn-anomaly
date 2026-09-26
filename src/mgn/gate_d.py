"""Gate D (pre-registered in docs/gate_d.md): does a surrogate beat the best rule on whole-run loading errors?"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from mgn.evaluate import RESULTS
from mgn.inject import GLOBAL_SEVERITY
from mgn.runlevel import RULE_STATS, SURROGATE_STATS, auroc_table
from mgn.train import RUNS


def shape_admitted(log):
    """Final val shape RMSE <= 0.5 x zero-displacement baseline."""
    m = re.findall(r"final val shape RMSE ([\d.e+-]+) \(baseline ([\d.e+-]+)\)", Path(log).read_text())
    return bool(m) and float(m[-1][0]) <= 0.5 * float(m[-1][1]), m[-1] if m else None


def holm(pvals, alpha=0.05):
    order, sig, m = np.argsort(pvals), set(), len(pvals)
    for rank, i in enumerate(order):
        if pvals[i] > alpha / (m - rank):
            break
        sig.add(i)
    return sig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=RESULTS / "runlevel" / "runs_valid.pkl")
    p.add_argument("--shape-log", type=Path, default=RUNS.parent / "runs" / "shape_train.log")
    a = p.parse_args()
    d = pickle.load(open(a.runs, "rb"))
    ok, rmse = shape_admitted(a.shape_log)
    surrogates = [s for s in SURROGATE_STATS if s in d["rows"][0] and (ok or not s.startswith("shape"))]
    rules = [s for s in RULE_STATS if s in d["rows"][0]]
    table, _ = auroc_table(d["rows"], surrogates + rules)
    tests = []
    for (kind, sev), v in table.items():
        best = max(rules, key=lambda s: v[s]["point"])
        for s in surrogates:
            delta = v[s]["boots"] - v[best]["boots"]
            tests.append(dict(cell=f"{kind}/{GLOBAL_SEVERITY[kind][sev]}", stat=s, rule=best,
                              delta=v[s]["point"] - v[best]["point"], lo=np.percentile(delta, 2.5),
                              p=(np.sum(delta <= 0) + 1) / (len(delta) + 1)))
    sig = holm(np.array([t["p"] for t in tests]))
    lines = [f"Gate D on {d['meta']}; shape model admitted: {ok} (final RMSE, baseline = {rmse}); "
             f"surrogate stats: {surrogates}\n",
             "| cell | surrogate stat | best rule | Δ AUROC | CI low | one-sided p | Holm |", "|---|---|---|---|---|---|---|"]
    for i, t in enumerate(tests):
        lines.append(f"| {t['cell']} | {t['stat']} | {t['rule']} | {t['delta']:+.3f} | {t['lo']:+.3f} | {t['p']:.4f} | "
                     f"{'*' if i in sig and t['delta'] > 0 else ''} |")
    passed = any(tests[i]["delta"] > 0 for i in sig)
    lines.append(f"\n**Gate D: {'PASS' if passed else 'FAIL'}**")
    (a.runs.parent / "gate_d.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
