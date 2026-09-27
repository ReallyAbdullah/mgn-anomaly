"""Gate K: bundle-outlier baseline after Kracker et al. (IJCrash 2023), adapted to simulations with different meshes.

Each frame -> mesh-independent feature vector (quantiles of node kinematics/smoothness in units of the run's step size,
plus inverted-element count and contact depth). Frame score = mean distance to the k nearest clean bank frames;
run score = mean frame score over the steady regime. See docs/gate_k.md.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mgn.data import NORMAL
from mgn.detect import contact_scores, jacobian_scores, laplacian_scores, velocity_laplacian_scores
from mgn.evaluate import RESULTS, choose_frames, seed
from mgn.inject import GLOBAL_SEVERITY, GLOBAL_TYPES, SEVERITY, TYPES, inject, inject_global, step_scale
from mgn.train import load_split

STEADY = np.arange(40, 340, 2)
Q = [50, 90, 99, 100]


def features(traj, wp, frames):
    frames = np.asarray(frames)
    normal = traj["node_type"] == NORMAL
    s = step_scale(dict(traj, world_pos=wp)) + 1e-12
    speed = np.linalg.norm(wp[frames] - wp[frames - 1], axis=-1)
    acc = np.linalg.norm(wp[frames] - 2 * wp[frames - 1] + wp[frames - 2], axis=-1)
    blocks = [speed, acc, laplacian_scores(traj, wp, frames), velocity_laplacian_scores(traj, wp, frames)]
    qs = [np.percentile(b[:, normal], Q, axis=1).T / s for b in blocks]
    inv = jacobian_scores(traj, wp, frames)[:, normal].sum(1, keepdims=True) / 4
    depth = contact_scores(traj, wp, frames).max(1, keepdims=True) / s
    return np.log1p(np.concatenate(qs + [inv, depth], 1))


class Bundle:
    def __init__(self, bank, k=5):
        self.mu, self.sd = bank.mean(0), bank.std(0) + 1e-9
        self.tree, self.k = cKDTree((bank - self.mu) / self.sd), k

    def score(self, X):
        d, _ = self.tree.query((X - self.mu) / self.sd, k=self.k)
        return d.mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=RESULTS / "gate_k")
    a = ap.parse_args()
    bank = np.concatenate([features(tr, tr["world_pos"], STEADY)
                           for tr in tqdm(load_split("valid", 70)[50:], desc="bank (valid 50-69)")])
    bundle = Bundle(bank)
    evals = load_split("valid", 50)[20:]
    sims = list(range(20, 50))

    # local failures: same copies and frames as the protocol v3 validation records
    V = pickle.load(open(RESULTS / "valid_v3" / "records.pkl", "rb"))["records"]
    ref = {(r["traj"], r["type"], r["sev"], r["t"]): r for r in V if 20 <= r["traj"] < 50}
    rows = []
    for i, tr in zip(sims, tqdm(evals, desc="local copies")):
        for kind in TYPES:
            for sev in SEVERITY:
                rng = np.random.default_rng(seed(i, kind, sev))
                wp, _, fm = inject(tr, kind, sev, rng)
                frames, labels, phase = choose_frames(fm, rng)
                sc = bundle.score(features(tr, wp, frames))
                for t, lab, ph, v in zip(frames, labels, phase, sc):
                    r = ref[(i, kind, sev, int(t))]
                    assert r["label"] == bool(lab)
                    rows.append(dict(type=kind, sev=sev, phase=str(ph), label=bool(lab), bundle=float(v),
                                     **{d: r[f"frame_{d}"] for d in ("gnn", "constvel", "laplacian", "vlap", "velocity",
                                                                     "jacobian", "contact")}))
    dets = ["bundle", "gnn", "constvel", "laplacian", "vlap", "velocity", "jacobian", "contact"]
    lines = ["**Local failures** — frame AUROC, validation sims 20–49, mean over 3 severities\n",
             "| failure | phase | " + " | ".join(dets) + " |", "|---|---|" + "---|" * len(dets)]
    res = {"local": {}, "global": {}}
    for kind in TYPES:
        for ph in ("onset", "sustained"):
            rs = [r for r in rows if r["type"] == kind and r["phase"] in (ph, "clean")]
            vals = {d: float(np.mean([roc_auc_score([r["label"] for r in rs if r["sev"] == s],
                                                    [r[d] for r in rs if r["sev"] == s]) for s in SEVERITY])) for d in dets}
            res["local"][f"{kind}/{ph}"] = vals
            lines.append(f"| {kind} | {ph} | " + " | ".join(f"{vals[d]:.2f}" for d in dets) + " |")

    # whole-run errors: run score = mean frame score over the steady regime
    run = {}
    for i, tr in zip(sims, tqdm(evals, desc="whole runs")):
        run[i, "clean", 0] = float(bundle.score(features(tr, tr["world_pos"], STEADY)).mean())
        for kind in GLOBAL_TYPES:
            for sev in GLOBAL_SEVERITY[kind]:
                run[i, kind, sev] = float(bundle.score(features(tr, inject_global(tr, kind, sev), STEADY)).mean())
    lines += ["\n**Whole-run errors** — run AUROC (clean vs corrupted), validation sims 20–49\n", "| error | severity | bundle |",
              "|---|---|---|"]
    for kind in GLOBAL_TYPES:
        for sev, p in GLOBAL_SEVERITY[kind].items():
            auc = roc_auc_score([0] * len(sims) + [1] * len(sims), [run[i, "clean", 0] for i in sims] + [run[i, kind, sev] for i in sims])
            res["global"][f"{kind}/{p}"] = float(auc)
            lines.append(f"| {kind} | {p} | {auc:.2f} |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate_k.json").write_text(json.dumps(res, indent=1))
    (a.out / "gate_k.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
