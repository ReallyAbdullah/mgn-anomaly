"""Build the injected-anomaly benchmark, run all detectors, and report AUROC with bootstrap CIs.

Protocol, per test trajectory x anomaly type x severity (one corrupted copy each):
  - up to 4 anomalous frames (incl. onset) and 4 clean frames from the same copy, outside a +-3-frame guard band
    (frames right after an event are fed corrupted inputs, so they are neither clearly clean nor anomalous);
  - residual detectors are calibrated per node by its median residual over every 10th frame (unsupervised:
    the copy is mostly clean). CLIP scores are already relative to a clean reference bank.
"""
import argparse
import pickle

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mgn.data import NORMAL, T_MAX
from mgn.detect import (ClipDetector, constvel_scores, frame_score, gnn_scores, jacobian_scores, laplacian_scores,
                        velocity_scores)
from mgn.inject import SEVERITY, TYPES, inject
from mgn.render import SIZE, Renderer, grid_cell
from mgn.train import RUNS, device, load_model, load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = RUNS.parent / "results"
DETECTORS = ["gnn", "constvel", "laplacian", "velocity", "jacobian", "clip_knn", "clip_zero"]
CALIB = np.arange(5, T_MAX, 10)


def seed(i, kind, sev):
    return i * 100 + TYPES.index(kind) * 10 + sev


def choose_frames(fmask, rng):
    ev = np.flatnonzero(fmask[:T_MAX])
    ev = ev[ev >= 2]
    anom = ev[np.unique(np.linspace(0, len(ev) - 1, min(4, len(ev))).astype(int))]
    ok = np.ones(T_MAX, bool)
    ok[:2] = False
    ok[max(ev[0] - 3, 0):ev[-1] + 4] = False
    clean = rng.choice(np.flatnonzero(ok), 4, replace=False)
    return np.concatenate([anom, clean]), np.r_[np.ones(len(anom)), np.zeros(4)].astype(bool)


def node_auroc(scores, mask):
    return roc_auc_score(mask, scores)


def run(a):
    dev = device()
    model = load_model(a.ckpt, dev)
    test = load_split("test", a.n_traj)
    clip = None
    if not a.no_clip:
        clip = ClipDetector(dev)
        refs = []
        for tr in load_split("valid", 20):
            r = Renderer(tr)
            refs += [r.render(tr["world_pos"][t]) for t in (50, 120, 190, 260, 330)]
        clip.fit(refs)

    records = []
    for i, tr in enumerate(tqdm(test, desc="trajectories")):
        normal = tr["node_type"] == NORMAL
        rend = Renderer(tr) if clip else None
        for kind in TYPES:
            for sev in SEVERITY:
                rng = np.random.default_rng(seed(i, kind, sev))
                wp, mask, fmask = inject(tr, kind, sev, rng)
                frames, labels = choose_frames(fmask, rng)
                node = {"gnn": gnn_scores(model, tr, wp, list(frames) + list(CALIB), dev),
                        "constvel": constvel_scores(wp, np.r_[frames, CALIB]),
                        "laplacian": laplacian_scores(tr, wp, np.r_[frames, CALIB]),
                        "velocity": velocity_scores(wp, frames),
                        "jacobian": jacobian_scores(tr, wp, frames)}
                fs = {}
                for d in ("gnn", "constvel", "laplacian"):
                    # per-node calibration (PaDiM-style): residual relative to that node's median residual
                    cal = np.median(node[d][len(frames):], axis=0)
                    node[d] = node[d][:len(frames)] / (cal + np.median(cal[normal]))
                    fs[d] = frame_score(node[d], normal)
                fs["velocity"] = frame_score(node["velocity"], normal)
                fs["jacobian"] = node["jacobian"][:, normal].sum(1) / 4  # inverted-tet count
                if clip:
                    images = [rend.render(wp[t]) for t in frames]
                    fs["clip_zero"], maps = clip.score(images)
                    fs["clip_knn"] = maps.reshape(len(frames), -1).max(1)
                    node["clip_knn"] = np.stack([clip.node_scores(m, rend.project(wp[t]), SIZE)
                                                 for m, t in zip(maps, frames)])
                for j, (t, lab) in enumerate(zip(frames, labels)):
                    rec = dict(traj=i, type=kind, sev=sev, t=int(t), label=bool(lab),
                               **{f"frame_{d}": float(v[j]) for d, v in fs.items()})
                    if lab:
                        for d, v in node.items():
                            rec[f"node_{d}"] = node_auroc(v[j][normal], mask[normal])
                        if rend:
                            rec["cell"] = grid_cell(rend.project(wp[t][mask]))
                            shift = rend.project(wp[t][mask]) - rend.project(tr["world_pos"][t][mask])
                            rec["px"] = float(np.linalg.norm(shift, axis=1).max())  # visible size in the render
                    records.append(rec)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "records.pkl", "wb") as f:
        pickle.dump(records, f)
    return records


def bootstrap(stat, groups, n=500, seed=0):
    """Point estimate + 95% CI of stat(indices), resampling whole simulations with replacement."""
    rng = np.random.default_rng(seed)
    idx = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(idx)
    boots = []
    for _ in range(n):
        take = np.concatenate([idx[keys[k]] for k in rng.integers(len(keys), size=len(keys))])
        try:
            boots.append(stat(take))
        except ValueError:  # resample with a single class
            pass
    return stat(np.arange(len(groups))), *np.percentile(boots, [2.5, 97.5])


def tpr_at_fpr(y, s, fpr=0.05):
    thr = np.quantile(s[~y], 1 - fpr)
    return (s[y] > thr).mean()


def report(records):
    dets = [d for d in DETECTORS if f"frame_{d}" in records[0]]
    rows, pixels = [], {}
    for kind in TYPES:
        for sev in SEVERITY:
            R = [r for r in records if r["type"] == kind and r["sev"] == sev]
            y = np.array([r["label"] for r in R])
            g = np.array([r["traj"] for r in R])
            S = {d: np.array([r[f"frame_{d}"] for r in R]) for d in dets}
            for d in dets:
                s = S[d]
                auc, lo, hi = bootstrap(lambda i: roc_auc_score(y[i], s[i]), g)
                nodes = [r[f"node_{d}"] for r in R if r["label"] and f"node_{d}" in r]
                row = dict(type=kind, sev=sev, detector=d, frame_auroc=auc, lo=lo, hi=hi, tpr5=tpr_at_fpr(y, s),
                           node_auroc=np.mean(nodes) if nodes else np.nan, delta=np.nan, dlo=np.nan, dhi=np.nan)
                if d != "gnn":  # paired bootstrap of the AUROC difference, GNN minus this detector
                    row["delta"], row["dlo"], row["dhi"] = bootstrap(
                        lambda i: roc_auc_score(y[i], S["gnn"][i]) - roc_auc_score(y[i], s[i]), g)
                rows.append(row)
            px = [r["px"] for r in R if "px" in r]
            pixels[kind, sev] = np.median(px) if px else np.nan
    with open(RESULTS / "metrics.csv", "w") as f:
        f.write("type,severity,detector,frame_auroc,ci_lo,ci_hi,tpr_at_5fpr,node_auroc,gnn_minus_det,d_lo,d_hi,"
                "median_px\n")
        for r in rows:
            f.write(f"{r['type']},{r['sev']},{r['detector']},{r['frame_auroc']:.4f},{r['lo']:.4f},{r['hi']:.4f},"
                    f"{r['tpr5']:.4f},{r['node_auroc']:.4f},{r['delta']:.4f},{r['dlo']:.4f},{r['dhi']:.4f},"
                    f"{pixels[r['type'], r['sev']]:.2f}\n")

    # markdown table: frame AUROC per type (rows) x detector (cols), one block per severity
    lines = []
    for sev in SEVERITY:
        lines.append(f"\n**Severity {sev}** ({SEVERITY[sev]:g}x RMS step displacement) — frame AUROC [95% CI] / node AUROC; px = median visible shift in the 448 px render\n")
        lines.append("| anomaly | px | " + " | ".join(dets) + " | GNN − constvel |")
        lines.append("|---|---|" + "---|" * (len(dets) + 1))
        for kind in TYPES:
            cells = []
            for d in dets:
                r = next(r for r in rows if r["type"] == kind and r["sev"] == sev and r["detector"] == d)
                node = "" if np.isnan(r["node_auroc"]) else f" / {r['node_auroc']:.2f}"
                cells.append(f"{r['frame_auroc']:.2f} [{r['lo']:.2f}–{r['hi']:.2f}]{node}")
            dc = next(r for r in rows if r["type"] == kind and r["sev"] == sev and r["detector"] == "constvel")
            lines.append(f"| {kind} | {pixels[kind, sev]:.1f} | " + " | ".join(cells)
                         + f" | {dc['delta']:+.2f} [{dc['dlo']:+.2f}, {dc['dhi']:+.2f}] |")
    (RESULTS / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(TYPES), figsize=(4 * len(TYPES), 3.4), sharey=True)
    for ax, kind in zip(axes, TYPES):
        for d in dets:
            rr = [r for r in rows if r["type"] == kind and r["detector"] == d]
            x = [SEVERITY[r["sev"]] for r in rr]
            ax.plot(x, [r["frame_auroc"] for r in rr], marker="o", label=d)
            ax.fill_between(x, [r["lo"] for r in rr], [r["hi"] for r in rr], alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(kind)
        ax.set_xlabel("severity (x step disp.)")
        ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    axes[0].set_ylabel("frame AUROC")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "auroc_vs_severity.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--n-traj", type=int, default=40)
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--report-only", action="store_true")
    a = p.parse_args()
    if a.report_only:
        with open(RESULTS / "records.pkl", "rb") as f:
            records = pickle.load(f)
    else:
        records = run(a)
    report(records)


if __name__ == "__main__":
    main()
