"""Build the injected-anomaly benchmark, run all detectors, and report AUROC with bootstrap CIs.

Protocol v2, per simulation x anomaly type x severity (one corrupted copy each):
  - every anomalous frame, labelled `onset` (first frame of the event) or `sustained` (the rest);
  - as many clean frames (min. 4) from the same copy, drawn from within 40 frames of the event (so clean and
    anomalous frames share the same phase of the loading) but outside a +-3-frame guard band (frames right after an
    event are fed corrupted inputs, so they are neither clearly clean nor anomalous);
  - residual detectors are calibrated per node by its median residual over every 10th frame (unsupervised:
    the copy is mostly clean); `*_causal` variants use only calibration frames before the scored frame, as an online
    monitor would. CLIP scores are already relative to a clean reference bank.
"""
import argparse
import hashlib
import os
import pickle
import subprocess

from pathlib import Path

import matplotlib
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mgn.data import NORMAL, T_MAX
from mgn.detect import (ClipDetector, constvel_scores, frame_score, gnn_scores, jacobian_scores, laplacian_scores,
                        velocity_scores)
from mgn.inject import SEVERITY, TYPES, inject
from mgn.render import SIZE, Renderer, grid_cell
from mgn.train import RUNS, device, load_model, load_split

PROTOCOL = "v2"
MIN_SIMS_FOR_CI = 20

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = RUNS.parent / "results"
DETECTORS = ["gnn", "constvel", "laplacian", "velocity", "jacobian", "clip_knn", "clip_zero",
             "gnn_causal", "constvel_causal", "laplacian_causal"]
CALIB = np.arange(5, T_MAX, 10)


def seed(i, kind, sev):
    return i * 100 + TYPES.index(kind) * 10 + sev


def choose_frames(fmask, rng, window=40):
    """All event frames (phase onset/sustained) + time-matched clean frames outside a +-3 guard band."""
    ev = np.flatnonzero(fmask[:T_MAX])
    ev = ev[ev >= 2]
    ok = np.zeros(T_MAX, bool)
    ok[max(ev[0] - window, 2):ev[-1] + window + 1] = True
    ok[max(ev[0] - 3, 0):ev[-1] + 4] = False
    cand = np.flatnonzero(ok)
    clean = np.sort(rng.choice(cand, min(max(4, len(ev)), len(cand)), replace=False))
    phase = np.array(["onset"] + ["sustained"] * (len(ev) - 1) + ["clean"] * len(clean))
    return np.concatenate([ev, clean]), phase != "clean", phase


def node_auroc(scores, mask):
    return roc_auc_score(mask, scores)


def score_copy(model, tr, wp, frames, dev):
    """Frame scores {detector: [F]} and node scores {detector: [F, N]} for the residual/rule detectors."""
    normal = tr["node_type"] == NORMAL
    node = {"gnn": gnn_scores(model, tr, wp, list(frames) + list(CALIB), dev),
            "constvel": constvel_scores(wp, np.r_[frames, CALIB]),
            "laplacian": laplacian_scores(tr, wp, np.r_[frames, CALIB]),
            "velocity": velocity_scores(wp, frames),
            "jacobian": jacobian_scores(tr, wp, frames)}
    fs = {}
    for d in ("gnn", "constvel", "laplacian"):
        raw, calib = node[d][:len(frames)], node[d][len(frames):]
        # per-node calibration (PaDiM-style): residual relative to that node's median residual
        cal = np.median(calib, axis=0)
        node[d] = raw / (cal + np.median(cal[normal]) + 1e-12)
        fs[d] = frame_score(node[d], normal)
        # causal variant: calibrate only on frames at least 4 before the scored one (min. first 3)
        causal = []
        for j, t in enumerate(frames):
            past = CALIB < t - 3
            c = np.median(calib[past if past.sum() >= 3 else slice(0, 3)], axis=0)
            causal.append(raw[j] / (c + np.median(c[normal]) + 1e-12))  # floor: pre-contact frames are all zero
        node[f"{d}_causal"] = np.stack(causal)
        fs[f"{d}_causal"] = frame_score(node[f"{d}_causal"], normal)
    fs["velocity"] = frame_score(node["velocity"], normal)
    fs["jacobian"] = node["jacobian"][:, normal].sum(1) / 4  # inverted-tet count
    return fs, node


def run(a):
    dev = device(a.device)
    model = load_model(a.ckpt, dev)
    sims = list(range(a.start, a.stop))
    test = load_split(a.split, a.stop)[a.start:]
    clip = None
    if not a.no_clip:
        clip = ClipDetector(dev)
        refs = []
        for tr in load_split("valid", 20):
            r = Renderer(tr)
            refs += [r.render(tr["world_pos"][t]) for t in (50, 120, 190, 260, 330)]
        clip.fit(refs)

    records = []
    for i, tr in zip(sims, tqdm(test, desc=f"{a.split} sims")):
        normal = tr["node_type"] == NORMAL
        rend = Renderer(tr) if clip else None
        for kind in TYPES:
            for sev in SEVERITY:
                rng = np.random.default_rng(seed(i, kind, sev))
                wp, mask, fmask = inject(tr, kind, sev, rng)
                frames, labels, phase = choose_frames(fmask, rng)
                fs, node = score_copy(model, tr, wp, frames, dev)
                if clip:
                    images = [rend.render(wp[t]) for t in frames]
                    fs["clip_zero"], maps = clip.score(images)
                    fs["clip_knn"] = maps.reshape(len(frames), -1).max(1)
                    node["clip_knn"] = np.stack([clip.node_scores(m, rend.project(wp[t]), SIZE)
                                                 for m, t in zip(maps, frames)])
                for j, (t, lab) in enumerate(zip(frames, labels)):
                    rec = dict(traj=i, type=kind, sev=sev, t=int(t), label=bool(lab), phase=str(phase[j]),
                               **{f"frame_{d}": float(v[j]) for d, v in fs.items()})
                    if lab:
                        for d, v in node.items():
                            rec[f"node_{d}"] = node_auroc(v[j][normal], mask[normal])
                        if rend:
                            rec["cell"] = grid_cell(rend.project(wp[t][mask]))
                            shift = rend.project(wp[t][mask]) - rend.project(tr["world_pos"][t][mask])
                            rec["px"] = float(np.linalg.norm(shift, axis=1).max())  # visible size in the render
                    records.append(rec)
    meta = dict(protocol=PROTOCOL, split=a.split, sims=sims, ckpt=a.ckpt, clip=clip is not None,
                ckpt_sha256=hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest(),
                commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "records.pkl", "wb") as f:
        pickle.dump(dict(meta=meta, records=records), f)
    return meta, records


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


def cell_rows(records, dets, kind, sev, phase=None):
    """Metrics for one type x severity; phase restricts the positives to onset or sustained frames."""
    R = [r for r in records if r["type"] == kind and r["sev"] == sev and (phase is None or r["phase"] in (phase, "clean"))]
    y = np.array([r["label"] for r in R])
    g = np.array([r["traj"] for r in R])
    if y.all() or not y.any():
        return []
    ci = len(np.unique(g)) >= MIN_SIMS_FOR_CI
    boot = (lambda stat: bootstrap(stat, g)) if ci else (lambda stat: (stat(np.arange(len(g))), np.nan, np.nan))
    S = {d: np.array([r[f"frame_{d}"] for r in R]) for d in dets}
    rows = []
    for d in dets:
        s = S[d]
        auc, lo, hi = boot(lambda i: roc_auc_score(y[i], s[i]))
        nodes = [r[f"node_{d}"] for r in R if r["label"] and f"node_{d}" in r]
        row = dict(type=kind, sev=sev, phase=phase or "all", detector=d, frame_auroc=auc, lo=lo, hi=hi,
                   tpr5=tpr_at_fpr(y, s), node_auroc=np.mean(nodes) if nodes else np.nan,
                   delta=np.nan, dlo=np.nan, dhi=np.nan)
        if d != "gnn" and "gnn" in S:  # paired bootstrap of the AUROC difference, GNN minus this detector
            row["delta"], row["dlo"], row["dhi"] = boot(
                lambda i: roc_auc_score(y[i], S["gnn"][i]) - roc_auc_score(y[i], s[i]))
        rows.append(row)
    return rows


CAUSAL_POOL = {"constvel_causal", "laplacian_causal", "jacobian"}  # causal or memoryless baselines
NONCAUSAL_POOL = {"constvel", "laplacian", "velocity", "jacobian", "clip_knn", "clip_zero"}


def best_baselines(metrics_csv, phase="all"):
    """Per type x severity, the best baseline in a (validation) metrics.csv for each GNN variant, like with like:
    {(type, sev): {"gnn": best non-causal baseline, "gnn_causal": best causal baseline}}. GNN variants never qualify."""
    best = {}
    for line in open(metrics_csv):
        if line.startswith("#") or line.startswith("type,"):
            continue
        kind, sev, ph, det, auc = line.split(",")[:5]
        if ph != phase:
            continue
        for variant, pool in (("gnn", NONCAUSAL_POOL), ("gnn_causal", CAUSAL_POOL)):
            cur = best.setdefault((kind, int(sev)), {}).get(variant, ("", -1))
            if det in pool and float(auc) > cur[1]:
                best[kind, int(sev)][variant] = (det, float(auc))
    return {k: {v: d[0] for v, d in vs.items()} for k, vs in best.items()}


def report(meta, records, out, best=None):
    dets = [d for d in DETECTORS if f"frame_{d}" in records[0]]
    rows, pixels = [], {}
    for kind in TYPES:
        for sev in SEVERITY:
            for phase in (None, "onset", "sustained"):
                rows += cell_rows(records, dets, kind, sev, phase)
            R = [r for r in records if r["type"] == kind and r["sev"] == sev]
            px = [r["px"] for r in R if "px" in r]
            pixels[kind, sev] = np.median(px) if px else np.nan
    with open(out / "metrics.csv", "w") as f:
        f.write(f"# {meta}\n")
        f.write("type,severity,phase,detector,frame_auroc,ci_lo,ci_hi,tpr_at_5fpr,node_auroc,gnn_minus_det,d_lo,d_hi,"
                "median_px\n")
        for r in rows:
            f.write(f"{r['type']},{r['sev']},{r['phase']},{r['detector']},{r['frame_auroc']:.4f},{r['lo']:.4f},{r['hi']:.4f},"
                    f"{r['tpr5']:.4f},{r['node_auroc']:.4f},{r['delta']:.4f},{r['dlo']:.4f},{r['dhi']:.4f},"
                    f"{pixels[r['type'], r['sev']]:.2f}\n")

    # markdown table: frame AUROC per type (rows) x detector (cols), one block per severity
    rows_all = [r for r in rows if r["phase"] == "all"]
    lines = [f"{len(meta['sims'])} {meta['split']} sims, protocol {meta['protocol']}, checkpoint "
             f"{meta['ckpt_sha256'][:12]}, commit {meta['commit']}" + ("" if len(meta["sims"]) >= MIN_SIMS_FOR_CI
                                                                        else " — too few sims, no CIs")]
    for sev in SEVERITY:
        lines.append(f"\n**Severity {sev}** ({SEVERITY[sev]:g}x RMS step displacement) — frame AUROC [95% CI] / node AUROC; px = median visible shift in the 448 px render\n")
        extra = " | best baseline (chosen on validation) | GNN − best" if best else ""
        lines.append("| anomaly | px | " + " | ".join(dets) + " | GNN − constvel" + extra + " |")
        lines.append("|---|---|" + "---|" * (len(dets) + 1 + 2 * bool(best)))
        for kind in TYPES:
            cells = []
            for d in dets:
                r = next(r for r in rows_all if r["type"] == kind and r["sev"] == sev and r["detector"] == d)
                node = "" if np.isnan(r["node_auroc"]) else f" / {r['node_auroc']:.2f}"
                cells.append(f"{r['frame_auroc']:.2f} [{r['lo']:.2f}–{r['hi']:.2f}]{node}")
            dc = next(r for r in rows_all if r["type"] == kind and r["sev"] == sev and r["detector"] == "constvel")
            tail = ""
            if best:
                b = next(r for r in rows_all if r["type"] == kind and r["sev"] == sev and r["detector"] == best[kind, sev]["gnn"])
                tail = f" | {b['detector']} | {b['delta']:+.2f} [{b['dlo']:+.2f}, {b['dhi']:+.2f}]"
            lines.append(f"| {kind} | {pixels[kind, sev]:.1f} | " + " | ".join(cells)
                         + f" | {dc['delta']:+.2f} [{dc['dlo']:+.2f}, {dc['dhi']:+.2f}]{tail} |")
    # onset vs. sustained: where does learned physics help?
    lines.append("\n**Frame AUROC by event phase** (onset = first anomalous frame, sustained = the rest; clean frames shared)\n")
    lines.append("| anomaly | severity | phase | " + " | ".join(dets) + " |")
    lines.append("|---|---|---|" + "---|" * len(dets))
    for kind in TYPES:
        for sev in SEVERITY:
            for phase in ("onset", "sustained"):
                rr = {r["detector"]: r for r in rows if r["type"] == kind and r["sev"] == sev and r["phase"] == phase}
                if rr:
                    lines.append(f"| {kind} | {sev} | {phase} | " + " | ".join(f"{rr[d]['frame_auroc']:.2f}" for d in dets) + " |")
    (out / "metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, axes = plt.subplots(1, len(TYPES), figsize=(4 * len(TYPES), 3.4), sharey=True)
    for ax, kind in zip(axes, TYPES):
        for d in dets:
            rr = [r for r in rows_all if r["type"] == kind and r["detector"] == d]
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
    fig.savefig(out / "auroc_vs_severity.png", dpi=150)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--start", type=int, default=8, help="test sims 0-7 were used for pilots; start after them")
    p.add_argument("--stop", type=int, default=100)
    p.add_argument("--device")
    p.add_argument("--out", type=Path, default=RESULTS)
    p.add_argument("--no-clip", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--best-from", type=Path, help="validation metrics.csv used to pick the best baseline per cell")
    a = p.parse_args()
    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 4)))
    if a.report_only:
        with open(a.out / "records.pkl", "rb") as f:
            d = pickle.load(f)
        meta, records = d["meta"], d["records"]
    else:
        meta, records = run(a)
    report(meta, records, a.out, best_baselines(a.best_from) if a.best_from else None)


if __name__ == "__main__":
    main()
