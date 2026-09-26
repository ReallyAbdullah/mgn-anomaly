"""Run-level detection of whole-run, loading-inconsistent errors ("is this simulation run wrong?").

Within-run per-node calibration would absorb a whole-run error, so every detector is reduced to a run statistic
without units, computed the same way for all of them on steady frames [40, 340) (every 2nd frame):
- residual/rule detectors: median over frames of the raw frame score (top-k plate nodes), divided by the run's own
  RMS per-step displacement
- shape surrogate: log of the least-squares scale of observed on predicted plate displacement (~ log(1+eps) for
  `scale`), and the relative residual RMS |u_obs - u_pred| / |u_pred|
- contact: median over frames of the minimum plate-actuator signed distance, and the max penetration depth, both
  divided by the run's RMS actuator step
Each statistic is turned into a two-sided robust z-score against CLEAN reference runs from other simulations
(validation 50-69 by default); the run score is |z|. AUROC is over runs (clean vs. corrupted), bootstrapped over sims.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mgn.data import NORMAL, OBSTACLE, T_MAX
from mgn.detect import constvel_scores, frame_score, gnn_scores, laplacian_scores, velocity_laplacian_scores, velocity_scores
from mgn.evaluate import RESULTS
from mgn.inject import GLOBAL_SEVERITY, GLOBAL_TYPES, actuator_distance, inject_global, step_scale
from mgn.train import RUNS, device, load_model, load_split

FRAMES = np.arange(40, T_MAX - 10, 2)
SURROGATE_STATS = ["shape_resid", "shape_scale", "gnn"]
RULE_STATS = ["constvel", "laplacian", "vlap", "velocity", "contact_gap", "contact_depth"]


def run_stats(traj, wp, gnn, shape_model, dev):
    normal = traj["node_type"] == NORMAL
    step = step_scale(dict(traj, world_pos=wp))
    act = traj["node_type"] == OBSTACLE
    act_step = np.sqrt((np.linalg.norm(np.diff(wp[:T_MAX][:, act], axis=0), axis=-1) ** 2).mean()) + 1e-12
    med = lambda node: float(np.median(frame_score(node, normal)))
    st = {"constvel": med(constvel_scores(wp, FRAMES)) / step,
          "laplacian": med(laplacian_scores(traj, wp, FRAMES)) / step,
          "vlap": med(velocity_laplacian_scores(traj, wp, FRAMES)) / step,
          "velocity": med(velocity_scores(wp, FRAMES))}
    plate = np.flatnonzero(normal)
    d = np.stack([actuator_distance(traj, wp[t], plate) for t in FRAMES])
    st["contact_gap"] = float(np.median(d.min(1))) / act_step
    st["contact_depth"] = float(np.maximum(0, -d).max()) / act_step
    if gnn is not None:
        st["gnn"] = med(gnn_scores(gnn, traj, wp, list(FRAMES), dev)) / step
    if shape_model is not None:
        from mgn.shape import predict_run
        up = predict_run(shape_model, traj, wp, list(FRAMES), dev)[:, normal]
        uo = (wp[FRAMES] - traj["mesh_pos"])[:, normal]
        st["shape_scale"] = float(np.log(max((uo * up).sum() / (up * up).sum(), 1e-6)))
        st["shape_resid"] = float(np.sqrt(((uo - up) ** 2).sum() / (up ** 2).sum()))
    return st


def score_runs(sims, split, gnn, shape_model, dev, corrupt=True):
    rows = []
    for i, tr in zip(sims, tqdm(load_split(split, max(sims) + 1)[min(sims):], desc=f"{split} runs")):
        rows.append(dict(sim=i, type="clean", sev=0, **run_stats(tr, tr["world_pos"], gnn, shape_model, dev)))
        if corrupt:
            for kind in GLOBAL_TYPES:
                for sev in GLOBAL_SEVERITY[kind]:
                    rows.append(dict(sim=i, type=kind, sev=sev,
                                     **run_stats(tr, inject_global(tr, kind, sev), gnn, shape_model, dev)))
    return rows


def zscore(rows, ref, stats):
    """|robust z| of each run statistic against the clean reference runs."""
    for s in stats:
        v = np.array([r[s] for r in ref])
        med, mad = np.median(v), np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-12
        for r in rows:
            r[f"z_{s}"] = abs(r[s] - med) / mad
    return rows


def auroc_table(rows, stats, n_boot=2000, seed=0):
    """Per (type, sev): AUROC of |z| for each stat, point + bootstrap over sims (clean and corrupted resampled together)."""
    sims = np.array(sorted({r["sim"] for r in rows}))
    by = {(r["sim"], r["type"], r["sev"]): r for r in rows}
    rng = np.random.default_rng(seed)
    boots = [rng.choice(sims, len(sims)) for _ in range(n_boot)]
    out = {}
    for kind in GLOBAL_TYPES:
        for sev in GLOBAL_SEVERITY[kind]:
            def auc(take, s):
                y = [0] * len(take) + [1] * len(take)
                x = [by[i, "clean", 0][f"z_{s}"] for i in take] + [by[i, kind, sev][f"z_{s}"] for i in take]
                return roc_auc_score(y, x)
            out[kind, sev] = {s: dict(point=auc(sims, s), boots=np.array([auc(b, s) for b in boots])) for s in stats}
    return out, boots


def spread(rows, ref, stats):
    """Between-sim spread of each clean statistic vs. the median shift each global error produces (in clean MADs)."""
    res = {}
    for s in stats:
        v = np.array([r[s] for r in ref])
        mad = np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-12
        res[s] = {"clean_cv": float(v.std() / (abs(v.mean()) + 1e-12))}
        for kind in GLOBAL_TYPES:
            for sev in GLOBAL_SEVERITY[kind]:
                shift = [r[s] - c[s] for r in rows if r["type"] == kind and r["sev"] == sev
                         for c in rows if c["type"] == "clean" and c["sim"] == r["sim"]]
                res[s][f"{kind}{sev}_shift_in_mads"] = float(np.median(shift) / mad)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gnn", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--shape", default=str(RUNS / "shape" / "model.pt"))
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--stop", type=int, default=50)
    p.add_argument("--ref-start", type=int, default=50)
    p.add_argument("--ref-stop", type=int, default=70)
    p.add_argument("--device")
    p.add_argument("--out", type=Path, default=RESULTS / "runlevel")
    a = p.parse_args()
    dev = device(a.device)
    gnn = load_model(a.gnn, dev) if a.gnn != "none" else None
    shape_model = load_model(a.shape, dev) if a.shape != "none" else None
    ref = score_runs(list(range(a.ref_start, a.ref_stop)), "valid", gnn, shape_model, dev, corrupt=False)
    rows = score_runs(list(range(a.start, a.stop)), a.split, gnn, shape_model, dev)
    stats = [s for s in SURROGATE_STATS + RULE_STATS if s in rows[0]]
    zscore(rows, ref, stats)
    a.out.mkdir(parents=True, exist_ok=True)
    pickle.dump(dict(meta=dict(split=a.split, sims=[a.start, a.stop], ref=[a.ref_start, a.ref_stop], gnn=a.gnn,
                               shape=a.shape), ref=ref, rows=rows), open(a.out / f"runs_{a.split}.pkl", "wb"))
    table, _ = auroc_table(rows, stats)
    lines = ["| type | sev | " + " | ".join(stats) + " |", "|---|---|" + "---|" * len(stats)]
    for (kind, sev), v in table.items():
        lines.append(f"| {kind} | {GLOBAL_SEVERITY[kind][sev]} | " + " | ".join(
            f"{v[s]['point']:.2f} [{np.percentile(v[s]['boots'], 2.5):.2f}–{np.percentile(v[s]['boots'], 97.5):.2f}]"
            for s in stats) + " |")
    (a.out / f"auroc_{a.split}.md").write_text("\n".join(lines) + "\n")
    (a.out / f"spread_{a.split}.json").write_text(json.dumps(spread(rows, ref, stats), indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
