"""Post-hoc, validation only (sims 20-49): velocity-Laplacian rule vs GNN on the exact Phase-1 validation copies.
Run at the phase1-frozen code (git checkout phase1-frozen -- src) so the copies match results/valid/records.pkl."""
import pickle, numpy as np
from sklearn.metrics import roc_auc_score
from mgn.data import NORMAL, load_traj, DATA
from mgn.inject import inject, TYPES, SEVERITY, adjacency
from mgn.evaluate import choose_frames, seed, CALIB, frame_score

V = pickle.load(open("results/valid/records.pkl", "rb"))
recs = {(r["traj"], r["type"], r["sev"], r["t"]): r for r in V["records"]}
out = []
for i in V["meta"]["sims"]:
    tr = load_traj(DATA / "valid" / f"traj_{i:04d}.npz"); normal = tr["node_type"] == NORMAL
    adj = adjacency(tr).astype(np.float32); deg = np.asarray(adj.sum(1)).clip(1)
    for k in TYPES:
        for s in SEVERITY:
            rng = np.random.default_rng(seed(i, k, s)); wp, mask, fm = inject(tr, k, s, rng)
            frames, labels, phase = choose_frames(fm, rng)
            fr = np.r_[frames, CALIB]
            vel = wp[fr] - wp[fr - 1]
            raw = np.stack([np.linalg.norm(v - (adj @ v) / deg, axis=-1) for v in vel])
            cal = np.median(raw[len(frames):], axis=0)
            fs = frame_score(raw[:len(frames)] / (cal + np.median(cal[normal]) + 1e-12), normal)
            for j, t in enumerate(frames):
                r = recs[(i, k, s, int(t))]
                assert r["label"] == labels[j] and r["phase"] == phase[j]
                out.append(dict(type=k, sev=s, phase=phase[j], label=labels[j], vlap=fs[j], **{d: r[f"frame_{d}"] for d in ("gnn", "constvel", "laplacian", "velocity", "jacobian")}))
dets = ["gnn", "vlap", "velocity", "constvel", "laplacian", "jacobian"]
print("type        phase     " + " ".join(f"{d:>9s}" for d in dets))
for k in TYPES:
    for ph in ("onset", "sustained"):
        rs = [r for r in out if r["type"] == k and r["phase"] in (ph, "clean")]
        vals = [np.mean([roc_auc_score([r["label"] for r in rs if r["sev"] == s], [r[d] for r in rs if r["sev"] == s]) for s in SEVERITY]) for d in dets]
        print(f"{k:11s} {ph:9s} " + " ".join(f"{v:9.2f}" for v in vals))
fz = [r for r in out if r["type"] == "frozen" and r["phase"] in ("sustained", "clean")]
for s in SEVERITY:
    rs = [r for r in fz if r["sev"] == s]
    print(f"frozen sev{s} sustained: gnn {roc_auc_score([r['label'] for r in rs],[r['gnn'] for r in rs]):.3f}  vlap {roc_auc_score([r['label'] for r in rs],[r['vlap'] for r in rs]):.3f}")
