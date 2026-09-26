"""Detection lead time / latency with causal calibration (what an online simulation monitor would see).

- blowup: grows 1.3x/frame from the simulation's clean-acceleration floor to a 100x-step-displacement cap, first reached
  at frame T (read from the data).
  Lead time = T - first alarm at or after onset (positive = warned before the cap). Censored if no alarm by T.
- frozen (severity 3, 30 frames): latency = first alarm after onset - onset. Censored if no alarm in the event.
Thresholds (Phase 2, `--method conformal`, default): conformal run-level alarms. Each detector's conformity score
is its max over a CLEAN run after a fixed burn-in (frame >= BURN_IN); the threshold is the conformal
ceil((n+1)(1-alpha))-th smallest of n clean validation runs (sims 70-99), so on exchangeable new runs
P(a clean run raises any alarm) <= alpha. Lead times are then compared at equal, guaranteed false-alarm rates.
`--method frame95` reproduces Phase 1 (95th percentile of pre-onset clean frames).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from mgn.evaluate import RESULTS, score_copy, seed
from mgn.inject import inject
from mgn.train import RUNS, device, load_model, load_split

BURN_IN = 40  # = start of the steady regime and the earliest anomaly onset; fixed, not tuned
RUN_FRAMES = np.arange(BURN_IN, 340, 2)
DETS = ["gnn_causal", "constvel_causal", "laplacian_causal", "vlap_causal", "velocity_causal", "jacobian", "contact"]  # causally calibrated or memoryless only


def run(a):
    dev = device(a.device)
    model = load_model(a.ckpt, dev)
    sims = list(range(a.start, a.stop))
    events = []
    for i, tr in zip(sims, tqdm(load_split(a.split, a.stop)[a.start:], desc=f"{a.split} sims")):
        for kind, sev in (("blowup", 1), ("frozen", 3)):
            rng = np.random.default_rng(seed(i, "frozen", sev) + (7 if kind == "blowup" else 0))
            wp, _, fmask = inject(tr, kind, sev, rng)
            t0 = int(np.flatnonzero(fmask)[0])
            dev_ = np.abs(wp - tr["world_pos"]).max(axis=(1, 2))
            end = int(np.flatnonzero(dev_ >= 0.999 * dev_.max())[0]) if kind == "blowup" else int(np.flatnonzero(fmask)[-1])
            frames = np.arange(max(t0 - 15, 2), end + 1)
            fs, _ = score_copy(model, tr, wp, frames, dev)
            events.append(dict(sim=i, kind=kind, t0=t0, end=end, ramp=end - t0, frames=frames.tolist(),
                               scores={d: fs[d].tolist() for d in DETS}))
    return events


def clean_run_max(model, split, sims, dev):
    """Per clean run: max frame score after the burn-in, per causal detector."""
    out = []
    for i, tr in zip(sims, tqdm(load_split(split, max(sims) + 1)[min(sims):], desc=f"clean {split} runs")):
        fs, _ = score_copy(model, tr, tr["world_pos"], RUN_FRAMES, dev)
        out.append({d: float(np.max(fs[d])) for d in DETS})
    return out


def conformal_threshold(scores, alpha):
    """ceil((n+1)(1-alpha))-th smallest of n calibration scores (inf if n is too small for this alpha)."""
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else float("inf")


def analyse(events, thr):
    out = {}
    for kind in ("blowup", "frozen"):
        E = [e for e in events if e["kind"] == kind]
        for d in DETS:
            vals, false_alarms, clean = [], 0, 0
            for e in E:
                f, s = np.array(e["frames"]), np.array(e["scores"][d])
                pre = f < e["t0"] - 3
                false_alarms += (s[pre] > thr[d]).sum()
                clean += pre.sum()
                hit = np.flatnonzero((f >= e["t0"]) & (s > thr[d]))
                if not len(hit):
                    vals.append(np.nan)  # censored: never alarmed in time
                elif kind == "blowup":
                    vals.append(e["end"] - f[hit[0]])
                else:
                    vals.append(f[hit[0]] - e["t0"])
            vals = np.array(vals, float)
            det = vals[~np.isnan(vals)]
            rng = np.random.default_rng(0)
            boots = [np.median(b) for b in (rng.choice(det, len(det)) for _ in range(1000))] if len(det) else [np.nan]
            out[f"{kind}/{d}"] = dict(detected=float((~np.isnan(vals)).mean()), median=float(np.median(det)) if len(det) else None,
                                      ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                                      false_alarm_rate=float(false_alarms / max(clean, 1)), n=len(vals))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--stop", type=int, default=100)
    p.add_argument("--device")
    p.add_argument("--thresholds", type=Path, help="thresholds.json from a --split valid run (required for test)")
    p.add_argument("--out", type=Path, default=RESULTS / "leadtime")
    p.add_argument("--method", default="conformal", choices=["conformal", "frame95"])
    p.add_argument("--alpha", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--calib-start", type=int, default=70)
    p.add_argument("--calib-stop", type=int, default=100)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.method == "conformal":
        dev = device(a.device)
        model = load_model(a.ckpt, dev)
        if a.thresholds:
            thrs = json.loads(a.thresholds.read_text())
        else:
            calib = clean_run_max(model, "valid", list(range(a.calib_start, a.calib_stop)), dev)
            thrs = {str(al): {d: conformal_threshold([c[d] for c in calib], al) for d in DETS} for al in a.alpha}
            (a.out / "conformal_thresholds.json").write_text(json.dumps(thrs, indent=1))
        clean = clean_run_max(model, a.split, list(range(a.start, a.stop)), dev)
        events = run(a)
        report = {}
        for al, thr in thrs.items():
            fpr = {d: float(np.mean([c[d] > thr[d] for c in clean])) for d in DETS}
            report[al] = dict(run_fpr=fpr, leadtime=analyse(events, thr))
            print(f"alpha={al}: run-level FPR on {a.split} " + ", ".join(f"{d} {v:.0%}" for d, v in fpr.items()))
            for k, v in report[al]["leadtime"].items():
                print(f"   {k:28s} detected {v['detected']:.0%}  median {v['median']} frames {v['ci']}")
        (a.out / f"conformal_{a.split}.json").write_text(json.dumps(dict(ckpt=a.ckpt, sims=[a.start, a.stop],
                                                                         thresholds=thrs, results=report), indent=1))
        return
    events = run(a)
    if a.split == "valid":
        clean = {d: np.concatenate([np.array(e["scores"][d])[np.array(e["frames"]) < e["t0"] - 3] for e in events])
                 for d in DETS}
        thr = {d: float(np.percentile(v, 95)) for d, v in clean.items()}
        (a.out / "thresholds.json").write_text(json.dumps(thr, indent=1))
    else:
        thr = json.loads(a.thresholds.read_text())
    res = analyse(events, thr)
    (a.out / f"leadtime_{a.split}.json").write_text(json.dumps(dict(ckpt=a.ckpt, sims=[a.start, a.stop], thresholds=thr,
                                                                     results=res), indent=1))
    for k, v in res.items():
        print(f"{k:28s} detected {v['detected']:.0%}  median {v['median']} frames {v['ci']}  false alarms {v['false_alarm_rate']:.1%}")


if __name__ == "__main__":
    main()
