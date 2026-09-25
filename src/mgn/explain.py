"""VLM explanations of simulation frames, scored for correctness (type, exact location) against the injected ground truth.

Deterministic first, LLM last: the VLM sees the render, the GNN residual heatmap and (in the `full` condition)
diagnostics computed from the mesh. Ablation: `visual` (images only) vs `full` (images + diagnostics).
"""
import argparse
import json
import pickle
import re

import matplotlib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

from mgn.data import NORMAL, OBSTACLE
from mgn.detect import gnn_scores
from mgn.evaluate import RESULTS, seed
from mgn.inject import TYPES, inject, signed_volumes
from mgn.render import SIZE, Renderer, grid_cell
from mgn.train import RUNS, device, load_model, load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = TYPES + ["none"]
CELLS = [f"{r}-{c}" for r in ("top", "middle", "bottom") for c in ("left", "center", "right")]

PROMPT = """You are a simulation QA engineer reviewing one frame of a finite-element simulation.
Setup: a hyperelastic plate is clamped along one edge; a blue actuator pushes it from below, so the plate bends.
LEFT panel: the deformed mesh. RIGHT panel: same camera, coloured by the disagreement between the observed motion and a
learned physics surrogate (red = the frame is physically inconsistent there, pale yellow = consistent).
{stats}
Classify the frame as exactly one of:
- hourglass: a patch of dozens of nodes with a checkerboard / zig-zag pattern of alternating offsets, away from the actuator
- penetration: a handful of plate nodes at the actuator contact pushed into the actuator (contact violated)
- inversion: a single element turned inside out: one node pushed through its element, very few nodes involved
- instability: a few nodes moving violently and erratically, far faster than the rest of the plate (numerical blow-up)
- frozen: a region that stops moving (much slower than the plate) while its surroundings keep deforming
- none: no anomaly; the frame is physically consistent
Answer with ONLY a JSON object:
{{"anomaly_type": "<one of: {labels}>", "location": "<3x3 grid cell of the LEFT panel: {cells}>",
"confidence": <0-1>, "explanation": "<one or two sentences citing the visual evidence>"}}"""


def diagnostics(traj, wp, t, res, rend, frame_ratio, clean_ref):
    """Deterministic, engineer-style diagnostics for frame t: (numeric dict, prompt text)."""
    nt, normal = traj["node_type"], traj["node_type"] == NORMAL
    rest_sign = np.sign(signed_volumes(traj["mesh_pos"], traj["cells"]))
    r = np.where(normal, res, 0)
    hot = np.argsort(r)[-10:]
    speed = np.linalg.norm(wp[t] - wp[t - 1], axis=1)
    d = dict(residual_ratio=frame_ratio,
             flagged_nodes=int((r > 5 * np.median(r[normal])).sum()),
             inverted=int((np.sign(signed_volumes(wp[t], traj["cells"])) != rest_sign).sum()),
             gap_mm=np.linalg.norm(wp[t][hot][:, None] - wp[t][nt == OBSTACLE][None], axis=-1).min() * 1000,
             speed_ratio=speed[hot].mean() / (np.median(speed[normal]) + 1e-12),
             hotspot=grid_cell(rend.project(wp[t][hot])))
    text = (f"Diagnostics: residual {d['residual_ratio']:.1f}x the simulation's typical level (95% of clean frames stay "
            f"below {clean_ref:.1f}x, with 0 inverted elements); {d['flagged_nodes']} nodes above 5x median residual; {d['inverted']} inverted "
            f"elements; hotspot-to-actuator distance {d['gap_mm']:.1f} mm; hotspot speed {d['speed_ratio']:.2f}x the "
            f"median plate node speed.")
    return d, text


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0))
        assert d["anomaly_type"] in LABELS
        return d
    except Exception:
        return None


def select(records, n_per_type, n_clean, rng):
    """Stratified random frames, NOT filtered on detection (filtering would bias accuracy upward);
    whether the GNN flagged each frame is recorded so both subsets can be reported."""
    picks = []
    for kind in TYPES:
        pool = [r for r in records if r["label"] and r["type"] == kind]
        picks += [pool[i] for i in rng.choice(len(pool), min(n_per_type, len(pool)), replace=False)]
    clean = [r for r in records if not r["label"]]
    picks += [clean[i] for i in rng.choice(len(clean), n_clean, replace=False)]
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--n-per-type", type=int, default=35)
    p.add_argument("--n-clean", type=int, default=25)
    a = p.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    with open(RESULTS / "records.pkl", "rb") as f:
        records = pickle.load(f)["records"]
    picks = select(records, a.n_per_type, a.n_clean, np.random.default_rng(0))
    clean_ref = np.percentile([r["frame_gnn"] for r in records if not r["label"]], 95)
    dev = device()
    gnn = load_model(a.ckpt, dev)
    test = load_split("test", max(r["traj"] for r in picks) + 1)
    vlm, processor = load(a.model)
    config = load_config(a.model)
    out_dir = RESULTS / "vlm"
    out_dir.mkdir(parents=True, exist_ok=True)

    def ask(prompt, image):
        chat = apply_chat_template(processor, config, prompt, num_images=1)
        for attempt in range(2):
            text = generate(vlm, processor, chat, [image], max_tokens=300, temperature=0.0, verbose=False).text
            if (d := parse(text)) is not None:
                return d, text
            chat = apply_chat_template(processor, config, prompt + "\nReturn ONLY valid JSON.", num_images=1)
        return None, text

    results = []
    for k, r in enumerate(picks):
        tr = test[r["traj"]]
        rng = np.random.default_rng(seed(r["traj"], r["type"], r["sev"]))
        wp, mask, _ = inject(tr, r["type"], r["sev"], rng)
        if not r["label"]:
            mask = np.zeros_like(mask)
        t = r["t"]
        res = gnn_scores(gnn, tr, wp, [t], dev)[0]
        rend = Renderer(tr)
        normal = tr["node_type"] == NORMAL
        heat = rend.render(wp[t], scalars=np.where(normal, res, 0), clim=(0, np.percentile(res[normal], 99.5)))
        path = out_dir / f"{k:03d}_{r['type']}_s{r['sev']}_t{t}_{'anom' if r['label'] else 'clean'}.png"
        Image.fromarray(np.hstack([rend.render(wp[t]), heat])).save(path)
        diag, stats = diagnostics(tr, wp, t, res, rend, r["frame_gnn"], clean_ref)
        truth = r["type"] if r["label"] else "none"
        cell = grid_cell(rend.project(wp[t][mask])) if r["label"] else None
        row = dict(image=path.name, truth=truth, sev=r["sev"], cell=cell, diag=diag, flagged=bool(r["frame_gnn"] > clean_ref))
        for cond, s in (("visual", ""), ("full", stats)):
            prompt = PROMPT.format(stats=s, labels=", ".join(LABELS), cells=", ".join(CELLS))
            d, raw = ask(prompt, str(path))
            row[cond], row[f"{cond}_raw"], row[f"{cond}_prompt"] = d, raw, prompt
        results.append(row)
        print(k, truth, "| visual:", (row["visual"] or {}).get("anomaly_type"), "| full:", (row["full"] or {}).get("anomaly_type"), flush=True)
    import mlx_vlm
    meta = dict(model=a.model, mlx_vlm=mlx_vlm.__version__, temperature=0.0, selection_seed=0, ckpt=a.ckpt)
    (RESULTS / "vlm_results.json").write_text(json.dumps(dict(meta=meta, frames=results), indent=1, default=float))
    score(results)


def diagnostics_baseline(results):
    """Does the VLM add anything beyond the numbers? A depth-4 decision tree on the diagnostics, 5-fold CV."""
    from sklearn.model_selection import cross_val_predict
    from sklearn.tree import DecisionTreeClassifier
    keys = ["residual_ratio", "flagged_nodes", "inverted", "gap_mm", "speed_ratio"]
    X = np.array([[r["diag"][k] for k in keys] for r in results])
    y = [r["truth"] for r in results]
    pred = cross_val_predict(DecisionTreeClassifier(max_depth=4, random_state=0), X, y, cv=5)
    return [dict(anomaly_type=p, location=r["diag"]["hotspot"]) for p, r in zip(pred, results)]


def score(results):
    y = [r["truth"] for r in results]
    cells = [r["cell"] for r in results if r["cell"]]
    majority_cell = max(set(cells), key=cells.count)
    arms = {"visual (VLM, images)": [r["visual"] for r in results],
            "full (VLM, images + diagnostics)": [r["full"] for r in results],
            "diagnostics-only decision tree (5-fold CV)": diagnostics_baseline(results),
            "majority class / majority cell": [dict(anomaly_type=max(set(y), key=y.count), location=majority_cell)] * len(y)}
    lines = [f"n = {len(y)} frames; uniform-random type accuracy = {1 / len(LABELS):.0%}\n",
             "| arm | valid JSON | type accuracy | macro-F1 | exact location cell | type acc. on GNN-flagged frames |",
             "|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, (name, preds) in enumerate(arms.items()):
        pred = [(p or {}).get("anomaly_type", "invalid") for p in preds]
        valid = np.mean([p is not None for p in preds])
        acc = np.mean([a == b for a, b in zip(y, pred)])
        f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
        loc = [p is not None and p.get("location") == r["cell"] for r, p in zip(results, preds) if r["cell"]]
        flag = [a == b for a, b, r in zip(y, pred, results) if r["flagged"]]
        lines.append(f"| {name} | {valid:.0%} | {acc:.0%} | {f1:.2f} | {np.mean(loc):.0%} | "
                     f"{np.mean(flag):.0%} (n={len(flag)}) |")
        if k < 3:
            ax = axes[k]
            cm = confusion_matrix(y, pred, labels=LABELS + ["invalid"])[:len(LABELS)]
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(LABELS) + 1), LABELS + ["invalid"], rotation=45, ha="right")
            ax.set_yticks(range(len(LABELS)), LABELS)
            for (i, j), v in np.ndenumerate(cm):
                if v:
                    ax.text(j, i, v, ha="center", va="center", fontsize=8)
            ax.set_title(name, fontsize=9)
            ax.set_xlabel("predicted")
    axes[0].set_ylabel("injected")
    fig.tight_layout()
    fig.savefig(RESULTS / "vlm_confusion.png", dpi=150)
    (RESULTS / "vlm_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
