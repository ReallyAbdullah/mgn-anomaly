"""VLM explanations of flagged frames, scored for faithfulness against the injected ground truth.

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


def diagnostics(traj, wp, t, res, rend, frame_ratio):
    """Deterministic, engineer-style diagnostics for frame t."""
    nt, normal = traj["node_type"], traj["node_type"] == NORMAL
    rest_sign = np.sign(signed_volumes(traj["mesh_pos"], traj["cells"]))
    inverted = int((np.sign(signed_volumes(wp[t], traj["cells"])) != rest_sign).sum())
    r = np.where(normal, res, 0)
    hot = np.argsort(r)[-10:]
    flagged = int((r > 5 * np.median(r[normal])).sum())
    gap = np.linalg.norm(wp[t][hot][:, None] - wp[t][nt == OBSTACLE][None], axis=-1).min() * 1000
    speed = np.linalg.norm(wp[t] - wp[t - 1], axis=1)
    speed_ratio = speed[hot].mean() / (np.median(speed[normal]) + 1e-12)
    return (f"Diagnostics: residual {frame_ratio:.1f}x the simulation's typical level; hotspot in the "
            f"{grid_cell(rend.project(wp[t][hot]))} cell; {flagged} nodes above 5x median residual; "
            f"{inverted} inverted elements; hotspot-to-actuator distance {gap:.1f} mm; "
            f"hotspot speed {speed_ratio:.2f}x the median plate node speed.")


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0))
        assert d["anomaly_type"] in LABELS
        return d
    except Exception:
        return None


def select(records, n_per_type, n_clean, rng):
    """Anomalous frames the GNN detector flags (score above the 95th percentile of clean frames), plus clean frames."""
    thr = np.percentile([r["frame_gnn"] for r in records if not r["label"]], 95)
    anom = [r for r in records if r["label"] and r["frame_gnn"] > thr]
    picks = []
    for kind in TYPES:
        pool = [r for r in anom if r["type"] == kind]
        picks += [pool[i] for i in rng.choice(len(pool), min(n_per_type, len(pool)), replace=False)]
    clean = [r for r in records if not r["label"]]
    picks += [clean[i] for i in rng.choice(len(clean), n_clean, replace=False)]
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--n-per-type", type=int, default=10)
    p.add_argument("--n-clean", type=int, default=10)
    a = p.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    with open(RESULTS / "records.pkl", "rb") as f:
        records = pickle.load(f)
    picks = select(records, a.n_per_type, a.n_clean, np.random.default_rng(0))
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
        stats = diagnostics(tr, wp, t, res, rend, r["frame_gnn"])
        truth = r["type"] if r["label"] else "none"
        cell = grid_cell(rend.project(wp[t][mask])) if r["label"] else None
        row = dict(image=path.name, truth=truth, sev=r["sev"], cell=cell, stats=stats)
        for cond, s in (("visual", ""), ("full", stats)):
            d, raw = ask(PROMPT.format(stats=s, labels=", ".join(LABELS), cells=", ".join(CELLS)), str(path))
            row[cond] = d
            row[f"{cond}_raw"] = raw
        results.append(row)
        print(k, truth, "| visual:", (row["visual"] or {}).get("anomaly_type"), "| full:", (row["full"] or {}).get("anomaly_type"), flush=True)
    (RESULTS / "vlm_results.json").write_text(json.dumps(results, indent=1))
    score(results)


def adjacent(a, b):
    (ra, ca), (rb, cb) = divmod(CELLS.index(a), 3), divmod(CELLS.index(b), 3)
    return abs(ra - rb) <= 1 and abs(ca - cb) <= 1


def score(results):
    lines = ["| condition | valid JSON | type accuracy | macro-F1 | location hit (±1 cell) |", "|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, cond in zip(axes, ("visual", "full")):
        y = [r["truth"] for r in results]
        pred = [(r[cond] or {}).get("anomaly_type", "invalid") for r in results]
        valid = np.mean([r[cond] is not None for r in results])
        acc = np.mean([a == b for a, b in zip(y, pred)])
        f1 = f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)
        loc = [adjacent(r["cell"], r[cond]["location"]) for r in results
               if r["cell"] and r[cond] and r[cond].get("location") in CELLS]
        lines.append(f"| {cond} | {valid:.0%} | {acc:.0%} | {f1:.2f} | {np.mean(loc) if loc else float('nan'):.0%} |")
        cm = confusion_matrix(y, pred, labels=LABELS + ["invalid"])[:len(LABELS)]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS) + 1), LABELS + ["invalid"], rotation=45, ha="right")
        ax.set_yticks(range(len(LABELS)), LABELS)
        for (i, j), v in np.ndenumerate(cm):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=8)
        ax.set_title(f"VLM — {cond}")
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("injected")
    fig.tight_layout()
    fig.savefig(RESULTS / "vlm_confusion.png", dpi=150)
    (RESULTS / "vlm_metrics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
