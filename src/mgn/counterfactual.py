"""Counterfactual explanation test: does the VLM tell a clean, highly deformed frame (A) apart from the same frame
with an injected hourglass pattern (B), or does it produce the same engineering story for both?

Three input arms: render only (the VLM's own vision), render + GNN residual heatmap, and + numeric diagnostics.
Only the render-only arm isolates the VLM's vision: the heatmap and diagnostics differ between A and B by
construction. Metrics per arm: P(none | A), P(hourglass | B), pair discrimination (A called none AND B called
anomalous), and the Jaccard overlap of engineering terms in the two explanations, against a shuffled-pair baseline.
"""
import argparse
import json
import re

import numpy as np
from PIL import Image

from mgn.data import NORMAL, T_MAX
from mgn.detect import gnn_scores
from mgn.evaluate import RESULTS, seed
from mgn.explain import CELLS, LABELS, PROMPT, clean_clim, diagnostics, make_asker
from mgn.inject import inject
from mgn.render import Renderer
from mgn.train import RUNS, device, load_model, load_split

_PANELS = ("LEFT panel: the deformed mesh. RIGHT panel: same camera, coloured by the disagreement between the observed "
           "motion and a\nlearned physics surrogate (red = the frame is physically inconsistent there, pale yellow = "
           "consistent).")
assert _PANELS in PROMPT
RENDER_PROMPT = PROMPT.replace(_PANELS, "The image shows the deformed mesh.").replace("of the LEFT panel", "of the image")
TERMS = ["hourglass", "checkerboard", "zig-zag", "alternating", "penetrat", "contact", "invert", "inside out",
         "instabil", "oscillat", "blow-up", "frozen", "stuck", "bend", "stress", "concentrat", "deform", "wrinkl",
         "buckl", "dent", "smooth", "consistent"]


def terms(text):
    t = (text or "").lower()
    return {w for w in TERMS if w in t}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    p.add_argument("--ckpt", default=str(RUNS / "mgn" / "model.pt"))
    p.add_argument("--split", default="test", choices=["valid", "test"])
    p.add_argument("--start", type=int, default=8)
    p.add_argument("--stop", type=int, default=48)  # 40 pairs
    a = p.parse_args()

    dev = device()
    gnn = load_model(a.ckpt, dev)
    ask = make_asker(a.model)
    out_dir = RESULTS / "counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    for i, tr in zip(range(a.start, a.stop), load_split(a.split, a.stop)[a.start:]):
        t = T_MAX - 20  # late in the loading: the plate is strongly bent, the hardest "clean" case
        wp_b, mask, _ = inject(tr, "hourglass", 3, np.random.default_rng(seed(i, "hourglass", 3)), t0=t - 2)
        rend, normal = Renderer(tr), tr["node_type"] == NORMAL
        clim = clean_clim(gnn, tr, dev)
        shift = rend.project(wp_b[t][mask]) - rend.project(tr["world_pos"][t][mask])
        pair = dict(sim=i, t=t, px=float(np.linalg.norm(shift, axis=1).max()))  # B must be visibly different
        for tag, wp in (("A", tr["world_pos"]), ("B", wp_b)):
            res = gnn_scores(gnn, tr, wp, [t], dev)[0]
            plain = rend.render(wp[t])
            heat = rend.render(wp[t], scalars=np.where(normal, res, 0), clim=clim)
            Image.fromarray(plain).save(out_dir / f"{i:03d}{tag}_render.png")
            Image.fromarray(np.hstack([plain, heat])).save(out_dir / f"{i:03d}{tag}_panels.png")
            _, stats = diagnostics(tr, wp, t, res, rend, float("nan"), float("nan"))
            stats = re.sub(r"residual nan.*?elements\); ", "", stats)  # no frame-level calibration for single frames
            fmt = dict(labels=", ".join(LABELS), cells=", ".join(CELLS))
            arms = {"render": (RENDER_PROMPT.format(stats="", **fmt), f"{i:03d}{tag}_render.png"),
                    "heatmap": (PROMPT.format(stats="", **fmt), f"{i:03d}{tag}_panels.png"),
                    "full": (PROMPT.format(stats=stats, **fmt), f"{i:03d}{tag}_panels.png")}
            for arm, (prompt, img) in arms.items():
                d, raw = ask(prompt, str(out_dir / img))
                pair[f"{tag}_{arm}"] = dict(pred=(d or {}).get("anomaly_type", "invalid"),
                                            explanation=(d or {}).get("explanation", raw))
        pairs.append(pair)
        print(i, {k: v["pred"] for k, v in pair.items() if isinstance(v, dict)}, flush=True)

    lines = [f"n = {len(pairs)} pairs (A = clean frame {T_MAX - 20}, B = same frame + hourglass severity 3; median visible shift "
             f"{np.median([q['px'] for q in pairs]):.0f} px)\n",
             "| arm | P(none \\| A) | P(hourglass \\| B) | pair discrimination | term Jaccard A↔B | shuffled-pair Jaccard | permutation p |",
             "|---|---|---|---|---|---|---|"]
    rng = np.random.default_rng(0)
    for arm in ("render", "heatmap", "full"):
        pa = [q[f"A_{arm}"] for q in pairs]
        pb = [q[f"B_{arm}"] for q in pairs]
        tA, tB = [terms(x["explanation"]) for x in pa], [terms(x["explanation"]) for x in pb]
        true_j = np.mean([jaccard(x, y) for x, y in zip(tA, tB)])
        shuffled = [np.mean([jaccard(x, tB[j]) for x, j in zip(tA, rng.permutation(len(tB)))]) for _ in range(1000)]
        pval = (np.sum(np.array(shuffled) >= true_j) + 1) / 1001  # high overlap vs. unrelated pairs => same story
        lines.append(f"| {arm} | {np.mean([x['pred'] == 'none' for x in pa]):.0%} | "
                     f"{np.mean([x['pred'] == 'hourglass' for x in pb]):.0%} | "
                     f"{np.mean([x['pred'] == 'none' and y['pred'] not in ('none', 'invalid') for x, y in zip(pa, pb)]):.0%} | "
                     f"{true_j:.2f} | {np.mean(shuffled):.2f} | {pval:.3f} |")
    (RESULTS / "counterfactual.json").write_text(json.dumps(dict(model=a.model, ckpt=a.ckpt, pairs=pairs), indent=1))
    (RESULTS / "counterfactual.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
