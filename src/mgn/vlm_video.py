"""Gate E video pilot (pre-registered in docs/gate_e_video.md): single render vs 8-frame clip, local Qwen3-VL-8B."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from mgn.counterfactual import RENDER_PROMPT
from mgn.evaluate import RESULTS, seed
from mgn.explain import CELLS, LABELS, parse, select
from mgn.inject import inject
from mgn.render import Renderer
from mgn.train import load_split
from mgn.vlm_pilot import select_frames

CLIP_LEN = 8
_ONE = "The image shows the deformed mesh."
CLIP_SENTENCE = (f"The {CLIP_LEN} images are consecutive frames of the deformed mesh, oldest first. Judge the LAST "
                 "frame, using the earlier frames as temporal context.")


def prompts():
    fmt = dict(stats="", labels=", ".join(LABELS), cells=", ".join(CELLS))
    single = RENDER_PROMPT.format(**fmt)
    assert _ONE in RENDER_PROMPT and "of the image" in RENDER_PROMPT
    clip = RENDER_PROMPT.replace(_ONE, CLIP_SENTENCE).replace("of the image", "of the last image").format(**fmt)
    return single, clip


def make_multi_asker(model_id):
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    vlm, processor = load(model_id)
    config = load_config(model_id)

    def ask(prompt, images):
        text = ""
        for extra in ("", "\nReturn ONLY valid JSON."):
            chat = apply_chat_template(processor, config, prompt + extra, num_images=len(images))
            text = generate(vlm, processor, chat, [str(p) for p in images], max_tokens=300, temperature=0.0,
                            verbose=False).text
            if (d := parse(text)) is not None:
                return d, text
        return None, text
    return ask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-VL-8B-Instruct-4bit")
    ap.add_argument("--out", type=Path, default=RESULTS / "vlm_video")
    a = ap.parse_args()
    img_dir = a.out / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)

    # recover (sim, type, sev, t) of the Phase 1 VLM frames (the selection reproduces all 200 image names)
    frames = json.loads((RESULTS / "vlm_results.json").read_text())["frames"]
    picks = select(pickle.load(open(RESULTS / "test" / "records.pkl", "rb"))["records"], 35, 25, np.random.default_rng(0))
    assert all(f"{k:03d}_{r['type']}_s{r['sev']}_t{r['t']}" in f["image"] for k, (r, f) in enumerate(zip(picks, frames)))
    idx = select_frames(frames, np.random.default_rng(0))
    pairs = json.loads((RESULTS / "counterfactual.json").read_text())["pairs"][:10]
    test = load_split("test", 100)

    items = []  # (key, truth, [clip image paths]) ; the single arm uses the last image
    for i in idx:
        r = picks[i]
        tr = test[r["traj"]]
        wp, _, _ = inject(tr, r["type"], r["sev"], np.random.default_rng(seed(r["traj"], r["type"], r["sev"])))
        if not r["label"]:  # clean frames: render the clean trajectory, so the clip can't include the nearby event
            wp = tr["world_pos"]
        items.append((f"frame:{i}", frames[i]["truth"], wp, tr, r["t"]))
    for q in pairs:
        tr, t = test[q["sim"]], q["t"]
        wp_b, _, _ = inject(tr, "hourglass", 3, np.random.default_rng(seed(q["sim"], "hourglass", 3)), t0=t - 2)
        items += [(f"pair:{q['sim']}:A", "none", tr["world_pos"], tr, t), (f"pair:{q['sim']}:B", "hourglass", wp_b, tr, t)]
    rendered = []
    for key, truth, wp, tr, t in items:
        rend = Renderer(tr)
        paths = []
        for j, tt in enumerate(range(t - CLIP_LEN + 1, t + 1)):
            p = img_dir / f"{key.replace(':', '_')}_{j}.png"
            if not p.exists():
                Image.fromarray(rend.render(wp[tt])).save(p)
            paths.append(p)
        rendered.append((key, truth, paths))

    single_prompt, clip_prompt = prompts()
    ask = make_multi_asker(a.model)
    out_path = a.out / "responses.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    for key, truth, paths in rendered:
        for arm, prompt, imgs in (("single", single_prompt, paths[-1:]), ("clip", clip_prompt, paths)):
            k = f"{key}|{arm}"
            if k in done:
                continue
            d, raw = ask(prompt, imgs)
            done[k] = dict(truth=truth, pred=(d or {}).get("anomaly_type", "invalid"), valid=d is not None,
                           explanation=(d or {}).get("explanation"), raw=raw)
            out_path.write_text(json.dumps(done, indent=1))
            print(k, truth, "->", done[k]["pred"], flush=True)
    report(done, a.out)


def report(done, out):
    def get(arm, prefix):
        return {k.split("|")[0]: v for k, v in done.items() if k.endswith(f"|{arm}") and k.startswith(prefix)}
    lines = [f"Gate E video pilot — local Qwen3-VL-8B, single render vs {CLIP_LEN}-frame clip (plain renders)\n",
             "| metric | single | clip |", "|---|---|---|"]
    res = {}
    for arm in ("single", "clip"):
        fr = get(arm, "frame:")
        anom = [v for v in fr.values() if v["truth"] != "none"]
        clean = [v for v in fr.values() if v["truth"] == "none"]
        pa = get(arm, "pair:")
        sims = sorted({k.split(":")[1] for k in pa})
        res[arm] = dict(acc=sum(v["pred"] == v["truth"] for v in anom), n=len(anom),
                        none=sum(v["pred"] == "none" for v in anom), clean_ok=sum(v["pred"] == "none" for v in clean),
                        n_clean=len(clean), sep=sum(pa[f"pair:{s}:A"]["pred"] == "none" and
                                                    pa[f"pair:{s}:B"]["pred"] not in ("none", "invalid") for s in sims),
                        n_pairs=len(sims), invalid=sum(not v["valid"] for v in list(fr.values()) + list(pa.values())))
    s, c = res["single"], res["clip"]
    lines += [f"| type accuracy, {s['n']} sev-3 anomalous frames | {s['acc']}/{s['n']} | {c['acc']}/{c['n']} |",
              f"| answered 'none' on anomalous frames | {s['none']}/{s['n']} | {c['none']}/{c['n']} |",
              f"| clean frames called 'none' | {s['clean_ok']}/{s['n_clean']} | {c['clean_ok']}/{c['n_clean']} |",
              f"| counterfactual pairs separated | {s['sep']}/{s['n_pairs']} | {c['sep']}/{c['n_pairs']} |",
              f"| invalid JSON | {s['invalid']} | {c['invalid']} |"]
    helps = (c["acc"] - s["acc"] >= 5) or (c["sep"] - s["sep"] >= 3)
    lines.append(f"\n**Pre-registered decision: {'temporal context HELPS' if helps else 'no evidence that temporal context helps this model'}**")
    (out / "video_pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
