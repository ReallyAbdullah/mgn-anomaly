"""Gate E pilot (pre-registered in docs/gate_e_pilot.md): a larger same-family VLM via OpenRouter, paired against the
local 8B on the same images and verbatim prompts. Budget-aware and resumable (free tier: 50 requests/day).

Needs OPENROUTER_API_KEY in the environment (kept in a gitignored .env; never committed).
"""
import argparse
import base64
import json
import os
import time
from pathlib import Path

import numpy as np
import requests

from mgn.counterfactual import RENDER_PROMPT
from mgn.evaluate import RESULTS
from mgn.explain import CELLS, LABELS, parse
from mgn.inject import TYPES

URL = "https://openrouter.ai/api/v1/chat/completions"


def ask(model, prompt, image_path, state):
    """One chat call with the image inlined; one retry on invalid JSON. Every HTTP call counts against the budget."""
    img = base64.b64encode(Path(image_path).read_bytes()).decode()
    raw = None
    for extra in ("", "\nReturn ONLY valid JSON."):
        if state["calls"] >= state["budget"]:
            raise RuntimeError("daily budget reached")
        r = requests.post(URL, timeout=300, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                          json={"model": model, "temperature": 0, "reasoning": {"enabled": True},
                                "messages": [{"role": "user", "content": [
                                    {"type": "text", "text": prompt + extra},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}]})
        state["calls"] += 1
        if r.status_code == 429:
            raise RuntimeError(f"rate limited: {r.text[:200]}")
        r.raise_for_status()
        raw = (r.json()["choices"][0]["message"].get("content") or "")
        if (d := parse(raw)) is not None:
            return d, raw
        time.sleep(1)
    return None, raw


def select_frames(frames, rng):
    picks = []
    for kind in TYPES:
        pool = [i for i, f in enumerate(frames) if f["truth"] == kind and f["sev"] == 3]
        picks += [pool[j] for j in rng.choice(len(pool), min(4, len(pool)), replace=False)]
    clean = [i for i, f in enumerate(frames) if f["truth"] == "none"]
    return picks + [clean[j] for j in rng.choice(len(clean), 5, replace=False)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen/qwen3.8-27b:free")
    p.add_argument("--budget", type=int, default=48, help="max HTTP calls this run (free tier: 50/day)")
    p.add_argument("--out", type=Path, default=RESULTS / "vlm_pilot")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    done_path = a.out / "responses.json"
    done = json.loads(done_path.read_text()) if done_path.exists() else {}
    state = {"calls": 0, "budget": a.budget}

    frames = json.loads((RESULTS / "vlm_results.json").read_text())["frames"]
    pairs = json.loads((RESULTS / "counterfactual.json").read_text())["pairs"][:10]
    jobs = [(f"frame:{i}", frames[i]["visual_prompt"], RESULTS / "vlm" / frames[i]["image"])
            for i in select_frames(frames, np.random.default_rng(0))]
    render_prompt = RENDER_PROMPT.format(stats="", labels=", ".join(LABELS), cells=", ".join(CELLS))
    jobs += [(f"pair:{q['sim']}:{tag}", render_prompt, RESULTS / "counterfactual" / f"{q['sim']:03d}{tag}_render.png")
             for q in pairs for tag in "AB"]
    try:
        for key, prompt, img in jobs:
            if key in done:
                continue
            d, raw = ask(a.model, prompt, img, state)
            done[key] = dict(pred=(d or {}).get("anomaly_type", "invalid"), location=(d or {}).get("location"),
                             explanation=(d or {}).get("explanation"), raw=raw, valid=d is not None)
            done_path.write_text(json.dumps(done, indent=1))
            print(key, done[key]["pred"], flush=True)
    except RuntimeError as e:
        print(f"stopped: {e}; {len(done)}/{len(jobs)} done — rerun tomorrow to resume")
    if len(done) < len(jobs):
        return
    report(done, frames, pairs, a)


def report(done, frames, pairs, a):
    fr = [(int(k.split(":")[1]), v) for k, v in done.items() if k.startswith("frame:")]
    anom = [(frames[i], v) for i, v in fr if frames[i]["truth"] != "none"]
    clean = [(frames[i], v) for i, v in fr if frames[i]["truth"] == "none"]
    acc27 = sum(v["pred"] == f["truth"] for f, v in anom)
    acc8 = sum((f["visual"] or {}).get("anomaly_type") == f["truth"] for f, v in anom)
    disc27 = sum(done[f"pair:{q['sim']}:A"]["pred"] == "none" and done[f"pair:{q['sim']}:B"]["pred"] not in ("none", "invalid")
                 for q in pairs)
    disc8 = sum(q["A_render"]["pred"] == "none" and q["B_render"]["pred"] not in ("none", "invalid") for q in pairs)
    loc = [v["location"] == f["cell"] for f, v in anom if v["location"] in CELLS]
    worth = (acc27 - acc8 >= 5) or disc27 >= 3
    lines = [f"Gate E pilot — {a.model} vs local Qwen3-VL-8B (4-bit), same images and prompts\n",
             "| metric | 27B | 8B |", "|---|---|---|",
             f"| type accuracy, {len(anom)} sev-3 anomalous frames | {acc27}/{len(anom)} | {acc8}/{len(anom)} |",
             f"| answered 'none' on anomalous frames | {sum(v['pred'] == 'none' for f, v in anom)}/{len(anom)} | "
             f"{sum((f['visual'] or {}).get('anomaly_type') == 'none' for f, v in anom)}/{len(anom)} |",
             f"| clean frames called 'none' | {sum(v['pred'] == 'none' for f, v in clean)}/{len(clean)} | "
             f"{sum((f['visual'] or {}).get('anomaly_type') == 'none' for f, v in clean)}/{len(clean)} |",
             f"| exact location cell (anomalous, valid answers) | {sum(loc)}/{len(loc)} | — |",
             f"| counterfactual pairs separated (render only) | {disc27}/{len(pairs)} | {disc8}/{len(pairs)} |",
             f"| invalid JSON | {sum(not v['valid'] for v in done.values())}/{len(done)} | — |",
             f"\n**Pre-registered decision: {'SCALE UP Gate E' if worth else 'do not scale up (no evidence the larger model fixes it)'}**"]
    (a.out / "pilot.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
