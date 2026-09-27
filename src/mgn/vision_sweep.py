"""Gate V (pre-registered in docs/gate_v.md): modern vision backbones x resolution on rendered failures (validation)."""
import argparse
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from mgn.data import NORMAL
from mgn.detect import ClipDetector
from mgn.evaluate import RESULTS, bootstrap, choose_frames, seed
from mgn.inject import TYPES, inject
from mgn.render import Renderer
from mgn.train import device, load_split

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def crop_box(rend, traj):
    """Plate bounding box in pixels from CLEAN positions at the first and last modelled frame (no anomaly leakage)."""
    plate = traj["node_type"] == NORMAL
    xy = np.concatenate([rend.project(traj["world_pos"][t][plate]) for t in (0, 349)])
    lo, hi = xy.min(0), xy.max(0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.08 + 4
    x0, y0 = np.clip((c - half).astype(int), 0, rend.size - 1)
    x1, y1 = np.clip((c + half).astype(int), 1, rend.size)
    return x0, y0, x1, y1


def to_tensor(img, size, mean, std):
    from PIL import Image
    im = Image.fromarray(img).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(((np.asarray(im) / 255.0 - mean) / std).transpose(2, 0, 1).astype(np.float32))


def tiles(img):
    h, w = img.shape[:2]
    return [img[:h // 2, :w // 2], img[:h // 2, w // 2:], img[h // 2:, :w // 2], img[h // 2:, w // 2:]]


class PatchBackbone:
    def __init__(self, name, dev):
        self.name, self.dev = name, dev
        if name == "dinov2_l":
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained("facebook/dinov2-large").to(dev).eval()
            self.mean, self.std, self.inp = MEAN, STD, 448
        else:
            arch, pre = {"clip_b16": ("ViT-B-16", "openai"), "clip_l14": ("ViT-L-14", "laion2b_s32b_b82k")}[name]
            self.model, _, pp = open_clip.create_model_and_transforms(arch, pretrained=pre, device=dev)
            self.model.eval()
            self.model.visual.output_tokens = True
            norm = [t for t in pp.transforms if t.__class__.__name__ == "Normalize"][0]
            self.mean, self.std, self.inp = np.array(norm.mean), np.array(norm.std), 224

    @torch.no_grad()
    def patches(self, images):
        """images: list of HxWx3 uint8 -> list of [P, D] L2-normalised patch features."""
        out = []
        for i in range(0, len(images), 16):
            x = torch.stack([to_tensor(im, self.inp, self.mean, self.std) for im in images[i:i + 16]]).to(self.dev)
            if self.name == "dinov2_l":
                f = self.model(pixel_values=x).last_hidden_state[:, 1:]
            else:
                f = self.model.visual(x)[1]
            out += list(torch.nn.functional.normalize(f, dim=-1))
        return out


class ZeroShot:
    """Anomaly score = cos(image, 'damaged' prompts) - cos(image, 'flawless' prompts); rank-equivalent to the
    two-class softmax used in Phase 1, and valid for SigLIP too (no patch tokens needed)."""

    def __init__(self, arch, pretrained, dev):
        self.dev = dev
        self.model, _, self.pp = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=dev)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            text = []
            for states in ClipDetector.STATES:
                prompts = [t.format(st.format("bent metal plate")) for st in states for t in ClipDetector.TEMPLATES]
                e = torch.nn.functional.normalize(self.model.encode_text(tok(prompts).to(dev)), dim=-1).mean(0)
                text.append(e / e.norm())
        self.text = torch.stack(text)

    @torch.no_grad()
    def score(self, images):
        from PIL import Image
        x = torch.stack([self.pp(Image.fromarray(im).convert("RGB")) for im in images]).to(self.dev)
        f = torch.nn.functional.normalize(self.model.encode_image(x), dim=-1) @ self.text.T
        return (f[:, 1] - f[:, 0]).cpu().tolist()


def views(img448, img896, box, backbone):
    """Condition (a): the full 448 render. Condition (b): plate crop from the 896 render (tiled 2x2 for 224-px models)."""
    x0, y0, x1, y1 = box
    crop = img896[y0:y1, x0:x1]
    return {"448": [img448], "896crop": [crop] if backbone == "dinov2_l" else tiles(crop)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--stop", type=int, default=40)
    ap.add_argument("--out", type=Path, default=RESULTS / "gate_v")
    a = ap.parse_args()
    dev = device()
    backbones = ["clip_b16", "clip_l14", "dinov2_l"]

    # 1. render everything once (both resolutions), keeping the per-sim crop boxes
    def render_pair(tr, frames_wp):
        r448, r896 = Renderer(tr), Renderer(tr, size=896)
        box = crop_box(r896, tr)
        return [(r448.render(wp), r896.render(wp)) for wp in frames_wp], box

    bank_imgs = []
    for tr in tqdm(load_split("valid", 20), desc="memory bank renders"):
        imgs, box = render_pair(tr, [tr["world_pos"][t] for t in (50, 120, 190, 260, 330)])
        bank_imgs += [(i4, i8, box) for i4, i8 in imgs]
    items = []  # (sim, type, sev, label, img448, img896, box)
    for i, tr in zip(range(a.start, a.stop), tqdm(load_split("valid", a.stop)[a.start:], desc="eval renders")):
        for kind in TYPES:
            for sev in (2, 3):
                rng = np.random.default_rng(seed(i, kind, sev))
                wp, _, fm = inject(tr, kind, sev, rng)
                frames, labels, phase = choose_frames(fm, rng)
                ev = frames[labels]
                pick = list(ev[[0] + list(np.unique(np.linspace(1, len(ev) - 1, min(2, len(ev) - 1)).astype(int)))]) if len(ev) > 1 else list(ev)
                pick += list(frames[~labels][:3])
                imgs, box = render_pair(tr, [wp[t] for t in pick])
                items += [(i, kind, sev, t in set(ev), i4, i8, box) for t, (i4, i8) in zip(pick, imgs)]

    # 2. patch-kNN scores per backbone x view
    scores = {}
    for bb in backbones:
        net = PatchBackbone(bb, dev)
        for view in ("448", "896crop"):
            bank = torch.cat([f for i4, i8, box in bank_imgs for f in net.patches(views(i4, i8, box, bb)[view])])
            s = []
            for it in tqdm(items, desc=f"{bb}/{view}"):
                feats = torch.cat(net.patches(views(it[4], it[5], it[6], bb)[view]))
                d = torch.cat([1 - (feats[j:j + 2048] @ bank.T).max(1).values for j in range(0, len(feats), 2048)])
                s.append(float(d.max()))
            scores[f"knn_{bb}_{view}"] = s
        del net
        torch.mps.empty_cache() if dev.type == "mps" else None
    # zero-shot (global image-text, Phase 1 WinCLIP prompt ensemble): CLIP B/16 and SigLIP2
    for tag, arch, pre in (("clip_b16", "ViT-B-16", "openai"), ("siglip2", "ViT-B-16-SigLIP2-512", "webli")):
        zs = ZeroShot(arch, pre, dev)
        for view in ("448", "896crop"):
            scores[f"zs_{tag}_{view}"] = [max(zs.score(views(it[4], it[5], it[6], "clip")[view]))
                                          for it in tqdm(items, desc=f"zs {tag}/{view}")]
        del zs

    # 3. AUROC per condition x type x severity, bootstrap over sims
    sims = np.array([it[0] for it in items])
    res, lines = {}, ["| condition | " + " | ".join(f"{k[:5]} s{s}" for k in TYPES for s in (2, 3)) + " | mean s3 |",
                      "|---|" + "---|" * (2 * len(TYPES) + 1)]
    for cond, s in scores.items():
        s = np.array(s)
        row, s3 = [], []
        for kind in TYPES:
            for sev in (2, 3):
                m = np.array([it[1] == kind and it[2] == sev for it in items])
                y = np.array([it[3] for it in items])[m]
                auc, lo, hi = bootstrap(lambda i: roc_auc_score(y[i], s[m][i]), sims[m])
                res[f"{cond}/{kind}/{sev}"] = dict(auc=auc, lo=lo, hi=hi)
                row.append(f"{auc:.2f}")
                if sev == 3:
                    s3.append(auc)
        res[f"{cond}/mean_s3"] = float(np.mean(s3))
        lines.append(f"| {cond} | " + " | ".join(row) + f" | **{np.mean(s3):.2f}** |")
    passed = [c for c in scores if res[f"{c}/mean_s3"] >= 0.80]
    lines.append(f"\n**Gate V: {'vision NOT blind — passing: ' + ', '.join(passed) if passed else 'no condition reaches 0.80 at severity 3'}**")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate_v.json").write_text(json.dumps(res, indent=1))
    (a.out / "gate_v.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
