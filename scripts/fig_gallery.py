"""Figure: what each injected failure looks like (validation sim 20, severity 3), plate coloured by deviation from the
clean trajectory; plus a counterfactual-pair panel with a difference map."""
import json

import matplotlib
import numpy as np
from PIL import Image

from mgn.data import NORMAL
from mgn.evaluate import seed
from mgn.inject import TYPES, inject
from mgn.render import Renderer
from mgn.train import load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

tr = load_split("valid", 21)[20]
rend, normal = Renderer(tr), tr["node_type"] == NORMAL
fig, axes = plt.subplots(1, len(TYPES), figsize=(10.5, 2.4))
for ax, kind in zip(axes, TYPES):
    wp, mask, fm = inject(tr, kind, 3, np.random.default_rng(seed(20, kind, 3)))
    ev = np.flatnonzero(fm)
    t = int(ev[min(2, len(ev) - 1)])
    dev = np.where(normal, np.linalg.norm(wp[t] - tr["world_pos"][t], axis=1), 0)
    img = rend.render(wp[t], scalars=dev, clim=(0, max(dev.max(), 1e-9)))
    c = rend.project(wp[t][mask]).mean(0).astype(int)  # crop around the injected nodes
    h = 70
    x0, y0 = np.clip(c - h, 0, img.shape[1] - 2 * h)
    ax.imshow(img[y0:y0 + 2 * h, x0:x0 + 2 * h])
    ax.set_title(kind, fontsize=9)
    ax.set_axis_off()
fig.tight_layout(pad=0.4)
fig.savefig("docs/report/figs/failure_gallery.png", dpi=200, facecolor="white")

# counterfactual pair with a magnified difference map
q = json.load(open("results/counterfactual.json"))["pairs"][0]
a = np.asarray(Image.open(f"results/counterfactual/{q['sim']:03d}A_render.png").convert("RGB")).astype(float)
b = np.asarray(Image.open(f"results/counterfactual/{q['sim']:03d}B_render.png").convert("RGB")).astype(float)
diff = np.abs(a - b).sum(-1)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9))
for ax, img, title in zip(axes, [a / 255, b / 255], ["A: clean", "B: + hourglass (sev. 3)"]):
    ax.imshow(img)
    ax.set_title(title, fontsize=9)
axes[2].imshow(diff > 30, cmap="Greys")
axes[2].set_title("|A − B| (changed pixels)", fontsize=9)
for ax in axes:
    ax.set_axis_off()
for ax in axes:
    ax.set_ylim(350, 115)  # trim empty space
fig.tight_layout(pad=0.8)
fig.savefig("docs/report/figs/counterfactual_pair.png", dpi=200, facecolor="white")
print("saved")
