"""Figure: what each injected failure looks like (validation sim 20, severity 3), plate coloured by deviation from the
clean trajectory; plus a counterfactual-pair panel with a difference map."""
import json

import matplotlib
import numpy as np
from PIL import Image

from mgn.data import NORMAL
from mgn.evaluate import seed
from mgn.inject import TYPES, inject
from mgn.train import load_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pyvista as pv  # noqa: E402

from mgn.data import OBSTACLE  # noqa: E402

pv.OFF_SCREEN = True
tr = load_split("valid", 21)[20]
nt, cells = tr["node_type"], tr["cells"]
is_obs = (nt[cells] == OBSTACLE).all(1)
normal = nt == NORMAL


def grid(c, pos):
    return pv.UnstructuredGrid(np.hstack([np.full((len(c), 1), 4), c]).ravel(), np.full(len(c), pv.CellType.TETRA),
                               pos.astype(np.float64))


def closeup(pos, focal, view, scalars=None, clim=None, bad_cells=None):
    """Close-up render around `focal`, camera along `view`; actuator translucent so contact failures stay visible."""
    pl = pv.Plotter(off_screen=True, window_size=(420, 340), lighting="three lights")
    pl.set_background("white")
    plate = grid(cells[~is_obs], pos).extract_surface(algorithm="dataset_surface")
    op = 0.25 if bad_cells is not None else 1.0  # cut-away view when highlighting inverted elements
    if bad_cells is not None and len(bad_cells):
        pl.add_mesh(grid(bad_cells, pos), color="#b2182b", show_edges=True, edge_color="black", line_width=1.2)
    if scalars is None or bad_cells is not None:
        pl.add_mesh(plate, color="lightgrey", show_edges=True, edge_color="grey", line_width=0.6, opacity=op)
    else:
        plate["d"] = scalars[plate["vtkOriginalPointIds"]]
        pl.add_mesh(plate, scalars="d", cmap="YlOrRd", clim=clim, show_scalar_bar=False, show_edges=True,
                    edge_color="grey", line_width=0.6)
    pl.add_mesh(grid(cells[is_obs], pos).extract_surface(algorithm="dataset_surface"), color="steelblue", opacity=0.3)
    v = np.asarray(view, float) / np.linalg.norm(view)
    pl.camera.focal_point = focal
    pl.camera.position = focal + 0.17 * v
    pl.camera.up = (0, 0, 1) if abs(v[2]) < 0.9 else (0, 1, 0)
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# per failure: view from the side where it happens (the actuator is on +z; the fixed vision camera looks from -z)
VIEWS = {"hourglass": (0.6, 0.3, -1), "penetration": (0.5, 0.3, 1), "inversion": (0.5, 0.3, 1),
         "instability": (0.6, 0.3, -1), "frozen": (0.6, 0.3, -1)}
fig, axes = plt.subplots(2, len(TYPES), figsize=(10.5, 4.1))
for j, kind in enumerate(TYPES):
    wp, mask, fm = inject(tr, kind, 3, np.random.default_rng(seed(20, kind, 3)))
    ev = np.flatnonzero(fm)
    t = int(ev[-1] if kind == "frozen" else ev[min(2, len(ev) - 1)])  # frozen: end of the 30-frame hold
    dev = np.where(normal, np.linalg.norm(wp[t] - tr["world_pos"][t], axis=1), 0)
    focal = tr["world_pos"][t][mask].mean(0)
    axes[0, j].imshow(closeup(tr["world_pos"][t], focal, VIEWS[kind]))
    bad = None
    if kind == "inversion":
        from mgn.inject import signed_volumes
        pc = cells[~is_obs]
        bad = pc[np.sign(signed_volumes(wp[t], pc)) != np.sign(signed_volumes(tr["mesh_pos"], pc))]
    axes[1, j].imshow(closeup(wp[t], focal, VIEWS[kind], scalars=dev, clim=(0, dev.max()), bad_cells=bad))
    axes[0, j].set_title(kind, fontsize=10, fontweight="bold")
    axes[1, j].set_xlabel(f"{dev.max() * 1000:.1f} mm max shift", fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#cccccc")
axes[0, 0].set_ylabel("clean", fontsize=9)
axes[1, 0].set_ylabel("with failure", fontsize=9)
fig.tight_layout(pad=0.4, w_pad=0.3, h_pad=0.3)
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
