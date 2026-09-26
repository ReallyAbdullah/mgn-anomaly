"""Shape-from-load surrogate: predicts the plate's total displacement from where the actuator is.

The problem is quasi-static and hyperelastic, so the deformed shape is (to first order) a function of the actuator
position alone. Unlike the velocity-conditioned one-step model, this model never sees the plate's own motion, so a run
whose plate is globally mis-scaled, lagged or time-stretched cannot hand it the error as input.

Graph (static per trajectory, so tensor shapes are constant within a run):
- node features: one-hot type; the actuator's displacement from rest (actuator nodes only); the mean actuator
  displacement broadcast to every node (global load, no message-passing reach limit); the node's rest position and its
  rest offset from the actuator's rest centroid (geometry). No plate world positions are ever used.
- mesh edges: rest geometry only [dmesh_pos, |dmesh_pos|]
- contact edges: each plate node <-> its <= 8 nearest actuator nodes within CONTACT_R at rest, with feature
  [actuator world position(t) - plate rest position, norm]
Target: plate displacement u = world_pos - mesh_pos (loss on NORMAL nodes).
"""
import numpy as np
import torch
from scipy.spatial import cKDTree

from mgn.data import NORMAL, NUM_TYPES, OBSTACLE, T_MAX, collate

CONTACT_R, CONTACT_K = 0.05, 8
NODE_IN, MESH_IN, WORLD_IN = NUM_TYPES + 12, 4, 4


def prepare(traj):
    """Cache the static contact edges (actuator a -> plate p pairs) on the trajectory dict."""
    if "shape_pairs" not in traj:
        nt, mp = traj["node_type"], traj["mesh_pos"]
        obs, plate = np.flatnonzero(nt == OBSTACLE), np.flatnonzero(nt != OBSTACLE)
        d, j = cKDTree(mp[obs]).query(mp[plate], k=CONTACT_K, distance_upper_bound=CONTACT_R)
        ok = np.isfinite(d)
        traj["shape_pairs"] = np.stack([obs[j[ok]], np.repeat(plate, CONTACT_K).reshape(-1, CONTACT_K)[ok]])
    return traj


def build_shape_graph(traj, act_pos):
    """`act_pos`: world positions [N, 3] of which only OBSTACLE rows are used (the load)."""
    prepare(traj)
    nt, mp = traj["node_type"], traj["mesh_pos"]
    is_obs = (nt == OBSTACLE)[:, None]
    act_disp = act_pos - mp
    load = np.broadcast_to(act_disp[nt == OBSTACLE].mean(0), mp.shape)
    to_act = mp - mp[nt == OBSTACLE].mean(0)
    x = np.concatenate([np.where(is_obs, act_disp, 0), load, mp, to_act, np.eye(NUM_TYPES)[nt]], 1).astype(np.float32)
    s, r = traj["mesh_edges"]
    dm = mp[s] - mp[r]
    a, p = traj["shape_pairs"]
    rel = act_pos[a] - mp[p]
    norm = lambda v: np.linalg.norm(v, axis=1, keepdims=True)
    return {"x": x,
            "mesh_edges": traj["mesh_edges"],
            "mesh_attr": np.concatenate([dm, norm(dm)], 1).astype(np.float32),
            "world_edges": np.concatenate([np.stack([a, p]), np.stack([p, a])], 1).astype(np.int64),
            "world_attr": np.concatenate([np.r_[rel, -rel], norm(np.r_[rel, rel])], 1).astype(np.float32)}


def sample(trajs, idx, pad=False):
    graphs, ys, masks = [], [], []
    for i, t in idx:
        tr = trajs[i]
        graphs.append(build_shape_graph(tr, tr["world_pos"][t]))
        ys.append((tr["world_pos"][t] - tr["mesh_pos"]).astype(np.float32))
        masks.append(tr["node_type"] == NORMAL)
    g, y, m = collate(graphs, pad), np.concatenate(ys), np.concatenate(masks)
    extra = len(g["x"]) - len(y)
    return g, np.pad(y, ((0, extra), (0, 0))), np.pad(m, (0, extra))


@torch.no_grad()
def predict_run(model, traj, wp, frames, dev, batch=8):
    """Predicted plate displacement [F, N, 3] for frames of a run, from that run's actuator positions."""
    from mgn.train import to_torch
    n, out = len(traj["node_type"]), []
    for i in range(0, len(frames), batch):
        fs = frames[i:i + batch]
        g = collate([build_shape_graph(traj, wp[t]) for t in fs], pad=True)
        u = model.predict(to_torch(g, dev)).cpu().numpy()
        out += [u[j * n:(j + 1) * n] for j in range(len(fs))]
    return np.stack(out)


@torch.no_grad()
def evaluate(model, val, dev):
    """Plate-displacement RMSE vs. the zero-displacement baseline, on fixed frames."""
    from mgn.train import to_torch
    model.eval()
    rng = np.random.default_rng(0)
    idx = [(i, t) for i in range(len(val)) for t in 1 + rng.choice(T_MAX - 1, 20, replace=False)]
    se = base = n = 0.0
    for j in range(0, len(idx), 8):
        g, y, m = sample(val, idx[j:j + 8], pad=True)
        pred = model.predict(to_torch(g, dev)).cpu().numpy()
        se += ((pred - y)[m] ** 2).sum()
        base += (y[m] ** 2).sum()
        n += m.sum()
    model.train()
    return np.sqrt(se / n), np.sqrt(base / n)
