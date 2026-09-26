"""Synthetic anomalies modelled on real FE / crash-simulation failure modes.

Each injector takes a clean trajectory and returns (corrupted world_pos, node_mask). Frame labels are derived
uniformly: a frame is anomalous iff any node deviates from the clean trajectory. Only NORMAL (plate) nodes are
ever corrupted; actuator and clamped nodes keep their prescribed kinematics.
"""
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from mgn.data import NORMAL, OBSTACLE, T_MAX

TYPES = ["hourglass", "penetration", "inversion", "instability", "frozen"]
# Phase 2: whole-run, spatially and temporally smooth errors that are inconsistent with the loading. The actuator
# prescribes displacement here, so a wrong material stiffness would mostly change stress, not shape; `scale` is
# therefore a response-scale / contact-consistency error (e.g. an output scaling bug), not a material-card error.
GLOBAL_TYPES = ["scale", "lag", "timescale"]
GLOBAL_SEVERITY = {"scale": {1: 0.05, 2: 0.10, 3: 0.20},   # plate displacement x (1 + eps)
                   "lag": {1: 1, 2: 3, 3: 10},             # plate response k frames behind the actuator
                   "timescale": {1: 0.95, 2: 0.90, 3: 0.80}}  # plate response at c x the correct rate
SEVERITY = {1: 1.0, 2: 10.0, 3: 100.0}  # amplitude in multiples of the RMS per-step displacement (~0.3 mm)


def inject_global(traj, kind, severity, param=None):
    """Whole-run global error on plate (NORMAL) nodes only; actuator and clamp keep their kinematics. Deterministic."""
    p = GLOBAL_SEVERITY[kind][severity] if param is None else param
    wp, mp = traj["world_pos"].copy(), traj["mesh_pos"]
    normal = traj["node_type"] == NORMAL
    u = wp[:, normal] - mp[normal]
    t = np.arange(len(wp))
    if kind == "scale":
        u2 = (1 + p) * u
    elif kind == "lag":
        u2 = u[np.maximum(t - int(p), 0)]
    elif kind == "timescale":
        tt = p * t
        lo = np.floor(tt).astype(int)
        w = (tt - lo)[:, None, None]
        u2 = (1 - w) * u[lo] + w * u[np.minimum(lo + 1, len(wp) - 1)]
    else:
        raise ValueError(kind)
    wp[:, normal] = mp[normal] + u2
    return wp


def adjacency(traj):
    s, r = traj["mesh_edges"]
    n = len(traj["node_type"])
    return coo_matrix((np.ones(len(s), bool), (s, r)), shape=(n, n)).tocsr()


def bfs_depth(adj, center, hops, allowed):
    """Hop distance from `center` for nodes within `hops` (restricted to `allowed`); -1 elsewhere."""
    depth = np.full(adj.shape[0], -1)
    depth[center], frontier = 0, [center]
    for h in range(1, hops + 1):
        nb = np.unique(adj[frontier].indices)
        frontier = nb[(depth[nb] < 0) & allowed[nb]]
        depth[frontier] = h
    return depth


def step_scale(traj):
    d = np.linalg.norm(np.diff(traj["world_pos"], axis=0), axis=-1)[:, traj["node_type"] == NORMAL]
    return np.sqrt((d ** 2).mean())


def actuator_surface(traj, pos):
    """Outward-oriented triangle surface of the actuator at positions `pos` (cell normals in cell_data["Normals"])."""
    nt, cells = traj["node_type"], traj["cells"]
    obs_cells = cells[(nt[cells] == OBSTACLE).all(1)]
    surf = pv.UnstructuredGrid(np.c_[np.full(len(obs_cells), 4), obs_cells].ravel(),
                               np.full(len(obs_cells), pv.CellType.TETRA), pos.astype(np.float64)
                               ).extract_surface(algorithm="dataset_surface").compute_normals(
                                   cell_normals=True, point_normals=False, auto_orient_normals=True)
    # enforce outward normals (away from the actuator centroid; actuators are convex). A no-op where auto-orient
    # already succeeded, which it does on the real meshes; it can fail on tiny surfaces such as a single tet.
    n = surf.cell_data["Normals"]
    flip = np.einsum("ij,ij->i", surf.cell_centers().points - surf.points.mean(0), n) < 0
    surf.cell_data["Normals"] = np.where(flip[:, None], -n, n)
    return surf


def actuator_distance(traj, pos, nodes):
    """Signed distance of `nodes` to the actuator surface (negative = inside the actuator). Inside/outside comes from
    a point-in-tet test on the actuator's volume mesh; face-normal signs are unreliable near edges and vertices."""
    nt, cells = traj["node_type"], traj["cells"]
    obs_cells = cells[(nt[cells] == OBSTACLE).all(1)]
    vol = pv.UnstructuredGrid(np.c_[np.full(len(obs_cells), 4), obs_cells].ravel(),
                              np.full(len(obs_cells), pv.CellType.TETRA), pos.astype(np.float64))
    p = pos[nodes].astype(np.float64)
    _, closest = vol.extract_surface(algorithm="dataset_surface").find_closest_cell(p, return_closest_point=True)
    dist = np.linalg.norm(p - closest, axis=1)
    return np.where(vol.find_containing_cell(p) >= 0, -dist, dist)


def signed_volumes(pos, cells):
    a, b, c, d = (pos[cells[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6


def _moving_center(traj, t0, rng):
    """A plate node in the top quartile of motion around t0 (so the anomaly isn't in a region that never moves)."""
    normal = np.flatnonzero(traj["node_type"] == NORMAL)
    motion = np.linalg.norm(traj["world_pos"][t0 + 10, normal] - traj["world_pos"][t0 - 10, normal], axis=1)
    return rng.choice(normal[motion >= np.quantile(motion, 0.75)])


def inject(traj, kind, severity, rng, t0=None):
    """Returns (world_pos, node_mask, frame_mask) for one anomaly event."""
    wp, nt = traj["world_pos"].copy(), traj["node_type"]
    normal = nt == NORMAL
    adj = adjacency(traj)
    a = SEVERITY[severity] * step_scale(traj)
    t0_given = t0 is not None
    t0 = int(rng.integers(40, 300)) if t0 is None else t0
    k = 5  # event duration in frames (except frozen/instability, see below)

    if kind == "hourglass":
        # checkerboard (alternating by hop parity) offset along one fixed direction: a zero-energy-like mode
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 2, normal)
        mask = depth >= 0
        sign = np.where(depth % 2 == 0, 1.0, -1.0)
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        wp[t0:t0 + k, mask] += a * sign[mask, None] * direction

    elif kind == "penetration":
        # plate nodes at the contact pushed through the actuator surface; severity = penetration depth
        obs = np.flatnonzero(nt == OBSTACLE)
        plate = np.flatnonzero(normal)
        if not t0_given:  # only start once the actuator actually touches the plate
            gaps = np.array([cKDTree(wp[t][obs]).query(wp[t][plate])[0].min() for t in range(40, 300)])
            contact = np.flatnonzero(gaps < 2e-3) + 40
            t0 = int(rng.choice(contact)) if len(contact) else int(np.argmin(gaps)) + 40
        _, cp = actuator_surface(traj, wp[t0]).find_closest_cell(wp[t0][plate].astype(np.float64), return_closest_point=True)
        dist = np.linalg.norm(cp - wp[t0][plate], axis=1)
        order = np.argsort(dist)[:10]
        close = plate[order[dist[order] <= max(1e-3, dist[order[0]])]]  # nodes touching the actuator (<1 mm)
        mask = np.zeros(len(nt), bool)
        mask[close] = True
        for t in range(t0, t0 + k):
            # closest point on the actuator surface, then `a` further inward along the surface normal
            surf = actuator_surface(traj, wp[t])
            cid, closest = surf.find_closest_cell(wp[t][close].astype(np.float64), return_closest_point=True)
            wp[t, close] = closest - a * surf.cell_data["Normals"][cid]

    elif kind == "inversion":
        # move one vertex of a plate tet through its opposite face -> negative Jacobian (inverted element)
        cells = traj["cells"]
        center = _moving_center(traj, t0, rng)
        cand = np.flatnonzero((cells == center).any(1) & normal[cells].all(1))
        cell = cells[rng.choice(cand)]
        others = cell[cell != center]
        frac = {1: 0.3, 2: 0.6, 3: 1.5}[severity]  # >0.5 of the reflection distance inverts the tet; 0.3 = distorted only
        mask = np.zeros(len(nt), bool)
        mask[center] = True
        for t in range(t0, t0 + k):
            p, q = wp[t, center], wp[t, others]
            n = np.cross(q[1] - q[0], q[2] - q[0])
            n /= np.linalg.norm(n)
            wp[t, center] = p - frac * 2 * np.dot(p - q[0], n) * n

    elif kind == "instability":
        # local oscillation growing geometrically over k frames (numerical blow-up)
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 1, normal)
        mask = depth >= 0
        k = 8
        noise = rng.normal(size=(mask.sum(), 3))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        for j in range(k):  # grows 1.6x per frame, sign-alternating, peaking at `a` in the last frame
            wp[t0 + j, mask] += a * 1.6 ** (j - k + 1) * (-1) ** j * noise

    elif kind == "frozen":
        # a region stops following the deformation (lost constraint / BC error); severity = duration
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 2, normal)
        mask = depth >= 0
        k = {1: 3, 2: 10, 3: 30}[severity]
        wp[t0 + 1:t0 + k, mask] = wp[t0, mask]

    elif kind == "blowup":
        # lead-time study only (not in TYPES): starts at this simulation's clean-acceleration floor (RMS second
        # difference of plate nodes, median over the steady regime) and grows 1.3x/frame to a cap of 100x the step
        # displacement, then holds at the cap for 5 frames. Onset is chosen so the cap lands before T_MAX.
        s0 = step_scale(traj)
        acc = np.linalg.norm(traj["world_pos"][2:T_MAX] - 2 * traj["world_pos"][1:T_MAX - 1] + traj["world_pos"][:T_MAX - 2], axis=-1)
        start = max(np.median(np.sqrt((acc[8:, normal] ** 2).mean(1))), 1e-9)
        n_grow = min(int(np.ceil(np.log(100 * s0 / start) / np.log(1.3))) + 1, T_MAX - 50)
        start = 100 * s0 / 1.3 ** (n_grow - 1)
        if not t0_given:
            t0 = int(rng.integers(40, T_MAX - n_grow - 5))
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 1, normal)
        mask = depth >= 0
        noise = rng.normal(size=(mask.sum(), 3))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        for j in range(n_grow + 5):
            wp[t0 + j, mask] += min(start * 1.3 ** j, 100 * s0) * (-1) ** j * noise

    else:
        raise ValueError(kind)

    frame_mask = np.abs(wp - traj["world_pos"]).max(axis=(1, 2)) > 0
    return wp, mask, frame_mask
