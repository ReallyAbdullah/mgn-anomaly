"""Synthetic anomalies modelled on real FE / crash-simulation failure modes.

Each injector takes a clean trajectory and returns (corrupted world_pos, node_mask). Frame labels are derived
uniformly: a frame is anomalous iff any node deviates from the clean trajectory. Only NORMAL (plate) nodes are
ever corrupted; actuator and clamped nodes keep their prescribed kinematics.
"""
import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from mgn.data import NORMAL, OBSTACLE

TYPES = ["hourglass", "penetration", "inversion", "instability", "frozen"]
BLOWUP_GROW = 28  # 0.1 * 1.3**j first reaches the 100x cap at j = 27
SEVERITY = {1: 1.0, 2: 10.0, 3: 100.0}  # amplitude in multiples of the RMS per-step displacement (~0.3 mm)


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
        obs_cells = traj["cells"][(nt[traj["cells"]] == OBSTACLE).all(1)]

        def actuator_surface(pos):
            return pv.UnstructuredGrid(np.c_[np.full(len(obs_cells), 4), obs_cells].ravel(),
                                       np.full(len(obs_cells), pv.CellType.TETRA), pos.astype(np.float64)
                                       ).extract_surface(algorithm="dataset_surface").compute_normals(
                                           cell_normals=True, point_normals=False, auto_orient_normals=True)

        _, cp = actuator_surface(wp[t0]).find_closest_cell(wp[t0][plate].astype(np.float64), return_closest_point=True)
        dist = np.linalg.norm(cp - wp[t0][plate], axis=1)
        order = np.argsort(dist)[:10]
        close = plate[order[dist[order] <= max(1e-3, dist[order[0]])]]  # nodes touching the actuator (<1 mm)
        mask = np.zeros(len(nt), bool)
        mask[close] = True
        for t in range(t0, t0 + k):
            # closest point on the actuator surface, then `a` further inward along the surface normal
            surf = actuator_surface(wp[t])
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
        # lead-time study only (not in TYPES): grows 1.3x/frame from 0.1x to a cap of 100x the step displacement
        # (reached at frame t0 + BLOWUP_GROW - 1), then holds at the cap for 5 frames
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 1, normal)
        mask = depth >= 0
        s0 = step_scale(traj)
        noise = rng.normal(size=(mask.sum(), 3))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        for j in range(BLOWUP_GROW + 5):
            wp[t0 + j, mask] += min(0.1 * 1.3 ** j, 100.0) * s0 * (-1) ** j * noise

    else:
        raise ValueError(kind)

    frame_mask = np.abs(wp - traj["world_pos"]).max(axis=(1, 2)) > 0
    return wp, mask, frame_mask
