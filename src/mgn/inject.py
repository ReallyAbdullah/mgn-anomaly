"""Synthetic anomalies modelled on real FE / crash-simulation failure modes.

Each injector takes a clean trajectory and returns (corrupted world_pos, node_mask). Frame labels are derived
uniformly: a frame is anomalous iff any node deviates from the clean trajectory. Only NORMAL (plate) nodes are
ever corrupted; actuator and clamped nodes keep their prescribed kinematics.
"""
import numpy as np
from scipy.sparse import coo_matrix

from mgn.data import NORMAL, OBSTACLE

TYPES = ["hourglass", "penetration", "inversion", "instability", "frozen"]
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
        # plate nodes nearest the actuator pushed further into it (contact constraint violated)
        obs = np.flatnonzero(nt == OBSTACLE)
        plate = np.flatnonzero(normal)
        gap = np.linalg.norm(wp[t0][plate][:, None] - wp[t0][obs][None], axis=-1)
        close = plate[np.argsort(gap.min(1))[:10]]
        mask = np.zeros(len(nt), bool)
        mask[close] = True
        for t in range(t0, t0 + k):
            nearest = obs[np.argmin(np.linalg.norm(wp[t][close][:, None] - wp[t][obs][None], axis=-1), axis=1)]
            direction = wp[t, nearest] - wp[t, close]
            wp[t, close] += a * direction / np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-9)

    elif kind == "inversion":
        # move one vertex of a plate tet through its opposite face -> negative Jacobian (inverted element)
        cells = traj["cells"]
        center = _moving_center(traj, t0, rng)
        cand = np.flatnonzero((cells == center).any(1) & normal[cells].all(1))
        cell = cells[rng.choice(cand)]
        others = cell[cell != center]
        frac = {1: 0.6, 2: 1.0, 3: 1.5}[severity]  # >0.5 of the reflection distance inverts the tet
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
        for j in range(k):
            wp[t0 + j, mask] += a * 0.3 * 1.6 ** j * (-1) ** j * noise

    elif kind == "frozen":
        # a region stops following the deformation (lost constraint / BC error); severity = duration
        depth = bfs_depth(adj, _moving_center(traj, t0, rng), 2, normal)
        mask = depth >= 0
        k = {1: 3, 2: 10, 3: 30}[severity]
        wp[t0 + 1:t0 + k, mask] = wp[t0, mask]

    else:
        raise ValueError(kind)

    frame_mask = np.abs(wp - traj["world_pos"]).max(axis=(1, 2)) > 0
    return wp, mask, frame_mask
