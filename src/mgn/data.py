"""deforming_plate: stream TFRecords -> .npz, and build MeshGraphNet input graphs.

No TensorFlow: TFRecord framing and the tf.train.Example protobuf are parsed by hand.
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
import requests
from scipy.spatial import cKDTree

URL = "https://storage.googleapis.com/dm-meshgraphnets/deforming_plate"
DATA = Path(__file__).resolve().parents[2] / "data"
NORMAL, OBSTACLE, HANDLE = 0, 1, 3
NUM_TYPES = 4  # one-hot over {0,1,2,3}; type 2 never occurs but keeps indices == type ids
RADIUS = 0.03
T_MAX = 350  # actuator speeds up ~9x near frame 360; we model/score the steady-actuator regime only


# ---------- TFRecord / protobuf parsing ----------

def _varint(buf, i):
    out = shift = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if b < 0x80:
            return out, i
        shift += 7


def _fields(buf):
    """Yield (field_number, bytes) for length-delimited fields of a protobuf message."""
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        assert key & 7 == 2, "only length-delimited fields expected"
        n, i = _varint(buf, i)
        yield key >> 3, buf[i:i + n]
        i += n


def parse_example(buf):
    """tf.train.Example -> {name: raw bytes}. Every MGN feature is a single-element bytes_list."""
    out = {}
    (_, features), = _fields(buf)            # Example.features
    for _, entry in _fields(features):       # Features.feature (map entries)
        kv = dict(_fields(entry))
        (_, bytes_list), = _fields(kv[2])    # Feature.bytes_list
        (_, value), = _fields(bytes_list)    # BytesList.value
        out[kv[1].decode()] = value
    return out


def read_records(stream):
    """Yield raw record payloads from a TFRecord byte stream (CRCs are skipped, not checked)."""
    while header := stream.read(12):
        (n,) = struct.unpack("<Q", header[:8])
        yield stream.read(n)
        stream.read(4)


def decode(raw, meta):
    out = {}
    for name, spec in meta["features"].items():
        if name == "stress":
            continue
        arr = np.frombuffer(raw[name], dtype=spec["dtype"]).reshape(spec["shape"])
        out[name] = arr[0] if spec["type"] == "static" else arr
    out["node_type"] = out["node_type"][:, 0]
    return out


def stream_convert(split, n):
    meta = requests.get(f"{URL}/meta.json", timeout=30).json()
    out = DATA / split
    out.mkdir(parents=True, exist_ok=True)
    with requests.get(f"{URL}/{split}.tfrecord", stream=True, timeout=60) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        for i, raw in enumerate(read_records(r.raw)):
            if i >= n:
                break
            path = out / f"traj_{i:04d}.npz"
            if not path.exists():
                np.savez(path, **decode(parse_example(raw), meta))
            print(f"{split} {i + 1}/{n}", end="\r", flush=True)
    print()


# ---------- graph construction ----------

def mesh_edges(cells):
    """Unique undirected tet edges, returned bidirectional as (senders, receivers)."""
    pairs = cells[:, [0, 0, 0, 1, 1, 2, 1, 2, 3, 2, 3, 3]].reshape(-1, 2, 6).transpose(0, 2, 1).reshape(-1, 2)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return np.concatenate([pairs, pairs[:, ::-1]]).T.astype(np.int64)


def world_edges(pos, node_type):
    """Obstacle <-> plate node pairs within RADIUS in world space (bidirectional)."""
    obs = np.flatnonzero(node_type == OBSTACLE)
    plate = np.flatnonzero(node_type != OBSTACLE)
    hits = cKDTree(pos[plate]).query_ball_point(pos[obs], RADIUS)
    s = np.repeat(obs, [len(h) for h in hits])
    r = plate[np.concatenate(hits).astype(np.int64)] if len(s) else np.zeros(0, np.int64)
    return np.stack([np.concatenate([s, r]), np.concatenate([r, s])]).astype(np.int64)


def load_traj(path):
    d = dict(np.load(path))
    d["mesh_edges"] = mesh_edges(d["cells"])
    return d


def build_graph(traj, prev_pos, pos, next_pos):
    """Inputs for step t -> t+1 (MGN cloth-style: current velocity in, acceleration out).

    `prev_pos`/`pos` may be noisy or corrupted; `next_pos` is only used for the prescribed actuator kinematics.
    """
    nt = traj["node_type"]
    onehot = np.eye(NUM_TYPES, dtype=np.float32)[nt]
    vel = pos - prev_pos
    obs_vel = np.where((nt == OBSTACLE)[:, None], next_pos - pos, 0)
    s, r = me = traj["mesh_edges"]
    dm = traj["mesh_pos"][s] - traj["mesh_pos"][r]
    dw = pos[s] - pos[r]
    we = world_edges(pos, nt)
    dww = pos[we[0]] - pos[we[1]]
    norm = lambda v: np.linalg.norm(v, axis=1, keepdims=True)
    return {
        "x": np.concatenate([vel, obs_vel, onehot], 1).astype(np.float32),
        "mesh_edges": me,
        "mesh_attr": np.concatenate([dm, norm(dm), dw, norm(dw)], 1).astype(np.float32),
        "world_edges": we,
        "world_attr": np.concatenate([dww, norm(dww)], 1).astype(np.float32),
    }


def collate(graphs, pad=False):
    """Concatenate graphs into one disconnected graph, offsetting edge indices.

    pad=True rounds node/edge counts up to fixed buckets (extra edges hit one dummy node). On MPS every new tensor
    shape compiles and caches a new kernel graph, so unpadded variable-size batches get slower and slower.
    """
    out, off = {}, 0
    for g in graphs:
        for k, v in g.items():
            out.setdefault(k, []).append(v + off if k.endswith("_edges") else v)
        off += len(g["x"])
    out = {k: np.concatenate(v, axis=1 if k.endswith("_edges") else 0) for k, v in out.items()}
    if pad:
        n = _bucket(off + 1, 512)
        out["x"] = np.pad(out["x"], ((0, n - off), (0, 0)))
        for e in ("mesh", "world"):
            extra = _bucket(out[f"{e}_edges"].shape[1] + 1, 1024) - out[f"{e}_edges"].shape[1]
            out[f"{e}_edges"] = np.pad(out[f"{e}_edges"], ((0, 0), (0, extra)), constant_values=n - 1)
            out[f"{e}_attr"] = np.pad(out[f"{e}_attr"], ((0, extra), (0, 0)))
    return out


def _bucket(n, mult):
    return -(-n // mult) * mult


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stream-convert deforming_plate TFRecords to .npz")
    p.add_argument("split", choices=["train", "valid", "test"])
    p.add_argument("-n", type=int, default=100)
    a = p.parse_args()
    stream_convert(a.split, a.n)
