import io
import struct

import numpy as np
import pytest
import torch

from mgn.data import build_graph, collate, mesh_edges, parse_example, read_records
from mgn.inject import TYPES, inject, signed_volumes
from mgn.model import MeshGraphNet


def _ld(field, payload):  # length-delimited protobuf field (payloads < 128 bytes)
    return bytes([field << 3 | 2, len(payload)]) + payload


def test_tfrecord_example_roundtrip():
    arr = np.arange(6, dtype=np.float32)
    feature = _ld(1, _ld(1, arr.tobytes()))                       # Feature{bytes_list{value}}
    example = _ld(1, _ld(1, _ld(1, b"mesh_pos") + _ld(2, feature)))  # Example{features{feature{key,value}}}
    record = struct.pack("<Q", len(example)) + b"\0" * 4 + example + b"\0" * 4
    (raw,) = read_records(io.BytesIO(record))
    np.testing.assert_array_equal(np.frombuffer(parse_example(raw)["mesh_pos"], np.float32), arr)


def _toy_traj(rng):
    """Two tets sharing a face, plus two actuator nodes; 60 frames of smooth motion."""
    mesh_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [.2, .2, 1.01], [.3, .3, 1.02]],
                        np.float32) * 0.02
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    if signed_volumes(mesh_pos, cells)[1] < 0:
        cells[1] = [2, 1, 3, 4]
    node_type = np.array([0, 0, 0, 0, 0, 1, 1])
    drift = np.linspace(0, 1, 400)[:, None, None] * rng.normal(size=(1, 7, 3)).astype(np.float32) * 1e-3
    return dict(mesh_pos=mesh_pos, cells=cells, node_type=node_type, world_pos=(mesh_pos + drift).astype(np.float32),
                mesh_edges=mesh_edges(cells))


def test_model_permutation_equivariant():
    rng = np.random.default_rng(0)
    traj = _toy_traj(rng)
    g = build_graph(traj, *traj["world_pos"][9:12])
    perm = rng.permutation(7)
    inv = np.argsort(perm)
    gp = dict(g, x=g["x"][perm], mesh_edges=inv[g["mesh_edges"]], world_edges=inv[g["world_edges"]])
    model = MeshGraphNet(steps=2, hidden=16).eval()
    t = lambda d: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
    out, outp = model(t(g)), model(t(gp))
    assert out.shape == (7, 3)
    torch.testing.assert_close(outp, out[perm], atol=1e-5, rtol=1e-5)
    assert len(collate([g, g])["x"]) == 14


@pytest.mark.parametrize("kind", TYPES)
def test_injectors(kind):
    rng = np.random.default_rng(1)
    traj = _toy_traj(rng)
    wp, mask, frames = inject(traj, kind, 3, rng, t0=50)
    assert mask.any() and frames.any()
    assert not mask[traj["node_type"] != 0].any(), "only plate nodes may be corrupted"
    changed = np.abs(wp - traj["world_pos"]).max(axis=(0, 2)) > 0
    assert not (changed & ~mask).any(), "changes must stay inside the labelled mask"
    if kind == "inversion":
        assert (signed_volumes(wp[50], traj["cells"]) < 0).any()
        assert (signed_volumes(traj["world_pos"][50], traj["cells"]) > 0).all()
    if kind == "frozen":
        np.testing.assert_array_equal(wp[51, mask], traj["world_pos"][50, mask])
