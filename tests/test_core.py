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
    """Two plate tets sharing a face, plus a one-tet actuator touching the plate; 400 frames of smooth motion."""
    mesh_pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
                         [0, 0, -.02], [1, 0, -1], [0, 1, -1], [.3, .3, -2]], np.float32) * 0.02
    cells = np.array([[0, 1, 2, 3], [1, 2, 3, 4], [5, 6, 7, 8]])
    for c in range(3):
        if signed_volumes(mesh_pos, cells)[c] < 0:
            cells[c, [0, 1]] = cells[c, [1, 0]]
    node_type = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    drift = np.linspace(0, 1, 400)[:, None, None] * rng.normal(size=(1, 9, 3)).astype(np.float32) * 1e-3
    return dict(mesh_pos=mesh_pos, cells=cells, node_type=node_type, world_pos=(mesh_pos + drift).astype(np.float32),
                mesh_edges=mesh_edges(cells))


def test_model_permutation_equivariant():
    rng = np.random.default_rng(0)
    traj = _toy_traj(rng)
    g = build_graph(traj, *traj["world_pos"][9:12])
    perm = rng.permutation(9)
    inv = np.argsort(perm)
    gp = dict(g, x=g["x"][perm], mesh_edges=inv[g["mesh_edges"]], world_edges=inv[g["world_edges"]])
    model = MeshGraphNet(steps=2, hidden=16).eval()
    t = lambda d: {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
    out, outp = model(t(g)), model(t(gp))
    assert out.shape == (9, 3)
    torch.testing.assert_close(outp, out[perm], atol=1e-5, rtol=1e-5)
    assert len(collate([g, g])["x"]) == 18


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


def test_geometric_detectors():
    from mgn.detect import jacobian_scores, laplacian_scores
    rng = np.random.default_rng(2)
    traj = _toy_traj(rng)
    wp, mask, _ = inject(traj, "inversion", 3, rng, t0=50)
    assert jacobian_scores(traj, wp, [50])[0, mask].min() >= 1
    assert jacobian_scores(traj, traj["world_pos"], [50]).sum() == 0
    shifted = traj["world_pos"] + np.float32([0.1, -0.2, 0.3])  # rigid translation has zero Laplacian residual
    plate = traj["node_type"] == 0  # the toy actuator nodes have no mesh neighbours
    np.testing.assert_allclose(laplacian_scores(traj, shifted, [50])[:, plate],
                               laplacian_scores(traj, traj["world_pos"], [50])[:, plate], atol=1e-5)


def test_choose_frames_protocol():
    from mgn.evaluate import choose_frames
    fmask = np.zeros(400, bool)
    fmask[100:110] = True
    frames, labels, phase = choose_frames(fmask, np.random.default_rng(0))
    np.testing.assert_array_equal(frames[labels], np.arange(100, 110))  # every event frame is scored
    assert phase[0] == "onset" and (phase[1:10] == "sustained").all()
    clean = frames[~labels]
    assert len(clean) == 10 and len(set(clean)) == 10
    assert ((clean < 97) | (clean > 112)).all(), "clean frames must sit outside the +-3 guard band"
    assert ((clean >= 60) & (clean <= 149)).all(), "clean frames must come from the same loading phase"


def test_severity_scaling():
    from mgn.inject import SEVERITY, step_scale
    rng = np.random.default_rng(3)
    traj = _toy_traj(rng)
    a = step_scale(traj)
    for kind in ("hourglass", "instability"):
        peak = {s: np.abs(inject(traj, kind, s, np.random.default_rng(0), t0=50)[0] - traj["world_pos"]).max()
                for s in SEVERITY}
        assert np.isclose(peak[2] / peak[1], SEVERITY[2] / SEVERITY[1], rtol=1e-3)
    wp, mask, _ = inject(traj, "instability", 1, np.random.default_rng(0), t0=50)
    np.testing.assert_allclose(np.linalg.norm(wp[57] - traj["world_pos"][57], axis=1)[mask].max(), a, rtol=1e-3)


def test_no_ci_below_min_sims():
    from mgn.evaluate import cell_rows
    rng = np.random.default_rng(0)
    records = [dict(traj=i, type="frozen", sev=1, label=bool(j % 2), phase="onset" if j % 2 else "clean",
                    frame_gnn=rng.random() + j % 2, frame_constvel=rng.random())
               for i in range(5) for j in range(4)]
    rows = cell_rows(records, ["gnn", "constvel"], "frozen", 1)
    assert all(np.isnan(r["lo"]) for r in rows) and rows[0]["frame_auroc"] > 0.9


def test_leadtime_analysis():
    from mgn.leadtime import DETS, analyse
    frames = list(range(85, 128))  # t0 = 100, cap reached at end = 127
    ramp = [0.0 if f < 115 else 10.0 for f in frames]  # alarms from frame 115 -> lead time 12 frames
    ev = dict(sim=0, kind="blowup", t0=100, end=127, frames=frames, scores={d: ramp for d in DETS})
    fr = dict(sim=0, kind="frozen", t0=100, end=129, frames=frames, scores={d: [0.0] * len(frames) for d in DETS})
    res = analyse([ev, fr], {d: 1.0 for d in DETS})
    assert res["blowup/gnn_causal"]["median"] == 12 and res["blowup/gnn_causal"]["false_alarm_rate"] == 0
    assert res["frozen/gnn_causal"]["detected"] == 0  # never alarmed -> censored, not counted as detected


def test_score_copy_finite_on_static_start():
    """Causal calibration on early frames sees only pre-motion (all-zero) residuals; scores must stay finite."""
    from mgn.evaluate import score_copy
    rng = np.random.default_rng(4)
    traj = _toy_traj(rng)
    traj["world_pos"][:40] = traj["world_pos"][0]  # plate at rest before contact
    model = MeshGraphNet(steps=1, hidden=8).eval()
    fs, _ = score_copy(model, traj, traj["world_pos"], np.array([10, 20, 45]), torch.device("cpu"))
    assert all(np.isfinite(v).all() for v in fs.values())
