"""Anomaly detectors. Each returns per-node scores [F, N] for the requested frames of a (possibly corrupted) trajectory.

- gnn:       |observed - MeshGraphNet prediction|   (learned physics)
- constvel:  |observed - constant-velocity extrapolation|   (the same predictor with zero learned acceleration)
- velocity:  robust z-score of node speed over the trajectory   (trivial statistics)
- laplacian: |displacement - mean neighbour displacement|   (physics-free spatial smoothness)
- jacobian:  inverted elements (signed tet volume flips sign vs. rest)   (geometric rule)
- vlap:      |velocity - mean neighbour velocity|   (physics-free spatial kinematic rule)
- velocity_causal: speed z-score against the node's own past only   (online trivial statistics)
- contact:   penetration depth of plate nodes into the actuator surface   (contact rule)
- clip_zero / clip_knn:  image-space detectors on renders (see ClipDetector)
"""
import numpy as np
import open_clip
import torch

from mgn.data import NORMAL, T_MAX, build_graph, collate
from mgn.inject import actuator_distance, adjacency, signed_volumes
from mgn.train import to_torch


def constvel_scores(wp, frames):
    f = np.asarray(frames)
    return np.linalg.norm(wp[f] - (2 * wp[f - 1] - wp[f - 2]), axis=-1)


@torch.no_grad()
def gnn_scores(model, traj, wp, frames, dev, batch=8):
    out = []
    for i in range(0, len(frames), batch):
        fs = frames[i:i + batch]
        g = collate([build_graph(traj, wp[t - 2], wp[t - 1], wp[t]) for t in fs], pad=True)
        acc = model.predict(to_torch(g, dev)).cpu().numpy()
        n = wp.shape[1]
        for j, t in enumerate(fs):
            pred = 2 * wp[t - 1] - wp[t - 2] + acc[j * n:(j + 1) * n]
            out.append(np.linalg.norm(wp[t] - pred, axis=-1))
    return np.stack(out)


def velocity_scores(wp, frames):
    speed = np.linalg.norm(np.diff(wp[:T_MAX], axis=0), axis=-1)  # speed[t-1] = |wp[t] - wp[t-1]|
    med = np.median(speed, axis=0)
    mad = np.median(np.abs(speed - med), axis=0) * 1.4826 + 1e-9
    return np.abs(speed[np.asarray(frames) - 1] - med) / mad


def laplacian_scores(traj, wp, frames):
    adj = adjacency(traj).astype(np.float32)
    deg = np.asarray(adj.sum(1)).clip(1)
    u = wp[np.asarray(frames)] - traj["mesh_pos"]
    return np.stack([np.linalg.norm(x - (adj @ x) / deg, axis=-1) for x in u])


def velocity_laplacian_scores(traj, wp, frames):
    adj = adjacency(traj).astype(np.float32)
    deg = np.asarray(adj.sum(1)).clip(1)
    f = np.asarray(frames)
    return np.stack([np.linalg.norm(v - (adj @ v) / deg, axis=-1) for v in wp[f] - wp[f - 1]])


def velocity_causal_scores(wp, frames, min_past=5):
    """Robust speed z-score using only frames at least 4 before the scored one (min. the first `min_past`)."""
    speed = np.linalg.norm(np.diff(wp[:T_MAX], axis=0), axis=-1)  # speed[t-1] = |wp[t] - wp[t-1]|
    out = []
    for t in frames:
        past = speed[:max(t - 4, min_past)]
        med = np.median(past, axis=0)
        mad = np.median(np.abs(past - med), axis=0) * 1.4826 + 1e-9
        out.append(np.abs(speed[t - 1] - med) / mad)
    return np.stack(out)


def contact_scores(traj, wp, frames):
    """Per plate node: penetration depth into the actuator surface (0 outside)."""
    plate = np.flatnonzero(traj["node_type"] == NORMAL)
    out = np.zeros((len(frames), len(traj["node_type"])))
    for j, t in enumerate(frames):
        out[j, plate] = np.maximum(0.0, -actuator_distance(traj, wp[t], plate))
    return out


def jacobian_scores(traj, wp, frames):
    """Per node: number of incident tets whose signed volume has flipped relative to the rest configuration."""
    cells = traj["cells"]
    rest = np.sign(signed_volumes(traj["mesh_pos"], cells))
    out = np.zeros((len(frames), len(traj["node_type"])))
    for j, t in enumerate(frames):
        bad = cells[np.sign(signed_volumes(wp[t], cells)) != rest]
        np.add.at(out[j], bad.ravel(), 1)
    return out


def frame_score(node_scores, normal, k=5):
    """Top-k mean over plate nodes."""
    s = np.sort(node_scores[:, normal], axis=1)
    return s[:, -k:].mean(1)


class ClipDetector:
    """(a) zero-shot WinCLIP-style prompt ensemble (image level);
    (b) few-shot patch kNN against a memory bank of clean reference renders (WinCLIP+/PatchCore idea)."""

    STATES = (["flawless {}", "perfect {}", "unblemished {}", "{} without flaw", "{} without defect"],
              ["damaged {}", "{} with flaw", "{} with defect", "broken {}", "{} with a damaged region"])
    TEMPLATES = ["a render of a {}.", "a 3D rendering of a {}.", "a cropped photo of the {}.", "a photo of a {}."]

    def __init__(self, dev, arch="ViT-B-16", pretrained="openai"):
        self.dev = dev
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=dev)
        self.model.eval()
        self.model.visual.output_tokens = True
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            text = []
            for states in self.STATES:
                prompts = [t.format(s.format("bent metal plate")) for s in states for t in self.TEMPLATES]
                e = self.model.encode_text(tok(prompts).to(dev))
                e = e / e.norm(dim=-1, keepdim=True)
                text.append(e.mean(0) / e.mean(0).norm())
            self.text = torch.stack(text)  # [2, D]: normal, anomalous
        self.bank = None

    @torch.no_grad()
    def _encode(self, images):
        from PIL import Image
        x = torch.stack([self.preprocess(Image.fromarray(im)) for im in images]).to(self.dev)
        pooled, tokens = self.model.visual(x)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        tokens = tokens / tokens.norm(dim=-1, keepdim=True)
        return pooled, tokens  # [B, D], [B, P, W]

    def fit(self, clean_images, batch=32):
        self.bank = torch.cat([self._encode(clean_images[i:i + batch])[1].flatten(0, 1)
                               for i in range(0, len(clean_images), batch)])

    @torch.no_grad()
    def score(self, images, batch=32):
        """Returns zero-shot p(anomalous) [B] and kNN patch-distance maps [B, g, g]."""
        zs, maps = [], []
        for i in range(0, len(images), batch):
            pooled, tokens = self._encode(images[i:i + batch])
            zs.append((100 * pooled @ self.text.T).softmax(-1)[:, 1])
            d = torch.stack([1 - (t @ self.bank.T).max(1).values for t in tokens])
            g = int(d.shape[1] ** 0.5)
            maps.append(d.view(-1, g, g))
        return torch.cat(zs).cpu().numpy(), torch.cat(maps).cpu().numpy()

    @staticmethod
    def node_scores(patch_map, xy, size):
        """Look up each node's patch score from its projected pixel position."""
        g = patch_map.shape[0]
        ij = np.clip((xy / size * g).astype(int), 0, g - 1)
        return patch_map[ij[:, 1], ij[:, 0]]
