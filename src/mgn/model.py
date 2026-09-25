"""MeshGraphNet (Pfaff et al., ICLR 2021) in plain PyTorch — scatter via index_add_, runs on MPS."""
import torch
from torch import nn


def mlp(din, dout, hidden=128, layer_norm=True):
    layers = [nn.Linear(din, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, dout)]
    return nn.Sequential(*layers, *([nn.LayerNorm(dout)] if layer_norm else []))


class Normalizer(nn.Module):
    """Online feature standardizer; accumulates statistics during the first `max_count` samples of training."""

    def __init__(self, dim, max_count=1_000_000, eps=1e-8):
        super().__init__()
        self.max_count, self.eps = max_count, eps
        self.register_buffer("count", torch.zeros(()))
        self.register_buffer("sum", torch.zeros(dim))
        self.register_buffer("sumsq", torch.zeros(dim))

    def forward(self, x, accumulate=False):
        if accumulate and self.count < self.max_count:
            self.count += len(x)
            self.sum += x.sum(0)
            self.sumsq += (x * x).sum(0)
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean

    @property
    def mean(self):
        return self.sum / self.count.clamp(min=1)

    @property
    def std(self):
        return (self.sumsq / self.count.clamp(min=1) - self.mean ** 2).clamp(min=self.eps).sqrt().clamp(min=self.eps)


def scatter_sum(src, index, n):
    return src.new_zeros(n, src.shape[1]).index_add_(0, index, src)


class Block(nn.Module):
    """One message-passing step over two edge sets (mesh + world/contact), with residual updates."""

    def __init__(self, h):
        super().__init__()
        self.mesh, self.world, self.node = mlp(3 * h, h), mlp(3 * h, h), mlp(3 * h, h)

    def forward(self, v, em, ew, mesh_edges, world_edges):
        (sm, rm), (sw, rw) = mesh_edges, world_edges
        em = em + self.mesh(torch.cat([em, v[sm], v[rm]], 1))
        ew = ew + self.world(torch.cat([ew, v[sw], v[rw]], 1))
        v = v + self.node(torch.cat([v, scatter_sum(em, rm, len(v)), scatter_sum(ew, rw, len(v))], 1))
        return v, em, ew


class MeshGraphNet(nn.Module):
    def __init__(self, node_in=10, mesh_in=8, world_in=4, out=3, hidden=128, steps=10):
        super().__init__()
        self.norm_x, self.norm_m, self.norm_w, self.norm_y = (
            Normalizer(node_in), Normalizer(mesh_in), Normalizer(world_in), Normalizer(out))
        self.enc_x, self.enc_m, self.enc_w = mlp(node_in, hidden), mlp(mesh_in, hidden), mlp(world_in, hidden)
        self.blocks = nn.ModuleList(Block(hidden) for _ in range(steps))
        self.dec = mlp(hidden, out, layer_norm=False)

    def forward(self, g, accumulate=False):
        """Returns the *normalized* per-node acceleration (next_pos - 2 pos + prev_pos)."""
        v = self.enc_x(self.norm_x(g["x"], accumulate))
        em = self.enc_m(self.norm_m(g["mesh_attr"], accumulate))
        ew = self.enc_w(self.norm_w(g["world_attr"], accumulate))
        for b in self.blocks:
            v, em, ew = b(v, em, ew, g["mesh_edges"], g["world_edges"])
        return self.dec(v)

    @torch.no_grad()
    def predict(self, g):
        """Denormalized acceleration per node; next_pos = pos + vel + predict(g)."""
        return self.norm_y.inverse(self(g))
