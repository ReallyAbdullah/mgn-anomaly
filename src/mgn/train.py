"""Train the MeshGraphNet one-step surrogate. Resumable; logs to runs/<name>/log.csv."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

from mgn.data import DATA, NORMAL, T_MAX, build_graph, collate, load_traj
from mgn.model import MeshGraphNet

RUNS = Path(__file__).resolve().parents[2] / "runs"


def device(name=None):
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))


def load_split(split, n=None):
    paths = sorted((DATA / split).glob("traj_*.npz"))[:n]
    return [load_traj(p) for p in paths]


def to_torch(g, dev):
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(dev) for k, v in g.items()}


def sample(trajs, idx, rng, noise, pad=False):
    """Batch of (graph, target acceleration, normal-node mask) for (traj, t) pairs, t in [1, T_MAX)."""
    graphs, ys, masks = [], [], []
    for i, t in idx:
        tr = trajs[i]
        normal = tr["node_type"] == NORMAL
        prev, pos, nxt = tr["world_pos"][t - 1:t + 2]
        if noise:
            pos = pos + rng.normal(0, noise, pos.shape).astype(np.float32) * normal[:, None]
        graphs.append(build_graph(tr, prev, pos, nxt))
        ys.append(nxt - 2 * pos + prev)
        masks.append(normal)
    g, y, m = collate(graphs, pad), np.concatenate(ys), np.concatenate(masks)
    extra = len(g["x"]) - len(y)
    return g, np.pad(y, ((0, extra), (0, 0))), np.pad(m, (0, extra))


def load_model(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    model = MeshGraphNet(**ck["cfg"]).to(dev)
    model.load_state_dict(ck["model"])
    return model.eval()


@torch.no_grad()
def evaluate(model, val, dev):
    """One-step next-position RMSE on NORMAL nodes vs. the constant-velocity baseline (= zero acceleration),
    on the same fixed frames at every call."""
    model.eval()
    rng = np.random.default_rng(0)
    idx = [(i, t) for i in range(len(val)) for t in 1 + rng.choice(T_MAX - 1, 20, replace=False)]
    se = base = n = 0.0
    for j in range(0, len(idx), 8):
        g, y, m = sample(val, idx[j:j + 8], None, 0, pad=True)
        pred = model.predict(to_torch(g, dev)).cpu().numpy()
        se += ((pred - y)[m] ** 2).sum()
        base += (y[m] ** 2).sum()
        n += m.sum()
    model.train()
    return np.sqrt(se / n), np.sqrt(base / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="mgn")
    p.add_argument("--n-train", type=int, default=250)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--noise", type=float, default=3e-4, help="teaches the surrogate to pull perturbed states back to equilibrium")
    p.add_argument("--iters", type=int, default=1_000_000)
    p.add_argument("--hours", type=float, default=float("inf"), help="wall-clock budget")
    p.add_argument("--warmup", type=int, default=500, help="steps that only accumulate normalizer stats")
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--device")
    p.add_argument("--mode", default="velocity", choices=["velocity", "shape"],
                   help="velocity: one-step acceleration model (Phase 1); shape: displacement from actuator position")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)  # Phase 1 runs predate this: their weight init was unseeded
    if a.mode == "shape":
        from mgn import shape

    dev = device(a.device)
    out = RUNS / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(hidden=a.hidden, steps=a.steps)
    if a.mode == "shape":
        cfg.update(node_in=shape.NODE_IN, mesh_in=shape.MESH_IN, world_in=shape.WORLD_IN)
    model = MeshGraphNet(**cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, fused=dev.type != "cpu")
    # decay lr by 100x over the run (as in MGN: 1e-4 -> 1e-6)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.01 ** min(s / a.iters, 1))
    step = 0
    if (out / "model.pt").exists():
        ck = torch.load(out / "model.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}")

    train, val = load_split("train", a.n_train), load_split("valid", a.n_val)
    print(f"{len(train)} train / {len(val)} val trajectories on {dev}; {sum(p.numel() for p in model.parameters()):,} params")
    rng = np.random.default_rng((a.seed, step))
    log = open(out / "log.csv", "a")
    t0, last, losses = time.time(), time.time(), []

    def save():
        meta = dict(mode=a.mode, noise=a.noise if a.mode == "velocity" else 0.0, n_train=a.n_train, batch=a.batch,
                    lr=a.lr, iters=a.iters, seed=a.seed, steps=a.steps)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, cfg=cfg,
                        meta=meta), out / "model.pt.tmp")
        (out / "model.pt.tmp").replace(out / "model.pt")  # atomic: readers never see a half-written checkpoint

    while step < a.iters and time.time() - t0 < a.hours * 3600:
        idx = [(rng.integers(len(train)), rng.integers(1, T_MAX)) for _ in range(a.batch)]
        warm = step < a.warmup  # normalizer stats must not see padding, so warmup batches are unpadded
        g, y, m = (shape.sample(train, idx, pad=not warm) if a.mode == "shape"
                   else sample(train, idx, rng, a.noise, pad=not warm))
        g, y, m = to_torch(g, dev), torch.from_numpy(y).to(dev), torch.from_numpy(m).to(dev)
        pred = model(g, accumulate=warm)
        target = model.norm_y(y[m], accumulate=warm)
        loss = ((pred[m] - target) ** 2).mean()
        if not warm:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        losses.append(loss.item())
        step += 1
        if step % 100 == 0:
            dt = (time.time() - last) / 100
            last = time.time()
            print(f"step {step} loss {np.mean(losses):.4f} {dt * 1000:.0f} ms/step lr {sched.get_last_lr()[0]:.2e}", flush=True)
            losses = []
        if step % a.eval_every == 0:
            rmse, base = (shape.evaluate if a.mode == "shape" else evaluate)(model, val, dev)
            print(f"  val {a.mode} RMSE {rmse:.2e} ({'zero-displacement' if a.mode == 'shape' else 'const-velocity'} "
                  f"baseline {base:.2e})", flush=True)
            log.write(f"{step},{time.time() - t0:.0f},{rmse:.6e},{base:.6e}\n")
            log.flush()
            save()
    save()
    rmse, base = (shape.evaluate if a.mode == "shape" else evaluate)(model, val, dev)
    print(f"final val {a.mode} RMSE {rmse:.2e} (baseline {base:.2e})")


if __name__ == "__main__":
    main()
