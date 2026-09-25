# Learned physics vs. vision vs. rules for anomaly detection in FE simulations

**When does a learned physics surrogate catch simulation failures that simple rules and generic vision models miss,
and can a VLM explain the flagged frames correctly?**

Contributions: (i) an injected-failure benchmark on FE meshes, with ground-truth type and location for every
anomaly; (ii) a quantified comparison of three detector families: a learned surrogate, physics-free rules
(constant-velocity extrapolation, Laplacian smoothness, inverted-element check, velocity z-score) and CLIP;
(iii) a correctness study of VLM explanations against a diagnostics-only classifier. Residual-based fault detection
with a surrogate is a classical digital-twin idea. What is new here is the controlled comparison and the explanation
study, not the detector itself.

A MeshGraphNet ([Pfaff et al., ICLR 2021](https://arxiv.org/abs/2010.03409)) is trained on DeepMind's
`deforming_plate` dataset (an actuator pressing into a hyperelastic plate; COMSOL, tetrahedral FE meshes). This is the
a public structural-mechanics benchmark. It is quasi-static and hyperelastic, which makes it a stand-in for crash
data rather than the real thing. Synthetic failures modelled on real FE failure modes are injected
into held-out simulations. Three kinds of detector are compared:

- the surrogate's one-step prediction error
- physics-free baselines
- CLIP-based image anomaly detection

A VLM then explains the flagged frames. Its answers are scored against the injected ground truth for type and exact
location, and compared with a decision tree trained on the numeric diagnostics alone. This measures correctness, not
full explanation faithfulness: free-text claims are not yet checked.

Runs end to end on a MacBook (M4 Pro, 24 GB): plain PyTorch on MPS, MLX for the VLM, no TensorFlow.

## Pipeline

```
TFRecords --stream--> .npz ──> MeshGraphNet (one-step surrogate, trained overnight on MPS)
                                   │
held-out sims ──inject 5 failure modes x 3 severities──> corrupted sims
                                   │
            ┌──────────────┬───────┴────────┬─────────────────────┐
  GNN residual  const-velocity  Laplacian  inverted-tet  velocity z  CLIP zero-shot / patch-kNN
            └──────────────┴───────┬────────┴─────────────────────┘
  frame/node AUROC, TPR@5%FPR, paired bootstrap of GNN - baseline
                                   │
     flagged frames ──render + residual heatmap (+ mesh diagnostics)──> Qwen3-VL (JSON) ──> vs. diagnostics-only tree
```

## Injected failure modes

| Injected anomaly | Crash-simulation failure it mimics | How |
|---|---|---|
| hourglass | hourglassing / zero-energy modes in under-integrated elements | checkerboard offsets (alternating by hop parity) on a 2-hop patch |
| penetration | contact failure | plate nodes touching the actuator (<1 mm) moved to the closest point on its surface, then pushed inward by the severity depth (checked: 100/100/92% of them end up inside the actuator) |
| inversion | element inversion (negative Jacobian) | one tet vertex moved towards its opposite face (severity 1 distorts without flipping; 2 and 3 invert) |
| instability | numerical blow-up | local sign-alternating oscillation growing 1.6x per frame for 8 frames, peaking at the severity amplitude |
| frozen | constraint / boundary-condition error | a region holds its position while the plate keeps deforming |

Severities are 1x, 10x and 100x the simulation's RMS per-step node displacement (≈0.3 mm). Inversion is geometric
(0.3x, 0.6x, 1.5x the reflection distance), and for frozen the severity is the duration (3, 10, 30 frames).
Penetration events start only once the actuator is in contact with the plate.

## Evaluation protocol (frozen before the final test run)

Everything below is fixed on validation simulations before test simulations 8–99 are scored, once:

- **Data.** Validation sims 0–19 serve as CLIP reference renders and training-time validation. Sims 20–99 are used
  for tuning and gates. Test sims 0–7 were used in pilot runs and are excluded, leaving 92 test sims.
- **Frames (protocol v2).** Every anomalous frame is scored and labelled `onset` (first frame) or `sustained`. Clean
  frames are drawn from the same copy within ±40 frames of the event, outside a ±3-frame guard band, so clean and
  anomalous frames come from the same loading phase.
- **Statistics.** 95% bootstrap CIs over simulations, and none are reported with fewer than 20 sims. Paired
  bootstrap for GNN-minus-baseline differences. TPR at 5% FPR. AUROC is reported per event phase.
- **Gates.** A: σ is chosen on validation (pre-registered criterion). B: the surrogate's one-step error vs
  constant velocity. C: the GNN must beat the best baseline (chosen per cell on validation) with a paired CI that
  excludes 0 before any claim of benefit.
- **Provenance.** Every records file stores the checkpoint SHA-256, the simulation IDs, the protocol version and the
  git commit.

## Results

RESULTS_PLACEHOLDER

## Design notes (what mattered)

- **MGN in plain PyTorch.** Aggregation uses `index_add_`, with two edge sets (mesh plus actuator↔plate contact edges
  within 3 cm), 10 message-passing steps and latent size 128, for 2.6M parameters. PyG's scatter extensions have no MPS
  kernels, and the model is about 80 lines anyway.
- **Velocity in, acceleration out.** The first version followed the paper's deforming-plate setup (positions in,
  displacement out, noise σ=3e-3). It failed to beat a zero-motion baseline within a laptop budget, because plate
  motion is tiny (~3e-4 per step) relative to edge lengths (~2e-2), so it only shows up as ~1e-4 strain signals.
  Switching to the MGN-cloth formulation fixed this: current velocity as a node input, acceleration as the target.
  A constant-velocity extrapolation then becomes the model's zero-output baseline, and it is reported as a detector
  in its own right. This makes the comparison honest: the GNN has to beat that baseline, not just a straw man.
- **Training noise is what makes the surrogate a detector.** A surrogate trained only on clean data learns "keep
  going at the same velocity", which is just the constant-velocity baseline. It flags the *onset* of a persistent
  anomaly, but not the frames after it, because the corrupted motion is smooth again by then. When noise is added to
  the input positions and the target keeps the clean next state, the model learns to pull perturbed states back to
  equilibrium. Inverted elements, frozen regions and hourglass patterns then leave a residual in *every* frame, since
  the observed simulation never relaxes back.

  The noise level sets a trade-off between that sensitivity and the error floor on clean data. For reference, the
  constant-velocity extrapolation's own one-step RMSE on clean frames is about 3e-6, so a noise-trained surrogate is
  15–70× *worse* than "do nothing" on clean data. It is a denoiser, not a better simulator.

  Pilot study (frame AUROC, GNN vs constant velocity, 8 test simulations that are excluded from the final
  evaluation). **This table is confounded and will be regenerated:** the σ=0 row is a 250-simulation, 10k-step run,
  while the others are 40-simulation, ~9k-step runs, and only the σ=3e-4 row used per-node calibration.

  | training noise σ | clean one-step RMSE | hourglass (100x) | inversion (100x) | frozen (10 frames) |
  |---|---|---|---|---|
  | 0 (clean only; 250 sims) | ≈ const-velocity | 0.73 vs 0.78 | 0.65 vs 0.69 | 0.64 vs 0.69 |
  | 3e-4 (chosen)    | 9.4e-5 (const-vel: 3e-6) | **0.91** vs 0.75 | **0.79** vs 0.71 | **0.80** vs 0.63 |
  | 1e-3             | 2.2e-4 | 0.85 vs 0.78 | 0.76 vs 0.69 | 0.71 vs 0.69 |

  Choosing σ on test simulations was a mistake, so it was re-checked on validation simulations (Gate A,
  pre-registered in [docs/gate_a.md](docs/gate_a.md)). At equal budget (40 train simulations, ~9k steps), σ=3e-4
  beats σ=1e-3 on mean GNN frame AUROC over hourglass, inversion and frozen: 0.83 vs 0.72, paired difference −0.105,
  95% CI [−0.129, −0.077], 30 validation simulations. So σ=3e-4 stays.
- **Steady-actuator regime only.** In every trajectory the actuator speeds up about 9x around frame 360. We train and
  score on frames under 350.
- **MPS shape bucketing.** Batches are padded to fixed node and edge buckets. Without this, every new graph size
  compiles a new Metal kernel graph, and step time climbed from 110 to 330 ms within minutes.
- **No TensorFlow.** TFRecord framing and `tf.train.Example` are parsed by hand in about 30 lines, and the data is
  streamed, so only the first 250 of the 1,000 training simulations are downloaded.
- **Evaluation protocol.** Each corrupted copy contributes up to 4 anomalous frames and 4 clean frames, taken from
  outside a ±3-frame guard band. Residual detectors are calibrated per node, without supervision, by their median
  over every 10th frame of the same simulation. That calibration uses future frames, which is fine for
  benchmarking but not valid for early-warning claims. CIs are 95% bootstrap intervals resampled over simulations,
  and GNN-vs-baseline differences use a paired bootstrap. The render column `px` gives the median visible shift of
  each anomaly in pixels, so "CLIP misses it" can be told apart from "it is sub-pixel".
- **Node AUROC below 0.5** (e.g. constant velocity on `frozen`) is not a bug. Frozen nodes have *lower* residuals
  than moving ones, and scores are one-sided.

## Reproduce

```bash
uv sync
uv run python -m mgn.data valid -n 100 && uv run python -m mgn.data test -n 100 && uv run python -m mgn.data train -n 250
uv run python -m mgn.train --name mgn --hours 8 --iters 260000      # ~110 ms/step on M4 Pro
uv run python -m mgn.evaluate --n-traj 100                          # detectors + AUROC table + figure
uv run python -m mgn.explain                                        # VLM explanations + correctness
uv run pytest
```

## Limitations and next steps

- Injected anomalies are additive, and the trajectory snaps back to the clean one afterwards, so it is not a
  self-consistent physical state. `frozen` at severity 1 (3 frames) is close to imperceptible.
- Single training seed, one dataset, one VLM (4-bit). The training-noise ablation is the only model ablation.
- `deforming_plate` is quasi-static and hyperelastic. Real crash simulations are explicit-dynamics, with plasticity,
  failure and self-contact. The failure modes here are injected, not produced by a solver.
- A laptop-scale surrogate (about 8 h, 250 simulations) is far from paper scale (8×H100, 20 h). Its one-step error
  sets the detection floor for subtle anomalies.
- Next steps: detection lead time with causal calibration and a growing instability; matched counterfactual
  pairs (clean high-stress frame vs. injected hourglass) for explanation faithfulness; checking claims in the
  free-text explanations; a larger or unquantised VLM; longer training and multi-scale or transformer processors (e.g. MGN-Transformer) for global
  consistency, solver-generated failures (e.g. reduced-integration runs that really hourglass), and a VLM fine-tuned
  on the diagnostics.
