# Gate C — pre-registered 2026-09-26, before any test simulation (8–99) was scored

**Question.** Does the learned surrogate (MeshGraphNet, σ=3e-4, the final 8 h checkpoint) detect injected failures
better than the best simple baseline?

**Code.** `src/mgn/gate_c.py`, committed together with this file. The analysis runs exactly as written there.

**Data.**
- Best baselines are chosen on validation sims 20–49 (protocol v2, no CLIP).
- The statistic is computed on test sims 8–99 (protocol v2, with CLIP), which are scored once.

**Primary hypothesis.** Learned physics helps with anomalies that persist, where the corrupted state stays physically
inconsistent after onset.

**Primary analysis.**
- Family: sustained-phase frame AUROC on {hourglass, inversion, frozen} × severity {1, 2, 3}.
- Comparison, like with like:
  - `gnn` against the best non-causal baseline among {constvel, laplacian, velocity, jacobian}
  - `gnn_causal` against the best causal or memoryless baseline among {constvel_causal, laplacian_causal, jacobian}
- The best baseline is chosen per cell on validation, using sustained-phase AUROC.
- CLIP is not in the pool, because it isn't run on validation. It is reported on test for context only.
- Statistic: the mean over the 9 cells of AUROC(GNN variant) − AUROC(best baseline), with a paired bootstrap over test
  sims (2,000 resamples, seed 0).
- Decision: claim "learned physics helps" only if the 95% CI lower bound is above 0. Otherwise report "no demonstrated
  benefit" and frame the work as a benchmark plus a negative result. In that case, no full-scale extra seeds.

**Secondary analysis (exploratory).** All 15 type × severity cells, all event frames, one-sided paired-bootstrap
p-values, Holm-corrected at α = 0.05.

**Not allowed after seeing test results:** changing injectors, frame selection, calibration, thresholds, the baseline
pool or the cell family.
