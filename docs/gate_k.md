# Gate K (bundle-outlier baseline) — specified 2026-09-27, before any result was seen

**Why.** Crash-simulation data mining detects outlier runs with kNN scores against a bundle of runs
(Kracker, Dhanasekaran, Schumacher & Garcke, IJCrash 2023). Reviewers from that community will expect it as a
baseline.

**Adaptation.** Their bundles share one mesh. Our simulations have different meshes, so each frame is described by a
fixed-length, mesh-independent vector: quantiles (50, 90, 99, max) over plate nodes of
- speed
- acceleration (second difference)
- position-Laplacian residual
- velocity-Laplacian residual

All four are divided by the run's RMS per-step displacement. The vector also includes the inverted-element count and
the maximum contact penetration depth (in step units), making 18 features, log1p-transformed and standardised on the
bank.

The bank is clean steady-regime frames (every 2nd frame in [40, 340)) from validation sims 50–69. The frame score is
the mean distance to the 5 nearest bank frames, and the run score is the mean frame score over the steady regime, as
in Kracker et al.

**Evaluation (validation sims 20–49 only; no test data).**
- Local failures: frame AUROC per type and phase (mean over severities), on the *same* copies and frames as the
  protocol v3 validation records. Compared with the GNN and the best rule on those frames.
- Whole-run errors: run AUROC (clean vs corrupted run) per type and severity, compared with the Gate D run-level
  table.

**Status.** Descriptive and secondary. This does not change Gates C or D. It is reported whether it wins or loses.
