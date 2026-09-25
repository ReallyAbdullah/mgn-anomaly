# Gate A — pre-registered 2026-09-26 ~07:18, before any validation results were seen
Question: at equal budget (40 train sims, ~9k steps), does sigma=1e-3 (runs/smoke_noise) beat sigma=3e-4 (runs/smoke_noise3e-4)?
Data: validation sims 20-99 (0-9 were training-time val, 0-19 are CLIP reference renders), protocol v2, per-node calibration, CPU.
Statistic: mean frame AUROC (gnn detector) over the 9 cells {hourglass, inversion, frozen} x severity {1,2,3};
  paired bootstrap over sims (500 resamples) of mean(1e-3) - mean(3e-4).
Decision: restart the 8h run with sigma=1e-3 ONLY if that difference is > 0 with the 95% CI excluding 0. Otherwise keep 3e-4.
Caveat: 9k-step pilots; the ranking may not hold at 260k steps.
AMENDED ~07:20 before results (timing only: 94 s/sim/ckpt on throttled CPU): data = validation sims 20-49 (30 sims).
NOTE ~07:24: restarted with 2 threads (training hit 120 ms/step). Both checkpoints use the same injector revision (penetration/inversion edits made before restart; only hourglass/inversion/frozen enter the statistic).
NOTE ~07:28: restarted again — torch.set_num_threads(4) was hard-coded, env vars ignored; now 2 threads.

RESULT 09:02: mean AUROC sigma=1e-3 0.724, sigma=3e-4 0.829; diff -0.105 [-0.129, -0.077], 30 sims. DECISION: KEEP sigma=3e-4 (no restart).
Note: report step of the 1e-3 run crashed on NaN in *_causal scores (0/0 on pre-contact frames); fixed afterwards; frame_gnn (the only input to this statistic) had no NaN.
