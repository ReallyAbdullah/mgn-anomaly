# Gate D — pre-registered 2026-09-26, before any run-level results on the Gate D data were seen

What had been seen before writing this: a pipeline smoke test on 2 validation sims × 3 reference sims, rules only.
The numbers were uninformative, with CIs spanning 0–1.

**Question.** For whole-run errors that are smooth in space and time but inconsistent with the loading, does a
learned surrogate beat the best simple rule?

**Code.** `src/mgn/runlevel.py`, and `src/mgn/inject.py` (`inject_global`, `GLOBAL_TYPES`, `GLOBAL_SEVERITY`), at
the commit that adds this file.

**Data.**
- Validation sims 20–49: one clean run and 9 corrupted runs each.
- Clean reference runs for the z-scores: validation sims 50–69.
- Frames: steady regime [40, 340), every 2nd frame.

**Errors × severities (fixed; no iteration after results):**
- `scale`: ε = 5, 10, 20%
- `lag`: k = 1, 3, 10 frames
- `timescale`: c = 0.95, 0.90, 0.80

**Statistics.** Each is a two-sided robust |z| against the clean reference runs.
- Surrogate family S = {`shape_resid`, `shape_scale`, `gnn`}.
- Rule family R = {`constvel`, `laplacian`, `vlap`, `velocity`, `contact_gap`, `contact_depth`}.

**Shape-model acceptance (checked first).** The shape surrogate (`runs/shape`, 8 h, seed 0) is admitted only if its
final validation RMSE is ≤ 0.5 × the zero-displacement baseline, as printed at the end of `runs/shape_train.log`.
Otherwise the shape statistics are reported descriptively and S = {`gnn`}.

**Test.**
- For each of the 9 cells, the best rule is the member of R with the highest point AUROC on these same data. This
  choice is conservative: it favours the rules.
- For each surrogate statistic s ∈ S, Δ = AUROC(s) − AUROC(best rule), with a paired bootstrap over sims (2,000
  resamples, seed 0). The one-sided p is the fraction of resamples with Δ ≤ 0.
- Holm correction over all 9 × |S| tests, at α = 0.05.

**Pass.** At least one Holm-significant positive Δ. The Phase 2 primary (H1) then uses exactly the passing cells and
statistics, re-tested once on P2 (train sims 250–349). **Fail:** the paper is the negative benchmark.

**Predictions written down in advance.**
- The contact rules will win `scale` at every severity, since the plate lifts off the actuator.
- The contact rules will win `lag` and `timescale` at the highest severity, since the actuator is driven into the
  plate.
- The informative cells are the low severities.
- We expect the velocity-input GNN to be weak here: it receives the corrupted velocity as input.
