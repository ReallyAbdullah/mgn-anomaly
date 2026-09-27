#set document(title: "Do learned surrogates, vision models and VLMs catch simulation failures?", author: "Muhammad Abdullah")
#set page(paper: "a4", margin: (x: 2.1cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => block(above: 1.2em, below: 0.7em, text(size: 12pt, weight: "bold", it))
#show heading.where(level: 2): it => block(above: 1em, below: 0.5em, text(size: 10.5pt, weight: "bold", it))
#show figure.caption: set text(size: 8.8pt)
#show table: set text(size: 8.6pt)
#let pending(body) = text(fill: rgb("#b00020"), [*[pending: #body]*])

#align(center)[
  #text(size: 15pt, weight: "bold")[Do learned surrogates, vision models and VLMs catch simulation failures?]\
  #v(2pt)
  #text(size: 11pt)[A pre-registered benchmark on finite-element trajectories]\
  #v(8pt)
  Muhammad Abdullah\
  #text(size: 9pt)[Independent researcher, Melbourne, Australia · abdullahfast95\@gmail.com]\
  #text(size: 9pt)[Technical report, September 2026 · code and results: github.com/ReallyAbdullah/mgn-anomaly]
]
#v(6pt)

#block(inset: (x: 1.2cm))[
  #text(size: 9.3pt)[
  *Abstract.* Automated checks of simulation output are proposed at every level, from learned surrogates that flag
  frames their prediction disagrees with, to vision and vision-language models (VLMs) that read renders and explain
  them. We test whether they work, and against what baseline. We inject synthetic failures modelled on real
  crash-simulation failure modes into DeepMind's `deforming_plate` finite-element trajectories: hourglassing, contact
  penetration, element inversion, numerical instability, frozen regions, and whole-run lag, time-scale and response-scale
  errors. Each has a known type, location and severity. We compare a noise-trained MeshGraphNet residual detector, a
  shape-from-load surrogate, one-line physics rules, CLIP-family, SigLIP2 and DINOv2 vision models, and Qwen-family
  VLM explanations. Every analysis was pre-registered in version control before it ran, and the test set was scored
  once. *Some one-line rule matches or beats the learned surrogate for every failure type* (92 test simulations). The
  surrogate's only apparent win, frozen regions, is matched by a velocity-Laplacian rule. Whole-run loading errors are
  caught by contact rules or by nothing: a 1–3 frame actuator-plate lag is undetectable by every method tested. No
  vision model reaches AUROC 0.80, and higher resolution does not help. A local 8B VLM labels 199 of 200 frames
  "no anomaly", and counterfactual pairs show its explanations are not grounded in the image. A 27B model from the same
family localizes the flagged region but still misnames the failure (2 of 20 correct). We release the benchmark
  and argue that learned detectors of simulation failures must be reported against their zero-output baseline and
  cheap physics rules.]
]

= Introduction

A finite-element (FE) crash simulation can fail silently. Under-integrated elements develop zero-energy
_hourglass_ modes, contact algorithms let nodes _penetrate_ surfaces, elements _invert_, time integration becomes
_unstable_, and constraints can fail and _freeze_ part of the model. Engineers catch these with energy-ratio checks
and by inspecting animations. As simulation volumes grow, and as simulation datasets are assembled to train ML
surrogates, there is pressure to automate this QA with learned models. Three families are natural candidates:
- a *learned surrogate* whose prediction residual flags implausible states
- *vision models* on rendered frames
- *vision-language models* that could also explain what went wrong

Whether these methods beat simple baselines is rarely tested. The critique of weak baselines in ML for PDEs
@mcgreivy applies here directly: a surrogate-residual detector has a trivial _zero-output_ counterpart, constant-velocity
extrapolation, and there are one-line spatial rules that any engineer could write. We ask:

+ Do learned surrogates detect FE failures better than their zero-output baseline and cheap physics rules?
+ Can generic and modern vision backbones detect the same failures from renders?
+ Do VLM explanations of flagged frames name the right failure, and are they grounded in the image?

*Contributions.*
- (i) An injected-failure benchmark on public FE trajectories, with ground-truth type, location and severity, and an
  onset/sustained split.
- (ii) A controlled comparison of surrogate, rule, vision and VLM detectors in one protocol.
- (iii) A pre-registered, gate-by-gate study design: every decision rule was committed to git before its data were
  seen, and the test set was scored once.
- (iv) A counterfactual grounding test for VLM explanations on engineering renders.

Most of our findings are negative. We report them as such.

= Related work

*ML checks of FE and crash results.* Plausibility checking of structural FE results with deep learning
@spruegel @bickel learns plausible-vs-implausible from existing simulations, focusing on static modelling errors.
Crash-simulation data mining at Fraunhofer SCAI and automotive partners finds outliers and bifurcations across bundles
of runs @kracker. These address behavioural outliers rather than numerical failure modes. To our knowledge, no prior
benchmark has ground-truth numerical failures on FE trajectories or compares learned detectors against physics rules.

*Simple baselines.* Weak baselines inflate claims in ML for PDEs @mcgreivy. In time-series anomaly detection, simple
detectors match deep ones @sarfraz, and many benchmarks are solvable "with one line of code" @wukeogh. We apply the
same standard to surrogate-residual detection.

*Vision and VLMs on engineering output.* WinCLIP @winclip and PatchCore @patchcore are standard zero- and few-shot
visual anomaly detectors, building on CLIP @clip and self-supervised backbones such as DINOv2 @dinov2. Multimodal LLMs
fall short of industrial anomaly-detection requirements @mmad and answer simulation questions poorly from images
@ezemba. LLM explanations of crash results have been proposed without checking faithfulness @mathieu. We test grounding
directly with counterfactual pairs.

= Benchmark

*Data.* We use DeepMind's MeshGraphNets `deforming_plate` @pfaff: a quasi-static, hyperelastic plate (COMSOL,
tetrahedral mesh, ≈1.3k nodes) pressed by a moving actuator, with 400 frames per trajectory. The actuator accelerates
about 9× near frame 360, so we model and score frames below 350. Splits are by simulation:
- train 0–249
- validation 20–49 for selection, 50–69 for fitting and reference, 70–99 for calibration
- test 8–99, scored once (0–7 were used in pilots)
- a second unseen set, training trajectories 250–349, reserved for replication

*Injected failures.* Only plate nodes are changed; the actuator and clamp keep their kinematics. See
@tab:inject. Severities are 1×, 10× and 100× the simulation's RMS per-step node displacement (≈0.3 mm). Inversion is
geometric: severity 1 distorts the element without flipping it. For frozen regions the severity is the duration. The
whole-run errors change the plate response over the entire run.

#figure(
  table(columns: (auto, auto, 1fr), inset: 4pt, stroke: 0.4pt + luma(180),
    table.header([*Failure*], [*Mimics*], [*Injection*]),
    [hourglass], [zero-energy modes], [checkerboard offsets on a 2-hop patch, 5 frames],
    [penetration], [contact failure], [plate nodes in contact pushed through the actuator surface by the severity depth],
    [inversion], [negative Jacobian], [one tet vertex moved towards its opposite face],
    [instability], [numerical blow-up], [sign-alternating oscillation growing 1.6× per frame],
    [frozen], [constraint/BC error], [a region holds its position for 3, 10 or 30 frames],
    [scale / lag / timescale], [output or sync errors], [whole-run: plate displacement ×(1+ε), k frames late, or at c× rate],
  ),
  caption: [Injected failure modes. Frame labels come from the corrupted positions: a frame is anomalous if any node
  deviates. The first frame of an event is _onset_ and the rest are _sustained_.],
) <tab:inject>

*Protocol.* For every simulation × type × severity there is one corrupted copy. All anomalous frames are scored, along
with an equal number of clean frames from the same copy, drawn within ±40 frames of the event but outside a ±3-frame
guard band. Metrics are frame AUROC and node AUROC. 95% CIs are bootstrapped over simulations, and none are reported
with fewer than 20 simulations. GNN-vs-baseline differences use a paired bootstrap with Holm correction.

= Detectors

*Learned surrogate.*
- A MeshGraphNet @pfaff written from scratch in PyTorch on Apple MPS: 10 message-passing steps, latent size 128,
  2.6M parameters, with mesh and actuator-contact edge sets.
- It follows the cloth formulation: current velocity in, next acceleration out. Constant-velocity extrapolation is
  therefore exactly its zero-output baseline.
- Trained 8 h on 250 trajectories with input noise σ = 3×10⁻⁴. The noise teaches it to pull perturbed states back
  towards equilibrium, and was chosen on validation (Gate A).
- A second seed reproduces the clean one-step error: 4.73 vs 4.85 ×10⁻⁵.
- The detection score is the residual |observed − predicted|, normalised per node.

A second, *shape-from-load* surrogate predicts total plate displacement from the actuator position alone, so a
whole-run error can't reach it through its inputs.

*Physics rules* (no learning):
- constant-velocity second difference
- position Laplacian (displacement minus neighbour mean)
- velocity Laplacian
- velocity z-score
- inverted-element (Jacobian) count
- contact penetration depth and gap

Causal variants use only past frames.

*Vision.*
- WinCLIP-style zero-shot and patch-kNN @winclip @patchcore on renders with a fixed camera per simulation.
- Backbones: CLIP ViT-B/16 @clip, OpenCLIP ViT-L/14, SigLIP2 @siglip2 and DINOv2-L @dinov2.
- Memory banks are built from clean validation renders.

*VLMs.*
- Qwen3-VL-8B (4-bit, local, MLX) @qwen3vl, and a pilot with Qwen3.8-27B via OpenRouter.
- Input: the render plus a panel coloured by the surrogate residual, optionally with numeric diagnostics.
- Output: JSON with type, 3×3 grid cell and explanation.

= Pre-registered gates

Every gate's question, data, statistic and decision rule was committed before its data were scored
(`docs/gate_*.md`). See @tab:gates.

#figure(
  table(columns: (auto, 1fr, auto), inset: 4pt, stroke: 0.4pt + luma(180),
    table.header([*Gate*], [*Question*], [*Outcome*]),
    [A], [Training noise σ = 10⁻³ vs 3×10⁻⁴ (validation)], [keep 3×10⁻⁴ (Δ −0.105 [−0.129, −0.077])],
    [B], [Surrogate one-step error vs constant velocity], [6.9× _worse_ on clean frames],
    [C], [GNN vs best baseline, sustained frames, 9 cells (test)], [non-causal: no benefit; causal: +0.023, from frozen only],
    [D], [Surrogates vs rules on whole-run errors (validation)], [fail],
    [V], [Modern vision backbones × resolution (validation)], [fail (best 0.65)],
    [E], [Larger same-family VLM (pilot)], [do not scale up (27B: 2/20 correct)],
    [E-video], [8-frame clip vs single frame, local 8B], [no help (0/20 in both arms)],
    [K], [Bundle-outlier baseline after @kracker (descriptive)], [most robust single detector (0.82–0.98)],
  ),
  caption: [Gate timeline. Full pre-registrations and results are in the repository.],
) <tab:gates>

= Results

== The surrogate is a denoiser, not a better simulator

On fixed validation frames, the surrogate's one-step RMSE is 4.85×10⁻⁵, against 7.0×10⁻⁶ for constant velocity
(Gate B). It is 6.9× _worse_ than doing nothing on clean data. Its detection signal comes from denoising: a corrupted
state that does not relax back leaves a residual in every frame. A clean-trained model (σ = 0) was indistinguishable
from constant velocity in pilots.

== Local failures: a rule always matches or wins

@tab:phase gives test-set frame AUROC by event phase. Constant velocity is near-perfect at the onset of every failure,
because any injected jump is a second-difference spike. It collapses on sustained frames (0.49–0.66), since the
corrupted motion is smooth again. The surrogate recovers part of that gap (0.70–0.79 on hourglass, penetration and
inversion), which is the restoring effect. But spatial rules are phase-invariant and recover more: the position
Laplacian scores 0.79–0.94. In the pre-registered Gate C, the non-causal GNN shows *no benefit* over the best baseline
(−0.058 [−0.073, −0.042]). The causal GNN passes (+0.023 [+0.006, +0.040]), entirely from frozen regions.

#figure(
  table(columns: 8, inset: 3.2pt, stroke: 0.4pt + luma(180), align: (left, left, center, center, center, center, center, center),
    table.header([*failure*], [*phase*], [*GNN*], [*const-vel*], [*pos. Lapl.*], [*vel. z*], [*inv. tet*], [*CLIP kNN*]),
    [hourglass], [onset], [1.00], [1.00], [0.85], [0.97], [0.72], [0.57],
    [], [sustained], [0.79], [0.62], [*0.85*], [0.50], [0.72], [0.57],
    [penetration], [onset], [0.99], [1.00], [0.79], [0.93], [0.56], [0.52],
    [], [sustained], [0.70], [0.66], [*0.79*], [0.55], [0.55], [0.51],
    [inversion], [onset], [1.00], [1.00], [0.94], [0.99], [0.84], [0.52],
    [], [sustained], [0.73], [0.63], [*0.94*], [0.52], [0.84], [0.51],
    [instability], [onset], [0.81], [*0.99*], [0.61], [0.77], [0.51], [0.50],
    [], [sustained], [0.90], [*1.00*], [0.74], [0.92], [0.60], [0.52],
    [frozen], [onset], [0.94], [*1.00*], [0.48], [0.86], [0.50], [0.50],
    [], [sustained], [*0.99*], [0.49], [0.63], [0.85], [0.51], [0.50],
  ),
  caption: [Test-set frame AUROC (92 simulations, mean over 3 severities) by event phase. Bold is the best in the row
  where it is unique. Regenerate with `scripts/phase_table.py`.],
) <tab:phase>

*The frozen win does not survive a complete rule pool.* After seeing the test results, we checked a velocity
Laplacian (a node's velocity minus its neighbours' mean) on validation simulations 20–49. It scores *1.00* on
sustained frozen frames at every severity, against the GNN's 0.97–0.99 (@fig:phase). This rule was not in the
pre-registered pool, so Gate C stands as reported. But the post-hoc reading "learned physics catches kinematic errors"
does not hold. Each local failure has some one-line rule that catches it: the second difference in time, the Laplacian
of positions, or the Laplacian of velocities.

#figure(image("figs/phase_dots.png", width: 100%),
  caption: [Onset vs sustained frame AUROC per detector (validation simulations 20–99, protocol v3, which includes
  the velocity Laplacian). The GNN never beats all three rules. On sustained frames, a different rule wins for each
  failure type.]) <fig:phase>

*A classical crash-data-mining baseline is stronger all-round.* Following @kracker, we describe each frame by 18
mesh-independent statistics (quantiles of speed, acceleration, and position and velocity Laplacian residuals in units
of the run's step size, plus the inverted-element count and contact depth), and score it by its mean distance to the 5
nearest clean frames from validation simulations 50–69. On validation simulations 20–49 this bundle-outlier detector
never collapses: frame AUROC is 0.82–0.98 in every type × phase cell. On sustained hourglass, penetration and inversion
it scores 0.82, 0.88 and 0.91, against the GNN's 0.79, 0.69 and 0.73. It doesn't beat the best single rule in any
cell, but no single rule is safe across all failure types, and the bundle is.

== Whole-run loading errors: contact rules or nothing

Rules that look for local irregularities can't see a run that is smooth everywhere but wrong for its loading. Gate D
tested whether a surrogate can. The shape-from-load surrogate did not meet its pre-registered admission bar: final
RMSE 1.99 cm against 3.18 cm for zero displacement, a ratio of 0.63 with a bar of 0.5. The velocity GNN never beat the
best rule (all Δ < 0). Contact rules detect response-scale errors (AUROC 0.85–1.00), time-scale errors (0.92–1.00) and
10-frame lag (0.91). *A 1–3 frame lag between actuator and plate is invisible to every detector* (AUROC 0.47–0.58).
That is a measured detection limit for positions-only QA.

== Vision: a modality limit, not a resolution limit

In Phase 1, CLIP patch-kNN stayed at 0.50–0.53 on penetration and inversion even at severity 3, where the shift is
16–20 px in a 448 px render. Gate V tested whether that is an artefact of an old backbone or low resolution.
@tab:vision shows the result. No condition reaches the pre-registered 0.80. DINOv2-L patch-kNN is the best (mean
0.65): it detects hourglassing (0.88 [0.79, 0.96]) but not penetration, inversion or frozen regions (0.55–0.62). An
896 px plate crop changes nothing. The renders lose the information that the field-level rules use.

#figure(
  table(columns: 7, inset: 3.2pt, stroke: 0.4pt + luma(180),
    table.header([*condition (sev. 3)*], [*hourglass*], [*penetration*], [*inversion*], [*instability*], [*frozen*], [*mean*]),
    [CLIP B/16 kNN, 448], [0.71], [0.50], [0.51], [0.56], [0.53], [0.56],
    [CLIP L/14 kNN, 448], [0.67], [0.55], [0.54], [0.54], [0.52], [0.56],
    [DINOv2-L kNN, 448], [*0.88*], [0.55], [0.58], [0.65], [0.59], [*0.65*],
    [DINOv2-L kNN, 896 crop], [0.86], [0.58], [0.62], [0.63], [0.56], [0.65],
    [SigLIP2 zero-shot, 896 crop], [0.66], [0.52], [0.53], [0.56], [0.52], [0.56],
  ),
  caption: [Gate V, validation simulations 20–39, frame AUROC at severity 3 (selected rows; full table in
  `results/gate_v`).],
) <tab:vision>

== VLM explanations are not grounded

On 200 test frames sampled without filtering on detection, the local Qwen3-VL-8B called 199 frames "none" from images
alone: 0% type accuracy on anomalies at every severity, including severity 3 (@tab:vlm). Given numeric diagnostics, it
reached 26%, labelling 72% of all frames "instability". A depth-4 decision tree, cross-validated on the same
diagnostics, reached 55%. The tree is supervised and the VLM zero-shot, so it marks what the diagnostics already
contain rather than a like-for-like competitor.

#figure(
  table(columns: 4, inset: 3.2pt, stroke: 0.4pt + luma(180),
    table.header([*arm (200 test frames)*], [*type acc.*], [*macro-F1*], [*exact cell*]),
    [Qwen3-VL-8B, images], [12%], [0.04], [55%],
    [Qwen3-VL-8B, images + diagnostics], [26%], [0.18], [41%],
    [decision tree on diagnostics (supervised, 5-fold)], [55%], [0.55], [79%],
    [majority class / majority cell], [18%], [0.05], [49%],
  ),
  caption: [VLM explanation correctness. Chance for 6 classes is 17%.],
) <tab:vlm>

*Counterfactual pairs.* For 40 pairs, frame A is a clean, strongly bent frame and frame B is the same frame with a
severity-3 hourglass (median visible shift 17 px; @fig:vlm).
- From the render alone, and with the heatmap, the VLM called every B frame "none".
- With diagnostics, it separated 78% of pairs but never named "hourglass".
- The term overlap between its explanations of A and of B (Jaccard 0.25) equals that of unrelated pairs (0.24).

The explanations are fluent but not grounded in what the model sees, which is an automation-bias risk in engineering
sign-off.

#figure(grid(columns: 2, gutter: 6pt, image("figs/vlm_example.png"), image("figs/counterfactual_pair.png")),
  caption: [Left: VLM input, the render and the surrogate-residual panel, for a severity-3 hourglass; the 8B model
  answered "none". Right: a counterfactual pair, clean (A) and the same frame with an injected hourglass (B).]) <fig:vlm>

*Does a larger model help?* A pre-registered pilot ran Qwen3.8-27B (OpenRouter free tier) on 20 severity-3 anomalous
frames, 5 clean frames and 10 counterfactual pairs, paired against the 8B on the same images and prompts.
- The 27B stops defaulting to "none" (6/20 anomalous frames, against 20/20 for the 8B).
- It localizes the residual hotspot: exact grid cell 11/20.
- But it names the right failure only 2/20 times, mostly answering "penetration", because the heatmap is often red
  near the actuator.
- It separates 1 of 10 counterfactual pairs.

The pre-registered rule says do not scale up.

*Does video help?* Crash results are reviewed as animations, so a second pre-registered pilot gave the local 8B an
8-frame clip (frames t−7 … t, plain renders) instead of a single frame, on the same 25 frames and 10 pairs. It
answered "none" to all 90 prompts in both arms: 0/20 correct and 0/10 pairs separated either way. Temporal context
does not rescue a model that doesn't see the failure in the first place. The larger model sees _where_ the surrogate flags a problem, but not
_what_ went wrong.

== Lead time

With causal calibration and per-frame thresholds at 95% specificity set on validation, constant velocity warns 39
frames before a growing instability reaches its cap, the GNN 18 frames and the position Laplacian 7. A conformal
run-level alarm with a guaranteed false-alarm rate was attempted. On validation, its false-alarm rate exceeded the
nominal α (up to 30% at α = 10%), so we do not claim the guarantee. We treat exchangeability across simulation ranges
as an open question.

= Discussion

*Practitioner checklist.*
+ Report a learned residual detector against its zero-output baseline (here, constant velocity) and cheap spatial
  rules. Here they win.
+ Evaluate on sustained frames, not only onset: onset flatters every temporal detector.
+ When fields are available, check fields rather than renders. Vision backbones lose the signal even at higher
  resolution.
+ Don't use free-text VLM explanations for sign-off without grounding tests. Fluent answers were not tied to image
  content.

What is left for learned models is narrower than hoped: failures that are smooth, locally plausible and only
inconsistent with the loading. Our shape surrogate did not reach the accuracy needed for that, and the smallest timing
errors defeated every detector.

= Limitations

The failures are injected, not produced by a solver. They snap back to the clean trajectory, and the dataset is one
quasi-static hyperelastic benchmark, not explicit crash dynamics. The surrogate is laptop-scale: 8 h and 250 of 1,000
trajectories. The VLM study uses one model family. The replication set (training trajectories 250–349) has not yet
been scored. Next steps are solver-generated failures, for example OpenRadioss runs with hourglass control disabled
and labelled by solver energies; a real crash dataset once one is public; temporal (video) VLM inputs; and
simulation-grounded explanation methods.

#v(4pt)
*Use of AI tools.* An AI coding assistant (Anthropic's Claude) was used as a research-engineering tool to
accelerate implementation, debugging, experiment monitoring and drafting, and to run analyses in parallel. The author
designed the study, set every pre-registered decision rule, and reviewed and takes responsibility for all code,
results and text.

#set text(size: 8.8pt)
#bibliography("refs.yml", title: "References", style: "ieee")
