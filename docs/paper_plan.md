# Paper plan (ARS plan mode, started 2026-09-26)

Targets: arXiv preprint + BMW application note first; ICLR 2027 workshop version cut from it.

## Pending experiments (thesis framing depends on these)
- **Phase 2**: smooth-everywhere but load-inconsistent runs. Assumed finished before writing.
- **Gate D (GNN scale)**: laptop only, longer overnight runs. Pre-register before running. Caveat: covers ~2-3x budget, so "gap stays" is weak evidence; "gap shrinks" is strong.
- **Gate E (VLM capacity)**: model TBD. Same 200 test frames + 40 counterfactual pairs, same prompts. Pre-register before running.

## INSIGHT collection

[INSIGHT: thesis_statement]
Core (survives any Gate D/E outcome):
- A: a surrogate-residual detector means nothing until it beats its own zero-output baseline (constant velocity) and cheap spatial rules.
- B: the contribution is the benchmark itself: injected failures with ground-truth type and location, onset vs. sustained split, pre-registered gates.
Supporting: C: the noise-trained surrogate is a denoiser, not a simulator; its signal ("state doesn't relax back") is also picked up by a simple rule.
Conditional: D (VLM explanations ungrounded, automation-bias risk) stays conditional until Gate E runs.

[INSIGHT: outcome_framing]
No Gate D/E outcome breaks the paper; they change framing only:
- D stays, E flat: negative result robust to scale and capacity.
- D shrinks, E flat: detection is budget-dependent; explanation finding stands.
- D stays, E improves: detection negative; explanation becomes a capacity threshold.
- Both improve: "what budget/capacity is needed"; benchmark carries the paper.
- D ambiguous (most likely): report as inconclusive over the tested range.

[INSIGHT: practitioner_takeaway]
Emphasised:
1. Run the cheap rules before trusting a learned surrogate.
2. Evaluate on sustained frames, not just onset.
5. Learned physics' possible niche: smooth but load-inconsistent runs (Phase 2).
Secondary: don't use VLM free text for sign-off; CLIP is useless at this pixel scale.

## Chapter plan
Skeleton: 1 Intro · 2 Related work · 3 Benchmark · 4 Detectors & protocol (Gates A-E) · 5 Detection results (D) ·
6 Explanation study (E) · 7 Discussion & practitioner guidance · 8 Limitations & conclusion.
Workshop cut: ch. 2 and 6 to one paragraph each, lead time to appendix.

### 1 Introduction
- Urgency: crash-simulation failures (hourglassing, penetration) get missed or caught late, and the ML fixes on offer haven't been tested fairly.
- Gap: residual detection is known, but no controlled comparison against its own zero-output baseline and simple rules exists, because no FE-failure benchmark with ground-truth type and location existed.
- Order: the comparison question leads (thesis A); the benchmark is introduced as the instrument (thesis B).

### 2 Related work
- Threads: (1) GNN surrogates for FE/crash (accuracy, not failure detection; crash surrogates folded in, one sentence for BMW);
  (2) residual/digital-twin fault detection (no fair baselines); (3) visual anomaly detection + VLMs on simulations (no correctness checks).
- Position against: residual fault detection as a genre (rarely compared with its own zero-output baseline); secondary:
  "Simulation vs. Hallucination" (QA, not grounding against injected ground truth).
- Lands on: "the detector is classical, the comparison is missing" → ch. 3-4.
- TODO: verify citations in research_notes.md (2026 arXiv IDs) with /ars-citation-check before drafting.

### 3 Benchmark
- "Not real physics" defence: concede; each injector mirrors a real FE failure signature and injection is the only route
  to ground-truth type + location; injected failures are a lower bound on difficulty. OpenRadioss named as next step, not run.
- Dataset: deforming_plate is the only public FE structural benchmark with MGN baselines; CarCrashNet unreleased; stand-in.
  Argue quasi-static is the surrogate's easy case (STRESS-TEST in Step 3: it is also constant velocity's easy case).
- Main-text weakness: additive injections snap back to the clean trajectory (not self-consistent).
  Limitations: frozen sev-1 near-imperceptible; severity relative to RMS displacement, not physical units.

### 4 Detectors & protocol
- "Undertrained" defence: Gate B states the gap openly (6.9x worse than constant velocity); Gate D gives the scaling
  trend over the tested range; Gate A shows sigma, not length, moves AUROC by ~0.10 at equal budget.
- Velocity Laplacian: reported openly as post-hoc, validation-only; Gate C stands; it refutes the kinematic-vs-geometric
  reading. It enters the Phase 2 and Gate D pre-registrations from the start (no extra gate, test sims not re-scored).
- Gates shown as one timeline figure: A -> B -> C -> post-hoc -> D/E/Phase 2, each with pre-registration date and decision.

### 5 Detection results
- Headline: in every failure type and phase, some one-line rule matches or beats the learned surrogate (frozen included,
  once the velocity Laplacian counts). Directly under it, the mechanism: constant velocity collapses on sustained
  anomalies; the surrogate recovers part of the gap, cheap spatial rules recover more.
- Causal Gate C split (+0.023): reported exactly as pre-registered, missing-baseline caveat in the same paragraph.
  TODO: build past-only velocity z-score (~1 h) and put it in the Gate D pre-registration.
- Order: Gate B -> Gate C -> onset/sustained table -> velocity-Laplacian refutation -> lead time -> Phase 2 -> Gate D.
  Workshop cut: lead time to appendix.

### 6 Explanation study
- Measures correctness + grounding (type/location vs injected ground truth; counterfactual pairs). States plainly that
  free-text faithfulness is not checked.
- Decision tree reframed as a ceiling ("what the diagnostics already contain"), not a competitor.
  TODO: zero-shot rule classifier (thresholds on validation, no training) as like-for-like baseline (~2 h).
- Visibility: px-shift column separates "unseen" from "unrecognised"; the heatmap arm shows the location and still gets
  "none", so visibility doesn't explain it. Gate E stays single-variable (capacity only; no crop arm).

### 7 Discussion
- Responds to residual/digital-twin detection first (report the zero-output baseline, thesis A), then MGN/surrogate
  work (used as a detector, the surrogate is a denoiser, thesis C).
- One sentence: ICLR cut "Benchmark a learned detector against its own zero-output baseline"; BMW note "Run the cheap
  rules before you trust the surrogate".
- Practitioner takeaways 1, 2, 5 as a numbered QA checklist box.

### 8 Limitations & conclusion
- Future work, realism before scale (Gate D can't reach paper scale): OpenRadioss solver-generated failures ->
  CarCrashNet once released -> scaling up the Phase 2 niche.
- Final paragraph: thesis A + B in two sentences, then the Phase 2 niche as the open question.
