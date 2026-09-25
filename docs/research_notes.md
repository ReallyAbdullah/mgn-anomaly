# Research notes (2026-09-26): prior art, real-failure data, venues

## Closest prior art (to cite, and to position against)
- **MeshGraphNets** (Pfaff et al., ICLR 2021), the surrogate architecture. The MGN-cloth formulation (velocity in,
  acceleration out, noise on the inputs) is what we use.
- **GNN surrogates for FE analysis**: [arXiv 2211.09373](https://arxiv.org/pdf/2211.09373) and the comprehensive
  evaluation [arXiv 2510.15750](https://arxiv.org/pdf/2510.15750). These target surrogate accuracy, not failure
  detection.
- **Crashworthiness surrogates**: [Mask-Morph Graph U-Net, arXiv 2605.15231](https://arxiv.org/pdf/2605.15231) and
  [CarCrashNet / CrashSolver, arXiv 2605.07098](https://arxiv.org/pdf/2605.07098). The closest domain match for BMW.
- **GNN + FE for damage identification**: ["What lies within", Structures 2025](https://www.sciencedirect.com/science/article/pii/S0141029625012337).
  This is structural damage detection, not detection of simulation failures.
- **Surrogate-residual fault detection** is a classical digital-twin idea, so our contribution is not the detector.
  It is the controlled benchmark (learned physics vs rules vs vision) and the explanation study.
- **Image anomaly detection**: WinCLIP (CVPR 2023), PatchCore (CVPR 2022), AnomalyCLIP (ICLR 2024).
- **VLMs on engineering simulations**: ["Simulation vs. Hallucination", DESTION 2025](https://dl.acm.org/doi/10.1145/3722573.3727826)
  evaluates VLM question answering on simulation outputs, the nearest prior work to our VLM study. Also the
  [mechanics / spatial-geometry awareness of LLMs, arXiv 2608.14615](https://arxiv.org/pdf/2608.14615).

## Real (solver-generated) failures — the biggest realism upgrade
- **OpenRadioss** (AGPL, [github](https://github.com/OpenRadioss/OpenRadioss)) is an industrial explicit crash
  solver. Its 8-node solids use one-point integration with Flanagan–Belytschko hourglass control, and it offers four
  hourglass formulations. Switching hourglass control off, or weakening it, produces *real* hourglassing. Contact
  settings can produce real penetration. This is the recommended route for solver-generated failures: a small plate
  impact model, run with and without hourglass control. It's a multi-day spike.
- **CarCrashNet** (CC BY 4.0) has 15,567 OpenRadioss crash simulations (bumper-beam pole impacts plus 825
  full-vehicle crashes, 6.65 TB, VTKHDF, with plastic strain and erosion flags). **Not yet downloadable**: the README
  says it will be released after peer review. Once it is, the bumper-beam subset would be a far better stand-in for
  BMW than `deforming_plate`.

## Venues (honest targets)
- NeurIPS 2026 workshops. Most calls for papers closed on Aug 30, and a few close on Sep 26
  ([tracker](https://aiworkshoptracker.com/conference/neurips/2026/)). That is too late for this cycle.
- Next realistic targets: ICLR 2027 workshops (AI for science / simulation), a NAFEMS or CAE-industry conference,
  or an arXiv preprint plus the BMW application.
