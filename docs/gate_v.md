# Gate V (vision baselines) — pre-registered 2026-09-27, before any result of this sweep was seen

**Question.** Phase 1 found CLIP near chance on rendered failures. Is that a real limit of vision on these renders,
or an artefact of a 2021 backbone at 224 px? Phase 1 shifts at 448 px were about half a CLIP patch.

**Data.**
- Validation sims 20–39, all 5 local failure types at severities 2 and 3, with the protocol v3 injectors and seeds.
- Per corrupted copy: the onset frame, up to 2 sustained frames (evenly spaced), and 3 clean frames from the same copy.
- Memory bank: clean renders of validation sims 0–19 at frames 50, 120, 190, 260 and 330.
- Test sims are not used.

**Conditions (backbone × input):**
- **Patch kNN:**
  - CLIP ViT-B/16 (OpenAI), the Phase 1 baseline
  - OpenCLIP ViT-L/14 (LAION-2B)
  - DINOv2-L
- **Zero-shot:** CLIP ViT-B/16 and SigLIP2 ViT-B/16-512, using the WinCLIP prompt ensemble from Phase 1.
- **Inputs:**
  - (a) the Phase 1 448 px render, resized to the model input
  - (b) an 896 px render cropped to the plate. The crop box is fixed per sim from the clean positions at frames 0
    and 349, so it can't leak the anomaly location. It is tiled 2×2 for the 224 px models and fed at 448 px to DINOv2.
- Frame score: the maximum patch distance to the memory bank (kNN, k = 1, cosine).

**Criterion.** Vision counts as "not blind" if any condition reaches a mean frame AUROC ≥ 0.80 over the 5 types at
severity 3. Report AUROC per type and severity, with a bootstrap over sims.

**Interpretation, fixed in advance.**
- If only (b) passes, it's resolution: the failures are visible given enough pixels.
- If nothing passes, it's modality: renders lose the information that field-level rules use.

In both cases the result bears on crash-video evaluation, where image resolution is also limited.

Note: a 1-sim pipeline smoke test (validation sim 20) ran before the full sweep; it is not part of the result.
