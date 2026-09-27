# Gate E video pilot — pre-registered 2026-09-27, before any video-arm output was seen

**Question.** Does a short *video* (a sequence of frames) help a VLM detect and name simulation failures, compared
with a single frame? The BMW position concerns crash-*video* evaluation, and failures such as frozen regions or
instability are temporal, so a single frame may be the wrong input format.

**Model.** The local Qwen3-VL-8B (4-bit, MLX), the Phase 1 model. It is free and has no request limit. Only the input
format changes.

**Items (paired, reused from the Gate E pilot).**
- The same 20 severity-3 anomalous frames and 5 clean frames. Selection: `mgn.vlm_pilot.select_frames`, seed 0.
  Their simulations and times are recovered by re-running the Phase 1 selection, which reproduces all 200 image names
  exactly.
- The same 10 counterfactual pairs: clean frame 330 (A) vs the same frame with a severity-3 hourglass (B).

**Arms.** Both use plain renders only, with no residual heatmap and no diagnostics, so only visual evidence is
available.
1. *single*: the render of frame t.
2. *clip*: renders of frames t−7 … t (8 frames, oldest first). For B pairs the hourglass is present from t−2, so the
   clip shows clean frames and then the onset.

Prompt: the Phase 1 render-only prompt. The clip arm adds one sentence saying the images are 8 consecutive frames,
oldest first, and asking it to judge the last frame. Temperature 0, one retry on invalid JSON.

**Decision rule (fixed now).** Video counts as helping if the clip arm names the correct type on at least 5 more of
the 20 anomalous frames than the single arm, *or* separates at least 3 more of the 10 pairs. Otherwise it is reported
as no evidence that temporal context helps this model.

**Reported regardless:** type accuracy, "none" rate, clean frames called "none", pair separation, invalid JSON.
- Clean frames are rendered from the clean trajectory, so their 8-frame clips can't contain the nearby injected event.
