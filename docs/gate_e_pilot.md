# Gate E pilot — pre-registered 2026-09-27, before any 27B output was seen

**Question.** Is a larger model from the same family, Qwen3.8-27B (`qwen/qwen3.8-27b:free` via OpenRouter), worth a
full Gate E run? The comparison point is the local Qwen3-VL-8B (4-bit, MLX), which labelled 199/200 test frames
"none" from images alone.

**Budget constraint.** The OpenRouter free tier allows 50 requests per day, so this is a pilot. It is sized to one
day's budget and paired against the 8B on the same images and prompts.

**Inputs, taken from the committed Phase 1 VLM study.** Prompts are reused verbatim.
- **Frames:** 20 severity-3 anomalous frames from `results/vlm_results.json`, 4 per type, drawn stratified with seed
  0. Where a type has fewer than 4, all of them are used. Plus 5 clean frames (seed 0). The arm is images only
  (`visual_prompt`, render and residual heatmap panels). That's 25 calls.
- **Pairs:** the first 10 counterfactual pairs in `results/counterfactual.json`, render-only arm, frames A and B.
  That's 20 calls.
- **Settings:** temperature 0, reasoning enabled, one retry on invalid JSON. Retries count against the budget.

**Decision rule (fixed now).** Scale Gate E up, by adding OpenRouter credit or spreading it over days, only if at
least one of these holds:
1. The 27B gets at least 5 more of the 20 anomalous frames' type right than the 8B did on the same frames.
2. The 27B separates at least 3 of the 10 pairs: it calls A "none" and calls B anomalous.

Otherwise, report as a pilot result: *no evidence that a larger model from the same family fixes it, at n = 20 / 10*.
No prompt changes, and no re-runs to fish for a better result.

**Reported regardless:**
- type accuracy
- the rate of "none" answers
- clean frames correctly called "none"
- exact-cell location accuracy
- pair discrimination
- invalid-JSON rate
- the raw responses

## Addendum (2026-09-27, before any Inkling output was seen): model-family variation

After the Qwen3.8-27B result, the same 25 frames and 10 pairs, prompts and settings are run with
`thinkingmachines/inkling:free` (OpenRouter), a different model family. The question is descriptive: does the
failure to name the failure persist across model families? The same table is reported next to the 27B and 8B, and the
original decision rule is applied unchanged. Output goes to `results/vlm_pilot_inkling/`. The free tier allows 50
requests/day, so this runs when the daily quota resets.

**Amendment (2026-09-27, before any output was seen).** `thinkingmachines/inkling:free` rejects plain API calls
(HTTP 403: free access only through approved agent apps), so the different-family run uses
`google/gemma-4-31b-it:free` instead, with identical items, prompts, settings and decision rule. Output:
`results/vlm_pilot_gemma/`.
