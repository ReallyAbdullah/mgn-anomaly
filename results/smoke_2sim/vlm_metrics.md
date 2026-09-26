n = 15 frames; uniform-random type accuracy = 17%

| arm | valid JSON | type accuracy | macro-F1 | exact location cell |
|---|---|---|---|---|
| visual (VLM, images) | 100% | 33% | 0.08 | 70% |
| full (VLM, images + diagnostics) | 100% | 13% | 0.04 | 100% |
| diagnostics-only decision tree (5-fold CV) | 100% | 53% | 0.37 | 100% |
| majority class / majority cell | 100% | 33% | 0.08 | 50% |
