n = 200 frames; uniform-random type accuracy = 17%

| arm | valid JSON | type accuracy | macro-F1 | exact location cell | type acc. on GNN-flagged frames |
|---|---|---|---|---|---|
| visual (VLM, images) | 100% | 12% | 0.04 | 55% | 2% (n=123) |
| full (VLM, images + diagnostics) | 100% | 26% | 0.18 | 41% | 24% (n=123) |
| diagnostics-only decision tree (5-fold CV) | 100% | 55% | 0.55 | 79% | 67% (n=123) |
| majority class / majority cell | 100% | 18% | 0.05 | 49% | 26% (n=123) |
