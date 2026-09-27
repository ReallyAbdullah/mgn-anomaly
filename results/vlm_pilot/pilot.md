Gate E pilot — qwen/qwen3.8-27b:free vs local Qwen3-VL-8B (4-bit), same images and prompts

| metric | 27B | 8B |
|---|---|---|
| type accuracy, 20 sev-3 anomalous frames | 2/20 | 0/20 |
| answered 'none' on anomalous frames | 6/20 | 20/20 |
| clean frames called 'none' | 4/5 | 5/5 |
| exact location cell (anomalous, valid answers) | 11/20 | — |
| counterfactual pairs separated (render only) | 1/10 | 0/10 |
| invalid JSON | 0/45 | — |

**Pre-registered decision: do not scale up (no evidence the larger model fixes it)**
