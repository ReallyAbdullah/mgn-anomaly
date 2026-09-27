**Local failures** — frame AUROC, validation sims 20–49, mean over 3 severities

| failure | phase | bundle | gnn | constvel | laplacian | vlap | velocity | jacobian | contact |
|---|---|---|---|---|---|---|---|---|---|
| hourglass | onset | 0.98 | 1.00 | 1.00 | 0.87 | 1.00 | 0.97 | 0.73 | 0.51 |
| hourglass | sustained | 0.82 | 0.79 | 0.63 | 0.87 | 0.49 | 0.48 | 0.73 | 0.51 |
| penetration | onset | 0.97 | 1.00 | 1.00 | 0.81 | 1.00 | 0.94 | 0.53 | 0.91 |
| penetration | sustained | 0.88 | 0.69 | 0.66 | 0.81 | 0.68 | 0.55 | 0.54 | 0.92 |
| inversion | onset | 0.98 | 1.00 | 1.00 | 0.96 | 1.00 | 0.98 | 0.82 | 0.49 |
| inversion | sustained | 0.91 | 0.73 | 0.62 | 0.96 | 0.65 | 0.53 | 0.82 | 0.49 |
| instability | onset | 0.92 | 0.83 | 0.99 | 0.62 | 0.99 | 0.81 | 0.51 | 0.49 |
| instability | sustained | 0.97 | 0.91 | 1.00 | 0.76 | 1.00 | 0.95 | 0.61 | 0.50 |
| frozen | onset | 0.97 | 0.92 | 1.00 | 0.48 | 1.00 | 0.82 | 0.50 | 0.51 |
| frozen | sustained | 0.95 | 0.98 | 0.49 | 0.65 | 1.00 | 0.82 | 0.52 | 0.52 |

**Whole-run errors** — run AUROC (clean vs corrupted), validation sims 20–49

| error | severity | bundle |
|---|---|---|
| scale | 0.05 | 0.48 |
| scale | 0.1 | 0.48 |
| scale | 0.2 | 0.48 |
| lag | 1 | 0.51 |
| lag | 3 | 0.54 |
| lag | 10 | 0.82 |
| timescale | 0.95 | 0.88 |
| timescale | 0.9 | 0.97 |
| timescale | 0.8 | 0.97 |
