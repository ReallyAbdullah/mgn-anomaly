Gate C on 92 test sims (checkpoint b917f77ffc26), best baselines chosen on /Users/muhammadabdullah/Claude Projects/mgn-anomaly/results/valid/metrics.csv

**Primary — gnn** (sustained phase, hourglass/inversion/frozen x 3 severities): mean AUROC difference vs best baseline -0.058, 95% CI [-0.073, -0.042] -> **no demonstrated benefit**
  best baselines per cell: hourglass/1=constvel, hourglass/2=laplacian, hourglass/3=laplacian, inversion/1=laplacian, inversion/2=jacobian, inversion/3=laplacian, frozen/1=velocity, frozen/2=velocity, frozen/3=velocity

Secondary — gnn vs best baseline per cell (all event frames; * = Holm-significant):

| cell | baseline | Δ AUROC | one-sided p |
|---|---|---|---|
| hourglass/1 | constvel | -0.004 | 0.6437 |
| hourglass/2 | laplacian | -0.176 | 1.0000 |
| hourglass/3 | laplacian | -0.018 | 1.0000 |
| penetration/1 | constvel | -0.023 | 0.9360 |
| penetration/2 | laplacian | -0.078 | 0.9980 |
| penetration/3 | laplacian | -0.134 | 1.0000 |
| inversion/1 | laplacian | -0.146 | 1.0000 |
| inversion/2 | jacobian | -0.225 | 1.0000 |
| inversion/3 | laplacian | -0.148 | 1.0000 |
| instability/1 | constvel | -0.298 | 1.0000 |
| instability/2 | constvel | -0.020 | 1.0000 |
| instability/3 | constvel | +0.000 | 0.5807 |
| frozen/1 | velocity | +0.089 * | 0.0005 |
| frozen/2 | velocity | +0.141 * | 0.0005 |
| frozen/3 | velocity | +0.145 * | 0.0005 |

**Primary — gnn_causal** (sustained phase, hourglass/inversion/frozen x 3 severities): mean AUROC difference vs best baseline +0.023, 95% CI [+0.006, +0.040] -> **learned physics helps**
  best baselines per cell: hourglass/1=constvel_causal, hourglass/2=laplacian_causal, hourglass/3=jacobian, inversion/1=laplacian_causal, inversion/2=jacobian, inversion/3=jacobian, frozen/1=laplacian_causal, frozen/2=laplacian_causal, frozen/3=laplacian_causal

Secondary — gnn_causal vs best baseline per cell (all event frames; * = Holm-significant):

| cell | baseline | Δ AUROC | one-sided p |
|---|---|---|---|
| hourglass/1 | constvel_causal | -0.003 | 0.5742 |
| hourglass/2 | laplacian_causal | -0.144 | 1.0000 |
| hourglass/3 | jacobian | -0.008 | 0.7736 |
| penetration/1 | constvel_causal | -0.020 | 0.9110 |
| penetration/2 | laplacian_causal | -0.060 | 0.9920 |
| penetration/3 | laplacian_causal | -0.098 | 1.0000 |
| inversion/1 | laplacian_causal | -0.113 | 1.0000 |
| inversion/2 | jacobian | -0.237 | 1.0000 |
| inversion/3 | jacobian | -0.153 | 1.0000 |
| instability/1 | constvel_causal | -0.318 | 1.0000 |
| instability/2 | constvel_causal | -0.015 | 0.9860 |
| instability/3 | constvel_causal | +0.000 | 0.5722 |
| frozen/1 | constvel_causal | +0.217 * | 0.0005 |
| frozen/2 | laplacian_causal | +0.384 * | 0.0005 |
| frozen/3 | laplacian_causal | +0.250 * | 0.0005 |

