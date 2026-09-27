Gate D on {'split': 'valid', 'sims': [20, 50], 'ref': [50, 70], 'gnn': '/Users/muhammadabdullah/Claude Projects/mgn-anomaly/runs/mgn/model.pt', 'shape': '/Users/muhammadabdullah/Claude Projects/mgn-anomaly/runs/shape/model.pt'}; shape model admitted: False (final RMSE, baseline = ('1.99e-02', '3.18e-02')); surrogate stats: ['gnn']

| cell | surrogate stat | best rule | Δ AUROC | CI low | one-sided p | Holm |
|---|---|---|---|---|---|---|
| scale/0.05 | gnn | contact_gap | -0.370 | -0.478 | 1.0000 |  |
| scale/0.1 | gnn | contact_gap | -0.470 | -0.560 | 1.0000 |  |
| scale/0.2 | gnn | contact_gap | -0.536 | -0.626 | 1.0000 |  |
| lag/1 | gnn | laplacian | -0.006 | -0.040 | 0.5547 |  |
| lag/3 | gnn | contact_depth | -0.085 | -0.197 | 0.9355 |  |
| lag/10 | gnn | contact_depth | -0.373 | -0.503 | 1.0000 |  |
| timescale/0.95 | gnn | contact_depth | -0.279 | -0.457 | 0.9995 |  |
| timescale/0.9 | gnn | contact_depth | -0.141 | -0.276 | 1.0000 |  |
| timescale/0.8 | gnn | contact_gap | -0.076 | -0.174 | 0.9995 |  |

**Gate D: FAIL**
