| A (model/features) | B | F1 difference A - B | 95% CI (paired time-step bootstrap) | Share of resamples with A ≤ B |
|---|---|---|---|---|
| rf/all | rf/local | +0.043 | [+0.031, +0.058] | 0.000 |
| rf/all | sage/all | +0.178 | [+0.136, +0.238] | 0.000 |
| rf/all | rf+sage/all | +0.170 | [+0.117, +0.255] | 0.000 |
| rf/local | rf+sage/local | +0.133 | [+0.072, +0.224] | 0.000 |
| mlp/local | sage/local | +0.099 | [+0.051, +0.178] | 0.000 |
| sage/local | gcn/local | +0.575 | [+0.394, +0.650] | 0.000 |
| sage/all | mlp/all | +0.074 | [+0.011, +0.146] | 0.010 |
| mlp/local | mlp/all | +0.204 | [+0.111, +0.326] | 0.000 |
