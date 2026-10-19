# PolyOpus training data and final evaluation


## Reported evaluation

| Property | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 7,000 | 24.579579 | 35.744916 | 0.926128 |
| Crystalline band gap (eV) | 4,800 | 0.701811 | 0.962368 | 0.790170 |
| AE (eV/atom) | 1,806 | 0.062148 | 0.075725 | 0.979460 |


```bash
python reproduction/data/evaluate_saved_predictions.py
```

The script verifies CSV hashes and counts, the AE rule, and all table metrics.
`--output NEW_FILE.json` saves the verification result.

## Sources and license

Property data were reused through [PolyNC](https://doi.org/10.1039/D3SC05079C) and
its [published repository](https://github.com/HKQiu/Unified_ML4Polymers), including
the [Kuenneth et al. compilation](https://doi.org/10.1016/j.patter.2021.100238).
PolyNC publishes these processed materials under
[Apache License 2.0](https://github.com/hkqiu/Unified_ML4Polymers/blob/1833503fc06a77e9a098a7b40daabd279632ba14/LICENSE);
a copy is included as [PolyNC_LICENSE.txt](PolyNC_LICENSE.txt).
`metadata.json` documents source attribution, instruction augmentation and corpus
composition; the training CSV preserves the supplied archive bytes.

Additional results: [structure and duplicate audit](structure_audit/README.md),
[CPU generation and GNN](../../reproduction/gnn/revision_run_20261015/README.md),
and [GPT-6-Astra comparison](../../reproduction/current_models/gpt6_astra_20261015/README.md).
