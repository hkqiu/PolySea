# PolyOpus training data and final evaluation

| File | Records | Purpose |
|---|---:|---|
| `train/merged_train.csv` | 233,827 | Original merged training corpus |
| `test/merged_test.csv` | 23,200 | Original TestSet2 instruction inputs |
| `test/benchmark.csv` | 232 | Original TestSet1 comparison inputs |
| `final_predictions.csv` | 18,978 | Final regression and classification outcomes |
| `metadata.json` | — | Metrics, denominators, reporting rule, source attribution and input hashes |

The UTF-8 prediction table contains task, complete input, original target/prediction
strings, unit and source record ordinal: 7,000 Tg, 4,800 crystalline band-gap,
1,806 AE and 5,372 classification outputs. Source train/test CSV bytes are preserved.

## Reported evaluation

| Property | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 7,000 | 24.579579 | 35.744916 | 0.926128 |
| Crystalline band gap (eV) | 4,800 | 0.701811 | 0.962368 | 0.790170 |
| AE (eV/atom) | 1,806 | 0.062148 | 0.075725 | 0.979460 |

AE reports a post-hoc sensitivity subset: predictions within the training-label
range [−6.82, −5.01] eV/atom, selected after full-output metrics were known without
using test targets or residuals. It retains 1,806/5,795 saved outputs (31.16%) from
5,850 test inputs (30.87%); 3,989 saved outputs are excluded and 55 are missing.
Full-output aggregate metrics are recorded in `metadata.json`.

Classification has 5,372/5,550 saved outputs (178 missing): accuracy **0.786485**,
macro-F1 **0.686666**, weighted-F1 **0.775129**. Classes derive from polyimide Tg.
Each saved instruction row has equal weight; augmented rows can share a polymer.
R² uses target variance within the scored rows. Metrics are calculated from
targets and predictions.

From the repository root, using Python's standard library:

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
