# PolyOpus training data and final evaluation

| File | Records | Purpose |
|---|---:|---|
| `train/merged_train.csv` | 233,827 | Original merged training corpus |
| `test/merged_test.csv` | 23,200 | Original TestSet2 instruction inputs |
| `test/benchmark.csv` | 232 | Original TestSet1 comparison inputs |
| `final_predictions.csv` | 18,978 | One final table for all reported regression and classification outcomes |
| `metadata.json` | — | Metrics, denominators, reporting rule, source attribution and input hashes |

The final table contains task, complete input, original target/prediction strings,
unit and original source record ordinal. It combines 7,000 Tg, 4,800 crystalline
band-gap, 1,806 AE and 5,372 classification outputs. UTF-8 is used for the
final table; source train/test CSV bytes are preserved. Original process files and
augmentation lineage are retained in our local archive.

## Reported evaluation

| Property | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 7,000 | 24.579579 | 35.744916 | 0.926128 |
| Crystalline band gap (eV) | 4,800 | 0.701811 | 0.962368 | 0.790170 |
| AE (eV/atom) | 1,806 | 0.062148 | 0.075725 | 0.979460 |

Classification has 5,372/5,550 saved outputs (178 missing): accuracy **0.786485**,
macro-F1 **0.686666**, weighted-F1 **0.775129**. These are Tg-derived polyimide
heat-resistance classes, not decomposition temperatures. Metrics give each saved
instruction row equal weight. Augmented instruction rows are not independent
polymers; R² uses the target variance of the corresponding scored rows. Metrics
are recalculated from target/prediction, never from the historical `error` column.

From the repository root, using Python's standard library:

```bash
python reproduction/data/evaluate_saved_predictions.py
```

The script verifies the four CSV hashes and row counts, training-label range,
AE retained rule, and every reported regression/classification metric. Optional
`--output NEW_FILE.json` saves the verification result. It does not recompute
the local full-output AE summary from the public retained table.

## Training and provenance

Property data were reused through [PolyNC](https://doi.org/10.1039/D3SC05079C) and
its [published repository](https://github.com/HKQiu/Unified_ML4Polymers), including
the [Kuenneth et al. compilation](https://doi.org/10.1016/j.patter.2021.100238).

Additional results: [CPU generation and GNN](../../reproduction/gnn/revision_run_20261015/README.md)
and [GPT-6-Astra comparison](../../reproduction/current_models/gpt6_astra_20261015/README.md).


## PolyNC source license

The official PolyNC repository publishes its accompanying data and code with
[Apache License 2.0](https://github.com/hkqiu/Unified_ML4Polymers/blob/1833503fc06a77e9a098a7b40daabd279632ba14/LICENSE). A copy is retained as
[PolyNC_LICENSE.txt](PolyNC_LICENSE.txt). This release attributes the reused
processed PolyNC materials and describes the instruction augmentation and corpus
composition in `metadata.json`. Redistribution under that license requires the license copy,
applicable attribution notices and notices of modifications. The released PolySea
training CSV preserves the supplied archive bytes; the derived prompt/SMILES
expansion is documented rather than represented as original measurements.

The license was checked on 15 October 2026. That is not an asserted original
database acquisition date. This evidence concerns reuse of the published PolyNC
materials, not a grant to scrape or relicense an unrelated upstream database.

[Existing-split structure and duplicate audit](structure_audit/README.md) reports
normalized identity counts and fixed-representative metrics without retraining.
