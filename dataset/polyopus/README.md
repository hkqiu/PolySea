# PolyOpus training data and final evaluation

| File | Records | Purpose |
|---|---:|---|
| `train/merged_train.csv` | 233,827 | Original merged training corpus, unchanged bytes |
| `test/merged_test.csv` | 23,200 | Original TestSet2 instruction inputs, unchanged bytes |
| `test/benchmark.csv` | 232 | Original TestSet1 comparison inputs, unchanged bytes |
| `final_predictions.csv` | 18,978 | One final table for all reported regression and classification outcomes |
| `metadata.json` | — | Metrics, denominators, reporting rule, source attribution and input hashes |

The final table contains task, complete input, original target/prediction strings,
unit and original source record ordinal. It combines 7,000 Tg, 4,800 crystalline
band-gap, 1,806 AE and 5,372 classification outputs. UTF-8 is used for the
final table; source train/test CSV bytes are preserved. Original process files and
augmentation lineage are retained in the authors' local archive.

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

The saved-order reconstruction found per-prompt-file block sizes of 10/20/15/15
for Tg/crystalline gap/AE/classes, repeated across ten core files. Two files were
identical, giving four distinct English and five Chinese templates. Original
duplicated rows and 9,990 classification targets with erroneous `eV` suffixes are
preserved. Metadata retains the concise reconstruction counts; inferred blocks
are not proven original measurement IDs or periodic-polymer equivalence. The
original component source of 4,291 QA/other rows remains unresolved.

Property data were reused through [PolyNC](https://doi.org/10.1039/D3SC05079C) and
its [published repository](https://github.com/HKQiu/Unified_ML4Polymers), including
the [Kuenneth et al. compilation](https://doi.org/10.1016/j.patter.2021.100238).
This is not a new automated acquisition from PolyInfo. Upstream acquisition
dates and the exact upstream row mapping for the crystalline-gap subset have
not been recovered; 2026-10-15 is a local archival date. The repository's code
license does not independently verify or replace all upstream data rights.

Additional results: [CPU generation and GNN](../../reproduction/gnn/revision_run_20261015/README.md)
and [GPT-6-Astra comparison](../../reproduction/current_models/gpt6_astra_20261015/README.md).
