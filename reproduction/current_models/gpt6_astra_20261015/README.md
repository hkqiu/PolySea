# GPT-6-Astra: final predictions

This release contains one prediction table with all **235 formal cases**: the three original SI questions and all 232 TestSet1 property rows. The 235 calls completed successfully; all 232 property outputs were parseable. Parameters were fixed at 8192 output tokens and low reasoning, with no tools, retrieval, or cross-case history.

`predictions.csv` preserves the exact user prompt, SMILES, target, parsed prediction, and unedited visible answer. QA answers are included in the same table. `metadata.json` gives the source hashes, shared system prompts, units, coverage, and final metrics.

| Task | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 70 | 20.41714 | 28.84507 | 0.947424 |
| Crystalline band gap (eV) | 48 | 0.986613 | 1.204328 | 0.671395 |
| Atomization energy (eV/atom) | 60 | 1.385500 | 1.884778 | −26.826465 |

Heat-resistance classification: **42/54 correct (77.78%)**, macro-F1 **0.738095**, balanced accuracy **69.44%**. These classes derive from polyimide Tg, not decomposition temperature.

The supplementary AE prediction-range filter [−6.82, −5.01] eV/atom retains **17/60 (28.33%)**: MAE 1.096471, RMSE 1.099909, R² −93.063008. This subset does not replace the unfiltered result; its R² uses its own target variance. The training-label range is not a universal physical bound.

Recompute every metric from the single table with Python's standard library:

```console
python recompute_metrics.py predictions.csv --check metadata.json
```

The script re-parses each visible output, checks the stored prediction, retains the complete task denominators, and verifies the recomputed values against the metadata. Parsing coverage is not the historical NHscore. The added prompt instructions mean historical scores are not claimed to be identical-prompt reruns. Model names are provider identifiers, not independently verified model weights.

Earlier budget calibration and the author-cancelled second-model branch remain in the local archive. They are excluded from this final prediction table. Detailed process logs and per-request files are retained locally rather than included in this release.
