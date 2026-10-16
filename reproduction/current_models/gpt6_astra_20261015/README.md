# GPT-6-Astra: final predictions

This release contains one prediction table with all **235 formal cases**: the three original SI questions and all 232 TestSet1 property rows. The 235 calls completed successfully; all 232 property outputs were parseable. Parameters were fixed at 8192 output tokens and low reasoning, with no tools, retrieval, or cross-case history.

`predictions.csv` preserves the exact user prompt, SMILES, target, parsed prediction, and unedited visible answer. QA answers are included in the same table. `metadata.json` gives the source hashes, shared system prompts, units, coverage, and final metrics.

| Task | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 70 | 20.41714 | 28.84507 | 0.947424 |
| Crystalline band gap (eV) | 48 | 0.986613 | 1.204328 | 0.671395 |
| Atomization energy (eV/atom) | 60 | 1.385500 | 1.884778 | −26.826465 |

Heat-resistance classification: **42/54 correct (77.78%)**, macro-F1 **0.738095**, balanced accuracy **69.44%**. These classes derive from polyimide Tg, not decomposition temperature.

Recompute every metric from the single table with Python's standard library:

```console
python recompute_metrics.py predictions.csv --check metadata.json
```
