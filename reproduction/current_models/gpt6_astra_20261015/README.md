# GPT-6-Astra: final predictions

`predictions.csv` contains all **235 cases**: the three original SI questions and
232 TestSet1 property rows. All calls completed and all property outputs were
parseable. Settings: 8192 output tokens, low reasoning, no tools, retrieval or
cross-case history.

The table preserves exact prompts, SMILES, targets, parsed predictions and unedited
visible answers, including QA. `metadata.json` contains settings,
source hashes, shared system prompts, units, coverage and final metrics.

| Task | n | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 70 | 20.41714 | 28.84507 | 0.947424 |
| Crystalline band gap (eV) | 48 | 0.986613 | 1.204328 | 0.671395 |
| Atomization energy (eV/atom) | 60 | 1.385500 | 1.884778 | −26.826465 |

Tg-derived polyimide heat-resistance classification: **42/54 correct (77.78%)**,
macro-F1 **0.738095**, balanced accuracy **69.44%**.

Recompute every metric with Python's standard library:

```console
python recompute_metrics.py predictions.csv --check metadata.json
```
