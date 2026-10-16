# PolyOpus generation and GNN assessment

PolyOpus generated two candidates at a nominal 6.5 eV target using FP16 CPU weights,
one call each for seeds 17 and 29. The historical PolySea candidate and both outputs
were assessed with a GraphConv model trained for this experiment. Each generation
was used as returned, without retries or repairs.

`final_predictions.csv` contains every train, validation, test and candidate
prediction. The release also includes the final checkpoint (`ckpt-2`),
`artifact.json` for model loading, `metadata.json` with the protocol, generation
text, software, model hashes, metrics and sources, and replay code.

| Partition | Rows | MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| train | 2,704 | 0.924229 | 1.092933 | 0.508667 |
| valid | 338 | 0.967516 | 1.178856 | 0.455147 |
| test | 338 | **0.926901** | 1.107326 | 0.490711 |

| Candidate | GNN (eV) | Absolute distance from 6.5 eV | Training graph overlap |
|---|---:|---:|---|
| historical PolySea | 5.759317 | 0.740683 | None |
| seed 17 | 5.372900 | 1.127100 | Source line 108; label 6.066 eV |
| seed 29 | 5.990445 | 0.509555 | Source line 578; label 6.7775 eV |

Both new candidates match canonical graphs in the GNN training partition;
their reference labels are existing single-chain data. These three case assessments
do not estimate a generation success rate.

## Reproduce

Use Python 3.11 and the package versions in `requirements.txt`. For CPU PyTorch,
use its [official index](https://download.pytorch.org/whl/cpu) for version 2.14.1.
From the repository root:

```bash
python reproduction/gnn/revision_run_20261015/replay.py verify
python reproduction/gnn/revision_run_20261015/replay.py predict --output-dir gnn_replay_new
```

`verify` uses the standard library to check hashes and recompute all split metrics.
`predict` reloads the checkpoint and compares the three candidate values
(rtol 1e-5, atol 1e-6 eV). Use a new or empty output directory outside the release;
on Windows, use an ASCII path for TensorFlow 2.15 checkpoint I/O.

The model used the 3,380-row [TransPolymer Egc.csv](../README.md#data-and-model)
single-chain dataset. The final table includes its reference values and exact splits.

```bash
python reproduction/gnn/train_graphconv.py train --csv Egc.csv --smiles-column smiles --target-column value --wildcard-policy preserve --seed 42 --epochs 10 --batch-size 100 --out gnn_train_new
```

The fixed protocol used grouped 80/10/10 splits (seed 42), layers 64/64, dense 128,
batch 100, learning rate 0.001, ten epochs, preserved wildcard atoms, and no target
normalization. The final model was not selected using validation/test scores.

With local merged [PolyOpus](https://huggingface.co/hkqiu/PolyOpus) weights matching
the hashes in metadata, replay generation with:

```bash
python reproduction/gnn/revision_run_20261015/replay.py generate --model-dir MODEL_DIRECTORY --output-dir polyopus_generation_new
```

The bare prompt is `Generate / design a polymer with band gap of 6.5 eV?`, without
a chat template. Settings: CPU FP16, no quantization, seeds 17/29, maximum 512 new
tokens, temperature 1.2, top-p 0.9, top-k 100, sampling enabled, stop `<STOP>`.
Sampled text may vary with software and hardware.
