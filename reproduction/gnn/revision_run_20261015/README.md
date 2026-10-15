# PolyOpus generation and GNN assessment

This revision experiment generated two PolyOpus candidates at a nominal 6.5 eV
target using original FP16 CPU weights, one call each for seeds 17 and 29. The
historical selected candidate and both new outputs were assessed with one newly
trained GraphConv model. No generated candidate was retried or repaired.

The compact release has one `final_predictions.csv` containing every train,
validation, test and candidate prediction; one final checkpoint (`ckpt-2`);
`artifact.json` for model loading; `metadata.json` for protocol, visible generation
text, software, model hashes, metrics and source information; and replay code.
Process logs, earlier checkpoints and redundant prediction tables are local only.

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

Both new candidates match canonical graphs in the GNN training partition. Their
assessments are not independent of those training structures. Existing source
labels are not new DFT values. No new DFT or Polymer Genome calculation was made;
this experiment does not resolve DFT functional sensitivity. Two new calls and
one historically selected example do not establish a reliable success rate.

## Reproduce

Use Python 3.11 and the package versions in `requirements.txt`. For CPU PyTorch,
use its [official index](https://download.pytorch.org/whl/cpu) for version 2.14.1.
From the repository root:

```bash
python reproduction/gnn/revision_run_20261015/replay.py verify
python reproduction/gnn/revision_run_20261015/replay.py predict --output-dir gnn_replay_new
```

`verify` needs only the standard library: it checks hashes and recomputes every
split metric from the single final table. `predict` reloads the final checkpoint
once and compares all three candidate values (rtol 1e-5, atol 1e-6 eV). On Windows,
use an ASCII output path for TensorFlow 2.15 checkpoint I/O. Outputs must be new or
empty and outside the release folder. No retraining occurs in prediction mode.

The GNN used [TransPolymer Egc.csv at commit 8399d48](https://media.githubusercontent.com/media/ChangwenXu98/TransPolymer/8399d4816ce772b64deba34f4455d91d9a764b2a/data/Egc.csv),
SHA-256 `d779e1777e7f25bf367dae2a2bf72659dba26345eb45dbc2cde73cdd461a9a14`.
This is **single-chain band gap**, from [Kuenneth et al.](https://doi.org/10.1016/j.patter.2021.100238),
distinct from the crystalline-gap endpoint in the PolyOpus evaluation corpus.
The final table includes all reference values and exact split assignments.

```bash
python reproduction/gnn/train_graphconv.py train --csv Egc.csv --smiles-column smiles --target-column value --wildcard-policy preserve --seed 42 --epochs 10 --batch-size 100 --out gnn_train_new
```

The fixed protocol used grouped 80/10/10 splits (seed 42), layers 64/64, dense 128,
batch 100, learning rate 0.001, ten epochs, preserved wildcard atoms, and no target
normalization. The final model was not selected using validation/test scores.
Canonical graph identity does not prove periodic-polymer equivalence. This is a
new revision checkpoint; missing historical GNN artifacts were not recovered.

With a local merged [PolyOpus](https://huggingface.co/hkqiu/PolyOpus) model matching
the hashes in metadata, a separate generation replay can be run:

```bash
python reproduction/gnn/revision_run_20261015/replay.py generate --model-dir MODEL_DIRECTORY --output-dir polyopus_generation_new
```

The bare prompt is `Generate / design a polymer with band gap of 6.5 eV?`, without
a chat template. Settings: CPU FP16, no quantization, seeds 17/29, maximum 512 new
tokens, temperature 1.2, top-p 0.9, top-k 100, sampling enabled, stop `<STOP>`.
Software/hardware differences may change sampled text; new outputs are separate
replay evidence and do not replace the released final observations.
