# Revision experiment: two PolyOpus samples and one reconstructed GNN

This capsule contains the real CPU experiment completed on 2026-10-15. It is a new revision experiment, not a reconstruction of the missing historical GNN checkpoint. Two additional PolyOpus candidates were generated at a nominal 6.5 eV target, with one formal call for seed 17 and one for seed 29. The previously selected PolySea example was then evaluated alongside both new candidates using the same newly trained GNN.

The released GNN used all 2,704 training rows, 338 validation rows and 338 test rows from the 3,380-row single-chain band-gap dataset. The fixed protocol used seed 42, graph convolution widths 64/64, dense width 128, batch size 100, Adam learning rate 0.001 and 10 epochs. Wildcard attachment atoms were preserved. No validation/test score selected a model, training duration or generation output. The saved checkpoint passed reload agreement.

| Split | Rows | MAE (eV) | RMSE (eV) | R² |
|---|---:|---:|---:|---:|
| train | 2,704 | 0.924229 | 1.092933 | 0.508667 |
| valid | 338 | 0.967516 | 1.178856 | 0.455147 |
| test | 338 | **0.926901** | 1.107326 | 0.490711 |

| Candidate | SMILES | New GNN (eV) | Absolute deviation from 6.5 eV | GNN training-data overlap |
|---|---|---:|---:|---|
| historical PolySea | `O=C(CCCCCCCCC(OCCC(O[*])C)=O)OC[*]` | 5.759317 | 0.740683 | None by canonical graph |
| seed 17 | `[*]CC([*])C(=O)OCC=C` | 5.372900 | 1.127100 | CSV line 108; existing label 6.066 eV |
| seed 29 | `[*]CC1CCC(COC(=O)CCCCC(=O)O[*])CC1` | 5.990445 | 0.509555 | CSV line 578; existing label 6.7775 eV |

All three new GNN predictions are below 6.5 eV. Both new candidates exactly match canonical graphs in the GNN training partition, so their GNN assessments are not independent of the training structures. The values 6.066 and 6.7775 eV are labels already present in the source dataset; they are not new DFT calculations. No new DFT or Polymer Genome calculation was performed. The test MAE of 0.9269 eV limits numerical interpretation, and these two prespecified samples plus a selected historical example do not establish a reliable success rate.

## Contents and provenance

- `artifact.json`, `checkpoint/`, `metrics.json`, `split_manifest.csv`, `excluded_rows.json`, and all three `*_predictions.csv` files contain the complete new GNN artifact and evaluation records. Both checkpoint versions written by training are retained; the checkpoint state file selects `ckpt-2`.
- `candidates_input.csv`, `candidates_predictions.csv`, `candidate_results.csv/json`, and `summary.json` retain all three candidates, numerical results, and graph-overlap evidence.
- `generation/seed_17_result.json` and `generation/seed_29_result.json` contain the complete visible input/output text, token IDs, sampling parameters, validation status and timings for the two formal calls. `generation/input.json` preserves the exact tokenized prompt. Both outputs were valid with two wildcard atoms; neither was retried or filtered out.
- `protocol.json`, `environment.json`, `generation/environment.json`, and `polyopus_model_manifest.json` identify the fixed protocol, software and exact merged-model file hashes. PolyOpus weights are not included.
- `verification_report.json` records a separate reload of this public capsule's checkpoint and comparison of all three candidate values. `sha256_manifest.json` is this capsule's independent byte manifest; it does not replace the parent data release manifest.

The GNN CSV is the [TransPolymer `data/Egc.csv` at commit `8399d4816ce772b64deba34f4455d91d9a764b2a`](https://media.githubusercontent.com/media/ChangwenXu98/TransPolymer/8399d4816ce772b64deba34f4455d91d9a764b2a/data/Egc.csv), SHA-256 `d779e1777e7f25bf367dae2a2bf72659dba26345eb45dbc2cde73cdd461a9a14`. The source endpoint is single-chain band gap in eV, described by Kuenneth et al., *Patterns* **2**, 100238 (2021), DOI [10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238). It is distinct from the crystalline band-gap endpoint in the PolyOpus property corpus.

The exact input split is provided. The split identity rule ties canonical molecular/model graphs and optional source IDs; it is not a proof of periodic repeat-unit equivalence. This release includes every prediction row and does not remove outliers.

## Verify and reload

Run the commands below from the repository root. Use Python 3.11 and a separate environment. `requirements.txt` records the primary packages actually used. GNN inference requires DeepChem 2.8.0, TensorFlow 2.15.1/Keras 2.15, NumPy 1.26.4 and RDKit 2023.9.6. The generation run additionally used Transformers 4.49.0, Accelerate 1.3.0, Safetensors 0.5.3 and PyTorch 2.14.1+cpu. For CPU-only PyTorch installation, use its [official package index](https://download.pytorch.org/whl/cpu) for version 2.14.1 before installing the other requirements.

```bash
python -m pip install -r reproduction/gnn/revision_run_20261015/requirements.txt
python reproduction/gnn/revision_run_20261015/replay.py verify
python reproduction/gnn/revision_run_20261015/replay.py predict --output-dir gnn_replay_new
```

`predict` uses the unchanged parent `train_graphconv.py`. It copies only the small checkpoint to the fresh output directory, reloads it once and predicts all three candidates. It saves `candidate_predictions.csv` and `verification.json`; agreement is checked with relative tolerance `1e-5` and absolute tolerance `1e-6` eV. It does not retrain. On Windows, supply an ASCII output path because TensorFlow 2.15 checkpoint I/O failed with a non-ASCII path in the original execution. The original failed attempt was retained in the authors' audit, then the same fixed protocol was restarted using an ASCII path alias; no model was selected by performance.

The existing prediction CLI can also consume the capsule directly when the model path is supported by TensorFlow:

```bash
python reproduction/gnn/train_graphconv.py predict --csv reproduction/gnn/revision_run_20261015/candidates_input.csv --id-column candidate_id --model-dir reproduction/gnn/revision_run_20261015 --out candidate_predictions_new.csv
```

To retrain under the fixed protocol, obtain the source CSV above and check its hash, then run the existing training script with a new output directory:

```bash
python reproduction/gnn/train_graphconv.py train --csv Egc.csv --smiles-column smiles --target-column value --wildcard-policy preserve --seed 42 --epochs 10 --batch-size 100 --out gnn_train_new
```

No standalone GNN source file or previous data release documentation was modified by adding this capsule. The parent training script hash is pinned in `protocol.json`. Numerical reproducibility can depend on the exact software and hardware; any new replay must retain its own results.

## Separate PolyOpus generation replay

Supply a local merged [PolyOpus](https://huggingface.co/hkqiu/PolyOpus) model matching `polyopus_model_manifest.json`. The script checks all model file hashes before loading. It never fetches or publishes weights.

```bash
python reproduction/gnn/revision_run_20261015/replay.py generate --model-dir MODEL_DIRECTORY --output-dir polyopus_generation_new
```

`--model-dir` and `--output-dir` are required for generation. The output directory must be new or empty and outside this capsule. Existing formal samples and replay outputs cannot be overwritten. The fixed CPU replay uses original float16 weights without quantization, the bare source prompt `Generate / design a polymer with band gap of 6.5 eV?`, maximum 512 new tokens, temperature 1.2, top-p 0.9, top-k 100, and `do_sample=true` inherited by the original source from the recovered model configuration. The stop string is `<STOP>`. The tokenizer chat template is not applied.

Seeds 17 and 29 each receive one call. Every result, including invalid or failed generations, is retained; candidates are not chemically repaired or selected from multiple alternatives. Different CPU libraries or package versions can change stochastic outputs despite fixed seeds. Such outputs are separate replay evidence and must not replace the two released formal results.
