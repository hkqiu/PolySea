# Polymer single-chain band-gap GraphConv model

We extended GraphConv training on the unchanged 2,704/338/338 train/validation/test split and selected the training-label-standardized checkpoint at epoch 259 by validation RMSE. Test MAE is 0.35806 eV and R² is 0.89890.

The [convergence study](convergence_20261016/README.md) provides the new checkpoint, learning curves, all split predictions and candidate results. From `reproduction/gnn/convergence_20261016`, run:

```bash
python train_convergence.py --variant standardized_from_scratch --out runs/standardized_replay
python predict_selected.py --model-dir selected_model --csv candidates_input.csv --out candidate_predictions_new.csv
```

The original ten-epoch checkpoint and CPU-generated candidates remain in [revision_run_20261015](revision_run_20261015/README.md). The sections below document that baseline training script.

## Data and model

The data source is Kuenneth et al., *Polymer informatics with multi-task learning*,
Patterns **2**, 100238 (2021), DOI [10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238).
The [TransPolymer Egc.csv copy](https://media.githubusercontent.com/media/ChangwenXu98/TransPolymer/8399d4816ce772b64deba34f4455d91d9a764b2a/data/Egc.csv)
contains 3,380 rows (`smiles`, `value`), with SHA-256
`d779e1777e7f25bf367dae2a2bf72659dba26345eb45dbc2cde73cdd461a9a14`.
Egc is single-chain band gap in eV; the PolyOpus evaluation uses crystalline band gap.
Other source CSVs can be supplied with explicit column names.

Use Python 3.11 in a separate environment and install `requirements.txt`.
DeepChem 2.8.0 settings are graph-convolution widths 64/64, dense width 128,
dropout 0, batch size 100, batch normalization, 75 ConvMol atom features,
Adam learning rate 0.001, and 10 epochs. The model uses `n_tasks=1`,
`mode="regression"`, and `uncertainty=False`; outputs are point predictions.
See the [DeepChem 2.8.0 API](https://deepchem.readthedocs.io/en/2.8.0/api_reference/models.html#graphconvmodel).

## Run

From `reproduction/gnn`:

```bash
python train_graphconv.py train --csv Egc.csv --smiles-column smiles --target-column value --wildcard-policy preserve --out run_chain
python train_graphconv.py predict --csv candidates.csv --smiles-column SMILES --model-dir run_chain --out candidate_predictions.csv
```

`--wildcard-policy preserve` passes the parsed repeat-unit graph, including `*`,
to ConvMol. `hydrogen_cap` replaces degree-one, single-bonded attachment markers
with hydrogen. These policies operate on molecular graphs, without constructing
periodic crystals.

The seeded 80/10/10 split groups equivalent molecular graphs, equivalent
preprocessed model inputs, and optional source IDs. Use `--split-column split`
for an existing train/valid/test assignment; cross-split graph/ID overlap is rejected.
Conflicting labels remain separate records within their group. Labels are neither
averaged nor normalized, and outliers are retained. Missing labels raise an error
unless `--missing-target drop` is specified; dropped lines are logged.

The fixed training budget is independent of test labels. Outputs include the split
manifest, configuration, input hash, software versions, MAE/RMSE/R², all split
predictions, and a TensorFlow checkpoint. A save/reload check compares test
predictions. `--prepare-only` audits inputs and writes the split manifest using
RDKit, without training. Use an empty output directory for each run.
