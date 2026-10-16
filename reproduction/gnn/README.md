# Polymer single-chain band-gap GraphConv model

Our original validator used DeepChem `GraphConvModel`, single-chain band-gap labels,
and default training parameters. We wrote this script during revision. We have not
recovered the historical weights, exact split, DeepChem version, or attachment-point
preprocessing, so this script does not reproduce the three published candidate
predictions.

The data source is Kuenneth et al., *Polymer informatics with multi-task learning*,
Patterns **2**, 100238 (2021), DOI [10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238).
The paper distinguishes single-chain and crystal band gaps. Obtain the permitted
single-chain data from the paper's linked [Khazana archive](https://khazana.gatech.edu/)
or a retained source file. The script does not substitute PolyOpus
crystal-band-gap labels or generated-polymer labels. CSV column selection is
explicit, and the input checksum is saved with every run.

Use Python 3.11 in a separate environment and install `requirements.txt`.
DeepChem 2.8.0 documents `[64,64]` graph-convolution widths, dense width 128,
dropout 0, batch size 100, batch normalization enabled and 75 ConvMol atom features.
The Keras wrapper defaults to Adam with learning rate 0.001; `fit` defaults to
10 epochs. The script sets `n_tasks=1`, `mode="regression"`, and
`uncertainty=False`. These are version-pinned defaults, not a recovered record of
the original installed version. See the [DeepChem 2.8.0 API](https://deepchem.readthedocs.io/en/2.8.0/api_reference/models.html#graphconvmodel).

```bash
python train_graphconv.py train --csv Egc.csv --smiles-column smiles --target-column value --wildcard-policy preserve --out run_chain
python train_graphconv.py predict --csv candidates.csv --smiles-column SMILES --model-dir run_chain --out candidate_predictions.csv
```

For an accessible example, the [TransPolymer public copy](https://media.githubusercontent.com/media/ChangwenXu98/TransPolymer/8399d4816ce772b64deba34f4455d91d9a764b2a/data/Egc.csv)
contains 3,380 rows with columns `smiles` and `value`. Its SHA256 is
`d779e1777e7f25bf367dae2a2bf72659dba26345eb45dbc2cde73cdd461a9a14`.
The source paper defines Egc as single-chain band gap in eV and Egb as bulk band gap.
This public copy was used for the [revision experiment](revision_run_20261015/README.md),
which provides the final checkpoint and one combined prediction table. It does
not reproduce our historical GNN training data selection. An archived
input file can instead be supplied with its actual column names.

The required wildcard policy makes preprocessing reviewable. `preserve` passes
the parsed repeat-unit graph, including `*`, to ConvMol. `hydrogen_cap` replaces
only degree-one, single-bonded attachment markers by hydrogen. Neither constructs
a periodic crystal or establishes equivalence under moving the repeat-unit boundary.
The original attachment-point preprocessing remains unverified; neither choice is
asserted to reproduce it.

When no split column is provided, the **new script** uses a seeded 80/10/10 split
over groups, tying equivalent molecular graphs, equivalent model inputs after
preprocessing, and optional source IDs together. This is a new reproducibility
choice, not a claim about the historical split. Supply `--split-column split` to
use an archived train/valid/test assignment; cross-split graph/ID overlap causes an
error. Conflicting measured values remain separate records within their group.
No label averaging, target normalization or outlier removal occurs. Missing labels
cause an error unless `--missing-target drop` is specified; dropped lines are logged.

The fixed training budget does not select a checkpoint using test labels. Outputs
include the complete split manifest, configuration, CSV hash, software versions,
MAE/RMSE/R² and every prediction for each split, plus a TensorFlow checkpoint.
A save/reload check compares the two sets of test predictions. `--prepare-only`
audits inputs and creates the split manifest without TensorFlow training; it requires
RDKit. Use an empty output directory for each run.

No quantitative uncertainty or applicability guarantee is supplied by this default
regression model. Source data, the original trained checkpoint, preprocessing and
split files are still needed to reproduce the historical design example.
