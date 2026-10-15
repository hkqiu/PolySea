# PolyOpus training data and archived evaluation outputs

This directory preserves the data used by the PolyOpus study and discloses the
subsequent revision analyses. The local archival date is **2026-10-15**. Original
upstream acquisition dates have not been recovered; this archival date is not an
upstream download date. Counts below are CSV data records, excluding headers.
Augmented instruction rows are not counts of independent polymers.

## Files and original bytes

| File | Records | Role |
| --- | ---: | --- |
| `train/merged_train.csv` | 233827 | Exact original training CSV; all values and labels preserved |
| `train/augmentation_source_files.tar.gz` | 20 CSV files | The other saved training files, each with its exact original bytes |
| `test/merged_test.csv` | 23200 | Original held-out instruction pool (TestSet2) |
| `test/benchmark.csv` | 232 | Original smaller comparison input set (TestSet1) |
| `predictions/raw/polysea_gen_v0327_cls.csv` | 5372 | Original saved heat-resistance classification outputs |
| `predictions/reported/polysea_gen_v0331_reg_reported_AE_training_range.csv` | 13606 | All 7000 Tg and 4800 band-gap rows, plus 1806 AE rows meeting the disclosed rule |
| `predictions/reported/regression_row_filter_flags.csv` | 17595 | Source ordinal, task and retained flag only; no excluded prompts or values |
| `predictions/reported/archived_full_regression_summary.json` | — | Complete-original SHA256 and aggregate metrics; no full per-row values |

The raw classification CSV is **GB18030**, preserved without transcoding. The
main training/test CSVs are UTF-8 with optional BOM, read as `utf-8-sig`. Individual
training archive encodings, columns, row counts and SHA256 hashes are listed in
[`augmentation/source_file_manifest.json`](augmentation/source_file_manifest.json).
Derived reporting CSVs use UTF-8 with BOM. Their original cell strings and row
order are retained; their encoding/serialization intentionally differs from the
locally retained GB18030 full regression file. The full original regression and
the 3989 excluded prediction records are **retained locally and not distributed
in this package**. [`release_manifest.json`](release_manifest.json) lists every release
file's SHA256 and byte count, CSV row counts, decompressed lineage hash, and all
archive members. The manifest excludes itself to avoid a circular checksum.
The repository's `.gitattributes` disables Git newline conversion for these
released files so their checksums remain valid after checkout on another platform.

## AE reporting rule and complete denominators

The reported AE subset is a **post-hoc sensitivity analysis requested after the
full-output metrics were known**. It keeps a saved AE prediction only when the
`PolySea` value lies within the inclusive minimum/maximum of the archived AE
training labels, **[-6.82, -5.01] eV/atom** (52650 training instruction rows).
The rule does not select on test targets or residuals. It does not establish a
physical bound and must not be interpreted as an unfiltered test-set result.
No clipping, sign correction, imputation, IQR filtering or target correction is
applied. Tg and crystalline band-gap rows are all retained.

AE has 5850 original inputs, 5795 saved outputs, 55 missing saved outputs, 1806
retained outputs and 3989 excluded outputs. The retained subset covers **31.1648%
of saved outputs and 30.8718% of original inputs**. Missing-output causes cannot be
recovered from these saved CSVs alone. Full regression values are retained in the
authors' local archive; only their checksum and aggregate summary are public.
All decisions are linked by `source_data_row`, a 1-based
CSV data-record ordinal excluding the header, not a physical text-line number.

Public reported metrics are recalculated from `target` and `PolySea`; the preserved
`error` column is not used. In the complete local original, that column differs
from the recalculated absolute residual in 139 Tg rows and 27 AE rows, as recorded
in the public aggregate summary. Original stored values have not been repaired.

| Saved-output evaluation | Scored rows | MAE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Tg (°C), all saved rows | 7000 | 24.579579 | 35.744916 | 0.926128 |
| Crystalline band gap (eV), all saved rows | 4800 | 0.701811 | 0.962368 | 0.790170 |
| AE (eV/atom), archived full-output summary only | 5795 | 6.748143 | 8.425614 | -633.066105 |
| AE (eV/atom), disclosed retained subset | 1806 | 0.062148 | 0.075725 | 0.979460 |

The full AE metrics above are an archived summary, not recomputable from the
public subset alone. Recomputing them requires the complete local original via
the script's optional `--full-regression` argument. R² uses target variance within
the corresponding scored subset. Metrics weight
instruction rows equally and do not pool different regression properties/units.
The original classification file contains 5372 of 5550 inputs (96.7928% coverage;
178 missing saved outputs): accuracy **0.786485**, macro-F1 **0.686666**, weighted-F1
**0.775129**. Class labels describe Tg-derived polyimide heat resistance; they are
not decomposition temperature or a separate thermal-degradation measurement.
The complete per-class scores and confusion matrix are in
[`predictions/reported/metrics.json`](predictions/reported/metrics.json).

## Training augmentation evidence

[`augmentation/reconstruction_summary.json`](augmentation/reconstruction_summary.json)
reconstructs saved-order blocks, component assembly and multiplicities from the
21 archived training CSVs. It does not infer historical code execution or recover
original measurement identifiers. Core blocks have 10 Tg, 20 crystalline-gap,
15 AE or 15 classification SMILES rows per prompt file, across ten prompt files.
`train1.csv` and `train5.csv` are exact duplicates: there are four distinct English
and five Chinese templates per core task. Their identical merged segments cannot
be assigned uniquely to those two filenames. The original duplication is retained.
**9990 classification targets erroneously carry an `eV` suffix** in the merged
training file; those raw labels are retained. Label normalization in reconstruction
is only for grouping and never rewrites training data.

The `augmentation/` files provide:

- `parent_registry.csv`, `core_parent_blocks.csv`, and `inverse_parent_rows.csv`:
  reconstructed parent surrogates and their basis; these are not verified original
  measurement IDs.
- `graph_label_registry.csv` and `graph_label_counts.csv`: canonical molecular
  graph + task + normalized-label groups, with conflicting labels kept separate.
- `multiplicity_distributions.csv`, `core_blocks_per_graph_label.csv`,
  `file_task_summary.csv`, and `prompt_language_counts.csv`: exact-record, SMILES,
  canonical-graph and graph-label multiplicities and language/template counts.
- `merged_ordered_segments.csv`, `merged_target_suffixes.csv`, and
  `merged_row_lineage.csv.gz`: ordered assembly and lineage for all 233827 rows;
  **4291 QA/other rows have unresolved original component sources**.

Graph identity uses RDKit sanitized canonical isomeric molecular graphs. It does
not establish periodic repeat-unit equivalence, normalize tautomers, strip salts,
or remove stereochemistry. Reconstruction alone does not establish original
measurement counts, temporal split order, or absence of train/test overlap.

## Reproduction

Run the standard-library verifier from the repository root:

```bash
python reproduction/data/verify_release.py --repo-root .
python reproduction/data/evaluate_saved_predictions.py --data-dir dataset/polyopus --output-dir reproduced_metrics
```

The evaluator reads the public reported CSV and flags, recomputes the reported
regression and original classification metrics, and writes deterministic
`metrics.json` and `evaluation_manifest.json`. Its archived full-output aggregate
is explicitly marked as not recomputed. Compare these two output files with
`predictions/reported/`. A holder of the complete original may additionally run:

```bash
python reproduction/data/evaluate_saved_predictions.py --data-dir dataset/polyopus --output-dir checked_with_local_full --full-regression /path/to/local/polysea_gen_v0331_reg.csv
```

That optional mode checks the original SHA256, every flag, the exact reported
subset, and the archived full metrics. It does not publish excluded values.
Reconstruct augmentation in an isolated Python 3.12
environment with `reproduction/data/requirements.txt` (RDKit 2026.9.1):

```bash
mkdir reconstructed_train
tar -xzf dataset/polyopus/train/augmentation_source_files.tar.gz -C reconstructed_train
cp dataset/polyopus/train/merged_train.csv reconstructed_train/merged_train.csv
python reproduction/data/reconstruct_augmentation.py --train-dir reconstructed_train --output-dir reconstructed_augmentation
python reproduction/data/pack_training_sources.py --train-dir reconstructed_train --manifest dataset/polyopus/augmentation/source_file_manifest.json --output rebuilt_training_sources.tar.gz
```

On Windows, `Copy-Item` can replace `cp`. The archive packer fixes tar member order,
permissions, ownership and timestamps and sets gzip `mtime=0` with no filename;
the rebuilt archive can be compared byte-for-byte. Reconstruction writes lineage
as an uncompressed CSV; its bytes match the decompressed released `.csv.gz`.
Output directories must be new or empty, so archived results are not overwritten.

The separate [`reproduction/gnn`](../../reproduction/gnn/README.md) script trains a
DeepChem GraphConv single-chain band-gap model from an explicitly supplied input.
It is revision-era reproducibility code with explicit split/preprocessing choices.
It does not recover the missing historical weights, split, package versions or
the original candidate predictions. Its Python 3.11 environment is separate from
the augmentation-reconstruction environment. No model weights are bundled here.

## Source attribution and rights

The property datasets were reused through **PolyNC**, DOI
[10.1039/D3SC05079C](https://doi.org/10.1039/D3SC05079C), and its published
[data/code repository](https://github.com/HKQiu/Unified_ML4Polymers). This release
does not represent a new automated acquisition from PolyInfo. The relevant
upstream compilation is Kuenneth et al., *Polymer informatics with multi-task
learning*, Patterns 2, 100238 (2021), DOI
[10.1016/j.patter.2021.100238](https://doi.org/10.1016/j.patter.2021.100238).
It identifies DFT atomization energies in eV/atom and distinguishes single-chain
and bulk/crystalline band gaps. PolyNC's reported source counts and the recovered
PolyOpus instruction-row counts are different quantities. The exact upstream
source-row mapping for the 236-record crystalline-gap subset has not been
recovered; the larger chain/bulk datasets are not substitutes for that mapping.

The repository's Apache-2.0 license applies to its source code as described in the
root LICENSE. It **does not assert that every upstream dataset's redistribution
terms have been independently verified or that all upstream data are relicensed
under Apache-2.0**. Upstream rights and required attribution remain with their
respective owners and publications. Original acquisition records and a complete
row-level provenance/license chain have not been recovered. The SHA256 manifests
establish the identity of this archived release, not ownership or a blanket
permission claim.
