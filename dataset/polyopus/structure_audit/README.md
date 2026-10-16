# Existing split and duplicate audit

Structural overlap and fixed-representative metrics for the original PolyOpus
training/test split, using saved predictions. `final_predictions.csv` contains
149 representative predictions.

| Endpoint | Training structures | Test structures | Same-task overlap | Test structures after removing overlap |
|---|---:|---:|---:|---:|
| Tg | 7,624 | 70 | 45 | 25 |
| Crystalline gap | 212 | 24 | 0 | 24 |
| Atomization energy | 351 | 39 | 0 | 39 |
| Heat-resistance class | 287 | 37 | 10 | 27 |

`canonicalize_psmiles` normalizes eligible two-attachment strings; other encodings
use separately labelled RDKit isomeric molecular-graph identifiers. One invalid
Tg training row has no identifier. These counts use the stated normalization,
which does not fully reduce branched or ring repeat multiplication. All same-task
overlap pairs were checked independently; non-identical monomer graphs were
checked by agreement of cyclic oligomers at two sizes.

For each endpoint and reporting population, the representative is the first saved
source-row prediction for each identifier, selected independently of prediction
error. Test labels are constant within an identifier. AE uses 18 structures from
the [1,806-row reporting subset](../README.md#reported-evaluation); the source
split counts above cover all 39 AE test structures.

| Endpoint | Representatives | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 70 | 46.61786 | 62.25919 | 0.77589 |
| Crystalline gap (eV) | 24 | 1.93760 | 2.52011 | -0.43888 |
| AE (eV/atom) | 18 | 0.06444 | 0.07638 | 0.97141 |

Classification: 23/37 correct (62.16%), macro-F1 0.36144. Excluding same-task
training overlaps gives Tg n=25, MAE 50.216 °C, and classification n=27,
18 correct (66.67%); gap and AE representatives are unchanged.
The table also flags matches across structured training fields, including other
property tasks and inverse-design targets. These indicate structure exposure;
same-target exposure is reported separately. Natural-language QA and base-model
pretraining are outside this structural audit.

## Reproduce

Install RDKit 2026.09.1 and the pinned canonicalizer in a separate environment:

```bash
pip install rdkit==2026.9.1
pip install git+https://github.com/Ramprasad-Group/canonicalize_psmiles.git@729dfd20b0e909df7cb245ffbfa5ab2187ffbf8b
python reproduction/data/audit_structure_split.py --output-dir structure_check
```

The external canonicalizer is governed by its GTRC license. The script verifies
its normalized source hash; `--canonicalizer-dir` accepts a checkout of that
revision.

The script rebuilds identities, overlap flags and representative predictions from
the public CSVs. Results and caches are written to the requested output directory.
`summary.json` records conventions, denominators, full-precision metrics and a
separate full-AE aggregate from the original archive; excluded AE per-row outputs
are not part of the public table.
