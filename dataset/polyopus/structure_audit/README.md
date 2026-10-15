# Existing split and duplicate audit

This analysis audits the original PolyOpus training/test files and rescores archived
predictions. It performs no new model training or inference and does not alter the
original split. One table contains the 149 final representative predictions.

| Endpoint | Training structures | Test structures | Same-task overlap | Test structures after removing overlap |
|---|---:|---:|---:|---:|
| Tg | 7,624 | 70 | 45 | 25 |
| Crystalline gap | 212 | 24 | 0 | 24 |
| Atomization energy | 351 | 39 | 0 | 39 |
| Heat-resistance class | 287 | 37 | 10 | 27 |

These counts use the declared structural identifiers. `canonicalize_psmiles`
normalizes eligible two-attachment strings; other encodings retain separately
labelled RDKit isomeric molecular-graph identifiers. One invalid Tg training row
is retained in the source-row count but has no structural identifier. The package
does not fully reduce all branched or ring repeat multiplication; these are
operational normalized-structure counts, not a universal proof of polymer identity.
Every same-task overlap pair was independently checked; non-identical monomer
graphs were checked by agreement of cyclic oligomers at two sizes.

For each endpoint and stated reporting population, the representative is the first
saved source-row prediction for each structural identifier. Neither target error
nor prediction quality chooses the representative. Original test labels are
constant within each identifier. This is a single-representative sensitivity
analysis, not an average-prediction ensemble and not a replacement for row-level
metrics. Reported AE representatives come from the previously documented AE rule:
18 structures among its 1,806 prediction records. Source split counts above still
describe the 39 structures in the original AE test input.

| Endpoint | Representatives | MAE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tg (°C) | 70 | 46.61786 | 62.25919 | 0.77589 |
| Crystalline gap (eV) | 24 | 1.93760 | 2.52011 | -0.43888 |
| AE (eV/atom) | 18 | 0.06444 | 0.07638 | 0.97141 |

Classification: 23/37 correct (62.16%), macro-F1 0.36144. After excluding same-task
training overlaps, Tg has n=25 and MAE 50.216 °C; classification has n=27 and
18 correct (66.67%). Gap and AE reporting representatives do not change under this
same-task exclusion. The final table also flags matches to all structured training
fields, including other property tasks and inverse-design targets. Such matches
indicate structure exposure, not necessarily exposure to the same target label.
Natural-language QA contents and earlier base-model pretraining are not audited.

## Reproduce

Install RDKit 2026.09.1 and the pinned canonicalizer in a separate environment:

```bash
pip install rdkit==2026.9.1
pip install git+https://github.com/Ramprasad-Group/canonicalize_psmiles.git@729dfd20b0e909df7cb245ffbfa5ab2187ffbf8b
python reproduction/data/audit_structure_split.py --output-dir structure_check
```

The canonicalizer is an external dependency with its own GTRC license; it is not
vendored or relicensed by this repository. This academic analysis follows that
package's documented scope. The script verifies its normalized source hash.
`--canonicalizer-dir` can instead point to a checkout of that exact revision.

The script rebuilds normalized identities, overlap flags and the final prediction
table from the public training/test and reporting CSVs. Generated caches and
registry files go into the requested local output folder, not this release.
`summary.json` records the conventions, denominators, full-precision metrics and
the local full-AE aggregate separately. The excluded original AE per-row outputs
remain in the authors' local archive.
