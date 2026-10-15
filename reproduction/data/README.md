# Data reproduction tools

The released data, source attributions, counts, limitations and commands are in
[`dataset/polyopus/README.md`](../../dataset/polyopus/README.md).

- `evaluate_saved_predictions.py`: Python standard library; recomputes the three
  reported regression tasks, the original classification results and AE filtering
  denominators from public flags. The optional `--full-regression` accepts a local
  complete original to verify the exact subset and recompute archived full metrics.
  Full original and excluded prediction values are not included in the public package.
- `reconstruct_augmentation.py`: Python 3.12 and the pinned RDKit requirement;
  recreates saved-order training augmentation evidence and parent/group registries.
- `pack_training_sources.py`: Python standard library; rebuilds the deterministic
  archive of the 20 auxiliary training CSVs from the source checksum manifest.
- `verify_release.py`: Python standard library; checks the publication allowlist,
  SHA256, byte counts, CSV row counts, decompressed lineage and all archive members.

Every path is supplied explicitly on the command line. Source data are read-only.
Use a fresh output directory/file for each run. The main training CSV preserves
the duplicated prompt files and erroneous classification `eV` suffixes exactly.
