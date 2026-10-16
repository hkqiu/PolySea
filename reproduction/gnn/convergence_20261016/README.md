# GraphConv convergence study

The validation-selected checkpoint uses training-label standardization and epoch 259. On the unchanged 338-record test set, MAE is 0.35805851 eV, RMSE 0.49335335 eV and R² 0.89890499. The ten-epoch baseline has test R² 0.49071098. Detailed split results and candidate values are in `summary.json`.

Use Python 3.11 and install `requirements.txt`. The relative input `data/split_manifest.csv` contains all 3,380 fixed source records, exact graph identifiers, model SMILES, eV targets and the unchanged train/valid/test assignments. It is the same 2,704/338/338 split used by the ten-epoch baseline. `baseline_10epoch` contains the original model configuration, checkpoint and reference metrics. No original local filesystem paths or execution logs are required.

Run commands from this directory, or replace the script names with their full paths:

```sh
python predict_selected.py --model-dir selected_model --csv candidates_input.csv --out candidate_predictions_new.csv
python train_convergence.py --variant standardized_from_scratch --out runs/standardized_replay
python train_convergence.py --variant continued_unscaled --out runs/unscaled_replay
python train_convergence.py --variant both --out runs/both_replay
```

The default `--baseline-dir` and `--split-manifest` resolve relative to the script, not the shell working directory. Both can be overridden explicitly. `--out` must be new or empty. On Windows, use an ASCII output directory for TensorFlow checkpoint writing. The program records all settings and row predictions under the output directory. It does not modify the published checkpoint.

The primary exact training recipe for the selected model is `--variant standardized_from_scratch --max-epochs 300 --min-stop-epoch 100 --patience 50`. Target mean 4.545666420118343 eV and population SD 1.5592129332106395 eV are fitted using training labels only. Every epoch is evaluated in original eV units. The best checkpoint is the minimum validation RMSE; early-stop patience uses a 0.0001 eV improvement tolerance. Test outputs are computed only after `selection.json` is written. Reproduction values can differ slightly across hardware and TensorFlow versions; the released checkpoint supplies the exact archived inference result.

The reported study compared two variants, both trained to 300 total epochs. The unscaled continuation selected epoch 295 (validation RMSE 0.51588501 eV); the standardized run selected epoch 259 (0.51259238 eV). Only the latter was evaluated on the test set for the new result. The original unscaled process had a file-sharing interruption after epoch 246 and resumed its saved epoch-230 checkpoint with the traversal RNG reset to 42. Both numerical curves are retained in `learning_curves`; a fresh uninterrupted unscaled replay may follow a different training trajectory. The selected standardized run was uninterrupted.

`predict_selected.py` applies the inverse target transform stored in the artifact. Do not use an older inference script that omits this transform. Checkpoint reload and a separate CLI replay reproduced all three candidate outputs with maximum absolute difference 0 eV. Exact graph matches of the two new generated candidates to training records remain marked in `selected_model/candidate_results.json`. These GNN results are model estimates, not new DFT or experimental measurements.

The checkpoint sidecars and data/prediction files are sealed in `sha256_manifest.json`.
