"""Train/reload a DeepChem GraphConvModel for polymer single-chain band gaps.

This is a reconstruction for the revision, not the missing historical checkpoint.
Model/fit defaults are pinned to DeepChem 2.8.0. Input preparation and split
choices are explicit because those historical details have not been recovered.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
from typing import Any

SOURCE_DOI = '10.1016/j.patter.2021.100238'
DEFAULTS = dict(graph_conv_layers=[64, 64], dense_layer_size=128,
                dropout=0.0, batch_size=100, learning_rate=0.001,
                batch_normalize=True, uncertainty=False, n_tasks=1,
                mode='regression', number_atom_features=75)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def graph_smiles(value: str, wildcard_policy: str) -> tuple[str, str]:
    from rdkit import Chem
    mol = Chem.MolFromSmiles(value)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty SMILES')
    identity = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if wildcard_policy == 'hydrogen_cap':
        # Only degree-one attachment markers can be capped unambiguously.
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                if atom.GetDegree() != 1 or atom.GetBonds()[0].GetBondType() != Chem.BondType.SINGLE:
                    raise ValueError('Hydrogen capping requires degree-one, single-bonded * atoms')
                atom.SetAtomicNum(1)
                atom.SetIsotope(0)
                atom.SetFormalCharge(0)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
    elif wildcard_policy != 'preserve':
        raise ValueError('Unknown wildcard policy')
    return identity, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def read_rows(args, prediction: bool = False) -> tuple[list[dict], list[dict]]:
    rows, excluded = [], []
    with args.csv.open(encoding=args.encoding, newline='') as stream:
        reader = csv.DictReader(stream)
        required = [args.smiles_column]
        if not prediction:
            required += [args.target_column]
            if args.split_column:
                required += [args.split_column]
        if args.id_column:
            required += [args.id_column]
        absent = set(required) - set(reader.fieldnames or [])
        if absent:
            raise ValueError(f'Missing CSV columns: {sorted(absent)}')
        for index, source in enumerate(reader, start=2):
            value = str(source[args.smiles_column]).strip()
            target = None
            if not prediction:
                raw = str(source[args.target_column]).strip()
                if raw.lower() in {'', 'nan', 'na', 'none', 'null'} and args.missing_target == 'drop':
                    excluded.append({'csv_line': index, 'reason': 'missing selected target'})
                    continue
                try:
                    target = float(raw)
                except ValueError as exc:
                    raise ValueError(f'Non-numeric target at CSV line {index}: {raw!r}') from exc
                if not math.isfinite(target) or target < 0:
                    raise ValueError(f'Band gap must be a finite, nonnegative eV value at line {index}')
            try:
                identity, model_smiles = graph_smiles(value, args.wildcard_policy)
            except Exception as exc:
                raise ValueError(f'CSV line {index}: {exc}') from exc
            row = dict(csv_line=index, source_id=source.get(args.id_column, '') if args.id_column else '',
                       smiles=value, canonical_smiles=identity, model_smiles=model_smiles)
            if not prediction:
                row['target_eV'] = target
                if args.split_column:
                    label = source[args.split_column].strip().lower()
                    if label == 'val': label = 'valid'
                    if label not in {'train', 'valid', 'test'}:
                        raise ValueError(f'Unknown split {label!r} at line {index}')
                    row['split'] = label
            rows.append(row)
    if not rows:
        raise ValueError('No records remain')
    return rows, excluded


def assign_splits(rows: list[dict], supplied: bool, seed: int,
                  train_fraction: float, valid_fraction: float) -> None:
    # Union by both original and actual model graphs, and optional source ID.
    # Capping can collapse distinct repeat-unit strings to the same model input.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        keys = [('original', row['canonical_smiles']), ('model', row['model_smiles'])]
        if row['source_id']:
            keys.append(('source_id', row['source_id']))
        for key in keys:
            if key in seen:
                parent[find(i)] = find(seen[key])
            seen[key] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    if supplied:
        for ids in groups.values():
            labels = {rows[i]['split'] for i in ids}
            if len(labels) != 1:
                raise ValueError(f'Train/valid/test graph or source-ID overlap at CSV lines {[rows[i]["csv_line"] for i in ids]}')
    else:
        if not (0 < train_fraction < 1 and 0 < valid_fraction < 1 and train_fraction + valid_fraction < 1):
            raise ValueError('Fractions must be positive and leave a nonzero test fraction')
        indices = list(groups)
        random.Random(seed).shuffle(indices)
        n_train = int(len(indices) * train_fraction)
        n_valid = int(len(indices) * valid_fraction)
        if min(n_train, n_valid, len(indices)-n_train-n_valid) < 1:
            raise ValueError('Too few independent groups for the requested split fractions')
        for j, key in enumerate(indices):
            label = 'train' if j < n_train else ('valid' if j < n_train+n_valid else 'test')
            for i in groups[key]: rows[i]['split'] = label
    if {row['split'] for row in rows} != {'train','valid','test'}:
        raise ValueError('All three splits must contain records')


def metrics(actual, predicted) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError('Prediction count differs from the nonempty target count')
    if not all(math.isfinite(float(v)) for v in predicted):
        raise ValueError('Nonfinite model predictions')
    n = len(actual)
    errors = [float(p)-y for y,p in zip(actual,predicted)]
    mean = sum(actual)/n
    variance = sum((y-mean)**2 for y in actual)
    return dict(n_rows=n, MAE_eV=sum(abs(e) for e in errors)/n,
                RMSE_eV=math.sqrt(sum(e*e for e in errors)/n),
                R2=1-sum(e*e for e in errors)/variance if n>1 and variance else None)


def backend(seed: int):
    # TF 2.15/Keras 2 is the documented reproducible environment below.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
    import numpy as np
    import tensorflow as tf
    import deepchem as dc
    if dc.__version__ != '2.8.0' or not tf.__version__.startswith('2.15.'):
        raise RuntimeError('Use the pinned Python 3.11 / DeepChem 2.8.0 / TensorFlow 2.15 environment; see requirements.txt')
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    return np, dc


def dataset(rows, np, dc):
    features = dc.feat.ConvMolFeaturizer().featurize([row['model_smiles'] for row in rows])
    for row, feature in zip(rows, features):
        if not hasattr(feature, 'get_num_atoms') or feature.get_num_atoms() == 0:
            raise ValueError(f'ConvMol featurization failed at CSV line {row["csv_line"]}')
    labels = np.asarray([[row.get('target_eV',0.0)] for row in rows],dtype=float)
    return dc.data.NumpyDataset(X=features, y=labels,
                                ids=np.asarray([str(row['csv_line']) for row in rows]))


def train(args):
    if args.epochs < 1 or args.batch_size < 2 or args.learning_rate <= 0:
        raise ValueError('Epochs and learning rate must be positive; GraphGather needs batch_size >= 2')
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Output directory must be empty; existing experiments are never overwritten')
    rows, excluded = read_rows(args)
    assign_splits(rows, bool(args.split_column), args.seed, args.train_fraction, args.valid_fraction)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'split_manifest.csv', rows)
    write_json(args.out/'excluded_rows.json', excluded)
    config = dict(DEFAULTS, batch_size=args.batch_size, learning_rate=args.learning_rate)
    metadata = dict(status='prepared', source_doi=SOURCE_DOI, target='single_chain_band_gap',
                    target_unit='eV', source_csv=str(args.csv.resolve()), csv_sha256=sha256(args.csv),
                    target_column=args.target_column, smiles_column=args.smiles_column,
                    wildcard_policy=args.wildcard_policy, seed=args.seed,
                    model_config=config, epochs=args.epochs, target_normalization='none',
                    split_rule='supplied' if args.split_column else 'group_random',
                    fractions=None if args.split_column else [args.train_fraction,args.valid_fraction,1-args.train_fraction-args.valid_fraction],
                    identity_scope='canonical molecular/model graph and optional source ID; no periodic equivalence proof',
                    implementation_status='revision reconstruction; not historical checkpoint reproduction',
                    counts={s:sum(r['split']==s for r in rows) for s in ['train','valid','test']})
    write_json(args.out/'artifact.json', metadata)
    if args.prepare_only:
        print(json.dumps(metadata, indent=2)); return
    np, dc = backend(args.seed)
    subsets = {s:[r for r in rows if r['split']==s] for s in ['train','valid','test']}
    datasets = {s:dataset(rs,np,dc) for s,rs in subsets.items()}
    model_dir = args.out/'checkpoint'
    model = dc.models.GraphConvModel(model_dir=str(model_dir), **config)
    # Fixed training budget. Neither validation nor test labels select a model.
    loss = model.fit(datasets['train'], nb_epoch=args.epochs)
    model.save_checkpoint()
    scores = {}
    for split in ['train','valid','test']:
        pred = np.asarray(model.predict(datasets[split])).reshape(-1)
        scores[split] = metrics([r['target_eV'] for r in subsets[split]],pred.tolist())
        write_csv(args.out/f'{split}_predictions.csv', [dict(r,predicted_band_gap_eV=float(v)) for r,v in zip(subsets[split],pred)])
    # Confirm that a saved artifact is sufficient to reproduce inference.
    restored = dc.models.GraphConvModel(model_dir=str(model_dir), **config)
    restored.restore()
    before = model.predict(datasets['test'])
    after = restored.predict(datasets['test'])
    if not np.allclose(before,after,rtol=1e-5,atol=1e-6):
        raise RuntimeError('Saved/reloaded predictions do not agree')
    metadata.update(status='trained',training_loss=float(loss),reload_verified=True,
                    versions={name:importlib.metadata.version(name) for name in ['deepchem','tensorflow','numpy','rdkit']})
    write_json(args.out/'metrics.json',scores)
    write_json(args.out/'artifact.json',metadata)
    print(json.dumps(scores,indent=2))


def predict(args):
    meta=json.loads((args.model_dir/'artifact.json').read_text(encoding='utf-8'))
    if meta['status'] != 'trained': raise ValueError('This artifact has not been trained')
    if args.out.exists(): raise ValueError('Prediction output already exists')
    args.wildcard_policy=meta['wildcard_policy']
    rows,_=read_rows(args,prediction=True)
    np,dc=backend(meta['seed'])
    model=dc.models.GraphConvModel(model_dir=str(args.model_dir/'checkpoint'),**meta['model_config'])
    model.restore()
    values=np.asarray(model.predict(dataset(rows,np,dc))).reshape(-1)
    if len(values)!=len(rows) or not np.isfinite(values).all(): raise ValueError('Invalid prediction result')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    write_csv(args.out,[dict(r,predicted_band_gap_eV=float(v)) for r,v in zip(rows,values)])


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    modes=p.add_subparsers(dest='command',required=True)
    for name in ['train','predict']:
        q=modes.add_parser(name)
        q.add_argument('--csv',type=Path,required=True)
        q.add_argument('--smiles-column',default='SMILES')
        q.add_argument('--id-column')
        q.add_argument('--encoding',default='utf-8-sig')
        q.add_argument('--out',type=Path,required=True)
        if name=='train':
            q.add_argument('--target-column',required=True,help='Select the single-chain band-gap column in eV, not crystal/bulk band gap')
            q.add_argument('--wildcard-policy',choices=['preserve','hydrogen_cap'],required=True)
            q.add_argument('--split-column')
            q.add_argument('--missing-target',choices=['error','drop'],default='error')
            q.add_argument('--train-fraction',type=float,default=0.8)
            q.add_argument('--valid-fraction',type=float,default=0.1)
            q.add_argument('--seed',type=int,default=42)
            q.add_argument('--epochs',type=int,default=10)
            q.add_argument('--batch-size',type=int,default=100)
            q.add_argument('--learning-rate',type=float,default=0.001)
            q.add_argument('--prepare-only',action='store_true')
            q.set_defaults(func=train)
        else:
            q.add_argument('--model-dir',type=Path,required=True)
            q.set_defaults(func=predict)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    args.func(args)
