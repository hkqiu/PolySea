"""Verify this capsule, reload its GNN, or make a separate two-seed CPU replay."""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CAPSULE = Path(__file__).resolve().parent

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def verify_capsule():
    manifest = read_json(CAPSULE/'sha256_manifest.json')
    for entry in manifest['files']:
        path = CAPSULE/entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Capsule integrity check failed: {entry["path"]}')
    return dict(status='passed', checked_files=len(manifest['files']))

def fresh_output(path):
    path = path.resolve()
    if path == CAPSULE or CAPSULE in path.parents:
        raise ValueError('Choose a new output directory outside the published capsule.')
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError('The output directory must be new or empty; previous runs are never overwritten.')
    path.mkdir(parents=True, exist_ok=True)
    return path

def thread_environment(threads):
    if not 1 <= threads <= 8:
        raise ValueError('Choose between 1 and 8 CPU threads; the recorded run used 8.')
    return dict(os.environ, PYTHONUNBUFFERED='1', PYTHONUTF8='1', OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
        TF_NUM_INTRAOP_THREADS=str(threads), TF_NUM_INTEROP_THREADS='1',
        CUDA_VISIBLE_DEVICES='-1', TF_CPP_MIN_LOG_LEVEL='2',
        HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false')

def predict(args):
    integrity = verify_capsule()
    if os.name == 'nt' and not str(args.output_dir.resolve()).isascii():
        raise ValueError('TensorFlow 2.15 on Windows needs an ASCII output path for this replay; choose an ASCII directory or path alias.')
    out = fresh_output(args.output_dir)
    script = args.gnn_script or CAPSULE.parent/'train_graphconv.py'
    if not script.is_file():
        raise FileNotFoundError('The parent train_graphconv.py is required; supply --gnn-script if the capsule was downloaded alone.')
    protocol = read_json(CAPSULE/'protocol.json')
    if sha256(script) != protocol['gnn_training_script_sha256']:
        raise RuntimeError('train_graphconv.py differs from the implementation used for this capsule.')
    # Stage only the small checkpoint so a Unicode source checkout also works on Windows.
    staged = out/'model'
    staged.mkdir()
    shutil.copyfile(CAPSULE/'artifact.json', staged/'artifact.json')
    shutil.copytree(CAPSULE/'checkpoint', staged/'checkpoint')
    command = [sys.executable, str(script), 'predict', '--csv', str(CAPSULE/'candidates_input.csv'),
        '--id-column', 'candidate_id', '--model-dir', str(staged), '--out', str(out/'candidate_predictions.csv')]
    started = now()
    subprocess.run(command, env=thread_environment(args.threads), check=True)
    with (out/'candidate_predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    expected = {row['candidate_id']:row['predicted_band_gap_eV'] for row in read_json(CAPSULE/'candidate_results.json')}
    observed = {row['source_id']:float(row['predicted_band_gap_eV']) for row in rows}
    if set(observed) != set(expected) or len(rows) != 3:
        raise RuntimeError('Candidate row identities differ from the three retained examples.')
    errors = {key:abs(observed[key]-expected[key]) for key in expected}
    agrees = all(math.isclose(observed[k], expected[k], rel_tol=1e-5, abs_tol=1e-6) for k in expected)
    report = dict(status='passed' if agrees else 'mismatch', started_utc=started, ended_utc=now(),
        capsule_integrity=integrity, candidate_predictions_eV=observed,
        expected_predictions_eV=expected, absolute_differences_eV=errors,
        maximum_absolute_difference_eV=max(errors.values()), relative_tolerance=1e-5, absolute_tolerance_eV=1e-6,
        checkpoint_files={p.name:sha256(p) for p in sorted((staged/'checkpoint').iterdir()) if p.is_file()},
        note='One reload of the released checkpoint; no retraining, resampling, or physical calculation.')
    save_json(out/'verification.json', report)
    print(json.dumps(report, indent=2))
    if not agrees:
        raise RuntimeError('Reloaded predictions exceed the recorded numerical tolerance.')

def generate(args):
    verify_capsule()
    if not args.model_dir.is_dir():
        raise ValueError('--model-dir must identify the locally supplied merged PolyOpus model.')
    out = fresh_output(args.output_dir)
    os.environ.update(thread_environment(args.threads))
    os.environ['USE_TF'] = '0'
    os.environ['USE_FLAX'] = '0'
    manifest = read_json(CAPSULE/'polyopus_model_manifest.json')
    for entry in manifest['files']:
        path = args.model_dir/entry['name']
        if not path.is_file() or path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise RuntimeError(f'Supplied PolyOpus file does not match the recorded model: {entry["name"]}')
    save_json(out/'model_verification.json', dict(status='passed', model_manifest=manifest, checked_utc=now()))
    import random
    import numpy as np
    import torch
    from rdkit import Chem
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map={'':'cpu'}, attn_implementation='eager')
    model.eval()
    generation = read_json(CAPSULE/'protocol.json')['generation']
    inputs = tokenizer(generation['prompt'], return_tensors='pt')
    length = inputs['input_ids'].shape[1]
    save_json(out/'input.json', dict(prompt=generation['prompt'], input_ids=inputs['input_ids'][0].tolist(),
        input_token_count=length, decoded=tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)))
    save_json(out/'environment.json', dict(device='cpu', weight_dtype=str(next(model.parameters()).dtype),
        quantization='none', threads=args.threads, python='.'.join(map(str,sys.version_info[:3])),
        versions={name:importlib.metadata.version(name) for name in ['torch','transformers','accelerate','safetensors','numpy','rdkit']},
        model_generation_config=model.generation_config.to_dict(),
        note='Separate replay outputs; hardware/software differences can change sampled text despite fixed seeds.'))
    class StopAtMarker(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return generation['stop_string'] in tokenizer.decode(input_ids[0], skip_special_tokens=True)
    def clean(text):
        if 'PolySea:' in text:
            text = text.split('PolySea:',1)[1]
        return text.split('<STOP>',1)[0].strip()
    for seed in generation['seeds']:
        path = out/f'seed_{seed}_result.json'
        record = dict(seed=seed, formal_attempt=1, status='started', started_utc=now(),
            prompt=generation['prompt'], target_eV=generation['nominal_target_eV'],
            weight_dtype='float16', quantization='none',
            parameters={k:generation[k] for k in ['max_new_tokens','temperature','top_p','top_k','do_sample']})
        save_json(path,record)
        start = time.perf_counter()
        try:
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            with torch.inference_mode():
                output = model.generate(**inputs, **record['parameters'],
                    stopping_criteria=StoppingCriteriaList([StopAtMarker()]),
                    eos_token_id=generation['eos_token_id'], pad_token_id=generation['eos_token_id'], use_cache=True)
            ids = output[0].tolist()
            full = tokenizer.decode(ids, skip_special_tokens=True)
            suffix = tokenizer.decode(ids[length:], skip_special_tokens=True)
            candidate = clean(full) if 'PolySea:' in full else clean(suffix)
            mol = Chem.MolFromSmiles(candidate) if candidate else None
            valid = mol is not None and mol.GetNumAtoms()>0
            record.update(status='completed',full_token_ids=ids,generated_token_ids=ids[length:],
                decoded_with_special_tokens=tokenizer.decode(ids,skip_special_tokens=False),
                full_decoded_text=full,generated_suffix=suffix,source_cleaned_text=clean(full),
                candidate_smiles=candidate,rdkit_valid=valid,
                canonical_smiles=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True) if valid else None,
                wildcard_count=sum(a.GetAtomicNum()==0 for a in mol.GetAtoms()) if valid else None,
                stop_reason='stop_string' if generation['stop_string'] in full else ('eos' if ids[-1]==generation['eos_token_id'] else 'max_new_tokens'))
        except Exception as exc:
            record.update(status='failed',rdkit_valid=False,error_type=type(exc).__name__,error_message=str(exc))
        record.update(ended_utc=now(),elapsed_seconds=time.perf_counter()-start)
        save_json(path,record)
        print(json.dumps(record,ensure_ascii=False))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest='mode',required=True)
    modes.add_parser('verify',help='Check the capsule byte hashes without scientific dependencies.')
    predict_parser = modes.add_parser('predict',help='Reload the released GNN once and verify all three candidate predictions.')
    predict_parser.add_argument('--output-dir',type=Path,required=True)
    predict_parser.add_argument('--gnn-script',type=Path)
    predict_parser.add_argument('--threads',type=int,default=8)
    generation_parser = modes.add_parser('generate',help='Make a separate CPU replay using a user-supplied local PolyOpus model.')
    generation_parser.add_argument('--model-dir',type=Path,required=True)
    generation_parser.add_argument('--output-dir',type=Path,required=True)
    generation_parser.add_argument('--threads',type=int,default=8)
    args = p.parse_args()
    if args.mode=='verify':print(json.dumps(verify_capsule(),indent=2))
    elif args.mode=='predict':predict(args)
    else:generate(args)

if __name__=='__main__':
    main()
