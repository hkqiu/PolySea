"""Reload a selected GraphConv artifact, including its train-only label transform."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).absolute().parent))
import graphconv_helpers as helper

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model-dir',type=Path,required=True)
parser.add_argument('--csv',type=Path,required=True)
parser.add_argument('--smiles-column',default='SMILES')
parser.add_argument('--id-column',default='candidate_id')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
if args.out.exists():
    raise RuntimeError('Prediction file already exists')
meta=json.loads((args.model_dir/'artifact.json').read_text(encoding='utf-8'))
args.wildcard_policy=meta['wildcard_policy'];args.encoding='utf-8-sig'
rows,_=helper.read_rows(args,prediction=True)
np,dc=helper.backend(meta['seed'])
model=dc.models.GraphConvModel(model_dir=str(args.model_dir/'checkpoint'),**meta['model_config'])
model.restore()
scale=meta['target_scaling']
pred=np.asarray(model.predict(helper.dataset(rows,np,dc))).reshape(-1)*scale['std_eV']+scale['mean_eV']
helper.write_csv(args.out,[dict(r,predicted_band_gap_eV=float(v)) for r,v in zip(rows,pred)])
print(json.dumps({'n':len(rows),'output':str(args.out),'selected_epoch':meta['selected_epoch']}))
