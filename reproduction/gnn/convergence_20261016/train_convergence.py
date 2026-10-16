"""Train a registered convergence variant using explicit, portable paths."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil

HERE=Path(__file__).absolute().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True,help='New empty output directory; use an ASCII path on Windows')
p.add_argument('--baseline-dir',type=Path,default=HERE/'baseline_10epoch')
p.add_argument('--split-manifest',type=Path,default=HERE/'data/split_manifest.csv')
p.add_argument('--variant',choices=['standardized_from_scratch','continued_unscaled','both'],default='standardized_from_scratch')
p.add_argument('--max-epochs',type=int,default=300)
p.add_argument('--min-stop-epoch',type=int,default=100)
p.add_argument('--patience',type=int,default=50)
a=p.parse_args()
if a.out.exists() and any(a.out.iterdir()): raise RuntimeError('Output directory must be new or empty')
a.out.mkdir(parents=True,exist_ok=True)
protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
protocol['variants']=[v for v in protocol['variants'] if a.variant=='both' or v['name']==a.variant]
protocol['max_epoch']=a.max_epochs;protocol['min_stop_epoch']=a.min_stop_epoch;protocol['patience_epochs']=a.patience
shutil.copytree(a.baseline_dir,a.out/'baseline_10epoch')
shutil.copy2(a.split_manifest,a.out/'baseline_10epoch/split_manifest.csv')
(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
os.environ['GNN_EXPERIMENT_ROOT']=str(a.out.absolute())
spec=importlib.util.spec_from_file_location('portable_convergence_core',HERE/'convergence_core.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main()
