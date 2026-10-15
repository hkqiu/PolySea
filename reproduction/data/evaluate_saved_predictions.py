"""Recompute the single final PolyOpus prediction table and verify source hashes."""
from pathlib import Path
from collections import defaultdict
from decimal import Decimal
import argparse,csv,hashlib,json,math,re

NUMBER = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?:eV(?:/atom)?|°C|℃|degC)?$",re.I)
def numeric(value):
    match=NUMBER.fullmatch(value.strip().replace('−','-'))
    if not match:raise ValueError('Invalid numeric value: '+value)
    result=float(Decimal(match.group(1)))
    if not math.isfinite(result):raise ValueError('Nonfinite value')
    return result
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def regression(rows):
    y=[numeric(r['target']) for r in rows];p=[numeric(r['prediction']) for r in rows]
    n=len(y);mean=math.fsum(y)/n
    sse=math.fsum((a-b)**2 for a,b in zip(y,p));sst=math.fsum((a-mean)**2 for a in y)
    return dict(n_rows=n,MAE=math.fsum(abs(a-b) for a,b in zip(y,p))/n,RMSE=math.sqrt(sse/n),R2=1-sse/sst)
def classification(rows):
    labels=['class 1','class 2','class 3'];matrix={a:{b:0 for b in labels} for a in labels}
    for row in rows:
        if row['target'] not in labels or row['prediction'] not in labels:raise ValueError('Unexpected class label')
        matrix[row['target']][row['prediction']]+=1
    per={}
    for a in labels:
        tp=matrix[a][a];support=sum(matrix[a].values());pred=sum(matrix[b][a] for b in labels)
        per[a]=dict(support=support,precision=tp/pred if pred else 0,recall=tp/support if support else 0,F1=2*tp/(support+pred) if support+pred else 0)
    return dict(n_rows=len(rows),accuracy=sum(matrix[a][a] for a in labels)/len(rows),macro_F1=sum(v['F1'] for v in per.values())/3,weighted_F1=sum(v['F1']*v['support'] for v in per.values())/len(rows),per_class=per,confusion_matrix_target_rows_prediction_columns=matrix)
def compare(actual,expected):
    for key,value in actual.items():
        if isinstance(value,dict):compare(value,expected[key])
        elif not math.isclose(value,expected[key],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('Metric mismatch: '+key)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'dataset/polyopus')
    parser.add_argument('--output',type=Path,help='Optional new JSON file; existing files are not overwritten')
    args=parser.parse_args();meta=json.loads((args.data_dir/'metadata.json').read_text(encoding='utf-8'))
    for name,expected in meta['files'].items():
        path=args.data_dir/name
        if path.stat().st_size!=expected['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:raise ValueError('Integrity mismatch: '+name)
        if len(read_csv(path))!=expected['rows']:raise ValueError('Record count mismatch: '+name)
    train=read_csv(args.data_dir/'train/merged_train.csv')
    ae=[numeric(r['target']) for r in train if any(t in r['prompt'].casefold() for t in ('atomization','binding energy','原子','解离能'))]
    if (len(ae),min(ae),max(ae))!=(52650,-6.82,-5.01):raise ValueError('AE training range changed')
    all_rows=read_csv(args.data_dir/'final_predictions.csv');groups=defaultdict(list)
    if len({r['prediction_id'] for r in all_rows})!=len(all_rows):raise ValueError('Duplicate prediction ID')
    for row in all_rows:
        task=row['task']
        if task not in meta['denominators']:raise ValueError('Unexpected task')
        if task=='atomization_energy':
            if row['reporting_scope']!='training_range_retained' or not min(ae)<=numeric(row['prediction'])<=max(ae):raise ValueError('AE reporting rule mismatch')
        elif row['reporting_scope']!='all_saved':raise ValueError('Unexpected row selection')
        groups[task].append(row)
    result={}
    for task,denom in meta['denominators'].items():
        if len(groups[task])!=denom['reported_rows']:raise ValueError('Task count mismatch: '+task)
        result[task]=classification(groups[task]) if task=='heat_class' else regression(groups[task])
        compare(result[task],meta['metrics'][task])
    report=dict(status='passed',prediction_rows=len(all_rows),metrics=result,AE_denominators=meta['denominators']['atomization_energy'],archived_full_regression_metrics_recomputed=False)
    encoded=json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:f.write(encoded)
    print(encoded)
if __name__=='__main__':main()
