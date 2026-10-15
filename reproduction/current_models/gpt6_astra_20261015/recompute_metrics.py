"""Recompute the released GPT-6-Astra benchmark using only Python's standard library.

Usage: python recompute_metrics.py predictions.csv --check metadata.json
This script performs no network calls and reads no model configuration or credentials.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

EXPECTED_TASKS={'QA':3,'Tg':70,'BG':48,'AE':60,'class':54}
AE_LOWER=-6.82
AE_UPPER=-5.01

def parse_output(text,task):
    value=text.strip().replace('\u2212','-')
    if task=='class':
        value=' '.join(value.lower().split())
        return value if re.fullmatch(r'class [123]',value) else None
    if re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?',value):
        number=float(value)
        if math.isfinite(number):return number
    return None

def regression(rows):
    valid=[r for r in rows if r['_valid']]
    if not valid:return {'total':len(rows),'valid':0,'coverage':0.0 if rows else None,'MAE':None,'RMSE':None,'R2':None}
    y=[float(r['target']) for r in valid];p=[r['_prediction'] for r in valid]
    errors=[b-a for a,b in zip(y,p)];mean=statistics.mean(y);sst=sum((a-mean)**2 for a in y)
    return {'total':len(rows),'valid':len(valid),'coverage':len(valid)/len(rows),
        'MAE':statistics.mean(abs(e) for e in errors),'RMSE':math.sqrt(statistics.mean(e*e for e in errors)),
        'R2':1-sum(e*e for e in errors)/sst if sst else None}

def classification(rows):
    labels=['class 1','class 2','class 3'];recalls=[];f1s=[]
    matrix={y:{p:0 for p in labels+['invalid']} for y in labels}
    for row in rows:matrix[row['target']][row['_prediction'] if row['_valid'] else 'invalid']+=1
    per=[]
    for label in labels:
        tp=matrix[label][label];fn=sum(matrix[label].values())-tp
        fp=sum(matrix[y][label] for y in labels if y!=label)
        recall=tp/(tp+fn) if tp+fn else 0;f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        recalls.append(recall);f1s.append(f1);per.append({'label':label,'support':tp+fn,'recall':recall,'f1':f1})
    return {'total':len(rows),'valid':sum(r['_valid'] for r in rows),
        'accuracy':sum(matrix[y][y] for y in labels)/len(rows),
        'macro_F1':statistics.mean(f1s),'balanced_accuracy':statistics.mean(recalls),
        'confusion_matrix':matrix,'per_class':per}

def calculate(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    counts=dict(Counter(r['task'] for r in rows))
    if counts!=EXPECTED_TASKS:raise ValueError(f'Expected the complete 235-case table, got {counts}')
    if len({r['case_id'] for r in rows})!=235:raise ValueError('Duplicate or missing case IDs')
    for row in rows:
        if row['task']=='QA':continue
        value=parse_output(row['output_text'],row['task'])
        supplied=row['parsed_prediction']
        if value is None:
            if supplied:raise ValueError('A parsed prediction was supplied for an unparseable output: '+row['case_id'])
        elif row['task']=='class':
            if supplied!=value:raise ValueError('Class prediction differs from the visible output: '+row['case_id'])
        elif not supplied or float(supplied)!=value:
            raise ValueError('Numeric prediction differs from the visible output: '+row['case_id'])
        row['_prediction']=value;row['_valid']=value is not None
    props=[r for r in rows if r['task']!='QA'];qa=[r for r in rows if r['task']=='QA']
    metrics={task+'_unfiltered':regression([r for r in props if r['task']==task]) for task in ['Tg','BG','AE']}
    ae=[r for r in props if r['task']=='AE']
    retained=[r for r in ae if r['_valid'] and AE_LOWER<=r['_prediction']<=AE_UPPER]
    metrics['AE_train_range_retained']=regression(retained)
    metrics['AE_train_range_retained'].update(full_AE_denominator=len(ae),retained_coverage=len(retained)/len(ae),
        excluded_parsed_predictions=sum(r['_valid'] for r in ae)-len(retained),
        parse_failures=sum(not r['_valid'] for r in ae),sensitivity_only=True)
    metrics['class']=classification([r for r in props if r['task']=='class'])
    coverage={'total_cases':len(rows),'task_counts':counts,'QA_visible_nonempty':sum(bool(r['output_text'].strip()) for r in qa),
        'property_parse_valid':sum(r['_valid'] for r in props),'property_total':len(props)}
    return {'coverage':coverage,'metrics':metrics}

def compare(actual,expected,path='result'):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise AssertionError('Different keys at '+path)
        for key in expected:compare(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        if len(actual)!=len(expected):raise AssertionError('Different list lengths at '+path)
        for index,(a,e) in enumerate(zip(actual,expected)):compare(a,e,f'{path}[{index}]')
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12):raise AssertionError(f'Numeric mismatch at {path}: {actual} != {expected}')
    elif actual!=expected:raise AssertionError('Mismatch at '+path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions',nargs='?',default='predictions.csv')
    parser.add_argument('--check',type=Path,help='Compare recomputed coverage and metrics to metadata.json')
    args=parser.parse_args();result=calculate(args.predictions)
    if args.check:
        metadata=json.loads(args.check.read_text(encoding='utf-8'))
        compare(result['coverage'],metadata['coverage'],'coverage')
        compare(result['metrics'],metadata['metrics'],'metrics')
        result['matches_metadata']=True
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
