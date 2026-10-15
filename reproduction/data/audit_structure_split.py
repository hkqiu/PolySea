"""Audit existing splits and rescore fixed archived predictions; never retrain.

The external canonicalize_psmiles package is an explicit normalization convention,
not a universal proof of periodic-polymer equivalence. Unsupported representations
remain separately labelled molecular-graph identities. Original files are read-only.
"""
from pathlib import Path
from collections import Counter,defaultdict
from functools import lru_cache
from decimal import Decimal
import csv,io,json,re,sys,hashlib,math,logging,subprocess

import argparse,inspect
parser=argparse.ArgumentParser(description="Rebuild the structural audit of the existing PolyOpus split; no training or inference.")
parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parents[2]/'dataset/polyopus')
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--canonicalizer-dir',type=Path,help='Optional source checkout for canonicalize_psmiles')
args=parser.parse_args()
PUBLIC=args.data_dir.resolve();AUDIT=args.output_dir.resolve()
if AUDIT==PUBLIC or PUBLIC in AUDIT.parents:raise ValueError('Output must be outside the input data directory.')
AUDIT.mkdir(parents=True,exist_ok=True)
if args.canonicalizer_dir:sys.path.insert(0,str(args.canonicalizer_dir.resolve()))
from rdkit import Chem,RDLogger,rdBase
from canonicalize_psmiles.canonicalize import canonicalize
canonicalizer_file=Path(inspect.getsourcefile(canonicalize))
if hashlib.sha256(canonicalizer_file.read_text(encoding='utf-8').encode()).hexdigest()!='4171f2fe09a9f80efc06b132649ba4179d939a9893cc110b491aa200e05ae826':
    raise ValueError('Use the pinned canonicalize_psmiles source revision in the README.')
RDLogger.DisableLog('rdApp.*')
logging.getLogger().setLevel(logging.CRITICAL)
TASKS=['Tg','BG','AE','class']

def read_csv(path):
    raw=path.read_bytes()
    try:text=raw.decode('utf-8-sig')
    except UnicodeDecodeError:text=raw.decode('gb18030')
    return list(csv.DictReader(io.StringIO(text)))

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def write_csv(path,rows,fields):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def task_of(prompt):
    p=prompt.lower()
    if re.search(r'heat resistance|thermal stability|temperature endurance|thermal resistance|耐热|热稳定',p):return 'class'
    if re.search(r'atomization|binding energy|individual atoms|原子|解离能',p):return 'AE'
    if re.search(r'bandgap|band gap|带隙|能带',p):return 'BG'
    if re.search(r'\btg\b|glass transition|玻璃化',p):return 'Tg'
    return 'other'

def number(value,task):
    value=value.strip().replace('−','-')
    m=re.fullmatch(r'class\s*([123])(?:eV)?',value,re.I) if task=='class' else re.match(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',value)
    if not m:raise ValueError((task,value))
    return int(m.group(1)) if task=='class' else float(m.group(1))

@lru_cache(None)
def graph(smiles):
    m=Chem.MolFromSmiles(smiles)
    if m is None:return None
    for atom in m.GetAtoms():atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True)

CACHE={}
cache_path=AUDIT/'identity_cache.jsonl'
if cache_path.exists():
    for line in cache_path.read_text(encoding='utf-8').splitlines():
        value=json.loads(line);CACHE[value['graph']]=value

def identity(smiles):
    g=graph(smiles)
    if g is None:return dict(graph='',key='',method='invalid',normalized='',reason='RDKit parse failed',stars=None)
    if g in CACHE:return CACHE[g]
    m=Chem.MolFromSmiles(g);stars=[a for a in m.GetAtoms() if a.GetAtomicNum()==0]
    normalized=g;method='graph_only';reason='requires two terminal attachment markers'
    eligible=len(stars)==2 and all(a.GetDegree()==1 for a in stars) and len(Chem.GetMolFrags(m))==1
    if eligible and stars[0].GetBonds()[0].GetBondType()!=stars[1].GetBonds()[0].GetBondType():
        eligible=False;reason='unequal boundary bond types'
    if eligible:
        try:
            candidate=canonicalize(g.replace('*','[*]'))
            target=Chem.MolFromSmiles(candidate)
            if target is None:raise ValueError('normalized output is invalid')
            if sum(a.GetAtomicNum()==0 for a in target.GetAtoms())!=2:raise ValueError('attachment count changed')
            original_counts=Counter(a.GetAtomicNum() for a in m.GetAtoms() if a.GetAtomicNum())
            reduced_counts=Counter(a.GetAtomicNum() for a in target.GetAtoms() if a.GetAtomicNum())
            ratios={original_counts[k]/reduced_counts[k] for k in original_counts if reduced_counts[k]}
            if set(original_counts)!=set(reduced_counts) or len(ratios)!=1 or next(iter(ratios))<1 or not next(iter(ratios)).is_integer():
                raise ValueError('element counts are not preserved under integer repeat reduction')
            normalized=graph(candidate)
            again=graph(canonicalize(normalized.replace('*','[*]')))
            if again!=normalized:raise ValueError('normalization is not idempotent')
            method='psmiles_normalized';reason=''
        except Exception as exc:
            normalized=g;method='graph_only';reason=type(exc).__name__+': '+str(exc)[:170]
    key=method+'|'+normalized
    result=dict(graph=g,key=key,structure_id=hashlib.sha256(key.encode()).hexdigest()[:20],method=method,normalized=normalized,reason=reason,stars=len(stars))
    CACHE[g]=result
    with cache_path.open('a',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
    if len(CACHE)%1000==0:print('Normalized',len(CACHE),'distinct graphs',flush=True)
    return result

def metrics(rows,task):
    if not rows:return {'n':0,'MAE':None,'RMSE':None,'R2':None} if task!='class' else {'n':0,'accuracy':None,'macro_F1':None,'balanced_accuracy':None}
    if task=='class':
        cm=[[sum(r['y']==a and r['p']==b for r in rows) for b in [1,2,3]] for a in [1,2,3]]
        f1=[];recalls=[]
        for i in range(3):
            tp=cm[i][i];support=sum(cm[i]);pred=sum(r[i] for r in cm)
            f1.append(2*tp/(support+pred) if support+pred else 0)
            if support:recalls.append(tp/support)
        return dict(n=len(rows),accuracy=sum(cm[i][i] for i in range(3))/len(rows),macro_F1=math.fsum(f1)/3,balanced_accuracy=math.fsum(recalls)/len(recalls),confusion_true_rows=cm)
    y=[r['y'] for r in rows];p=[r['p'] for r in rows];mean=math.fsum(y)/len(y)
    sse=math.fsum((a-b)**2 for a,b in zip(y,p));sst=math.fsum((a-mean)**2 for a in y)
    return dict(n=len(rows),MAE=math.fsum(abs(a-b) for a,b in zip(y,p))/len(y),RMSE=math.sqrt(sse/len(y)),R2=1-sse/sst if len(y)>1 and sst else None)

def main():
    paths={'train':PUBLIC/'train/merged_train.csv','test':PUBLIC/'test/merged_test.csv','TestSet1':PUBLIC/'test/benchmark.csv','predictions':PUBLIC/'final_predictions.csv'}
    hashes={key:sha(path) for key,path in paths.items()}
    data={key:read_csv(path) for key,path in paths.items()}
    data['regression']=[];data['classification']=[]
    for row in data['predictions']:
        key='classification' if row['task']=='heat_class' else 'regression'
        data[key].append({'prompt-new':row['input'],'target':row['target'],'PolySea':row['prediction'],'_source_row':row['source_data_row']})
    # All these are independent of prediction errors and are fixed before scoring.
    smoke={s:identity(s)['normalized'] for s in ['[*]CC[*]','[*]CCCC[*]','[*]CCO[*]','[*]OCC[*]','[*]CCOCCO[*]','[*]CC([*])C','[*]CC(CC([*])C)C']}
    assert smoke['[*]CC[*]']==smoke['[*]CCCC[*]']
    assert smoke['[*]CCO[*]']==smoke['[*]OCC[*]']==smoke['[*]CCOCCO[*]']
    # Upstream reducer does not generally reduce branched/ring repeat multiplication.
    smoke['branched_repeat_limitation']=smoke['[*]CC([*])C']!=smoke['[*]CC(CC([*])C)C']
    write_json(AUDIT/'normalization_checks.json',smoke)
    records={};training_all=set();split_rows=[]
    for split in ['train','test','TestSet1']:
        by_task={task:[] for task in TASKS}
        for i,row in enumerate(data[split],1):
            sm=row['SMILES'].strip();task=task_of(row['prompt'])
            if split=='train' and not sm:
                candidate=row['target'].strip()
                if graph(candidate):
                    ident=identity(candidate)
                    if ident['key']:training_all.add(ident['key'])
                continue
            if task not in TASKS or not sm:continue
            ident=identity(sm)
            rec=dict(source_row=i,task=task,smiles=sm,target=row['target'],y=number(row['target'],task),**ident)
            by_task[task].append(rec)
            if split=='train' and ident['key']:training_all.add(ident['key'])
        records[split]=by_task
        print(split,Counter({k:len(v) for k,v in by_task.items()}),flush=True)
    train_keys={t:{r['key'] for r in records['train'][t] if r['key']} for t in TASKS}
    summary={}
    for task in TASKS:
        split_summary={}
        for split in ['train','test','TestSet1']:
            rows=records[split][task];keys={r['key'] for r in rows if r['key']}
            target_sets=defaultdict(set)
            for r in rows:
                if r['key']:target_sets[r['key']].add(r['y'])
            split_summary[split]=dict(rows=len(rows),molecular_graphs=len({r['graph'] for r in rows if r['graph']}),normalized_structures=len(keys),normalized_repeat_units=len({r['key'] for r in rows if r['method']=='psmiles_normalized'}),graph_only_identities=len({r['key'] for r in rows if r['method']=='graph_only'}),invalid_rows=sum(r['method']=='invalid' for r in rows),structures_with_multiple_targets=sum(len(v)>1 for v in target_sets.values()))
        test_keys={r['key'] for r in records['test'][task] if r['key']}
        overlap=test_keys&train_keys[task]
        split_summary['overlap']=dict(same_task_structures=len(overlap),same_task_test_rows=sum(r['key'] in overlap for r in records['test'][task]),any_structured_training_structures=len(test_keys&training_all),same_task_graph_overlap=len({r['graph'] for r in records['train'][task] if r['graph']}&{r['graph'] for r in records['test'][task] if r['graph']}))
        core_keys={r['key'] for r in records['train'][task] if 7827<=r['source_row']<=214326 and r['key']}
        split_summary['core_training_normalized_structures']=len(core_keys)
        split_summary['combined_train_test_normalized_structures']=len(test_keys|train_keys[task])
        summary[task]=split_summary
        for split in ['train','test','TestSet1']:
            first={}
            for r in records[split][task]:
                if r['key'] and r['key'] not in first:first[r['key']]=r
            for r in first.values():
                split_rows.append(dict(split=split,task=task,structure_id=r['structure_id'],identity_method=r['method'],normalized_structure=r['normalized'],example_smiles=r['smiles'],first_source_row=r['source_row'],target=r['target'],same_task_train_overlap=r['key'] in train_keys[task],any_structured_train_overlap=r['key'] in training_all))
    write_csv(AUDIT/'structure_registry.csv',split_rows,list(split_rows[0]))
    # Match prediction prompts to original test inputs before considering a fallback.
    prompt_map={}
    for row in data['test']:
        prompt_map[' '.join((row['prompt'].strip()+' '+row['SMILES'].strip()).split())]=row['SMILES'].strip()
    predictions={t:[] for t in TASKS};unmatched=[];modes=Counter()
    test_target_keys={t:{(r['key'],r['y']) for r in records['test'][t]} for t in TASKS}
    for name in ['regression','classification']:
        for ordinal,row in enumerate(data[name],1):
            ordinal=int(row['_source_row'])
            raw_input=row['prompt-new']
            task='class' if name=='classification' else task_of(raw_input)
            text=' '.join(raw_input.split())
            sm=prompt_map.get(text);mode='exact_test_prompt'
            if sm is None:
                sm=text.rsplit(' ',1)[-1];mode='trailing_smiles'
            ident=identity(sm)
            if task not in TASKS or not ident['key']:
                unmatched.append(dict(file=name,row=ordinal,task=task));continue
            y=number(row['target'],task);p=number(row['PolySea'],task)
            if (ident['key'],y) not in test_target_keys[task]:
                unmatched.append(dict(file=name,row=ordinal,task=task,reason='structure/target not in test'));continue
            modes[mode]+=1
            predictions[task].append(dict(source=name,source_row=ordinal,task=task,input=raw_input,smiles=sm,target=row['target'],prediction=row['PolySea'],y=y,p=p,**ident))
    write_json(AUDIT/'prediction_alignment.json',dict(matched_modes=modes,unmatched=unmatched))
    assert not unmatched,('unmatched prediction rows',unmatched[:5])
    results={};representatives=[]
    for task,rows in predictions.items():
        cohorts={'reported':rows}
        results[task]={}
        for cohort,selected in cohorts.items():
            first={}
            for r in selected:
                if r['key'] not in first:first[r['key']]=r
            unique=list(first.values())
            no_same=[r for r in unique if r['key'] not in train_keys[task]]
            no_any=[r for r in unique if r['key'] not in training_all]
            results[task][cohort]=dict(original_rows=len(selected),unique_structures=metrics(unique,task),no_same_task_train_overlap=metrics(no_same,task),no_structured_train_overlap=metrics(no_any,task))
            for r in unique:
                representatives.append(dict(cohort=cohort,task=task,source=r['source'],source_row=r['source_row'],structure_id=r['structure_id'],identity_method=r['method'],normalized_structure=r['normalized'],SMILES=r['smiles'],target=r['target'],prediction=r['prediction'],same_task_train_overlap=r['key'] in train_keys[task],any_structured_train_overlap=r['key'] in training_all))
    public_rows=[r for r in representatives if r['cohort']=='reported']
    write_csv(AUDIT/'final_predictions.csv',public_rows,list(public_rows[0]))
    for task in TASKS:
        clean={r['structure_id'] for r in public_rows if r['task']==task and not r['same_task_train_overlap']}
        training={r['structure_id'] for r in split_rows if r['split']=='train' and r['task']==task}
        assert not clean&training
    assert hashes=={key:sha(path) for key,path in paths.items()}
    version='729dfd20b0e909df7cb245ffbfa5ab2187ffbf8b'
    report=dict(source_hashes=hashes,rdkit_version=rdBase.rdkitVersion,canonicalizer_commit=version,
        normalization='canonicalize_psmiles 0.1.2 plus RDKit isomeric SMILES; atom maps removed; two terminal markers required',
        normalization_limitations=['Not a universal periodic-equivalence proof; branched/ring repeat multiplication is not fully reduced by the upstream algorithm.','Unsupported attachment encodings retain graph-only identifiers; invalid inputs are counted separately.'],
        sampling_rule='One first saved source-row prediction per normalized identity within each disclosed cohort; no selection by target error. Conflicting targets remain in the raw source and are counted.',
        no_training_or_new_inference=True,no_source_split_changed=True,counts=summary,metrics=results,
        prediction_alignment=dict(matched_modes=modes,unmatched_count=len(unmatched)),
        all_original_bytes_unchanged=True,reported_final_prediction_rows=len(public_rows))
    write_json(AUDIT/'summary.json',report)
    print(json.dumps(dict(counts=summary,metrics=results,reported_rows=len(public_rows)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
