"""Prespecified structured-answer scoring; no condition-dependent rules."""
from pathlib import Path
import json,re,csv,math,statistics,hashlib

ROOT=Path(__file__).resolve().parents[1]
def number(v):
    if isinstance(v,bool) or v is None:return None
    try:return float(v)
    except (TypeError,ValueError):
        m=re.search(r'[-+]?\d+(?:\.\d+)?',str(v))
        return float(m.group()) if m else None
def close(v,target,tol=.02):
    v=number(v);return v is not None and math.isfinite(v) and abs(v-target)<=tol
def ids(v):
    if isinstance(v,list):return {str(x).strip().lower() for x in v}
    return set(re.findall(r'\br[1-4]\b|\b[A-C]\b',str(v or ''),re.I))
def bool_is(v,b):return v is b
def lower(v):return str(v or '').lower()
def symbols(v):
    if not isinstance(v,list):return set()
    return {str(x).strip().upper() for x in v if str(x).strip().upper() not in {'H','*'}}
def labels(v):
    if not isinstance(v,list):return []
    return [number(x) for x in v]
def temp_checks(r,expected_inputs):
    rows=r.get('temperatures')
    if not isinstance(rows,list) or len(rows)!=len(expected_inputs):return False,False
    consistent=True;kelvin=True
    for row,c in zip(rows,expected_inputs):
        if not isinstance(row,dict):return False,False
        u=lower(row.get('unit')).replace('°','').strip()
        inp=close(row.get('input_c'),c)
        if u in {'k','kelvin'}:target=c+273.15
        elif u in {'c','celsius','degrees celsius','degc'}:target=c
        elif u in {'f','fahrenheit'}:target=c*9/5+32
        else:target=float('inf')
        consistent &= inp and close(row.get('value'),target)
        kelvin &= inp and u in {'k','kelvin'} and close(row.get('value'),c+273.15)
    return bool(consistent),bool(kelvin)
def energy_consistent(r):
    t=number(r.get('target_eV'));e=r.get('absolute_errors_eV')
    return t is not None and isinstance(e,dict) and all(close(e.get(k),abs(v-t)) for k,v in {'A':6.58,'B':6.21,'C':7.9}.items())
def parent_rule(r):return bool(re.search(r'parent|polymer\s*identity|equivalent.*polymer|原始|父',lower(r.get('grouping_key'))))
def add_screen(science,r):
    e=r.get('elements_by_candidate') or {}
    for c,syms in {'A':{'C'},'B':{'C','F'},'C':{'C','O'}}.items():science['elements_'+c]=symbols(e.get(c))==syms
def add_hrc(science,r):
    science['classes']=labels(r.get('classes'))==[3.,2.,1.]
    science['tg_basis']=bool(re.search(r'\bt[_ -]?g\b|glass.*transition',lower(r.get('underlying_property'))))
    science['not_decomposition']=bool_is(r.get('decomposition_temperature_classes'),False)
def add_tao(science,r,full=True):
    science['15_descriptors']=close(r.get('descriptor_count'),15,0)
    science['rdkit']=bool(re.search(r'rdkit',lower(r.get('calculation_tool'))))
    if full:
        science['pi1m']='pi1m' in lower(r.get('structural_database')).replace(' ','')
        science['not_measured_bulk']=bool_is(r.get('experimentally_measured_bulk_labels'),False)
def add_target(memory,r):
    memory['target_6_5']=close(r.get('target_eV'),6.5)
    obj=lower(r.get('objective'))
    memory['nominal_absolute_objective']=bool(re.search(r'absolute|\|.*\||abs\(',obj)) and not ('maximize' in obj)
    ranking=r.get('ranking')
    memory['ranking']=isinstance(ranking,list) and [str(x).strip().upper() for x in ranking]==['A','B','C']
    err=r.get('absolute_errors_eV') or {}
    memory['errors']=all(close(err.get(k),v) for k,v in {'A':.08,'B':.29,'C':1.4}.items())
def source_check(parsed,doi,prefix,title_hint,source_records):
    citations=parsed.get('citations') or []
    text=json.dumps(citations,ensure_ascii=False).lower()
    if doi.lower() in text:return True
    if title_hint.lower() in text:return True
    for key in re.findall(r'\b(?:PNC|MT|TAO)_\d{3}_\d+\b',json.dumps(citations),re.I):
        if key.upper() in source_records and key.upper().startswith(prefix):return True
    return False

def score(case,raw,records):
    parsed=raw.get('parsed_answer') if raw.get('json_valid') else {}
    if not isinstance(parsed,dict):parsed={}
    r=parsed.get('results') or {}
    if not isinstance(r,dict):r={}
    scientific={};memory={};source=None
    cid=case['id']
    if cid in {'M1','C1'}:
        valid,k=temp_checks(r,[25,100,200] if cid=='M1' else [280,350,420])
        scientific['temperature_arithmetic']=valid;memory['saved_kelvin_convention']=k
    if cid in {'M2','C3'}:
        add_screen(scientific,r);memory['halogen_free_selection']=ids(r.get('eligible_ids'))=={'a','c'}
    if cid in {'M3','C2'}:
        scientific['parent_grouping']=parent_rule(r)
        memory['holdout_parents']=ids(r.get('holdout_ids'))=={'r1','r2'}
        memory['eligible_train']=ids(r.get('train_ids'))=={'r3','r4'}
    if cid in {'M4','C4'}:
        scientific['arithmetic_for_reported_target']=energy_consistent(r)
        scientific['no_experimental_validation']=bool_is(r.get('experimental_dielectric_validation'),False)
        add_target(memory,r)
    if cid=='E1':
        scientific.update(train_fraction=close(r.get('train_fraction'),.9,.001),test_fraction=close(r.get('test_fraction'),.1,.001),
                          split_before_augmentation=bool_is(r.get('split_before_augmentation'),True),
                          alternative_smiles=bool(re.search('smiles',lower(r.get('augmentation_method')))) and bool(re.search(r'random|enumer|alternat|augment',lower(r.get('augmentation_method')))),
                          not_other_project_verification=bool_is(r.get('proves_other_project_no_leakage'),False))
    if cid=='C2':
        scientific.update(train_fraction=close(r.get('published_train_fraction'),.9,.001),test_fraction=close(r.get('published_test_fraction'),.1,.001),
                          split_before_augmentation=bool_is(r.get('published_split_before_augmentation'),True))
    if cid=='E2':
        scientific.update(ev='ev' in lower(r.get('energy_unit')),per_atom='atom' in lower(r.get('normalization_basis')) and 'repeat' not in lower(r.get('normalization_basis')),
                          dft=bool(re.search(r'dft|density.functional',lower(r.get('data_source_type')))),n390=close(r.get('record_count'),390,0),
                          needs_atom_counts=bool_is(r.get('conversion_without_atom_counts_possible'),False))
    if cid in {'E3','C1'}:add_hrc(scientific,r)
    if cid in {'E4','C3','C4'}:add_tao(scientific,r,full=(cid!='C4'))
    if cid in {'E1','E3','C1','C2'}:
        source=source_check(parsed,'10.1039/D3SC05079C','PNC','natural and chemical language model',records)
    if cid=='E2':source=source_check(parsed,'10.1016/j.patter.2021.100238','MT','polymer informatics with multi-task learning',records)
    if cid in {'E4','C3','C4'}:source=source_check(parsed,'10.1038/s41524-024-01466-5','TAO','on-demand reverse design of polymers',records)
    valid=bool(raw.get('success') and raw.get('json_valid'))
    allok=valid and all(scientific.values()) and all(memory.values()) and source is not False
    return {'run_id':raw['run_id'],'case_id':cid,'group':case['group'],'condition':raw['condition'],'replicate':raw['replicate'],
            'valid_json':valid,'content_score':sum(scientific.values())/len(scientific),
            'memory_score':sum(memory.values())/len(memory) if memory else None,'source_identified':int(source) if source is not None else None,
            'complete_task':int(allok),'needs_clarification':parsed.get('needs_clarification'),
            'content_checks':scientific,'memory_checks':memory,'elapsed_s':raw.get('elapsed_s'),'usage':raw.get('usage',{}),
            'finish_reason':raw.get('finish_reason')}

def evaluate():
    cases={c['id']:c for c in json.loads((ROOT/'cases.json').read_text(encoding='utf-8'))}
    records={r['id'].upper():r for r in [json.loads(l) for l in (ROOT/'snapshot'/'data'/'evidence'/'corpus.jsonl').read_text(encoding='utf-8').splitlines()]}
    rows=[]
    for file in sorted((ROOT/'raw').glob('*_r*.json')):
        raw=json.loads(file.read_text(encoding='utf-8'));rows.append(score(cases[raw['case_id']],raw,records))
    (ROOT/'results'/'scored_answers.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    if not rows:print('No answers yet');return
    fields=['run_id','case_id','group','condition','replicate','valid_json','content_score','memory_score','source_identified','complete_task','needs_clarification','elapsed_s','finish_reason']
    with (ROOT/'results'/'scored_answers.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    summaries=[]
    for condition in ['full','without_FAB','without_LEC','without_both']:
        rr=[r for r in rows if r['condition']==condition]
        if not rr:continue
        item={'condition':condition,'n_answers':len(rr),'n_tasks':len({r['case_id'] for r in rr})}
        for key in ['content_score','memory_score','source_identified','complete_task','valid_json']:
            vv=[float(r[key]) for r in rr if r[key] is not None];item[key]=statistics.mean(vv) if vv else None
        lat=[r['elapsed_s'] for r in rr if r['elapsed_s'] is not None];item['median_latency_s']=statistics.median(lat) if lat else None
        item['output_tokens']=sum(r['usage'].get('output_tokens',r['usage'].get('completion_tokens',0)) for r in rr)
        summaries.append(item)
    (ROOT/'results'/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':evaluate()
