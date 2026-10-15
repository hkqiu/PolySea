"""Reconstruct training-file augmentation using saved rows only (no test data).

Run with Python 3.12 and RDKit; source CSVs are never modified. All row numbers
are 1-based DATA RECORD ordinals, excluding the header, not physical line numbers.
"""
from pathlib import Path
from collections import Counter, defaultdict
from functools import lru_cache
from decimal import Decimal, InvalidOperation
import argparse
import csv
import hashlib
import io
import itertools
import json
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--train-dir', type=Path, required=True,
                    help='Directory containing all 21 archived training CSV files')
parser.add_argument('--output-dir', type=Path, required=True,
                    help='New or empty directory for reconstructed evidence')
args = parser.parse_args()
if args.output_dir.exists() and any(args.output_dir.iterdir()):
    parser.error('--output-dir must be absent or empty; archived results are never overwritten')
from rdkit import Chem, RDLogger, rdBase
RDLogger.DisableLog('rdApp.*')
csv.field_size_limit(100_000_000)
args.output_dir.mkdir(parents=True, exist_ok=True)
TASKS = ['Tg', 'bandgap_crystal', 'atomization_energy', 'heat_class']
BLOCK = {'Tg': 10, 'bandgap_crystal': 20, 'atomization_energy': 15, 'heat_class': 15}
CORE = [f'train{i}{suffix}.csv' for i in range(1, 6) for suffix in ('', '-CN')]
INVERSE = [f'band gap{i}.csv' for i in range(1, 6)]
INVERSE_ALL = INVERSE + ['band gap.csv', 'band gap_can_smiles.csv']
EXTRA = ['tg_raw.csv', 'PI-glass transition temperature.csv']
MERGED = ['merged_train.csv', 'merged_train copy.csv']
FILES = CORE + EXTRA + INVERSE_ALL + MERGED
manifest = []
data = {}

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write_json(name, obj):
    (args.output_dir / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def write_csv(name, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_dir / name).open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

for name in FILES:
    path = args.train_dir / name
    raw = path.read_bytes()
    try:
        text = raw.decode('utf-8-sig')
        encoding = 'utf-8-sig'
    except UnicodeDecodeError:
        text = raw.decode('gb18030')
        encoding = 'gb18030'
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    assert all(None not in row for row in rows), name
    data[name] = rows
    manifest.append({'file': name, 'bytes': len(raw), 'sha256': sha(raw), 'encoding': encoding,
                     'data_rows': len(rows), 'columns': reader.fieldnames})

@lru_cache(None)
def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms():
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)

NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
SCALAR = re.compile(r'(' + NUMBER + r')\s*(℃|°C|C|K|eV(?:\s*/\s*atom)?)?', re.I)
CLASS = re.compile(r'class\s*([123])\s*(eV)?', re.I)
VALUE_IN_PROMPT = re.compile(r'(' + NUMBER + r')\s*eV', re.I)

def decimal_text(value):
    val = Decimal(value)
    assert val.is_finite()
    return format(val.normalize(), 'f') if val else '0'

def label(target, task):
    target = target.strip().replace('−', '-')
    if task == 'heat_class':
        found = CLASS.fullmatch(target)
        if found:
            return 'class ' + found.group(1)
    found = SCALAR.fullmatch(target)
    if found:
        return decimal_text(found.group(1))
    return target

prompt_tasks = {}
for name in CORE:
    prompts = list(dict.fromkeys(r['prompt'] for r in data[name]))
    assert len(prompts) == 4, name
    for prompt, task in zip(prompts, TASKS):
        assert prompt not in prompt_tasks or prompt_tasks[prompt] == task
        prompt_tasks[prompt] = task
for name in EXTRA:
    for row in data[name]:
        assert row['prompt'] not in prompt_tasks or prompt_tasks[row['prompt']] == 'Tg'
        prompt_tasks[row['prompt']] = 'Tg'
inverse_prompts = {r['prompt'] for name in INVERSE_ALL for r in data[name]}

def task_of(row, name):
    if name in INVERSE_ALL or row['prompt'] in inverse_prompts:
        return 'inverse_design'
    return prompt_tasks.get(row['prompt'], 'QA_or_other')

def language(prompt):
    return 'ZH' if re.search(r'[\u4e00-\u9fff]', prompt) else 'EN'

def prompt_template(prompt, task):
    return VALUE_IN_PROMPT.sub('<bandgap_eV> eV', prompt) if task == 'inverse_design' else prompt

def graph_label(row, task):
    if task == 'QA_or_other':
        return '', '', ''
    if task == 'inverse_design':
        smiles = row['target'].strip()
        found = VALUE_IN_PROMPT.findall(row['prompt'])
        assert len(found) == 1, row
        value = decimal_text(found[0])
    else:
        smiles = row.get('SMILES', '').strip()
        value = label(row['target'], task)
    return smiles, canonical(smiles), value

def group_id(task, graph, value):
    return 'GL-' + sha(json.dumps([task, graph, value], ensure_ascii=False, separators=(',', ':')).encode())[:24]

registry = {}
row_cache = {}
for name, rows in data.items():
    annotated = []
    for number, row in enumerate(rows, 1):
        task = task_of(row, name)
        smiles, graph, value = graph_label(row, task)
        ident = group_id(task, graph, value) if graph else ''
        if ident:
            record = {'graph_label_group_id': ident, 'task': task, 'canonical_isomeric_smiles': graph,
                      'normalized_label': value}
            assert ident not in registry or registry[ident] == record
            registry[ident] = record
        annotated.append({'data_row': number, 'task': task, 'smiles': smiles, 'graph': graph,
                          'label': value, 'group_id': ident, 'language': language(row['prompt']),
                          'template': prompt_template(row['prompt'], task)})
    row_cache[name] = annotated

summaries, distributions, prompt_counts, group_counts = [], [], [], []

def summarize(scope, raw, records):
    by_task = defaultdict(list)
    for row, rec in zip(raw, records):
        by_task[rec['task']].append((row, rec))
    for task, paired in by_task.items():
        rs = [rec for _, rec in paired]
        valid = [rec for rec in rs if rec['graph']]
        metrics = {
            'exact_smiles': Counter(rec['smiles'] for rec in rs if rec['smiles']),
            'exact_smiles_label': Counter((rec['smiles'], rec['label']) for rec in rs if rec['smiles']),
            'canonical_graph': Counter(rec['graph'] for rec in valid),
            'canonical_graph_label': Counter(rec['group_id'] for rec in valid),
            'exact_full_record': Counter((row['prompt'], row.get('SMILES', ''), row['target']) for row, _ in paired),
        }
        entry = {'scope': scope, 'task': task, 'rows': len(rs), 'EN_rows': sum(r['language']=='EN' for r in rs),
                 'ZH_rows': sum(r['language']=='ZH' for r in rs), 'distinct_prompt_templates': len({r['template'] for r in rs}),
                 'valid_graph_rows': len(valid), 'invalid_graph_rows': len(rs)-len(valid) if task!='QA_or_other' else 0}
        for kind, counter in metrics.items():
            entry['distinct_' + kind] = len(counter)
            for multiplicity, groups in sorted(Counter(counter.values()).items()):
                distributions.append({'scope': scope, 'task': task, 'identity': kind,
                                      'row_multiplicity': multiplicity, 'groups': groups,
                                      'accounted_rows': multiplicity * groups})
        for ident, n in sorted(metrics['canonical_graph_label'].items()):
            group_counts.append({'scope': scope, 'task': task, 'graph_label_group_id': ident, 'rows': n})
        summaries.append(entry)
        for (lang, template), count in Counter((r['language'], r['template']) for r in rs).items():
            prompt_counts.append({'scope': scope, 'task': task, 'language': lang,
                                  'prompt_template': template, 'rows': count})

for name in FILES:
    summarize(name, data[name], row_cache[name])
summarize('CORE_TEN_FILES', [r for n in CORE for r in data[n]], [r for n in CORE for r in row_cache[n]])

base = data['train1.csv']
base_records = row_cache['train1.csv']
comparisons = []
base_sequence = [(r['SMILES'], r['target']) for r in base]
base_counter = Counter(base_sequence)
for name in CORE:
    sequence = [(r['SMILES'], r['target']) for r in data[name]]
    result = {'file': name, 'reference': 'train1.csv', 'same_exact_smiles_label_sequence': sequence == base_sequence,
              'same_exact_smiles_label_multiset': Counter(sequence) == base_counter,
              'same_canonical_graph_label_sequence': [r['group_id'] for r in row_cache[name]] == [r['group_id'] for r in base_records],
              'same_complete_records_including_prompt': data[name] == base}
    assert result['same_exact_smiles_label_sequence']
    comparisons.append(result)

blocks, block_summary, block_dist = [], [], []
base_row_parent = {}
for task in TASKS:
    records = [r for r in base_records if r['task'] == task]
    size = BLOCK[task]
    assert len(records) % size == 0
    run_lengths = Counter(len(list(run)) for _, run in itertools.groupby(r['group_id'] for r in records))
    task_blocks = []
    for start in range(0, len(records), size):
        chunk = records[start:start + size]
        assert len({r['group_id'] for r in chunk}) == 1 and chunk[0]['group_id']
        parent = f'CORE-{task}-{start // size + 1:06d}'
        smiles_counts = Counter(r['smiles'] for r in chunk)
        row = {'parent_surrogate_id': parent, 'task': task, 'reference_file': 'train1.csv',
               'start_data_row': chunk[0]['data_row'], 'end_data_row': chunk[-1]['data_row'],
               'rows_per_prompt_file': size, 'graph_label_group_id': chunk[0]['group_id'],
               'distinct_smiles_strings': len(smiles_counts), 'duplicate_string_rows_beyond_first': size-len(smiles_counts),
               'string_multiplicity_distribution': json.dumps(dict(sorted(Counter(smiles_counts.values()).items())))}
        for rec in chunk:
            base_row_parent[rec['data_row']] = parent
        blocks.append(row)
        task_blocks.append(row)
    parent_counts = Counter(r['graph_label_group_id'] for r in task_blocks)
    for count, n in sorted(Counter(parent_counts.values()).items()):
        block_dist.append({'task': task, 'blocks_per_graph_label': count, 'graph_label_groups': n})
    block_summary.append({'task': task, 'reconstructed_fixed_size_blocks': len(task_blocks),
                         'rows_per_block_per_file': size, 'rows_per_file': len(records), 'core_files': 10,
                         'rows_per_block_across_core_files': size*10, 'core_rows': len(records)*10,
                         'canonical_graphs': len({r['graph'] for r in records}),
                         'canonical_graph_label_groups': len(parent_counts),
                         'distinct_EN_prompt_templates': len({r['template'] for n in CORE if '-CN' not in n for r in row_cache[n] if r['task']==task}),
                         'distinct_ZH_prompt_templates': len({r['template'] for n in CORE if '-CN' in n for r in row_cache[n] if r['task']==task}),
                         'observed_contiguous_graph_label_run_length_distribution': dict(sorted(run_lengths.items())),
                         'within_block_distinct_smiles_count_distribution': dict(sorted(Counter(r['distinct_smiles_strings'] for r in task_blocks).items()))})

# Recover the exact ordered assembly. Match row sequences; infer observed target
# suffixes, never silently normalize a numerical value to force a source match.
merged = data['merged_train.csv']
segments, lineage, suffix_rows = [], [], []
cursor = 0

def exact_segment(name):
    global cursor
    src = data[name]
    assert merged[cursor:cursor+len(src)] == src, name
    segments.append({'segment': len(segments)+1, 'source_candidates': name,
                     'start_data_row': cursor+1, 'end_data_row': cursor+len(src),
                     'rows': len(src), 'match': 'exact complete row sequence'})
    for j, rec in enumerate(row_cache[name], 1):
        lineage.append({'merged_data_row': cursor+j, 'source_candidates': name, 'source_data_row': j,
                        'parent_surrogate_id': ('TGRAW' if name=='tg_raw.csv' else 'PI') + f'-ROW-{j:06d}',
                        'parent_basis': 'saved source row; augmentation history not inferred',
                        'graph_label_group_id': rec['group_id'], 'target_suffix_added': '', 'input_space_filled': False})
    cursor += len(src)

exact_segment('PI-glass transition temperature.csv')
exact_segment('tg_raw.csv')
used_candidates = Counter()
for segment_index in range(10):
    length = len(base)
    chunk = merged[cursor:cursor+length]
    matches = []
    transformations = {}
    for name in CORE:
        src = data[name]
        if all(m['prompt']==s['prompt'] and m['SMILES']==s['SMILES'] and m['target'].startswith(s['target']) for m,s in zip(chunk,src)):
            task_suffix = defaultdict(set)
            for m,s,rec in zip(chunk,src,row_cache[name]):
                task_suffix[rec['task']].add(m['target'][len(s['target']):])
            if all(len(values)==1 for values in task_suffix.values()):
                matches.append(name)
                transformations[name] = {task: next(iter(values)) for task,values in task_suffix.items()}
    assert matches, segment_index
    assert len({json.dumps(transformations[name],sort_keys=True) for name in matches}) == 1
    candidate_text = '|'.join(matches)
    used_candidates[candidate_text] += 1
    name = matches[0]
    suffixes = transformations[name]
    segments.append({'segment': len(segments)+1, 'source_candidates': candidate_text,
                     'start_data_row': cursor+1, 'end_data_row': cursor+length, 'rows': length,
                     'match': 'exact prompt/input/value sequence with verified task-specific target suffix'})
    for task in TASKS:
        suffix_rows.append({'segment': len(segments), 'source_candidates': candidate_text, 'task': task,
                            'suffix_added_to_target': suffixes[task],
                            'rows': sum(r['task']==task for r in row_cache[name])})
    for j, (m,s,rec) in enumerate(zip(chunk,data[name],row_cache[name]),1):
        assert m['target']==s['target']+suffixes[rec['task']]
        lineage.append({'merged_data_row': cursor+j, 'source_candidates': candidate_text, 'source_data_row': j,
                        'parent_surrogate_id': base_row_parent[j], 'parent_basis': 'fixed-size saved-order block',
                        'graph_label_group_id': rec['group_id'], 'target_suffix_added': suffixes[rec['task']],
                        'input_space_filled': False})
    cursor += length
assert used_candidates['train1.csv|train5.csv'] == 2
assert all(used_candidates[name]==1 for name in CORE if name not in ('train1.csv','train5.csv'))

inverse_length = len(data['band gap.csv'])
qa_length = len(merged)-cursor-inverse_length
qa = merged[cursor:cursor+qa_length]
assert all(task_of(row,'merged_train.csv')=='QA_or_other' for row in qa)
segments.append({'segment': len(segments)+1, 'source_candidates': 'unmatched_QA_or_other_in_merged',
                 'start_data_row': cursor+1, 'end_data_row': cursor+qa_length, 'rows': qa_length,
                 'match': 'residual contiguous records; original source file not recovered in train directory'})
for j in range(qa_length):
    lineage.append({'merged_data_row': cursor+j+1, 'source_candidates': 'unmatched_QA_or_other_in_merged',
                    'source_data_row': '', 'parent_surrogate_id': '', 'parent_basis': 'unresolved source',
                    'graph_label_group_id': '', 'target_suffix_added': '', 'input_space_filled': False})
cursor += qa_length
chunk = merged[cursor:]
src = data['band gap.csv']
assert len(chunk) == len(src)
assert all(m['prompt']==s['prompt'] and m['target']==s['target'] and m['SMILES']==' ' and s['SMILES']=='' for m,s in zip(chunk,src))
segments.append({'segment': len(segments)+1, 'source_candidates': 'band gap.csv',
                 'start_data_row': cursor+1, 'end_data_row': len(merged), 'rows': len(src),
                 'match': 'exact prompt/target sequence; empty SMILES input changed to one space'})
for j, rec in enumerate(row_cache['band gap.csv'],1):
    lineage.append({'merged_data_row': cursor+j, 'source_candidates': 'band gap.csv', 'source_data_row': j,
                    'parent_surrogate_id': f'INVERSE-ROW-{(j-1)%len(data[INVERSE[0]])+1:06d}',
                    'parent_basis': 'aligned row in each of five inverse template files',
                    'graph_label_group_id': rec['group_id'], 'target_suffix_added': '', 'input_space_filled': True})
assert len(lineage)==len(merged)
assert [r['merged_data_row'] for r in lineage]==list(range(1,len(merged)+1))
assert data['merged_train copy.csv'] == merged[len(data[EXTRA[1]]):]

inverse_base = row_cache[INVERSE[0]]
inverse_checks = []
for name in INVERSE:
    sequence = [(r['smiles'],r['label']) for r in row_cache[name]]
    assert sequence == [(r['smiles'],r['label']) for r in inverse_base]
    inverse_checks.append({'file':name,'same_target_smiles_and_requested_bandgap_sequence_as_band_gap1':True,
                           'rows':len(sequence),'language':row_cache[name][0]['language']})
assert [r for name in INVERSE for r in data[name]] == [{k:v for k,v in r.items() if k!='SMILES'} for r in data['band gap.csv']]
assert [{k:v for k,v in r.items() if k!='can_smiles'} for r in data['band gap_can_smiles.csv']] == data[INVERSE[0]]
canonical_column_exact = sum(r['can_smiles']==canonical(r['target']) for r in data['band gap_can_smiles.csv'])
canonical_column_same_graph = sum(canonical(r['can_smiles'])==canonical(r['target']) for r in data['band gap_can_smiles.csv'])
inverse_parent = []
for rec in inverse_base:
    inverse_parent.append({'parent_surrogate_id': f'INVERSE-ROW-{rec["data_row"]:06d}',
                           'band_gap1_data_row': rec['data_row'], 'graph_label_group_id': rec['group_id'],
                           'requested_bandgap_eV': rec['label'], 'rows_across_five_prompt_files': 5})

# Independently account for each merged task using source counts, without deduplication.
merged_task_counts = Counter(r['task'] for r in row_cache['merged_train.csv'])
expected_tasks = Counter(r['task'] for name in CORE+EXTRA+['band gap.csv'] for r in row_cache[name])
expected_tasks['QA_or_other'] += qa_length
assert merged_task_counts == expected_tasks
source_unchanged = all(sha((args.train_dir/entry['file']).read_bytes())==entry['sha256'] for entry in manifest)
assert source_unchanged
for scope in FILES+['CORE_TEN_FILES']:
    for task in TASKS+['inverse_design']:
        row = next((s for s in summaries if s['scope']==scope and s['task']==task), None)
        if row:
            accounted = sum(d['accounted_rows'] for d in distributions if d['scope']==scope and d['task']==task and d['identity']=='canonical_graph_label')
            assert accounted == row['valid_graph_rows']

report = {
    'scope': 'train directory only; no training, inference, or train/test comparisons',
    'source_train_directory': 'caller-supplied --train-dir; see source_file_manifest.json', 'rdkit_version': rdBase.rdkitVersion,
    'identity_rule': 'RDKit sanitized canonical isomeric molecular graph; no periodic repeat-unit equivalence, tautomer normalization, salt stripping, or stereochemistry removal',
    'label_rule': 'Within a task, exact numeric value normalized with Decimal and documented stored unit suffix removed; class 1/2/3 parsed even when saved with erroneous eV suffix. This is not a physical unit conversion. Raw full-record multiplicities are also reported.',
    'row_number_rule': 'All row indices are 1-based CSV data-record ordinals, excluding the header; multiline records are not counted as extra rows.',
    'parent_rule': 'CORE IDs identify homogeneous fixed-size blocks in train1 saved order. Graph-label IDs identify deduplicated task+canonical-graph+normalized-label groups. Additional Tg and inverse IDs identify saved rows. None is a verified original measurement ID.',
    'core': block_summary,
    'cross_file_checks': comparisons,
    'core_distinct_template_note': 'Five English files contain four distinct templates per task because train5.csv equals train1.csv; five Chinese files contain five distinct templates. Ten file copies therefore correspond to nine distinct templates.',
    'merged_composition': {'total_rows':len(merged),'core_rows':sum(len(data[n]) for n in CORE),
                           'tg_raw_rows':len(data[EXTRA[0]]),'PI_Tg_rows':len(data[EXTRA[1]]),
                           'inverse_design_rows':inverse_length,'QA_or_other_rows':qa_length,
                           'tasks':dict(merged_task_counts),'ordered_segments':segments,
                           'merged_copy_equals_final_without_leading_PI_rows':True,
                           'train1_train5_source_filename_ambiguity':'The two exact-identical English file copies cannot be individually assigned to their two merged segments; either assignment gives identical records and parent IDs.'},
    'inverse_design': {'rows_per_file':len(inverse_base),'files':5,'EN_templates':3,'ZH_templates':2,
                       'checks':inverse_checks,'band_gap_csv_equals_five_file_concatenation':True,
                       'canonical_column_exact_rdkit_string_matches':canonical_column_exact,
                       'canonical_column_same_rdkit_graph_matches':canonical_column_same_graph,
                       'band_gap_can_smiles_is_annotation_of_band_gap1':True},
    'verification': {'all_assertions_passed':True,'source_hashes_unchanged':source_unchanged,
                     'merged_rows_with_lineage':len(lineage),
                     'merged_rows_with_matched_component_source':len(merged)-qa_length,
                     'QA_rows_with_source_unresolved':qa_length,
                     'task_totals_independently_equal_component_sum':True,
                     'multiplicity_histograms_account_for_all_valid_graph_rows':True,
                     'script_sha256':sha(Path(__file__).read_bytes())},
    'limitations': [
        'Fixed-size block multiplicity is directly supported by saved rows, but chronological augmentation code execution and original measurement counts are not recovered from these CSVs.',
        'Repeated canonical graph-label blocks can be duplicate source rows or independently recorded identical labels; saved train rows alone do not distinguish those cases.',
        'Different labels for one graph remain separate graph-label groups; no label aggregation or curation is performed.',
        'Additional tg_raw and PI rows have no uniform saved SMILES-enumeration block factor established here; their filenames do not prove that their rows are raw measurements.',
        'Inverse design contains five aligned prompt copies per saved base row. Any augmentation before band gap1.csv is not inferable from this factor.',
        'The units appended in merged targets are formatting evidence. In particular, class labels in the two identical English segments incorrectly carry an eV suffix; this audit preserves and reports that fact.',
    ]}
write_json('reconstruction_summary.json', report)
write_json('source_file_manifest.json', manifest)
write_csv('file_task_summary.csv', summaries)
write_csv('prompt_language_counts.csv', prompt_counts)
write_csv('multiplicity_distributions.csv', distributions)
write_csv('core_parent_blocks.csv', blocks)
write_csv('core_blocks_per_graph_label.csv', block_dist)
write_csv('graph_label_registry.csv', registry.values())
write_csv('graph_label_counts.csv', group_counts)
write_csv('merged_ordered_segments.csv', segments)
write_csv('merged_target_suffixes.csv', suffix_rows)
write_csv('merged_row_lineage.csv', lineage)
write_csv('inverse_parent_rows.csv', inverse_parent)

# A merged-training registry links reconstructed parent surrogates to graph-label
# groups without treating them as verified original measurement identifiers.
parent_registry = {}
for record in lineage:
    ident = record['parent_surrogate_id']
    if not ident:
        continue
    fixed = {'parent_surrogate_id': ident,
             'parent_basis': record['parent_basis'],
             'graph_label_group_id': record['graph_label_group_id']}
    if ident not in parent_registry:
        parent_registry[ident] = {**fixed, 'merged_rows': 0}
    assert all(parent_registry[ident][key] == value for key, value in fixed.items())
    parent_registry[ident]['merged_rows'] += 1
write_csv('parent_registry.csv', parent_registry.values())

print(json.dumps({'output':str(args.output_dir),'rdkit_version':rdBase.rdkitVersion,'core':block_summary,
                  'merged_task_counts':dict(merged_task_counts),'verification':report['verification']},ensure_ascii=True))
