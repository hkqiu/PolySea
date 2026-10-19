"""Recompute the frozen primary and secondary FAB/LEC scores without model calls."""
from pathlib import Path
import copy
import hashlib
import json
import statistics
import sys
from frozen_scorer import score

ROOT = Path(__file__).resolve().parent


def parse_json(text):
    """Use the parser from the original experiment runner."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char == '{':
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                pass
    raise ValueError('No valid JSON object in visible answer')


def normalize_for_secondary_audit(raw):
    """Apply the two uniform, recorded formatting adaptations; do not fill values."""
    fixed = copy.deepcopy(raw)
    parsed = fixed.get('parsed_answer')
    if isinstance(parsed, dict):
        if 'results' not in parsed and any(k in parsed for k in
                ['train_fraction', 'target_eV', 'classes', 'temperatures', 'descriptor_count']):
            fixed['parsed_answer'] = {
                'results': parsed, 'citations': parsed.get('citations', []),
                'explanation': parsed.get('explanation', ''),
                'needs_clarification': parsed.get('needs_clarification')}
            parsed = fixed['parsed_answer']
        results = parsed.get('results')
        if (isinstance(results, dict) and isinstance(results.get('classes'), list)
                and results['classes'] and all(isinstance(x, dict) and 'class' in x
                                               for x in results['classes'])):
            results['classes'] = [x['class'] for x in results['classes']]
    return fixed


def main():
    study = json.loads((ROOT / 'study_results.json').read_text(encoding='utf-8'))
    requests = json.loads((ROOT / 'requests.json').read_text(encoding='utf-8'))
    registry = json.loads((ROOT / 'evidence_index.json').read_text(encoding='utf-8'))
    evidence = {r['id'].upper(): r for r in registry}
    cases = {r['id']: r for r in study['tasks']}
    outputs = {r['run_id']: r for r in study['visible_outputs']}
    expected_primary = {r['run_id']: r for r in study['primary_scores']}
    expected_audited = {r['run_id']: r for r in study['audited_scores']}
    assert len(cases) == 12 and len(outputs) == 96 and len(requests) == 96
    assert set(outputs) == set(expected_primary) == set(expected_audited) == {r['run_id'] for r in requests}
    primary, audited = [], []
    for run_id, saved in outputs.items():
        raw = dict(saved)
        if raw['json_valid']:
            raw['parsed_answer'] = parse_json(raw['answer'])
        actual = score(cases[raw['case_id']], raw, evidence)
        secondary = score(cases[raw['case_id']], normalize_for_secondary_audit(raw), evidence)
        assert actual == expected_primary[run_id], ('Primary score mismatch', run_id)
        assert secondary == expected_audited[run_id], ('Secondary score mismatch', run_id)
        primary.append(actual)
        audited.append(secondary)
    for request in requests:
        output = outputs[request['run_id']]
        assert all(request[k] == output[k] for k in ['case_id', 'condition', 'replicate'])
        condition = request['condition']
        trace = request['context_trace']
        assert trace['fab_enabled'] == (condition not in ['without_FAB', 'without_both'])
        assert trace['lec_enabled'] == (condition not in ['without_LEC', 'without_both'])
        assert request['user'] == cases[request['case_id']]['prompt']
    summary = []
    for condition in ['full', 'without_FAB', 'without_LEC', 'without_both']:
        rows = [r for r in primary if r['condition'] == condition]
        assert len(rows) == 24
        summary.append({
            'condition': condition, 'n': len(rows),
            'all_checks_passed': sum(r['complete_task'] for r in rows),
            'content_mean': statistics.mean(r['content_score'] for r in rows),
            'memory_mean': statistics.mean(r['memory_score'] for r in rows if r['memory_score'] is not None),
            'source_identified': sum(r['source_identified'] for r in rows if r['source_identified'] is not None),
            'valid_json': sum(r['valid_json'] for r in rows),
        })
    assert [r['all_checks_passed'] for r in summary] == [18, 8, 8, 1]
    assert all(a['complete_task'] == b['complete_task'] for a, b in zip(primary, audited))
    if '--json' in sys.argv:
        print(json.dumps({'verified_answers': 96, 'primary_and_audited_scores_match': True,
                          'summary': summary}, indent=2))
    else:
        print('Verified all 96 primary scores and all 96 secondary-audit scores.')
        print('Condition          All checks passed   Parseable JSON')
        for row in summary:
            print(f"{row['condition']:<18} {row['all_checks_passed']:>2}/24               {row['valid_json']:>2}/24")
        print('No model calls were made. Empty and length-limited outputs remain included.')


if __name__ == '__main__':
    main()
