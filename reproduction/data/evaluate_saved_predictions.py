"""Recompute metrics from public reported data, optionally checking local full data.

Python standard library only. Source CSV bytes and values are never modified.
All row identifiers are 1-based CSV data-record ordinals, excluding the header.
The saved error column is preserved in exports but never used for scoring or
selection. No pooled metric across incompatible regression units is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re

REGRESSION = ('Tg', 'bandgap_crystal', 'atomization_energy')
UNITS = {'Tg': 'degC', 'bandgap_crystal': 'eV', 'atomization_energy': 'eV/atom'}
NUMBER = re.compile(r'([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?)\s*(?:eV(?:/atom)?|℃|°C)?')


def read_csv(path):
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode('utf-8-sig'), 'utf-8-sig'
    except UnicodeDecodeError:
        text, encoding = raw.decode('gb18030'), 'gb18030'
    reader = csv.DictReader(io.StringIO(text, newline=''))
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV: {path.name}')
    return rows, reader.fieldnames, {
        'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
        'encoding': encoding, 'data_rows': len(rows), 'columns': reader.fieldnames,
    }


def task_of(prompt):
    value = prompt.casefold()
    rules = {
        'Tg': ('tg', 'glass transition', '玻璃'),
        'atomization_energy': ('atomization', 'binding energy', '原子', '解离能'),
        'bandgap_crystal': ('bandgap', 'band gap', '带隙', '能带'),
        'heat_class': ('heat resistance', 'temperature endurance', 'thermal', '耐热', '热稳定'),
    }
    matches = [task for task, terms in rules.items() if any(term in value for term in terms)]
    if len(matches) != 1:
        raise ValueError('Unrecognized or ambiguous property prompt')
    return matches[0]


def numeric(value):
    match = NUMBER.fullmatch(value.strip().replace('−', '-'))
    if not match:
        raise ValueError('Non-numeric value or unexpected unit suffix')
    result = Decimal(match.group(1))
    if not result.is_finite():
        raise ValueError('Nonfinite value')
    return result


def regression_metrics(rows):
    actual = [float(numeric(row['target'])) for row in rows]
    predicted = [float(numeric(row['PolySea'])) for row in rows]
    n = len(actual)
    mean = math.fsum(actual) / n
    errors = [p - y for y, p in zip(actual, predicted)]
    sse = math.fsum(error * error for error in errors)
    sst = math.fsum((value - mean) ** 2 for value in actual)
    return {'n_rows': n, 'MAE': math.fsum(abs(error) for error in errors) / n,
            'RMSE': math.sqrt(sse / n), 'R2': 1 - sse / sst if sst else None,
            'R2_denominator': 'target variance within the scored subset',
            'target_range': [min(actual), max(actual)],
            'prediction_range': [min(predicted), max(predicted)]}


def classification_metrics(rows):
    labels = ['class 1', 'class 2', 'class 3']
    matrix = {target: {predicted: 0 for predicted in labels} for target in labels}
    for row in rows:
        if row['target'] not in labels or row['PolySea'] not in labels:
            raise ValueError('Unexpected class; no invalid rows may be silently dropped')
        matrix[row['target']][row['PolySea']] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        predicted = sum(matrix[other][label] for other in labels)
        support = sum(matrix[label].values())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[label] = {'support': support, 'precision': precision, 'recall': recall, 'F1': f1}
    return {'n_rows': len(rows), 'accuracy': sum(matrix[label][label] for label in labels) / len(rows),
            'macro_F1': sum(item['F1'] for item in per_class.values()) / len(labels),
            'weighted_F1': sum(item['F1'] * item['support'] for item in per_class.values()) / len(rows),
            'per_class': per_class, 'confusion_matrix_target_rows_prediction_columns': matrix}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Released dataset/polyopus directory')
    parser.add_argument('--output-dir', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--full-regression', type=Path,
                        help='Optional complete original regression CSV supplied locally; not included in the public release')
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('--output-dir must be absent or empty')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def load(name):
        rows, fields, details = read_csv(args.data_dir / name)
        inputs[name] = details
        return rows, fields

    train, _ = load('train/merged_train.csv')
    test, _ = load('test/merged_test.csv')
    reported, reg_fields = load('predictions/reported/polysea_gen_v0331_reg_reported_AE_training_range.csv')
    flags, flag_fields = load('predictions/reported/regression_row_filter_flags.csv')
    classification, _ = load('predictions/raw/polysea_gen_v0327_cls.csv')
    summary_path = args.data_dir / 'predictions/reported/archived_full_regression_summary.json'
    summary_raw = summary_path.read_bytes()
    archived = json.loads(summary_raw)
    inputs['predictions/reported/archived_full_regression_summary.json'] = {
        'bytes': len(summary_raw), 'sha256': hashlib.sha256(summary_raw).hexdigest()}
    prompt_tasks = {row['prompt']: task_of(row['prompt']) for row in test}
    ordered_prompts = sorted(prompt_tasks, key=len, reverse=True)
    original_counts = Counter(prompt_tasks[row['prompt']] for row in test)

    def prediction_task(row):
        for prompt in ordered_prompts:
            if row['prompt-new'].startswith(prompt + ' '):
                return prompt_tasks[prompt]
        raise ValueError('Saved prediction does not match any archived test prompt')

    ae_train = [numeric(row['target']) for row in train
                if any(term in row['prompt'].casefold()
                       for term in ('atomization', 'binding energy', '原子', '解离能'))]
    lower, upper = min(ae_train), max(ae_train)
    if (len(ae_train), lower, upper) != (52650, Decimal('-6.82'), Decimal('-5.01')):
        raise ValueError('Training-label range differs from the archived release')
    if flag_fields != ['source_data_row', 'task', 'keep_reported']:
        raise ValueError('Public filter flags must contain only ordinal, task and retained flag')
    if [int(row['source_data_row']) for row in flags] != list(range(1, len(flags) + 1)):
        raise ValueError('Filter record ordinals must be complete and ordered')
    if any(row['keep_reported'] not in {'0', '1'} or row['task'] not in REGRESSION for row in flags):
        raise ValueError('Invalid public filter flags')
    saved_counts = Counter(row['task'] for row in flags)
    retained_flags = [row for row in flags if row['keep_reported'] == '1']
    if len(retained_flags) != len(reported):
        raise ValueError('Public row flags do not account for all reported rows')
    grouped, stale_errors = defaultdict(list), Counter()
    for flag, row in zip(retained_flags, reported):
        task = prediction_task(row)
        if flag['task'] != task:
            raise ValueError('Reported task order differs from retained row flags')
        target, predicted = numeric(row['target']), numeric(row['PolySea'])
        if task == 'atomization_energy' and not lower <= predicted <= upper:
            raise ValueError('Published AE prediction fails the disclosed range rule')
        grouped[task].append(row)
        if abs(numeric(row['error']) - abs(predicted - target)) > Decimal('0.00000001'):
            stale_errors[task] += 1
    if dict(saved_counts) != {'Tg': 7000, 'bandgap_crystal': 4800, 'atomization_energy': 5795}:
        raise ValueError('Unexpected archived regression row counts')
    if {task: len(rows) for task, rows in grouped.items()} != {
        'Tg': 7000, 'bandgap_crystal': 4800, 'atomization_energy': 1806}:
        raise ValueError('Unexpected public reporting subset')
    if any(row['task'] != 'atomization_energy' for row in flags if row['keep_reported'] == '0'):
        raise ValueError('Only AE rows may be excluded')
    if any(prediction_task(row) != 'heat_class' for row in classification):
        raise ValueError('Non-classification row in classification file')

    metrics = {
        'metric_weighting': 'saved instruction rows; augmented rows are not independent polymers',
        'regression': {}, 'classification': classification_metrics(classification),
        'reported_error_column_mismatch_counts': dict(stale_errors),
        'saved_error_column_used_for_selection_or_metrics': False,
        'full_original_regression_values_in_public_package': False,
        'archived_all_saved_regression_metrics': {
            'status': 'archived summary only; not recomputed without --full-regression',
            'full_original_sha256': archived['full_regression_file_sha256'],
            'metrics': archived['all_saved_outputs'],
        },
    }
    for task in REGRESSION:
        total, saved, subset = original_counts[task], saved_counts[task], grouped[task]
        metrics['regression'][task] = {
            'units': UNITS[task], 'original_input_rows': total,
            'saved_output_rows_from_public_flags': saved, 'missing_saved_outputs': total - saved,
            'saved_output_coverage_of_original_inputs': saved / total,
            'reported_subset': regression_metrics(subset),
            'reported_rows': len(subset), 'excluded_saved_rows': saved - len(subset),
            'reported_coverage_of_saved_outputs': len(subset) / saved,
            'reported_coverage_of_original_inputs': len(subset) / total,
        }
    metrics['classification'].update({
        'original_input_rows': original_counts['heat_class'],
        'missing_saved_outputs': original_counts['heat_class'] - len(classification),
        'saved_output_coverage_of_original_inputs': len(classification) / original_counts['heat_class'],
    })
    if args.full_regression:
        full, full_fields, details = read_csv(args.full_regression)
        inputs['optional_local_full_regression'] = details
        if details['sha256'] != archived['full_regression_file_sha256'] or full_fields != reg_fields:
            raise ValueError('Optional full file differs from the archived original SHA256/schema')
        if len(full) != len(flags):
            raise ValueError('Optional full-file row count differs from public flags')
        full_groups, retained = defaultdict(list), []
        for flag, row in zip(flags, full):
            task = prediction_task(row)
            keep = task != 'atomization_energy' or lower <= numeric(row['PolySea']) <= upper
            if flag['task'] != task or int(flag['keep_reported']) != int(keep):
                raise ValueError('Optional full-file decision differs from public row flags')
            full_groups[task].append(row)
            if keep:
                retained.append(row)
        if retained != reported:
            raise ValueError('Public reported rows differ from the exact full-file subset')
        full_results = {task: regression_metrics(rows) for task, rows in full_groups.items()}
        for task, result in full_results.items():
            for key in ['n_rows', 'MAE', 'RMSE', 'R2']:
                if abs(result[key] - archived['all_saved_outputs'][task][key]) > 1e-12:
                    raise ValueError('Recomputed full metrics differ from the archived summary')
        metrics['archived_all_saved_regression_metrics'] = {
            'status': 'recomputed from optional user-supplied local full file; exact subset and all flags verified',
            'full_original_sha256': details['sha256'], 'metrics': full_results}
    write_json(args.output_dir / 'metrics.json', metrics)
    raw = (args.output_dir / 'metrics.json').read_bytes()
    write_json(args.output_dir / 'evaluation_manifest.json', {
        'inputs': inputs, 'outputs': {'metrics.json': {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}})
    print(json.dumps({'regression_reported_recomputed': len(reported),
                      'regression_full_count_from_flags': len(flags),
                      'AE_excluded_count_from_flags': 3989, 'classification_saved': len(classification),
                      'optional_full_regression_recomputed': bool(args.full_regression)}))


if __name__ == '__main__':
    main()
