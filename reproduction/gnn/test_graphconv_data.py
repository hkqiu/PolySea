"""Data-integrity tests; these do not simulate model training or predictions."""
import argparse
import csv
from pathlib import Path
import tempfile
import unittest

from train_graphconv import graph_smiles, assign_splits, read_rows, metrics


def row(smi, label=None, source_id='', policy='preserve'):
    identity, model = graph_smiles(smi, policy)
    result = dict(csv_line=2, source_id=source_id, canonical_smiles=identity, model_smiles=model)
    if label: result['split'] = label
    return result


class DataIntegrityTests(unittest.TestCase):
    def test_supplied_split_rejects_alternative_smiles(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assign_splits([row('CCO','train'),row('OCC','test'),row('CCC','valid')],True,42,.8,.1)

    def test_capping_collision_cannot_cross_splits(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assign_splits([row('*CC*','train',policy='hydrogen_cap'),row('CC','test',policy='hydrogen_cap'),row('CO','valid')],True,42,.8,.1)

    def test_source_identity_ties_different_graph_representations(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            assign_splits([row('*CC*','train','P1'),row('*CCCC*','test','P1'),row('CO','valid','P2')],True,42,.8,.1)

    def test_generated_splits_are_reproducible_and_keep_duplicates(self):
        a=[row('C'*i) for i in range(1,21)]+[row('C(C)C')]
        b=[dict(x) for x in a]
        assign_splits(a,False,42,.8,.1);assign_splits(b,False,42,.8,.1)
        self.assertEqual([x['split'] for x in a],[x['split'] for x in b])
        self.assertEqual(a[2]['split'],a[-1]['split'])
        self.assertEqual({x['split'] for x in a},{'train','valid','test'})

    def test_preprocessing_rejects_invalid_or_ambiguous_attachment(self):
        for value in ['not_a_smiles', '']:
            with self.assertRaises(ValueError): graph_smiles(value,'preserve')
        with self.assertRaises(ValueError): graph_smiles('C*C','hydrogen_cap')
        original, capped=graph_smiles('*CC*','hydrogen_cap')
        self.assertIn('*',original);self.assertEqual(capped,'CC')

    def test_missing_targets_are_only_dropped_explicitly(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'data.csv'
            with p.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['SMILES','chain_eV']);w.writerow(['CC','2.1']);w.writerow(['CO',''])
            args=argparse.Namespace(csv=p,encoding='utf-8',smiles_column='SMILES',target_column='chain_eV',
                                    split_column=None,id_column=None,wildcard_policy='preserve',missing_target='error')
            with self.assertRaises(ValueError): read_rows(args)
            args.missing_target='drop'
            rows,dropped=read_rows(args)
            self.assertEqual(len(rows),1);self.assertEqual(dropped,[{'csv_line':3,'reason':'missing selected target'}])

    def test_metrics_do_not_discard_nonfinite_outputs(self):
        with self.assertRaises(ValueError):metrics([1.,2.],[1.,float('nan')])
        result=metrics([1.,2.,3.],[2.,2.,2.])
        self.assertAlmostEqual(result['MAE_eV'],2/3)
        self.assertAlmostEqual(result['R2'],0.)


if __name__=='__main__': unittest.main()
