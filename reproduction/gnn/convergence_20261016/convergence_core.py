"""Validation-selected convergence experiment on the immutable 20261015 split."""
from pathlib import Path
import csv
import datetime
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time

ROOT = Path(os.environ.get('GNN_EXPERIMENT_ROOT', str(Path(__file__).absolute().parent)))
for key, value in {'TF_CPP_MIN_LOG_LEVEL':'2', 'OMP_NUM_THREADS':'8',
                   'MKL_NUM_THREADS':'8', 'OPENBLAS_NUM_THREADS':'8',
                   'TF_NUM_INTRAOP_THREADS':'8', 'TF_NUM_INTEROP_THREADS':'1',
                   'CUDA_VISIBLE_DEVICES':'-1'}.items():
    os.environ.setdefault(key, value)
sys.path.insert(0, str(ROOT))
import graphconv_helpers as helper


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(25):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 24:
                raise
            time.sleep(0.2)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    started = time.perf_counter()
    protocol = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    amendment_path=ROOT/'execution_amendment_01.json'
    amendment=json.loads(amendment_path.read_text(encoding='utf-8')) if amendment_path.exists() else None
    if (ROOT/'selection.json').exists() or ((ROOT/'continued_unscaled').exists() and not amendment):
        raise RuntimeError('Existing run outputs will not be overwritten')
    rows = read_csv(ROOT/'baseline_10epoch/split_manifest.csv')
    for row in rows:
        row['csv_line'] = int(row['csv_line'])
        row['target_eV'] = float(row['target_eV'])
    assert helper.sha256(ROOT/'baseline_10epoch/split_manifest.csv') == protocol['split_sha256']
    helper.assign_splits(rows, True, 42, 0.8, 0.1)
    subsets = {s:[r for r in rows if r['split']==s] for s in ['train','valid','test']}
    assert {s:len(r) for s,r in subsets.items()} == protocol['fixed_counts']
    # Test rows are not featurized or predicted until the selection is locked below.
    np, dc = helper.backend(42)
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    features = {s:helper.dataset(subsets[s],np,dc) for s in ['train','valid']}
    train_y = np.array([r['target_eV'] for r in subsets['train']], dtype=float)
    mean, std = float(np.mean(train_y)), float(np.std(train_y))
    assert std > 0
    dump(ROOT/'target_scaling.json', {'fit_partition':'train', 'n':len(train_y),
         'mean_eV':mean,'std_eV_ddof0':std,'min_eV':float(train_y.min()),'max_eV':float(train_y.max())})
    config = json.loads((ROOT/'baseline_10epoch/artifact.json').read_text(encoding='utf-8'))['model_config']
    results = []
    for variant in protocol['variants']:
        name = variant['name']; run = ROOT/name
        resuming=bool(amendment and name=='continued_unscaled')
        if not resuming: run.mkdir()
        tf.keras.backend.clear_session()
        random.seed(42); np.random.seed(42); tf.random.set_seed(42)
        scale = {'mean_eV':mean,'std_eV':std} if name=='standardized_from_scratch' else {'mean_eV':0.0,'std_eV':1.0}
        fit_data = dc.data.NumpyDataset(features['train'].X,
                   (features['train'].y-scale['mean_eV'])/scale['std_eV'],
                   features['train'].w, features['train'].ids)
        model = dc.models.GraphConvModel(model_dir=str(run/'working_checkpoint'), **config)
        first = amendment['restore_epoch'] if resuming else variant['start_epoch']
        best_score = float('inf'); best_epoch = None; patience_score = float('inf'); wait = 0
        curve=[];elapsed_offset=0.0
        if resuming:
            old_curve=read_csv(ROOT/'interrupted_attempt_01/learning_curve.csv')
            for row in old_curve:
                if int(row['epoch'])<=first:
                    curve.append({k:(int(v) if k=='epoch' else (float(v) if v else None)) for k,v in row.items()})
            elapsed_offset=curve[-1]['elapsed_s']

        def evaluate(split):
            raw = np.asarray(model.predict(features[split])).reshape(-1)
            pred = raw*scale['std_eV']+scale['mean_eV']
            return pred, helper.metrics([r['target_eV'] for r in subsets[split]], pred.tolist())

        if first:
            model.restore(model_dir=str(run/'resume_checkpoint' if resuming else ROOT/'baseline_10epoch/checkpoint'))
            old_pred = np.array([float(r['predicted_band_gap_eV']) for r in read_csv(run/'best_valid_predictions.csv' if resuming else ROOT/'baseline_10epoch/valid_predictions.csv')])
            pred, metric = evaluate('valid')
            error = float(np.max(np.abs(pred-old_pred)))
            assert np.allclose(pred,old_pred,rtol=1e-5,atol=1e-6), error
            # Deferred restoration retains saved Adam slots for the first new fit.
            dump(run/('resume_restore_check.json' if resuming else 'baseline_restore_check.json'),{'valid_max_abs_difference_eV':error,
                 'global_step':model.get_global_step(),'baseline_epoch':first})
            _, train_metric = evaluate('train')
            if not resuming:
                curve.append({'epoch':first,'fit_loss':None,'elapsed_s':0.0,
                     **{'train_'+k:v for k,v in train_metric.items() if k!='n_rows'},
                     **{'valid_'+k:v for k,v in metric.items() if k!='n_rows'}})
            best_score=patience_score=metric['RMSE_eV'];best_epoch=first
            model.save_checkpoint(max_checkpoints_to_keep=1, model_dir=str(run/'best_checkpoint'))
        run_start = time.perf_counter()
        for epoch in range(first+1, protocol['max_epoch']+1):
            loss = model.fit(fit_data, nb_epoch=1, checkpoint_interval=0, deterministic=False)
            _, train_metric = evaluate('train')
            valid_pred, valid_metric = evaluate('valid')
            score = valid_metric['RMSE_eV']
            if score < best_score:
                best_score=score; best_epoch=epoch
                model.save_checkpoint(max_checkpoints_to_keep=1,model_dir=str(run/'best_checkpoint'))
                helper.write_csv(run/'best_valid_predictions.csv',
                    [dict(r,predicted_band_gap_eV=float(v)) for r,v in zip(subsets['valid'],valid_pred)])
            if score < patience_score-protocol['early_stopping_min_delta_eV']:
                patience_score=score;wait=0
            else:
                wait+=1
            curve.append({'epoch':epoch,'fit_loss':float(loss),'elapsed_s':elapsed_offset+time.perf_counter()-run_start,
                 **{'train_'+k:v for k,v in train_metric.items() if k!='n_rows'},
                 **{'valid_'+k:v for k,v in valid_metric.items() if k!='n_rows'}})
            helper.write_csv(run/'learning_curve.csv',curve)
            dump(ROOT/'progress.json',{'status':'training','variant':name,'epoch':epoch,
                  'best_epoch':best_epoch,'best_validation_RMSE_eV':best_score,
                  'current_validation_RMSE_eV':score,'current_validation_R2':valid_metric['R2'],
                  'current_train_R2':train_metric['R2'],'patience_used':wait,'updated_utc':now()})
            if epoch%10==0 or epoch==first+1:
                print(json.dumps({'variant':name,'epoch':epoch,'train_R2':train_metric['R2'],
                   'valid_R2':valid_metric['R2'],'valid_RMSE_eV':score,'best_epoch':best_epoch,
                   'elapsed_s':time.perf_counter()-run_start}),flush=True)
            if epoch>=protocol['min_stop_epoch'] and wait>=protocol['patience_epochs']:
                break
        model.save_checkpoint(max_checkpoints_to_keep=1,model_dir=str(run/'last_checkpoint'))
        result={'variant':name,'best_epoch':best_epoch,'last_epoch':epoch,
                'best_validation_RMSE_eV':best_score,'target_scaling':scale,
                'stop_reason':'validation_patience' if epoch<protocol['max_epoch'] else 'maximum_epoch',
                'elapsed_s':elapsed_offset+time.perf_counter()-run_start,'model_config':config,
                'checkpoint':str(run/'best_checkpoint')}
        dump(run/'result.json',result);results.append(result)
        del model
    # Predeclared validation-only winner, locked before the first new test prediction.
    winner = min(results,key=lambda r:r['best_validation_RMSE_eV'])
    selection={'locked_utc':now(),'criterion':'minimum validation RMSE_eV',
               'test_used_for_selection':False,'variants':results,'selected':winner}
    dump(ROOT/'selection.json',selection)
    target=ROOT/'selected_model';target.mkdir()
    shutil.copytree(Path(winner['checkpoint']),target/'checkpoint')
    scale=winner['target_scaling']
    tf.keras.backend.clear_session()
    selected=dc.models.GraphConvModel(model_dir=str(target/'checkpoint'),**config)
    selected.restore()
    features['test']=helper.dataset(subsets['test'],np,dc)
    final={};replay={};saved={}
    for split in ['train','valid','test']:
        pred=np.asarray(selected.predict(features[split])).reshape(-1)*scale['std_eV']+scale['mean_eV']
        saved[split]=pred
        final[split]=helper.metrics([r['target_eV'] for r in subsets[split]],pred.tolist())
        helper.write_csv(target/f'{split}_predictions.csv',[dict(r,predicted_band_gap_eV=float(v)) for r,v in zip(subsets[split],pred)])
    candidates=json.loads((ROOT/'baseline_10epoch/candidate_results.json').read_text(encoding='utf-8'))
    cand_rows=[{'csv_line':i+2,'model_smiles':helper.graph_smiles(r['smiles'],'preserve')[1]} for i,r in enumerate(candidates)]
    candidate_data=helper.dataset(cand_rows,np,dc)
    cpred=np.asarray(selected.predict(candidate_data)).reshape(-1)*scale['std_eV']+scale['mean_eV']
    for row,value in zip(candidates,cpred):
        row['baseline_10epoch_prediction_eV']=row['predicted_band_gap_eV']
        row['predicted_band_gap_eV']=float(value)
        row['absolute_target_deviation_eV']=abs(float(value)-6.5)
        row['predicted_at_least_target']=bool(value>=6.5)
    reloaded=dc.models.GraphConvModel(model_dir=str(target/'checkpoint'),**config);reloaded.restore()
    for split in ['train','valid','test']:
        repeated=np.asarray(reloaded.predict(features[split])).reshape(-1)*scale['std_eV']+scale['mean_eV']
        replay[split+'_max_abs_difference_eV']=float(np.max(np.abs(repeated-saved[split])))
        assert np.allclose(repeated,saved[split],rtol=1e-6,atol=1e-6)
    repeated=np.asarray(reloaded.predict(candidate_data)).reshape(-1)*scale['std_eV']+scale['mean_eV']
    replay['candidates_max_abs_difference_eV']=float(np.max(np.abs(repeated-cpred)))
    assert np.allclose(repeated,cpred,rtol=1e-6,atol=1e-6)
    replay['verified']=True
    dump(target/'metrics.json',final);dump(target/'candidate_results.json',candidates);dump(target/'reload_verification.json',replay)
    artifact={'status':'trained','target':'single_chain_band_gap','target_unit':'eV',
              'source_doi':'10.1016/j.patter.2021.100238','seed':42,'wildcard_policy':'preserve',
              'model_config':config,'target_scaling':scale,'selected_variant':winner['variant'],
              'selected_epoch':winner['best_epoch'],'split_sha256':protocol['split_sha256'],
              'reload_verified':True,'versions':{k:importlib.metadata.version(k) for k in ['deepchem','tensorflow','numpy','rdkit']}}
    dump(target/'artifact.json',artifact)
    summary={'completed_utc':now(),'elapsed_s':time.perf_counter()-started,
             'baseline_10epoch_metrics':json.loads((ROOT/'baseline_10epoch/metrics.json').read_text(encoding='utf-8')),
             'selection':selection,'final_metrics':final,'candidates':candidates,'reload_verification':replay}
    dump(ROOT/'summary.json',summary)
    dump(ROOT/'progress.json',{'status':'completed','selected_variant':winner['variant'],
           'selected_epoch':winner['best_epoch'],'test_metrics':final['test'],'updated_utc':now()})
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
