"""One frozen clean holdout evaluation with prediction-before-label ordering.

No model selection, downloading, preparation, target assets or alternate
configurations. A persistent candidate-side claim permits only the same output
directory on resume. Geometry-only predictions for every period commit before
the first heldout supervision array is accessed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from .common import decode, write_json
from .download import sha256
from .ensemble import validate_members, load_member, predict_member, weighted_average
from .run_lock import run_lock, AlreadyRunningError


HOLDOUT=('J03WMX','J03WR9')
REGISTRY_DIR=Path('model-runs/frozen-holdout-evaluation')


def read(path):
    return json.loads(Path(path).read_text())


def payload_hash(payload):
    normalized=json.loads(json.dumps(payload,allow_nan=False))
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_sealed(path,payload):
    write_json(path,dict(**payload,payload_sha256=payload_hash(payload)))


def read_sealed(path):
    record=read(path)
    if not isinstance(record,dict):
        raise ValueError('Frozen score payload must be an object')
    checksum=record.pop('payload_sha256',None)
    if checksum!=payload_hash(record):
        raise ValueError(f'Frozen score payload checksum changed: {path}')
    return record


def validate_candidate(path):
    candidate=read(path); freeze=read(path.parent/'freeze.json')
    digest=sha256(path); split=read('modeling/splits.json')
    if freeze.get('status')!='frozen' or freeze.get('candidate_sha256')!=digest:
        raise ValueError('A matching frozen candidate hash is required before evaluation')
    if (isinstance(candidate.get('schema_version'),bool) or candidate.get('schema_version')!=1 or
            candidate.get('task')!='possession_HAD_25fps' or candidate.get('switch_cost')!=3):
        raise ValueError('Unsupported frozen candidate schema/task/decoder')
    if split['holdout']!=list(HOLDOUT):
        raise ValueError('Only the pre-existing J03WMX/J03WR9 holdouts are supported')
    if (set(split['train'])&set(split['validation']) or
            set(split['holdout'])&(set(split['train'])|set(split['validation']))):
        raise ValueError('Frozen train/validation/holdout roles must be disjoint')
    for role in ('train','validation','holdout'):
        if candidate.get(role)!=split[role]:
            raise ValueError(f'Frozen {role} roles must exactly match the existing split')
    try:
        frozen=datetime.fromisoformat(candidate['frozen_at_utc'].replace('Z','+00:00'))
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError('A valid frozen_at_utc timestamp is required') from error
    if frozen.tzinfo is None or frozen.utcoffset().total_seconds()!=0:
        raise ValueError('frozen_at_utc must include UTC timezone')
    if not isinstance(candidate.get('members'),list):
        raise ValueError('Frozen members must be a list of artifact specifications')
    members=validate_members(candidate['members'])
    for member in members:
        if member.get('train')!=split['train'] or member.get('validation')!=split['validation']:
            raise ValueError('Member provenance differs from the frozen training/validation roles')
    weights=candidate.get('weights')
    if not isinstance(weights,dict) or set(weights)!={m['id'] for m in members}:
        raise ValueError('One frozen weight is required for every member id')
    if any(isinstance(w,bool) or not isinstance(w,(int,float)) for w in weights.values()):
        raise ValueError('Frozen weights must be numeric')
    values=np.asarray(list(weights.values()),np.float64)
    if not np.isfinite(values).all() or (values<0).any() or not np.isfinite(values.sum()) or values.sum()<=0:
        raise ValueError('Frozen weights must be finite nonnegative with positive total')
    return candidate,digest


def claim_evaluation(candidate_path,candidate_hash,output):
    path=candidate_path.parent/'evaluation-claim.json'
    if path.exists():
        claim=read(path)
        if claim.get('candidate_sha256')!=candidate_hash or Path(claim['output']).resolve()!=output.resolve():
            raise ValueError('Candidate evaluation is already claimed; only its original output may resume')
    else:
        write_json(path,dict(schema_version=1,candidate_sha256=candidate_hash,output=str(output.resolve())))


def claim_workspace(candidate,candidate_hash,output):
    path=REGISTRY_DIR/'claim.json'
    if path.exists():
        claim=read(path)
        if (claim.get('candidate_sha256')!=candidate_hash or claim.get('holdout')!=candidate['holdout'] or
                Path(claim.get('output','')).resolve()!=output.resolve()):
            raise ValueError('Workspace holdouts already belong to one frozen candidate/output')
    else:
        write_json(path,dict(schema_version=1,candidate_sha256=candidate_hash,
                            holdout=candidate['holdout'],output=str(output.resolve())))


def period_paths(data,matches):
    files=[data/f'{match}-{half}.npz' for match in matches for half in ('firstHalf','secondHalf')]
    if len(files)!=4 or any(not path.is_file() for path in files):
        raise ValueError('All four already-prepared frozen holdout periods are required; this evaluator never downloads/prepares data')
    return files


def load_geometry(path):
    """Never request the labels key, including during resumed prediction."""
    with np.load(path,allow_pickle=False) as archive:
        data={name:archive[name] for name in ('players','ball','teams','pitch')}
        if 'fps' in archive.files:
            data['fps']=archive['fps']
    if not len(data['ball']):
        raise ValueError('Empty holdout period')
    return data


def load_labels(path):
    """Called only after the durable predictions-complete barrier."""
    with np.load(path,allow_pickle=False) as archive:
        labels=archive['labels']
    if labels.ndim!=1 or labels.dtype.kind not in 'iu' or not np.isin(labels,[0,1,2,3]).all():
        raise ValueError('Invalid heldout H/A/D/X supervision')
    return labels


def verify_hashes(hashes):
    for filename,digest in hashes.items():
        if sha256(Path(filename))!=digest:
            raise ValueError(f'Frozen evaluation artifact changed: {filename}')


def save_prediction(path,probabilities,prediction,metadata):
    path.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=path.stem+'-',suffix='.npz',dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary,probabilities=probabilities,prediction=prediction)
        os.replace(temporary,path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    write_json(path.with_suffix('.json'),dict(**metadata,rows=len(prediction),sha256=sha256(path)))


def cached_prediction(path,expected):
    metadata=path.with_suffix('.json')
    if not path.exists() or not metadata.exists():
        return None
    record=read(metadata)
    if (not isinstance(record,dict) or set(record)!=set(expected)|{'rows','sha256'} or
            any(record.get(key)!=value for key,value in expected.items()) or
            record['sha256']!=sha256(path)):
        raise ValueError('Frozen prediction cache bytes/provenance changed')
    with np.load(path,allow_pickle=False) as archive:
        p,pred=archive['probabilities'],archive['prediction']
    weighted_average([p],[1.])  # finite, normalized T x 3 contract
    if (type(record['rows']) is not int or record['rows']<1 or len(pred)!=record['rows'] or
            pred.shape!=(len(p),) or pred.dtype.kind not in 'iu' or not np.isin(pred,[0,1,2]).all()):
        raise ValueError('Invalid frozen prediction cache contract')
    return pred


def make_protocol(path,candidate,digest,data,output,files):
    import torch
    import sklearn
    names=('frozen_evaluate.py','ensemble.py','neural.py','motion.py','robust_features.py',
           'common.py','official_metrics.py','download.py','run_lock.py','splits.json')
    member_hashes={m['path']:m['sha256'] for m in candidate['members']}
    member_hashes.update({str(Path(m['path']).parent/'report.json'):m['report_sha256']
                         for m in candidate['members'] if 'report_sha256' in m})
    return dict(schema_version=1,candidate_path=str(path.resolve()),candidate_sha256=digest,
        freeze_sha256=sha256(path.parent/'freeze.json'),data=str(data.resolve()),output=str(output.resolve()),
        task='one_frozen_clean_holdout_evaluation',holdout=candidate['holdout'],switch_cost=3,
        weights=candidate['weights'],input_hashes={str(p):sha256(p) for p in files},member_hashes=member_hashes,
        source_hashes={str(Path('modeling')/n):sha256(Path('modeling')/n) for n in names},
        runtime=dict(numpy=np.__version__,torch=torch.__version__,sklearn=sklearn.__version__),threads=1,
        transition_matching='Official typed-to matches isolated within each input period and graded scoring run',
        scope='One pre-frozen configuration on two untouched external clean matches; no alternatives, target assets or target export')


def prediction_provenance(path,protocol,protocol_hash):
    return dict(candidate_sha256=protocol['candidate_sha256'],protocol_sha256=protocol_hash,
        input_sha256=protocol['input_hashes'][str(path)],period=path.stem,
        member_hashes_sha256=payload_hash(protocol['member_hashes']))


def validate_predictions_marker(marker,protocol,protocol_hash,files,output):
    expected={path.stem:output/'predictions'/f'{path.stem}.npz' for path in files}
    artifacts={str(p) for dest in expected.values() for p in (dest,dest.with_suffix('.json'))}
    if (not isinstance(marker,dict) or marker.get('status')!='completed' or
            marker.get('candidate_sha256')!=protocol['candidate_sha256'] or
            marker.get('protocol_sha256')!=protocol_hash or
            not isinstance(marker.get('periods'),dict) or set(marker['periods'])!=set(expected) or
            not isinstance(marker.get('artifact_hashes'),dict) or set(marker['artifact_hashes'])!=artifacts):
        raise ValueError('Frozen predictions-complete barrier contract differs')
    for stem,dest in expected.items():
        record=marker['periods'][stem]
        if (not isinstance(record,dict) or set(record)!={'path','rows'} or record.get('path')!=str(dest) or
                type(record.get('rows')) is not int or record['rows']<1):
            raise ValueError('Frozen prediction period path/rows contract differs')
    verify_hashes(marker['artifact_hashes'])
    for path in files:
        dest=expected[path.stem]
        prediction=cached_prediction(dest,prediction_provenance(path,protocol,protocol_hash))
        if prediction is None or marker['periods'][path.stem]['rows']!=len(prediction):
            raise ValueError('Frozen prediction barrier/cache row contract differs')
    return marker


def predictions_phase(candidate,models,protocol,protocol_hash,files,output):
    barrier=output/'predictions-complete.json'
    if barrier.exists():
        return validate_predictions_marker(read(barrier),protocol,protocol_hash,files,output)
    records={}; artifacts={}
    for path in files:
        dest=output/'predictions'/f'{path.stem}.npz'
        provenance=prediction_provenance(path,protocol,protocol_hash)
        prediction=cached_prediction(dest,provenance)
        if prediction is None:
            geometry=load_geometry(path)
            probabilities=[predict_member(m,models[m['id']],geometry) for m in candidate['members']]
            averaged=weighted_average(probabilities,[candidate['weights'][m['id']] for m in candidate['members']])
            prediction=decode(averaged,switch_cost=3)
            save_prediction(dest,averaged,prediction,provenance)
        records[path.stem]=dict(path=str(dest),rows=len(prediction))
        for artifact in (dest,dest.with_suffix('.json')):
            artifacts[str(artifact)]=sha256(artifact)
        write_json(output/'status.json',dict(state='running',stage='geometry_only_predictions',period=path.stem))
    # This atomic marker is the first point where any supervision may be read.
    marker=dict(status='completed',candidate_sha256=protocol['candidate_sha256'],protocol_sha256=protocol_hash,
                periods=records,artifact_hashes=artifacts)
    write_json(barrier,marker)
    return validate_predictions_marker(marker,protocol,protocol_hash,files,output)


def score_metadata(candidate,protocol,protocol_hash,output):
    return dict(candidate_sha256=protocol['candidate_sha256'],protocol_sha256=protocol_hash,
                predictions_complete_sha256=sha256(output/'predictions-complete.json'))


def score_progress(candidate,protocol,protocol_hash,output):
    metadata=score_metadata(candidate,protocol,protocol_hash,output)
    report_path=output/'full-reports.json'
    report=read_sealed(report_path) if report_path.exists() else dict(status='running',**metadata,matches={})
    if (any(report.get(k)!=v for k,v in metadata.items()) or report.get('status') not in ('running','completed') or
            not isinstance(report.get('matches'),dict) or not set(report['matches'])<=set(candidate['holdout'])):
        raise ValueError('Frozen score report protocol/match contract differs')
    commits={}
    for match in candidate['holdout']:
        path=output/'score-commits'/f'{match}.json'
        if not path.exists():
            continue
        commit=read_sealed(path)
        if (set(commit)!=set(metadata)|{'status','match','metrics'} or commit.get('status')!='committed' or
                commit.get('match')!=match or any(commit.get(k)!=v for k,v in metadata.items()) or
                not isinstance(commit.get('metrics'),dict)):
            raise ValueError('Frozen per-match score commit contract differs')
        commits[match]=commit['metrics']
    for match,metrics in report['matches'].items():
        if match not in commits or metrics!=commits[match]:
            raise ValueError('Frozen score progress differs from its per-match commit')
    if report['status']=='completed' and set(report['matches'])!=set(candidate['holdout']):
        raise ValueError('Completed frozen score report is missing a match')
    # A crash after committing a match but before updating the aggregate can
    # safely recover its score without rereading labels.
    report['matches'].update(commits)
    return report


def compact_summary(candidate,protocol,protocol_hash,report):
    keys=('spell_f1_mean','macro_f1','alive_dead_f1','n_gt_spells','n_pred_spells','n_graded')
    compact={m:{key:report['matches'][m][key] for key in keys} for m in candidate['holdout']}
    return dict(status='completed',candidate_sha256=protocol['candidate_sha256'],protocol_sha256=protocol_hash,
        matches=compact,mean={key:float(np.mean([compact[m][key] for m in candidate['holdout']]))
        for key in ('spell_f1_mean','macro_f1','alive_dead_f1')},limitations=protocol['scope'])


def period_metrics(periods):
    """Official primary metrics; transition matches cannot cross scoring runs."""
    from . import official_metrics as metric
    truths,predictions=[],[]
    for truth,prediction in periods:
        truths.extend([truth,np.array([3],np.int8)]); predictions.extend([prediction,np.array([3],np.int8)])
    result=metric.evaluate(np.concatenate(truths),np.concatenate(predictions))
    for tolerance in metric.TAUS:
        rows=[]; offset=0
        for truth,prediction in periods:
            runs=metric.valid_runs(truth); filled=metric.fill_abstentions(prediction,runs)
            for run in runs:
                gt=[(t+offset,a,b) for t,a,b in metric.transitions(truth,[run])]
                pred=[(t+offset,a,b) for t,a,b in metric.transitions(filled,[run])]
                rows.append(metric.transition_scores(gt,pred,tolerance))
            offset+=len(truth)+1
        keys=('n_gt','n_pred','n_matched','n_missed','n_spurious','n_ambiguous')
        totals={key:sum(row[key] for row in rows) for key in keys}
        events={key:[event for row in rows for event in row[key]] for key in ('pairs','missed','spurious')}
        tp=totals['n_matched']; precision=tp/totals['n_pred'] if totals['n_pred'] else 0.
        recall=tp/totals['n_gt'] if totals['n_gt'] else 0.
        latency=np.array([pred[0]-gt[0] for gt,pred in events['pairs']],np.float64)
        result['transitions'][tolerance]=dict(**totals,**events,precision=precision,recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
            mean_latency=float(latency.mean()) if len(latency) else None,
            median_latency=float(np.median(latency)) if len(latency) else None,
            mean_abs_latency=float(np.abs(latency).mean()) if len(latency) else None,
            p90_abs_latency=float(np.percentile(np.abs(latency),90)) if len(latency) else None)
    result['transition_matching']='Official typed-to event matching isolated per period/graded scoring run; decoder predictions unchanged'
    return result


def score_phase(candidate,protocol,protocol_hash,marker,files,output):
    report_path=output/'full-reports.json'
    report=score_progress(candidate,protocol,protocol_hash,output)
    for match in candidate['holdout']:
        if match in report['matches']:
            continue
        periods=[]
        for path in [p for p in files if p.stem.startswith(match+'-')]:
            verify_hashes({str(path):protocol['input_hashes'][str(path)]})
            labels=load_labels(path)
            record=marker['periods'][path.stem]
            if len(labels)!=record['rows']:
                raise ValueError('Committed prediction/heldout label length mismatch')
            committed=output/'predictions'/f'{path.stem}.npz'
            verify_hashes({str(committed):marker['artifact_hashes'][str(committed)]})
            with np.load(committed,allow_pickle=False) as archive:
                predicted=archive['prediction']
            periods.append((labels,predicted))
        metrics=period_metrics(periods)
        write_sealed(output/'score-commits'/f'{match}.json',dict(status='committed',match=match,
            metrics=metrics,**score_metadata(candidate,protocol,protocol_hash,output)))
        report['matches'][match]=metrics
        write_sealed(report_path,report)
    report['status']='completed'; write_sealed(report_path,report)
    write_json(output/'summary.json',compact_summary(candidate,protocol,protocol_hash,report))


def completion_artifacts(candidate,output):
    return [output/n for n in ('protocol.json','predictions-complete.json','full-reports.json','summary.json')]+[
        output/'score-commits'/f'{match}.json' for match in candidate['holdout']]


def validate_completion(candidate,protocol,protocol_hash,output):
    marker=read(output/'complete.json')
    expected={str(p) for p in completion_artifacts(candidate,output)}
    if (marker.get('status')!='completed' or marker.get('candidate_sha256')!=protocol['candidate_sha256'] or
            marker.get('protocol_sha256')!=protocol_hash or not isinstance(marker.get('artifact_hashes'),dict) or
            set(marker['artifact_hashes'])!=expected):
        raise ValueError('Frozen completion marker status/candidate/artifact contract differs')
    verify_hashes(marker['artifact_hashes'])
    report=score_progress(candidate,protocol,protocol_hash,output)
    if report['status']!='completed' or set(report['matches'])!=set(candidate['holdout']):
        raise ValueError('Frozen completion marker has incomplete committed scores')
    if read(output/'summary.json')!=compact_summary(candidate,protocol,protocol_hash,report):
        raise ValueError('Frozen completed summary differs from its committed scores')


def run_evaluation(candidate_path,output,data):
    candidate_path,output,data=Path(candidate_path),Path(output),Path(data)
    with run_lock(REGISTRY_DIR),run_lock(candidate_path.parent/'evaluation-lock'),run_lock(output):
        candidate,digest=validate_candidate(candidate_path)
        claim_workspace(candidate,digest,output)
        claim_evaluation(candidate_path,digest,output)
        # Artifact class/provenance checks precede even holdout geometry loading.
        models={m['id']:load_member(m) for m in candidate['members']}
        files=period_paths(data,candidate['holdout'])
        protocol=make_protocol(candidate_path,candidate,digest,data,output,files)
        protocol_path=output/'protocol.json'
        if protocol_path.exists() and read(protocol_path)!=protocol:
            raise ValueError('Frozen protocol/input/model/source changed; alternate output is prohibited by the evaluation claim')
        write_json(protocol_path,protocol); protocol_hash=sha256(protocol_path)
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(1)
        with threadpool_limits(limits=1):
            marker=predictions_phase(candidate,models,protocol,protocol_hash,files,output)
            if (output/'complete.json').exists():
                validate_completion(candidate,protocol,protocol_hash,output); return
            if sha256(candidate_path)!=digest or sha256(candidate_path.parent/'freeze.json')!=protocol['freeze_sha256']:
                raise ValueError('Candidate/freeze changed before the scoring barrier')
            for group in ('input_hashes','member_hashes','source_hashes'):
                verify_hashes(protocol[group])
            score_phase(candidate,protocol,protocol_hash,marker,files,output)
        artifacts=completion_artifacts(candidate,output)
        write_json(output/'complete.json',dict(status='completed',candidate_sha256=digest,protocol_sha256=protocol_hash,
            artifact_hashes={str(p):sha256(p) for p in artifacts}))
        write_json(output/'status.json',dict(state='completed',stage='one_frozen_evaluation'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('model-runs/final-evaluation-v1'))
    parser.add_argument('--data',type=Path,default=Path('data-external/prepared'))
    args=parser.parse_args()
    try:
        run_evaluation(args.candidate,args.out,args.data)
    except Exception as error:
        if not isinstance(error,AlreadyRunningError):
            write_json(args.out/'status.json',dict(state='failed',error=str(error)))
        raise
