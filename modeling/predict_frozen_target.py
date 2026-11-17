"""Local target export from one frozen ensemble; never reads target supervision."""
import argparse
import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import numpy as np
from .common import decode, write_json
from .download import sha256
from .ensemble import load_member, predict_member, weighted_average
from .ensemble_run import save_npz
from .frozen_evaluate import validate_candidate
from .predict_target import read_target
from .run_lock import run_lock, AlreadyRunningError


def source_paths(repo):
    spec = importlib.util.spec_from_file_location('frozen_target_games', repo/'games.py')
    games = importlib.util.module_from_spec(spec); spec.loader.exec_module(games)
    if games.GAMES['BAR-ATH']['fps'] != 25:
        raise ValueError('Frozen target adapter requires the supplied 25-Hz recording')
    assets = games.asset_paths(data_dir=str(repo/'data'))
    # The allowlist excludes all target labels/events/video and unrelated assets.
    return [repo/'games.py', Path(assets['atd_raw']), Path(assets['atd_meta'])]


def predict_period(candidate, loaded, data):
    probabilities = [predict_member(member, loaded[member['id']], data) for member in candidate['members']]
    combined = weighted_average(probabilities, [candidate['weights'][member['id']] for member in candidate['members']])
    return combined, decode(combined, switch_cost=candidate['switch_cost'])


def assemble(n, periods):
    labels = np.full(n, 'X', dtype='<U1')
    covered = np.zeros(n, bool)
    for lo, hi, pred in periods:
        if not 0 <= lo < hi <= n or len(pred) != hi-lo or covered[lo:hi].any():
            raise ValueError('Invalid or overlapping target period bounds')
        if np.asarray(pred).ndim != 1 or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError('Period predictions must use H/A/D codes')
        labels[lo:hi] = np.array(list('HAD'))[pred]
        covered[lo:hi] = True
    return labels


def main(a):
    import json
    import pandas as pd
    import torch
    from threadpoolctl import threadpool_limits
    from .official_metrics import validate_submission
    with run_lock(a.out), threadpool_limits(limits=1):
        candidate, digest = validate_candidate(a.candidate)
        loaded = {m['id']: load_member(m) for m in candidate['members']}
        torch.set_num_threads(1)
        inputs = source_paths(a.repo)
        stats = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
        code = ['predict_frozen_target.py', 'predict_target.py', 'ensemble.py', 'neural.py', 'motion.py',
                'robust_features.py', 'common.py', 'official_metrics.py']
        manifest = dict(candidate_sha256=digest, candidate=str(a.candidate),
            source_hashes={str(p): sha256(p) for p in inputs},
            code_hashes={str(Path('modeling')/name): sha256(Path('modeling')/name) for name in code},
            mode='Frozen tracking-only ensemble; candidate weights; cost3; original ATD observations',
            scope='Local prediction export only; target possession ground truth was not read; no video recovery, submission or publishing')
        manifest_path = a.out/'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Frozen target inputs/configuration changed; use a new output directory')
        write_json(manifest_path, manifest)
        if (a.out/'complete.json').exists():
            record = json.loads((a.out/'complete.json').read_text())
            if record.get('status') != 'completed' or record.get('manifest_sha256') != sha256(manifest_path):
                raise ValueError('Target completion contract changed')
            for path, expected in record['artifact_hashes'].items():
                if sha256(path) != expected:
                    raise ValueError('Completed target artifact changed')
            return
        write_json(a.out/'status.json', dict(state='running', stage='target_geometry_only'))
        n, blocks = read_target(a.repo)
        predictions, descriptions, artifacts = [], [], []
        for i, (lo, hi, data) in enumerate(blocks):
            probability, predicted = predict_period(candidate, loaded, data)
            path = a.out/f'period-{i+1}-predictions.npz'
            save_npz(path, probabilities=probability, prediction=predicted)
            artifacts.append(path)
            predictions.append((lo, hi, predicted))
            descriptions.append(dict(start=lo, stop_exclusive=hi, frames=hi-lo,
                ball_missing_fraction=float((~np.isfinite(data['ball']).all(-1)).mean())))
            print('FROZEN TARGET PERIOD', i+1, lo, hi, flush=True)
        labels = assemble(n, predictions)
        output = a.out/'possession.csv'
        descriptor, temporary = tempfile.mkstemp(prefix='possession-', suffix='.csv', dir=a.out)
        os.close(descriptor)
        try:
            with open(temporary, 'w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream, lineterminator='\n'); writer.writerow(['frame', 'label'])
                writer.writerows(enumerate(labels))
            error = validate_submission(pd.read_csv(temporary), n)
            if error is not None:
                raise ValueError('Mechanical prediction validation failed: '+error)
            os.replace(temporary, output)
        finally:
            Path(temporary).unlink(missing_ok=True)
        for path in inputs:
            if (path.stat().st_size, path.stat().st_mtime_ns) != stats[str(path)]:
                raise ValueError('Target input changed during inference')
        if sha256(a.candidate) != digest:
            raise ValueError('Frozen candidate changed during inference')
        write_json(a.out/'summary.json', dict(status='completed', frames=n, periods=descriptions,
            predicted_label_counts={label: int((labels == label).sum()) for label in 'HADX'},
            csv_sha256=sha256(output), candidate_sha256=digest, mechanical_validation='passed',
            limitations='No target accuracy is measured; X is retained outside provided playing periods. Original/recovered video proposals are not fused.'))
        artifacts += [output, a.out/'summary.json', manifest_path]
        write_json(a.out/'complete.json', dict(status='completed', manifest_sha256=sha256(manifest_path),
            artifact_hashes={str(path): sha256(path) for path in artifacts}))
        write_json(a.out/'status.json', dict(state='completed', stage='validated_local_export'))
        print('Frozen target export complete:', output, n, 'frames; no target labels read.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', type=Path, default=Path('model-runs/frozen-candidate-v1/manifest.json'))
    p.add_argument('--repo', type=Path, default=Path('hackathon-windows-transfer/more-than-a-hack-2027'))
    p.add_argument('--out', type=Path, default=Path('model-runs/frozen-target-v1'))
    a = p.parse_args()
    try:
        main(a)
    except Exception as error:
        if not isinstance(error, AlreadyRunningError):
            write_json(a.out/'status.json', dict(state='failed', error=str(error)))
        raise
