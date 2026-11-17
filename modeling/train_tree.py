"""CPU training; train and validation matches must be disjoint."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .common import load, corrupt, decode, evaluate, paths, write_json, nearest_baseline
from .robust_features import robust_features


def main(a):
    velocity_lag = getattr(a, 'velocity_lag', 1)
    if isinstance(velocity_lag, bool) or not isinstance(velocity_lag, int) or velocity_lag < 1:
        raise ValueError('velocity_lag must be a positive integer')
    if set(a.train) & set(a.validation):
        raise ValueError('Training/validation match overlap')
    split = json.loads(Path('modeling/splits.json').read_text())
    if (set(a.train) | set(a.validation)) & set(split['holdout']):
        raise ValueError('Final holdouts cannot be used by the development trainer')
    if not set(a.train) <= set(split['train']) or not set(a.validation) <= set(split['validation']):
        raise ValueError('Requested matches do not respect the frozen split roles')
    xs, ys = [], []
    for path in paths(a.data, a.train):
        d = load(path)
        for variant in (d, corrupt(d)):
            x, names = robust_features(variant, lag=velocity_lag)
            ix = np.arange(0, len(x), a.stride)
            ix = ix[d['labels'][ix] < 3]
            xs.append(x[ix]); ys.append(d['labels'][ix])
    x, y = np.concatenate(xs), np.concatenate(ys)
    print(f'Training {len(y)} examples, {x.shape[1]} features', flush=True)
    model = HistGradientBoostingClassifier(max_iter=a.iterations, max_leaf_nodes=15,
             l2_regularization=1.0, early_stopping=False, random_state=42)
    with threadpool_limits(limits=4):
        model.fit(x, y)
    if not np.array_equal(model.classes_, [0, 1, 2]):
        raise ValueError('All H/A/D classes required')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / 'tree.pkl', 'wb') as f:
        pickle.dump({'model': model, 'features': names, 'switch_cost': a.switch_cost,
                     'velocity_lag': velocity_lag}, f)
    report = {'train': a.train, 'validation': a.validation, 'seed': 42,
              'training_rows': len(y), 'iterations': a.iterations, 'stride': a.stride,
              'velocity_lag': velocity_lag,
              'switch_cost': a.switch_cost, 'periods': {},
              'limitations': 'Provider-label proxy; no target score or semantic equivalence claim.'}
    validation = paths(a.data, a.validation) if a.validation else []
    for path in validation:
        d = load(path)
        for name, variant in [('clean', d), ('ball_gaps_dropout', corrupt(d, 123))]:
            x, _ = robust_features(variant, lag=velocity_lag)
            with threadpool_limits(limits=4):
                prob = model.predict_proba(x)
            pred = decode(prob, a.switch_cost)
            baseline = nearest_baseline(variant)
            report['periods'][f'{path.stem}/{name}'] = {
                'nearest_smooth50': evaluate(d['labels'], baseline),
                'raw': evaluate(d['labels'], prob.argmax(1)),
                'decoded': evaluate(d['labels'], pred)}
            np.savez_compressed(out / f'{path.stem}-{name}-pred.npz', probabilities=prob, prediction=pred, baseline=baseline)
    if not validation:
        report['status'] = 'Training smoke run only; no held-out evaluation performed'
    else:
        # Aggregate at match level, separating halves with ungraded X to prevent merging.
        report['matches'] = {}
        for match in a.validation:
            for scenario in ('clean', 'ball_gaps_dropout'):
                truth, prediction, baseline = [], [], []
                for path in paths(a.data, [match]):
                    truth.extend([load(path)['labels'], np.array([3], np.int8)])
                    result = load(out / f'{path.stem}-{scenario}-pred.npz')
                    prediction.extend([result['prediction'], np.array([3], np.int8)])
                    baseline.extend([result['baseline'], np.array([3], np.int8)])
                report['matches'][f'{match}/{scenario}'] = {
                    'decoded': evaluate(np.concatenate(truth), np.concatenate(prediction)),
                    'nearest_smooth50': evaluate(np.concatenate(truth), np.concatenate(baseline))}
    write_json(out / 'report.json', report)
    print(f'Saved model and report to {out}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data-external/prepared')
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', nargs='*', default=[])
    p.add_argument('--out', default='model-runs/tree')
    p.add_argument('--iterations', type=int, default=100)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--switch-cost', type=float, default=3.0)
    p.add_argument('--velocity-lag', type=int, default=1)
    main(p.parse_args())
