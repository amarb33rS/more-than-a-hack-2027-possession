"""Hash-pinned, development-only comparison of the three completed motion seeds.

Cached standard comparisons run with one CPU thread. Fresh inference waits for
the current robust-motion driver, so it does not compete with active training.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from .common import load, paths, write_json
from .download import sha256
from .improve_run import completed, now
from .run_lock import run_lock, AlreadyRunningError


def read(path):
    return json.loads(Path(path).read_text())


def configurations():
    result = []
    motion = {f'motion_{s}': 1/3 for s in (42, 43, 44)}
    for weight in (0, .25, .5, .6):
        weights = {k: (1-weight)*v for k, v in motion.items()}
        if weight:
            weights['tree'] = weight
        for cost in (3, 10):
            result.append(dict(id=f'motion3_tree_{weight:g}_cost_{cost}',
                               weights=weights, switch_cost=cost))
    result += [dict(id='reference_motion43_44_tree_025_cost_3',
                    weights={'motion_43': .375, 'motion_44': .375, 'tree': .25}, switch_cost=3),
               dict(id='geometry_fallback_cost_3',
                    weights={**{f'motion_{s}': .5/3 for s in (42, 43, 44)},
                             **{f'geometry_{s}': .25/3 for s in (42, 43, 44)}, 'tree': .25},
                    switch_cost=3),
               dict(id='geometry3_cost_3',
                    weights={f'geometry_{s}': 1/3 for s in (42, 43, 44)}, switch_cost=3)]
    result.extend(dict(id=f'tree_cost_{cost}', weights={'tree': 1.}, switch_cost=cost)
                  for cost in (3, 10))
    return result


def build_plan():
    from .ensemble import describe_member, validate_members
    split = read('modeling/splits.json')
    roots = {'motion_42': Path('model-runs/improvements-v1/motion-pilot'),
             'motion_43': Path('model-runs/improvements-v1/motion-confirmation'),
             'motion_44': Path('model-runs/improvements-v1/motion-confirmation')}
    roots.update({f'geometry_{s}': Path('model-runs/development-mixed-v1') for s in (42, 43, 44)})
    members, caches = [], {}
    for name, root in roots.items():
        if not completed(root/'complete.json'):
            raise ValueError(f'Incomplete source experiment: {root}')
        seed = int(name.split('_')[-1])
        selection = read(root/f'neural-seed-{seed}'/'selection.json')
        if not selection.get('complete'):
            raise ValueError(f'Incomplete source selection: {name}')
        folder = root/f'neural-seed-{seed}'/'best'
        spec = describe_member(name, 'neural', folder/'last.pt')
        expected = 'set-tcn-motion-v3' if name.startswith('motion') else 'set-tcn-geometry-v2'
        if (spec['architecture'] != expected or spec['seed'] != seed or
                spec['epoch'] != selection['best_epoch'] or spec['sampling'] != 'mixed' or
                spec.get('train') != split['train'] or spec.get('validation') != split['validation']):
            raise ValueError(f'Source selection/configuration mismatch: {name}')
        members.append(spec)
        caches[name] = str(folder)
    tree = Path('model-runs/development-mixed-v1/tree/tree.pkl')
    members.append(describe_member('tree', 'tree', tree))
    caches['tree'] = str(tree.parent)
    validate_members(members)
    prepared = paths('data-external/prepared', split['validation'])
    cache_files = [Path(folder)/f'{p.stem}-{scenario}-pred.npz'
                   for folder in caches.values() for p in prepared
                   for scenario in ('clean', 'ball_gaps_dropout')]
    return dict(version=1, scope='Repeated external development only; no final holdouts or target labels',
                train=split['train'], validation=split['validation'], members=members,
                configurations=configurations(), cached_folders=caches,
                input_hashes={str(p): sha256(p) for p in prepared},
                cached_prediction_hashes={str(p): sha256(p) for p in cache_files},
                scenarios=['clean', 'ball_gaps_dropout', 'noise_025m', 'target_ball_gaps'],
                standard_corruption_seed=123, stress_seed=2026,
                replay_seed_rule='2026 + period index within each match',
                wait_for='model-runs/motion-robust-v4/driver-complete.json',
                source_hashes={n: sha256(Path('modeling')/n) for n in
                               ('ensemble.py', 'ensemble_run.py', 'common.py', 'reliability.py',
                                'transfer_benchmark.py', 'predict_target.py', 'official_metrics.py',
                                'neural.py', 'motion.py', 'robust_features.py')},
                interpretation='Equal seed weights; fixed small tree sweep plus one geometry fallback. '
                'Development ranking is not an unbiased generalization estimate. '
                'The existing two-seed blend remains an explicit reference, not a selected deployment.')


def combine_case(periods, configurations_):
    """Preserve periods explicitly; unknown labels affect scoring, never resets."""
    from .ensemble import weighted_average
    from .reliability import brief
    ys, lengths, probability_parts, observed = [], [], {}, []
    for labels, probabilities, seen in periods:
        ys.extend([labels, np.array([3], np.int8)])
        lengths.extend([len(labels), 1])
        observed.extend([seen, np.array([False])])
        for name, prob in probabilities.items():
            if len(prob) != len(labels):
                raise ValueError('Prediction/period row mismatch')
            probability_parts.setdefault(name, []).extend([prob, np.full((1, 3), 1/3)])
    y, seen = np.concatenate(ys), np.concatenate(observed)
    probabilities = {name: np.concatenate(parts) for name, parts in probability_parts.items()}
    results = {}
    for config in configurations_:
        names = list(config['weights'])
        p = weighted_average([probabilities[name] for name in names],
                             [config['weights'][name] for name in names])
        result = brief(y, p, switch_cost=config['switch_cost'], lengths=lengths)
        result['conditional_raw_accuracy'] = {}
        for name, ix in [('observed', seen & (y < 3)), ('missing', ~seen & (y < 3))]:
            result['conditional_raw_accuracy'][name] = dict(frames=int(ix.sum()),
                accuracy=float((p[ix].argmax(1) == y[ix]).mean()) if ix.any() else None)
        results[config['id']] = result
    return dict(frames=int((y < 3).sum()), ball_missing_fraction=float((~seen[y < 3]).mean()),
                models=results)


def summarize(report):
    groups = {'standard': ['clean', 'ball_gaps_dropout'], 'noise': ['noise_025m'],
              'target_replay': ['target_ball_gaps']}
    summary = {}
    for group, scenarios in groups.items():
        rows = [v for key, v in report['cases'].items() if key.split('/')[0] in scenarios]
        expected = len(scenarios)*len(report['validation'])
        if len(rows) != expected:
            continue
        summary[group] = {name: {metric: float(np.mean([v['models'][name][metric] for v in rows]))
                                for metric in ('spell_f1_mean', 'macro_f1', 'alive_dead_f1')}
                          for name in rows[0]['models']}
    return summary


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem+'-', suffix='.npz', dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return sha256(path)


def save_prediction(path, probability):
    digest = save_npz(path, probabilities=probability)
    write_json(path.with_suffix('.json'), dict(sha256=digest, rows=len(probability)))


def cached_prediction(path):
    metadata = path.with_suffix('.json')
    if not path.exists() or not metadata.exists():
        return None
    if read(metadata)['sha256'] != sha256(path):
        raise ValueError(f'Damaged inference cache: {path}')
    return load(path)['probabilities']


def main(a):
    import torch
    from threadpoolctl import threadpool_limits
    from .ensemble import load_member, predict_member
    root = a.out
    with run_lock(root):
        torch.set_num_threads(1)
        plan = build_plan()
        if (root/'manifest.json').exists() and read(root/'manifest.json') != plan:
            raise ValueError('Pinned ensemble inputs/code differ; use a new output root')
        write_json(root/'manifest.json', plan)
        if completed(root/'complete.json'):
            return
        report = read(root/'report.json') if (root/'report.json').exists() else dict(
            status='running', validation=plan['validation'], cases={}, manifest_sha256=sha256(root/'manifest.json'),
            limitations=plan['interpretation']+' Stress noise is synthetic; target replay contributes only observation masks.')
        if report['manifest_sha256'] != sha256(root/'manifest.json'):
            raise ValueError('Report/manifest mismatch')
        def save_case(key, periods):
            with threadpool_limits(limits=1):
                report['cases'][key] = combine_case(periods, plan['configurations'])
            write_json(root/'report.json', report)
            write_json(root/'summary.json', summarize(report))
            write_json(root/'status.json', dict(state='running', stage=key, updated_at_utc=now()))
            print('ENSEMBLE', key, {k: round(v['spell_f1_mean'], 4)
                                  for k, v in report['cases'][key]['models'].items()}, flush=True)
        for scenario in ('clean', 'ball_gaps_dropout'):
            for match in plan['validation']:
                key = f'{scenario}/{match}'
                if key in report['cases']:
                    continue
                periods = []
                for p in paths('data-external/prepared', [match]):
                    d = load(p)
                    from .common import corrupt
                    changed = corrupt(d, seed=123) if scenario == 'ball_gaps_dropout' else d
                    probs = {name: load(Path(folder)/f'{p.stem}-{scenario}-pred.npz')['probabilities']
                             for name, folder in plan['cached_folders'].items()}
                    periods.append((d['labels'], probs, np.isfinite(changed['ball']).all(-1)))
                save_case(key, periods)
        # The cached comparison above is light. Serialize all fresh inference
        # after the existing training/reliability/transfer suite's final marker.
        while not completed(Path(plan['wait_for'])):
            status_path = Path(plan['wait_for']).parent/'status.json'
            if status_path.exists() and read(status_path).get('state') == 'failed':
                raise RuntimeError('Robust-motion dependency failed; repair it before continuing')
            write_json(root/'status.json', dict(state='waiting', stage='robust_motion_driver', updated_at_utc=now()))
            print('Waiting for robust-motion driver completion', flush=True)
            time.sleep(30)
        from .predict_target import read_target
        from .transfer_benchmark import variant
        from .reliability import stress
        _, target = read_target(a.repo)
        patterns = [np.isfinite(d['ball']).all(-1) for _, _, d in target]
        del target
        # Save only masks, with their source provenance, never target labels.
        mask_path = root/'target-observation-masks.npz'
        if not mask_path.exists() or not mask_path.with_suffix('.json').exists():
            digest = save_npz(mask_path, **{f'period_{i}': p for i, p in enumerate(patterns)})
            write_json(mask_path.with_suffix('.json'), dict(sha256=digest,
                source='read_target: ATD coordinates/metadata only, no possession labels',
                patterns=[dict(frames=len(p), missing_fraction=float((~p).mean())) for p in patterns]))
        else:
            previous = load(mask_path)
            if sha256(mask_path) != read(mask_path.with_suffix('.json'))['sha256'] or any(
                    not np.array_equal(previous[f'period_{i}'], p) for i, p in enumerate(patterns)):
                raise ValueError('Target observation masks changed')
        mask_hash = sha256(mask_path)
        if report.get('target_observation_masks_sha256', mask_hash) != mask_hash:
            raise ValueError('Pinned report observation masks changed')
        report['target_observation_masks_sha256'] = mask_hash
        write_json(root/'report.json', report)
        loaded = {m['id']: load_member(m) for m in plan['members']}
        for scenario in ('noise_025m', 'target_ball_gaps'):
            for match in plan['validation']:
                key = f'{scenario}/{match}'
                if key in report['cases']:
                    continue
                periods = []
                for i, p in enumerate(paths('data-external/prepared', [match])):
                    d = load(p)
                    changed = stress(d, scenario) if scenario == 'noise_025m' else variant(
                        d, scenario, patterns[i % len(patterns)], seed=2026+i)
                    probabilities = {}
                    for member in plan['members']:
                        cache = root/'predictions'/scenario/f"{p.stem}-{member['id']}.npz"
                        probability = cached_prediction(cache)
                        if probability is None:
                            with threadpool_limits(limits=1):
                                probability = predict_member(member, loaded[member['id']], changed)
                            save_prediction(cache, probability)
                        probabilities[member['id']] = probability
                    periods.append((d['labels'], probabilities, np.isfinite(changed['ball']).all(-1)))
                save_case(key, periods)
        report['status'] = 'completed'
        write_json(root/'report.json', report)
        write_json(root/'summary.json', summarize(report))
        write_json(root/'complete.json', dict(status='completed', manifest_sha256=sha256(root/'manifest.json'),
            report_sha256=sha256(root/'report.json'), target_observation_masks_sha256=mask_hash))
        write_json(root/'status.json', dict(state='completed', updated_at_utc=now()))
        print('Three-motion-seed ensemble evaluation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('model-runs/motion-ensemble-v1'))
    parser.add_argument('--repo', type=Path, default=Path('hackathon-windows-transfer/more-than-a-hack-2027'))
    a = parser.parse_args()
    try:
        main(a)
    except Exception as error:
        if not isinstance(error, AlreadyRunningError):
            write_json(a.out/'status.json', dict(state='failed', error=str(error), updated_at_utc=now()))
        raise
