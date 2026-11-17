"""Development-only robustness, duration, calibration and fusion experiments."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from threadpoolctl import threadpool_limits
from .common import load, paths, decode, evaluate, write_json
from . import official_metrics as metric
from .robust_features import tree_features


def calibration(y, prob):
    valid = y < 3
    y, prob = y[valid], prob[valid]
    conf, pred = prob.max(1), prob.argmax(1)
    bins = []
    for b in range(10):
        ix = np.minimum((conf*10).astype(int), 9) == b
        bins.append(dict(count=int(ix.sum()), confidence=float(conf[ix].mean()) if ix.any() else None,
                         accuracy=float((pred[ix] == y[ix]).mean()) if ix.any() else None))
    ece = sum(v['count']*abs(v['confidence']-v['accuracy']) for v in bins if v['count'])/len(y)
    return dict(nll=float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-8, 1)).mean()),
                ece=float(ece), bins=bins)


def duration_recall(y, pred):
    runs = metric.valid_runs(y)
    truth, estimated = metric.possession_spells(y, runs), metric.possession_spells(pred, runs)
    flags = metric.matched_gt_flags(metric.spell_overlaps(truth, estimated), len(truth), .5)
    durations = np.array([(b-a)/25 for a, b, _ in truth])
    result = {}
    for name, lo, hi in [('under_2s', 0, 2), ('2_to_10s', 2, 10), ('10s_plus', 10, np.inf)]:
        ix = (durations >= lo) & (durations < hi)
        result[name] = dict(count=int(ix.sum()), matched=int(flags[ix].sum()),
                            recall_at_iou_05=float(flags[ix].mean()) if ix.any() else None)
    return result


def brief(y, prob, boundary=None, switch_cost=3, lengths=None):
    # Decode periods independently, using explicit input lengths, never GT labels.
    lengths = [len(y)] if lengths is None else lengths
    if sum(lengths) != len(y):
        raise ValueError('Period lengths do not match')
    predictions, lo = [], 0
    for length in lengths:
        hi = lo+length
        predictions.append(decode(prob[lo:hi], switch_cost,
                                  boundary=None if boundary is None else boundary[lo:hi]))
        lo = hi
    pred = np.concatenate(predictions)
    result = evaluate(y, pred)
    return {**{k: result[k] for k in ('spell_f1_mean', 'macro_f1', 'alive_dead_f1', 'n_gt_spells', 'n_pred_spells')},
            'calibration': calibration(y, prob), 'duration_recall': duration_recall(y, pred)}


def stress(d, kind):
    """Fixed, reproducible stress tests; independent of labels."""
    out = {k: v.copy() for k, v in d.items()}
    rng = np.random.default_rng(2026)
    if kind.startswith('gap_'):
        frames = int(kind.split('_')[1])
        for lo in range(250, len(d['ball']), 1000):
            out['ball'][lo:lo+frames] = np.nan
    elif kind == 'dropout_20pct':
        out['players'][rng.random(d['players'].shape[:2]) < .2] = np.nan
    elif kind == 'noise_025m':
        for field in ('players', 'ball'):
            out[field] += rng.normal(0, .25, out[field].shape).astype('float32')
    else:
        raise ValueError(kind)
    return out


def temperature(prob, value):
    z = np.log(np.clip(prob, 1e-8, 1))/value
    p = np.exp(z-z.max(1, keepdims=True))
    return p/p.sum(1, keepdims=True)


def main(root):
    plan = json.loads((root/'plan.json').read_text())
    split = json.loads(Path('modeling/splits.json').read_text())
    if plan['validation'] != split['validation'] or not (root/'complete.json').exists():
        raise ValueError('Requires completed frozen development experiment')
    summary = json.loads((root/'summary.json').read_text())
    best_seed = max(summary['neural'], key=lambda s: summary['neural'][s]['score'])
    neural_folders = [root/f'neural-seed-{s}'/'best' for s in plan['seeds']]
    selected = root/f'neural-seed-{best_seed}'/'best'
    out = root/'reliability'; out.mkdir(exist_ok=True)
    fusion, per_model, temperatures = {}, {}, {}
    for match in plan['validation']:
        for scenario in ('clean', 'ball_gaps_dropout'):
            ys, tp, ns, bs = [], [], [], []
            for path in paths('data-external/prepared', [match]):
                ys.extend([load(path)['labels'], np.array([3], np.int8)])
                tp.extend([load(root/'tree'/f'{path.stem}-{scenario}-pred.npz')['probabilities'], np.full((1, 3), 1/3)])
                results = [load(f/f'{path.stem}-{scenario}-pred.npz') for f in neural_folders]
                ns.extend([np.mean([r['probabilities'] for r in results], axis=0), np.full((1, 3), 1/3)])
                bs.extend([np.mean([r['boundary'] for r in results], axis=0), np.zeros(1)])
            y, tree, neural, boundary = map(np.concatenate, (ys, tp, ns, bs))
            lengths = [len(v) for v in ys]
            key = f'{match}/{scenario}'
            per_model[key] = {'tree': brief(y, tree, lengths=lengths), 'neural_seed_average': brief(y, neural, lengths=lengths),
                              'neural_with_boundary': brief(y, neural, boundary=boundary, lengths=lengths)}
            # A bounded score-to-cost sweep, not a claim of calibrated boundary probabilities.
            for strength in (5, 20):
                mapped = .75*np.clip(strength*boundary, 0, 1)
                per_model[key][f'neural_boundary_strength_{strength}_cost_10'] = brief(
                    y, neural, boundary=mapped, switch_cost=10, lengths=lengths)
            per_model[key]['neural_static_cost_10'] = brief(y, neural, switch_cost=10, lengths=lengths)
            from sklearn.metrics import average_precision_score
            bv = (y[1:] < 3) & (y[:-1] < 3)
            changed = (y[1:] != y[:-1])[bv]
            per_model[key]['boundary_quality'] = dict(prevalence=float(changed.mean()),
                average_precision=float(average_precision_score(changed, boundary[1:][bv])) if changed.any() else None,
                mapping_note='Strength sweep caps penalty reduction at 75%; development tuning only')
            valid = y < 3
            twrong, nwrong = tree[valid].argmax(1) != y[valid], neural[valid].argmax(1) != y[valid]
            per_model[key]['error_overlap'] = dict(both_wrong=float((twrong & nwrong).mean()),
                only_tree_wrong=float((twrong & ~nwrong).mean()), only_neural_wrong=float((~twrong & nwrong).mean()))
            for weight in (.25, .5, .6, .75):
                fusion[f'{weight}/{key}'] = brief(y, weight*tree+(1-weight)*neural, lengths=lengths)
                fusion[f'{weight}/cost_10/{key}'] = brief(y, weight*tree+(1-weight)*neural,
                                                       switch_cost=10, lengths=lengths)
            for value in (.75, 1, 1.5, 2):
                temperatures[f'{value}/{key}'] = calibration(y, temperature(neural, value))
    write_json(out/'fusion-and-calibration.json', dict(selected_seed=best_seed, models=per_model,
        tree_weight_sweep=fusion, neural_temperature_sweep=temperatures,
        limitations='All tuning and scores use development validation. Confirm choices on final holdouts only after freezing the model. ECE uses raw frame probabilities; duration recall uses one-to-one IoU .5 matching.'))
    print('Fusion, boundary, duration and calibration analysis saved', flush=True)
    import torch
    from .neural import PossessionNet, predict
    torch.set_num_threads(4)
    state = torch.load(selected/'last.pt', map_location='cpu', weights_only=False)
    model = PossessionNet(architecture=state['architecture']); model.load_state_dict(state['model'])
    with open(root/'tree'/'tree.pkl', 'rb') as f:
        tree_artifact = pickle.load(f)
    tree = tree_artifact['model']
    results = {}
    for kind in ('gap_25', 'gap_75', 'gap_250', 'dropout_20pct', 'noise_025m'):
        for match in plan['validation']:
            ys, tp, npred = [], [], []
            for path in paths('data-external/prepared', [match]):
                d = load(path); variant = stress(d, kind)
                x, _ = tree_features(variant, tree_artifact)
                with threadpool_limits(limits=4):
                    p = tree.predict_proba(x)
                n, _ = predict(model, variant)
                ys.extend([d['labels'], np.array([3], np.int8)])
                tp.extend([p, np.full((1, 3), 1/3)])
                npred.extend([n, np.full((1, 3), 1/3)])
            y = np.concatenate(ys)
            lengths = [len(v) for v in ys]
            results[f'{kind}/{match}'] = {'tree': brief(y, np.concatenate(tp), lengths=lengths),
                                         'selected_neural': brief(y, np.concatenate(npred), lengths=lengths)}
            write_json(out/'stress.json', results)
            print('STRESS', kind, match, flush=True)
    write_json(out/'complete.json', {'status': 'completed', 'selected_neural_seed': best_seed})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('model-runs/development-v2'))
    main(p.parse_args().root)
