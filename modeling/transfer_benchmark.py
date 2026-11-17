"""Replay unlabelled target observation patterns onto labelled external matches."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .common import load, paths, write_json
from .robust_features import tree_features
from .predict_target import read_target
from .neural import PossessionNet, predict
from .reliability import brief
from .download import sha256


def observation_mask(pattern, n, seed):
    pattern = np.asarray(pattern, bool)
    if not len(pattern):
        raise ValueError('Empty observation pattern')
    offset = int(np.random.default_rng(seed).integers(len(pattern)))
    return pattern[(np.arange(n)+offset) % len(pattern)]


def variant(d, name, pattern, seed=2026):
    out = {k: v.copy() for k, v in d.items()}
    rng = np.random.default_rng(seed)
    n = len(out['ball'])
    if name in ('target_ball_gaps', 'iid_same_missing_count', 'combined'):
        seen = observation_mask(pattern, n, seed)
        if name == 'iid_same_missing_count':
            seen = rng.permutation(seen)
        out['ball'][~seen] = np.nan
    if name in ('identity_shuffle', 'combined'):
        # Change slot identity while preserving each frame's team-labelled point set.
        for lo in range(250, n, 250):
            hi = min(n, lo+250)
            for team in (0, 1):
                slots = np.flatnonzero(d['teams'] == team)
                order = rng.permutation(slots)
                out['players'][lo:hi, slots] = d['players'][lo:hi, order]
    if name in ('duplicate_bursts', 'combined'):
        # Short duplicate bursts on ~2% of frames; candidates, not a target replica.
        copies = np.full((n, 2, 2), np.nan, np.float32)
        for lo in range(250, n, 2500):
            for team in (0, 1):
                slots = np.flatnonzero((d['teams'] == team) & np.isfinite(d['players'][lo]).all(-1))
                if len(slots):
                    slot = int(rng.choice(slots))
                    copies[lo:lo+50, team] = d['players'][lo:lo+50, slot]
        out['players'] = np.concatenate([out['players'], copies], axis=1)
        out['teams'] = np.r_[out['teams'], [0, 1]].astype(np.int8)
    if name not in ('clean', 'target_ball_gaps', 'iid_same_missing_count', 'identity_shuffle', 'duplicate_bursts', 'combined'):
        raise ValueError(name)
    return out


def main(a):
    root = a.root
    split = json.loads(Path('modeling/splits.json').read_text())
    folder = root/f'neural-seed-{a.seed}'/'best'
    selected = json.loads((folder/'report.json').read_text())
    if selected['train'] != split['train'] or selected['validation'] != split['validation']:
        raise ValueError('Checkpoint splits differ from frozen development split')
    torch.set_num_threads(4)
    checkpoint = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    model = PossessionNet(architecture=checkpoint['architecture'])
    model.load_state_dict(checkpoint['model'])
    tree_path = a.tree_model or root/'tree'/'tree.pkl'
    with open(tree_path, 'rb') as f:
        tree_artifact = pickle.load(f)
    tree = tree_artifact['model']
    _, target = read_target(a.repo)
    patterns = [np.isfinite(d['ball']).all(-1) for _, _, d in target]
    del target
    report = dict(neural_seed=a.seed, neural_epoch=checkpoint['epoch'],
        tree_model=str(tree_path), tree_sha256=sha256(tree_path),
        tree_velocity_lag=tree_artifact.get('velocity_lag', 1),
        neural_checkpoint_sha256=sha256(folder/'last.pt'),
        neural_sampling=checkpoint.get('sampling', 'halfwise'),
        limitations='External development evaluation only. Target contributes observation masks, not labels or coordinates. Replayed masks preserve gap structure but not its association with match events. Circular replay creates a possible wrap seam. Identity shuffles preserve spatial sets; duplication bursts are a synthetic stress test.',
        matches={})
    out = a.out or root/f'transfer-benchmark-seed-{a.seed}.json'
    for name in ('clean', 'target_ball_gaps', 'iid_same_missing_count', 'identity_shuffle', 'duplicate_bursts', 'combined'):
        for match in split['validation']:
            ys, tp, nn, masks = [], [], [], []
            for i, path in enumerate(paths('data-external/prepared', [match])):
                d = load(path); changed = variant(d, name, patterns[i % len(patterns)], seed=2026+i)
                x, _ = tree_features(changed, tree_artifact)
                with threadpool_limits(limits=4):
                    tree_prob = tree.predict_proba(x)
                neural_prob, _ = predict(model, changed)
                ys.extend([d['labels'], np.array([3], np.int8)])
                tp.extend([tree_prob, np.full((1, 3), 1/3)])
                nn.extend([neural_prob, np.full((1, 3), 1/3)])
                masks.extend([np.isfinite(changed['ball']).all(-1), np.array([False])])
            y, t, p, seen = map(np.concatenate, (ys, tp, nn, masks))
            lengths = [len(v) for v in ys]
            result = {'ball_missing_fraction': float((~seen[y < 3]).mean()), 'models': {}}
            for label, probabilities in [('tree', t), ('neural', p)]:
                metrics = {str(cost): brief(y, probabilities, switch_cost=cost, lengths=lengths) for cost in (3, 10)}
                conditional = {}
                for label_seen, ix in [('observed', seen & (y < 3)), ('missing', ~seen & (y < 3))]:
                    conditional[label_seen] = dict(frames=int(ix.sum()), raw_frame_accuracy=float((probabilities[ix].argmax(1) == y[ix]).mean()) if ix.any() else None)
                result['models'][label] = dict(decoder=metrics, conditional_accuracy=conditional)
            report['matches'][f'{name}/{match}'] = result
            write_json(out, report)
            print('TRANSFER', name, match, 'missing', round(result['ball_missing_fraction'], 3),
                  {label: round(v['decoder']['3']['spell_f1_mean'], 4) for label, v in result['models'].items()}, flush=True)
    report['status'] = 'completed'
    write_json(out, report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('model-runs/development-v2'))
    p.add_argument('--repo', type=Path, default=Path('hackathon-windows-transfer/more-than-a-hack-2027'))
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tree-model', type=Path)
    p.add_argument('--out', type=Path)
    main(p.parse_args())
