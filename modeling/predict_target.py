"""Local-only ATD inference. Never opens target ground truth. Experimental output."""
import argparse
import importlib.util
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from threadpoolctl import threadpool_limits
from .common import decode, write_json
from .robust_features import tree_features
from .download import sha256


def read_target(repo):
    spec = importlib.util.spec_from_file_location('target_games', repo/'games.py')
    games = importlib.util.module_from_spec(spec); spec.loader.exec_module(games)
    config = games.GAMES['BAR-ATH']
    assets = games.asset_paths(data_dir=str(repo/'data'))
    root = ET.parse(assets['atd_meta']).getroot()
    ids = [t.get('id') for t in root.iter('Team')]
    if len(ids) != 2:
        raise ValueError('Expected two teams')
    teams = np.array([ids.index(p.get('teamId')) for p in root.iter('Player')], np.int8)
    n = config['n_video_frames']
    players = np.full((n, len(teams), 2), np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    pitch = np.array(config['pitch'], np.float32)
    seen = np.zeros(n, bool)
    with open(assets['atd_raw']) as f:
        for line in f:
            frame, ps, bs = line.strip().split(':')
            i = int(frame)-1
            if not 0 <= i < n or seen[i]:
                raise ValueError('Unexpected or duplicate ATD frame')
            seen[i] = True
            groups = ps.split(';')
            if len(groups) != len(teams):
                raise ValueError('Player slot mismatch')
            players[i] = [[float(v) for v in group.split(',')[:2]] for group in groups]
            ball[i] = [float(v) for v in bs.split(',')[:2]]
    if not seen.all():
        raise ValueError('ATD does not cover the expected video frames')
    players = (players - .5) * pitch
    ball = (ball - .5) * pitch
    blocks = []
    for half, (start, stop) in config['halves'].items():
        lo, hi = start-config['offsets'][half], stop-config['offsets'][half]+1
        blocks.append((lo, hi, dict(players=players[lo:hi], ball=ball[lo:hi], teams=teams, pitch=pitch)))
    return n, blocks


def main(a):
    with open(a.model, 'rb') as f:
        saved = pickle.load(f)  # only load our own trusted training artifact
    n, blocks = read_target(Path(a.repo))
    labels = np.full(n, 'X', dtype='<U1')
    for lo, hi, data in blocks:
        x, names = tree_features(data, saved)
        with threadpool_limits(limits=4):
            prob = saved['model'].predict_proba(x)
        labels[lo:hi] = np.array(list('HAD'))[decode(prob, saved['switch_cost'])]
    dest = Path(a.out); dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, 'w', newline='') as f:
        f.write('frame,label\n')
        for i, label in enumerate(labels):
            f.write(f'{i},{label}\n')
    write_json(dest.with_suffix('.json'), {'status': 'Experimental; not selected for submission',
               'model_sha256': sha256(a.model), 'csv_sha256': sha256(dest), 'frames': n})
    print(f'Wrote {n} frames to {dest}; target labels were not read.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default='hackathon-windows-transfer/more-than-a-hack-2027')
    p.add_argument('--model', default='model-runs/tree/tree.pkl')
    p.add_argument('--out', default='model-runs/tree/experimental-target.csv')
    main(p.parse_args())
