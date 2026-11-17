"""Sequential local development experiment with resumable validation selection."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from .common import load, paths, decode, evaluate, write_json
from .run_lock import run_lock


class EarlyStopping:
    def __init__(self, patience=3, min_delta=.002, best=None, stale=0):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.stale = best, stale

    def update(self, score):
        if not np.isfinite(score):
            raise ValueError('Nonfinite selection score')
        improved = self.best is None or score > self.best + self.min_delta
        if improved:
            self.best, self.stale = score, 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def run(*args):
    print('RUN', *args, flush=True)
    subprocess.run([sys.executable, '-m', *map(str, args)], check=True)


def validation_summary(folder, matches):
    report = json.loads((folder/'report.json').read_text())
    metrics = {}
    for match in matches:
        for scenario in ('clean', 'ball_gaps_dropout'):
            labels, probs = [], []
            for path in paths('data-external/prepared', [match]):
                y = load(path)['labels']; valid = y < 3
                labels.append(y[valid])
                probs.append(load(folder/f'{path.stem}-{scenario}-pred.npz')['probabilities'][valid])
            y, prob = np.concatenate(labels), np.concatenate(probs)
            nll = float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-8, 1)).mean())
            m = report['matches'][f'{match}/{scenario}']['decoded']
            metrics[f'{match}/{scenario}'] = dict(nll=nll, spell_f1=m['spell_f1_mean'], macro_f1=m['macro_f1'])
    return {'score': float(np.mean([v['spell_f1'] for v in metrics.values()])),
            'validation_nll': float(np.mean([v['nll'] for v in metrics.values()])), 'matches': metrics}


def analyze(folder, matches):
    """Bounded decoder sweep plus class errors and probability calibration."""
    from sklearn.metrics import confusion_matrix
    reports = {}
    for penalty in (0, 1, 3, 6, 10):
        scores = []
        for match in matches:
            for scenario in ('clean', 'ball_gaps_dropout'):
                truths, predictions = [], []
                confusion = np.zeros((3, 3), dtype=np.int64)
                bins = np.zeros((10, 3))  # count, confidence sum, correct sum
                for path in paths('data-external/prepared', [match]):
                    y = load(path)['labels']
                    prob = load(folder/f'{path.stem}-{scenario}-pred.npz')['probabilities']
                    pred = decode(prob, penalty)
                    truths.extend([y, np.array([3], np.int8)])
                    predictions.extend([pred, np.array([3], np.int8)])
                    valid = y < 3
                    confusion += confusion_matrix(y[valid], pred[valid], labels=[0, 1, 2])
                    confidence = prob[valid].max(1)
                    correct = prob[valid].argmax(1) == y[valid]
                    for b in range(10):
                        mask = np.minimum((confidence*10).astype(int), 9) == b
                        bins[b] += [mask.sum(), confidence[mask].sum(), correct[mask].sum()]
                m = evaluate(np.concatenate(truths), np.concatenate(predictions))
                scores.append(m['spell_f1_mean'])
                reports[f'{penalty}/{match}/{scenario}'] = dict(
                    spell_f1=m['spell_f1_mean'], macro_f1=m['macro_f1'],
                    true_spells=m['n_gt_spells'], confusion=confusion.tolist(),
                    raw_probability_ece=float(np.abs(bins[:, 1]-bins[:, 2]).sum()/bins[:, 0].sum()))
        reports[f'{penalty}/mean_spell_f1'] = float(np.mean(scores))
    write_json(folder/'error-analysis.json', reports)


def run_experiment(a):
    root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
    split = json.loads(Path('modeling/splits.json').read_text())
    train, validation = split['train'], split['validation']
    plan = dict(train=train, validation=validation, seeds=a.seeds,
        max_epochs=a.max_epochs, check_every=a.check_every, patience=a.patience,
        min_delta=.002, selection='Equal mean spell F1 across validation matches and clean/corrupted scenarios',
        limitations='Development selection only; no final holdouts or target labels')
    if a.sampling != 'halfwise':
        plan['sampling'] = a.sampling
    if a.architecture != 'set-tcn-geometry-v2':
        plan['architecture'] = a.architecture
    if a.tree_from:
        plan['tree_from'] = str(a.tree_from)
    if (root/'plan.json').exists() and json.loads((root/'plan.json').read_text()) != plan:
        raise ValueError('Existing experiment plan differs; use a fresh output folder')
    write_json(root/'plan.json', plan)
    for match in train + validation:
        run('modeling.download', '--matches', match)
        if not (Path('data-external/prepared')/f'{match}-audit.json').exists():
            run('modeling.prepare', '--match', match)
    tree = root/'tree'
    if a.tree_from and not (tree/'report.json').exists():
        source = json.loads((a.tree_from/'report.json').read_text())
        if source['train'] != train or source['validation'] != validation:
            raise ValueError('Reused tree has different data splits')
        shutil.copytree(a.tree_from, tree, dirs_exist_ok=True)
    if not (tree/'report.json').exists():
        run('modeling.train_tree', '--train', *train, '--validation', *validation, '--out', tree)
    summary = {'tree': validation_summary(tree, validation), 'neural': {}}
    write_json(root/'summary.json', summary)
    for seed in a.seeds:
        folder = root/f'neural-seed-{seed}'; folder.mkdir(exist_ok=True)
        progress_path = folder/'selection.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {'checks': [], 'complete': False}
        stop = EarlyStopping(a.patience, best=progress.get('best_score'), stale=progress.get('stale', 0))
        first = progress['checks'][-1]['epoch'] + a.check_every if progress['checks'] else a.check_every
        if not progress['complete']:
            for epoch in range(first, a.max_epochs+1, a.check_every):
                args = ['modeling.train_neural', '--train', *train, '--validation', *validation,
                        '--architecture', a.architecture, '--seed', seed, '--team-swap',
                        '--epochs', epoch, '--steps-per-half', 100, '--out', folder]
                if a.sampling != 'halfwise':
                    args += ['--sampling', a.sampling]
                if (folder/'last.pt').exists():
                    args += ['--resume', folder/'last.pt']
                run(*args)
                result = validation_summary(folder, validation)
                improved, done = stop.update(result['score'])
                if improved:
                    best = folder/'best'; best.mkdir(exist_ok=True)
                    for file in [folder/'last.pt', folder/'report.json', *folder.glob('*-pred.npz')]:
                        tmp = best/(file.name+'.tmp'); shutil.copy2(file, tmp); tmp.replace(best/file.name)
                    progress['best_epoch'] = epoch
                progress['checks'].append(dict(epoch=epoch, **result))
                progress.update(best_score=stop.best, stale=stop.stale, complete=done or epoch==a.max_epochs)
                write_json(progress_path, progress)
                print('VALIDATION', seed, epoch, result['score'], 'NLL', result['validation_nll'], 'best epoch', progress['best_epoch'], flush=True)
                if progress['complete']:
                    break
        summary['neural'][str(seed)] = dict(best_epoch=progress['best_epoch'], **validation_summary(folder/'best', validation))
        write_json(root/'summary.json', summary)
    for folder in [tree] + [root/f'neural-seed-{seed}'/'best' for seed in a.seeds]:
        if not (folder/'error-analysis.json').exists():
            analyze(folder, validation)
    write_json(root/'complete.json', {'status': 'completed', 'seeds': a.seeds, 'architecture': a.architecture,
        'scope': 'Expanded data, tree, neural seeds, checkpoint selection, decoder/error/calibration analysis'})
    print('Development experiment complete', flush=True)


def main(a):
    with run_lock(a.out):
        run_experiment(a)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='model-runs/development-v2')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--check-every', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--sampling', choices=['halfwise', 'mixed'], default='halfwise')
    p.add_argument('--tree-from', type=Path)
    p.add_argument('--architecture', choices=['set-tcn-geometry-v2', 'set-tcn-motion-v3', 'set-tcn-motion-robust-v4'], default='set-tcn-geometry-v2')
    a = p.parse_args()
    if a.max_epochs < a.check_every or a.max_epochs % a.check_every or a.check_every < 1 or a.patience < 1:
        p.error('Use positive intervals/patience and max epochs divisible by check interval')
    main(a)
