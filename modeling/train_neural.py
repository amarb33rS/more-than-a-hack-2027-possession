"""Single-GPU/CPU trainer with checkpoint resume; only public external inputs."""
import argparse
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import random
import time
import platform
import numpy as np
import torch
from torch.nn import functional as F
from .common import load, paths, corrupt, evaluate, decode, write_json, nearest_baseline
from .neural import PossessionNet, tensors, predict, HALO, swap_team_labels


def training_windows(training_paths, rng, steps, window, epoch, sampling, motion=False, motion_lag=1):
    """Same window budget; optional balanced conditions and interleaved periods."""
    if sampling == 'mixed':
        if steps % 2:
            raise ValueError('Mixed sampling requires an even steps-per-half')
        # Development data fits in host RAM; each half has only one clean/noisy copy.
        periods = [load(path) for path in training_paths]
        noisy = [corrupt(d, 42+epoch) for d in periods]
        if motion:
            from .motion import augment_motion
            periods = [augment_motion(d, velocity_lag=motion_lag) for d in periods]
            noisy = [augment_motion(d, velocity_lag=motion_lag) for d in noisy]
        tasks = [(i, condition) for i in range(len(periods))
                 for condition in (0, 1) for _ in range(steps//2)]
        rng.shuffle(tasks)
        for i, condition in tasks:
            d = (periods, noisy)[condition][i]
            n = len(d['ball'])
            if n < window+2*HALO:
                raise ValueError('Period shorter than training context')
            yield d, int(rng.integers(HALO, n-window-HALO+1))
    elif sampling == 'halfwise':
        # Preserve the original sequence, RNG consumption and memory footprint.
        for path in training_paths:
            d = load(path)
            d = corrupt(d, 42+epoch) if epoch % 2 else d
            if motion:
                from .motion import augment_motion
                d = augment_motion(d, velocity_lag=motion_lag)
            n = len(d['ball'])
            if n < window+2*HALO:
                raise ValueError('Period shorter than training context')
            for _ in range(steps):
                yield d, int(rng.integers(HALO, n-window-HALO+1))
    else:
        raise ValueError('Unknown sampling strategy')


def main(a):
    if set(a.train) & set(a.validation):
        raise ValueError('Training/validation match overlap')
    split = json.loads(Path('modeling/splits.json').read_text())
    if (set(a.train) | set(a.validation)) & set(split['holdout']):
        raise ValueError('Holdouts are excluded from development')
    if not set(a.train) <= set(split['train']) or not set(a.validation) <= set(split['validation']):
        raise ValueError('Requested matches do not respect the frozen split roles')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if a.require_cuda and device != 'cuda':
        raise RuntimeError('No CUDA GPU attached. Enable a GPU runtime first.')
    torch.use_deterministic_algorithms(True)
    model = PossessionNet(architecture=a.architecture).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    rng = np.random.default_rng(a.seed)
    start_epoch = 0
    if a.resume:
        state = torch.load(a.resume, map_location=device, weights_only=False)
        if state.get('architecture', 'set-tcn-v1') != a.architecture:
            raise ValueError('Resume architecture differs; specify the checkpoint architecture')
        if state['train'] != a.train:
            raise ValueError('Resume training split differs')
        if state.get('seed', 42) != a.seed or state.get('team_swap', False) != a.team_swap:
            raise ValueError('Resume seed or augmentation differs')
        if state.get('sampling', 'halfwise') != a.sampling:
            raise ValueError('Resume sampling differs; use a fresh experiment')
        for key in ('window', 'steps_per_half'):
            if key in state and state[key] != getattr(a, key):
                raise ValueError(f'Resume {key} differs')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        rng.bit_generator.state = state['rng']
    training_paths = paths(a.data, a.train)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    history = state.get('history', []) if a.resume else []
    started = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        model.train()
        losses = []
        for d, lo in training_windows(training_paths, rng, a.steps_per_half, a.window, epoch, a.sampling,
                                       motion=a.architecture in ('set-tcn-motion-v3', 'set-tcn-motion-robust-v4'),
                                       motion_lag=5 if a.architecture == 'set-tcn-motion-robust-v4' else 1):
            hi = lo + a.window
            inputs = [t.unsqueeze(0).to(device) for t in tensors(d, lo-HALO, hi+HALO)]
            labels = torch.tensor(d['labels'][lo:hi], dtype=torch.long, device=device)
            valid = labels < 3
            if not valid.any():
                continue
            previous = torch.tensor(d['labels'][lo-1:hi-1], device=device)
            if a.team_swap and rng.random() < .5:
                inputs[2] = swap_team_labels(inputs[2])
                labels = swap_team_labels(labels)
                previous = swap_team_labels(previous)
            logits, boundary = model(*inputs)
            logits = logits[0, HALO:HALO+a.window]
            boundary = boundary[0, HALO:HALO+a.window]
            bv = valid & (previous < 3)
            changed = (labels != previous).float()
            state_loss = F.cross_entropy(logits[valid], labels[valid])
            boundary_loss = F.binary_cross_entropy_with_logits(boundary[bv], changed[bv]) if bv.any() else state_loss*0
            loss = state_loss + .2*boundary_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step(); losses.append(float(loss.detach()))
        if not losses:
            raise ValueError('No labelled training windows')
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses))})
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch+1,
                    'rng': rng.bit_generator.state, 'train': a.train, 'history': history,
                    'architecture': a.architecture, 'seed': a.seed, 'team_swap': a.team_swap,
                    'sampling': a.sampling, 'window': a.window, 'steps_per_half': a.steps_per_half}, out/'last.pt.tmp')
        (out/'last.pt.tmp').replace(out/'last.pt')
        print(f'Epoch {epoch+1}: loss {np.mean(losses):.4f} ({device})', flush=True)
    report = {'device': device, 'architecture': a.architecture, 'seed': a.seed, 'team_swap': a.team_swap, 'sampling': a.sampling, 'window': a.window, 'steps_per_half': a.steps_per_half, 'train': a.train, 'validation': a.validation, 'history': history,
              'periods': {}, 'training_seconds': time.perf_counter()-started,
              'parameters': sum(p.numel() for p in model.parameters()),
              'torch': torch.__version__, 'python': platform.python_version(),
              'peak_gpu_memory_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'status': 'Smoke training only' if not a.validation else 'External provider-label validation'}
    for path in paths(a.data, a.validation):
        d = load(path)
        for scenario, variant in [('clean', d), ('ball_gaps_dropout', corrupt(d, 123))]:
            prob, boundary = predict(model, variant, device)
            pred = decode(prob)  # auxiliary boundary head uncalibrated; not fused yet
            report['periods'][f'{path.stem}/{scenario}'] = evaluate(d['labels'], pred)
            np.savez_compressed(out/f'{path.stem}-{scenario}-pred.npz', probabilities=prob,
                                boundary=boundary, prediction=pred, baseline=nearest_baseline(variant))
    report['matches'] = {}
    for match in a.validation:
        for scenario in ('clean', 'ball_gaps_dropout'):
            truth, prediction, baseline = [], [], []
            for path in paths(a.data, [match]):
                result = load(out/f'{path.stem}-{scenario}-pred.npz')
                for target_list, array in ((truth, load(path)['labels']), (prediction, result['prediction']), (baseline, result['baseline'])):
                    target_list.extend([array, np.array([3], np.int8)])
            report['matches'][f'{match}/{scenario}'] = {
                'decoded': evaluate(np.concatenate(truth), np.concatenate(prediction)),
                'nearest_smooth50': evaluate(np.concatenate(truth), np.concatenate(baseline))}
    write_json(out/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data-external/prepared')
    p.add_argument('--train', nargs='+', required=True)
    p.add_argument('--validation', nargs='*', default=[])
    p.add_argument('--out', default='model-runs/neural')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps-per-half', type=int, default=100)
    p.add_argument('--window', type=int, default=512)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--team-swap', action='store_true')
    p.add_argument('--sampling', choices=['halfwise', 'mixed'], default='halfwise')
    p.add_argument('--architecture', choices=['set-tcn-v1', 'set-tcn-geometry-v2', 'set-tcn-motion-v3', 'set-tcn-motion-robust-v4'], default='set-tcn-v1')
    main(p.parse_args())
