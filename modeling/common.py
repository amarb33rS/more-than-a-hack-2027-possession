"""Shared input features, corruption, Viterbi and official evaluation."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def corrupt(a, seed=42):
    """Fixed stress scenario, not a fitted replica of target missingness."""
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in a.items()}
    n = len(a['ball'])
    for start in range(0, n, 500):
        length = int(rng.integers(50, 300))
        out['ball'][start:start + length] = np.nan
    dropped = rng.random(out['players'].shape[:2]) < .08
    out['players'][dropped] = np.nan
    return out


def features(a):
    """Geometry only: the label array is never accessed here."""
    p, b = a['players'], a['ball']
    seen = np.isfinite(b).all(-1)
    ps = np.isfinite(p).all(-1)
    velocity = np.full_like(p, np.nan)
    velocity[1:] = (p[1:] - p[:-1]) * 25
    speed = np.linalg.norm(velocity, axis=-1)
    speed[speed > 15] = np.nan
    bs = np.full(len(b), np.nan, np.float32)
    bs[1:] = np.linalg.norm(b[1:] - b[:-1], axis=-1) * 25
    bs[bs > 50] = np.nan
    idx = np.arange(len(b))
    prev = np.maximum.accumulate(np.where(seen, idx, -100000))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10000000)[::-1])[::-1]
    cols = [seen, np.minimum(idx - prev, 250) / 25, np.minimum(nxt - idx, 250) / 25,
            b[:, 0] / a['pitch'][0], b[:, 1] / a['pitch'][1], bs]
    names = ['ball_seen', 'since_ball_s', 'until_ball_s', 'ball_x', 'ball_y', 'ball_speed']
    for team in (0, 1):
        subset = a['teams'] == team
        q = p[:, subset]
        valid = ps[:, subset]
        d = np.linalg.norm(q - b[:, None], axis=-1)
        ordered = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
        for j in range(3):
            c = ordered[:, j] if ordered.shape[1] > j else np.full(len(b), np.nan)
            cols.append(np.where(np.isfinite(c), c, np.nan))
            names.append(f'{team}_distance_{j}')
        count = valid.sum(1)
        center = np.nansum(q, axis=1) / np.maximum(count[:, None], 1)
        sp = speed[:, subset]
        mean_speed = np.nansum(sp, axis=1) / np.maximum(np.isfinite(sp).sum(1), 1)
        cols.extend([count, center[:, 0], center[:, 1], mean_speed, (d < 5).sum(1)])
        names.extend([f'{team}_{x}' for x in ('count', 'center_x', 'center_y', 'speed', 'near_ball')])
    x = np.column_stack(cols).astype(np.float32)
    # Offline context, clipped to this period. No crossing half-time.
    context = []
    for lag in (-25, 25):
        context.append(x[np.clip(idx + lag, 0, len(x) - 1)])
    return np.concatenate([x] + context, axis=1), names + [f'{n}_{l}' for l in (-25, 25) for n in names]


def decode(prob, switch_cost=3.0, boundary=None):
    """Exact Viterbi; boundary confidence can reduce the transition penalty."""
    prob = np.asarray(prob)
    if prob.ndim != 2 or prob.shape[1] != 3 or not len(prob):
        raise ValueError('Expected nonempty T x 3 probabilities')
    if not np.isfinite(prob).all() or (prob < 0).any():
        raise ValueError('Invalid probabilities')
    emission = np.log(np.clip(prob, 1e-8, 1))
    score = emission[0].copy()
    back = np.zeros((len(prob), 3), np.int8)
    for t in range(1, len(prob)):
        penalty = switch_cost if boundary is None else switch_cost * (1 - boundary[t])
        candidates = score[:, None] - penalty * (1 - np.eye(3))
        back[t] = candidates.argmax(0)
        score = candidates.max(0) + emission[t]
    out = np.empty(len(prob), np.int8)
    out[-1] = score.argmax()
    for t in range(len(prob) - 1, 0, -1):
        out[t-1] = back[t, out[t]]
    return out


def evaluate(y, pred):
    from .official_metrics import evaluate as official
    return official(y, pred)


def nearest_baseline(a, half_width=50):
    """Organizer nearest-ball/majority logic, adapted to the shared geometry."""
    distance = np.linalg.norm(a['players'] - a['ball'][:, None], axis=-1)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    nearest = distance.argmin(1)
    labels = np.where(np.isfinite(distance.min(1)), a['teams'][nearest], 2)
    if not half_width:
        return labels
    idx = np.arange(len(labels))
    lo = np.maximum(0, idx-half_width)
    hi = np.minimum(len(labels), idx+half_width+1)
    votes = []
    for team in (0, 1):
        count = np.r_[0, np.cumsum(labels == team)]
        votes.append(count[hi]-count[lo])
    return np.where(votes[0]+votes[1] == 0, 2, np.where(votes[0] >= votes[1], 0, 1))


def paths(folder, ids):
    result = []
    for match in ids:
        found = sorted(Path(folder).glob(f'{match}-*.npz'))
        if len(found) != 2:
            raise ValueError(f'Expected two prepared halves for {match}, found {len(found)}')
        result.extend(found)
    return result


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name+'.', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            json.dump(obj, f, indent=2, allow_nan=False)
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.05)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
