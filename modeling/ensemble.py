"""Hash-pinned probability inference for trusted local H/A/D model artifacts.

Pickle and torch checkpoints are executable formats: hashing pins already-trusted
project artifacts; it does not make arbitrary downloaded files safe to open.
Paths are resolved against the process working directory. Inputs are full 25-Hz
periods in metres, and outputs always use class order H=0, A=1, D=2.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np


ARCHITECTURES = ('set-tcn-v1', 'set-tcn-geometry-v2', 'set-tcn-motion-v3',
                 'set-tcn-motion-robust-v4')
CLASS_ORDER = [0, 1, 2]


def _read(path, expected=None):
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Artifact hash mismatch: {path}')
    return data, digest


def _inspect(member_id, kind, path, expected=None, expected_report=None):
    if not isinstance(member_id, str) or not member_id.strip():
        raise ValueError('Member id must be a nonempty string')
    if kind not in ('neural', 'tree'):
        raise ValueError(f'Unsupported member kind: {kind}')
    path = Path(path)
    data, digest = _read(path, expected)
    spec = dict(id=member_id, kind=kind, path=str(path), sha256=digest,
                class_order=CLASS_ORDER.copy())
    if kind == 'neural':
        import torch
        artifact = torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)
        architecture = artifact.get('architecture', 'set-tcn-v1')
        if architecture not in ARCHITECTURES:
            raise ValueError(f'Unsupported architecture: {architecture}')
        spec.update(architecture=architecture, seed=int(artifact.get('seed', 42)),
                    epoch=int(artifact['epoch']), sampling=artifact.get('sampling', 'halfwise'))
    else:
        import pickle
        artifact = pickle.loads(data)
        if not np.array_equal(artifact['model'].classes_, CLASS_ORDER):
            raise ValueError('Tree class order must be exactly H/A/D [0, 1, 2]')
        names = artifact['features']
        if not isinstance(names, list) or not names or not all(isinstance(v, str) for v in names):
            raise ValueError('Tree feature names must be a nonempty string list')
        lag = artifact.get('velocity_lag', 1)
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
            raise ValueError('Tree velocity lag must be a positive integer')
        spec.update(velocity_lag=int(lag), features=names)
    report = path.parent/'report.json'
    provenance = {}
    if report.exists():
        raw, report_digest = _read(report, expected_report)
        provenance = json.loads(raw)
        spec['report_sha256'] = report_digest
    elif expected_report is not None:
        raise ValueError(f'Pinned companion report is missing: {report}')
    for role in ('train', 'validation'):
        value = provenance.get(role, artifact.get(role))
        if value is not None:
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise ValueError(f'Invalid {role} provenance')
            if role in provenance and role in artifact and artifact[role] != value:
                raise ValueError(f'Checkpoint/report {role} mismatch')
            spec[role] = value
    if set(spec.get('train', [])) & set(spec.get('validation', [])):
        raise ValueError('Member training/validation roles overlap')
    return spec, artifact


def describe_member(member_id, kind, path):
    """Inspect one trusted project artifact and return its JSON manifest entry."""
    return _inspect(member_id, kind, path)[0]


def validate_members(specs):
    """Validate member identities and compatible class/split contracts, no loads."""
    specs = list(specs)
    if not specs:
        raise ValueError('At least one ensemble member is required')
    ids, paths = set(), set()
    roles = {}
    for spec in specs:
        member_id = spec.get('id')
        if not isinstance(member_id, str) or not member_id.strip() or member_id in ids:
            raise ValueError('Member ids must be unique nonempty strings')
        ids.add(member_id)
        path = Path(spec['path']).resolve()
        if path in paths:
            raise ValueError('Duplicate ensemble artifact path')
        paths.add(path)
        if spec.get('kind') not in ('neural', 'tree') or spec.get('class_order') != CLASS_ORDER:
            raise ValueError('Unsupported member kind or class contract')
        if spec['kind'] == 'neural' and spec.get('architecture') not in ARCHITECTURES:
            raise ValueError('Unsupported neural architecture')
        for role in ('train', 'validation'):
            if role in spec:
                value = spec[role]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ValueError(f'Invalid {role} provenance')
                if role in roles and roles[role] != value:
                    raise ValueError(f'Ensemble {role} provenance mismatch')
                roles[role] = value
    if set(roles.get('train', [])) & set(roles.get('validation', [])):
        raise ValueError('Ensemble training/validation roles overlap')
    return specs


def load_member(spec):
    """Verify pinned bytes/metadata before returning a CPU model or tree artifact."""
    validate_members([spec])
    current, artifact = _inspect(spec['id'], spec['kind'], spec['path'],
                                 expected=spec['sha256'], expected_report=spec.get('report_sha256'))
    for key, value in current.items():
        if spec.get(key) != value:
            raise ValueError(f'Pinned member metadata mismatch: {key}')
    if spec['kind'] == 'tree':
        return artifact
    from .neural import PossessionNet
    model = PossessionNet(architecture=current['architecture'])
    model.load_state_dict(artifact['model'])
    model.eval()
    return model


def _probabilities(probabilities):
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p):
        raise ValueError('Expected nonempty T x 3 probabilities')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError('Probabilities must be finite, nonnegative and normalized')
    return p


def predict_member(spec, loaded, data):
    """Predict one full period, rebuilding features from coordinates every time.

    The allowlist excludes supervision and cached motion arrays. This prevents
    stale derivatives after coordinate corruption and v2/v3 feature-cache reuse.
    """
    geometry = {key: data[key] for key in ('players', 'ball', 'teams', 'pitch')}
    if 'fps' in data and not np.isclose(float(data['fps']), 25):
        raise ValueError('This ensemble requires 25-Hz periods')
    if spec['kind'] == 'tree':
        from .robust_features import tree_features
        if not np.array_equal(loaded['model'].classes_, CLASS_ORDER):
            raise ValueError('Tree class order must be exactly [0, 1, 2]')
        x, _ = tree_features(geometry, loaded)
        result = loaded['model'].predict_proba(x)
    elif spec['kind'] == 'neural':
        from .neural import predict
        if loaded.architecture != spec['architecture']:
            raise ValueError('Loaded neural architecture mismatch')
        result, _ = predict(loaded, geometry)
    else:
        raise ValueError('Unsupported member kind')
    result = _probabilities(result)
    if len(result) != len(geometry['ball']):
        raise ValueError('Prediction length differs from period length')
    return result


def weighted_average(probabilities, weights):
    """Blend class probabilities; normalize weights, never invalid model rows."""
    arrays = [_probabilities(p) for p in probabilities]
    w = np.asarray(weights, dtype=np.float64)
    if not arrays or w.ndim != 1 or len(w) != len(arrays):
        raise ValueError('One weight is required per probability array')
    if not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(w.sum()) or w.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative and have positive sum')
    if any(p.shape != arrays[0].shape for p in arrays):
        raise ValueError('Ensemble probability shapes differ')
    out = np.zeros(arrays[0].shape, dtype=np.float64)
    for p, weight in zip(arrays, w/w.sum()):
        out += weight*p
    return _probabilities(out)
