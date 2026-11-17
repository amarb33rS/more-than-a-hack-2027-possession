"""Verify this private local release, then run an unlabelled synthetic parity check."""
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits

root = Path(__file__).resolve().parent
if Path.cwd().resolve() != root:
    raise SystemExit('Run from the extracted bundle directory')
manifest = json.loads((root/'FILES.json').read_text())
for name, spec in manifest['files'].items():
    path = root/name
    if path.stat().st_size != spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise SystemExit('Release artifact changed: '+name)
from modeling.frozen_evaluate import validate_candidate
from modeling.ensemble import load_member
from modeling.predict_frozen_target import predict_period
candidate, digest = validate_candidate(Path('candidate/manifest.json'))
torch.set_num_threads(1)
with threadpool_limits(limits=1):
    loaded = {m['id']: load_member(m) for m in candidate['members']}
    with np.load('verification/geometry-only.npz', allow_pickle=False) as a:
        geometry = {key: a[key] for key in ('players','ball','teams','pitch','fps')}
    probabilities, prediction = predict_period(candidate, loaded, geometry)
with np.load('verification/expected.npz', allow_pickle=False) as expected:
    if not np.array_equal(probabilities, expected['probabilities']) or not np.array_equal(prediction, expected['prediction']):
        raise SystemExit('Synthetic numerical parity failed; inspect runtime versions')
print('Verified all file hashes and bit-exact synthetic inference parity. No match labels opened.')
