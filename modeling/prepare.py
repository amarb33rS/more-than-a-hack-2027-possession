"""Two-pass, bounded-memory IDSSE XML adapter. Unknown targets stay masked."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .download import sha256


def frames(path):
    """Yield frame attributes and frameset metadata without retaining the XML tree."""
    fs = None
    stack = []
    for event, elem in ET.iterparse(path, events=('start', 'end')):
        if event == 'start':
            stack.append(elem)
            if elem.tag == 'FrameSet':
                fs = dict(elem.attrib)
        else:
            if elem.tag == 'Frame' and fs is not None:
                yield fs, elem.attrib
            if elem.tag == 'FrameSet':
                fs = None
            if len(stack) > 1:
                stack[-2].remove(elem)
            elem.clear()
            stack.pop()


def target(status, owner):
    if status == '0':
        return 2
    if status == '1' and owner in ('1', '2'):
        return int(owner) - 1
    return 3  # unknown is X, never silently dead or away


def prepare(match, raw, out):
    meta = raw / f'{match}-meta.xml'
    tracking = raw / f'{match}-tracking.xml'
    root = ET.parse(meta).getroot()
    general = root.find('.//General')
    ids = [general.get('HomeTeamId'), general.get('GuestTeamId')]
    if len(set(ids)) != 2 or None in ids:
        raise ValueError('Invalid home/away metadata')
    env = root.find('.//Environment')
    pitch = np.array([float(env.get('PitchX')), float(env.get('PitchY'))], np.float32)
    bounds, objects = {}, {}
    statuses, owners = Counter(), Counter()
    print('Scanning XML schema and period bounds', flush=True)
    for fs, f in frames(tracking):
        period = fs['GameSection']
        if period not in ('firstHalf', 'secondHalf'):
            raise ValueError(f'Unexpected period {period}')
        n = int(f['N'])
        lo, hi = bounds.get(period, (n, n))
        bounds[period] = min(lo, n), max(hi, n)
        team = fs['TeamId']
        if team in ids:
            objects.setdefault(period, {})[fs['PersonId']] = ids.index(team)
        elif team == 'BALL':
            statuses[f.get('BallStatus', 'missing')] += 1
            owners[f.get('BallPossession', 'missing')] += 1
    arrays = {}
    for period, (lo, hi) in bounds.items():
        slots = sorted(objects[period])
        n = hi - lo + 1
        arrays[period] = dict(players=np.full((n, len(slots), 2), np.nan, np.float32),
                              teams=np.array([objects[period][s] for s in slots], np.int8),
                              ball=np.full((n, 2), np.nan, np.float32),
                              labels=np.full(n, 3, np.int8), frame=np.arange(lo, hi + 1),
                              pitch=pitch, slot_ids=np.array(slots))
    lookup = {p: {s: i for i, s in enumerate(a['slot_ids'])} for p, a in arrays.items()}
    seen = {p: set() for p in arrays}
    print('Extracting geometry and separate supervision', flush=True)
    for fs, f in frames(tracking):
        p = fs['GameSection']
        i = int(f['N']) - bounds[p][0]
        a = arrays[p]
        xy = [float(f.get('X', 'nan')), float(f.get('Y', 'nan'))]
        if fs['TeamId'] == 'BALL':
            if i in seen[p]:
                raise ValueError('Duplicate ball frame')
            seen[p].add(i)
            a['ball'][i] = xy
            a['labels'][i] = target(f.get('BallStatus'), f.get('BallPossession'))
        elif fs['PersonId'] in lookup[p]:
            a['players'][i, lookup[p][fs['PersonId']]] = xy
    out.mkdir(parents=True, exist_ok=True)
    report = dict(match=match, home=ids[0], away=ids[1], fps=25,
                  raw_sha256=sha256(tracking), metadata_sha256=sha256(meta),
                  status_counts=dict(statuses), ownership_counts=dict(owners), periods={},
                  semantics='Provider labels are proxy supervision; agreement with competition control semantics is not yet established.')
    for period, a in arrays.items():
        np.savez_compressed(out / f'{match}-{period}.npz', **a)
        report['periods'][period] = dict(frames=len(a['frame']),
            labels={k: int((a['labels'] == v).sum()) for v, k in enumerate('HADX')},
            player_slots=len(a['teams']), ball_frames=len(seen[period]))
    (out / f'{match}-audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--match', required=True)
    p.add_argument('--raw', type=Path, default=Path('data-external'))
    p.add_argument('--out', type=Path, default=Path('data-external/prepared'))
    a = p.parse_args()
    prepare(a.match, a.raw, a.out)
