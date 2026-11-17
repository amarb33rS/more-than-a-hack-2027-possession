"""Aggregate unlabelled target tracking quality; never opens possession labels."""
import argparse
from pathlib import Path
import numpy as np
from .predict_target import read_target
from .common import write_json


def run_lengths(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)


def summary(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0}
    return dict(count=int(len(values)), mean=float(values.mean()),
                **{f'p{q}': float(np.percentile(values, q)) for q in (10, 50, 90, 95, 99)},
                maximum=float(values.max()))


def profile(d):
    ball_seen = np.isfinite(d['ball']).all(-1)
    player_seen = np.isfinite(d['players']).all(-1)
    gaps = run_lengths(~ball_seen)
    edges = np.diff(np.r_[False, ~ball_seen, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    interior_gaps = (stops-starts)[(starts > 0) & (stops < len(ball_seen))]
    ball_hist = np.histogram(gaps, bins=[0, 25, 75, 125, 250, 500, 1500, np.inf])[0]
    lifetimes = np.concatenate([run_lengths(player_seen[:, i]) for i in range(player_seen.shape[1])])
    delta = np.linalg.norm(np.diff(d['players'], axis=0), axis=-1)*25
    good = np.isfinite(delta)
    counts, near_coincident = {}, {}
    for team in (0, 1):
        member = d['teams'] == team
        counts[str(team)] = summary(player_seen[:, member].sum(1))
        pair_counts = []
        for p in d['players'][::25, member]:
            p = p[np.isfinite(p).all(-1)]
            distance = np.linalg.norm(p[:, None]-p[None, :], axis=-1)
            pair_counts.append(int(np.triu(distance < .5, k=1).sum()))
        near_coincident[str(team)] = dict(sampled_frames=len(pair_counts),
            frames_with_pair=int(np.count_nonzero(pair_counts)), pairs=summary(pair_counts))
    return dict(frames=len(ball_seen), player_slots=len(d['teams']),
        ball_missing_fraction=float((~ball_seen).mean()),
        ball_gap_seconds=summary(gaps/25), interior_ball_gap_seconds=summary(interior_gaps/25), ball_gap_histogram=dict(
            frame_bins=['0-24', '25-74', '75-124', '125-249', '250-499', '500-1499', '1500+'], counts=ball_hist.tolist()),
        visible_slot_run_seconds=summary(lifetimes/25), observed_players_by_team=counts,
        consecutive_observation_speed_mps=summary(delta[good]),
        speed_over_15_mps_fraction=float((delta[good] > 15).mean()) if good.any() else None,
        same_team_pairs_within_05m_at_1hz=near_coincident)


def main(a):
    _, periods = read_target(a.repo)
    report = {'source': 'Provided unlabelled tactical tracking and metadata only',
        'fps': 25, 'periods': {},
        'limitations': 'Slot runs are observation continuity, not verified person identities. Near-coincident pairs are candidates, not proof of duplicate tracks. Gaps touching period edges may be censored. No possession labels or video annotations read.'}
    for i, (lo, hi, d) in enumerate(periods, 1):
        report['periods'][str(i)] = dict(video_start=lo, video_stop=hi, **profile(d))
    write_json(a.out, report)
    for half, value in report['periods'].items():
        print('Period', half, 'ball missing', value['ball_missing_fraction'],
              'gap seconds', value['ball_gap_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('hackathon-windows-transfer/more-than-a-hack-2027'))
    p.add_argument('--out', type=Path, default=Path('model-runs/target-tracking-profile.json'))
    main(p.parse_args())
