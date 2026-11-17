import itertools
import tempfile
import unittest
from pathlib import Path
import numpy as np
from .common import decode, evaluate, features
from .prepare import target, prepare


def sample(n=300):
    rng = np.random.default_rng(4)
    return {'players': rng.normal(size=(n, 6, 2)).astype('float32'),
            'ball': rng.normal(size=(n, 2)).astype('float32'),
            'teams': np.array([0, 0, 0, 1, 1, 1]), 'pitch': np.array([105, 68]),
            'labels': np.zeros(n, np.int8)}


class CoreTests(unittest.TestCase):
    def test_atomic_report_preserves_previous_on_serialization_failure(self):
        import json
        from .common import write_json
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'report.json'
            write_json(path, {'complete': True})
            with self.assertRaises(ValueError):
                write_json(path, {'bad': float('nan')})
            self.assertEqual(json.loads(path.read_text()), {'complete': True})
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_transfer_corruptions_preserve_labels_and_spatial_identity(self):
        from .transfer_benchmark import variant
        d = sample(600)
        pattern = np.array([True]*20 + [False]*30)
        replay = variant(d, 'target_ball_gaps', pattern)
        iid = variant(d, 'iid_same_missing_count', pattern)
        self.assertEqual(np.isnan(replay['ball']).all(-1).sum(), np.isnan(iid['ball']).all(-1).sum())
        np.testing.assert_array_equal(replay['labels'], d['labels'])
        shuffled = variant(d, 'identity_shuffle', pattern)
        for team in (0, 1):
            ids = d['teams'] == team
            np.testing.assert_array_equal(np.sort(shuffled['players'][:, ids, 0], axis=1), np.sort(d['players'][:, ids, 0], axis=1))
        duplicate = variant(d, 'duplicate_bursts', pattern)
        self.assertEqual(duplicate['players'].shape[1], d['players'].shape[1]+2)
        np.testing.assert_array_equal(duplicate['labels'], d['labels'])

    def test_unlabelled_tracking_profile(self):
        from .profile_target import run_lengths, profile
        np.testing.assert_array_equal(run_lengths([True, True, False, True]), [2, 1])
        d = sample(300); del d['labels']
        d['players'][:] = 0
        d['ball'][25:75] = np.nan
        result = profile(d)
        self.assertAlmostEqual(result['ball_missing_fraction'], 50/300)
        self.assertEqual(result['ball_gap_seconds']['p50'], 2)
        self.assertEqual(result['speed_over_15_mps_fraction'], 0)
        self.assertEqual(result['observed_players_by_team']['0']['p50'], 3)

    def test_reliability_duration_calibration_and_stress(self):
        from .reliability import calibration, duration_recall, stress, brief
        y = np.array([0]*25 + [1]*100 + [2]*25 + [0]*300, np.int8)
        prob = np.eye(3)[y]
        self.assertEqual(calibration(y, prob)['ece'], 0)
        self.assertEqual(calibration(y, prob)['nll'], 0)
        for row in duration_recall(y, y).values():
            self.assertEqual(row['recall_at_iou_05'], 1)
        self.assertEqual(brief(y, prob, lengths=[150, 300])['spell_f1_mean'], 1)
        d = sample(600)
        original = d['ball'].copy()
        changed = stress(d, 'gap_25')
        self.assertTrue(np.isnan(changed['ball'][250:275]).all())
        np.testing.assert_array_equal(d['ball'], original)
        np.testing.assert_array_equal(changed['labels'], d['labels'])

    def test_unknown_labels_not_dead_or_away(self):
        self.assertEqual(target('1', '0'), 3)
        self.assertEqual(target(None, '1'), 3)
        self.assertEqual(target('0', '0'), 2)
        self.assertEqual(target('1', '2'), 1)

    def test_features_cannot_read_labels(self):
        d = sample()
        before, _ = features(d)
        del d['labels']
        after, _ = features(d)
        np.testing.assert_equal(before, after)

    def test_viterbi_matches_exhaustive_search(self):
        prob = np.array([[.8, .1, .1], [.2, .7, .1], [.6, .3, .1], [.1, .8, .1]])
        penalty = .7
        def score(path):
            return sum(np.log(prob[i, c]) for i, c in enumerate(path)) - penalty*sum(a != b for a, b in zip(path, path[1:]))
        expected = max(itertools.product(range(3), repeat=4), key=score)
        np.testing.assert_array_equal(decode(prob, penalty), expected)
        boundary = np.array([0, .1, .75, .3])
        def boundary_score(path):
            return sum(np.log(prob[i, c]) for i, c in enumerate(path)) - sum(
                penalty*(1-boundary[t]) for t in range(1, len(path)) if path[t] != path[t-1])
        expected = max(itertools.product(range(3), repeat=4), key=boundary_score)
        np.testing.assert_array_equal(decode(prob, penalty, boundary), expected)

    def test_metrics_boundaries_and_ungraded_separator(self):
        y = np.array([0]*30 + [1]*20 + [2]*10 + [0]*20, np.int8)
        self.assertEqual(evaluate(y, y)['spell_f1_mean'], 1)
        self.assertLess(evaluate(y, np.roll(y, 10))['spell_f1_mean'], 1)
        separated = np.array([0]*20 + [3] + [0]*20, np.int8)
        self.assertEqual(evaluate(separated, separated)['n_gt_spells'], 2)

    def test_adapter_reorders_frames_preserves_missing_and_teams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'test-meta.xml').write_text('<Root><General HomeTeamId="h" GuestTeamId="a"/><Environment PitchX="105" PitchY="68"/></Root>')
            (root/'test-tracking.xml').write_text('''<Root>
            <FrameSet GameSection="firstHalf" TeamId="h" PersonId="p"><Frame N="11" X="2" Y="3"/><Frame N="10" X="1" Y="2"/></FrameSet>
            <FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="b"><Frame N="10" X="0" Y="0" BallStatus="1" BallPossession="0"/><Frame N="12" X="0" Y="0" BallStatus="0" BallPossession="1"/></FrameSet></Root>''')
            prepare('test', root, root/'out')
            with np.load(root/'out/test-firstHalf.npz') as z:
                np.testing.assert_array_equal(z['frame'], [10, 11, 12])
                np.testing.assert_array_equal(z['labels'], [3, 3, 2])
                self.assertTrue(np.isnan(z['ball'][1]).all())
                self.assertEqual(z['players'][0, 0, 0], 1)


class NeuralTests(unittest.TestCase):
    def test_mixed_sampler_balances_and_interleaves_without_crossing_periods(self):
        from collections import Counter
        from .train_neural import training_windows
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for period in range(2):
                d = sample(600); d['period'] = np.array(period)
                path = Path(tmp)/f'{period}.npz'; np.savez(path, **d); files.append(path)
            def sequence():
                result = []
                for d, lo in training_windows(files, np.random.default_rng(4), 4, 128, 0, 'mixed'):
                    result.append((int(d['period']), bool(np.isnan(d['ball']).any()), lo))
                    self.assertGreaterEqual(lo, 63)
                    self.assertLessEqual(lo+128+63, 600)
                    np.testing.assert_array_equal(d['labels'], np.zeros(600, np.int8))
                return result
            draws = sequence()
            self.assertEqual(draws, sequence())
            counts = Counter((period, noisy) for period, noisy, _ in draws)
            self.assertEqual(counts, {(p, c): 2 for p in range(2) for c in (False, True)})
            self.assertNotEqual([(p, c) for p, c, _ in draws], sorted((p, c) for p, c, _ in draws))

    def test_team_swap_preserves_dead_unknown_and_boundaries(self):
        import torch
        from .neural import swap_team_labels
        y = torch.tensor([0, 0, 1, 2, 3, -1])
        swapped = swap_team_labels(y)
        self.assertEqual(swapped.tolist(), [1, 1, 0, 2, 3, -1])
        self.assertTrue(torch.equal(swap_team_labels(swapped), y))
        self.assertTrue(torch.equal(y[1:] != y[:-1], swapped[1:] != swapped[:-1]))

    def test_training_motion_cache_is_built_after_corruption(self):
        from .train_neural import training_windows
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'period.npz'
            np.savez(path, **sample(600))
            for d, _ in training_windows([path], np.random.default_rng(4), 4, 128, 0, 'mixed', motion=True):
                missing = ~np.isfinite(d['ball']).all(-1)
                self.assertEqual(bool((d['motion_ball'][:, 4] > 0).any()), bool(missing.any()))
                self.assertFalse(d['motion_ball'][missing, 2].any())

    def test_early_stopping_and_resume(self):
        from .development_run import EarlyStopping
        stop = EarlyStopping(patience=2, min_delta=.01)
        self.assertEqual(stop.update(.3), (True, False))
        self.assertEqual(stop.update(.305), (False, False))
        resumed = EarlyStopping(patience=2, min_delta=.01, best=stop.best, stale=stop.stale)
        self.assertEqual(resumed.update(.29), (False, True))
        self.assertEqual(resumed.update(.32), (True, False))
        with self.assertRaises(ValueError):
            resumed.update(float('nan'))

    def test_permutation_and_chunked_inference(self):
        for architecture in ('set-tcn-v1', 'set-tcn-geometry-v2', 'set-tcn-motion-v3', 'set-tcn-motion-robust-v4'):
            with self.subTest(architecture=architecture):
                self.check_inference(architecture)

    def check_inference(self, architecture):
        import torch
        from .neural import PossessionNet, tensors, predict
        torch.set_num_threads(2); torch.manual_seed(4)
        model = PossessionNet(architecture=architecture).eval()
        d = sample()
        if architecture in ('set-tcn-motion-v3', 'set-tcn-motion-robust-v4'):
            from .motion import augment_motion
            full_data = augment_motion(d, velocity_lag=5 if architecture == 'set-tcn-motion-robust-v4' else 1)
        else:
            full_data = d
        with torch.no_grad():
            full = model(*[t.unsqueeze(0) for t in tensors(full_data)])[0][0].softmax(-1).numpy()
        chunked, _ = predict(model, d, chunk=60)
        np.testing.assert_allclose(chunked, full, atol=1e-6)
        order = np.array([5, 2, 0, 3, 1, 4])
        d['players'] = d['players'][:, order]; d['teams'] = d['teams'][order]
        permuted, _ = predict(model, d)
        np.testing.assert_allclose(permuted, full, atol=1e-6)

    def test_geometry_missing_ball_and_players(self):
        import torch
        from .neural import PossessionNet, tensors
        model = PossessionNet(architecture='set-tcn-geometry-v2').eval()
        d = sample(10)
        d['players'][:] = np.nan
        d['ball'][:] = np.nan
        del d['labels']
        with torch.no_grad():
            logits, boundary = model(*[t.unsqueeze(0) for t in tensors(d)])
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(boundary).all())


if __name__ == '__main__':
    unittest.main()
