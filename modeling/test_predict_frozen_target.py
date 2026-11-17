import unittest
from unittest.mock import patch
import numpy as np
from .predict_frozen_target import assemble, predict_period


class FrozenTargetTests(unittest.TestCase):
    def test_provided_periods_have_explicit_bounds_and_outside_abstention(self):
        got = assemble(9, [(1, 4, np.array([0, 1, 2])), (6, 8, np.array([1, 0]))])
        self.assertEqual(''.join(got), 'XHADXXAHX')
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            assemble(9, [(1, 4, np.zeros(3, int)), (3, 6, np.zeros(3, int))])
        with self.assertRaisesRegex(ValueError, 'bounds'):
            assemble(9, [(1, 4, np.zeros(2, int))])

    def test_blend_is_before_decoding_and_never_depends_on_labels(self):
        candidate = dict(members=[dict(id='a'), dict(id='b')], weights={'a': .25, 'b': .75}, switch_cost=3)
        pa = np.array([[.9, .05, .05], [.1, .1, .8]])
        pb = np.array([[.1, .8, .1], [.2, .3, .5]])
        def predict(spec, loaded, data):
            return pa if spec['id'] == 'a' else pb
        with patch('modeling.predict_frozen_target.predict_member', side_effect=predict), patch(
                'modeling.predict_frozen_target.decode', return_value=np.array([1, 2])) as decoder:
            probability, pred = predict_period(candidate, {'a': None, 'b': None}, {'ball': np.zeros((2, 2))})
        np.testing.assert_allclose(probability, .25*pa+.75*pb)
        np.testing.assert_array_equal(decoder.call_args.args[0], probability)
        self.assertEqual(decoder.call_args.kwargs['switch_cost'], 3)


if __name__ == '__main__':
    unittest.main()
