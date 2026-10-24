import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from pbm.pipeline import generate, validate, curate, split, FEATURES, capacity, psi

class PipelineTests(unittest.TestCase):
    def test_seed_reproduces_data(self):
        pd.testing.assert_frame_equal(generate(100, 1), generate(100, 1))

    def test_duplicate_grain_rejected(self):
        d = generate(100)
        with self.assertRaises(ValueError):
            validate(pd.concat([d, d.iloc[:1]]))

    def test_resolution_before_submission_rejected(self):
        d = generate(100)
        d.loc[0, 'resolved_at'] = '2024-01-01'
        with self.assertRaises(ValueError):
            validate(d)

    def test_invalid_categories_rejected(self):
        d = generate(100)
        d.loc[0, 'channel'] = 'unknown'
        with self.assertRaises(ValueError):
            validate(d)

    def test_maturity_and_boundary_labels(self):
        d = generate(5)
        d.initial_status = 'rejected'
        d.reject_reason = 'coverage_check'
        d.submitted_at = ['2025-01-01 00:00:00'] * 4 + ['2025-08-28 00:00:00']
        d.resolved_at = ['2025-01-08 00:00:00', '2025-01-08 01:00:00', None, '2025-01-02 00:00:00', None]
        with tempfile.TemporaryDirectory() as t:
            rows = curate(d, Path(t) / 'test.sqlite')
        self.assertEqual(rows.unresolved_7d.tolist(), [0, 1, 1, 0])

    def test_reruns_do_not_duplicate(self):
        d = generate(1000)
        with tempfile.TemporaryDirectory() as t:
            db = Path(t) / 'test.sqlite'
            a, b = curate(d, db), curate(d, db)
            pd.testing.assert_frame_equal(a, b)

    def test_time_embargo_and_no_feature_leakage(self):
        with tempfile.TemporaryDirectory() as t:
            a, b, c = split(curate(generate(10000), Path(t) / 'test.sqlite'))
        self.assertLess(pd.to_datetime(a.submitted_at).max() + pd.Timedelta(days=7), pd.to_datetime(b.submitted_at).min())
        self.assertLess(pd.to_datetime(b.submitted_at).max() + pd.Timedelta(days=7), pd.to_datetime(c.submitted_at).min())
        self.assertTrue(set(a.claim_id).isdisjoint(c.claim_id))
        self.assertTrue(set(a.member_id).isdisjoint(c.member_id))
        self.assertFalse(set(FEATURES) & {'resolved_at', 'unresolved_7d', 'claim_id', 'member_id', 'submitted_at'})

    def test_nonfinite_features_rejected(self):
        d = generate(100)
        d.loc[0, 'submitted_cost'] = float('inf')
        with self.assertRaises(ValueError):
            validate(d)

    def test_capacity_counts(self):
        m = capacity([1, 0, 1, 0], [.8, .9, .7, .1], .5)
        self.assertEqual((m['reviewed'], m['captured'], m['recall']), (2, 1, .5))

    def test_psi_identical_and_shifted(self):
        a = np.linspace(.01, .49, 100)
        self.assertAlmostEqual(psi(a, a), 0)
        self.assertGreater(psi(a, a + .5), .2)

if __name__ == '__main__':
    unittest.main()
