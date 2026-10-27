"""Logic tests for the formulary pipeline on tiny hand-built fixtures (test inputs only).

The real analysis runs on CMS files; these fixtures only check grain, linkage, and leakage rules.
"""
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from pbm import formulary as F


def form_frame(year, rows):
    return pd.DataFrame([{'FORMULARY_ID': f, 'CONTRACT_YEAR': str(year), 'RXCUI': r, 'NDC': n,
                          'TIER_LEVEL_VALUE': str(t), 'QUANTITY_LIMIT_YN': q, 'PRIOR_AUTHORIZATION_YN': pa,
                          'STEP_THERAPY_YN': st} for f, r, n, t, q, pa, st in rows])


def plan_frame(rows):
    return pd.DataFrame([{'CONTRACT_ID': c, 'PLAN_ID': p, 'SEGMENT_ID': '000', 'FORMULARY_ID': f, 'SNP': snp}
                         for c, p, f, snp in rows])


class Curation(unittest.TestCase):
    def test_pair_grain_takes_any_ndc_flag(self):
        f = form_frame(2025, [('F1', '100', 'a', 2, 'N', 'N', 'N'), ('F1', '100', 'b', 3, 'Y', 'Y', 'N')])
        p = F.pairs(f)
        self.assertEqual(len(p), 1)
        self.assertEqual(int(p.pa[0]), 1)
        self.assertEqual(int(p.tier[0]), 2)

    def test_validation_rejects_bad_flag_and_year(self):
        with self.assertRaises(ValueError):
            F.validate_formulary(form_frame(2025, [('F1', '1', 'a', 1, 'N', 'X', 'N')]), 2025)
        with self.assertRaises(ValueError):
            F.validate_formulary(form_frame(2024, [('F1', '1', 'a', 1, 'N', 'N', 'N')]), 2025)

    def test_link_uses_majority_of_continuing_plans(self):
        cur = F.plan_table(plan_frame([('S1', '001', 'N1', '0'), ('S1', '002', 'N1', '0'), ('S1', '003', 'N1', '0')]))
        prev = F.plan_table(plan_frame([('S1', '001', 'O1', '0'), ('S1', '002', 'O1', '0'), ('S1', '003', 'O2', '0')]))
        link = F.link_formularies(cur, prev)
        self.assertEqual(link.prev_formulary_id.tolist(), ['O1'])

    def test_features_use_prior_year_only_and_keep_grain(self):
        prev = F.pairs(form_frame(2024, [('O1', '1', 'a', 1, 'N', 'Y', 'N'), ('O1', '2', 'b', 2, 'N', 'N', 'N')]))
        cur = F.pairs(form_frame(2025, [('N1', '1', 'a', 1, 'N', 'N', 'N'), ('N1', '3', 'c', 4, 'N', 'Y', 'Y')]))
        pp = F.plan_table(plan_frame([('S1', '001', 'O1', '0')]))
        cp = F.plan_table(plan_frame([('S1', '001', 'N1', '0')]))
        d = F.build(cur, cp, prev, pp).set_index('rxcui')
        self.assertEqual(len(d), 2)
        # Drug 1 carries last year's PA even though this year it dropped: features must not see Y labels.
        self.assertEqual(d.loc['1', 'prev_pair_pa'], 1)
        self.assertEqual(d.loc['1', F.LABEL], 0)
        self.assertEqual(d.loc['3', 'prev_pair_listed'], 0)
        self.assertEqual(d.loc['3', 'drug_listed_prev'], 0)
        self.assertTrue(np.isnan(d.loc['3', 'drug_pa_rate_prev']))
        self.assertFalse(set(F.FEATURES) & {F.LABEL, 'formulary_id', 'rxcui', 'prev_formulary_id'})


class EndToEnd(unittest.TestCase):
    def test_run_on_fixture_zips(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            raw, out = Path(t) / 'raw', Path(t) / 'out'
            raw.mkdir()
            for year in F.SOURCES:
                rows, plans = [], []
                for k in range(12):
                    fid = f'{year}{k:03d}'
                    plans.append((('S' if k % 2 else 'H') + f'{k:04d}', '001', fid, '0'))
                    for r in range(60):
                        pa = 'Y' if (r % 5 == 0) ^ (rng.random() < .1) else 'N'
                        rows.append((fid, str(r), f'n{r}', 1 + r % 5, 'N', pa, 'N'))
                inner = io.BytesIO()
                with zipfile.ZipFile(inner, 'w') as z:
                    z.writestr('basic drugs formulary file  20250101.txt', form_frame(year, rows).to_csv(sep='|', index=False))
                    z.writestr('plan information  20250101.txt', plan_frame(plans).to_csv(sep='|', index=False))
                with zipfile.ZipFile(raw / f'spuf_{year}.zip', 'w') as z:
                    z.writestr('nested.zip', inner.getvalue())
            r = F.run(out, raw, seed=1, boot_reps=20)
            self.assertEqual(r['test']['rows'], 12 * 60)
            self.assertEqual(r['formulary_link_rate_test'], 1.0)
            ref = json.loads((out / 'monitoring_reference.json').read_text())
            self.assertEqual(ref['features'], F.FEATURES)
            self.assertTrue((out / 'reference_pairs.json').exists())


if __name__ == '__main__':
    unittest.main()
