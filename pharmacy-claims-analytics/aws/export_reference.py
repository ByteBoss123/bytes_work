"""Export the first 1,000 validation-period synthetic episodes (features only) for live validation.

Usage: PYTHONPATH=. python aws/export_reference.py CLAIMS_SQLITE OUT_JSON
"""
import json
import sqlite3
import sys

import pandas as pd

from pbm.pipeline import FEATURES

con = sqlite3.connect(sys.argv[1])
ref = pd.read_sql_query("SELECT * FROM rejected_features WHERE submitted_at >= '2025-05-18' "
                        "AND submitted_at < '2025-06-23'", con)
rec = ref[FEATURES].to_dict(orient='records')[:1000]
for r in rec:
    for k in ['days_supply', 'prior_rejections_90d', 'member_tenure_days', 'weekend']:
        r[k] = int(r[k])
    r['submitted_cost'] = float(r['submitted_cost'])
json.dump({'note': 'Synthetic validation-period episodes (seed 42), features only', 'episodes': rec},
          open(sys.argv[2], 'w'))
