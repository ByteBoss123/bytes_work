"""Exercise a deployed scoring service (e.g. Cloud Run) over HTTPS with synthetic traffic.

Phases: normal -> invalid payloads -> shifted features -> recovery, then an optional
timed soak of normal traffic. Records client-side end-to-end latency (network included)
and the service's own /monitoring snapshots. Writes reports/cloud_run_validation.json.

Scope: synthetic test traffic sent by the author. Not real users, not real claims.

Usage (after `python -m pbm.pipeline`, which creates reports/claims.sqlite):
    python tests/remote_validation.py --url https://SERVICE-URL --soak-minutes 30
Auth: by default runs `gcloud auth print-identity-token` and refreshes it every 30 min.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pbm.pipeline import FEATURES  # noqa: E402


class Client:
    def __init__(self, base, token_cmd):
        self.base = base.rstrip('/')
        self.token_cmd = token_cmd
        self.token, self.token_at = None, 0.0
        self.latencies, self.statuses = [], []

    def headers(self):
        if self.token_cmd and time.time() - self.token_at > 1800:
            self.token = subprocess.run(self.token_cmd, shell=True, check=True, capture_output=True,
                                        text=True).stdout.strip()
            self.token_at = time.time()
        h = {'Content-Type': 'application/json'}
        if self.token:
            h['Authorization'] = 'Bearer ' + self.token
        return h

    def call(self, method, path, payload=None, track=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(self.base + path, data=data, headers=self.headers(), method=method)
        start = time.perf_counter()
        try:
            with urlopen(req, timeout=30) as r:
                status, body = r.status, r.read().decode()
        except HTTPError as e:
            status, body = e.code, e.read().decode()
        except (URLError, TimeoutError, ConnectionError) as e:
            status, body = 0, json.dumps({'error': type(e).__name__})
        if track:
            self.latencies.append((time.perf_counter() - start) * 1000)
            self.statuses.append(status)
        return status, body

    def predict(self, episodes):
        status, body = self.call('POST', '/predict', {'episodes': episodes}, track=True)
        return status

    def monitoring(self):
        status, body = self.call('GET', '/monitoring')
        if status != 200:
            raise RuntimeError(f'/monitoring returned {status}: {body[:200]}')
        return json.loads(body)


def summarize(latencies, statuses):
    lat = np.array(latencies) if latencies else np.array([np.nan])
    return {'requests': len(statuses),
            'http_200': int(sum(s == 200 for s in statuses)),
            'http_4xx': int(sum(400 <= s < 500 for s in statuses)),
            'http_5xx': int(sum(s >= 500 for s in statuses)),
            'connection_errors': int(sum(s == 0 for s in statuses)),
            'client_latency_ms': {'p50': float(np.nanpercentile(lat, 50)),
                                  'p95': float(np.nanpercentile(lat, 95)),
                                  'p99': float(np.nanpercentile(lat, 99))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--token-cmd', default='gcloud auth print-identity-token',
                    help='Shell command printing a bearer token; empty string for no auth')
    ap.add_argument('--soak-minutes', type=float, default=0)
    ap.add_argument('--soak-interval', type=float, default=2.0, help='Seconds between soak requests')
    ap.add_argument('--out', type=Path, default=Path('reports/cloud_run_validation.json'))
    args = ap.parse_args()

    db = Path('reports/claims.sqlite')
    if not db.exists():
        sys.exit('reports/claims.sqlite missing: run `python -m pbm.pipeline` first')
    with sqlite3.connect(db) as con:
        ref = pd.read_sql_query("SELECT * FROM rejected_features WHERE submitted_at >= '2025-05-18' "
                                "AND submitted_at < '2025-06-23'", con)
    records = ref[FEATURES].to_dict(orient='records')
    batch = lambda i: records[(i % 20) * 50:(i % 20 + 1) * 50]  # noqa: E731

    c = Client(args.url, args.token_cmd)
    status, body = c.call('GET', '/health')
    if status != 200:
        sys.exit(f'/health returned {status}: {body[:300]}')
    health = json.loads(body)
    started = datetime.now(timezone.utc).isoformat()
    checks = {}

    # 1. Normal traffic: 20 requests x 50 validation-period episodes.
    for i in range(20):
        assert c.predict(batch(i)) == 200, 'normal request failed'
    normal = c.monitoring()
    checks['normal_no_drift_alert'] = not normal['alerts']['score_drift']

    # 2. Invalid payloads. 10 failures is >5% of any 100-request window, even if the
    #    instance already holds earlier normal traffic.
    for _ in range(10):
        assert c.predict([{'resolved_at': 'future'}]) == 400, 'invalid payload was not rejected'
    invalid = c.monitoring()
    checks['failure_rate_alert_fired'] = invalid['alerts']['request_failure_rate']

    # 3. Shifted features: 20 x 50 = 1,000 scores, filling the whole PSI window.
    stress = [dict(r, reject_reason='authorization_required',
                   prior_rejections_90d=r['prior_rejections_90d'] + 4) for r in records[:50]]
    for _ in range(20):
        assert c.predict(stress) == 200, 'shifted request failed'
    shifted = c.monitoring()
    checks['drift_alert_fired'] = shifted['alerts']['score_drift']

    # 4. Recovery: 100 normal requests flush both windows.
    for i in range(100):
        assert c.predict(batch(i)) == 200, 'recovery request failed'
    recovered = c.monitoring()
    checks['alerts_cleared'] = (not recovered['alerts']['score_drift']
                                and not recovered['alerts']['request_failure_rate'])
    phases = summarize(c.latencies, c.statuses)

    # 5. Optional soak: normal traffic at a fixed interval.
    soak = None
    if args.soak_minutes > 0:
        c.latencies, c.statuses = [], []
        end = time.time() + args.soak_minutes * 60
        i = 0
        while time.time() < end:
            c.predict(batch(i))
            i += 1
            time.sleep(args.soak_interval)
        soak = summarize(c.latencies, c.statuses)
        soak['minutes'] = args.soak_minutes
        soak['interval_seconds'] = args.soak_interval
        soak['success_rate'] = soak['http_200'] / soak['requests'] if soak['requests'] else None
        soak['final_monitoring'] = c.monitoring()

    result = {'test': 'Deployed-service validation with synthetic traffic',
              'url_host': args.url.split('/')[2],
              'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'model_sha256': health['model_sha256'],
              'phase_requests': phases,
              'monitoring': {'normal': normal, 'invalid_payload_stage': invalid,
                             'shifted': shifted, 'recovered': recovered},
              'checks': checks, 'passed': all(checks.values()), 'soak': soak,
              'scope': ('Synthetic traffic sent by the author to a deployed endpoint. Not real users, '
                        'real claims, or production workload. Client latency includes network.')}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'passed': result['passed'], 'checks': checks,
                      'phase_requests': result['phase_requests'],
                      'soak': None if soak is None else {k: soak[k] for k in
                                                         ('requests', 'success_rate', 'client_latency_ms')}},
                     indent=2))
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
