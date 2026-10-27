"""Validate the deployed Lambda scoring service by replaying real CMS formulary feature rows, via the Lambda Invoke API.

Sends Function-URL-format (payload v2.0) events, so requests reach the HTTP service through
the Lambda Web Adapter exactly as Function URL calls do, without exposing the URL publicly.

Phases: 20 normal -> 10 invalid -> 20 shifted -> 100 normal (recovery), then an optional soak.
Latency here is client-side Invoke round trip (includes AWS API overhead), not server time.

Usage:
    python aws/validate_lambda.py --soak-minutes 20
Needs AWS credentials with lambda:InvokeFunction and s3:GetObject on the reference file.
Scope: test traffic sent by the author, replaying public plan-design rows. Not real users or production load.
Shift phase: real rows whose pair required PA the prior year (a subpopulation shift, values unaltered).
"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np

REGION = 'us-east-1'
FUNCTION = 'pbm-claims-scoring'
BUCKET = 'pbm-claims-scoring-331604528033'
REFERENCE_KEY = 'validation/reference_pairs.json'


def event(method, path, body=''):
    return {'version': '2.0', 'routeKey': '$default', 'rawPath': path, 'rawQueryString': '',
            'headers': {'content-type': 'application/json', 'host': 'lambda-invoke'},
            'requestContext': {'accountId': 'anonymous', 'apiId': 'invoke', 'domainName': 'lambda-invoke',
                               'domainPrefix': 'invoke', 'requestId': str(uuid.uuid4()), 'routeKey': '$default',
                               'stage': '$default', 'time': '', 'timeEpoch': int(time.time() * 1000),
                               'http': {'method': method, 'path': path, 'protocol': 'HTTP/1.1',
                                        'sourceIp': '127.0.0.1', 'userAgent': 'pbm-validation'}},
            'body': body, 'isBase64Encoded': False}


class Client:
    def __init__(self):
        self.lam = boto3.client('lambda', region_name=REGION)
        self.latencies, self.statuses = [], []

    def call(self, method, path, payload=None, track=False):
        body = json.dumps(payload) if payload is not None else ''
        start = time.perf_counter()
        res = self.lam.invoke(FunctionName=FUNCTION, Payload=json.dumps(event(method, path, body)).encode())
        out = json.loads(res['Payload'].read())
        status = out.get('statusCode', 0) if 'FunctionError' not in res else 0
        if track:
            self.latencies.append((time.perf_counter() - start) * 1000)
            self.statuses.append(status)
        return status, out.get('body', '')

    def predict(self, pairs):
        return self.call('POST', '/predict', {'pairs': pairs}, track=True)[0]

    def monitoring(self):
        status, body = self.call('GET', '/monitoring')
        assert status == 200, f'/monitoring returned {status}'
        return json.loads(body)


def summarize(latencies, statuses):
    lat = np.array(latencies)
    return {'requests': len(statuses), 'http_200': int(sum(s == 200 for s in statuses)),
            'http_4xx': int(sum(400 <= s < 500 for s in statuses)), 'http_5xx': int(sum(s >= 500 for s in statuses)),
            'invoke_errors': int(sum(s == 0 for s in statuses)),
            'invoke_latency_ms': {q: float(np.percentile(lat, int(q[1:]))) for q in ('p50', 'p95', 'p99')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--soak-minutes', type=float, default=0)
    ap.add_argument('--soak-interval', type=float, default=2.0)
    ap.add_argument('--out', type=Path, default=Path('reports/aws_lambda_validation.json'))
    args = ap.parse_args()
    ref = json.loads(boto3.client('s3', region_name=REGION).get_object(Bucket=BUCKET, Key=REFERENCE_KEY)['Body'].read())
    records = ref['pairs'][:1000]
    prior_pa = [r for r in ref['pairs'] if r['prev_pair_pa'] == 1][:50]
    batch = lambda i: records[(i % 20) * 50:(i % 20 + 1) * 50]  # noqa: E731
    c = Client()
    status, body = c.call('GET', '/health')
    assert status == 200, f'/health returned {status}'
    started = datetime.now(timezone.utc).isoformat()
    checks = {}
    for i in range(20):
        assert c.predict(batch(i)) == 200
    normal = c.monitoring()
    checks['normal_no_drift_alert'] = not normal['alerts']['score_drift']
    for _ in range(10):
        assert c.predict([{'pa_required': 1}]) == 400
    invalid = c.monitoring()
    checks['failure_rate_alert_fired'] = invalid['alerts']['request_failure_rate']
    for _ in range(20):
        assert c.predict(prior_pa) == 200
    shifted = c.monitoring()
    checks['drift_alert_fired'] = shifted['alerts']['score_drift']
    for i in range(100):
        assert c.predict(batch(i)) == 200
    recovered = c.monitoring()
    checks['alerts_cleared'] = not recovered['alerts']['score_drift'] and not recovered['alerts']['request_failure_rate']
    phases = summarize(c.latencies, c.statuses)
    soak = None
    if args.soak_minutes > 0:
        c.latencies, c.statuses = [], []
        end, i = time.time() + args.soak_minutes * 60, 0
        while time.time() < end:
            c.predict(batch(i)); i += 1
            time.sleep(args.soak_interval)
        soak = summarize(c.latencies, c.statuses)
        soak.update(minutes=args.soak_minutes, final_monitoring=c.monitoring())
    result = {'test': 'AWS Lambda deployment validation replaying real CMS formulary feature rows (Lambda Invoke, Function URL event format)',
              'function': FUNCTION, 'region': REGION, 'started_utc': started,
              'finished_utc': datetime.now(timezone.utc).isoformat(), 'model_sha256': json.loads(body)['model_sha256'],
              'phase_requests': phases,
              'monitoring': {'normal': normal, 'invalid_payload_stage': invalid, 'shifted': shifted, 'recovered': recovered},
              'checks': checks, 'passed': all(checks.values()), 'soak': soak,
              'scope': 'Test traffic sent by the author replaying public plan-design rows. Not real users or production workload.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({'passed': result['passed'], 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
