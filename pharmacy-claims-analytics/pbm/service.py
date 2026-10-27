"""HTTP inference with real request telemetry and persisted alerts.

Binds to loopback by default. Container mode (--host 0.0.0.0) is intended only
behind an authenticated ingress such as Cloud Run with IAM invocation; see
deploy/README.md. Demonstration service, not a hardened public production server.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import joblib
import numpy as np
import pandas as pd
from pbm.monitoring import psi


def log_event(event, severity, **fields):
    """One JSON line on stdout. Cloud Logging parses it into jsonPayload with this severity.

    Never includes request payloads or identifiers.
    """
    print(json.dumps({'severity': severity, 'event': event, **fields}), flush=True)


class ScoringService:
    def __init__(self, model_path, reference_path, telemetry_path):
        self.model = joblib.load(model_path)  # Load only locally generated trusted artifacts.
        self.model_sha = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        reference = json.loads(Path(reference_path).read_text())
        if reference['model_sha256'] != self.model_sha:
            raise ValueError('Reference/model version mismatch')
        self.reference = np.array(reference['validation_scores'])
        self.features = list(reference['features'])
        self.label = reference['label']
        self.db = str(telemetry_path)
        self.lock = threading.Lock()
        self.started = time.time()
        with sqlite3.connect(self.db) as con:
            con.executescript('''CREATE TABLE IF NOT EXISTS requests (
              id INTEGER PRIMARY KEY, timestamp REAL, status INTEGER, latency_ms REAL,
              model_sha TEXT);
              CREATE TABLE IF NOT EXISTS scores (
              id INTEGER PRIMARY KEY, request_id INTEGER, probability REAL, model_sha TEXT);
              CREATE TABLE IF NOT EXISTS alerts (
              id INTEGER PRIMARY KEY, timestamp REAL, name TEXT, active INTEGER,
              value REAL, model_sha TEXT);''')

    BINARY = {'prev_pair_listed', 'prev_pair_pa', 'prev_pair_st', 'prev_pair_ql', 'drug_listed_prev',
              'form_linked', 'any_snp'}
    RATES = {'drug_pa_rate_prev', 'drug_st_rate_prev', 'drug_ql_rate_prev', 'drug_breadth_prev',
             'form_pa_share_prev', 'is_pdp'}

    def predict(self, body):
        if not isinstance(body, dict) or set(body) != {'pairs'}:
            raise ValueError('Expected pairs array only')
        rows = body['pairs']
        if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
            raise ValueError('Batch must contain 1 to 100 pairs')
        expected = set(self.features)
        if any(not isinstance(row, dict) or set(row) != expected for row in rows):
            raise ValueError('Send prior-year features only; no identifiers, labels, or extra fields')
        for row in rows:
            for k, v in row.items():
                if v is None:
                    continue  # prior-year history can be absent; the model handles missing values
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                    raise ValueError('Features must be finite nonnegative numbers or null')
                if k in self.BINARY and v not in (0, 1):
                    raise ValueError('Indicator out of range: ' + k)
                if k in self.RATES and v > 1:
                    raise ValueError('Rate out of range: ' + k)
        df = pd.DataFrame(rows, columns=self.features).astype(float)
        return self.model.predict_proba(df)[:, 1].tolist()

    def record(self, status, latency, scores):
        with self.lock, sqlite3.connect(self.db) as con:
            cursor = con.execute('INSERT INTO requests(timestamp,status,latency_ms,model_sha) VALUES(?,?,?,?)',
                                 (time.time(), status, latency, self.model_sha))
            con.executemany('INSERT INTO scores(request_id,probability,model_sha) VALUES(?,?,?)',
                            [(cursor.lastrowid, p, self.model_sha) for p in scores])
        self.monitor()

    def monitor(self):
        with self.lock, sqlite3.connect(self.db) as con:
            rows = con.execute('SELECT status,latency_ms FROM requests WHERE model_sha=? ORDER BY id DESC LIMIT 100', (self.model_sha,)).fetchall()
            p = [x[0] for x in con.execute('SELECT probability FROM scores WHERE model_sha=? ORDER BY id DESC LIMIT 1000', (self.model_sha,)).fetchall()]
            total = con.execute('SELECT COUNT(*) FROM requests WHERE model_sha=?', (self.model_sha,)).fetchone()[0]
            value = psi(self.reference, p) if len(p) >= 200 else None
            error_rate = sum(s >= 400 for s, _ in rows) / len(rows) if rows else None
            p95 = float(np.quantile([ms for _, ms in rows], .95)) if rows else None
            alerts = {'score_drift': (value is not None and value > .20, value),
                      'request_failure_rate': (len(rows) >= 20 and error_rate > .05, error_rate),
                      'latency': (len(rows) >= 20 and p95 > 2000, p95)}
            for name, (active, metric) in alerts.items():
                previous = con.execute('SELECT active FROM alerts WHERE name=? AND model_sha=? ORDER BY id DESC LIMIT 1', (name,self.model_sha)).fetchone()
                if previous is None or bool(previous[0]) != bool(active):
                    con.execute('INSERT INTO alerts(timestamp,name,active,value,model_sha) VALUES(?,?,?,?,?)',
                                (time.time(), name, int(active), metric, self.model_sha))
                    log_event('pbm_alert_transition', 'WARNING' if active else 'INFO', alert=name,
                              active=bool(active), value=metric, model_sha256=self.model_sha)
            return {'deployment':'HTTP scoring service; test traffic replays real CMS formulary feature rows', 'model_sha256':self.model_sha,
                    'requests_total':total, 'request_window':len(rows), 'score_window':len(p),
                    'request_failure_rate':error_rate, 'latency_p95_ms':p95, 'score_psi':value,
                    'alerts':{k:bool(v[0]) for k,v in alerts.items()},
                    'thresholds':{'psi':.20, 'failure_rate':.05, 'latency_ms':2000},
                    'minimum_samples':{'requests':20,'scores':200},
                    'performance_monitoring':'Not live: labels arrive with the next contract-year formulary file',
                    'alert_delivery':'Persisted SQLite transitions and HTTP endpoint; no external paging'}


def make_server(service, port=8765, host='127.0.0.1'):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # Do not log payloads or identifiers.
        def send(self, code, body, content_type='application/json'):
            payload = body.encode() if isinstance(body, str) else json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        def do_GET(self):
            path = urlsplit(self.path).path  # ignore query strings (Lambda URLs send '?')
            if path == '/health':
                return self.send(200, {'status':'ready', 'model_sha256':service.model_sha})
            if path == '/monitoring':
                return self.send(200,service.monitor())
            if path == '/metrics':
                m=service.monitor()
                lines=['# TYPE pbm_requests_total counter',f'pbm_requests_total {m["requests_total"]}']
                for key in ['request_failure_rate','latency_p95_ms','score_psi']:
                    if m[key] is not None:
                        lines += [f'# TYPE pbm_{key} gauge',f'pbm_{key} {m[key]}']
                for name,active in m['alerts'].items():
                    lines.append(f'pbm_alert_active{{name="{name}"}} {int(active)}')
                return self.send(200,'\n'.join(lines)+'\n','text/plain; version=0.0.4')
            return self.send(404,{'error':'not found'})
        def do_POST(self):
            if urlsplit(self.path).path != '/predict':
                return self.send(404, {'error':'not found'})
            start=time.perf_counter(); status=200; scores=[]
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0 < size <= 100000:
                    raise ValueError('Body length must be 1–100000 bytes')
                self.connection.settimeout(5)
                body=json.loads(self.rfile.read(size))
                scores=service.predict(body)
                response={'probabilities':scores,'model_sha256':service.model_sha,'label':service.label}
            except (ValueError, TypeError, KeyError, TimeoutError):
                status=400; response={'error':'Invalid payload; consult SERVICE.md schema'}
            except Exception:
                status=500; response={'error':'internal scoring failure'}
            service.record(status,(time.perf_counter()-start)*1000,scores)
            self.send(status,response)
    return ThreadingHTTPServer((host,port),Handler)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--reports',type=Path,default=Path('reports'))
    ap.add_argument('--host',default='127.0.0.1',help='Use 0.0.0.0 only behind authenticated ingress')
    ap.add_argument('--port',type=int,default=int(os.environ.get('PORT','8765')))
    ap.add_argument('--telemetry',type=Path,default=None,help='SQLite path (default: <reports>/telemetry.sqlite)')
    args=ap.parse_args()
    s=ScoringService(args.reports/'model.joblib',args.reports/'monitoring_reference.json',
                     args.telemetry or args.reports/'telemetry.sqlite')
    server=make_server(s,args.port,args.host)
    log_event('pbm_service_started','INFO',host=args.host,port=server.server_port,model_sha256=s.model_sha)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
