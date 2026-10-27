"""Exercise actual loopback HTTP requests, drift detection, and alert recovery.

Traffic replays real CMS formulary feature rows (reports/reference_pairs.json, public plan-design data).
The shift phase replays only real rows whose pair required PA last year: a genuine subpopulation
shift, not altered values.
"""
import json
import sqlite3
import tempfile
import threading
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from pbm.service import ScoringService,make_server


def exercise():
    with tempfile.TemporaryDirectory() as tmp:
        service=ScoringService('reports/model.joblib','reports/monitoring_reference.json',Path(tmp)/'telemetry.sqlite')
        server=make_server(service,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        base=f'http://127.0.0.1:{server.server_port}'
        def get(path):
            with urlopen(base+path,timeout=10) as r:return r.read().decode()
        def post(payload):
            request=Request(base+'/predict',json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            try:
                with urlopen(request,timeout=10) as r:return r.status,json.load(r)
            except HTTPError as e:return e.code,json.load(e)
        try:
            assert json.loads(get('/health'))['status']=='ready'
            records=json.loads(Path('reports/reference_pairs.json').read_text())[:1000]
            prior_pa=[r for r in json.loads(Path('reports/reference_pairs.json').read_text()) if r['prev_pair_pa']==1][:50]
            assert len(prior_pa)==50
            for i in range(20):
                status,body=post({'pairs':records[i*50:(i+1)*50]})
                assert status==200 and len(body['probabilities'])==50
            normal=json.loads(get('/monitoring'))
            assert not normal['alerts']['score_drift'] and normal['requests_total']==20
            for _ in range(5):assert post({'pairs':[{'pa_required':1}]})[0]==400
            invalid=json.loads(get('/monitoring'))
            assert invalid['alerts']['request_failure_rate']
            for _ in range(20):assert post({'pairs':prior_pa})[0]==200
            shifted=json.loads(get('/monitoring'))
            assert shifted['alerts']['score_drift']
            for i in range(100):assert post({'pairs':records[(i%20)*50:(i%20+1)*50]})[0]==200
            recovered=json.loads(get('/monitoring'))
            assert not recovered['alerts']['score_drift'] and not recovered['alerts']['request_failure_rate']
            assert 'pbm_requests_total 145' in get('/metrics')
            with sqlite3.connect(service.db) as con:
                transitions=con.execute('SELECT name,active,value FROM alerts ORDER BY id').fetchall()
                assert con.execute('SELECT COUNT(*) FROM scores').fetchone()[0]==7000
            result={'test':'Actual HTTP service integration replaying real CMS formulary feature rows','requests':145,'scored_pairs':7000,
                    'normal':normal,'invalid_payload_stage':invalid,'shifted':shifted,'recovered':recovered,
                    'alert_transitions':transitions,'passed':True,
                    'scope':'Transient loopback deployment. Replayed public plan-design rows; not real-user production traffic.'}
            Path('reports/service_validation.json').write_text(json.dumps(result,indent=2))
            print(json.dumps({'passed':True,'requests':145,'scored_pairs':7000,'drift_detected':True,'failure_alert_detected':True,'recovery_verified':True},indent=2))
        finally:server.shutdown();server.server_close();thread.join()

if __name__=='__main__':exercise()
