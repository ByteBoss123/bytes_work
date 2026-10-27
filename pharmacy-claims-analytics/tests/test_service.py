import json
import tempfile
import unittest
from pathlib import Path
from pbm.service import ScoringService


@unittest.skipUnless(Path('reports/model.joblib').exists(), 'Run pbm.formulary to create the trusted local model')
class ServiceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=Path(self.tmp.name)/'telemetry.sqlite'
        self.s=ScoringService('reports/model.joblib','reports/monitoring_reference.json',self.db)
        self.row=json.loads(Path('reports/reference_pairs.json').read_text())[0]
    def tearDown(self):self.tmp.cleanup()
    def test_scores_real_reference_row(self):
        p=self.s.predict({'pairs':[self.row]})
        self.assertEqual(len(p),1);self.assertTrue(0<=p[0]<=1)
    def test_missing_history_allowed(self):
        self.assertEqual(len(self.s.predict({'pairs':[dict(self.row,prev_pair_pa=None,prev_pair_tier=None)]})),1)
    def test_reject_label_or_identifier(self):
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,pa_required=1)]})
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,formulary_id='00026001')]})
    def test_reject_out_of_range(self):
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,drug_pa_rate_prev=1.5)]})
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,prev_pair_pa=2)]})
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,form_size_prev=float('inf'))]})
        with self.assertRaises(ValueError):self.s.predict({'pairs':[dict(self.row,is_pdp=True)]})
    def test_reject_oversized_batch(self):
        with self.assertRaises(ValueError):self.s.predict({'pairs':[self.row]*101})
    def test_monitor_warmup_and_restart_persistence(self):
        self.s.record(200,12,[.3])
        s2=ScoringService('reports/model.joblib','reports/monitoring_reference.json',self.db)
        m=s2.monitor()
        self.assertEqual(m['requests_total'],1)
        self.assertIsNone(m['score_psi'])
        self.assertFalse(m['alerts']['score_drift'])
    def test_alert_transition_emits_structured_log(self):
        import contextlib
        import io
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf):
            for _ in range(20):self.s.record(400,5,[])
        events=[json.loads(line) for line in buf.getvalue().splitlines()]
        raised=[e for e in events if e['alert']=='request_failure_rate' and e['active']]
        self.assertEqual(len(raised),1)
        self.assertEqual(raised[0]['severity'],'WARNING')
        self.assertEqual(raised[0]['event'],'pbm_alert_transition')
        self.assertNotIn('pairs',buf.getvalue())
    def test_routes_ignore_query_string(self):
        # Lambda Function URLs (via the Web Adapter) forward '/health?' even with no query.
        import threading
        from urllib.request import Request, urlopen
        from pbm.service import make_server
        server=make_server(self.s,0)
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
        base=f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base+'/health?',timeout=10) as r:self.assertEqual(r.status,200)
            req=Request(base+'/predict?x=1',json.dumps({'pairs':[self.row]}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=10) as r:self.assertEqual(r.status,200)
        finally:server.shutdown();server.server_close();t.join()
    def test_reference_version_mismatch(self):
        p=Path(self.tmp.name)/'wrong.json'
        p.write_text(json.dumps({'model_sha256':'wrong','validation_scores':[.5],'features':['x'],'label':'x'}))
        with self.assertRaises(ValueError):ScoringService('reports/model.joblib',p,self.db)

if __name__=='__main__':unittest.main()
