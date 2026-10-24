"""Reproducible descriptive workflow analysis of real de-identified MIMIC demo data."""
from pathlib import Path
import hashlib
import json
import sqlite3
from urllib.request import urlopen
import pandas as pd

BASE='https://physionet.org/files/mimic-iv-demo/2.2/'

def run(folder=Path('data/mimic'),out=Path('reports')):
    folder=Path(folder);out=Path(out);folder.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
    tables={};provenance={}
    for name in ['prescriptions','pharmacy','admissions']:
        p=folder/(name+'.csv.gz');url=BASE+'hosp/'+p.name
        if not p.exists():
            with urlopen(url,timeout=60) as r:p.write_bytes(r.read())
        provenance[name]={'source':url,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
        tables[name]=pd.read_csv(p,low_memory=False)
    rx,ph,ad=[tables[x] for x in ['prescriptions','pharmacy','admissions']]
    key=['subject_id','hadm_id','pharmacy_id']
    if ph[key].isna().any().any() or ph.duplicated(key).any():raise ValueError('Nonunique pharmacy key')
    if ad[['subject_id','hadm_id']].isna().any().any() or ad.duplicated(['subject_id','hadm_id']).any():raise ValueError('Nonunique admission key')
    # Prescription rows may represent components; do not deduplicate on pharmacy_id.
    with sqlite3.connect(':memory:') as con:
        rx.to_sql('rx',con,index=False);ph.to_sql('ph',con,index=False);ad[['subject_id','hadm_id']].to_sql('ad',con,index=False)
        sql='''SELECT rx.*, CASE WHEN ph.pharmacy_id IS NULL THEN 0 ELSE 1 END AS pharmacy_match,
            CASE WHEN ad.hadm_id IS NULL THEN 0 ELSE 1 END AS admission_match
            FROM rx LEFT JOIN ph ON rx.subject_id=ph.subject_id AND rx.hadm_id=ph.hadm_id AND rx.pharmacy_id=ph.pharmacy_id
            LEFT JOIN ad ON rx.subject_id=ad.subject_id AND rx.hadm_id=ad.hadm_id'''
        linked=pd.read_sql_query(sql,con)
    assert len(linked)==len(rx), 'Join inflated prescription grain'
    missing={c:int((rx[c].isna() | rx[c].fillna('').astype(str).str.strip().eq('')).sum()) for c in ['drug','route','dose_val_rx','dose_unit_rx']}
    start=pd.to_datetime(rx.starttime,errors='coerce');stop=pd.to_datetime(rx.stoptime,errors='coerce')
    entered=pd.to_datetime(ph.entertime,errors='coerce');verified=pd.to_datetime(ph.verifiedtime,errors='coerce')
    hours=(verified-entered).dt.total_seconds()/3600
    valid=hours[hours.notna() & (hours>=0)]
    counts=rx.groupby(['subject_id','hadm_id']).size()
    result={'dataset':'MIMIC-IV Clinical Database Demo v2.2','data_type':'Real de-identified patient-level EHR, open demo; not PBM claims',
        'source_doi':'https://doi.org/10.13026/dp1f-ex47','license':'ODbL-1.0',
        'prescription_rows':len(rx),'pharmacy_rows':len(ph),'admission_rows':len(ad),'patients':int(rx.subject_id.nunique()),
        'admissions_with_prescriptions':len(counts),'unique_pharmacy_orders_in_prescriptions':int(rx.pharmacy_id.nunique()),
        'pharmacy_linked_rows':int(linked.pharmacy_match.sum()),'admission_linked_rows':int(linked.admission_match.sum()),
        'join_preserved_row_count':len(linked)==len(rx),'exact_duplicate_prescription_rows':int(rx.duplicated().sum()),
        'missing_fields':missing,'stop_precedes_start_rows':int((stop<start).sum()),
        'prescription_rows_per_admission_quantiles':{str(q):float(counts.quantile(q)) for q in [.25,.5,.75,.9]},
        'verification_hours':{'denominator_all_pharmacy_rows':len(ph),'missing_pairs':int(hours.isna().sum()),
                              'negative_intervals_flagged':int((hours<0).sum()),'nonnegative_pairs':len(valid),
                              'median_nonnegative':float(valid.median()),'p90_nonnegative':float(valid.quantile(.9))},
        'business_use':'Assess medication-order reconciliation workload, join coverage and documentation completeness before operational reporting.',
        'limitations':['No clinical recommendations, intervention outcomes, costs or rejection labels.',
                      'Missing or negative fields are data-quality signals, not proof of clinical errors.',
                      'Prescription components are not independent prescriptions; row count is not unique order count.',
                      'Dates are shifted per patient; do not infer calendar trends or compare absolute dates across patients.',
                      '100-patient demo is small and not representative of PBM members.'],
        'provenance':provenance,'sql':sql}
    (out/'mimic_workflow.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','sql','limitations']},indent=2))
    return result

if __name__=='__main__':run()
