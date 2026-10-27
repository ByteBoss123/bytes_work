"""Bounded real CMS aggregate-data analysis (provider-drug spending), separate from the PA model."""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

URL='https://data.cms.gov/data-api/v1/dataset/9552739e-3d05-4c1b-8eff-ecabf391e2e5/data?size=5000&offset=0'


def run(out=Path('reports')):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    with urlopen(URL,timeout=60) as r:
        raw=r.read()
    rows=json.loads(raw)
    if not isinstance(rows,list) or not rows:
        raise ValueError('CMS returned no records')
    d=pd.DataFrame(rows)
    grain=['Prscrbr_NPI','Brnd_Name','Gnrc_Name']
    if d[grain].isna().any().any() or d.duplicated(grain).any():
        raise ValueError('Missing or duplicate provider/drug grain')
    for c in ['Tot_Clms','Tot_30day_Fills','Tot_Drug_Cst']:
        d[c]=pd.to_numeric(d[c],errors='raise')
        if d[c].isna().any() or (d[c]<0).any():
            raise ValueError('Invalid totals')
    # Names/NPI are unnecessary for the downstream business summary; do not persist them.
    with sqlite3.connect(':memory:') as con:
        d[['Gnrc_Name','Tot_Clms','Tot_30day_Fills','Tot_Drug_Cst']].to_sql('cms',con,index=False)
        query='''SELECT Gnrc_Name AS generic_name, COUNT(*) AS source_rows,
          SUM(Tot_Clms) AS claims, SUM(Tot_30day_Fills) AS fills_30day,
          ROUND(SUM(Tot_Drug_Cst),2) AS drug_cost,
          SUM(Tot_Drug_Cst)/NULLIF(SUM(Tot_30day_Fills),0) AS cost_per_30day_fill
          FROM cms GROUP BY Gnrc_Name ORDER BY drug_cost DESC'''
        summary=pd.read_sql_query(query,con)
    summary.to_csv(out/'cms_spend_by_generic.csv',index=False)
    result={'source_url':URL,'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
        'raw_response_sha256':hashlib.sha256(raw).hexdigest(),'source_rows':len(d),
        'unique_generic_names':len(summary),'claim_count':int(d.Tot_Clms.sum()),
        'total_drug_cost':float(d.Tot_Drug_Cst.sum()),
        'top10_spend_share':float(summary.head(10).drug_cost.sum()/d.Tot_Drug_Cst.sum()),
        'query':query, 'top10':summary.head(10).to_dict(orient='records'),
        'scope':'First 5000 API rows in source ordering; convenience extract, not random or representative. Not national or Virginia totals.',
        'data_type':'Real public provider-drug aggregates; no patient-level records or rejection outcomes.',
        'business_use':'Identify high-spend drug groups within this extract for analyst review; no savings, prescribing changes, or appropriateness claims.',
        'limitations':'Source suppression and selection affect coverage. Beneficiary counts are not summed. Drug cost is not net PBM cost after rebates. Cost variation is not evidence of inappropriate care. No labels for unresolved claims.'}
    (out/'cms_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k not in ['top10','query']},indent=2))
    return result

if __name__=='__main__':run()
