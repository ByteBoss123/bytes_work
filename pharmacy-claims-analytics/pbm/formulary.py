"""Real-data prior-authorization forecasting from CMS Part D formulary public use files.

Question: for each (formulary, drug) pair a Part D plan lists in contract year Y, which pairs
will require prior authorization (PA)? PA is a utilization-management rule that a pharmacy
claim must satisfy before it pays, so knowing next year's PA list ahead of time helps an
operations team plan review capacity and member outreach.

Every feature comes from the PRIOR contract year's public file (Y-1), plus structural plan
type. Labels come from year Y. Training: CY2024 features -> CY2025 labels. Test: CY2025
features -> CY2026 labels, evaluated once after model selection on a formulary-grouped
validation split of the training year.

Source: CMS "Quarterly Prescription Drug Plan Formulary, Pharmacy Network, and Pricing
Information" (data.cms.gov). These are plan-design records, not claims or patient data.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from pbm.monitoring import psi

ROOT = Path(__file__).resolve().parents[1]

# Exact public files (data.cms.gov, listed in the catalog.data.gov dataset record).
SOURCES = {
    2024: 'https://data.cms.gov/sites/default/files/2025-05/e015bae0-4b23-455d-8243-65aef830ab3c/SPUF_2024_20250430_2024Q3.zip',
    2025: 'https://data.cms.gov/sites/default/files/2025-10/6b0620f5-9259-4f1f-929c-70feb8857887/SPUF_2025_20251009.zip',
    2026: 'https://data.cms.gov/sites/default/files/2026-01/5942aa7e-a0c4-4e65-bd56-32608c33649f/SPUF_2026_20260107.zip',
}

FEATURES = [
    'prev_pair_listed', 'prev_pair_pa', 'prev_pair_st', 'prev_pair_ql', 'prev_pair_tier',
    'drug_listed_prev', 'drug_pa_rate_prev', 'drug_st_rate_prev', 'drug_ql_rate_prev',
    'drug_breadth_prev', 'drug_mean_tier_prev',
    'form_linked', 'form_pa_share_prev', 'form_size_prev',
    'is_pdp', 'any_snp',
]
LABEL = 'pa_required'


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- download and read

def download(year, raw):
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f'spuf_{year}.zip'
    if not path.exists():
        log('downloading', SOURCES[year])
        req = urllib.request.Request(SOURCES[year], headers={'User-Agent': 'pbm-portfolio-research'})
        with urllib.request.urlopen(req, timeout=600) as r, open(path, 'wb') as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return path, h.hexdigest()


WANTED = [('basic', 'formulary'), ('plan', 'information')]


def wanted(name):
    n = name.lower().replace('_', ' ')
    return any(all(k in n for k in ks) for ks in WANTED)


def extract(zpath, dest):
    """Extract only the needed files, descending into nested zips via temporary files on disk."""
    dest.mkdir(parents=True, exist_ok=True)
    done = dest / '.extracted'
    if not done.exists():
        listing = []
        def walk(zf, label):
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = Path(info.filename).name  # flatten; never trust archive paths
                listing.append(f'{label}{name} ({info.file_size} bytes)')
                if name.lower().endswith('.zip'):
                    tmp = dest / ('_nested_' + name)
                    with zf.open(info) as src, open(tmp, 'wb') as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
                    with zipfile.ZipFile(tmp) as inner:
                        walk(inner, f'{label}{name}/')
                    tmp.unlink()
                elif wanted(name):
                    with zf.open(info) as src, open(dest / name, 'wb') as dst:
                        while chunk := src.read(1 << 20):
                            dst.write(chunk)
        walk(zipfile.ZipFile(zpath), '')
        (dest / '.listing').write_text('\n'.join(listing))
        done.write_text('ok')
    return sorted(p for p in dest.iterdir() if not p.name.startswith('.'))


def find(files, *keywords):
    hits = [p for p in files if all(k in p.name.lower().replace('_', ' ') for k in keywords)]
    if len(hits) != 1:
        raise ValueError(f'expected one file matching {keywords}, found {[p.name for p in hits]} among {[p.name for p in files]}')
    return hits[0]


def read_table(path, usecols):
    with open(path, 'rb') as f:
        head = f.readline().decode('latin-1')
    sep = '|' if head.count('|') >= head.count(',') else ','
    cols = [c.strip().upper() for c in head.strip().split(sep)]
    missing = [c for c in usecols if c not in cols]
    if missing:
        raise ValueError(f'{path.name}: missing columns {missing}; header={cols}')
    df = pd.read_csv(path, sep=sep, dtype=str, encoding='latin-1', keep_default_na=False,
                     usecols=lambda c: c.strip().upper() in usecols)
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def load_year(year, raw):
    zpath, sha = download(year, raw)
    files = extract(zpath, Path(raw) / f'spuf_{year}')
    fpath = find(files, 'basic', 'formulary')
    ppath = find(files, 'plan', 'information')
    form = read_table(fpath, ['FORMULARY_ID', 'CONTRACT_YEAR', 'RXCUI', 'NDC', 'TIER_LEVEL_VALUE',
                              'QUANTITY_LIMIT_YN', 'PRIOR_AUTHORIZATION_YN', 'STEP_THERAPY_YN'])
    plans = read_table(ppath, ['CONTRACT_ID', 'PLAN_ID', 'SEGMENT_ID', 'FORMULARY_ID', 'SNP'])
    source = {'url': SOURCES[year], 'sha256': sha, 'bytes': zpath.stat().st_size,
              'archive_listing': (Path(raw) / f'spuf_{year}' / '.listing').read_text().splitlines(), 'formulary_file': fpath.name, 'plan_file': ppath.name,
              'formulary_rows': len(form), 'plan_rows': len(plans)}
    return form, plans, source


# ---------------------------------------------------------------- curation

def validate_formulary(form, year):
    for c in ['PRIOR_AUTHORIZATION_YN', 'STEP_THERAPY_YN', 'QUANTITY_LIMIT_YN']:
        bad = set(form[c].str.strip().str.upper().unique()) - {'Y', 'N'}
        if bad:
            raise ValueError(f'{year} {c}: unexpected values {sorted(bad)[:10]}')
    if (form.FORMULARY_ID.str.strip() == '').any() or (form.RXCUI.str.strip() == '').any():
        raise ValueError(f'{year}: blank formulary or RXCUI')
    years = set(form.CONTRACT_YEAR.str.strip())
    if years != {str(year)}:
        raise ValueError(f'expected contract year {year}, found {sorted(years)}')


def pairs(form):
    """Collapse NDC rows to one row per (formulary, RXCUI); a flag is set if any NDC has it."""
    f = pd.DataFrame({
        'formulary_id': form.FORMULARY_ID.str.strip(), 'rxcui': form.RXCUI.str.strip(),
        'pa': (form.PRIOR_AUTHORIZATION_YN.str.strip().str.upper() == 'Y').astype(np.int8),
        'st': (form.STEP_THERAPY_YN.str.strip().str.upper() == 'Y').astype(np.int8),
        'ql': (form.QUANTITY_LIMIT_YN.str.strip().str.upper() == 'Y').astype(np.int8),
        'tier': pd.to_numeric(form.TIER_LEVEL_VALUE.str.strip(), errors='coerce')})
    g = f.groupby(['formulary_id', 'rxcui'], sort=False)
    out = g.agg(pa=('pa', 'max'), st=('st', 'max'), ql=('ql', 'max'), tier=('tier', 'min'),
                ndc_rows=('pa', 'size')).reset_index()
    return out


def plan_table(plans):
    p = pd.DataFrame({
        'plan_key': plans.CONTRACT_ID.str.strip() + '-' + plans.PLAN_ID.str.strip() + '-' + plans.SEGMENT_ID.str.strip(),
        'contract_id': plans.CONTRACT_ID.str.strip(),
        'formulary_id': plans.FORMULARY_ID.str.strip(),
        'snp': plans.SNP.str.strip()}).drop_duplicates(['plan_key', 'formulary_id'])
    return p


def link_formularies(plans_cur, plans_prev):
    """Map each current formulary to the prior-year formulary most of its continuing plans used."""
    m = plans_cur[['plan_key', 'formulary_id']].merge(
        plans_prev[['plan_key', 'formulary_id']].rename(columns={'formulary_id': 'prev_formulary_id'}),
        on='plan_key', how='inner')
    if m.empty:
        return pd.DataFrame(columns=['formulary_id', 'prev_formulary_id', 'link_share'])
    c = m.groupby(['formulary_id', 'prev_formulary_id']).size().rename('n').reset_index()
    c['link_share'] = c.n / c.groupby('formulary_id').n.transform('sum')
    c = c.sort_values(['formulary_id', 'n', 'prev_formulary_id'], ascending=[True, False, True])
    return c.drop_duplicates('formulary_id')[['formulary_id', 'prev_formulary_id', 'link_share']]


def build(cur_pairs, cur_plans, prev_pairs, prev_plans):
    """Features for year-Y pairs, using only year Y-1 rules plus structural plan type."""
    n_forms_prev = prev_pairs.formulary_id.nunique()
    drug = prev_pairs.groupby('rxcui').agg(
        drug_pa_rate_prev=('pa', 'mean'), drug_st_rate_prev=('st', 'mean'),
        drug_ql_rate_prev=('ql', 'mean'), drug_mean_tier_prev=('tier', 'mean'),
        drug_forms=('formulary_id', 'nunique')).reset_index()
    drug['drug_breadth_prev'] = drug.pop('drug_forms') / n_forms_prev
    drug['drug_listed_prev'] = 1
    form_prev = prev_pairs.groupby('formulary_id').agg(
        form_pa_share_prev=('pa', 'mean'), form_size_prev=('rxcui', 'size')).reset_index()
    link = link_formularies(cur_plans, prev_plans)
    ptype = cur_plans.groupby('formulary_id').agg(
        is_pdp=('contract_id', lambda s: float(s.str.startswith('S').mean())),
        any_snp=('snp', lambda s: int((~s.isin(['0', ''])).any()))).reset_index()

    d = cur_pairs.rename(columns={'pa': LABEL})[['formulary_id', 'rxcui', LABEL]]
    d = d.merge(link, on='formulary_id', how='left')
    d['form_linked'] = d.prev_formulary_id.notna().astype(int)
    prev = prev_pairs.rename(columns={'formulary_id': 'prev_formulary_id', 'pa': 'prev_pair_pa',
                                      'st': 'prev_pair_st', 'ql': 'prev_pair_ql', 'tier': 'prev_pair_tier'})
    d = d.merge(prev[['prev_formulary_id', 'rxcui', 'prev_pair_pa', 'prev_pair_st', 'prev_pair_ql', 'prev_pair_tier']],
                on=['prev_formulary_id', 'rxcui'], how='left')
    d['prev_pair_listed'] = d.prev_pair_pa.notna().astype(int)
    d = d.merge(drug, on='rxcui', how='left')
    d['drug_listed_prev'] = d.drug_listed_prev.fillna(0).astype(int)
    d['drug_breadth_prev'] = d.drug_breadth_prev.fillna(0.0)
    d = d.merge(form_prev.rename(columns={'formulary_id': 'prev_formulary_id'}), on='prev_formulary_id', how='left')
    d = d.merge(ptype, on='formulary_id', how='left')
    d['is_pdp'] = d.is_pdp.fillna(0.0)
    d['any_snp'] = d.any_snp.fillna(0).astype(int)
    if d.duplicated(['formulary_id', 'rxcui']).any():
        raise ValueError('pair grain violated after joins')
    if len(d) != len(cur_pairs):
        raise ValueError('row count changed during joins')
    return d


# ---------------------------------------------------------------- modeling

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = {'n': int(len(y)), 'positives': int(y.sum()), 'prevalence': float(y.mean())}
    if 0 < y.sum() < len(y):
        out.update(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)))
    out['brier_score'] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    return out


def top_k(y, p, k):
    order = np.argsort(-np.asarray(p), kind='stable')[:k]
    hit = int(np.asarray(y)[order].sum())
    return {'flagged': int(k), 'true_pa': hit, 'precision': hit / k if k else 0.0,
            'recall': hit / int(np.sum(y)) if np.sum(y) else 0.0}


def baselines(d):
    """Rules an analyst could apply without a model, using prior-year data only."""
    drug = d.drug_pa_rate_prev.fillna(d.drug_pa_rate_prev.mean() if d.drug_pa_rate_prev.notna().any() else 0.0)
    carry = np.where(d.prev_pair_listed == 1, d.prev_pair_pa, drug)
    return {'carry_forward_then_drug_rate': carry.astype(float), 'drug_rate_only': drug.to_numpy(float)}


def cluster_bootstrap(y, p, b, groups, reps, seed):
    """AUC difference (model minus baseline), resampling whole formularies."""
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(groups)
    idx_by = [np.flatnonzero(codes == i) for i in range(len(uniq))]
    diffs = []
    for _ in range(reps):
        pick = rng.integers(0, len(uniq), len(uniq))
        ix = np.concatenate([idx_by[i] for i in pick])
        yy = y[ix]
        if 0 < yy.sum() < len(yy):
            diffs.append(roc_auc_score(yy, p[ix]) - roc_auc_score(yy, b[ix]))
    return [float(x) for x in np.quantile(diffs, [.025, .5, .975])], len(diffs)


def run(out, raw, seed=42, boot_reps=200):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    years, sources = {}, {}
    for y in sorted(SOURCES):
        form, plans, src = load_year(y, Path(raw))
        validate_formulary(form, y)
        years[y] = (pairs(form), plan_table(plans))
        src['pairs'] = len(years[y][0])
        src['formularies'] = int(years[y][0].formulary_id.nunique())
        src['rxcuis'] = int(years[y][0].rxcui.nunique())
        src['pa_pair_rate'] = float(years[y][0].pa.mean())
        sources[y] = src
        log(y, {k: v for k, v in src.items() if k != 'archive_listing'})
        del form, plans

    train = build(*years[2025], *years[2024])
    test = build(*years[2026], *years[2025])
    log('train', train.shape, 'test', test.shape)

    # Formulary-grouped validation inside the training year for model selection.
    rng = np.random.default_rng(seed)
    forms = np.array(sorted(train.formulary_id.unique()))
    val_forms = set(rng.choice(forms, size=max(1, len(forms) // 5), replace=False))
    is_val = train.formulary_id.isin(val_forms).to_numpy()
    tr, va = train[~is_val], train[is_val]

    models = {
        'logistic_regression': make_pipeline(SimpleImputer(strategy='median', add_indicator=True),
                                             StandardScaler(), LogisticRegression(max_iter=2000)),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            max_iter=300, learning_rate=.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)}
    validation = {}
    for name, m in models.items():
        m.fit(tr[FEATURES], tr[LABEL])
        validation[name] = metrics(va[LABEL], m.predict_proba(va[FEATURES])[:, 1])
        log('validation', name, validation[name])
    chosen = max(validation, key=lambda n: validation[n]['average_precision'])
    model = models[chosen]
    val_scores = model.predict_proba(va[FEATURES])[:, 1]
    model.fit(train[FEATURES], train[LABEL])  # refit on the full training year; test untouched so far

    y = test[LABEL].to_numpy()
    p = model.predict_proba(test[FEATURES])[:, 1]
    base = baselines(test)
    res = {'model': metrics(y, p), **{k: metrics(y, v) for k, v in base.items()}}

    k = int(y.sum())
    res['top_k_at_true_pa_count'] = {'k': k, 'model': top_k(y, p, k),
                                     **{n: top_k(y, v, k) for n, v in base.items()}}

    seg = {}
    masks = {
        'pairs_listed_prior_year': test.prev_pair_listed.to_numpy() == 1,
        'pairs_new_to_formulary': test.prev_pair_listed.to_numpy() == 0,
        'drugs_new_to_part_d': test.drug_listed_prev.to_numpy() == 0,
        'prior_year_no_pa': (test.prev_pair_listed.to_numpy() == 1) & (test.prev_pair_pa.to_numpy() == 0),
        'prior_year_pa': (test.prev_pair_listed.to_numpy() == 1) & (test.prev_pair_pa.to_numpy() == 1),
    }
    for name, msk in masks.items():
        if msk.sum() == 0:
            continue
        seg[name] = {'model': metrics(y[msk], p[msk]), **{b: metrics(y[msk], v[msk]) for b, v in base.items()}}
    ci = {}
    for name in ['pairs_new_to_formulary', 'prior_year_no_pa']:
        msk = masks[name]
        if msk.sum() and 0 < y[msk].sum() < msk.sum():
            q, n_ok = cluster_bootstrap(y[msk], p[msk], base['carry_forward_then_drug_rate'][msk],
                                        test.formulary_id.to_numpy()[msk], boot_reps, seed + 1)
            ci[name] = {'auc_diff_vs_carry_forward_ci95': [q[0], q[2]], 'median': q[1], 'replicates': n_ok}
    ci_all, n_all = cluster_bootstrap(y, p, base['carry_forward_then_drug_rate'], test.formulary_id.to_numpy(), boot_reps, seed + 2)
    ci['all_pairs'] = {'auc_diff_vs_carry_forward_ci95': [ci_all[0], ci_all[2]], 'median': ci_all[1], 'replicates': n_all}

    # Year-over-year changes the operations team would care about.
    lp = test[test.prev_pair_listed == 1]
    changes = {'listed_both_years': int(len(lp)),
               'pa_added': int(((lp.prev_pair_pa == 0) & (lp[LABEL] == 1)).sum()),
               'pa_removed': int(((lp.prev_pair_pa == 1) & (lp[LABEL] == 0)).sum()),
               'new_pairs': int((test.prev_pair_listed == 0).sum()),
               'new_pairs_with_pa': int(((test.prev_pair_listed == 0) & (test[LABEL] == 1)).sum())}

    test_psi = psi(val_scores, p)
    monitoring = {'reference': 'validation-formulary scores from the training year (CY2025 labels)',
                  'current': 'CY2026 test scores', 'psi': test_psi, 'threshold': .20, 'alert': bool(test_psi > .20)}

    imp = None
    if chosen == 'logistic_regression':
        lr = model.steps[-1][1]
        names = list(model.steps[0][1].get_feature_names_out(FEATURES))
        imp = dict(sorted(zip(names, lr.coef_[0].tolist()), key=lambda t: -abs(t[1])))

    result = {
        'data': 'Real CMS Part D formulary public use files (plan-design records, not claims or patient data)',
        'task': 'Predict whether a (formulary, RXCUI) pair listed in contract year Y requires prior authorization, using year Y-1 rules only',
        'unit': 'formulary_id x RXCUI; PA = any NDC under the RXCUI flagged PRIOR_AUTHORIZATION_YN=Y',
        'sources': {str(k): {kk: vv for kk, vv in v.items()} for k, v in sources.items()},
        'train': {'feature_year': 2024, 'label_year': 2025, 'rows': len(train), 'prevalence': float(train[LABEL].mean()),
                  'validation_formularies': len(val_forms), 'validation_rows': int(is_val.sum())},
        'test': {'feature_year': 2025, 'label_year': 2026, 'rows': len(test), 'formularies': int(test.formulary_id.nunique())},
        'formulary_link_rate_test': float(test.form_linked.mean()),
        'selected_model': chosen, 'selection': 'Highest average precision on held-out training-year formularies; refit on full training year; test evaluated once',
        'validation': validation, 'results': res, 'segments': seg, 'bootstrap': ci,
        'bootstrap_note': f'{boot_reps} formulary-cluster bootstrap replicates; sampling uncertainty for this year pair only',
        'year_over_year_changes': changes, 'monitoring': monitoring, 'coefficients': imp,
        'features': FEATURES,
        'limitations': ['Plan-design records only: no claims, rejections, member counts, or utilization.',
                        'PA at RXCUI level is the max over NDC rows; NDC-specific nuance is collapsed.',
                        'Formulary linkage uses continuing plan keys; unlinked formularies get no pair history.',
                        'One train year pair and one test year pair; not evidence of stability across other years.'],
    }
    (out / 'pa_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=float))

    joblib.dump(model, out / 'model.joblib')
    sha = hashlib.sha256((out / 'model.joblib').read_bytes()).hexdigest()
    ref = np.quantile(val_scores, np.linspace(0, 1, 2001)).tolist()
    (out / 'monitoring_reference.json').write_text(json.dumps(
        {'model_sha256': sha, 'features': FEATURES, 'label': 'prior authorization required (CMS formulary, real data)',
         'validation_scores': ref}, indent=1))
    # Public, non-identifying plan-design feature rows for live validation traffic.
    sample = test.sample(n=min(2000, len(test)), random_state=seed)
    sample[FEATURES].to_json(out / 'reference_pairs.json', orient='records')
    # Aggregates for the dashboard / README.
    tier = test.groupby('prev_pair_tier', dropna=False)[LABEL].agg(['size', 'mean']).reset_index()
    tier.to_csv(out / 'pa_rate_by_prior_tier.csv', index=False)
    manifest = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'seed': seed, 'model_sha256': sha,
                'source_sha256': {str(pth.relative_to(ROOT)): hashlib.sha256(pth.read_bytes()).hexdigest()
                                  for pth in [Path(__file__), ROOT / 'pbm/monitoring.py']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    log(json.dumps({'selected': chosen, 'results': res, 'segments_new': seg.get('pairs_new_to_formulary'),
                    'bootstrap': ci, 'changes': changes, 'monitoring': monitoring}, indent=1, default=float))
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'reports')
    ap.add_argument('--raw', type=Path, default=ROOT / 'data' / 'cms_formulary')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--boot', type=int, default=200)
    a = ap.parse_args()
    run(a.out, a.raw, a.seed, a.boot)
