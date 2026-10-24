"""Reproducible simulation, SQL curation, temporal validation, and monitoring."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sqlite3
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
import joblib

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['days_supply', 'submitted_cost', 'prior_rejections_90d', 'member_tenure_days', 'weekend']
CATEGORICAL = ['reject_reason', 'channel', 'drug_type']
FEATURES = NUMERIC + CATEGORICAL
REASONS = ['coverage_check', 'authorization_required', 'refill_timing', 'submission_error']


def generate(n=60000, seed=42):
    """One synthetic initial claim episode per row. All distributions are assumptions."""
    rng = np.random.default_rng(seed)
    day = rng.integers(0, 240, n)
    hour = rng.integers(0, 24, n)
    submitted = pd.Timestamp('2025-01-01') + pd.to_timedelta(day, unit='D') + pd.to_timedelta(hour, unit='h')
    specialty = rng.random(n) < .16
    prior = rng.poisson(1.2, n)
    tenure = rng.integers(1, 1600, n)
    weekend = (submitted.dayofweek >= 5).astype(int)
    rejected = rng.random(n) < (0.17 + .12 * specialty + .035 * np.minimum(prior, 5))
    reason = rng.choice(REASONS, n, p=[.30, .25, .25, .20])
    cost = np.round(rng.lognormal(4 + 2 * specialty, .8, n), 2)
    channel = rng.choice(['retail', 'mail', 'specialty'], n, p=[.70, .20, .10])
    # Known simulator mechanism, not an empirical clinical relationship.
    latent = (-1.9 + 1.1 * (reason == 'authorization_required') + .7 * (reason == 'coverage_check')
              + .45 * specialty + .22 * prior + .45 * weekend + .5 * (tenure < 90)
              + rng.normal(0, .8, n))
    never = rng.random(n) < 1 / (1 + np.exp(-latent))
    hours = np.maximum(1, np.round(rng.gamma(2, 22, n) * (1 + (reason == 'authorization_required') + .4 * weekend)))
    resolved = pd.Series(submitted + pd.to_timedelta(hours, unit='h'))
    resolved[never & rejected] = pd.NaT
    resolved[~rejected] = submitted[~rejected]
    out = pd.DataFrame({
        'claim_id': [f'SYN-C{i:07d}' for i in range(n)],
        'member_id': [f'SYN-M{i:07d}' for i in range(n)],
        'submitted_at': submitted.astype(str), 'initial_status': np.where(rejected, 'rejected', 'paid'),
        'reject_reason': np.where(rejected, reason, 'none'), 'channel': channel,
        'drug_type': np.where(specialty, 'specialty', 'standard'),
        'days_supply': rng.choice([30, 60, 90], n), 'submitted_cost': cost,
        'prior_rejections_90d': prior, 'member_tenure_days': tenure,
        'weekend': weekend, 'resolved_at': resolved.dt.strftime('%Y-%m-%d %H:%M:%S')})
    return out


def validate(df):
    if df.empty or df.claim_id.duplicated().any():
        raise ValueError('Empty dataset or duplicate claim grain')
    required = ['claim_id', 'member_id', 'submitted_at', 'initial_status'] + FEATURES
    if df[required].isna().any().any():
        raise ValueError('Null required field')
    if not df.initial_status.isin(['paid', 'rejected']).all():
        raise ValueError('Invalid status')
    if not df.reject_reason.isin(REASONS + ['none']).all():
        raise ValueError('Invalid reason')
    if ((df.initial_status == 'paid') != (df.reject_reason == 'none')).any():
        raise ValueError('Status/reason mismatch')
    if not df.channel.isin(['retail', 'mail', 'specialty']).all() or not df.drug_type.isin(['standard', 'specialty']).all():
        raise ValueError('Invalid category')
    if not np.isfinite(df[NUMERIC].to_numpy(dtype=float)).all() or (df[NUMERIC] < 0).any().any() or not df.days_supply.isin([30, 60, 90]).all():
        raise ValueError('Invalid numeric range')
    submitted = pd.to_datetime(df.submitted_at, errors='raise')
    resolved = pd.to_datetime(df.resolved_at, errors='raise')
    if (resolved.dropna() < submitted[resolved.notna()]).any():
        raise ValueError('Resolution precedes submission')


def curate(df, db):
    """Validate before writing; reruns replace the table rather than append duplicates."""
    validate(df)
    with sqlite3.connect(db) as con:
        df.to_sql('claims', con, if_exists='replace', index=False)
        con.execute('CREATE UNIQUE INDEX claim_grain ON claims(claim_id)')
        con.executescript((ROOT / 'sql/curate.sql').read_text())
        return pd.read_sql_query('SELECT * FROM rejected_features ORDER BY submitted_at, claim_id', con)


def split(df):
    t = pd.to_datetime(df.submitted_at)
    # 7-day embargo permits every prior partition label to mature.
    train = df[t < '2025-05-11'].copy()
    val = df[(t >= '2025-05-18') & (t < '2025-06-23')].copy()
    test = df[(t >= '2025-06-30') & (t < '2025-08-21')].copy()
    if any(len(x) < 50 or x.unresolved_7d.nunique() < 2 for x in [train, val, test]):
        raise ValueError('Insufficient mature examples/classes in temporal partitions')
    return train, val, test


def capacity(y, p, fraction=.20):
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, math.ceil(len(y) * fraction))
    selected = np.argsort(-p, kind='stable')[:k]
    found = int(y[selected].sum())
    return {'reviewed': k, 'captured': found, 'total_unresolved': int(y.sum()),
            'precision': found / k, 'recall': found / int(y.sum()) if y.sum() else 0.0}


def metrics(y, p):
    return {'roc_auc': float(roc_auc_score(y, p)),
            'average_precision': float(average_precision_score(y, p)),
            'brier_score': float(brier_score_loss(y, p)),
            'review_20pct': capacity(y, p)}


def psi(reference, current):
    """Fixed probability bins, smoothed counts; threshold is an illustrative setting."""
    edges = np.linspace(0, 1, 11)
    a = np.histogram(reference, edges)[0] + .5
    b = np.histogram(current, edges)[0] + .5
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def run(out, n=60000, seed=42):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate(n, seed)
    curated = curate(df, out / 'claims.sqlite')
    df.head(100).to_csv(out / 'sample_synthetic_claims.csv', index=False)
    train, val, test = split(curated)
    def prep():
        return ColumnTransformer([('numeric', StandardScaler(), NUMERIC),
                                  ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    models = {
        'logistic_regression': make_pipeline(prep(), LogisticRegression(max_iter=1000, random_state=seed)),
        'hist_gradient_boosting': make_pipeline(prep(), HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, l2_regularization=5, random_state=seed))}
    validation = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train.unresolved_7d)
        validation[name] = metrics(val.unresolved_7d, model.predict_proba(val[FEATURES])[:, 1])
    chosen = max(validation, key=lambda name: validation[name]['average_precision'])
    model = models[chosen]  # Frozen training model; no test-driven tuning or refit.
    p = model.predict_proba(test[FEATURES])[:, 1]
    v = model.predict_proba(val[FEATURES])[:, 1]
    rates = train.groupby('reject_reason').unresolved_7d.mean()
    rule = test.reject_reason.map(rates).to_numpy()
    baseline = np.repeat(train.unresolved_7d.mean(), len(test))
    results = {'selected_model': metrics(test.unresolved_7d, p),
               'reason_rate_baseline': metrics(test.unresolved_7d, rule),
               'constant_prevalence_baseline': metrics(test.unresolved_7d, baseline)}
    rng = np.random.default_rng(seed + 1)
    boot = []
    y = test.unresolved_7d.to_numpy()
    for _ in range(300):
        ix = rng.integers(0, len(y), len(y))
        boot.append(roc_auc_score(y[ix], p[ix]))
    stress = test[FEATURES].copy()
    stress['prior_rejections_90d'] += 4
    stress['reject_reason'] = 'authorization_required'
    stress_p = model.predict_proba(stress)[:, 1]
    natural_psi, stress_psi = psi(v, p), psi(v, stress_p)
    monitoring = {'type': 'offline simulation', 'reference': 'validation score distribution',
                  'psi_alert_threshold': .20, 'test_score_psi': natural_psi,
                  'stress_score_psi': stress_psi, 'test_alert': natural_psi > .20,
                  'stress_alert': stress_psi > .20,
                  'stress_scenario': 'Add 4 prior rejections and set reason to authorization_required; labels are not re-evaluated.'}
    scored = test[['claim_id', 'submitted_at', 'reject_reason', 'unresolved_7d']].copy()
    scored['predicted_probability'] = p
    scored.to_csv(out / 'test_predictions.csv', index=False)
    with sqlite3.connect(out / 'claims.sqlite') as con:
        operations = pd.read_sql_query((ROOT / 'sql/operations.sql').read_text(), con)
    operations.to_csv(out / 'operations_by_reason.csv', index=False)
    segments = []
    for reason in REASONS:
        mask = test.reject_reason == reason
        segments.append({'reason': reason, 'n': int(mask.sum()), 'prevalence': float(y[mask].mean()),
                         'brier': float(brier_score_loss(y[mask], p[mask]))})
    result = {'synthetic_only': True, 'seed': seed, 'episodes': n,
              'initial_rejected': int((df.initial_status == 'rejected').sum()),
              'initial_rejection_rate': float((df.initial_status == 'rejected').mean()),
              'snapshot_at': '2025-08-29 00:00:00',
              'splits': {name: {'n': len(part), 'start': part.submitted_at.min(), 'end': part.submitted_at.max(),
                                 'unresolved_rate': float(part.unresolved_7d.mean())}
                         for name, part in [('train', train), ('validation', val), ('test', test)]},
              'selected_model': chosen, 'selection': 'Highest validation average precision; test untouched until final evaluation',
              'validation': validation, 'test': results,
              'test_roc_auc_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist(),
              'bootstrap_note': '300 claim-level bootstrap samples; single-seed synthetic uncertainty only',
              'monitoring': monitoring, 'segments': segments,
              'capacity': [{'fraction': f, 'model': capacity(y, p, f), 'reason_baseline': capacity(y, rule, f)} for f in [.05, .10, .20, .30, .50]],
              'operations': operations.to_dict(orient='records')}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    joblib.dump(model, out / 'model.joblib')
    manifest = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                'sklearn': sklearn.__version__, 'seed': seed,
                'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), ROOT / 'sql/curate.sql', ROOT / 'sql/operations.sql']}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    from pbm.dashboard import render
    render(result, out / 'dashboard.html')
    print(json.dumps({'selected_model': chosen, 'test': results, 'monitoring': monitoring}, indent=2))
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT / 'reports')
    ap.add_argument('--n', type=int, default=60000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    run(args.out, args.n, args.seed)
