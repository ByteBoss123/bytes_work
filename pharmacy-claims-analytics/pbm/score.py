"""Batch scoring entry point for locally generated, trusted model artifacts."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pbm.pipeline import FEATURES, NUMERIC, CATEGORICAL


def score(model_path, input_path, output_path):
    data = pd.read_csv(input_path)
    required = ['claim_id', 'initial_status'] + FEATURES
    if not set(required).issubset(data.columns):
        raise ValueError('Input missing required columns: ' + ', '.join(sorted(set(required) - set(data.columns))))
    if data[required].isna().any().any() or data.claim_id.duplicated().any():
        raise ValueError('Missing required data or duplicate claim_id')
    if not data.initial_status.isin(['paid', 'rejected']).all():
        raise ValueError('Invalid initial status')
    if not np.isfinite(data[NUMERIC].to_numpy(dtype=float)).all() or (data[NUMERIC] < 0).any().any():
        raise ValueError('Invalid numeric features')
    rows = data.loc[data.initial_status == 'rejected'].copy()
    if rows.empty:
        raise ValueError('No rejected episodes to score')
    # Joblib/pickle can execute code: only load artifacts you generated or trust.
    model = joblib.load(model_path)
    transformer = model.steps[0][1]
    encoder = transformer.named_transformers_['category']
    for column, categories in zip(CATEGORICAL, encoder.categories_):
        if not rows[column].isin(categories).all():
            raise ValueError('Unseen category in ' + column + '; investigate schema drift before scoring')
    rows['predicted_unresolved_7d'] = model.predict_proba(rows[FEATURES])[:, 1]
    result = rows[['claim_id', 'predicted_unresolved_7d']].sort_values('predicted_unresolved_7d', ascending=False)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    score(a.model, a.input, a.output)
