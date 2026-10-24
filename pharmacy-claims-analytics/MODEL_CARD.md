# Model card

## Intended use
Portfolio demonstration of prioritizing synthetic initially rejected pharmacy claim episodes for operational review. Positive label: no paid resolution by exactly seven days after initial submission. A resolution at exactly seven days is negative. Missing resolution or resolution later than seven days is positive, provided the label is mature.

Not for denying claims, changing coverage, choosing treatment, or determining access to care. No real healthcare deployment, prospective validation, or outcome improvement is established.

## Data and model
60,000 generated episodes, one episode per synthetic member. Reason/channel/drug categories and numeric features are invented. Prior rejection count is a simulated pre-submission attribute, not a longitudinal aggregation. Demographic attributes are omitted; no protected-group fairness claim is possible. Member uniqueness simplifies evaluation but removes repeated-member behavior.

Training-only StandardScaler and one-hot encoder. Logistic regression and histogram gradient boosting have fixed configurations. Selection uses validation average precision. Test metrics are reported once after selection; no test tuning or generator adjustment to inflate scores. Baselines: constant training prevalence and training mean outcome by rejection reason. Scores are risk estimates under this simulator, not calibrated real-world probabilities.

## Evaluation
ROC-AUC measures ranking across thresholds; average precision reflects positive-class ranking; Brier score measures probability error. Review capacity uses ceil(N*fraction) highest scores with stable chronological tie-breaking. Constant-score capacity approximates a chronological queue, not randomized selection. The reason baseline uses chronological order within tied reasons.

The ROC-AUC interval bootstraps 300 test samples at episode level. It covers sampling variability for a single fixed synthetic generator/model, not data-generation uncertainty, deployment shift, or repeated model selection. The 28-case improvement is descriptive; it has not been established as a statistically significant improvement. No causal claim is made.

Segment counts, prevalence, and Brier scores by rejection reason are in metrics.json. These are descriptive diagnostics, not fairness certification.

## Monitoring simulation
Reference: validation predicted-risk histogram with fixed [0,1] bins and count smoothing. Compare with unmodified test scores, then a separate stress scenario that increases prior rejections by four and forces authorization_required. PSI > 0.20 raises a demo alert. The stress scenario is intentionally strong and not empirically calibrated. It does not measure concept drift or actual post-shift model performance because stressed outcomes are not regenerated.

Labels arrive after seven days. A real monitoring service would also track schema/data quality, delayed labeled performance, missingness, operational backlog, and workflow harms. This implementation runs offline, has no live paging integration, and does not automatically retrain.

## Limitations and release status
Local batch inference, a loopback HTTP scoring/monitoring service, and an offline dashboard. No GCP/AWS deployment, LLM component, live PBM integration, clinical notes, real NCPDP codes, formulary rules, reversals, or actual intervention experiment. Requirements are pinned to the tested environment. Full data and model artifacts are generated locally rather than committed. No PHI is generated or used.


## Service extension
The HTTP service measures actual processing latency, request failures, and score drift during integration traffic, with persisted alert transitions. Traffic remains synthetic; thresholds are illustrative. See SERVICE.md for scope. The separate real CMS aggregate analysis is not a model-training source and supplies no clinical outcome labels.
