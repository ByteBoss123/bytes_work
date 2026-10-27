# Model card: next-year prior-authorization forecast

## Intended use
Rank (formulary, drug) pairs listed for the next contract year by the chance they require prior authorization, to help plan PA review capacity and member outreach. Portfolio demonstration on public CMS plan-design data. Not for coverage decisions, clinical decisions, or any individual member.

## Data
Real CMS Part D quarterly formulary public use files for contract years 2024, 2025 and 2026 (URLs and SHA-256 in `reports/pa_metrics.json`). Plan-design records only: no claims, members or patients. Unit: formulary × RXCUI; PA is 1 if any NDC under the RXCUI is flagged.

## Training and selection
Features from year Y-1, label from year Y. Train 2024→2025 (1,298,449 pairs), test 2025→2026 (1,121,429 pairs). Formularies are linked across years through continuing contract-plan-segment keys. Logistic regression (median imputation with missing indicators, scaling) and histogram gradient boosting were compared on 75 held-out training-year formularies; gradient boosting won on average precision and was refit on the full training year. The test year was evaluated once.

## Performance (2026 test year)
| | Model | Carry-forward rule | Drug PA rate |
|---|---:|---:|---:|
| ROC-AUC | 0.995 | 0.986 | 0.982 |
| Average precision | 0.990 | 0.969 | 0.960 |
| Brier score | 0.0166 | 0.0163 | 0.0458 |
| Precision at true-PA count | 96.2% | 96.4% | 88.0% |

Segments: pairs gaining PA among prior no-PA pairs (6,242 of 718,543) AUC 0.840 vs 0.852 for the drug-rate rule; pairs new to a formulary AUC 0.979 vs 0.978 (difference CI includes zero); drugs new to Part D AUC 0.556.

## Known weaknesses
- Does not beat the carry-forward rule at the operating cutoff, and is slightly worse than the drug-rate rule at detecting PA additions.
- Near chance for drugs with no Part D history.
- One year pair for training and one for testing; stability across other years is untested.
- Scores are probabilities under this data-generating process, not guarantees about any plan's decision.

## Monitoring
The service compares the last 1,000 scores with the validation-score reference (PSI, alert above 0.20). Test-year PSI was 0.041. True performance can only be checked when the next contract year's file is published, so the service cannot monitor accuracy live.
