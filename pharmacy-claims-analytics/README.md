# Pharmacy Benefit Operations Analytics

**Real public data only.** A prior-authorization (PA) forecast trained on CMS Part D formulary files, an HTTP scoring service with drift, failure-rate and latency alerts, and two real healthcare-data analyses (de-identified MIMIC-IV medication records and CMS Part D spending aggregates). Python, SQL, scikit-learn.

This is a portfolio project. It is not professional PBM experience and is not affiliated with CVS Health or Caremark. No claims, rejections, member records or patient-level PBM data are used anywhere: those are not public.

## The operational question

Each fall, Part D plans publish next year's formulary rules. A prior-authorization requirement means a pharmacy claim for that drug will reject until the PA is approved, so PA changes drive rejection volume, call volume and member outreach.

**For each (formulary, drug) pair a plan lists next year, will it require prior authorization, and which pairs are *newly* gaining PA?** Every feature comes from the prior contract year's public file; the label is the next year's PA flag.

## Data

CMS Quarterly Prescription Drug Plan Formulary, Pharmacy Network, and Pricing Information files ([data.cms.gov](https://catalog.data.gov/dataset/quarterly-prescription-drug-plan-formulary-pharmacy-network-and-pricing-information)). Exact URLs and SHA-256 hashes of each download are in [`reports/pa_metrics.json`](reports/pa_metrics.json). Raw files are not committed.

| Contract year | Formulary-drug pairs | Formularies | Drugs (RXCUI) | Pairs with PA |
|---|---:|---:|---:|---:|
| 2024 | 1,490,827 | 421 | 6,522 | 26.6% |
| 2025 | 1,298,449 | 378 | 6,092 | 27.7% |
| 2026 | 1,121,429 | 331 | 5,994 | 28.1% |

Unit: formulary × RXCUI (PA = any NDC under the RXCUI flagged). Formularies get new IDs every year, so each 2026 formulary is linked to its 2025 predecessor through the continuing plans (contract-plan-segment) that use it; 90.8% of 2026 pairs belong to a linked formulary.

## Design

- **Train:** 2024 features → 2025 labels (1,298,449 pairs). **Test:** 2025 features → 2026 labels (1,121,429 pairs), evaluated once.
- **Model selection:** 75 held-out training-year formularies (255,908 pairs). Histogram gradient boosting beat logistic regression on average precision (0.992 vs 0.987), then was refit on the full training year.
- **Features (16):** last year's PA/step-therapy/quantity-limit flags and tier for the same pair; the drug's prior-year PA, step-therapy and quantity-limit rates across all formularies, breadth and mean tier; the predecessor formulary's PA share and size; plan type (PDP vs MA) and special-needs status. No year-Y rule enters the features.
- **Baselines an analyst could run without a model:** (1) carry forward last year's PA flag, falling back to the drug's prior-year PA rate for new pairs; (2) the drug's prior-year PA rate alone.
- **Uncertainty:** 200-replicate bootstrap that resamples whole formularies.

## Results (CY2026 test year)

| | Model | Carry-forward rule | Drug PA rate only |
|---|---:|---:|---:|
| ROC-AUC, all 1.12M pairs | **0.995** | 0.986 | 0.982 |
| Average precision | **0.990** | 0.969 | 0.960 |
| Brier score (lower is better) | 0.0166 | **0.0163** | 0.0458 |
| Precision when flagging the top 314,740 pairs (the true PA count) | 96.2% | **96.4%** | 88.0% |

**What this means**
- The model ranks pairs better overall (AUC +0.009, 95% CI +0.008 to +0.010), but **at the operating point it does not beat the carry-forward rule**: precision is 96.2% vs 96.4%, and calibration is no better. Last year's PA flag explains most of next year's.
- **The hard part is change.** 985,605 pairs appeared in both years; only 6,242 gained PA (0.87%) and 4,528 lost it. For flagging the pairs that gain PA, the model (AUC 0.840, AP 0.034) is slightly **worse** than simply ranking by the drug's prior-year PA rate (AUC 0.852, AP 0.040). Carry-forward cannot rank these at all (AUC 0.5).
- **Pairs new to a formulary** (135,824; 33.8% PA): model AUC 0.979 vs 0.978 for the drug-rate fallback; the difference is not significant (CI −0.0003 to +0.0016).
- **Drugs new to Part D** (7,912 pairs; 46.5% PA): AUC 0.556. Prior-year formulary data carries almost no signal for them.
- PA rate by last year's tier: 78.6% at tier 5 (specialty) vs 8.5% at tier 2 ([`pa_rate_by_prior_tier.csv`](reports/pa_rate_by_prior_tier.csv)).
- Score drift between the training-year validation formularies and the 2026 test year: PSI 0.041, under the 0.20 alert threshold.

**Recommendation from this run:** for next-year PA planning, start from the carry-forward list and rank non-PA pairs by the drug's PA rate across other plans; the gradient-boosted model adds ranking polish, not a better operating list. The open problem is predicting PA additions and new-to-market drugs, which needs signals this file does not have (launch dates, list price, clinical class).

## Scoring service and monitoring

[`pbm/service.py`](pbm/service.py) serves the CI-trained model over HTTP (`/predict`, `/health`, `/monitoring`, `/metrics`). It refuses to start unless the model's SHA-256 matches the monitoring reference, rejects labels, identifiers and out-of-range values, and tracks:

- request failure rate over the last 100 requests (alert above 5%)
- p95 processing latency (alert above 2,000 ms)
- PSI of the last 1,000 scores against the validation-score reference (alert above 0.20)

The loopback integration test ([`reports/service_validation.json`](reports/service_validation.json)) sends 145 HTTP requests replaying real 2026 feature rows. A shift phase replays only pairs that required PA last year (real rows, unaltered values), which raised the drift alert; invalid payloads raised the failure-rate alert; both cleared on recovery.

**AWS (deployed Oct 27, 2026):** the service runs as a container image on AWS Lambda behind an IAM-authenticated Function URL, built by CodeBuild from commit `f21ee17` with the CI-trained model (fingerprint `529e5e44…`). A 150-request validation replaying real 2026 formulary rows returned every expected status (140 × 200, 10 × 400, 0 × 5xx). The drift and failure-rate alerts raised and cleared, and both CloudWatch alarms fired and sent their SNS notifications 48 s and 73 s after the service raised them. Evidence: [`reports/aws_lambda_validation.json`](reports/aws_lambda_validation.json).

**Dashboard:** `/dashboard` on the service (screenshots: [desktop](reports/dashboard_desktop.png), [dark](reports/dashboard_dark.png), [phone](reports/dashboard_mobile.png), rendered in headless Chromium by [`dashboard-screenshot.yml`](../.github/workflows/dashboard-screenshot.yml)).

## Medication extraction from real clinical notes

[`pbm/notes_extraction.py`](pbm/notes_extraction.py), run by [`notes-extraction.yml`](../.github/workflows/notes-extraction.yml). Data: the [OHSU Glaucoma Medication Dataset](https://github.com/jche253/Glaucoma_Med_Dataset) (Chen et al., TVST 2022; BSD-3-Clause, commit `15e67ea`): 480 de-identified ophthalmology visit notes from Casey Eye Institute with clinician sentence-level labels. Results: [`reports/notes_extraction.json`](reports/notes_extraction.json); every test prediction is in [`reports/notes_test_predictions.jsonl`](reports/notes_test_predictions.jsonl).

- **Split:** cut at a note boundary. Train 4,411 sentences; test 1,081 sentences from 74 held-out notes. 19 test sentences whose gold drug, frequency or route was masked by de-identification are excluded, and 9 unparseable annotation lines are dropped.
- **Baseline:** drug, frequency and route lexicons learned from the training notes only, keeping strings that were labeled at least half the times they appeared.
- **LLM:** Qwen2.5-7B-Instruct (Q4_K_M GGUF, Apache-2.0) on a 4-core CPU runner, 7 few-shot training examples, temperature 0, JSON-schema-constrained output. Each returned string is mapped back onto the sentence; strings not found in it count as errors.

| Strict span F1 (held-out notes) | Lexicon baseline | LLM | Difference (95% CI, note bootstrap) |
|---|---:|---:|---:|
| Drug (338 gold) | **0.953** | 0.930 | −0.023 (−0.046 to −0.001) |
| Frequency (286) | **0.963** | 0.960 | −0.003 (−0.024 to +0.017) |
| Route (278) | 0.711 | **0.909** | **+0.198** (+0.134 to +0.255) |
| All | 0.875 | **0.933** | **+0.058** (+0.033 to +0.080) |

The JSON output format itself caps F1 at 0.977 (gold labels pushed through the same mapping). The LLM's gain is almost entirely in routes: the lexicon tags every "OS"/"OU" in exam findings, while the LLM separates medication routes from exam text. On drugs the LLM is slightly worse; its false positives include "drops" and "POAG", which is the glaucoma diagnosis, not a drug. 6.1% of the strings it returned (60 of 980) did not appear in the sentence. It produced no invalid JSON and averaged 3.4 s per sentence (p95 6.4 s).

**Link to Part D rules.** The 25 most-mentioned drug terms were mapped to RxNorm (NLM RxNav) and joined to the real CY2026 formulary file. 14 matched products listed on 224–331 of 331 formularies. Among formularies listing them: brinzolamide (Azopt) has step therapy on 27%, Rhopressa on 13% and latanoprost on 10%; Restasis has quantity limits on 72%. The 11 unmatched terms are mostly over-the-counter artificial-tear abbreviations that Part D does not cover. Caveats: a brand name's RxNorm products include its generic equivalents, and the join is not route-specific (prednisolone's 45% PA rate reflects oral products).

## Drug-label and Part D coverage question answering (RAG)

[`pbm/drug_rag.py`](pbm/drug_rag.py). Results: [`reports/drug_rag.json`](reports/drug_rag.json). Run on AWS CodeBuild (build #8, commit `e9eb6da`) with Qwen3-32B on Amazon Bedrock. Per-question predictions and the fetched label cache are in `s3://pbm-claims-scoring-331604528033/drug-rag/8/`.

- **Drugs:** the 80 highest-spend generic names in the real CMS Part D prescriber extract. 59 matched an FDA prescribing-information label through openFDA, by exact name, salt-stripped name, or an RxNorm approximate match plus product RxCUI. The 21 unmatched are mostly truncated combination names, biologics and devices such as pen needles.
- **Corpus (775 chunks):** label sections (boxed warning, indications, dosage and administration, dosage forms, contraindications, warnings; 600 words each at most), plus one CY2026 coverage summary per drug computed from the real formulary file (55 drugs).
- **173 questions with gold answers from structured fields the system never sees:**
  - route of administration, from openFDA route metadata
  - whether the label has a boxed warning
  - the share of listing formularies that require prior authorization (±0.5 points)
- **Retrieval:** hybrid BM25 + bge-small with reciprocal-rank fusion, restricted to the drug named in the question. The drug is matched exactly, then fuzzily (similarity ≥ 0.85), then inferred when one drug dominates the top-10 hits; otherwise the search is unfiltered.

| Accuracy (173 questions, 59 drugs) | Closed-book | RAG | Difference (95% CI, drug bootstrap) |
|---|---:|---:|---:|
| Prior-authorization share (55) | 0% (all abstained) | **100%** | +100 points |
| Boxed warning yes/no (59) | 45.8% | **89.8%** | +44.1 (+32.2 to +57.6) |
| Route (59) | **83.1%** | 71.2% | −11.9 (−25.4 to +1.7) |
| All | 43.9% | **86.7%** | **+42.8 (+36.2 to +49.7)** |

- **Retrieval:** every retriever found the right drug's documents in the top 5 for all 173 original questions. The difference was precision. Without the drug filter, 34.4% of boxed-warning questions about drugs that have no boxed warning also retrieved another drug's boxed-warning passage. With the filter, 0%.
- **Robustness (222 rephrased questions):**
  - **Misspelled brands** (3rd and 4th letters swapped): top-5 hit rate rose from 53.4% to 96.6% with the filter, and other drugs' boxed-warning passages in the top 5 fell from 90.3% to 6.5%.
  - **Generic names instead of brands:** 98.1% to 100%.
  - **Answers to misspelled boxed-warning questions:** 81.0% vs 69.0% unfiltered (+12.1, CI −1.7 to +25.9), but 3 false "yes" answers vs 1. The filter is not a free win at the answer level.
- **Route is where RAG loses.** The model already knows most routes. The 600-word section cap sometimes cut the sentence that states the route, which led to 8 abstentions. A fix would be indexing the full dosage-and-administration and dosage-forms sections.
- **Grounding:** every RAG answer with valid citations (162 of 163) cited only passages from the correct drug.
- **Format compliance:** 72 of 173 RAG answers ignored the forced tool call and answered in prose. Those were parsed (answer text, [C#] citations, "passages do not specify" counted as an abstention) and are counted under `text_fallback`.

## Real healthcare-data analyses

- [MIMIC-IV medication workflow](MIMIC_WORKFLOW.md): 18,087 prescription-component rows from 100 de-identified patients; join validation and timestamp/documentation quality checks.
- [CMS Part D spending](REAL_DATA.md): 5,000 provider-drug aggregate rows covering $36.7M in recorded drug cost.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p "test_*.py" -v
python -m pbm.formulary --out reports      # downloads ~8 GB of CMS archives
PYTHONPATH=. python tests/service_integration.py
python -m pbm.service --reports reports
```

The real-data run executes in GitHub Actions ([`.github/workflows/pa-formulary.yml`](../.github/workflows/pa-formulary.yml)), which commits the metrics, model and monitoring reference back to this branch. Log: [`reports/formulary_run.log`](reports/formulary_run.log).

## Limitations

- Plan-design records only: no claims, rejection counts, member enrollment or utilization, so impact on rejection volume is not measured.
- One training year pair and one test year pair. The CY2027 file (expected January 2027) would allow a prospective test of the frozen model.
- PA is collapsed to RXCUI level; NDC-specific rules are lost.
- 9.2% of 2026 pairs sit in formularies with no linked predecessor and get no pair history.

## Supported resume wording

> Forecast 2026 prior-authorization requirements for 1.12M real CMS Part D formulary-drug pairs (331 formularies) from prior-year rules; gradient boosting reached 0.995 ROC-AUC but matched a carry-forward rule at the operating cutoff (96.2% vs 96.4% precision), so recommended the simpler rule.

Keep "formulary" or "plan-design" data. Do not describe this as claims data, rejection data or production PBM work.
