# Real public CMS data: spending prioritization

## Business question
Which generic-drug groups account for the largest share of recorded drug spending in a bounded CMS extract, and should be prioritized for further analyst review?

This is a separate real-data analysis. It does not supply rejection labels or replace the synthetic model's training data.

## Source and scope
CMS Medicare Part D Prescribers — by Provider and Drug:
https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug

The exact API request, UTC retrieval time, raw-response SHA-256, SQL, and scope are stored in `reports/cms_analysis.json`. It contains the first 5,000 rows from the source API's ordering, not a random, national, or Virginia cohort. This endpoint was indexed as a historical release in the HHS catalog; do not describe this as current-year claims data. Provider/drug aggregate rows are not patient records.

## Implemented analysis

`python -m pbm.cms_data` downloads the bounded extract, checks provider/drug grain and numeric totals, and runs SQL aggregation by generic name. It produces `cms_spend_by_generic.csv` and a JSON evidence summary. Provider names and identifiers are not persisted because they are unnecessary for this business question. The public dataset contains suppression; do not infer zeros or sum beneficiary counts across drugs.

Measured extract: **5,000 aggregate rows**, **553 generic names**, **286,832 reported claims**, and **$36,679,472.47 in recorded drug cost**. The top ten generic groups account for **44.1%** of this extract's spending. This identifies an analyst review queue, not inappropriate prescribing or proven savings. Cost excludes any assumption about net PBM economics after rebates. Case mix, formulations, prescribing patterns, and selection affect comparison.

## Limitations

No patient-level features, rejected claims, clinical notes, resolution timestamps, or labels for the synthetic prediction task are available here. Real patient-level claims would require an authorized source and appropriate data handling. Do not describe these aggregates as patient records or use spending concentration to make clinical decisions.

## Supported resume wording

Analyzed 5,000 real CMS provider-drug aggregate records using Python and SQL, profiling $36.7M in recorded drug spending and identifying ten generic-drug groups accounting for 44.1% of spending within the extract for further analyst review.

Keep the word **aggregate** and the extract scope. This work supports real public healthcare-data familiarity, not professional clinical experience.

The raw API response is not committed. The hash records the fetched response, and the committed drug-level aggregate CSV preserves the analysis output. API reruns may differ if the source changes.
