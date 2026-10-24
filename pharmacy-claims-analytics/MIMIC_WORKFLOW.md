# Real patient-level medication workflow analysis

This module uses **real, de-identified patient-level electronic health records** from the openly available MIMIC-IV Clinical Database Demo v2.2. It is separate from the synthetic PBM claims model; inpatient prescriptions are not pharmacy-benefit claims.

## Operational problem
Before estimating medication-reconciliation workload or building pharmacy reports, analysts need to know whether prescription components join to orders/admissions reliably and whether timestamps and dose fields are usable.

Run `python -m pbm.mimic_workflow` from this project directory. The script downloads three public tables, validates join cardinality, runs SQL joins, and writes aggregate findings to `reports/mimic_workflow.json`. Raw files remain local under ignored `data/`; no patient-level rows or identifiers are committed.

## Measured findings
- 18,087 prescription-component rows from 100 de-identified patients.
- 15,306 pharmacy records and 275 admission records in the source tables.
- Prescription rows cover 250 admissions and 15,225 distinct pharmacy orders.
- All 18,087 prescription rows linked to pharmacy and admission records without inflating the row count.
- 753 rows have stop timestamps earlier than start timestamps; these are flagged for investigation, not asserted to be clinical errors.
- 6 missing route values, 9 missing dose-value fields, and 9 missing dose-unit fields (counts may overlap).
- Median prescription-component workload: 53 rows per admission with prescriptions; 90th percentile: 146.2 rows.
- Entered-to-verified median and p90 were both zero among 15,300 nonnegative pairs. This is insufficient evidence for a bottleneck claim and may reflect source recording practices. Six pairs were missing.

## Interpretation
The workflow establishes a reproducible medication-order data-quality and workload baseline. It does not demonstrate improved care, reduced turnaround time, fewer denials, or cost savings. Missing fields and timestamp anomalies require domain review before downstream use. Prescription rows can be order components, so 18,087 is not a count of independent prescriptions.

Dates are shifted per patient; absolute cross-patient calendar comparisons, day-of-week inference, and seasonal trends are inappropriate. The 100-patient demo is small, historical, and not representative of PBM members. No clinical decisions, re-identification, patient risk scores, or live patient-data serving are performed.

## Attribution and license
Contains information from MIMIC-IV Clinical Database Demo v2.2, made available under the Open Data Commons Open Database License v1.0: https://opendatacommons.org/licenses/odbl/1-0/ . The aggregate output and reproducible derivative query are provided under that same database license where applicable. This notice does not relicense unrelated project software.

Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet. https://doi.org/10.13026/dp1f-ex47

PhysioNet citation: Pollard, T., Moody, B. E., Lehman, L., Gow, B., Fernandes, C., Xie, C., Johnson, A., Mark, R. G., & Heldt, T. (2026). PhysioNet as a global platform for biomedical research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z

Source: https://physionet.org/content/mimic-iv-demo/2.2/
Schema: https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html

## Supported resume wording
Analyzed 18,087 prescription-component records from 100 de-identified patients using Python and SQL; reconciled pharmacy/admission joins without row inflation and flagged timestamp and documentation gaps to support medication-workflow reporting.
