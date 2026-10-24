# Data dictionary and simulation contract

Grain: one initial claim episode, not one transaction. Identifiers carry a SYN prefix. All dates are timezone-naive within a single assumed operational timezone; there is no daylight-saving handling. No real identifiers or standardized drug/rejection codes.

| Field | Definition | Available at prediction? |
|---|---|---|
| claim_id | Unique synthetic episode identifier | Yes, excluded from model |
| member_id | Unique member per episode in this simplified simulator | Yes, excluded |
| submitted_at | Initial submission timestamp | Yes, excluded except derived weekend |
| initial_status | Initially paid or rejected | Yes; only rejected enter model |
| reject_reason | Invented workflow category; none for paid | Yes |
| channel | Simulated retail/mail/specialty routing | Yes |
| drug_type | Synthetic standard/specialty category | Yes |
| days_supply | 30, 60, or 90; independent synthetic attribute | Yes |
| submitted_cost | Simulated nonnegative amount; not savings, paid amount, or member liability | Yes |
| prior_rejections_90d | Invented pre-submission history count | Yes |
| member_tenure_days | Invented membership tenure | Yes |
| weekend | Submission falls on Saturday/Sunday | Yes |
| resolved_at | First simulated paid-resolution time, or null | NO: label-only |
| unresolved_7d | No resolution within seven days, mature rows only | NO: target |

## Simulator assumptions

Submission days are uniform across 240 days starting Jan 1, 2025. Specialty share is 16%; reason mix is 30% coverage_check, 25% authorization_required, 25% refill_timing, 20% submission_error. Initial rejection probability depends on synthetic specialty and prior rejection count. Nonresolution uses a noisy logistic mechanism involving reason, specialty, prior rejections, weekend, and tenure. Resolution hours follow a gamma distribution modified by reason/weekend. These are arbitrary pedagogical assumptions, not estimates from claims data.

No NCPDP mappings, real NDCs, clinical conditions, payer contracts, or reimbursement logic are represented. Some future resolution timestamps are present in generated raw rows for label construction; snapshot-level operations explicitly exclude resolutions after the snapshot. Immature seven-day outcomes are excluded from model and rate denominators, including early-resolving immature cases to avoid selection bias.

## Metrics

- Initial rejection rate = initially rejected / all initial episodes.
- Seven-day unresolved rate = positive labels / mature rejected episodes.
- Mean hours among resolved = total observed resolution hours / resolved rejected episodes by snapshot. This excludes unresolved episodes and is explicitly conditional.
- Review recall = true unresolved cases in selected top-k / all unresolved test cases.
- Review precision = true unresolved cases in selected top-k / k.
