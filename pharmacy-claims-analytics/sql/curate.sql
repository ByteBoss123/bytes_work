-- Snapshot is 2025-08-29. Only fully observed 7-day outcomes enter modeling.
-- Prediction happens immediately after initial rejection; resolved_at is label-only.
DROP VIEW IF EXISTS rejected_features;
CREATE VIEW rejected_features AS
SELECT claim_id, member_id, submitted_at,
       reject_reason, channel, drug_type, days_supply, submitted_cost,
       prior_rejections_90d, member_tenure_days, weekend,
       CASE WHEN resolved_at IS NULL
                  OR julianday(resolved_at) - julianday(submitted_at) > 7
            THEN 1 ELSE 0 END AS unresolved_7d
FROM claims
WHERE initial_status = 'rejected'
  AND julianday(submitted_at) + 7 <= julianday('2025-08-29 00:00:00');
