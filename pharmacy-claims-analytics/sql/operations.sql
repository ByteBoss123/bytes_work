-- All rejected episodes, including immature outcomes; null 7-day labels excluded
-- from the mature denominator. Resolution summaries exclude unresolved/censored rows.
SELECT c.reject_reason, COUNT(*) AS rejected_episodes,
       COUNT(f.unresolved_7d) AS mature_episodes,
       AVG(f.unresolved_7d) AS unresolved_7d_rate,
       SUM(CASE WHEN c.resolved_at <= '2025-08-29 00:00:00' THEN 1 ELSE 0 END) AS resolved_by_snapshot,
       AVG(CASE WHEN c.resolved_at <= '2025-08-29 00:00:00'
           THEN (julianday(c.resolved_at)-julianday(c.submitted_at))*24 END) AS mean_hours_among_resolved
FROM claims c LEFT JOIN rejected_features f ON c.claim_id=f.claim_id
WHERE c.initial_status='rejected'
GROUP BY c.reject_reason ORDER BY rejected_episodes DESC;
