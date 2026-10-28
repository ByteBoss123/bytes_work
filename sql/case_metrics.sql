-- Build status spells and one row per incident from the privacy-gated event table `ev`.
-- A spell is the time an incident sits in one status until its next status change.
-- All durations are ELAPSED calendar hours in a status, not effort.

CREATE OR REPLACE TABLE spells AS
WITH ordered AS (
    SELECT
        case_id, ts, status, substatus, grp, org, impact, product,
        ROW_NUMBER() OVER (PARTITION BY case_id ORDER BY ts, seq) AS step_no,
        LEAD(ts) OVER (PARTITION BY case_id ORDER BY ts, seq)     AS next_ts,
        LAG(grp) OVER (PARTITION BY case_id ORDER BY ts, seq)     AS prev_grp
    FROM ev
),
first_done AS (
    SELECT case_id, MIN(ts) AS resolved_at FROM ev WHERE status = 'Completed' GROUP BY case_id
)
SELECT
    o.*,
    f.resolved_at,
    EXTRACT(EPOCH FROM (o.next_ts - o.ts)) / 3600.0 AS dur_h,
    o.ts < f.resolved_at                             AS pre_resolution,
    CASE
        WHEN o.status = 'Queued'                                  THEN 'queue'
        WHEN o.substatus IN ('In Progress', 'Assigned')           THEN 'active_status'
        WHEN o.substatus = 'Wait - User'                          THEN 'wait_user'
        WHEN o.substatus LIKE 'Wait%'                             THEN 'wait_other'
        WHEN o.status = 'Completed'                               THEN 'completed'
        ELSE 'other'
    END AS spell_type,
    (o.prev_grp IS NOT NULL AND o.grp IS DISTINCT FROM o.prev_grp) AS is_transfer
FROM ordered o
LEFT JOIN first_done f USING (case_id);

CREATE OR REPLACE TABLE case_metrics AS
WITH pre AS (
    SELECT * FROM spells WHERE pre_resolution
),
agg AS (
    SELECT
        case_id,
        SUM(CASE WHEN spell_type = 'queue'         THEN dur_h ELSE 0 END) AS queue_h,
        SUM(CASE WHEN spell_type = 'active_status' THEN dur_h ELSE 0 END) AS active_status_h,
        SUM(CASE WHEN spell_type = 'wait_user'     THEN dur_h ELSE 0 END) AS wait_user_h,
        SUM(CASE WHEN spell_type = 'wait_other'    THEN dur_h ELSE 0 END) AS wait_other_h,
        SUM(CASE WHEN spell_type = 'other'         THEN dur_h ELSE 0 END) AS other_h,
        SUM(CASE WHEN is_transfer THEN 1 ELSE 0 END)                       AS transfers,
        COUNT(DISTINCT grp)                                                AS groups_involved,
        MAX(CASE WHEN spell_type = 'wait_user' THEN 1 ELSE 0 END)          AS used_wait_user,
        MAX(CASE WHEN grp LIKE '%3rd' THEN 3 WHEN grp LIKE '%2nd' THEN 2 ELSE 1 END) AS max_line
    FROM pre GROUP BY case_id
),
-- ping-pong: incident leaves a group and later returns to it before resolution
grp_seq AS (
    SELECT case_id, grp, step_no FROM pre WHERE is_transfer OR step_no = 1
),
pingpong AS (
    SELECT case_id, MAX(CASE WHEN n_visits > 1 THEN 1 ELSE 0 END) AS ping_pong
    FROM (SELECT case_id, grp, COUNT(*) AS n_visits FROM grp_seq GROUP BY case_id, grp)
    GROUP BY case_id
),
bounds AS (
    SELECT
        case_id,
        MIN(ts)                                       AS opened_at,
        ANY_VALUE(resolved_at)                        AS resolved_at,
        MAX(ts)                                       AS last_event_at,
        arg_min(org, ts)                              AS owning_org,
        arg_min(grp, ts)                              AS first_group,
        ANY_VALUE(impact)                             AS impact,
        ANY_VALUE(product)                            AS product,
        MAX(CASE WHEN status <> 'Completed' AND ts > resolved_at THEN 1 ELSE 0 END) AS reopened
    FROM spells GROUP BY case_id
)
SELECT
    b.*,
    EXTRACT(EPOCH FROM (b.resolved_at - b.opened_at)) / 3600.0 AS resolution_h,
    COALESCE(a.queue_h, 0) AS queue_h, COALESCE(a.active_status_h, 0) AS active_status_h,
    COALESCE(a.wait_user_h, 0) AS wait_user_h, COALESCE(a.wait_other_h, 0) AS wait_other_h,
    COALESCE(a.other_h, 0) AS other_h,
    COALESCE(a.transfers, 0) AS transfers, COALESCE(a.groups_involved, 1) AS groups_involved,
    COALESCE(a.used_wait_user, 0) AS used_wait_user, COALESCE(a.max_line, 1) AS max_line,
    COALESCE(p.ping_pong, 0) AS ping_pong
FROM bounds b
LEFT JOIN agg a USING (case_id)
LEFT JOIN pingpong p USING (case_id);
