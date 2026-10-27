"""Summarize the scheduled canary: Lambda invocations, errors, duration percentiles, alarm changes.

Usage: python aws/canary_report.py [--out reports/aws_canary_report.json]
Reads CloudWatch for the schedule's window (start of schedule to now or its end date).
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import boto3

REGION, FN = 'us-east-1', 'pbm-claims-scoring'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('reports/aws_canary_report.json'))
    args = ap.parse_args()
    s = boto3.Session(region_name=REGION)
    sch = s.client('scheduler').get_schedule(Name='pbm-claims-canary')
    cw = s.client('cloudwatch')
    end = min(dt.datetime.now(dt.timezone.utc), sch['EndDate'])
    start = sch['CreationDate']
    dims = [{'Name': 'FunctionName', 'Value': FN}]

    def total(metric):
        pts = cw.get_metric_statistics(Namespace='AWS/Lambda', MetricName=metric, Dimensions=dims, StartTime=start,
                                       EndTime=end, Period=3600, Statistics=['Sum'])['Datapoints']
        return sum(p['Sum'] for p in pts)

    dur = cw.get_metric_statistics(Namespace='AWS/Lambda', MetricName='Duration', Dimensions=dims, StartTime=start,
                                   EndTime=end, Period=3600, ExtendedStatistics=['p50', 'p95', 'p99'])['Datapoints']
    history = {}
    for a in cw.describe_alarms(AlarmNamePrefix='pbm-scoring-')['MetricAlarms']:
        items = cw.describe_alarm_history(AlarmName=a['AlarmName'], HistoryItemType='StateUpdate',
                                          StartDate=start, EndDate=end)['AlarmHistoryItems']
        history[a['AlarmName']] = [f"{i['Timestamp'].isoformat()} {i['HistorySummary']}" for i in items]
    report = {'window_utc': [start.isoformat(), end.isoformat()],
              'hours': round((end - start).total_seconds() / 3600, 2),
              'invocations': total('Invocations'), 'errors': total('Errors'), 'throttles': total('Throttles'),
              'duration_ms_hourly_percentiles': sorted(
                  [{'hour': d['Timestamp'].isoformat(), **d['ExtendedStatistics']} for d in dur], key=lambda x: x['hour']),
              'alarm_state_changes': history,
              'note': 'Invocations include the canary (1/min) plus any manual validation traffic in the window. '
                      'Author-sent canary traffic replaying public formulary rows.'}
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ['window_utc', 'hours', 'invocations', 'errors', 'throttles']}, indent=2))


if __name__ == '__main__':
    main()
