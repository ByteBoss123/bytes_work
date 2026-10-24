"""Delete everything aws/deploy_aws.py created (us-east-1)."""
import boto3
from botocore.exceptions import ClientError

REGION, NAME = 'us-east-1', 'pbm-claims-scoring'


def quiet(fn, **kw):
    try:
        fn(**kw)
        print('deleted', fn.__name__, kw)
    except ClientError as e:
        print('skip', fn.__name__, e.response['Error']['Code'])


s = boto3.Session(region_name=REGION)
acct = s.client('sts').get_caller_identity()['Account']
lam, cw, logs, sns, cb, ecr, iam = (s.client(x) for x in
                                       ['lambda', 'cloudwatch', 'logs', 'sns', 'codebuild', 'ecr', 'iam'])
quiet(s.client('scheduler').delete_schedule, Name='pbm-claims-canary')
alarms = [a['AlarmName'] for a in cw.describe_alarms(AlarmNamePrefix='pbm-scoring-')['MetricAlarms']]
if alarms:
    quiet(cw.delete_alarms, AlarmNames=alarms)
for f in ['score_drift', 'request_failure_rate', 'latency']:
    quiet(logs.delete_metric_filter, logGroupName=f'/aws/lambda/{NAME}', filterName=f'pbm-{f}-raised')
quiet(lam.delete_function, FunctionName=NAME)
quiet(logs.delete_log_group, logGroupName=f'/aws/lambda/{NAME}')
quiet(sns.delete_topic, TopicArn=f'arn:aws:sns:{REGION}:{acct}:{NAME}-alerts')
quiet(cb.delete_project, name='pbm-claims-build')
quiet(logs.delete_log_group, logGroupName='/aws/codebuild/pbm-claims-build')
quiet(ecr.delete_repository, repositoryName=NAME, force=True)
for r in ['pbm-claims-codebuild', 'pbm-claims-lambda', 'pbm-claims-scheduler']:
    try:
        for p in iam.list_role_policies(RoleName=r)['PolicyNames']:
            quiet(iam.delete_role_policy, RoleName=r, PolicyName=p)
        for p in iam.list_attached_role_policies(RoleName=r)['AttachedPolicies']:
            quiet(iam.detach_role_policy, RoleName=r, PolicyArn=p['PolicyArn'])
    except ClientError as e:
        print('skip role', r, e.response['Error']['Code'])
    quiet(iam.delete_role, RoleName=r)
b = boto3.resource('s3', region_name=REGION).Bucket(f'{NAME}-{acct}')
try:
    b.object_versions.delete()
    b.objects.all().delete()
    b.delete()
    print('deleted bucket', b.name)
except ClientError as e:
    print('skip bucket', e.response['Error']['Code'])
