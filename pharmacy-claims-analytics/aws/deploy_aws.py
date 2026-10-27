"""Deploy the scoring service to AWS Lambda with CloudWatch alarms (boto3, idempotent).

Creates, in us-east-1:
  S3 bucket (build source + validation reference), ECR repo, CodeBuild project + role,
  Lambda function (container image + AWS Lambda Web Adapter) + role, IAM-auth Function URL,
  3 log metric filters on the service's alert-transition logs, 6 CloudWatch alarms, SNS topic.

Usage (AWS credentials with admin-level rights in the target account):
    python aws/deploy_aws.py                    # build + deploy + alarms
    python aws/deploy_aws.py --email you@x.com  # also subscribe an email to the SNS topic
Then: python aws/validate_lambda.py --soak-minutes 20
Remove everything: python aws/teardown_aws.py
"""
import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

REGION = 'us-east-1'
NAME = 'pbm-claims-scoring'
ROOT = Path(__file__).resolve().parents[1]

BUILDSPEC = """version: 0.2
phases:
  pre_build:
    commands:
      - sha256sum pbm/service.py pbm/formulary.py reports/model.joblib reports/monitoring_reference.json requirements.txt
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $REPO_URI
  build:
    commands:
      - export TAG=$(sha256sum pbm/service.py reports/model.joblib | sha256sum | cut -c1-12)
      - docker build --provenance=false -f aws/Dockerfile.lambda -t $REPO_URI:$TAG .
      - python3 -c "import json;json.dump({'note':'Real CMS formulary feature rows (CY2026 test year), features only','pairs':json.load(open('reports/reference_pairs.json'))},open('/tmp/reference_pairs.json','w'))"
      - aws s3 cp /tmp/reference_pairs.json s3://$BUCKET/validation/reference_pairs.json
  post_build:
    commands:
      - docker push $REPO_URI:$TAG
      - echo "IMAGE=$REPO_URI:$TAG"
"""


def ensure(fn, *args, ignore=('AlreadyExists', 'EntityAlreadyExists', 'ResourceConflict',
                              'RepositoryAlreadyExists', 'BucketAlreadyOwnedByYou', 'ResourceAlreadyExists'), **kw):
    try:
        return fn(*args, **kw)
    except ClientError as e:
        if any(code in e.response['Error']['Code'] for code in ignore):
            return None
        raise


def role(iam, name, service, policy=None, managed=None):
    trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': service},
                                                     'Action': 'sts:AssumeRole'}]}
    ensure(iam.create_role, RoleName=name, AssumeRolePolicyDocument=json.dumps(trust))
    if policy:
        iam.put_role_policy(RoleName=name, PolicyName=name, PolicyDocument=json.dumps(policy))
    if managed:
        iam.attach_role_policy(RoleName=name, PolicyArn=managed)
    return iam.get_role(RoleName=name)['Role']['Arn']


def source_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in ['requirements.txt', 'aws/Dockerfile.lambda', 'aws/export_reference.py',
                    *[str(p.relative_to(ROOT)) for p in (ROOT / 'pbm').glob('*.py')],
                    *[str(p.relative_to(ROOT)) for p in (ROOT / 'sql').glob('*.sql')]]:
            z.write(ROOT / rel, rel)
        z.writestr('buildspec.yml', BUILDSPEC)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--email', default='')
    ap.add_argument('--canary-hours', type=float, default=6, help='0 disables the scheduled canary')
    args = ap.parse_args()
    s = boto3.Session(region_name=REGION)
    acct = s.client('sts').get_caller_identity()['Account']
    bucket = f'{NAME}-{acct}'
    s3, ecr, iam, cb = s.client('s3'), s.client('ecr'), s.client('iam'), s.client('codebuild')
    lam, logs, cw, sns = s.client('lambda'), s.client('logs'), s.client('cloudwatch'), s.client('sns')

    ensure(s3.create_bucket, Bucket=bucket)
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        'BlockPublicAcls': True, 'IgnorePublicAcls': True, 'BlockPublicPolicy': True, 'RestrictPublicBuckets': True})
    ensure(ecr.create_repository, repositoryName=NAME, imageScanningConfiguration={'scanOnPush': True})
    repo_uri = f'{acct}.dkr.ecr.{REGION}.amazonaws.com/{NAME}'

    cb_role = role(iam, 'pbm-claims-codebuild', 'codebuild.amazonaws.com', policy={'Version': '2012-10-17', 'Statement': [
        {'Effect': 'Allow', 'Action': ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
         'Resource': f'arn:aws:logs:{REGION}:{acct}:log-group:/aws/codebuild/pbm-claims-build*'},
        {'Effect': 'Allow', 'Action': ['s3:GetObject', 's3:GetObjectVersion'], 'Resource': f'arn:aws:s3:::{bucket}/build/*'},
        {'Effect': 'Allow', 'Action': ['s3:PutObject'], 'Resource': f'arn:aws:s3:::{bucket}/validation/*'},
        {'Effect': 'Allow', 'Action': ['ecr:GetAuthorizationToken'], 'Resource': '*'},
        {'Effect': 'Allow', 'Action': ['ecr:BatchCheckLayerAvailability', 'ecr:InitiateLayerUpload', 'ecr:UploadLayerPart',
                                       'ecr:CompleteLayerUpload', 'ecr:PutImage', 'ecr:BatchGetImage',
                                       'ecr:GetDownloadUrlForLayer'],
         'Resource': f'arn:aws:ecr:{REGION}:{acct}:repository/{NAME}'}]})
    fn_role = role(iam, 'pbm-claims-lambda', 'lambda.amazonaws.com',
                   managed='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    time.sleep(10)  # IAM propagation

    s3.put_object(Bucket=bucket, Key='build/source.zip', Body=source_zip())
    project = {'name': 'pbm-claims-build', 'source': {'type': 'S3', 'location': f'{bucket}/build/source.zip'},
               'artifacts': {'type': 'NO_ARTIFACTS'}, 'serviceRole': cb_role, 'timeoutInMinutes': 30,
               'environment': {'type': 'LINUX_CONTAINER', 'image': 'aws/codebuild/standard:7.0',
                               'computeType': 'BUILD_GENERAL1_MEDIUM', 'privilegedMode': True,
                               'environmentVariables': [{'name': 'REPO_URI', 'value': repo_uri},
                                                        {'name': 'BUCKET', 'value': bucket}]},
               'logsConfig': {'cloudWatchLogs': {'status': 'ENABLED', 'groupName': '/aws/codebuild/pbm-claims-build'}}}
    try:
        cb.create_project(**project)
    except ClientError:
        cb.update_project(**project)
    build_id = cb.start_build(projectName='pbm-claims-build')['build']['id']
    print('Building', build_id)
    while (b := cb.batch_get_builds(ids=[build_id])['builds'][0])['buildStatus'] == 'IN_PROGRESS':
        time.sleep(15)
    if b['buildStatus'] != 'SUCCEEDED':
        raise SystemExit(f'Build {b["buildStatus"]}: see CloudWatch log group /aws/codebuild/pbm-claims-build')
    newest = max(ecr.describe_images(repositoryName=NAME)['imageDetails'], key=lambda d: d['imagePushedAt'])
    image = f'{repo_uri}@{newest["imageDigest"]}'

    try:
        lam.create_function(FunctionName=NAME, PackageType='Image', Code={'ImageUri': image}, Role=fn_role,
                            MemorySize=2048, Timeout=30, Architectures=['x86_64'],
                            Description='Prior-authorization forecast scoring service (portfolio demo, CMS formulary data)')
    except ClientError as e:
        if 'ResourceConflict' not in e.response['Error']['Code']:
            raise
        lam.update_function_code(FunctionName=NAME, ImageUri=image)
    lam.get_waiter('function_updated_v2').wait(FunctionName=NAME)
    ensure(lam.create_function_url_config, FunctionName=NAME, AuthType='AWS_IAM')
    url = lam.get_function_url_config(FunctionName=NAME)['FunctionUrl']

    log_group = f'/aws/lambda/{NAME}'
    ensure(logs.create_log_group, logGroupName=log_group)
    logs.put_retention_policy(logGroupName=log_group, retentionInDays=30)
    topic = sns.create_topic(Name=f'{NAME}-alerts')['TopicArn']
    if args.email:
        sns.subscribe(TopicArn=topic, Protocol='email', Endpoint=args.email)
        print('Confirm the SNS subscription email sent to', args.email)
    rules = {'score_drift': ('ScoreDriftAlertRaised', 'PSI of last 1,000 scores vs validation > 0.20'),
             'request_failure_rate': ('RequestFailureRateAlertRaised', '4xx+5xx share of last 100 requests > 5%'),
             'latency': ('LatencyAlertRaised', 'p95 server processing latency of last 100 requests > 2,000 ms')}
    for alert, (metric, rule) in rules.items():
        logs.put_metric_filter(logGroupName=log_group, filterName=f'pbm-{alert}-raised',
                               filterPattern='{ ($.event = "pbm_alert_transition") && ($.alert = "%s") && ($.active IS TRUE) }' % alert,
                               metricTransformations=[{'metricName': metric, 'metricNamespace': 'PBMScoring',
                                                       'metricValue': '1', 'unit': 'Count'}])
        cw.put_metric_alarm(AlarmName=f'pbm-scoring-{alert}', AlarmDescription=f'Service raised {alert}: {rule}',
                            Namespace='PBMScoring', MetricName=metric, Statistic='Sum', Period=60,
                            EvaluationPeriods=1, Threshold=1, ComparisonOperator='GreaterThanOrEqualToThreshold',
                            TreatMissingData='notBreaching', AlarmActions=[topic], OKActions=[topic])
    for name, metric, stat, threshold in [('lambda-errors', 'Errors', 'Sum', 0), ('lambda-throttles', 'Throttles', 'Sum', 0),
                                          ('lambda-duration-p95', 'Duration', 'p95', 2000)]:
        p = dict(AlarmName=f'pbm-scoring-{name}', Namespace='AWS/Lambda', MetricName=metric,
                 Dimensions=[{'Name': 'FunctionName', 'Value': NAME}], Period=300, EvaluationPeriods=1,
                 Threshold=threshold, ComparisonOperator='GreaterThanThreshold', TreatMissingData='notBreaching',
                 AlarmActions=[topic], OKActions=[topic])
        p['ExtendedStatistic' if stat.startswith('p') else 'Statistic'] = stat
        cw.put_metric_alarm(**p)
    if args.canary_hours > 0:
        create_canary(s, acct, bucket, args.canary_hours)
    print(json.dumps({'image': image, 'function_url (AWS_IAM)': url, 'sns_topic': topic}, indent=2))


def create_canary(s, acct, bucket, hours):
    """EventBridge Scheduler: one 100-pair /predict request per minute, auto-ending.

    Replays real reference pairs 900-999 (check their PSI vs. validation before relying on the
    canary staying under the 0.20 drift threshold).
    """
    import datetime
    fn_arn = f'arn:aws:lambda:{REGION}:{acct}:function:{NAME}'
    iam, sch = s.client('iam'), s.client('scheduler')
    trust = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': 'scheduler.amazonaws.com'},
                                                     'Action': 'sts:AssumeRole',
                                                     'Condition': {'StringEquals': {'aws:SourceAccount': acct}}}]}
    ensure(iam.create_role, RoleName='pbm-claims-scheduler', AssumeRolePolicyDocument=json.dumps(trust))
    iam.put_role_policy(RoleName='pbm-claims-scheduler', PolicyName='invoke-scorer', PolicyDocument=json.dumps(
        {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'lambda:InvokeFunction',
                                                 'Resource': [fn_arn, fn_arn + ':*']}]}))
    time.sleep(10)
    ref = json.loads(s.client('s3').get_object(Bucket=bucket, Key='validation/reference_pairs.json')['Body'].read())
    event = {'version': '2.0', 'routeKey': '$default', 'rawPath': '/predict', 'rawQueryString': '',
             'headers': {'content-type': 'application/json', 'host': 'scheduler-canary'},
             'requestContext': {'accountId': 'anonymous', 'apiId': 'canary', 'domainName': 'scheduler-canary',
                                'domainPrefix': 'canary', 'requestId': 'canary', 'routeKey': '$default',
                                'stage': '$default', 'time': '', 'timeEpoch': 0,
                                'http': {'method': 'POST', 'path': '/predict', 'protocol': 'HTTP/1.1',
                                         'sourceIp': '127.0.0.1', 'userAgent': 'pbm-scheduler-canary'}},
             'body': json.dumps({'pairs': ref['pairs'][900:1000]}), 'isBase64Encoded': False}
    params = dict(Name='pbm-claims-canary', ScheduleExpression='rate(1 minute)', FlexibleTimeWindow={'Mode': 'OFF'},
                  EndDate=datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=hours),
                  Target={'Arn': fn_arn, 'RoleArn': f'arn:aws:iam::{acct}:role/pbm-claims-scheduler',
                          'Input': json.dumps(event), 'RetryPolicy': {'MaximumRetryAttempts': 0}}, State='ENABLED')
    try:
        sch.create_schedule(**params)
    except ClientError:
        sch.update_schedule(**params)
    print(f'Canary scheduled for {hours} h')


if __name__ == '__main__':
    main()
