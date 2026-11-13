#!/usr/bin/env bash
# EC2 user-data for an AWS Deep Learning AMI (Ubuntu, PyTorch). Placeholders
# __CODE_S3__, __RESULTS_S3__, __MAX_HOURS__ are filled in at launch.
# The instance must be launched with InstanceInitiatedShutdownBehavior=terminate
# and an instance profile that can read the code object and write RESULTS_S3.
set -uxo pipefail
CODE_S3="__CODE_S3__"; RESULTS_S3="__RESULTS_S3__"; MAX_HOURS="__MAX_HOURS__"
exec > /var/log/speech-run.log 2>&1
# hard cost cap: power off (=> terminate) after MAX_HOURS no matter what
shutdown -h +$(( MAX_HOURS * 60 ))
for f in /opt/pytorch/bin/activate /opt/conda/bin/activate; do [ -f "$f" ] && source "$f" && break; done
WORK=/opt/dlami/nvme; [ -d "$WORK" ] || WORK=/home/ubuntu
mkdir -p $WORK/data && cd $WORK
aws s3 cp "$CODE_S3" code.zip && unzip -q -o code.zip
cd speaker-aware-speech-llm
pip install -q -r requirements.txt
nvidia-smi
DATA=$WORK/data RESULTS_S3="$RESULTS_S3" bash scripts/run_all.sh
aws s3 cp /var/log/speech-run.log "$RESULTS_S3/logs/bootstrap.log"
shutdown -h now
