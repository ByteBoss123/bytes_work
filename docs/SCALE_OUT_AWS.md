# Scaling past Kaggle: 4–8 GPUs on AWS (optional, costs money)

Every training script already runs under `torchrun` with DDP, so the same code scales from 1 to N GPUs on one machine. Kaggle stops at 2x T4. This runbook repeats the scaling measurements on a bigger EC2 GPU instance in your own AWS account.

**Nothing here runs automatically.** Before launching anything:
* check the instance's current on-demand price and your GPU service quota (G or P instances) in the target region;
* terminate the instance when you're done.

## 1. Instance

| Instance | GPUs | Use |
|---|---|---|
| g5.12xlarge | 4x A10G 24 GB | 1→2→4 GPU scaling curve, 3B LLM with larger batches |
| g5.48xlarge | 8x A10G 24 GB | 1→8 GPU scaling curve |

Use an AWS Deep Learning AMI (PyTorch, Ubuntu) and at least 300 GB of gp3 storage. A10G GPUs support bf16, but the scripts use fp16 autocast, so results stay comparable with the T4 runs.

## 2. Setup on the instance

```bash
git clone https://github.com/ByteBoss123/speaker-aware-speech-llm.git && cd speaker-aware-speech-llm
pip install -r requirements.txt
export DATA=/opt/dlami/nvme/data   # local NVMe if present, else any large volume
bash scripts/prepare_data.sh $DATA slurp vox1test libri_test libri_train100
python scripts/sv_slurp.py split --slurp_dir $DATA/slurp_repo/dataset/slurp --out_dir results/sv
```

## 3. Throughput scaling (same job, more GPUs)

```bash
S="--slurp_dir $DATA/slurp_repo/dataset/slurp --audio_dir $DATA/slurp_audio/slurp_real --cache_dir $DATA/slurp_cache"
for N in 1 2 4 8; do   # drop 8 on g5.12xlarge
  torchrun --nproc_per_node $N scripts/sv_finetune_ddp.py $S --max_steps 300 --save 0 --tag ws$N
  torchrun --nproc_per_node $N scripts/sllm_train.py --data_root $DATA --max_steps 100 \
           --batch 8 --grad_accum 1 --out_dir runs/sllm_ws$N
done
```

Scaling efficiency at N GPUs is `samples_per_sec(N) / (N * samples_per_sec(1))`. Read it from `runs/ecapa_ft/train_stats_wsN.json` and `runs/sllm_wsN/train_stats.json`. The speaker run also logs `data_wait_fraction`. If that fraction grows with N, the input pipeline is the limit, not the GPUs.

## 4. Bigger models

The 3B run in `notebooks/05_llm_scaling.ipynb` works unchanged with `--nproc_per_node 4`. With 24 GB per GPU you can raise `--batch` and lower `--grad_accum` to keep the global batch at 32, so the comparison with the T4 runs stays fair.
