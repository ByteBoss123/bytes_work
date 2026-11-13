"""Generate the three Kaggle notebooks (GPU T4 x2, Internet ON)."""
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/ByteBoss123/speaker-aware-speech-llm.git"
SETUP = f'''# Settings: GPU T4 x2, Internet ON. Persistent outputs: results/ and runs/ in /kaggle/working.
import os, shutil, subprocess
REPO_URL = "{REPO}"
WORK = "/kaggle/working/speaker-aware-speech-llm"
if not os.path.exists(WORK):
    src = "/kaggle/input/speaker-aware-speech-llm"   # alternative: upload the repo as a Kaggle dataset
    if os.path.exists(src):
        shutil.copytree(src, WORK)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, WORK], check=True)
os.chdir(WORK)
DATA = "/tmp/data"   # large ephemeral disk; /kaggle/working is capped at ~20 GB
os.environ["DATA"] = DATA'''
INSTALL = "!pip install -q -r requirements.txt\n!nvidia-smi --query-gpu=name,memory.total --format=csv\n!df -h /tmp /kaggle/working | tail -2"
SUMMARY = "!python scripts/summarize.py results runs\n!cd /kaggle/working && zip -qr results_{tag}.zip speaker-aware-speech-llm/results speaker-aware-speech-llm/runs -x '*.npz'"


def nb(title, cells, tag):
    n = nbf.v4.new_notebook()
    n.cells = [nbf.v4.new_markdown_cell(title), nbf.v4.new_code_cell(SETUP), nbf.v4.new_code_cell(INSTALL)]
    for c in cells:
        n.cells.append(nbf.v4.new_markdown_cell(c[2:]) if c.startswith("# ") and "\n" not in c else nbf.v4.new_code_cell(c))
    n.cells.append(nbf.v4.new_code_cell(SUMMARY.format(tag=tag)))
    n.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                  "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True}}
    return n


SLURP = "--slurp_dir $DATA/slurp_repo/dataset/slurp"
AUDIO = "--audio_dir $DATA/slurp_audio/slurp_real"
CACHE = "--cache_dir $DATA/slurp_cache"
nb1 = nb("# 01 Speaker verification on far-field voice-assistant audio (SLURP) + VoxCeleb1-O check", [
    "!bash scripts/prepare_data.sh $DATA slurp vox1test",
    "# Pipeline check: the pretrained model card reports 0.80% EER on VoxCeleb1-O (cleaned)",
    "!python scripts/sv_vox.py --vox_root $DATA/vox1 --out results/sv_vox1o.json",
    "# Speaker-disjoint split (the official SLURP splits share speakers)",
    f"!python scripts/sv_slurp.py split {SLURP} --out_dir results/sv",
    f"!python scripts/sv_slurp.py eval {SLURP} {AUDIO} --out_dir results/sv --tag pretrained",
    f"!python scripts/sv_slurp.py noise {SLURP} {AUDIO} --out_dir results/sv --tag pretrained",
    "# DDP throughput: same job on 1 and 2 GPUs (300 steps each, not saved)",
    f"!torchrun --nproc_per_node 1 scripts/sv_finetune_ddp.py {SLURP} {AUDIO} {CACHE} --max_steps 300 --save 0 --tag ws1\n"
    f"!torchrun --nproc_per_node 2 scripts/sv_finetune_ddp.py {SLURP} {AUDIO} {CACHE} --max_steps 300 --save 0 --tag ws2",
    "# Full multi-condition fine-tune on train-split speakers (2 GPUs)",
    f"!torchrun --nproc_per_node 2 scripts/sv_finetune_ddp.py {SLURP} {AUDIO} {CACHE} --epochs 2 --tag main",
    f"!python scripts/sv_slurp.py eval {SLURP} {AUDIO} --out_dir results/sv --ckpt runs/ecapa_ft/embedding_model_main.pt --tag finetuned\n"
    f"!python scripts/sv_slurp.py noise {SLURP} {AUDIO} --out_dir results/sv --ckpt runs/ecapa_ft/embedding_model_main.pt --tag finetuned",
    "# Did fine-tuning hurt out-of-domain verification?",
    "!python scripts/sv_vox.py --vox_root $DATA/vox1 --ckpt runs/ecapa_ft/embedding_model_main.pt --out results/sv_vox1o_finetuned.json",
], "sv")

nb2 = nb("# 02 Speech LLM: Whisper-small encoder + Qwen2.5-0.5B (LoRA), ASR + SLU from audio", [
    "!bash scripts/prepare_data.sh $DATA slurp libri_test libri_train100",
    "# Whisper-small baseline (2 GPUs in parallel shards)",
    "!python scripts/asr_whisper.py --data_root $DATA --set librispeech_test_clean --out results/asr",
    "!CUDA_VISIBLE_DEVICES=0 python scripts/asr_whisper.py --data_root $DATA --set slurp_test --shard 0/2 --out results/asr > /tmp/a0.log 2>&1 & "
    "CUDA_VISIBLE_DEVICES=1 python scripts/asr_whisper.py --data_root $DATA --set slurp_test --shard 1/2 --out results/asr > /tmp/a1.log 2>&1; wait\n"
    "!tail -3 /tmp/a0.log /tmp/a1.log\n!python scripts/asr_whisper.py --data_root $DATA --set slurp_test --merge --out results/asr",
    "# Train (DDP, 2 GPUs). Raise MAX_STEPS if the throughput printed in the log leaves time in the session.",
    "MAX_STEPS = 3000\n!torchrun --nproc_per_node 2 scripts/sllm_train.py --data_root $DATA --out_dir runs/sllm --max_steps {MAX_STEPS}",
    "!python scripts/sllm_eval.py --data_root $DATA --run_dir runs/sllm --out results/sllm",
    "# GRPO directly on the speech LLM (audio -> JSON), reference = merged SFT model",
    "!python scripts/sllm_grpo.py --data_root $DATA --sft_dir runs/sllm --out_dir runs/sllm_grpo --steps 200",
    "!python scripts/sllm_eval.py --data_root $DATA --run_dir runs/sllm --grpo_dir runs/sllm_grpo --out results/sllm_grpo",
], "sllm")

nb3 = nb("# 03 Cascade SLU on real Whisper transcripts: SFT then GRPO (verifiable reward)", [
    "!bash scripts/prepare_data.sh $DATA slurp",
    "# Whisper-small transcripts: 24k train recordings, 2k devel, full test (2-GPU shards)",
    "%%bash\nfor S in slurp_train slurp_devel slurp_test; do\n  M=0; [ $S = slurp_train ] && M=24000; [ $S = slurp_devel ] && M=2000\n"
    "  CUDA_VISIBLE_DEVICES=0 python scripts/asr_whisper.py --data_root $DATA --set $S --max_items $M --shard 0/2 --out results/asr > /tmp/$S.0.log 2>&1 &\n"
    "  CUDA_VISIBLE_DEVICES=1 python scripts/asr_whisper.py --data_root $DATA --set $S --max_items $M --shard 1/2 --out results/asr > /tmp/$S.1.log 2>&1\n"
    "  wait\n  python scripts/asr_whisper.py --data_root $DATA --set $S --merge --out results/asr | tail -4\ndone",
    "!python scripts/slu_text.py sft --data_root $DATA --asr_dir results/asr --run_dir runs/slu",
    "!python scripts/slu_text.py grpo --data_root $DATA --asr_dir results/asr --run_dir runs/slu --grpo_steps 300",
    "!python scripts/slu_text.py eval --data_root $DATA --asr_dir results/asr --run_dir runs/slu --out results/slu",
], "slu")

nb4 = nb("# 04 Device-directed speech detection with leave-corpus-out evaluation and shortcut probes", [
    "!bash scripts/prepare_data.sh $DATA slurp vox1test",
    "# Export SLURP + MInDS-14 (directed) and AMI + VoxCeleb1 (non-directed) to 16 kHz wav",
    "!python scripts/ddsd_prepare.py --data_root $DATA --out $DATA/ddsd",
    "!python scripts/ddsd_features.py --ddsd_dir $DATA/ddsd --out results/ddsd",
    "# A model only counts as learning directedness where it beats the channel-only and length-only probes cross-corpus",
    "!python scripts/ddsd_eval.py --feat_dir results/ddsd",
], "ddsd")

SIZES = [("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", 8, 2), ("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", 8, 2),
         ("3B", "Qwen/Qwen2.5-3B-Instruct", 4, 4)]
cells5 = ["!bash scripts/prepare_data.sh $DATA slurp libri_test libri_train100",
          "# Same data, same optimizer steps, same global batch (32) for every LLM size; only the LLM changes",
          "STEPS = 1500"]
for tag, name, b, acc in SIZES:
    cells5.append(f"!torchrun --nproc_per_node 2 scripts/sllm_train.py --data_root $DATA --llm {name} "
                  f"--batch {b} --grad_accum {acc} --max_steps {{STEPS}} --out_dir runs/sllm_scale_{tag}\n"
                  f"!python scripts/sllm_eval.py --data_root $DATA --llm {name} --run_dir runs/sllm_scale_{tag} "
                  f"--max_per_set 2000 --out results/sllm_scale_{tag}")
nb5 = nb("# 05 LLM-size scaling study: 0.5B vs 1.5B vs 3B speech LLM (DDP, 2x T4)", cells5, "scale")

out = Path(__file__).resolve().parents[1] / "notebooks"
out.mkdir(exist_ok=True)
for name, n in (("01_speaker_verification.ipynb", nb1), ("02_speech_llm.ipynb", nb2), ("03_slu_grpo.ipynb", nb3),
                ("04_directed_speech.ipynb", nb4), ("05_llm_scaling.ipynb", nb5)):
    nbf.write(n, out / name)
    print("wrote", name)
