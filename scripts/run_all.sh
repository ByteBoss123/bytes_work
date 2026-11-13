#!/usr/bin/env bash
# Run every pipeline on one multi-GPU Linux machine (e.g. an EC2 g5 instance).
# Mirrors notebooks/01-05. Each stage writes a .done marker, so a re-run skips
# finished stages; results are synced to S3 after every stage when
# RESULTS_S3 is set (e.g. s3://my-bucket/speech-run1).
#
#   DATA=/opt/dlami/nvme/data RESULTS_S3=s3://bucket/run bash scripts/run_all.sh [stage ...]
# Stages: sv sllm slu ddsd scale  (default: all, in that order)
set -uo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:?set DATA to a large local disk path}
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || { echo "no GPUs found (set NGPU to override)"; exit 1; }
STAGES=${*:-${RUN_STAGES:-"sv sllm slu ddsd scale"}}
SLLM_STEPS=${SLLM_STEPS:-3000}
SCALE_STEPS=${SCALE_STEPS:-1500}
# caps for a smoke test (empty = full run)
SV_FT_ARGS=${SV_FT_ARGS:---epochs 2}          # e.g. "--max_steps 30"
SV_MAX_TEST=${SV_MAX_TEST:-}                  # max test recordings per speaker
WS_STEPS=${WS_STEPS:-300}
EVAL_MAX=${EVAL_MAX:-0}                       # speech-LLM eval cap per set (0 = full)
SCALE_EVAL_MAX=${SCALE_EVAL_MAX:-2000}
SLLM_GRPO_STEPS=${SLLM_GRPO_STEPS:-200}
ASR_TRAIN_MAX=${ASR_TRAIN_MAX:-24000}
ASR_TEST_MAX=${ASR_TEST_MAX:-0}
SLU_ARGS=${SLU_ARGS:-}                        # e.g. "--sft_epochs 0.05"
SLU_GRPO_STEPS=${SLU_GRPO_STEPS:-300}
SLU_TEST_MAX=${SLU_TEST_MAX:-0}
DDSD_MAX=${DDSD_MAX:-5000}
LIBRI_MAX=${LIBRI_MAX:-20000}
SCALE_SIZES=${SCALE_SIZES:-"0.5B 1.5B 3B"}
DATASETS=${DATASETS:-"slurp vox1test libri_test libri_train100"}
GLOBAL_BATCH=32
mkdir -p logs results runs
export DATA

sync() {
  [ -n "${RESULTS_S3:-}" ] || return 0
  aws s3 sync results "$RESULTS_S3/results" --exclude "*.npz" --only-show-errors
  aws s3 sync runs "$RESULTS_S3/runs" --only-show-errors
  aws s3 sync logs "$RESULTS_S3/logs" --only-show-errors
}
stage() {  # stage NAME CMD...
  local name=$1; shift
  if [ -f "logs/$name.done" ]; then echo "skip $name (done)"; return 0; fi
  echo "=== $name  $(date -u +%FT%TZ)"
  if [ -n "${DRY_RUN:-}" ]; then echo "  $*" | tr -s ' '; return 0; fi
  if bash -euo pipefail -c "$*" 2>&1 | tee "logs/$name.log" | sed -u "s/^/[$name] /"; then
    touch "logs/$name.done"; echo "ok $name"
  else
    echo "FAILED $name (see logs/$name.log)"; tail -20 "logs/$name.log"
  fi
  sync
}
accum() { echo $(( GLOBAL_BATCH / ($1 * NGPU) > 0 ? GLOBAL_BATCH / ($1 * NGPU) : 1 )); }

S="--slurp_dir $DATA/slurp_repo/dataset/slurp"
A="--audio_dir $DATA/slurp_audio/slurp_real"
C="--cache_dir $DATA/slurp_cache"
TR="torchrun --nproc_per_node $NGPU"

stage data "bash scripts/prepare_data.sh $DATA $DATASETS"

for st in $STAGES; do case $st in
sv)
  stage sv_vox "[ ! -d $DATA/vox1/wav ] && echo VoxCeleb1 unavailable, skipped || python scripts/sv_vox.py --vox_root $DATA/vox1 --out results/sv_vox1o.json"
  stage sv_split "python scripts/sv_slurp.py split $S --out_dir results/sv"
  stage sv_eval "python scripts/sv_slurp.py eval $S $A --out_dir results/sv --tag pretrained $([ -n "$SV_MAX_TEST" ] && echo --max_test_per_speaker $SV_MAX_TEST) && \
                 python scripts/sv_slurp.py noise $S $A --out_dir results/sv --tag pretrained $([ -n "$SV_MAX_TEST" ] && echo --max_test_per_speaker $SV_MAX_TEST)"
  # throughput at 1, 2, 4, ... GPUs for the scaling-efficiency curve
  for n in 1 2 4 8; do
    [ "$n" -le "$NGPU" ] && stage sv_ws$n "torchrun --nproc_per_node $n scripts/sv_finetune_ddp.py $S $A $C --max_steps $WS_STEPS --save 0 --tag ws$n"
  done
  stage sv_ft "$TR scripts/sv_finetune_ddp.py $S $A $C $SV_FT_ARGS --tag main"
  stage sv_ft_eval "python scripts/sv_slurp.py eval $S $A --out_dir results/sv --ckpt runs/ecapa_ft/embedding_model_main.pt --tag finetuned $([ -n "$SV_MAX_TEST" ] && echo --max_test_per_speaker $SV_MAX_TEST) && \
                    python scripts/sv_slurp.py noise $S $A --out_dir results/sv --ckpt runs/ecapa_ft/embedding_model_main.pt --tag finetuned $([ -n "$SV_MAX_TEST" ] && echo --max_test_per_speaker $SV_MAX_TEST) && \
                    { [ ! -d $DATA/vox1/wav ] || python scripts/sv_vox.py --vox_root $DATA/vox1 --ckpt runs/ecapa_ft/embedding_model_main.pt --out results/sv_vox1o_finetuned.json; }"
  ;;
sllm)
  stage asr_libri "python scripts/asr_whisper.py --data_root $DATA --set librispeech_test_clean --out results/asr"
  stage asr_slurp_test "python scripts/asr_whisper.py --data_root $DATA --set slurp_test --max_items $ASR_TEST_MAX --out results/asr"
  stage sllm_train "$TR scripts/sllm_train.py --data_root $DATA --out_dir runs/sllm --batch 8 --grad_accum $(accum 8) --max_steps $SLLM_STEPS --libri_max $LIBRI_MAX"
  stage sllm_eval "python scripts/sllm_eval.py --data_root $DATA --run_dir runs/sllm --max_per_set $EVAL_MAX --out results/sllm"
  stage sllm_grpo "python scripts/sllm_grpo.py --data_root $DATA --sft_dir runs/sllm --out_dir runs/sllm_grpo --steps $SLLM_GRPO_STEPS"
  stage sllm_grpo_eval "python scripts/sllm_eval.py --data_root $DATA --run_dir runs/sllm --grpo_dir runs/sllm_grpo --max_per_set $EVAL_MAX --out results/sllm_grpo"
  ;;
slu)
  stage asr_slurp_train "python scripts/asr_whisper.py --data_root $DATA --set slurp_train --max_items $ASR_TRAIN_MAX --out results/asr"
  stage asr_slurp_devel "python scripts/asr_whisper.py --data_root $DATA --set slurp_devel --max_items 2000 --out results/asr"
  stage asr_slurp_test2 "test -f results/asr/hyp_slurp_test.jsonl || python scripts/asr_whisper.py --data_root $DATA --set slurp_test --max_items $ASR_TEST_MAX --out results/asr"
  stage slu_sft "python scripts/slu_text.py sft --data_root $DATA --asr_dir results/asr --run_dir runs/slu $SLU_ARGS"
  stage slu_grpo "python scripts/slu_text.py grpo --data_root $DATA --asr_dir results/asr --run_dir runs/slu --grpo_steps $SLU_GRPO_STEPS"
  stage slu_eval "python scripts/slu_text.py eval --data_root $DATA --asr_dir results/asr --run_dir runs/slu --max_test $SLU_TEST_MAX --out results/slu"
  ;;
ddsd)
  stage ddsd_prepare "python scripts/ddsd_prepare.py --data_root $DATA --out $DATA/ddsd --max_per_corpus $DDSD_MAX"
  stage ddsd_features "python scripts/ddsd_features.py --ddsd_dir $DATA/ddsd --out results/ddsd"
  stage ddsd_eval "python scripts/ddsd_eval.py --feat_dir results/ddsd"
  stage ddsd_jax "python scripts/ddsd_jax.py --feat_dir results/ddsd"
  ;;
multi)
  stage multi_prepare "python scripts/multi_slu.py prepare --data $DATA/smassive --train_per_lang ${MULTI_TRAIN_PER_LANG:-6000} --eval_per_lang ${MULTI_EVAL_PER_LANG:-500}"
  stage multi_train "python scripts/multi_slu.py train --data $DATA/smassive --run_dir runs/multi --steps ${MULTI_STEPS:-1500}"
  stage multi_eval "python scripts/multi_slu.py eval --data $DATA/smassive --run_dir runs/multi --out results/multi"
  ;;
scale)
  for spec in "0.5B Qwen/Qwen2.5-0.5B-Instruct 8" "1.5B Qwen/Qwen2.5-1.5B-Instruct 8" "3B Qwen/Qwen2.5-3B-Instruct 4"; do
    set -- $spec
    [[ " $SCALE_SIZES " == *" $1 "* ]] || continue
    stage scale_$1 "$TR scripts/sllm_train.py --data_root $DATA --llm $2 --batch $3 --grad_accum $(accum $3) \
                    --max_steps $SCALE_STEPS --libri_max $LIBRI_MAX --out_dir runs/sllm_scale_$1 && \
                    python scripts/sllm_eval.py --data_root $DATA --llm $2 --run_dir runs/sllm_scale_$1 \
                    --max_per_set $SCALE_EVAL_MAX --out results/sllm_scale_$1"
  done
  ;;
*) echo "unknown stage $st" ;;
esac; done

python scripts/summarize.py results runs > logs/summary.txt 2>&1 || true
sync
echo "all stages finished $(date -u +%FT%TZ)"
ls logs/*.done | sed 's#logs/##' | tr '\n' ' '; echo
