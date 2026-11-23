#!/usr/bin/env bash
# All training runs of the paper, in order of priority.  Every step is resumable: finished runs
# are skipped, so the script can simply be started again after an interruption.  A failing step is
# recorded in outputs/logs/train/pipeline.log and the pipeline moves on.
#   usage: scripts/pipeline_train.sh [STAGE ...]     (default: all stages)
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
TAG=${LLM_TAG:-qwen25}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/train
mkdir -p $LOG
STAGES=${*:-main controls lora ablation distill lodo extra live}
DOMAINS="社会生活 财经商业 医药健康 灾难事故 文体娱乐 政治"

note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }
# step LOGFILE CMD...  : run a command, append its output to LOGFILE, record failures
step() {
  local lf=$1; shift
  "$@" >> "$LOG/$lf" 2>&1 || note "FAILED ($lf): $*"
}
run() { local lf=$1; shift; step "$lf" $PY scripts/run.py --amp --perturb "$@"; }
sel() { $PY scripts/selected_lr.py "$@" || { note "FAILED selected_lr $*"; return 1; }; }

for STAGE in $STAGES; do
note "stage $STAGE start"
case $STAGE in
main)        # main comparison: the same learning-rate grid for every SLM-side method, 5 seeds
  for D in ARG-CN ARG-EN; do
    DS=outputs/datasets/$D.$TAG.json
    run main_$D.log --dataset $DS --sweep main    --seeds 1 2 3 4 5
    run main_$D.log --dataset $DS --sweep main_zs --seeds 1 2 3 4 5
  done ;;
controls)    # ARG with CLARA's mismatch negatives
  for D in ARG-CN ARG-EN; do
    LR=$(sel $D.$TAG arg) || continue
    run controls_$D.log --dataset outputs/datasets/$D.$TAG.json --sweep controls --lr $LR --seeds 1 2 3 4 5
  done ;;
lora)        # LoRA fine-tuned 7B classifier
  for D in ARG-CN ARG-EN; do
    EV=""
    for K in style tamper; do F=outputs/datasets/$D-$K.$TAG.json; [ -f $F ] && EV="$EV --eval $K=$F"; done
    F=outputs/datasets/$D-occl.none.json; [ -f $F ] && EV="$EV --eval occl=$F"
    for S in 1 2 3; do
      step lora_$D.log $PY scripts/lora_baseline.py --dataset outputs/datasets/$D.$TAG.json --seed $S --lr 1e-4 $EV
    done
  done ;;
ablation)    # CLARA ablations at the learning rate selected on validation (5 seeds zh, 3 seeds en)
  for D in ARG-CN ARG-EN; do
    LR=$(sel $D.$TAG clara) || continue
    SEEDS="1 2 3 4 5"; [ $D = ARG-EN ] && SEEDS="1 2 3"
    run ablation_$D.log --dataset outputs/datasets/$D.$TAG.json --sweep ablation --lr $LR --seeds $SEEDS
  done ;;
distill)     # LLM-free students
  for D in ARG-CN ARG-EN; do
    LR=$(sel $D.$TAG clara) || continue
    run distill_$D.log --dataset outputs/datasets/$D.$TAG.json --sweep distill --lr $LR --seeds 1 2 3
  done ;;
lodo)        # leave-one-domain-out on Weibo21, each method at its validation-selected learning rate
  NAMES=$(sel ARG-CN.$TAG --names bert clara arg) && ZS=$(sel ARG-CN.$TAG --names bert_zs) && \
  for DOM in $DOMAINS; do
    step lodo.log $PY scripts/run.py --amp --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main \
        --only $NAMES --holdout $DOM --seeds 1 2 3
    step lodo.log $PY scripts/run.py --amp --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main_zs \
        --only $ZS --holdout $DOM --seeds 1 2 3
  done ;;
extra)       # number of claims (both languages); sensitivity to the a-priori weights (Chinese)
  for D in ARG-CN ARG-EN; do
    LR=$(sel $D.$TAG clara) || continue
    run extra_$D.log --dataset outputs/datasets/$D.$TAG.json --sweep claims_k --lr $LR --seeds 1 2 3
  done
  LR=$(sel ARG-CN.$TAG clara) && \
  run extra_ARG-CN.log --dataset outputs/datasets/ARG-CN.$TAG.json --sweep sensitivity --lr $LR --seeds 1 2 3 ;;
live)        # protocol check: recompute the frozen layers every step (no cache, no mixed precision)
  NAMES=$(sel ARG-CN.$TAG --names bert arg_gpt clara) && \
  step live.log $PY scripts/run.py --live --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main \
      --only $NAMES --tag ARG-CN.$TAG-live --seeds 1 2 3 ;;
*) note "unknown stage $STAGE" ;;
esac
note "stage $STAGE done"
done
note "pipeline finished"
