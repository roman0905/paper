#!/usr/bin/env bash
# All training runs of the paper, in order of priority.  Every step is resumable: finished runs
# are skipped, so the script can simply be started again after an interruption.
#   usage: scripts/pipeline_train.sh [STAGE ...]     (default: all stages)
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
TAG=${LLM_TAG:-qwen25}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/train
mkdir -p $LOG
RUN="$PY scripts/run.py --amp --perturb"
STAGES=${*:-main lora ablation distill lodo extra live}
DOMAINS="社会生活 财经商业 医药健康 灾难事故 文体娱乐 政治"

for STAGE in $STAGES; do
case $STAGE in
main)        # main comparison: equal learning-rate grid for every SLM-side method, 5 seeds
  for D in ARG-CN ARG-EN; do
    DS=outputs/datasets/$D.$TAG.json
    $RUN --dataset $DS --sweep main    --seeds 1 2 3 4 5 >> $LOG/main_$D.log 2>&1
    $RUN --dataset $DS --sweep main_zs --seeds 1 2 3 4 5 >> $LOG/main_$D.log 2>&1
  done ;;
ablation)    # CLARA ablations at the learning rate selected on validation (5 seeds zh, 3 seeds en)
  for D in ARG-CN ARG-EN; do
    DS=outputs/datasets/$D.$TAG.json; LR=$($PY scripts/selected_lr.py $D.$TAG clara) || exit 1
    SEEDS="1 2 3 4 5"; [ $D = ARG-EN ] && SEEDS="1 2 3"
    $RUN --dataset $DS --sweep ablation --lr $LR --seeds $SEEDS >> $LOG/ablation_$D.log 2>&1
  done ;;
distill)     # LLM-free students
  for D in ARG-CN ARG-EN; do
    DS=outputs/datasets/$D.$TAG.json; LR=$($PY scripts/selected_lr.py $D.$TAG clara) || exit 1
    $RUN --dataset $DS --sweep distill --lr $LR --seeds 1 2 3 >> $LOG/distill_$D.log 2>&1
  done ;;
lora)        # LoRA fine-tuned 7B classifier
  for D in ARG-CN ARG-EN; do
    EV=""
    for K in style tamper; do F=outputs/datasets/$D-$K.$TAG.json; [ -f $F ] && EV="$EV --eval $K=$F"; done
    F=outputs/datasets/$D-occl.none.json; [ -f $F ] && EV="$EV --eval occl=$F"
    for S in 1 2 3; do
      $PY scripts/lora_baseline.py --dataset outputs/datasets/$D.$TAG.json --seed $S --lr 1e-4 $EV >> $LOG/lora_$D.log 2>&1
    done
  done ;;
extra)       # number of claims; loss-weight sensitivity (Chinese only)
  for D in ARG-CN ARG-EN; do
    DS=outputs/datasets/$D.$TAG.json; LR=$($PY scripts/selected_lr.py $D.$TAG clara) || exit 1
    $RUN --dataset $DS --sweep claims_k --lr $LR --seeds 1 2 3 >> $LOG/extra_$D.log 2>&1
  done
  LR=$($PY scripts/selected_lr.py ARG-CN.$TAG clara) || exit 1
  $RUN --dataset outputs/datasets/ARG-CN.$TAG.json --sweep sensitivity --lr $LR --seeds 1 2 3 >> $LOG/extra_ARG-CN.log 2>&1 ;;
lodo)        # leave-one-domain-out on Weibo21, each method at its validation-selected learning rate
  NAMES=$($PY scripts/selected_lr.py ARG-CN.$TAG --names bert clara arg factguard) || exit 1
  for DOM in $DOMAINS; do
    $PY scripts/run.py --amp --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main --only $NAMES \
        --holdout $DOM --seeds 1 2 3 >> $LOG/lodo.log 2>&1
  done ;;
live)        # protocol check: recompute the frozen layers every step (no cache, no AMP)
  NAMES=$($PY scripts/selected_lr.py ARG-CN.$TAG --names bert arg_gpt) || exit 1
  $PY scripts/run.py --live --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main --only $NAMES \
      --tag ARG-CN.$TAG-live --seeds 1 2 3 >> $LOG/live.log 2>&1 ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "$(date '+%F %T') stage $STAGE done" >> $LOG/pipeline.log
done
