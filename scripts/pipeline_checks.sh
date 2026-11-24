#!/usr/bin/env bash
# Checks that complement the main pipelines:
#   cross       CLARA on tampered texts paired with the analysis of the ORIGINAL text: can the
#               relevance gate detect an on-topic analysis that is unfaithful in one place?
#   fp32        protocol check (Weibo21): cached hidden states, full precision in the trainable layers
#   fp32_main   the selected configuration of every SLM-side method without mixed precision, 5 seeds,
#               both datasets: does mixed precision change the main comparison?
#   live_clara  protocol check (Weibo21): CLARA with the frozen layers recomputed every step (slow)
# usage: scripts/pipeline_checks.sh [--wait] [STAGE ...]     (--wait: start after pipeline_extra.sh)
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
TAG=${LLM_TAG:-qwen25}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/train
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }

if [ "${1:-}" = "--wait" ]; then
  shift
  until grep -q "extra pipeline finished" $LOG/pipeline.log 2>/dev/null; do sleep 60; done
fi
STAGES=${*:-cross fp32 live_clara}

for STAGE in $STAGES; do
note "check stage $STAGE start"
case $STAGE in
cross)
  for D in ARG-CN ARG-EN; do
    NAMES=$($PY scripts/selected_lr.py $D.$TAG --names clara) && \
    $PY scripts/run.py --amp --perturb --cross --dataset outputs/datasets/$D.$TAG.json --sweep main \
        --only $NAMES --tag $D.$TAG-cross --seeds 1 2 3 >> $LOG/checks.log 2>&1 || note "FAILED check cross $D"
  done ;;
fp32)
  NAMES=$($PY scripts/selected_lr.py ARG-CN.$TAG --names bert) && \
  $PY scripts/run.py --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main \
      --only $NAMES arg_gpt@lr2e-05 --tag ARG-CN.$TAG-fp32 --seeds 1 2 3 >> $LOG/checks.log 2>&1 \
      || note "FAILED check fp32" ;;
fp32_main)
  for D in ARG-CN ARG-EN; do
    NAMES=$($PY scripts/selected_lr.py $D.$TAG --names bert arg factguard clara) && \
    $PY scripts/run.py --dataset outputs/datasets/$D.$TAG.json --sweep main \
        --only $NAMES --tag $D.$TAG-fp32 --seeds 1 2 3 4 5 >> $LOG/checks.log 2>&1 || note "FAILED check fp32_main $D"
    ZS=$($PY scripts/selected_lr.py $D.$TAG --names bert_zs) && \
    $PY scripts/run.py --dataset outputs/datasets/$D.$TAG.json --sweep main_zs \
        --only $ZS --tag $D.$TAG-fp32 --seeds 1 2 3 4 5 >> $LOG/checks.log 2>&1 || note "FAILED check fp32_main zs $D"
  done
  # the official-ARG comparison configuration: two more seeds
  $PY scripts/run.py --dataset outputs/datasets/ARG-CN.$TAG.json --sweep main \
      --only arg_gpt@lr2e-05 --tag ARG-CN.$TAG-fp32 --seeds 4 5 >> $LOG/checks.log 2>&1 || note "FAILED check fp32_main arg_gpt" ;;
live_clara)
  bash scripts/pipeline_train.sh live_clara ;;
*) note "unknown check stage $STAGE" ;;
esac
note "check stage $STAGE done"
done
note "checks finished"
