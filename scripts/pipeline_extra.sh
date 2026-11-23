#!/usr/bin/env bash
# Runs after scripts/pipeline_train.sh has finished:
#   inputs   text-only classifiers on style-normalised renderings of the news (analysis rows)
#   llm2     robustness to the analyst LLM (second model: MODEL2_PATH / MODEL2_TAG)
#   usage: scripts/pipeline_extra.sh [--wait] [STAGE ...]
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
TAG=${LLM_TAG:-qwen25}
MODEL2_PATH=${MODEL2_PATH:-/media/dell/DATA/models/Qwen3-8B}
MODEL2_TAG=${MODEL2_TAG:-qwen3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/train
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }

if [ "${1:-}" = "--wait" ]; then
  shift
  until grep -q "pipeline finished" $LOG/pipeline.log 2>/dev/null; do sleep 60; done
fi
STAGES=${*:-inputs llm2}

for STAGE in $STAGES; do
note "extra stage $STAGE start"
case $STAGE in
inputs)
  for D in ARG-CN ARG-EN; do
    LR=$($PY scripts/selected_lr.py $D.$TAG bert) || { note "FAILED selected_lr bert $D"; continue; }
    $PY scripts/run.py --amp --perturb --dataset outputs/datasets/$D.$TAG.json --sweep inputs --lr $LR \
        --seeds 1 2 3 4 5 >> $LOG/inputs_$D.log 2>&1 || note "FAILED inputs $D"
  done ;;
llm2)
  bash scripts/pipeline_llm2.sh "$MODEL2_PATH" "$MODEL2_TAG" --no-think ;;
*) note "unknown extra stage $STAGE" ;;
esac
note "extra stage $STAGE done"
done
note "extra pipeline finished"
