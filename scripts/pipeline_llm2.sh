#!/usr/bin/env bash
# Robustness to the choice of the analyst LLM: regenerate every LLM output with a second model
# and rerun the LLM-dependent methods at the learning rates selected with the first model.
#   usage: scripts/pipeline_llm2.sh MODEL_PATH TAG [extra llm_generate.py args, e.g. --no-think]
set -u
MODEL=$1; TAG2=$2; shift 2
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
TAG=${LLM_TAG:-qwen25}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/train
mkdir -p $LOG
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }

note "llm2 ($TAG2) generation start"
bash scripts/gen_main.sh "$MODEL" "$TAG2" "$@" || note "FAILED generation $TAG2"
bash scripts/build_all.sh "$TAG2" >> $LOG/llm2.log 2>&1 || { note "FAILED build_all $TAG2"; exit 1; }
for D in ARG-CN ARG-EN; do
  $PY scripts/llm_only.py outputs/datasets/$D.$TAG2.json --json results/llm_only/$D.$TAG2.json >> $LOG/llm2.log 2>&1
  NAMES=$($PY scripts/selected_lr.py $D.$TAG --names clara arg factguard) || { note "FAILED selected_lr"; continue; }
  $PY scripts/run.py --amp --dataset outputs/datasets/$D.$TAG2.json --sweep main --only $NAMES \
      --seeds 1 2 3 >> $LOG/llm2.log 2>&1 || note "FAILED llm2 main $D"
  NAMES=$($PY scripts/selected_lr.py $D.$TAG --names bert_zs) || continue
  $PY scripts/run.py --amp --dataset outputs/datasets/$D.$TAG2.json --sweep main_zs --only $NAMES \
      --seeds 1 2 3 >> $LOG/llm2.log 2>&1 || note "FAILED llm2 main_zs $D"
done
note "llm2 ($TAG2) done"
