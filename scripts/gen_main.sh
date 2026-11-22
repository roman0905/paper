#!/usr/bin/env bash
# Full offline generation for the two main datasets with one LLM.
# usage: scripts/gen_main.sh MODEL_PATH OUT_TAG [extra llm_generate.py args]
set -u
MODEL=$1; TAG=$2; shift 2
PY=${VLLM_PY:-/home/dell/anaconda3/envs/vllm/bin/python}
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SPECS=()
for sp in train val test; do
  SPECS+=(--spec "ARG-CN.$sp=data/ARG-CN/$sp.json:zh" --spec "ARG-EN.$sp=data/ARG-EN/$sp.json:en")
done
mkdir -p "outputs/llm/$TAG" outputs/logs
for TASKS in "clara" "td cs tc zs cot"; do
  $PY scripts/llm_generate.py --model "$MODEL" --out-dir "outputs/llm/$TAG" --tasks $TASKS \
      "${SPECS[@]}" --gpu-util 0.90 "$@" >> "outputs/logs/gen_$TAG.log" 2>&1
  echo "tasks=[$TASKS] exit=$?" >> "outputs/logs/gen_$TAG.log"
done
