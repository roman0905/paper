#!/usr/bin/env bash
# Build the controlled test sets (style attack, single-sentence tampering, occlusion variants).
#   perturbations are written by REWRITER, analysed by ANALYST (two different LLMs on purpose)
# usage: scripts/gen_perturb.sh REWRITER_PATH ANALYST_PATH ANALYST_TAG
set -u
REWRITER=$1; ANALYST=$2; TAG=$3
VPY=${VLLM_PY:-/home/dell/anaconda3/envs/vllm/bin/python}
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LOG=outputs/logs/gen_perturb.log
mkdir -p outputs/llm/rewriter outputs/logs
declare -A LANG=( [ARG-CN]=zh [ARG-EN]=en )

# 1. rewrites and edits by the rewriter model
for D in ARG-CN ARG-EN; do
  $PY scripts/make_perturb.py tamper-input --dataset outputs/datasets/$D.$TAG.json \
      --out outputs/perturb/$D-tamper/input.json >> $LOG 2>&1
done
$VPY scripts/llm_generate.py --model "$REWRITER" --out-dir outputs/llm/rewriter --tasks style --split test \
    --no-think --gpu-util 0.90 \
    --spec "ARG-CN.test=outputs/datasets/ARG-CN.$TAG.json:zh" \
    --spec "ARG-EN.test=outputs/datasets/ARG-EN.$TAG.json:en" >> $LOG 2>&1
echo "style exit=$?" >> $LOG
$VPY scripts/llm_generate.py --model "$REWRITER" --out-dir outputs/llm/rewriter --tasks tamper \
    --no-think --gpu-util 0.90 \
    --spec "ARG-CN.tamperin=outputs/perturb/ARG-CN-tamper/input.json:zh" \
    --spec "ARG-EN.tamperin=outputs/perturb/ARG-EN-tamper/input.json:en" >> $LOG 2>&1
echo "tamper exit=$?" >> $LOG

# 2. raw test files of the perturbed sets
for D in ARG-CN ARG-EN; do
  $PY scripts/make_perturb.py style-build --dataset outputs/datasets/$D.$TAG.json \
      --gen outputs/llm/rewriter/$D.test.style.jsonl --out outputs/perturb/$D-style/test.json >> $LOG 2>&1 || exit 1
  $PY scripts/make_perturb.py tamper-build --input outputs/perturb/$D-tamper/input.json \
      --gen outputs/llm/rewriter/$D.tamperin.tamper.jsonl --out outputs/perturb/$D-tamper/test.json >> $LOG 2>&1 || exit 1
  $PY scripts/make_perturb.py occlude-build --tamper outputs/perturb/$D-tamper/test.json \
      --lang ${LANG[$D]} --name $D-occl --out outputs/datasets/$D-occl.none.json >> $LOG 2>&1 || exit 1
done

# 3. analyses of the perturbed texts by the analyst model (same prompts as the main data)
SPECS=()
for D in ARG-CN ARG-EN; do for K in style tamper; do
  SPECS+=(--spec "$D-$K.test=outputs/perturb/$D-$K/test.json:${LANG[$D]}")
done; done
for TASKS in "clara" "td cs tc zs"; do
  $VPY scripts/llm_generate.py --model "$ANALYST" --out-dir "outputs/llm/$TAG" --tasks $TASKS \
      "${SPECS[@]}" --gpu-util 0.90 >> $LOG 2>&1
  echo "analyst [$TASKS] exit=$?" >> $LOG
done

# 4. dataset files
for D in ARG-CN ARG-EN; do for K in style tamper; do
  $PY scripts/build_dataset.py --data outputs/perturb/$D-$K --name $D-$K --lang ${LANG[$D]} \
      --llm-dir "outputs/llm/$TAG" --llm "$TAG" --splits test --out "outputs/datasets/$D-$K.$TAG.json" >> $LOG 2>&1 || exit 1
done; done
echo "perturb done" >> $LOG
