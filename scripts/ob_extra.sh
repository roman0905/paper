#!/usr/bin/env bash
# Open-book experiments that complement scripts/ob_pipeline.sh.
#   refresh   redo the evaluation-only / cheap runs whose result files predate an evaluation
#             condition that was added later (zero-shot verifier, RoBERTa-large with evidence)
#   llm2      second LLM (Qwen3-8B): LoRA with first-stage evidence and SIEVE on RAWFC, 3 seeds
#   usage: scripts/ob_extra.sh [--wait] [STAGE ...]     (--wait: start after ob_pipeline.sh has finished)
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/ob
MODEL2=${MODEL2_PATH:-/media/dell/DATA/models/Qwen3-8B}
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }
run() { local log=$1; shift; "$@" >> $LOG/$log 2>&1 || note "FAILED ($log): $*"; }
SIEVE="--ev-weight 0.5 --train-sample --swap-rate 0.15 --select"

if [ "${1:-}" = "--wait" ]; then
  shift
  until grep -q "pipeline finished" $LOG/pipeline.log 2>/dev/null; do sleep 120; done
fi
for STAGE in "${@:-refresh}"; do
note "extra stage $STAGE start"
case $STAGE in
refresh)
  for D in RAWFC LIAR-RAW; do
    E=10; N=""; [ $D = LIAR-RAW ] && E=5 && N="--perturb-n 400"
    # result files of evidence-reading runs without the latest condition
    $PY - "$D" <<'PYEOF' > /tmp/ob_refresh.$$
import glob, json, os, sys
D = sys.argv[1]
for path in sorted(glob.glob("results/ob/runs/{}.strict/*/s*.json".format(D))):
    name, seed = os.path.basename(os.path.dirname(path)), os.path.basename(path)[1:-5]
    if not (name == "zs-bm25-m16" or name.startswith("roberta-large-bm25@")):
        continue
    if "test@nearswap" not in json.load(open(path, encoding="utf-8"))["metrics"]:
        print(name, seed)
PYEOF
    while read -r NAME SEED; do
      if [ "$NAME" = zs-bm25-m16 ]; then
        run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0 --force $N
      else
        run slm.$D.log $PY scripts/ob_slm.py --dataset $D --seed $SEED --lr ${NAME##*@lr} --epochs $E --name $NAME --force
      fi
    done < /tmp/ob_refresh.$$
    rm -f /tmp/ob_refresh.$$
  done ;;
llm2)
  D=RAWFC
  run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D --model $MODEL2 --epochs 0 --closed --name qwen3-zs-closed
  run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D --model $MODEL2 --epochs 0 --name qwen3-zs-bm25-m16
  for s in 1 2 3; do
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D --model $MODEL2 --seed $s --closed --name qwen3-lora-closed
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D --model $MODEL2 --seed $s --name qwen3-lora-bm25-m16
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D --model $MODEL2 --seed $s --name qwen3-sieve $SIEVE
  done ;;
*) note "unknown extra stage $STAGE" ;;
esac
note "extra stage $STAGE done"
done
note "extras finished"
