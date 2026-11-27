#!/usr/bin/env bash
# All open-book experiments (RAWFC / LIAR-RAW), by stage.  Every run is skipped when its result
# file exists, so a stage can be restarted at any time; a failing run is recorded in
# outputs/logs/ob/pipeline.log and the stage moves on.
#   usage: scripts/ob_pipeline.sh STAGE [STAGE ...]        (normally started by scripts/ob_queue.sh)
#   stages, in the order they are meant to run:
#     smoke                   GPU check of the code paths that later stages rely on (writes no results)
#     refresh                 redo the two RAWFC runs whose result files predate the last evaluation conditions
#     rawfc_main liar_main    main comparison, strict protocol
#     rawfc_j liar_j          the same verifier trained with the explanation target (sieve-j)
#     decide                  validation-only rule: is the explanation target part of the final method?
#     all_protocol            the final method trained and tested with all reports
#     ablation                the final method with one component removed; evidence-only control
#     reeval                  re-evaluate saved adapters whose runs predate the per-claim usability scores
#     llm2                    second backbone (Llama-2-7B-chat) on RAWFC
#   To stop cleanly, create outputs/logs/ob/STOP: the current run finishes, nothing else is started.
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/ob
mkdir -p $LOG
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }
run() {   # run LOGFILE CMD...
  local log=$1; shift
  if [ -e $LOG/STOP ]; then note "STOP file present, not started: $*"; exit 0; fi
  "$@" >> $LOG/$log 2>&1 || note "FAILED ($log): $*"
}
# the proposed method (hyper-parameters fixed a priori, never searched) and the explanation target
SIEVE="--ev-weight 0.5 --train-sample --swap-rate 0.15 --select"
EXPL="--explain-weight 0.5 --justify"
# second backbone; its tokenizer needs ~15% more tokens for the same text, hence the larger limit
# (so that, as with the first backbone, practically no query loses evidence sentences)
MODEL2=${MODEL2_PATH:-/media/dell/DATA/models/llama-2-7b-chat-hf}
M2=llama2
X2="--model $MODEL2 --max-tokens 1300"
# name and flags of the final method: decided on validation data by scripts/ob_decide.py (stage "decide")
final_name()  { $PY scripts/ob_decide.py 2>> $LOG/pipeline.log; }
flags_of()    { if [ "$1" = sieve-j ]; then echo "$SIEVE $EXPL"; else echo "$SIEVE"; fi; }
missing() {   # missing RESULT_JSON CONDITION : true if the result file lacks the evaluation condition
  $PY - "$1" "$2" <<'PYEOF'
import json, os, sys
p, c = sys.argv[1], sys.argv[2]
sys.exit(0 if not os.path.exists(p) or c not in json.load(open(p, encoding="utf-8"))["metrics"] else 1)
PYEOF
}
slm() {   # slm DATASET EPOCHS SEEDS...   (two learning rates; the better validation mean is reported)
  local d=$1 ep=$2; shift 2
  for lr in 1e-5 2e-5; do for s in "$@"; do
    run slm.$d.log $PY scripts/ob_slm.py --dataset $d --seed $s --lr $lr --epochs $ep --name roberta-large-bm25@lr$lr
    run slm.$d.log $PY scripts/ob_slm.py --dataset $d --seed $s --lr $lr --epochs $ep --closed --name roberta-large-closed@lr$lr
  done; done
}

for STAGE in "$@"; do
note "stage $STAGE start (pid $$)"
case $STAGE in
smoke)        # nothing is written with --limit; the output goes to smoke.log
  run smoke.log $PY scripts/ob_verifier.py --dataset RAWFC --seed 1 --epochs 1 --limit 48 --name smoke $SIEVE $EXPL --transfer
  run smoke.log $PY scripts/ob_verifier.py --dataset LIAR-RAW --seed 1 --epochs 1 --limit 48 --perturb-n 16 --name smoke $SIEVE $EXPL
  run smoke.log $PY scripts/ob_verifier.py --dataset RAWFC $X2 --seed 1 --epochs 1 --limit 32 --name smoke $SIEVE $EXPL ;;
refresh)      # same configuration and seed as before; only the set of evaluation conditions is completed
  D=RAWFC
  missing results/ob/runs/$D.strict/zs-bm25-m16/s0.json test@nearswap && \
    run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0 --force
  missing results/ob/runs/$D.strict/roberta-large-bm25@lr2e-5/s1.json test@nearswap && \
    run slm.$D.log $PY scripts/ob_slm.py --dataset $D --seed 1 --lr 2e-5 --epochs 10 --name roberta-large-bm25@lr2e-5 --force ;;
rawfc_main)   # main comparison on RAWFC, strict protocol, 5 seeds
  D=RAWFC
  run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0 --closed
  run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0
  slm $D 10 1 2 3 4 5
  for s in 1 2 3 4 5; do run lora.$D.log $PY scripts/ob_verifier.py --dataset $D --closed --seed $s; done
  for s in 1 2 3 4 5; do
    X=""; [ $s -le 3 ] && X="--transfer"
    run lora.$D.log  $PY scripts/ob_verifier.py --dataset $D --seed $s $X
    run sieve.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve $SIEVE $X
  done ;;
liar_main)    # main comparison on LIAR-RAW, strict protocol, 3 seeds, 2 epochs for the LoRA runs
  D=LIAR-RAW; E="--epochs 2 --perturb-n 400"
  run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0 --closed
  run zs.$D.log $PY scripts/ob_verifier.py --dataset $D --epochs 0 --perturb-n 400
  slm $D 5 1 2 3
  for s in 1 2 3; do run lora.$D.log $PY scripts/ob_verifier.py --dataset $D --closed --seed $s --epochs 2; done
  for s in 1 2 3; do
    run lora.$D.log  $PY scripts/ob_verifier.py --dataset $D --seed $s $E --transfer
    run sieve.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s $E --name sieve $SIEVE --transfer
  done ;;
rawfc_j)      # SIEVE trained with the explanation target, RAWFC, 5 seeds
  D=RAWFC
  for s in 1 2 3 4 5; do
    X=""; [ $s -le 3 ] && X="--transfer"
    run sievej.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve-j $SIEVE $EXPL $X
  done ;;
liar_j)       # the same on LIAR-RAW, 3 seeds
  D=LIAR-RAW
  for s in 1 2 3; do
    run sievej.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --epochs 2 --perturb-n 400 --name sieve-j $SIEVE $EXPL --transfer
  done ;;
decide)       # validation-only decision about the explanation target (written once)
  F=$(final_name) && note "final method: $F" || note "no decision could be made" ;;
all_protocol) # the final method trained AND tested with all reports (the setting of the published numbers), 3 seeds
  F=$(final_name) || { note "stage $STAGE skipped: no decision"; continue; }
  D=LIAR-RAW
  for s in 1 2 3; do
    run all.$D.log $PY scripts/ob_verifier.py --dataset $D --protocol all --seed $s --epochs 2 --perturb-n 400 --name $F $(flags_of $F)
  done
  D=RAWFC
  for s in 1 2 3; do
    run all.$D.log $PY scripts/ob_verifier.py --dataset $D --protocol all --seed $s --name $F $(flags_of $F)
  done ;;
ablation)     # RAWFC, 3 seeds: the final method with one component removed, and the evidence-only control
  F=$(final_name) || { note "stage $STAGE skipped: no decision"; continue; }
  D=RAWFC; FE=""; [ "$F" = sieve-j ] && FE="$EXPL"
  for s in 1 2 3; do
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name $F-noswap   --ev-weight 0.5 --train-sample --select $FE
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name $F-nosample --ev-weight 0.5 --swap-rate 0.15 --select $FE
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name $F-nohead   --train-sample --swap-rate 0.15 $FE
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --no-claim
  done ;;
reeval)       # the first RAWFC "sieve" runs predate the storing of per-claim usability scores
  F=$(final_name) || { note "stage $STAGE skipped: no decision"; continue; }
  D=RAWFC
  if [ "$F" = sieve ]; then
    for s in 1 2 3 4 5; do
      missing results/ob/runs/$D.strict/sieve/s$s.reeval.json test && \
        run reeval.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve $SIEVE --eval-only
    done
  fi ;;
llm2)         # second backbone on RAWFC, 3 seeds
  F=$(final_name) || { note "stage $STAGE skipped: no decision"; continue; }
  D=RAWFC
  run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D $X2 --epochs 0 --closed --name $M2-zs-closed
  run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D $X2 --epochs 0 --name $M2-zs-bm25-m16
  for s in 1 2 3; do
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D $X2 --seed $s --closed --name $M2-lora-closed
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D $X2 --seed $s --name $M2-lora-bm25-m16
    run llm2.$D.log $PY scripts/ob_verifier.py --dataset $D $X2 --seed $s --name $M2-$F $(flags_of $F)
  done ;;
*) note "unknown stage $STAGE" ;;
esac
note "stage $STAGE done"
done
