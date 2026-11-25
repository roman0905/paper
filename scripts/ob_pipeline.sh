#!/usr/bin/env bash
# All open-book experiments (RAWFC / LIAR-RAW), by stage.  Every run is skipped when its result
# file exists, so the script can be restarted at any time; a failing run is recorded in
# outputs/logs/ob/pipeline.log and the pipeline moves on.
#   usage: scripts/ob_pipeline.sh [STAGE ...]      stages: rawfc_main liar_main ablation all_protocol
set -u
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/logs/ob
mkdir -p $LOG
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }
run() {   # run LOGFILE CMD...
  local log=$1; shift
  "$@" >> $LOG/$log 2>&1 || note "FAILED ($log): $*"
}
# the proposed method and its ablations (hyper-parameters fixed a priori, never searched)
SIEVE="--ev-weight 0.5 --train-sample --swap-rate 0.15 --select"
slm() {   # slm DATASET EPOCHS SEEDS...   (two learning rates; the better validation mean is reported)
  local d=$1 ep=$2; shift 2
  for lr in 1e-5 2e-5; do for s in "$@"; do
    run slm.$d.log $PY scripts/ob_slm.py --dataset $d --seed $s --lr $lr --epochs $ep --name roberta-large-bm25@lr$lr
    run slm.$d.log $PY scripts/ob_slm.py --dataset $d --seed $s --lr $lr --epochs $ep --closed --name roberta-large-closed@lr$lr
  done; done
}

for STAGE in "${@:-rawfc_main}"; do
note "stage $STAGE start"
case $STAGE in
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
ablation)     # RAWFC, 3 seeds: one component removed at a time, and the evidence-only control
  D=RAWFC
  for s in 1 2 3; do
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve-noswap   --ev-weight 0.5 --train-sample --select
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve-nohead   --train-sample --swap-rate 0.15
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --name sieve-nosample --ev-weight 0.5 --swap-rate 0.15 --select
    run abl.$D.log $PY scripts/ob_verifier.py --dataset $D --seed $s --no-claim
  done ;;
all_protocol) # trained and tested with ALL reports (the setting of the published numbers)
  D=RAWFC
  for s in 1 2 3; do
    run all.$D.log $PY scripts/ob_verifier.py --dataset $D --protocol all --seed $s
    run all.$D.log $PY scripts/ob_verifier.py --dataset $D --protocol all --seed $s --name sieve $SIEVE
  done
  D=LIAR-RAW
  run all.$D.log $PY scripts/ob_verifier.py --dataset $D --protocol all --seed 1 --epochs 2 --perturb-n 400 --name sieve $SIEVE ;;
*) note "unknown stage $STAGE" ;;
esac
note "stage $STAGE done"
done
note "pipeline finished"
