#!/usr/bin/env bash
# Runs the stages listed in outputs/logs/ob/queue.txt (one per line, '#' comments allowed), in order.
# The list may be edited while the driver runs: a stage is removed from it when it starts, and
# scripts/ob_pipeline.sh is copied at that moment, so later edits of that script cannot disturb a
# running stage.  To stop cleanly, create outputs/logs/ob/STOP (see ob_pipeline.sh).
#   usage (from the repository root): setsid nohup bash scripts/ob_queue.sh [PID] &
#   PID: wait for this process (a run that is still in progress) to exit before starting
set -u
LOG=outputs/logs/ob
Q=$LOG/queue.txt
mkdir -p $LOG/stages
note() { echo "$(date '+%F %T') $*" >> $LOG/pipeline.log; }
if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
note "queue driver started (pid $$)"
while true; do
  if [ -e $LOG/STOP ]; then note "STOP file present: queue driver stops"; exit 0; fi
  STAGE=$(grep -v '^[[:space:]]*\(#.*\)\?$' "$Q" 2>/dev/null | head -n 1)
  [ -z "$STAGE" ] && break
  awk -v s="$STAGE" '!done && $0 == s { done = 1; next } { print }' "$Q" > "$Q.tmp" && mv -f "$Q.tmp" "$Q"
  cp scripts/ob_pipeline.sh "$LOG/stages/$STAGE.sh"
  bash "$LOG/stages/$STAGE.sh" "$STAGE"
done
note "queue empty: driver stopped"
