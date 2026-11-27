#!/usr/bin/env bash
# Build the thesis in a scratch directory from a preview analysis, without touching thesis/gen.
#
#   bash scripts/dev/thesis_preview.sh [sieve|sieve-j|decided]
#
# "sieve" / "sieve-j": pretend that run is the final method (before results/ob/decision.json exists);
# "decided": use the decision as it is.  Output: $DEST/thesis/main.pdf (default DEST=/tmp/thesis_preview).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
DEST=${DEST:-/tmp/thesis_preview}
FINAL=${1:-sieve}
mkdir -p "$DEST/thesis"
rsync -a --delete --exclude main.pdf --exclude '*.aux' --exclude '*.log' --exclude '*.bbl' --exclude '*.blg' \
      --exclude '*.toc' --exclude '*.out' "$ROOT/thesis/" "$DEST/thesis/"
cd "$ROOT"
if [ "$FINAL" = decided ]; then
  $PY scripts/ob_analyze.py --out "$DEST/analysis.json"
else
  $PY scripts/ob_analyze.py --out "$DEST/analysis.json" --assume-final "$FINAL"
fi
MPLBACKEND=Agg $PY scripts/ob_assets.py --analysis "$DEST/analysis.json" --out "$DEST/thesis/gen" | grep -v '^wrote' || true
make -C "$DEST/thesis"
echo "preview: $DEST/thesis/main.pdf"
