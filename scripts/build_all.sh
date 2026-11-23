#!/usr/bin/env bash
# Merge raw data and LLM caches into dataset files.  usage: scripts/build_all.sh [LLM_TAG]
set -eu
TAG=${1:-qwen25}
PY=${CLARA_PY:-$HOME/.venvs/clara/bin/python}
$PY scripts/build_dataset.py --data data/ARG-CN --name ARG-CN --lang zh --llm-dir "outputs/llm/$TAG" \
    --llm "$TAG" --out "outputs/datasets/ARG-CN.$TAG.json" --domains outputs/datasets/ARG-CN.domains.json
$PY scripts/build_dataset.py --data data/ARG-EN --name ARG-EN --lang en --llm-dir "outputs/llm/$TAG" \
    --llm "$TAG" --out "outputs/datasets/ARG-EN.$TAG.json"
