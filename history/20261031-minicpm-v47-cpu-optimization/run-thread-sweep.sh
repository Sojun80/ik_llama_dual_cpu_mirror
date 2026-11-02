#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization
P=$(cat context-16.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test "$(wc -l < context-16.jsonl)" -eq 3
exec python3 -u thread-sweep.py
