#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat chunk4-test.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
exec python3 -u variants.py
