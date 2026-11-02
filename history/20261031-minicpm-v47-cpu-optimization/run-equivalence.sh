#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization
P=$(cat ik-server-build.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test -f IK_SERVER_BUILD_FINISHED
exec python3 -u equivalence.py
