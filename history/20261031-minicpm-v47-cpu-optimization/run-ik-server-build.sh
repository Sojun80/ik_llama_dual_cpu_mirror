#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization
P=$(cat verify-deep.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test -f VERIFY_DEEP_FINISHED
numactl --physcpubind=48-63 cmake --build ik-build -j 8 --target llama-server > ik-server-build.log 2>&1
touch IK_SERVER_BUILD_FINISHED
