#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat numa-build.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test -f NUMA_BUILD_COMPLETE
ulimit -c 0
exec python3 -u numa-test.py
