#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat repro-server.pid)
while kill -0 "$P" 2>/dev/null; do
 if grep -q "n_tokens = 9437" repro-server.log; then
  date -u > postcheckpoint-profile-start.txt
  sudo -n perf record -F 99 -g -p "$P" -o postcheckpoint-perf.data -- sleep 8 > postcheckpoint-perf-capture.log 2>&1
  sudo -n perf report --stdio -i postcheckpoint-perf.data --no-children --percent-limit 1 > postcheckpoint-perf-report.txt 2>&1
  exit 0
 fi
 sleep 1
done
exit 1
