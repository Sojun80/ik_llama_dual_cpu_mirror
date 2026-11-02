#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4
pid=$(cat download-parallel.pid)
while kill -0 "$pid" 2>/dev/null; do
  state=$(ps -o stat= -p "$pid" || true)
  case "$state" in Z*|"") break;; esac
  sleep 20
done
grep -qx assembled download-parallel.log || { echo "Download did not complete"; exit 1; }
test -x build/bin/llama-server
sha256sum build/bin/llama-server > server.sha256
git -C source rev-parse HEAD > source-commit.txt
python3 -u run.py
