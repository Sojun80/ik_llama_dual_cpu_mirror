#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat profile.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
cmake -S ik-q8k-source -B ik-q8k-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > q8k-config.log 2>&1
numactl --physcpubind=48-63 cmake --build ik-q8k-build -j 8 --target llama-server > q8k-build.log 2>&1
touch Q8K_BUILD_COMPLETE
