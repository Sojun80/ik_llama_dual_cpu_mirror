#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat corpus.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test -f CORPUS_COMPLETE
cmake -S ik-chunk4-source -B ik-chunk4-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > chunk4-config.log 2>&1
numactl --physcpubind=48-63 cmake --build ik-chunk4-build -j 8 --target llama-server > chunk4-build.log 2>&1
touch CHUNK4_BUILD_COMPLETE
