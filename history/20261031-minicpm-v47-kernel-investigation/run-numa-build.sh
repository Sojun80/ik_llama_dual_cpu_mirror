#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat variants2.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
cmake -S ik-numa-source -B ik-numa-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > numa-build-config.log 2>&1
numactl --physcpubind=48-63 cmake --build ik-numa-build -j 8 --target llama-server > numa-build.log 2>&1
touch NUMA_BUILD_COMPLETE
