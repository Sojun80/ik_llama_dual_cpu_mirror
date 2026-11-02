#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-cpu-optimization
P=$(cat dual.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
test -f DUAL_FINISHED
git -C ik-source rev-parse HEAD > ik-commit.txt
cmake -S ik-source -B ik-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DLLAMA_CURL=OFF > ik-config.log 2>&1
numactl --physcpubind=48-63 cmake --build ik-build -j 8 --target llama-bench > ik-build.log 2>&1
touch IK_BUILD_FINISHED
