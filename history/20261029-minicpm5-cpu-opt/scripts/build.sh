#!/usr/bin/env bash
# Zen 2 CPU builds of upstream llama.cpp. Runs on node 1 cores only, away from production (node 0 cores 0-15).
# Usage: build.sh <name> <extra cmake args...>
set -euo pipefail
E=/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt
name=$1; shift
B=$E/build-$name
ZEN2="-march=znver2 -mtune=znver2"
numactl --cpunodebind=1 --membind=1 cmake -S $E/src -B $B -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON \
    -DGGML_AVX512=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
    -DCMAKE_C_FLAGS="$ZEN2" -DCMAKE_CXX_FLAGS="$ZEN2" "$@" > $E/build-logs/$name-config.log 2>&1
numactl --cpunodebind=1 --membind=1 nice -n 5 cmake --build $B -j 24 --target llama-server llama-bench \
    > $E/build-logs/$name-build.log 2>&1
grep -E "CMAKE_C_FLAGS|OpenMP|Adding CPU backend|ggml commit" $E/build-logs/$name-config.log || true
ls -la $B/bin/llama-server $B/bin/llama-bench
