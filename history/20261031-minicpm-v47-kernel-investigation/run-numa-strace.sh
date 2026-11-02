#!/bin/bash
set -eu
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
P=$(cat nockpt-corpus.pid)
while kill -0 "$P" 2>/dev/null; do state=$(ps -o stat= -p "$P" || true); case "$state" in Z*|"") break;; esac; sleep 3; done
ulimit -c 0
export IK_EXPERT_NUMA=1 GLIBC_TUNABLES=glibc.malloc.hugetlb=1
set +e
timeout 90 strace -f -e trace=mbind,set_mempolicy -o numa-strace.txt numactl --physcpubind=16-95 --membind=1 ./ik-numa-build/bin/llama-server -m /home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf --no-mmap -ngl 0 -t 32 -tb 48 -c 16384 --host 127.0.0.1 --port 19234 --parallel 1 > numa-strace-server.log 2>&1
printf "%s\n" "$?" > numa-strace-exit.txt
