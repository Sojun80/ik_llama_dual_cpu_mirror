#!/usr/bin/env bash
# Phase 2c: 24 decode threads (2 per CCX) under the S4 layout.
set -uo pipefail
E=/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt
cd $E/scripts
while [ ! -e $E/raw/PHASE2B_DONE ]; do sleep 10; done
MS24=$(python3 -c 'from topo import *; print(hex(sum(1 << c for c in spread(1,24))))')
MALL=$(python3 -c 'print(hex(sum(1 << c for c in range(48, 96))))')
python3 serve_bench.py 'S10 prod-build spread24 tb48 THP' prod 48-95 1 24 48 --env GLIBC_TUNABLES=glibc.malloc.hugetlb=1 --extra "-C $MS24 --cpu-strict 1 -Cb $MALL --cpu-strict-batch 1"
touch $E/raw/PHASE2C_DONE
