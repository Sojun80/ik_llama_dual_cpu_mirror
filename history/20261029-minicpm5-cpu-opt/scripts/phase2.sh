#!/usr/bin/env bash
# Phase 2: real Casebook prompts through llama-server, node 1 only (production lives on node 0 cores 0-15).
set -uo pipefail
E=/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt
cd $E/scripts
THP="--env GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
P16=48-63
S16=$(python3 -c 'from topo import *; print(cpulist(spread(1,16)))')
MS16=$(python3 -c 'from topo import *; print(hex(sum(1 << c for c in spread(1,16))))')
MALL=$(python3 -c 'print(hex(sum(1 << c for c in range(48, 96))))')
sb() { python3 serve_bench.py "$@"; }
sb "S0 prod-build packed16 4K"          prod       $P16    1 16 16
sb "S1 prod-build packed16 THP"         prod       $P16    1 16 16 $THP
sb "S2 prod-build spread16 THP"         prod       $S16    1 16 16 $THP
sb "S3 znver2 spread16 THP"             znver2-omp $S16    1 16 16 $THP
# Decode on 16 spread cores, prompt batches on all 48 node-1 cores (separate strict-pinned threadpools).
sb "S4 prod-build spread16 tb48 THP"    prod       48-95   1 16 48 $THP --extra "-C $MS16 --cpu-strict 1 -Cb $MALL --cpu-strict-batch 1"
sb "S5 prod-build spread16 fa-off THP"  prod       $S16    1 16 16 $THP --extra "-fa off"
touch $E/raw/PHASE2_DONE
