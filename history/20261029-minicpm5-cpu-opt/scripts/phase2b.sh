#!/usr/bin/env bash
# Phase 2b: variants of S4 (decode strict-pinned on spread cores, batch on all 48 node-1 cores, THP).
set -uo pipefail
E=/home/ubuntu/llm-evaluations/20261029-minicpm5-cpu-opt
cd $E/scripts
THP="--env GLIBC_TUNABLES=glibc.malloc.hugetlb=1"
MS16=$(python3 -c 'from topo import *; print(hex(sum(1 << c for c in spread(1,16))))')
MS12=$(python3 -c 'from topo import *; print(hex(sum(1 << c for c in spread(1,12))))')
MALL=$(python3 -c 'print(hex(sum(1 << c for c in range(48, 96))))')
PIN16="-C $MS16 --cpu-strict 1 -Cb $MALL --cpu-strict-batch 1"
sb() { python3 serve_bench.py "$@"; }
sb "S5 prod-build spread16 tb48 THP fa-off"  prod       48-95 1 16 48 $THP --extra "$PIN16 -fa off"
sb "S6 prod-build spread16 tb48 THP fa-on"   prod       48-95 1 16 48 $THP --extra "$PIN16 -fa on"
sb "S7 znver2 spread16 tb48 THP"             znver2-omp 48-95 1 16 48 $THP --extra "$PIN16"
sb "S8 prod-build spread12 tb48 THP"         prod       48-95 1 12 48 $THP --extra "-C $MS12 --cpu-strict 1 -Cb $MALL --cpu-strict-batch 1"
sb "S9 prod-build spread16 tb48 THP repeat"  prod       48-95 1 16 48 $THP --extra "$PIN16"
touch $E/raw/PHASE2B_DONE
