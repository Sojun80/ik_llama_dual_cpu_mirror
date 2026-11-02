#!/bin/bash
# Bandwidth sweep over per-CCX placements and job sizes. Node-1 memory only.
cd "$(dirname "$0")"
B=./bwprobe
place() { # $1 = cores per CCX, $2 = number of CCX (default 12)
  local k=$1 n=${2:-12} out=""
  for c in $(seq 0 $((n-1))); do for i in $(seq 0 $((k-1))); do out="$out,$((48+4*c+i))"; done; done
  echo ${out#,}
}
BASE32="48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61"
echo "# streaming"
for k in 1 2 3 4; do numactl --membind=1 $B $(place $k) 0 4096; done
numactl --membind=1 $B $BASE32 0 4096 | sed 's/^/baseline32 /'
for k in 2 3 4; do numactl --membind=1 $B $(place $k) 0 4096 512; done
echo "# one CCD (2 CCX)"
for k in 1 2 4; do numactl --membind=1 $B $(place $k 2) 0 2048; done
echo "# one core"
numactl --membind=1 $B 48 0 1024
echo "# decode-like jobs, barrier after each"
for jb in 65536 262144 589824 1048576 4718592 13762560; do
  for k in 2 3 4; do numactl --membind=1 $B $(place $k) $jb 3072; done
  numactl --membind=1 $B $BASE32 $jb 3072 | sed 's/^/baseline32 /'
done
