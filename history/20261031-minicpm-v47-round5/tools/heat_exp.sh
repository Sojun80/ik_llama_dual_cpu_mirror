#!/bin/bash
# FA cold-fetch test: decode profile of s32 with a memory heater on spare node-1 cpu 95 (see memheat.c)
# usage: tools/heat_exp.sh OUT_PREFIX variant...   variants: none | tick<ns> | stream<mb>
cd /home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5
P=$1; shift
B=$PWD/build-prof/bin/llama-server
E=IK_R5_DYN=1,IK_R5_FA=1,IK_R5_HUGE=1,IK_R5_PINONCE=1,IK_R5_FMOE=1,IK_R5_FMOE_UG=1${XENV:+,$XENV}
for v in "$@"; do
    out=runs/$P-$v
    hp=
    case $v in
        tick*)   numactl --physcpubind=95 --membind=1 tools/memheat tick ${v#tick} 3600 2> $out.heat.log > /dev/null & hp=$! ;;
        stream*) numactl --physcpubind=95 --membind=1 tools/memheat stream ${v#stream} 3600 2> $out.heat.log > /dev/null & hp=$! ;;
    esac
    python3 tools/r5bench.py --out $out --cases medium,long --rounds 1 --n-predict 256 --prof --config "s32=$B:32:base:$E" > $out.log 2>&1
    [ -n "$hp" ] && kill $hp && wait $hp 2>/dev/null
    for c in medium long; do
        f=$out/prof-s32-$c-r0.txt
        printf "%-14s %-7s %s  FA %s\n" $v $c "$(sed -n 2p $f | grep -o 'total=[0-9.]*')" "$(grep '^FLASH_ATTN_EXT' $f | awk -F'|' '{print $2}' | awk '{print $3}')"
    done
    [ -f $out.heat.log ] && tail -1 $out.heat.log
done
