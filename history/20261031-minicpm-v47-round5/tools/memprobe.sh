#!/bin/bash
# memprobe.sh OUT CASE ENVSTR
# Single-socket server (build-mirror3, 32 threads, baseline order). During decode:
#   * /proc/vmstat numa_* and page-fault counters, delta over a 6 s window
#   * perf stat page-faults / minor-faults for the server
#   * perf mem (AMD IBS) load sampling for 6 s
set -u
OUT=$1; CASE=$2; ENVSTR=$3
R=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round5
KI=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-kernel-investigation
MODEL=/home/ubuntu/llm-evaluations/20261031-minicpm-v47-round4/model.complete.gguf
ORDER=48,49,52,56,57,60,64,65,68,72,73,76,80,84,88,92,81,89,50,58,66,74,82,90,51,59,67,75,83,91,53,61,69,77,85,93,54,62,70,78,86,94,55,63,71,79,87,95
PORT=19189
mkdir -p "$OUT"
env GLIBC_TUNABLES=glibc.malloc.hugetlb=1 $ENVSTR numactl --physcpubind=48-95 --membind=1 $R/build-mirror3/bin/llama-server \
    -m $MODEL -ngl 0 --no-mmap -t 32 -tb 48 --cpu-range $ORDER -c 16384 --host 127.0.0.1 --port $PORT --parallel 1 > "$OUT/server.log" 2>&1 &
SRV=$!
for i in $(seq 600); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
python3 - "$KI/corpus-$CASE.txt" $PORT > "$OUT/request.json" <<'PY' &
import json, sys, urllib.request
txt = open(sys.argv[1]).read()
req = urllib.request.Request(f'http://127.0.0.1:{sys.argv[2]}/completion',
      json.dumps({'prompt': txt, 'temperature': 0, 'n_predict': 512, 'cache_prompt': False, 'ignore_eos': True}).encode(),
      {'Content-Type': 'application/json'})
print(json.dumps(json.load(urllib.request.urlopen(req, timeout=900))['timings']))
PY
REQ=$!
if [ "$CASE" = long ]; then sleep 32; else sleep 11; fi
grep -E '^(numa_|pgfault|pgmajfault|thp_fault|thp_split|pgmigrate)' /proc/vmstat > "$OUT/vmstat-before.txt"
sudo -n perf stat -e page-faults,minor-faults,major-faults,context-switches,cpu-migrations -p $SRV -- sleep 6 2> "$OUT/perfstat.txt"
grep -E '^(numa_|pgfault|pgmajfault|thp_fault|thp_split|pgmigrate)' /proc/vmstat > "$OUT/vmstat-after.txt"
sudo -n perf mem -t load record -o "$OUT/mem.data" -p $SRV -- sleep 4 > "$OUT/memrecord.log" 2>&1
wait $REQ
kill $SRV; wait $SRV 2>/dev/null
sudo -n chown ubuntu:ubuntu "$OUT/mem.data" 2>/dev/null
echo "== vmstat delta over 6 s of decode"
join "$OUT/vmstat-before.txt" "$OUT/vmstat-after.txt" | awk '{d=$3-$2; if (d!=0) print $1, d}'
echo "== perf stat"; grep -E 'faults|switches|migrations|elapsed' "$OUT/perfstat.txt"
tail -2 "$OUT/memrecord.log"; cat "$OUT/request.json"
